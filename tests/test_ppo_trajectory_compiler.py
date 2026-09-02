from __future__ import annotations

import torch
from torch import nn

from learning.normalization import FixedContextNormalizer
from learning.ppo_trajectory_compiler import (
    StateImpulse,
    compile_ppo_trajectory,
    metric_maximum_absolute_differences,
    replay_compiled_trajectory,
)
from learning.sequential_sac_env import (
    ATTACHMENT_RELATIVE_SPEED_SHAPING,
    TARGET_ALIGNED_SAGITTAL_ACTION_MODE,
    SequentialWhipEnvironment,
)
from planning.task import load_canonical_whip_task
from simulator.simulator import CoupledSimulator
from simulator.state import SimulatorState
from simulator.uav.model import FullStateUAVModel

from ._common import PROJECT_ROOT, SETTINGS


class _ZeroAgent:
    def __init__(self) -> None:
        self.policy = nn.Identity()

    @staticmethod
    def deterministic_action(observation: torch.Tensor) -> torch.Tensor:
        return torch.zeros((observation.shape[0], 6), dtype=observation.dtype)


def _environment(
    *,
    terminate_on_success: bool = False,
    action_mode: str = "full_6d",
    directed_speed_shaping_reference: str = "world_tip",
) -> SequentialWhipEnvironment:
    simulator = CoupledSimulator(
        SETTINGS.cable_configuration,
        SETTINGS.parameters,
        dt_s=SETTINGS.dt_s,
        device="cpu",
        dtype=torch.float64,
        uav_model=FullStateUAVModel(),
        attachment_offset_body_m=SETTINGS.attachment_offset_body_m,
        attachment_tangent_body=SETTINGS.attachment_tangent_body,
    )
    initial = simulator.reset(
        torch.tensor(SETTINGS.initial_uav_position_m, dtype=torch.float64),
        uav_orientation_xyzw=torch.tensor(
            SETTINGS.initial_uav_orientation_xyzw, dtype=torch.float64
        ),
    )
    return SequentialWhipEnvironment(
        simulator,
        load_canonical_whip_task(PROJECT_ROOT / "config" / "tasks" / "canonical_whip_v1.json"),
        initial,
        FixedContextNormalizer(torch.zeros(83), torch.ones(83), 1),
        batch_size=1,
        episode_duration_s=0.03,
        control_dt_s=0.01,
        action_mode=action_mode,
        directed_speed_shaping_reference=directed_speed_shaping_reference,
        success_mode="task_whip_once",
        reward_mode="whip_potential",
        terminate_on_success=terminate_on_success,
        record_fullstate_commands=True,
        record_state_trajectory=True,
    )


def test_target_aligned_relative_speed_environment_executes_end_to_end() -> None:
    environment = _environment(
        action_mode=TARGET_ALIGNED_SAGITTAL_ACTION_MODE,
        directed_speed_shaping_reference=ATTACHMENT_RELATIVE_SPEED_SHAPING,
    )
    observation = environment.reset()
    result = environment.step(torch.zeros((1, 3), dtype=torch.float64))
    assert observation.shape == (1, 84)
    assert result.next_observation.shape == (1, 84)
    assert bool(torch.isfinite(result.reward).all())
    assert environment.previous_normalized_action.shape == (1, 3)


def test_successful_hit_is_terminal_and_post_hit_padding_is_frozen() -> None:
    environment = _environment(terminate_on_success=True)
    environment.reset()
    # Isolate terminal handling from contact geometry: this row has already
    # registered a valid c10-first task event when the control interval ends.
    environment.episode_scientific_event.fill_(True)
    environment.episode_first_entry_marker.fill_(10)
    action = torch.zeros((1, 6), dtype=torch.float64)

    hit = environment.step(action)
    assert bool(hit.newly_successful.item())
    assert bool(hit.done.item())
    assert bool(hit.include_transition.item())
    hit_uav_position = environment.state.uav.position_m.clone()
    hit_cable_positions = environment.state.cable.positions_m.clone()

    padded = environment.step(action)
    assert bool(padded.done.item())
    assert not bool(padded.include_transition.item())
    assert float(padded.reward.item()) == 0.0
    torch.testing.assert_close(
        environment.state.uav.position_m, hit_uav_position, atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        environment.state.cable.positions_m,
        hit_cable_positions,
        atol=0.0,
        rtol=0.0,
    )


def test_compiler_records_resolved_commands_and_replays_exactly() -> None:
    environment = _environment()
    compiled = compile_ppo_trajectory(environment, _ZeroAgent())
    assert compiled.actor_queries == 3
    assert compiled.physics_steps == 3
    assert compiled.commands.step_count == 3
    assert compiled.commands.batch_size == 1
    # Live-state p/v re-anchoring is captured at each physical step.
    torch.testing.assert_close(
        compiled.commands.positions_m,
        environment.recorded_state_trajectory(clone=False)["uav_position_m"][:-1],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        compiled.commands.velocities_m_s,
        environment.recorded_state_trajectory(clone=False)["uav_velocity_m_s"][:-1],
        atol=0.0,
        rtol=0.0,
    )
    replay = replay_compiled_trajectory(
        environment.simulator,
        environment._initial_state,  # noqa: SLF001 - exact compiler reset source
        environment.task,
        compiled.commands,
        reference_state_trajectory=environment.recorded_state_trajectory(clone=False),
    )
    assert max(replay.reference_maximum_absolute_difference.values()) == 0.0
    assert all(
        value == 0
        for value in metric_maximum_absolute_differences(
            compiled.feedback_metrics, replay.metrics
        ).values()
    )


def test_post_planning_impulse_changes_state_only_at_requested_step() -> None:
    environment = _environment()
    state = environment._initial_state  # noqa: SLF001 - diagnostic state fixture
    impulse = StateImpulse(
        "lateral",
        physics_step=2,
        uav_velocity_delta_world_m_s=(0.0, 0.1, 0.0),
        cable_velocity_delta_world_m_s=(0.0, 0.2, 0.0),
        cable_node_start=10,
    )
    unchanged = impulse(1, state)
    assert unchanged is state
    changed = impulse(2, state)
    torch.testing.assert_close(
        changed.uav.velocity_m_s - state.uav.velocity_m_s,
        torch.tensor([[0.0, 0.1, 0.0]], dtype=torch.float64),
    )
    assert isinstance(changed, SimulatorState)
    assert float(changed.cable.velocities_m_s[0, -1, 1]) == 0.2
