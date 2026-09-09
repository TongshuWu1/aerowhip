from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from learning.simple_ppo import PPORollout
from learning.point_force_env import (
    POINT_FORCE_OBSERVATION_DIM,
    PointForceWhipEnvironment,
    near_target_speed_quality,
    tip_velocity_strike_gate,
)


ROOT = Path(__file__).resolve().parents[2]


def _configs() -> tuple[dict, dict, dict]:
    return tuple(
        json.loads((ROOT / "config" / name).read_text(encoding="utf-8"))
        for name in ("model.json", "task.json", "ppo.json")
    )


def test_zero_policy_action_is_exact_hanging_hover_force() -> None:
    model, task, ppo = _configs()
    environment = PointForceWhipEnvironment(
        model, task, ppo, batch_size=3, device=torch.device("cpu")
    )
    observation = environment.reset()
    force = environment.physical_force(torch.zeros((3, 3)))
    hover = environment.model.hover_force_world_n(
        dtype=torch.float64, device="cpu"
    )
    assert observation.shape == (3, POINT_FORCE_OBSERVATION_DIM)
    torch.testing.assert_close(force, hover[None].expand(3, -1))


def test_reward_strike_quality_has_no_angle_input_or_angle_credit() -> None:
    distance = torch.tensor([0.05, 0.05], dtype=torch.float64)
    directed_speed = torch.tensor([4.0, 4.0], dtype=torch.float64)
    quality = near_target_speed_quality(
        distance,
        directed_speed,
        proximity_scale_m=0.15,
        directed_speed_cap_m_s=4.0,
    )
    torch.testing.assert_close(quality[0], quality[1])


def test_actual_tip_velocity_uses_directed_speed_and_binary_angle_gate() -> None:
    desired = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64).expand(6, -1)
    velocity = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [4.0, 4.0, 0.0],
            [4.0, 4.01, 0.0],
            [3.99, 0.0, 0.0],
            [-5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
        ],
        dtype=torch.float64,
    )
    passed = tip_velocity_strike_gate(
        velocity,
        desired,
        minimum_directed_speed_m_s=4.0,
        maximum_direction_error_deg=45.0,
    )
    assert passed.tolist() == [True, True, False, False, False, False]


def test_short_environment_episode_is_finite_and_terminates_at_horizon() -> None:
    model, task, ppo = _configs()
    # Isolate timeout/time accounting from tiny calibrated-model drift shaping.
    ppo = copy.deepcopy(ppo)
    ppo['reward']['progress_weight'] = 0.
    ppo['reward']['strike_quality_improvement_weight'] = 0.
    task = copy.deepcopy(task)
    task["episode_duration_s"] = 0.2
    environment = PointForceWhipEnvironment(
        model, task, ppo, batch_size=2, device=torch.device("cpu")
    )
    observation = environment.reset()
    for step_index in range(environment.control_step_count):
        result = environment.step(torch.zeros((2, 3)))
        observation = result.next_observation
    assert torch.isfinite(observation).all()
    assert torch.isfinite(environment.episode_reward).all()
    assert result.done[:, 0].bool().all()
    assert not environment.episode_success.any()
    assert environment.episode_timed_out.all()
    expected_timeout_reward = -(
        ppo["reward"]["timeout_penalty"]
        + ppo["reward"]["time_to_success_weight_per_s"]
        * task["episode_duration_s"]
    )
    torch.testing.assert_close(
        environment.episode_reward,
        torch.full((2,), expected_timeout_reward, dtype=torch.float64),
    )


def test_point_displacement_integral_cost_is_charged_every_physics_step() -> None:
    model, task, ppo = _configs()
    task = copy.deepcopy(task)
    task["episode_duration_s"] = 0.1
    environment = PointForceWhipEnvironment(
        model, task, ppo, batch_size=1, device=torch.device("cpu")
    )
    result = environment.step(torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float64))
    expected = (
        -ppo["reward"]["point_displacement_integral_weight"]
        * environment.episode_point_displacement_cost_integral_s
    )
    torch.testing.assert_close(
        result.components.point_displacement_integral[:, 0], expected
    )
    assert environment.episode_point_displacement_integral_m_s.item() > 0.0
    assert environment.episode_point_displacement_cost_integral_s.item() > 0.0


def test_invalid_tip_entry_is_recorded_but_episode_continues_until_timeout() -> None:
    model, task, ppo = _configs()
    task = copy.deepcopy(task)
    task['success']['first_contact_only'] = False  # Saved legacy runs retain their contract.
    cable_length = sum(model["cable"]["marker_interval_lengths_m"])
    task["target_position_m"] = [0.0, 0.0, 1.5 - cable_length]
    task["success"]["tip_target_distance_m"] = 0.01
    task["episode_duration_s"] = 0.2
    environment = PointForceWhipEnvironment(
        model, task, ppo, batch_size=1, device=torch.device("cpu")
    )
    result = environment.step(torch.zeros((1, 3)))
    assert result.newly_invalid_tip_entry.item() is True
    assert result.done.item() == 0.0
    assert environment.active.item() is True
    assert result.components.invalid_tip_entry.item() == -ppo["reward"][
        "invalid_tip_entry_penalty"
    ]
    for step in range(1, environment.control_step_count):
        result = environment.step(torch.zeros((1, 3)))
        assert result.newly_invalid_tip_entry.item() is False
        if step < environment.control_step_count - 1:
            assert not result.newly_timed_out.item()
            assert not result.done.item()
    assert result.newly_timed_out.item() is True
    assert result.done.item() == 1.0


def test_short_point_force_rollout_updates_ppo() -> None:
    from run_ppo import build_agent, collect_rollout

    torch.set_num_threads(1)
    model, task, ppo = _configs()
    task = copy.deepcopy(task)
    task["episode_duration_s"] = 0.2
    # This case checks rollout-to-update accounting; full PID recovery has
    # separate coverage. Keep this isolated training smoke test short.
    ppo['deployment']['recovery_duration_s'] = .05
    device = torch.device("cpu")
    environment = PointForceWhipEnvironment(
        model, task, ppo, batch_size=4, device=device
    )
    agent = build_agent(ppo, device)
    rollout = PPORollout.allocate(
        environment.control_step_count,
        4,
        POINT_FORCE_OBSERVATION_DIM,
        3,
        device=device,
    )
    collect_rollout(environment, agent, rollout)
    metrics = agent.update(
        rollout,
        minibatch_size=8,
        epochs=2,
        generator=torch.Generator().manual_seed(9),
    )
    assert metrics.valid_transitions == environment.control_step_count * environment.batch_size
    assert torch.isfinite(rollout.rewards).all()
