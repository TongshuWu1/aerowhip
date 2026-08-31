"""Batch one-shot policy evaluation through the frozen production simulator.

This module deliberately accepts a complete action tensor, never a callable
``policy(state_t)``.  The only policy query therefore occurs before this API is
entered; the decoded FullState trajectory is fixed for the entire rollout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from planning.cem_task import VariableDurationWhipTask
from planning.metrics import PopulationRolloutResult
from planning.rl_reward import RLWhipRewardComponents, RLWhipRewardConfig
from planning.rollout import DeterministicReplay
from planning.variable_duration import (
    replay_variable_trajectory,
    run_variable_population_rollout,
)
from simulator.simulator import CoupledSimulator

from .policy_action import DecodedPolicyAction, decode_policy_action
from .policy_context import PhysicsContext, PolicyContext


@dataclass(frozen=True, slots=True)
class OneShotEpisodeResult:
    """Compact GPU-resident results for a batch of complete open-loop maneuvers."""

    reward: torch.Tensor
    feasibility_ranking_reward: torch.Tensor
    task_success: torch.Tensor
    feasible: torch.Tensor
    tip_min_distance_m: torch.Tensor
    max_tip_speed_m_s: torch.Tensor
    hit_time_s: torch.Tensor
    directed_tip_speed_m_s: torch.Tensor
    direction_error_deg: torch.Tensor
    first_entry_marker: torch.Tensor
    max_uav_displacement_m: torch.Tensor
    max_uav_speed_m_s: torch.Tensor
    max_command_acceleration_m_s2: torch.Tensor
    rollout_finite: torch.Tensor
    task_cost: torch.Tensor
    decoded_action: DecodedPolicyAction
    population_metrics: PopulationRolloutResult
    trajectory: DeterministicReplay | None = None
    reward_components: RLWhipRewardComponents | None = None

    @property
    def batch_size(self) -> int:
        return int(self.reward.shape[0])

    def row(self, index: int) -> dict[str, object]:
        """Materialize one compact row for diagnostics, not for the physics loop."""

        def scalar(value: torch.Tensor) -> float:
            return float(value[index].detach().cpu())

        marker = int(self.first_entry_marker[index].detach().cpu())
        success = bool(self.task_success[index].detach().cpu())
        result = {
            "reward": scalar(self.reward),
            "feasibility_ranking_reward": scalar(self.feasibility_ranking_reward),
            "task_success": success,
            "feasible": bool(self.feasible[index].detach().cpu()),
            "tip_min_distance_m": scalar(self.tip_min_distance_m),
            "max_tip_speed_m_s": scalar(self.max_tip_speed_m_s),
            "hit_time_s": scalar(self.hit_time_s) if success else None,
            "directed_tip_speed_m_s": scalar(self.directed_tip_speed_m_s),
            "direction_error_deg": scalar(self.direction_error_deg),
            "first_entry_marker": marker,
            "first_entry_marker_label": None if marker == 0 else f"c{marker}",
            "max_uav_displacement_m": scalar(self.max_uav_displacement_m),
            "max_uav_speed_m_s": scalar(self.max_uav_speed_m_s),
            "max_command_acceleration_m_s2": scalar(
                self.max_command_acceleration_m_s2
            ),
            "rollout_finite": bool(self.rollout_finite[index].detach().cpu()),
            "task_cost": scalar(self.task_cost),
            "duration_s": scalar(self.decoded_action.duration_s),
        }
        if self.reward_components is not None:
            components = self.reward_components
            result["reward_components"] = {
                "progress": scalar(components.progress),
                "strike": scalar(components.strike),
                "success": scalar(components.success),
                "safety": scalar(components.safety),
                "non_tip": scalar(components.non_tip),
                "control": scalar(components.control),
                "time": scalar(components.time),
                "normalized_progress": scalar(components.normalized_progress),
                "initial_tip_distance_m": scalar(components.initial_tip_distance_m),
                "minimum_tip_distance_m": scalar(components.minimum_tip_distance_m),
            }
        return result


def _verify_frozen_physics_context(
    simulator: CoupledSimulator,
    context: PolicyContext,
) -> None:
    expected = PhysicsContext.from_parameters(
        simulator.parameters,
        batch_size=context.batch_size,
        dtype=context.to_tensor().dtype,
        device=context.to_tensor().device,
    ).raw_tensor()
    actual = context.physics.raw_tensor()
    if not torch.allclose(actual, expected, rtol=1.0e-6, atol=1.0e-9):
        raise ValueError(
            "Milestone 5A permits only the current frozen physics context; "
            "physics randomization is not enabled."
        )


def evaluate_open_loop_batch(
    simulator: CoupledSimulator,
    contexts: PolicyContext,
    normalized_actions: torch.Tensor,
    task: VariableDurationWhipTask,
    *,
    record_trajectory: bool = False,
    rl_reward_config: RLWhipRewardConfig | None = None,
    evaluation_time_s: float | None = None,
) -> OneShotEpisodeResult:
    """Decode and execute a complete batch of maneuvers with no feedback query.

    ``normalized_actions`` is the complete output of a future policy call.  No
    policy object is accepted here and no action is requested inside the fixed
    physics loop.  ``reward`` is the negative of the existing tuned task cost;
    hard feasibility and success remain explicit result fields.  The separate
    feasibility-ranking reward mirrors CEM ordering and is not a model-fitting
    loss.
    """

    if task.model_freeze != "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI":
        raise ValueError("One-shot evaluation requires the frozen PRE_MPPI model.")
    legacy = task.legacy_run_online_objective
    if legacy is None or legacy.profile != "legacy_run_online_strike_margin_tuned_v4":
        raise ValueError("One-shot evaluation requires the accepted tuned legacy reward.")
    if task.real_flight_authorized or task.protected_test_evaluation_allowed:
        raise ValueError("One-shot Milestone 5A evaluation must remain simulation-only.")
    _verify_frozen_physics_context(simulator, contexts)
    decoded = decode_policy_action(
        torch.as_tensor(normalized_actions, device=simulator.device),
        task,
    )
    if decoded.batch_size != contexts.batch_size:
        raise ValueError("Policy context and complete-action batch sizes must match.")

    knots_world = contexts.frame.vectors_to_world(
        decoded.acceleration_knots_local_m_s2
    )
    initial_state = contexts.initial_state_world
    # Production DDER is outer no-grad, but its existing bending-force kernel
    # uses a local autograd derivative internally.  ``inference_mode`` would
    # disable that derivative; match the proven planner's ``no_grad`` contract.
    with torch.no_grad():
        rollout_time_s = (
            task.cem.duration_max_initial_s
            if evaluation_time_s is None
            else float(evaluation_time_s)
        )
        population = run_variable_population_rollout(
            simulator,
            initial_state,
            knots_world,
            decoded.duration_s,
            task,
            maximum_time_s=rollout_time_s,
            evaluation_time_s=evaluation_time_s,
            command_initial_positions_m=contexts.command_initial_position_world_m,
            command_initial_velocities_m_s=contexts.command_initial_velocity_world_m_s,
            command_yaws_rad=contexts.command_yaw_world_rad,
            target_positions_m=contexts.target_position_world_m(),
            desired_directions=contexts.target_direction_world(),
            initial_uav_positions_m=contexts.command_initial_position_world_m,
            rl_reward_config=rl_reward_config,
        )

    trajectory: DeterministicReplay | None = None
    if record_trajectory:
        if contexts.batch_size != 1:
            raise ValueError("Trajectory recording is intentionally limited to batch size one.")
        with torch.no_grad():
            trajectory = replay_variable_trajectory(
                simulator,
                initial_state,
                knots_world,
                float(decoded.duration_s[0].detach().cpu()),
                task,
                command_initial_position_m=contexts.command_initial_position_world_m,
                command_initial_velocity_m_s=contexts.command_initial_velocity_world_m_s,
                command_yaw_rad=contexts.command_yaw_world_rad,
                target_position_m=contexts.target_position_world_m(),
                desired_direction=contexts.target_direction_world(),
                initial_uav_position_m=contexts.command_initial_position_world_m,
                evaluation_time_s=evaluation_time_s,
            )

    nan = torch.full_like(population.first_entry_time_s, float("nan"))
    valid_hit_time = torch.where(
        population.success, population.first_entry_time_s, nan
    )
    directed = torch.where(
        population.success,
        population.first_entry_directed_speed_m_s,
        population.best_event_directed_speed_m_s,
    )
    direction = torch.where(
        population.success,
        population.first_entry_direction_angle_deg,
        population.best_event_direction_angle_deg,
    )
    components = population.rl_reward_components
    reward = -population.task_cost if components is None else components.total
    return OneShotEpisodeResult(
        reward=reward,
        feasibility_ranking_reward=-population.cost,
        task_success=population.success,
        feasible=population.feasible,
        tip_min_distance_m=population.minimum_tip_target_distance_m,
        max_tip_speed_m_s=population.maximum_tip_speed_m_s,
        hit_time_s=valid_hit_time,
        directed_tip_speed_m_s=directed,
        direction_error_deg=direction,
        first_entry_marker=population.first_entry_marker,
        max_uav_displacement_m=population.maximum_uav_displacement_m,
        max_uav_speed_m_s=population.maximum_uav_speed_m_s,
        max_command_acceleration_m_s2=population.maximum_command_acceleration_m_s2,
        rollout_finite=population.finite,
        task_cost=population.task_cost,
        decoded_action=decoded,
        population_metrics=population,
        trajectory=trajectory,
        reward_components=components,
    )
