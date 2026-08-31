"""Training-only hard-safety prefix evaluation for one-shot policies.

The production simulator is not changed.  Rows are propagated through the
normal frozen model until the first hard safety violation.  Task statistics
exclude the violating state and every later state, while the first violation
is retained as a binary safety-learning target and diagnostic record.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from planning.cem_task import VariableDurationWhipTask
from planning.variable_duration import VariableWhipAccumulator, variable_duration_fullstate
from planning.rollout import clone_state_batch
from planning.rl_reward import RLWhipRewardConfig
from simulator.cable.dder import DderState
from simulator.simulator import CoupledSimulator
from simulator.state import SimulatorState
from simulator.uav.state import FullStateCommand, ResidualHistoryState, UAVState

from .one_shot_env import OneShotEpisodeResult, _verify_frozen_physics_context
from .policy_action import decode_policy_action
from .policy_context import PolicyContext


SAFETY_FAILURE_DISPLACEMENT = 1
SAFETY_FAILURE_SPEED = 2
SAFETY_FAILURE_COMMAND_ACCELERATION = 4
SAFETY_FAILURE_NONFINITE = 8
HARD_SAFETY_FAILURE_QUARANTINE = "HARD_SAFETY_FAILURE_QUARANTINE"
RL_V2_HARD_SAFETY_PREFIX = "RL_V2_HARD_SAFETY_PREFIX"


def _finite_state(state: SimulatorState) -> torch.Tensor:
    values = (
        state.uav.position_m,
        state.uav.velocity_m_s,
        state.uav.orientation_xyzw,
        state.uav.angular_velocity_world_rad_s,
        state.cable.positions_m,
        state.cable.velocities_m_s,
    )
    result = torch.ones(
        state.uav.batch_size, dtype=torch.bool, device=state.uav.position_m.device
    )
    for value in values:
        result &= torch.isfinite(value).reshape(value.shape[0], -1).all(dim=1)
    return result


def _select_tensor(
    candidate: torch.Tensor | None,
    previous: torch.Tensor | None,
    use_candidate: torch.Tensor,
) -> torch.Tensor | None:
    if candidate is None or previous is None:
        if candidate is not previous:
            raise RuntimeError("Simulator optional-state structure changed during rollout.")
        return candidate
    shape = (use_candidate.shape[0],) + (1,) * (candidate.ndim - 1)
    return torch.where(use_candidate.reshape(shape), candidate, previous)


def _select_state(
    candidate: SimulatorState,
    previous: SimulatorState,
    use_candidate: torch.Tensor,
) -> SimulatorState:
    candidate_history = candidate.uav.residual_history
    previous_history = previous.uav.residual_history
    if candidate_history is None or previous_history is None:
        if candidate_history is not previous_history:
            raise RuntimeError("Residual-history structure changed during rollout.")
        history = candidate_history
    else:
        history = ResidualHistoryState(
            _select_tensor(
                candidate_history.features, previous_history.features, use_candidate
            )
        )
    uav = UAVState(
        _select_tensor(candidate.uav.position_m, previous.uav.position_m, use_candidate),
        _select_tensor(candidate.uav.velocity_m_s, previous.uav.velocity_m_s, use_candidate),
        _select_tensor(
            candidate.uav.orientation_xyzw,
            previous.uav.orientation_xyzw,
            use_candidate,
        ),
        _select_tensor(
            candidate.uav.angular_velocity_world_rad_s,
            previous.uav.angular_velocity_world_rad_s,
            use_candidate,
        ),
        history,
        _select_tensor(
            candidate.uav.residual_acceleration_m_s2,
            previous.uav.residual_acceleration_m_s2,
            use_candidate,
        ),
    )
    cable = DderState(
        _select_tensor(
            candidate.cable.positions_m, previous.cable.positions_m, use_candidate
        ),
        _select_tensor(
            candidate.cable.velocities_m_s, previous.cable.velocities_m_s, use_candidate
        ),
        _select_tensor(
            candidate.cable.endpoint_orientations,
            previous.cable.endpoint_orientations,
            use_candidate,
        ),
        _select_tensor(
            candidate.cable.endpoint_twist_rad,
            previous.cable.endpoint_twist_rad,
            use_candidate,
        ),
    )
    return SimulatorState(candidate.time_s, uav, cable)


def _quarantine_command(
    command: FullStateCommand,
    state: SimulatorState,
    active: torch.Tensor,
) -> FullStateCommand:
    """Replace failed rows by a benign zero-error hold solely for tensor shape."""

    vector_mask = active[:, None]
    zero = torch.zeros_like(command.acceleration_m_s2)
    return FullStateCommand(
        torch.where(vector_mask, command.position_m, state.uav.position_m),
        torch.where(vector_mask, command.velocity_m_s, state.uav.velocity_m_s),
        torch.where(vector_mask, command.acceleration_m_s2, zero),
        torch.where(active[:, None], command.orientation_xyzw, state.uav.orientation_xyzw),
        torch.where(vector_mask, command.angular_velocity_body_rad_s, zero),
    )


@dataclass(frozen=True, slots=True)
class QuarantineDiagnostics:
    contract: str
    reward_contract: str
    safety_failed: torch.Tensor
    safety_cost: torch.Tensor
    failure_time_s: torch.Tensor
    failure_gate_mask: torch.Tensor
    failure_uav_position_m: torch.Tensor
    failure_uav_velocity_m_s: torch.Tensor
    failure_cable_positions_m: torch.Tensor
    failure_cable_velocities_m_s: torch.Tensor
    propagated_step_count: torch.Tensor
    uav_positions_m: torch.Tensor | None = None
    uav_velocities_m_s: torch.Tensor | None = None
    cable_positions_m: torch.Tensor | None = None
    cable_velocities_m_s: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class QuarantinedEpisodeResult:
    episode: OneShotEpisodeResult
    diagnostics: QuarantineDiagnostics


def record_full_batch_trajectory(
    simulator: CoupledSimulator,
    contexts: PolicyContext,
    normalized_actions: torch.Tensor,
    task: VariableDurationWhipTask,
) -> dict[str, torch.Tensor]:
    """Record an ordinary, unquarantined production batch for verification."""

    _verify_frozen_physics_context(simulator, contexts)
    decoded = decode_policy_action(
        torch.as_tensor(normalized_actions, device=simulator.device), task
    )
    knots_world = contexts.frame.vectors_to_world(
        decoded.acceleration_knots_local_m_s2
    )
    command = variable_duration_fullstate(
        knots_world,
        decoded.duration_s,
        initial_position_m=contexts.command_initial_position_world_m,
        initial_velocity_m_s=contexts.command_initial_velocity_world_m_s,
        yaw_rad=contexts.command_yaw_world_rad,
        maximum_time_s=task.cem.duration_max_initial_s,
        dt_s=simulator.dt_s,
    )
    state = clone_state_batch(contexts.initial_state_world, contexts.batch_size)
    uav_p = [state.uav.position_m.clone()]
    uav_v = [state.uav.velocity_m_s.clone()]
    cable_p = [state.cable.positions_m.clone()]
    cable_v = [state.cable.velocities_m_s.clone()]
    sequence = command.simulator_sequence()
    with torch.no_grad():
        for index in range(sequence.step_count):
            state = simulator._propagate(  # noqa: SLF001
                state,
                sequence.command_at(index),
                simulator.parameters,
                create_graph=False,
            )
            uav_p.append(state.uav.position_m.clone())
            uav_v.append(state.uav.velocity_m_s.clone())
            cable_p.append(state.cable.positions_m.clone())
            cable_v.append(state.cable.velocities_m_s.clone())
    return {
        "uav_positions_m": torch.stack(uav_p),
        "uav_velocities_m_s": torch.stack(uav_v),
        "cable_positions_m": torch.stack(cable_p),
        "cable_velocities_m_s": torch.stack(cable_v),
    }


def evaluate_open_loop_batch_quarantined(
    simulator: CoupledSimulator,
    contexts: PolicyContext,
    normalized_actions: torch.Tensor,
    task: VariableDurationWhipTask,
    *,
    rl_reward_config: RLWhipRewardConfig,
    record_trajectories: bool = False,
) -> QuarantinedEpisodeResult:
    """Evaluate complete actions using the valid hard-safety prefix only."""

    if task.model_freeze != "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI":
        raise ValueError("Quarantine evaluation requires the frozen PRE_MPPI model.")
    if rl_reward_config.profile != "rl_whip_reward_v2":
        raise ValueError("Quarantine evaluation requires unchanged rl_whip_reward_v2.")
    _verify_frozen_physics_context(simulator, contexts)
    decoded = decode_policy_action(
        torch.as_tensor(normalized_actions, device=simulator.device), task
    )
    if decoded.batch_size != contexts.batch_size:
        raise ValueError("Policy context and action batches must match.")
    knots_world = contexts.frame.vectors_to_world(
        decoded.acceleration_knots_local_m_s2
    )
    command = variable_duration_fullstate(
        knots_world,
        decoded.duration_s,
        initial_position_m=contexts.command_initial_position_world_m,
        initial_velocity_m_s=contexts.command_initial_velocity_world_m_s,
        yaw_rad=contexts.command_yaw_world_rad,
        maximum_time_s=task.cem.duration_max_initial_s,
        dt_s=simulator.dt_s,
    )
    state = clone_state_batch(contexts.initial_state_world, contexts.batch_size)
    accumulator = VariableWhipAccumulator(
        task,
        decoded.duration_s.clone(),
        knots_world,
        target_positions_m=contexts.target_position_world_m(),
        desired_directions=contexts.target_direction_world(),
        initial_uav_positions_m=contexts.command_initial_position_world_m,
        rl_reward_config=rl_reward_config,
    )
    accumulator.observe(
        0.0,
        uav_position_m=state.uav.position_m,
        uav_velocity_m_s=state.uav.velocity_m_s,
        cable_positions_m=state.cable.positions_m,
        cable_velocities_m_s=state.cable.velocities_m_s,
    )
    batch = contexts.batch_size
    dtype, device = state.uav.position_m.dtype, state.uav.position_m.device
    failed = torch.zeros(batch, dtype=torch.bool, device=device)
    failure_time = torch.full((batch,), float("nan"), dtype=dtype, device=device)
    gate_mask = torch.zeros(batch, dtype=torch.int64, device=device)
    failure_uav_p = torch.full_like(state.uav.position_m, float("nan"))
    failure_uav_v = torch.full_like(state.uav.velocity_m_s, float("nan"))
    failure_cable_p = torch.full_like(state.cable.positions_m, float("nan"))
    failure_cable_v = torch.full_like(state.cable.velocities_m_s, float("nan"))
    propagated_steps = torch.zeros(batch, dtype=torch.int64, device=device)
    uav_p_history = [state.uav.position_m.clone()] if record_trajectories else None
    uav_v_history = [state.uav.velocity_m_s.clone()] if record_trajectories else None
    cable_p_history = [state.cable.positions_m.clone()] if record_trajectories else None
    cable_v_history = [state.cable.velocities_m_s.clone()] if record_trajectories else None

    sequence = command.simulator_sequence()
    with torch.no_grad():
        for index in range(sequence.step_count):
            step_command = sequence.command_at(index)
            command_bad = (
                torch.linalg.vector_norm(step_command.acceleration_m_s2, dim=-1)
                > task.maximum_command_acceleration_m_s2 + 1.0e-5
            )
            new_command_failure = ~failed & command_bad
            if bool(new_command_failure.any()):
                now = torch.full_like(failure_time, index * simulator.dt_s)
                failure_time = torch.where(new_command_failure, now, failure_time)
                gate_mask = torch.where(
                    new_command_failure,
                    torch.full_like(gate_mask, SAFETY_FAILURE_COMMAND_ACCELERATION),
                    gate_mask,
                )
                failure_uav_p = torch.where(
                    new_command_failure[:, None], state.uav.position_m, failure_uav_p
                )
                failure_uav_v = torch.where(
                    new_command_failure[:, None], state.uav.velocity_m_s, failure_uav_v
                )
                failure_cable_p = torch.where(
                    new_command_failure[:, None, None],
                    state.cable.positions_m,
                    failure_cable_p,
                )
                failure_cable_v = torch.where(
                    new_command_failure[:, None, None],
                    state.cable.velocities_m_s,
                    failure_cable_v,
                )
                accumulator.durations = torch.where(
                    new_command_failure,
                    torch.full_like(accumulator.durations, index * simulator.dt_s),
                    accumulator.durations,
                )
                failed |= new_command_failure

            propagate_mask = ~failed
            masked_command = _quarantine_command(step_command, state, propagate_mask)
            candidate = simulator._propagate(  # noqa: SLF001
                state, masked_command, simulator.parameters, create_graph=False
            )
            propagated_steps += propagate_mask.to(torch.int64)
            finite = _finite_state(candidate)
            displacement = torch.linalg.vector_norm(
                candidate.uav.position_m
                - contexts.command_initial_position_world_m,
                dim=-1,
            )
            speed = torch.linalg.vector_norm(candidate.uav.velocity_m_s, dim=-1)
            displacement_bad = displacement > task.maximum_uav_displacement_m
            speed_bad = speed > task.maximum_uav_speed_m_s
            new_failure = propagate_mask & (~finite | displacement_bad | speed_bad)
            if bool(new_failure.any()):
                bits = (
                    displacement_bad.to(torch.int64) * SAFETY_FAILURE_DISPLACEMENT
                    + speed_bad.to(torch.int64) * SAFETY_FAILURE_SPEED
                    + (~finite).to(torch.int64) * SAFETY_FAILURE_NONFINITE
                )
                now = torch.full_like(failure_time, (index + 1) * simulator.dt_s)
                failure_time = torch.where(new_failure, now, failure_time)
                gate_mask = torch.where(new_failure, bits, gate_mask)
                failure_uav_p = torch.where(
                    new_failure[:, None], candidate.uav.position_m, failure_uav_p
                )
                failure_uav_v = torch.where(
                    new_failure[:, None], candidate.uav.velocity_m_s, failure_uav_v
                )
                failure_cable_p = torch.where(
                    new_failure[:, None, None],
                    candidate.cable.positions_m,
                    failure_cable_p,
                )
                failure_cable_v = torch.where(
                    new_failure[:, None, None],
                    candidate.cable.velocities_m_s,
                    failure_cable_v,
                )
                # The task-valid prefix ends at the previous state.
                accumulator.durations = torch.where(
                    new_failure,
                    torch.full_like(accumulator.durations, index * simulator.dt_s),
                    accumulator.durations,
                )

            accept = propagate_mask & ~new_failure
            state = _select_state(candidate, state, accept)
            time_s = (index + 1) * simulator.dt_s
            accumulator.observe(
                time_s,
                uav_position_m=state.uav.position_m,
                uav_velocity_m_s=state.uav.velocity_m_s,
                cable_positions_m=state.cable.positions_m,
                cable_velocities_m_s=state.cable.velocities_m_s,
            )
            if bool(new_failure.any()):
                # Preserve the measured first violation for the unchanged v2
                # safety penalty while excluding it from task/event metrics.
                safe_displacement = torch.nan_to_num(
                    displacement, nan=float("inf"), posinf=float("inf"), neginf=float("inf")
                )
                safe_speed = torch.nan_to_num(
                    speed, nan=float("inf"), posinf=float("inf"), neginf=float("inf")
                )
                accumulator.maximum_uav_displacement = torch.where(
                    new_failure,
                    safe_displacement,
                    accumulator.maximum_uav_displacement,
                )
                accumulator.maximum_uav_speed = torch.where(
                    new_failure, safe_speed, accumulator.maximum_uav_speed
                )
                accumulator.finite = torch.where(
                    new_failure, finite, accumulator.finite
                )
                accumulator.success &= ~new_failure
                failed |= new_failure
            if record_trajectories:
                assert uav_p_history is not None and uav_v_history is not None
                assert cable_p_history is not None and cable_v_history is not None
                uav_p_history.append(state.uav.position_m.clone())
                uav_v_history.append(state.uav.velocity_m_s.clone())
                cable_p_history.append(state.cable.positions_m.clone())
                cable_v_history.append(state.cable.velocities_m_s.clone())

    population = accumulator.finalize(command)
    population = replace(
        population,
        success=population.success & ~failed,
        feasible=population.feasible & ~failed,
    )
    nan = torch.full_like(population.first_entry_time_s, float("nan"))
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
    if components is None:
        raise RuntimeError("Hard-safety prefix requires RL reward components.")
    episode = OneShotEpisodeResult(
        reward=components.total,
        feasibility_ranking_reward=-population.cost,
        task_success=population.success,
        feasible=population.feasible,
        tip_min_distance_m=population.minimum_tip_target_distance_m,
        max_tip_speed_m_s=population.maximum_tip_speed_m_s,
        hit_time_s=torch.where(population.success, population.first_entry_time_s, nan),
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
        trajectory=None,
        reward_components=components,
    )
    diagnostics = QuarantineDiagnostics(
        HARD_SAFETY_FAILURE_QUARANTINE,
        RL_V2_HARD_SAFETY_PREFIX,
        failed,
        failed.to(dtype),
        failure_time,
        gate_mask,
        failure_uav_p,
        failure_uav_v,
        failure_cable_p,
        failure_cable_v,
        propagated_steps,
        None if uav_p_history is None else torch.stack(uav_p_history),
        None if uav_v_history is None else torch.stack(uav_v_history),
        None if cable_p_history is None else torch.stack(cable_p_history),
        None if cable_v_history is None else torch.stack(cable_v_history),
    )
    return QuarantinedEpisodeResult(episode, diagnostics)
