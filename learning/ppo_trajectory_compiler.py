"""Compile sequential PPO feedback into physical FullState command trajectories.

The compiler deliberately records commands *after* the sequential environment
has resolved its live-state position/velocity re-anchoring.  Replaying the
result therefore requires neither the actor nor cable feedback.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np
import torch

from planning.rollout import clone_state_batch
from planning.task import CanonicalWhipTask
from simulator.cable.dder import DderState
from simulator.parameters import CableParameters, SimulatorParameters
from simulator.simulator import CoupledSimulator
from simulator.state import SimulatorState
from simulator.uav.state import (
    FullStateCommandSequence,
    ResidualHistoryState,
    UAVState,
)

from .sequential_sac_env import (
    SequentialWhipEnvironment,
    _replace_rows,
    _row_finite,
    diagnostic_scientific_whip_success,
    simple_endpoint_success,
    task_whip_success,
)


StatePostprocessor = Callable[[int, SimulatorState], SimulatorState]


@dataclass(frozen=True, slots=True)
class WhipExecutionMetrics:
    task_success: torch.Tensor
    endpoint_success: torch.Tensor
    legacy_scientific_success: torch.Tensor
    finite: torch.Tensor
    first_entry_marker: torch.Tensor
    first_entry_physics_step: torch.Tensor
    first_entry_time_s: torch.Tensor
    first_entry_tip_distance_m: torch.Tensor
    first_entry_tip_speed_m_s: torch.Tensor
    first_entry_directed_speed_m_s: torch.Tensor
    first_entry_direction_error_deg: torch.Tensor
    minimum_tip_distance_m: torch.Tensor
    maximum_uav_displacement_m: torch.Tensor
    maximum_uav_speed_m_s: torch.Tensor
    maximum_command_acceleration_m_s2: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.task_success.numel())


@dataclass(frozen=True, slots=True)
class CompiledPPOWhip:
    commands: FullStateCommandSequence
    feedback_metrics: WhipExecutionMetrics
    actor_queries: int
    physics_steps: int


@dataclass(frozen=True, slots=True)
class CompiledReplayResult:
    metrics: WhipExecutionMetrics
    final_state: SimulatorState
    reference_maximum_absolute_difference: dict[str, float]


@dataclass(frozen=True, slots=True)
class StateImpulse:
    """One deterministic velocity impulse applied after a physics step."""

    name: str
    physics_step: int
    uav_velocity_delta_world_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cable_velocity_delta_world_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cable_node_start: int = 11

    def __post_init__(self) -> None:
        if self.physics_step < 1:
            raise ValueError("A disturbance physics step must be positive.")
        if not 0 <= self.cable_node_start < 12:
            raise ValueError("cable_node_start must be in [0,11].")
        values = (*self.uav_velocity_delta_world_m_s, *self.cable_velocity_delta_world_m_s)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("State-impulse components must be finite.")

    def __call__(self, physics_step: int, state: SimulatorState) -> SimulatorState:
        if physics_step != self.physics_step:
            return state
        uav_delta = torch.tensor(
            self.uav_velocity_delta_world_m_s,
            dtype=state.uav.velocity_m_s.dtype,
            device=state.uav.velocity_m_s.device,
        ).view(1, 3)
        cable_delta = torch.tensor(
            self.cable_velocity_delta_world_m_s,
            dtype=state.cable.velocities_m_s.dtype,
            device=state.cable.velocities_m_s.device,
        ).view(1, 1, 3)
        cable_velocity = state.cable.velocities_m_s.clone()
        node_count = 12 - self.cable_node_start
        if node_count > 0 and bool(torch.any(cable_delta != 0.0)):
            taper = torch.linspace(
                1.0 / node_count,
                1.0,
                node_count,
                dtype=cable_velocity.dtype,
                device=cable_velocity.device,
            ).view(1, node_count, 1)
            cable_velocity[:, self.cable_node_start :] += taper * cable_delta
        return SimulatorState(
            state.time_s,
            UAVState(
                state.uav.position_m,
                state.uav.velocity_m_s + uav_delta,
                state.uav.orientation_xyzw,
                state.uav.angular_velocity_world_rad_s,
                state.uav.residual_history,
                state.uav.residual_acceleration_m_s2,
            ),
            DderState(
                state.cable.positions_m,
                cable_velocity,
                state.cable.endpoint_orientations,
                state.cable.endpoint_twist_rad,
            ),
        )


def compose_state_postprocessors(
    *postprocessors: StatePostprocessor,
) -> StatePostprocessor | None:
    if not postprocessors:
        return None

    def composed(physics_step: int, state: SimulatorState) -> SimulatorState:
        result = state
        for postprocessor in postprocessors:
            result = postprocessor(physics_step, result)
        return result

    return composed


def scaled_cable_parameters(
    nominal: SimulatorParameters,
    *,
    ei_scale: float = 1.0,
    cb_scale: float = 1.0,
) -> SimulatorParameters:
    if ei_scale <= 0.0 or cb_scale <= 0.0:
        raise ValueError("Cable-parameter scales must be positive.")
    return SimulatorParameters(
        cable=CableParameters(
            EI=torch.as_tensor(nominal.cable.EI) * float(ei_scale),
            Cb=torch.as_tensor(nominal.cable.Cb) * float(cb_scale),
        ),
        uav=nominal.uav,
    )


def metrics_from_feedback_environment(
    environment: SequentialWhipEnvironment,
) -> WhipExecutionMetrics:
    finite = ~environment.failed
    task_success = task_whip_success(
        endpoint_event_found=environment.episode_scientific_event,
        first_entry_marker=environment.episode_first_entry_marker,
        finite=finite,
    )
    legacy = diagnostic_scientific_whip_success(
        endpoint_event_found=task_success,
        first_entry_marker=environment.episode_first_entry_marker,
        maximum_uav_displacement_m=environment.episode_maximum_displacement,
        maximum_uav_speed_m_s=environment.episode_maximum_uav_speed,
        maximum_command_acceleration_m_s2=(
            environment.episode_maximum_command_acceleration
        ),
        finite=finite,
        maximum_uav_displacement_limit_m=environment.task.maximum_uav_displacement_m,
        maximum_uav_speed_limit_m_s=environment.task.maximum_uav_speed_m_s,
        maximum_command_acceleration_limit_m_s2=(
            environment.task.maximum_command_acceleration_m_s2
        ),
    )
    return WhipExecutionMetrics(
        task_success=task_success.clone(),
        endpoint_success=environment.episode_endpoint_success.clone(),
        legacy_scientific_success=legacy.clone(),
        finite=finite.clone(),
        first_entry_marker=environment.episode_first_entry_marker.clone(),
        first_entry_physics_step=environment.episode_first_entry_physics_step.clone(),
        first_entry_time_s=environment.episode_first_entry_time_s.clone(),
        first_entry_tip_distance_m=environment.episode_first_entry_tip_distance.clone(),
        first_entry_tip_speed_m_s=environment.episode_first_entry_tip_speed.clone(),
        first_entry_directed_speed_m_s=(
            environment.episode_first_entry_directed_speed.clone()
        ),
        first_entry_direction_error_deg=(
            environment.episode_first_entry_direction_error_deg.clone()
        ),
        minimum_tip_distance_m=environment.episode_minimum_tip_distance.clone(),
        maximum_uav_displacement_m=environment.episode_maximum_displacement.clone(),
        maximum_uav_speed_m_s=environment.episode_maximum_uav_speed.clone(),
        maximum_command_acceleration_m_s2=(
            environment.episode_maximum_command_acceleration.clone()
        ),
    )


@torch.no_grad()
def compile_ppo_trajectory(
    environment: SequentialWhipEnvironment,
    agent: Any,
) -> CompiledPPOWhip:
    """Roll deterministic feedback PPO once and retain resolved 0.01-s commands."""

    if not environment.record_fullstate_commands:
        raise ValueError("The compiler environment must record FullState commands.")
    was_training = bool(agent.policy.training)
    agent.policy.eval()
    observation = environment.reset()
    for _ in range(environment.control_step_count):
        action = agent.deterministic_action(observation)
        observation = environment.step(action).next_observation
    agent.policy.train(was_training)
    return CompiledPPOWhip(
        environment.recorded_command_sequence(),
        metrics_from_feedback_environment(environment),
        environment.control_step_count,
        environment.physics_step_count,
    )


def reindex_command_sequence(
    commands: FullStateCommandSequence,
    source_indices: torch.Tensor,
) -> FullStateCommandSequence:
    index = torch.as_tensor(
        source_indices, dtype=torch.int64, device=commands.positions_m.device
    ).reshape(-1)
    if index.numel() < 1 or int(index.min()) < 0 or int(index.max()) >= commands.batch_size:
        raise IndexError("Compiled-command source index is outside the batch.")
    return FullStateCommandSequence(
        commands.positions_m[:, index],
        commands.velocities_m_s[:, index],
        commands.accelerations_m_s2[:, index],
        commands.orientations_xyzw[:, index],
        commands.angular_velocities_body_rad_s[:, index],
    )


def _reference_difference(
    reference: dict[str, torch.Tensor] | None,
    index: int,
    state: SimulatorState,
    maxima: dict[str, float],
) -> None:
    if reference is None:
        return
    values = {
        "uav_position_m": state.uav.position_m,
        "uav_velocity_m_s": state.uav.velocity_m_s,
        "uav_orientation_xyzw": state.uav.orientation_xyzw,
        "uav_angular_velocity_world_rad_s": state.uav.angular_velocity_world_rad_s,
        "cable_positions_m": state.cable.positions_m,
        "cable_velocities_m_s": state.cable.velocities_m_s,
    }
    for name, value in values.items():
        difference = float(torch.max(torch.abs(value - reference[name][index])).cpu())
        maxima[name] = max(maxima.get(name, 0.0), difference)


@torch.no_grad()
def replay_compiled_trajectory(
    simulator: CoupledSimulator,
    initial_state: SimulatorState,
    task: CanonicalWhipTask,
    commands: FullStateCommandSequence,
    *,
    parameters: SimulatorParameters | None = None,
    state_postprocessor: StatePostprocessor | None = None,
    reference_state_trajectory: dict[str, torch.Tensor] | None = None,
) -> CompiledReplayResult:
    """Execute fixed FullState commands with no actor, context, or feedback."""

    batch_size = commands.batch_size
    state = clone_state_batch(initial_state, batch_size)
    reset_state = clone_state_batch(initial_state, batch_size)
    active_parameters = simulator.parameters if parameters is None else parameters
    target = torch.tensor(
        task.target_position_m, dtype=simulator.dtype, device=simulator.device
    ).view(1, 3).expand(batch_size, -1)
    direction = torch.tensor(
        task.desired_direction, dtype=simulator.dtype, device=simulator.device
    ).view(1, 3).expand(batch_size, -1)
    failed = torch.zeros(batch_size, dtype=torch.bool, device=simulator.device)
    endpoint_success = torch.zeros_like(failed)
    task_event = torch.zeros_like(failed)
    first_entry_marker = torch.zeros(
        batch_size, dtype=torch.int64, device=simulator.device
    )
    first_entry_step = torch.zeros_like(first_entry_marker)
    nan = torch.full(
        (batch_size,), torch.nan, dtype=simulator.dtype, device=simulator.device
    )
    first_entry_time = nan.clone()
    first_entry_distance = nan.clone()
    first_entry_speed = nan.clone()
    first_entry_directed_speed = nan.clone()
    first_entry_direction_error = nan.clone()
    minimum_tip_distance = torch.linalg.vector_norm(
        state.cable.positions_m[:, -1] - target, dim=-1
    )
    maximum_displacement = torch.zeros_like(minimum_tip_distance)
    maximum_speed = torch.linalg.vector_norm(state.uav.velocity_m_s, dim=-1)
    maximum_command_acceleration = torch.zeros_like(minimum_tip_distance)
    marker_ids = torch.arange(1, 11, dtype=torch.int64, device=simulator.device)[None]
    reference_maxima: dict[str, float] = {}
    _reference_difference(reference_state_trajectory, 0, state, reference_maxima)

    for index in range(commands.step_count):
        command = commands.command_at(index)
        maximum_command_acceleration = torch.maximum(
            maximum_command_acceleration,
            torch.linalg.vector_norm(command.acceleration_m_s2, dim=-1),
        )
        propagated = simulator._propagate(
            state, command, active_parameters, create_graph=False
        )
        physics_step = index + 1
        if state_postprocessor is not None:
            propagated = state_postprocessor(physics_step, propagated)
        finite = _row_finite(propagated)
        failed |= ~finite
        state = _replace_rows(propagated, reset_state, failed)
        eligible = ~failed
        tip_position = state.cable.positions_m[:, -1]
        tip_velocity = state.cable.velocities_m_s[:, -1]
        distance = torch.linalg.vector_norm(tip_position - target, dim=-1)
        speed = torch.linalg.vector_norm(tip_velocity, dim=-1)
        directed_speed = torch.sum(tip_velocity * direction, dim=-1)
        direction_error = torch.rad2deg(
            torch.acos((directed_speed / speed.clamp_min(torch.finfo(speed.dtype).eps)).clamp(-1.0, 1.0))
        )
        minimum_tip_distance = torch.where(
            eligible, torch.minimum(minimum_tip_distance, distance), minimum_tip_distance
        )
        displacement = torch.linalg.vector_norm(
            state.uav.position_m - reset_state.uav.position_m, dim=-1
        )
        uav_speed = torch.linalg.vector_norm(state.uav.velocity_m_s, dim=-1)
        maximum_displacement = torch.where(
            eligible, torch.maximum(maximum_displacement, displacement), maximum_displacement
        )
        maximum_speed = torch.where(
            eligible, torch.maximum(maximum_speed, uav_speed), maximum_speed
        )
        point_endpoint = eligible & simple_endpoint_success(
            tip_position,
            tip_velocity,
            target,
            direction,
            radius_m=task.success_radius_m,
            minimum_directed_speed_m_s=task.minimum_directed_speed_m_s,
            maximum_direction_error_deg=task.maximum_direction_error_deg,
        )
        marker_distance = torch.linalg.vector_norm(
            state.cable.positions_m[:, 2:12] - target[:, None], dim=-1
        )
        inside = marker_distance <= task.success_radius_m
        candidate_marker = torch.where(
            inside, marker_ids, torch.full_like(marker_ids, 11)
        ).amin(dim=1)
        new_entry = eligible & (first_entry_marker == 0) & (candidate_marker <= 10)
        first_entry_marker = torch.where(
            new_entry, candidate_marker, first_entry_marker
        )
        step_tensor = torch.full_like(first_entry_step, physics_step)
        first_entry_step = torch.where(new_entry, step_tensor, first_entry_step)
        first_entry_time = torch.where(
            new_entry,
            step_tensor.to(simulator.dtype) * simulator.dt_s,
            first_entry_time,
        )
        first_entry_distance = torch.where(new_entry, distance, first_entry_distance)
        first_entry_speed = torch.where(new_entry, speed, first_entry_speed)
        first_entry_directed_speed = torch.where(
            new_entry, directed_speed, first_entry_directed_speed
        )
        first_entry_direction_error = torch.where(
            new_entry, direction_error, first_entry_direction_error
        )
        endpoint_success |= point_endpoint
        task_event |= point_endpoint & new_entry & (candidate_marker == 10)
        _reference_difference(
            reference_state_trajectory, physics_step, state, reference_maxima
        )

    finite = ~failed
    task_success = task_whip_success(
        endpoint_event_found=task_event,
        first_entry_marker=first_entry_marker,
        finite=finite,
    )
    legacy_success = diagnostic_scientific_whip_success(
        endpoint_event_found=task_success,
        first_entry_marker=first_entry_marker,
        maximum_uav_displacement_m=maximum_displacement,
        maximum_uav_speed_m_s=maximum_speed,
        maximum_command_acceleration_m_s2=maximum_command_acceleration,
        finite=finite,
        maximum_uav_displacement_limit_m=task.maximum_uav_displacement_m,
        maximum_uav_speed_limit_m_s=task.maximum_uav_speed_m_s,
        maximum_command_acceleration_limit_m_s2=(
            task.maximum_command_acceleration_m_s2
        ),
    )
    return CompiledReplayResult(
        WhipExecutionMetrics(
            task_success,
            endpoint_success,
            legacy_success,
            finite,
            first_entry_marker,
            first_entry_step,
            first_entry_time,
            first_entry_distance,
            first_entry_speed,
            first_entry_directed_speed,
            first_entry_direction_error,
            minimum_tip_distance,
            maximum_displacement,
            maximum_speed,
            maximum_command_acceleration,
        ),
        state,
        reference_maxima,
    )


def _finite_summary(values: torch.Tensor) -> dict[str, float | None]:
    array = values[torch.isfinite(values)].detach().cpu().double().numpy()
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "p05": None, "p95": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def summarize_execution_metrics(metrics: WhipExecutionMetrics) -> dict[str, Any]:
    success = metrics.task_success
    entered = metrics.first_entry_marker > 0
    return {
        "episodes": metrics.batch_size,
        "task_successes": int(success.sum().cpu()),
        "task_success_rate": float(success.float().mean().cpu()),
        "endpoint_successes": int(metrics.endpoint_success.sum().cpu()),
        "legacy_scientific_successes": int(
            metrics.legacy_scientific_success.sum().cpu()
        ),
        "finite_rate": float(metrics.finite.float().mean().cpu()),
        "tip_first_count": int((metrics.first_entry_marker == 10).sum().cpu()),
        "first_entry_tip_distance_m": _finite_summary(
            metrics.first_entry_tip_distance_m[entered]
        ),
        "first_entry_directed_speed_m_s": _finite_summary(
            metrics.first_entry_directed_speed_m_s[entered]
        ),
        "first_entry_direction_error_deg": _finite_summary(
            metrics.first_entry_direction_error_deg[entered]
        ),
        "first_entry_time_s": _finite_summary(metrics.first_entry_time_s[entered]),
        "minimum_tip_distance_m": _finite_summary(metrics.minimum_tip_distance_m),
        "maximum_uav_displacement_m": _finite_summary(
            metrics.maximum_uav_displacement_m
        ),
        "maximum_uav_speed_m_s": _finite_summary(metrics.maximum_uav_speed_m_s),
        "maximum_command_acceleration_m_s2": _finite_summary(
            metrics.maximum_command_acceleration_m_s2
        ),
    }


def metric_maximum_absolute_differences(
    first: WhipExecutionMetrics,
    second: WhipExecutionMetrics,
) -> dict[str, float | int]:
    differences: dict[str, float | int] = {}
    for name in (
        "task_success",
        "endpoint_success",
        "legacy_scientific_success",
        "finite",
        "first_entry_marker",
        "first_entry_physics_step",
    ):
        left = getattr(first, name)
        right = getattr(second, name)
        differences[name] = int(torch.count_nonzero(left != right).cpu())
    for name in (
        "first_entry_time_s",
        "first_entry_tip_distance_m",
        "first_entry_tip_speed_m_s",
        "first_entry_directed_speed_m_s",
        "first_entry_direction_error_deg",
        "minimum_tip_distance_m",
        "maximum_uav_displacement_m",
        "maximum_uav_speed_m_s",
        "maximum_command_acceleration_m_s2",
    ):
        left = getattr(first, name)
        right = getattr(second, name)
        both_nan = torch.isnan(left) & torch.isnan(right)
        delta = torch.where(both_nan, torch.zeros_like(left), torch.abs(left - right))
        differences[name] = float(torch.nan_to_num(delta, nan=math.inf).max().cpu())
    return differences
