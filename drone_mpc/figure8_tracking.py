"""Continuous free-tip figure-eight tracking with the existing DDER-MPPI backend.

This module deliberately defines a new task beside the impact/whip task.  It
reuses the production point-mass root plant, accelerated DDER rollouts, MPPI
sampling/update rule, warm-start shifting, and existing endpoint-only state
conditioning.  No impact/contact objective or parameter adaptation is used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
import json
import math
from pathlib import Path
import time
from typing import Callable, Literal

import numpy as np
import torch

from cable_twin.shared.dder import DderState

from .history_observer import (
    DderHistoryObserver,
    EndpointHistoryObservation,
    HistoryObserverUpdate,
)
from .mppi import (
    MppiSettings,
    _bound_vectors,
    _sample_perturbations,
    interpolate_control_knots,
)
from .receding_mppi import endpoint_conditioned_state, shift_control_knots
from .simulator import DroneCableState, TensorRollout, WhipSimulator


ObservationMode = Literal["full", "endpoint", "history"]
ExecutionObservationMode = Literal["full", "endpoint", "history", "mixed"]
ProgressCallback = Callable[[str], None]
CancellationCallback = Callable[[], bool]
PauseCallback = Callable[[], None]
ObservationModeProvider = Callable[[], ObservationMode]
PlantStateHook = Callable[[float, DroneCableState], DroneCableState]


@dataclass(frozen=True, slots=True)
class Figure8Reference:
    """Flat closed figure-eight path with no prescribed timing."""

    center_position_m: tuple[float, float, float]
    amplitude_x_m: float = 0.7
    amplitude_y_m: float = 0.5

    def __post_init__(self) -> None:
        center = np.asarray(self.center_position_m, dtype=np.float64)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("Figure-eight center must contain three finite values.")
        values = (self.amplitude_x_m, self.amplitude_y_m)
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Figure-eight amplitudes must be positive.")

    def point_numpy(self, progress_cycles: np.ndarray | float) -> np.ndarray:
        values = np.asarray(progress_cycles, dtype=np.float64)
        phase = 2.0 * math.pi * values
        center = np.asarray(self.center_position_m, dtype=np.float64)
        x_offset = self.amplitude_x_m * np.sin(phase)
        y_offset = self.amplitude_y_m * np.sin(2.0 * phase)
        return np.stack(
            (
                center[0] + x_offset,
                center[1] + y_offset,
                np.full_like(values, center[2]),
            ),
            axis=-1,
        )

    def point_torch(self, progress_cycles: torch.Tensor) -> torch.Tensor:
        center = torch.as_tensor(
            self.center_position_m,
            dtype=progress_cycles.dtype,
            device=progress_cycles.device,
        )
        phase = 2.0 * math.pi * progress_cycles
        x_offset = self.amplitude_x_m * torch.sin(phase)
        y_offset = self.amplitude_y_m * torch.sin(2.0 * phase)
        z = torch.full_like(progress_cycles, center[2])
        return torch.stack((center[0] + x_offset, center[1] + y_offset, z), dim=-1)

    def unit_tangent_torch(self, progress_cycles: torch.Tensor) -> torch.Tensor:
        """Return the forward unit tangent of the flat path."""

        phase = 2.0 * math.pi * progress_cycles
        derivative = torch.stack(
            (
                2.0 * math.pi * self.amplitude_x_m * torch.cos(phase),
                4.0 * math.pi * self.amplitude_y_m * torch.cos(2.0 * phase),
                torch.zeros_like(progress_cycles),
            ),
            dim=-1,
        )
        return derivative / torch.clamp(
            torch.linalg.vector_norm(derivative, dim=-1, keepdim=True),
            min=1.0e-9,
        )

    def unit_tangent_numpy(self, progress_cycles: np.ndarray | float) -> np.ndarray:
        """Return the forward unit tangent for execution diagnostics."""

        values = np.asarray(progress_cycles, dtype=np.float64)
        phase = 2.0 * math.pi * values
        derivative = np.stack(
            (
                2.0 * math.pi * self.amplitude_x_m * np.cos(phase),
                4.0 * math.pi * self.amplitude_y_m * np.cos(2.0 * phase),
                np.zeros_like(values),
            ),
            axis=-1,
        )
        return derivative / np.maximum(
            np.linalg.norm(derivative, axis=-1, keepdims=True), 1.0e-9
        )

    def path_numpy(self, sample_count: int = 721) -> np.ndarray:
        if sample_count < 8:
            raise ValueError("Figure-eight path needs at least eight samples.")
        return self.point_numpy(np.linspace(0.0, 1.0, sample_count))

    @property
    def approximate_length_m(self) -> float:
        path = self.path_numpy(1441)
        return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))

    def project_sequence_numpy(
        self,
        positions_m: np.ndarray,
        start_progress_cycles: float,
        *,
        sample_count: int = 720,
        backward_fraction: float = 0.03,
        forward_fraction: float = 0.25,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Causally project positions onto the nearby forward branch."""

        positions = np.asarray(positions_m, dtype=np.float64).reshape(-1, 3)
        path = self.point_numpy(np.arange(sample_count) / sample_count)
        progress = float(start_progress_cycles)
        references: list[np.ndarray] = []
        progresses: list[float] = []
        errors: list[float] = []
        backward = max(1, int(round(backward_fraction * sample_count)))
        forward = max(2, int(round(forward_fraction * sample_count)))
        offsets = np.arange(-backward, forward + 1, dtype=np.int64)
        for position in positions:
            base = int(round(progress * sample_count))
            unwrapped = base + offsets
            candidates = path[np.mod(unwrapped, sample_count)]
            distances = np.linalg.norm(candidates - position[None], axis=1)
            selected = int(np.argmin(distances))
            selected_progress = float(unwrapped[selected] / sample_count)
            references.append(candidates[selected])
            progress = max(progress, selected_progress)
            progresses.append(progress)
            errors.append(float(distances[selected]))
        return (
            np.asarray(references),
            np.asarray(progresses),
            np.asarray(errors),
        )


@dataclass(frozen=True, slots=True)
class TrackingCostSettings:
    """Geometric contouring objective with weak motion regularization."""

    tracking_weight: float = 5000.0
    path_progress_reward_weight: float = 5.0
    control_effort_weight: float = 2.0e-4
    control_smoothness_weight: float = 5.0e-2
    tip_motion_smoothness_weight: float = 5.0e-1
    speed_limit_weight: float = 80.0
    ground_safety_weight: float = 100.0
    ground_clearance_m: float = 0.02

    def __post_init__(self) -> None:
        positive = (
            self.tracking_weight,
            self.path_progress_reward_weight,
            self.speed_limit_weight,
            self.ground_safety_weight,
            self.ground_clearance_m,
        )
        nonnegative = (
            self.control_effort_weight,
            self.control_smoothness_weight,
            self.tip_motion_smoothness_weight,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Tracking and safety scales must be finite and positive.")
        if any(not math.isfinite(value) or value < 0.0 for value in nonnegative):
            raise ValueError("Tracking regularization weights must be non-negative.")


@dataclass(frozen=True, slots=True)
class Figure8ExecutionSettings:
    replan_interval_s: float = 0.10
    observation_mode: ObservationMode = "full"
    duration_s: float | None = None
    realtime_pacing: bool = True
    periodic_swing_bootstrap: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.replan_interval_s) or self.replan_interval_s <= 0.0:
            raise ValueError("Figure-eight replanning interval must be positive.")
        if self.observation_mode not in {"full", "endpoint", "history"}:
            raise ValueError(
                "Observation mode must be 'full', 'endpoint', or 'history'."
            )
        if self.duration_s is not None and (
            not math.isfinite(self.duration_s) or self.duration_s <= 0.0
        ):
            raise ValueError("Figure-eight duration must be positive when specified.")


@dataclass(frozen=True, slots=True)
class TrackingPlan:
    controls_m_s2: np.ndarray
    control_knots_m_s2: np.ndarray
    cost: float
    terms: dict[str, float]
    predicted_time_s: np.ndarray
    predicted_drone_positions_m: np.ndarray
    predicted_tip_positions_m: np.ndarray
    predicted_reference_positions_m: np.ndarray
    initialization: str


@dataclass(frozen=True, slots=True)
class Figure8LiveUpdate:
    update_index: int
    time_s: float
    planning_wall_time_s: float
    drone_position_m: np.ndarray
    drone_velocity_m_s: np.ndarray
    cable_positions_m: np.ndarray
    cable_velocities_m_s: np.ndarray
    reference_tip_position_m: np.ndarray
    path_progress_cycles: float
    command_m_s2: np.ndarray
    tracking_error_m: float
    rmse_m: float
    mean_error_m: float
    maximum_error_m: float
    p95_error_m: float
    mean_planning_wall_time_s: float
    p95_planning_wall_time_s: float
    drone_displacement_from_start_m: float
    maximum_drone_displacement_from_start_m: float
    drone_horizontal_displacement_from_start_m: float
    maximum_drone_horizontal_displacement_from_start_m: float
    drone_vertical_displacement_m: float
    recent_time_s: np.ndarray
    recent_actual_tip_positions_m: np.ndarray
    recent_reference_tip_positions_m: np.ndarray
    recent_tracking_errors_m: np.ndarray
    observation_mode: ObservationMode
    observer_ready: bool
    observer_accepted: bool
    observer_history_rmse_m: float
    observer_wall_time_s: float


@dataclass(frozen=True, slots=True)
class Figure8Execution:
    time_s: np.ndarray
    reference_tip_positions_m: np.ndarray
    path_progress_cycles: np.ndarray
    actual_tip_positions_m: np.ndarray
    actual_tip_velocities_m_s: np.ndarray
    drone_positions_m: np.ndarray
    drone_velocities_m_s: np.ndarray
    commands_m_s2: np.ndarray
    cable_positions_m: np.ndarray
    cable_velocities_m_s: np.ndarray
    estimated_cable_positions_m: np.ndarray
    estimated_cable_velocities_m_s: np.ndarray
    tracking_errors_m: np.ndarray
    planning_wall_times_s: np.ndarray
    sequential_update_wall_times_s: np.ndarray
    observer_update_times_s: np.ndarray
    observer_history_rmse_before_m: np.ndarray
    observer_history_rmse_after_m: np.ndarray
    observer_update_accepted: np.ndarray
    observation_modes: np.ndarray
    observation_mode: ExecutionObservationMode
    reference: Figure8Reference
    controller_model_sha256: str
    plant_model_sha256: str

    @property
    def summary(self) -> dict[str, float]:
        errors = np.asarray(self.tracking_errors_m, dtype=np.float64)
        planning = np.asarray(self.planning_wall_times_s, dtype=np.float64)
        sequential = np.asarray(
            self.sequential_update_wall_times_s, dtype=np.float64
        )
        drone_delta = (
            np.asarray(self.drone_positions_m, dtype=np.float64)
            - np.asarray(self.drone_positions_m[0], dtype=np.float64)
        )
        drone_displacement = np.linalg.norm(drone_delta, axis=1)
        horizontal_displacement = np.linalg.norm(drone_delta[:, :2], axis=1)
        tip_displacement = np.linalg.norm(
            np.asarray(self.actual_tip_positions_m, dtype=np.float64)
            - np.asarray(self.actual_tip_positions_m[0], dtype=np.float64),
            axis=1,
        )
        maximum_drone_displacement = float(np.max(drone_displacement))
        maximum_tip_displacement = float(np.max(tip_displacement))
        tip_velocity = np.asarray(self.actual_tip_velocities_m_s, dtype=np.float64)
        tangent = self.reference.unit_tangent_numpy(self.path_progress_cycles)
        tangent_speed = np.sum(tip_velocity * tangent, axis=1)
        normal_velocity = tip_velocity - tangent_speed[:, None] * tangent
        normal_speed = np.linalg.norm(normal_velocity, axis=1)
        sample_dt_s = (
            float(np.median(np.diff(self.time_s))) if len(self.time_s) > 1 else 1.0
        )
        tangent_acceleration = np.diff(tangent_speed) / max(sample_dt_s, 1.0e-9)
        command_changes = np.diff(
            np.asarray(self.commands_m_s2, dtype=np.float64), axis=0
        )
        return {
            "tip_position_rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
            "tip_position_mean_error_m": float(np.mean(errors)),
            "tip_position_maximum_error_m": float(np.max(errors)),
            "tip_position_p95_error_m": float(np.percentile(errors, 95.0)),
            "mppi_mean_update_time_s": (
                float(np.mean(planning)) if len(planning) else math.nan
            ),
            "mppi_p95_update_time_s": (
                float(np.percentile(planning, 95.0)) if len(planning) else math.nan
            ),
            "sequential_mean_update_time_s": (
                float(np.mean(sequential)) if len(sequential) else math.nan
            ),
            "sequential_p95_update_time_s": (
                float(np.percentile(sequential, 95.0))
                if len(sequential)
                else math.nan
            ),
            "drone_rms_displacement_from_start_m": float(
                np.sqrt(np.mean(np.square(drone_displacement)))
            ),
            "drone_maximum_displacement_from_start_m": maximum_drone_displacement,
            "drone_horizontal_rms_displacement_from_start_m": float(
                np.sqrt(np.mean(np.square(horizontal_displacement)))
            ),
            "drone_horizontal_maximum_displacement_from_start_m": float(
                np.max(horizontal_displacement)
            ),
            "drone_minimum_vertical_displacement_m": float(
                np.min(drone_delta[:, 2])
            ),
            "drone_maximum_vertical_displacement_m": float(
                np.max(drone_delta[:, 2])
            ),
            "tip_maximum_displacement_from_start_m": maximum_tip_displacement,
            "tip_to_drone_motion_ratio": (
                maximum_tip_displacement / max(maximum_drone_displacement, 1.0e-9)
            ),
            "tip_normal_velocity_rms_m_s": float(
                np.sqrt(np.mean(np.square(normal_speed)))
            ),
            "tip_normal_velocity_p95_m_s": float(
                np.percentile(normal_speed, 95.0)
            ),
            "tip_forward_tangent_speed_mean_m_s": float(
                np.mean(np.maximum(tangent_speed, 0.0))
            ),
            "tip_tangent_acceleration_rms_m_s2": (
                float(np.sqrt(np.mean(np.square(tangent_acceleration))))
                if len(tangent_acceleration)
                else 0.0
            ),
            "tip_tangent_deceleration_rms_m_s2": (
                float(
                    np.sqrt(
                        np.mean(np.square(np.minimum(tangent_acceleration, 0.0)))
                    )
                )
                if len(tangent_acceleration)
                else 0.0
            ),
            "command_change_rms_m_s2": (
                float(np.sqrt(np.mean(np.sum(np.square(command_changes), axis=1))))
                if len(command_changes)
                else 0.0
            ),
            "completed_path_cycles": float(
                self.path_progress_cycles[-1] - self.path_progress_cycles[0]
            ),
        }


LiveUpdateCallback = Callable[[Figure8LiveUpdate], None]


def endpoint_only_observation_state(
    predicted_state: DroneCableState,
    plant_state: DroneCableState,
) -> DroneCableState:
    """Expose plant root/tip state while replacing truth interior with prediction.

    The existing endpoint conditioner accepts a ``DroneCableState`` for API
    compatibility.  Sanitizing that object here ensures the plant's interior
    positions and velocities never cross the observation boundary, even if the
    conditioner is modified later.
    """

    if predicted_state.batch_size != 1 or plant_state.batch_size != 1:
        raise ValueError("Endpoint-only observation expects single states.")
    positions = predicted_state.cable.positions_m.clone()
    velocities = predicted_state.cable.velocities_m_s.clone()
    positions[:, -1] = plant_state.cable.positions_m[:, -1]
    velocities[:, -1] = plant_state.cable.velocities_m_s[:, -1]
    return DroneCableState(
        plant_state.drone_position_m,
        plant_state.drone_velocity_m_s,
        DderState(positions, velocities),
    )


def path_following_metrics(
    tip_positions_m: torch.Tensor,
    reference: Figure8Reference,
    start_progress_cycles: float,
    *,
    sample_count: int = 720,
    local_search_fraction: float = 0.08,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized projection around an arc-length-derived unwrapped branch.

    Cumulative predicted tip travel supplies a branch-continuous center for the
    local search. This avoids crossing jumps and the old quarter-cycle horizon
    limit without launching a separate sequence of GPU kernels for every time
    step in every MPPI rollout.
    """

    if tip_positions_m.ndim != 3 or tip_positions_m.shape[-1] != 3:
        raise ValueError("Tip positions must have shape [batch, time, 3].")
    if sample_count < 32 or not 0.0 < local_search_fraction < 0.25:
        raise ValueError("Path projection resolution/search fraction is invalid.")
    grid_progress = torch.arange(
        sample_count + 1,
        dtype=tip_positions_m.dtype,
        device=tip_positions_m.device,
    ) / sample_count
    grid_positions = reference.point_torch(grid_progress)
    segment_lengths = torch.linalg.vector_norm(
        grid_positions[1:] - grid_positions[:-1], dim=1
    )
    path_length = torch.sum(segment_lengths)
    cumulative_arc_fraction = torch.cat(
        (
            torch.zeros(
                (1,),
                dtype=tip_positions_m.dtype,
                device=tip_positions_m.device,
            ),
            torch.cumsum(segment_lengths, dim=0) / path_length,
        )
    )

    start_lap = math.floor(float(start_progress_cycles))
    start_local_progress = float(start_progress_cycles) - start_lap
    start_index = min(
        sample_count, max(0, int(round(start_local_progress * sample_count)))
    )
    start_arc_cycles = start_lap + cumulative_arc_fraction[start_index]
    step_distance = torch.linalg.vector_norm(
        tip_positions_m[:, 1:] - tip_positions_m[:, :-1], dim=2
    )
    traveled_distance = torch.cat(
        (
            torch.zeros(
                (tip_positions_m.shape[0], 1),
                dtype=tip_positions_m.dtype,
                device=tip_positions_m.device,
            ),
            torch.cumsum(step_distance, dim=1),
        ),
        dim=1,
    )
    estimated_arc_cycles = start_arc_cycles + traveled_distance / path_length
    estimated_laps = torch.floor(estimated_arc_cycles)
    estimated_local_arc = estimated_arc_cycles - estimated_laps
    center_local_index = torch.searchsorted(
        cumulative_arc_fraction,
        estimated_local_arc.contiguous(),
    ).clamp(max=sample_count)
    center_progress = estimated_laps + center_local_index.to(
        tip_positions_m.dtype
    ) / sample_count
    lower_progress = center_progress - local_search_fraction
    upper_progress = center_progress + local_search_fraction
    selected_progress = center_progress
    for _iteration in range(5):
        phase = 2.0 * math.pi * selected_progress
        selected_positions = reference.point_torch(selected_progress)
        path_derivative = torch.stack(
            (
                2.0 * math.pi * reference.amplitude_x_m * torch.cos(phase),
                4.0 * math.pi * reference.amplitude_y_m * torch.cos(2.0 * phase),
                torch.zeros_like(selected_progress),
            ),
            dim=2,
        )
        residual = selected_positions - tip_positions_m
        projection_step = torch.sum(residual * path_derivative, dim=2) / torch.clamp(
            torch.sum(path_derivative.square(), dim=2), min=1.0e-9
        )
        projection_step = torch.clamp(
            projection_step,
            min=-0.5 * local_search_fraction,
            max=0.5 * local_search_fraction,
        )
        selected_progress = torch.minimum(
            torch.maximum(selected_progress - projection_step, lower_progress),
            upper_progress,
        )
    selected_positions = reference.point_torch(selected_progress)
    error = torch.linalg.vector_norm(
        selected_positions - tip_positions_m, dim=2
    )
    return error, selected_progress, selected_positions


def tracking_rollout_objective(
    rollout: TensorRollout,
    experiment_initial_state: DroneCableState,
    reference: Figure8Reference,
    start_path_progress_cycles: float,
    simulator: WhipSimulator,
    weights: TrackingCostSettings,
    previous_acceleration_m_s2: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Evaluate geometric path following without a prescribed traversal speed."""

    tip = rollout.cable_positions_m[:, :, -1]
    error, path_progress, _reference_positions = path_following_metrics(
        tip,
        reference,
        start_path_progress_cycles,
    )
    tracking_cost = weights.tracking_weight * torch.mean(error[:, 1:].square(), dim=1)

    path_length = reference.approximate_length_m
    monotonic_progress = torch.cummax(
        torch.clamp(path_progress, min=float(start_path_progress_cycles)), dim=1
    ).values
    progress_delta_m = (
        monotonic_progress[:, -1] - float(start_path_progress_cycles)
    ) * path_length
    path_progress_reward = -weights.path_progress_reward_weight * progress_delta_m
    # Tangent-speed quantities remain evaluation diagnostics. Consecutive tip
    # velocities additionally define a causal physical acceleration penalty so
    # that the free endpoint moves smoothly without prescribing its speed.
    tip_velocity = rollout.cable_velocities_m_s[:, :, -1]
    root_velocity = rollout.cable_velocities_m_s[:, :, 0]
    tangent = reference.unit_tangent_torch(path_progress)
    world_tangent_speed = torch.sum(tip_velocity * tangent, dim=2)
    relative_tangent_speed = torch.sum(
        (tip_velocity - root_velocity) * tangent, dim=2
    )
    if tip_velocity.shape[1] > 1:
        tip_acceleration = (
            tip_velocity[:, 1:] - tip_velocity[:, :-1]
        ) / simulator.settings.simulation_dt_s
        tip_acceleration_squared = torch.mean(
            torch.sum(tip_acceleration.square(), dim=2), dim=1
        )
    else:
        tip_acceleration_squared = torch.zeros(
            rollout.batch_size,
            dtype=tip_velocity.dtype,
            device=tip_velocity.device,
        )
    acceleration_scale_squared = (
        simulator.settings.maximum_acceleration_m_s2 ** 2
    )
    tip_motion_smoothness_cost = (
        weights.tip_motion_smoothness_weight
        * tip_acceleration_squared
        / acceleration_scale_squared
    )

    initial_drone = experiment_initial_state.drone_position_m
    if initial_drone.shape[0] == 1:
        initial_drone = initial_drone.expand(rollout.batch_size, -1)
    displacement_vector = rollout.drone_positions_m - initial_drone[:, None]
    horizontal_displacement = torch.linalg.vector_norm(
        displacement_vector[:, :, :2], dim=2
    )
    vertical_displacement = displacement_vector[:, :, 2]
    displacement = torch.linalg.vector_norm(displacement_vector, dim=2)
    acceleration = rollout.accelerations_m_s2
    effort = torch.mean(torch.sum(acceleration.square(), dim=2), dim=1)
    if acceleration.shape[1] > 1:
        changes = acceleration[:, 1:] - acceleration[:, :-1]
        smoothness = torch.mean(torch.sum(changes.square(), dim=2), dim=1)
    else:
        smoothness = torch.zeros_like(effort)
    if previous_acceleration_m_s2 is not None:
        previous = previous_acceleration_m_s2.to(
            dtype=acceleration.dtype, device=acceleration.device
        ).reshape(1, 3)
        continuity = torch.sum((acceleration[:, 0] - previous).square(), dim=1)
        smoothness = smoothness + continuity
    control_cost = (
        weights.control_effort_weight * effort
        + weights.control_smoothness_weight * smoothness
    )

    drone_speed = torch.linalg.vector_norm(rollout.drone_velocities_m_s, dim=2)
    maximum_drone_speed = torch.amax(drone_speed, dim=1)
    speed_violation = torch.relu(
        maximum_drone_speed / simulator.settings.maximum_speed_m_s - 1.0
    ).square()
    minimum_scene_height = torch.minimum(
        torch.amin(rollout.drone_positions_m[:, :, 2], dim=1),
        torch.amin(rollout.cable_positions_m[:, :, :, 2], dim=(1, 2)),
    )
    ground_violation = torch.relu(
        (weights.ground_clearance_m - minimum_scene_height)
        / weights.ground_clearance_m
    ).square()
    safety_cost = (
        weights.speed_limit_weight * speed_violation
        + weights.ground_safety_weight * ground_violation
    )
    total = (
        tracking_cost
        + path_progress_reward
        + tip_motion_smoothness_cost
        + control_cost
        + safety_cost
    )
    diagnostics = {
        "tracking_rmse_m": torch.sqrt(torch.mean(error.square(), dim=1)),
        "tracking_mean_error_m": torch.mean(error, dim=1),
        "tracking_maximum_error_m": torch.amax(error, dim=1),
        "terminal_tracking_error_m": error[:, -1],
        "path_progress_cycles": monotonic_progress[:, -1],
        "path_progress_delta_cycles": (
            monotonic_progress[:, -1] - float(start_path_progress_cycles)
        ),
        "path_progress_m": progress_delta_m,
        "mean_forward_world_tangent_speed_m_s": torch.mean(
            torch.relu(world_tangent_speed[:, 1:]), dim=1
        ),
        "mean_forward_relative_tangent_speed_m_s": torch.mean(
            torch.relu(relative_tangent_speed[:, 1:]), dim=1
        ),
        "maximum_forward_relative_tangent_speed_m_s": torch.amax(
            torch.relu(relative_tangent_speed), dim=1
        ),
        "maximum_drone_speed_m_s": maximum_drone_speed,
        "maximum_drone_displacement_m": torch.amax(displacement, dim=1),
        "maximum_drone_horizontal_displacement_m": torch.amax(
            horizontal_displacement, dim=1
        ),
        "maximum_drone_vertical_displacement_m": torch.amax(
            torch.abs(vertical_displacement), dim=1
        ),
        "tracking_cost": tracking_cost,
        "path_progress_reward": path_progress_reward,
        "tip_acceleration_rms_m_s2": torch.sqrt(tip_acceleration_squared),
        "tip_motion_smoothness_cost": tip_motion_smoothness_cost,
        "control_cost": control_cost,
        "safety_cost": safety_cost,
        "mppi_objective": total,
    }
    return total, diagnostics


def periodic_swing_seed_bank(
    knot_count: int,
    horizon_s: float,
    maximum_acceleration_m_s2: float,
    *,
    direction_count: int = 16,
    amplitude_fractions: tuple[float, ...] = (1.0 / 3.0, 2.0 / 3.0, 1.0),
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Create generic zero-net out-and-back stroke families.

    A smooth one-second zero-net acceleration stroke returns root velocity and
    displacement to their initial values in the continuous limit. Longer MPC
    horizons pad that same physical seed with zero acceleration instead of
    stretching it into a different maneuver. The bank varies only planar
    direction and magnitude; DDER and the real objective select the family.
    """

    if knot_count < 2 or horizon_s <= 0.0 or maximum_acceleration_m_s2 <= 0.0:
        raise ValueError("Periodic seed dimensions, horizon, and limit must be positive.")
    if direction_count < 4:
        raise ValueError("Periodic seed bank needs at least four planar directions.")
    if not amplitude_fractions or any(
        not math.isfinite(value) or value <= 0.0 or value > 1.0
        for value in amplitude_fractions
    ):
        raise ValueError("Periodic seed amplitudes must lie in (0, 1].")
    times = torch.linspace(0.0, horizon_s, knot_count, dtype=dtype, device=device)
    stroke_duration_s = min(1.0, horizon_s)
    normalized_time = torch.clamp(times / stroke_duration_s, 0.0, 1.0)
    waveform = (
        torch.cos(2.0 * math.pi * normalized_time)
        - torch.cos(4.0 * math.pi * normalized_time)
    )
    waveform = waveform / torch.clamp(torch.amax(torch.abs(waveform)), min=1.0e-9)
    waveform = torch.where(times <= stroke_duration_s, waveform, 0.0)
    seeds = [torch.zeros((knot_count, 3), dtype=dtype, device=device)]
    labels = ["zero"]
    for fraction in amplitude_fractions:
        magnitude = maximum_acceleration_m_s2 * fraction
        for direction_index in range(direction_count):
            angle = 2.0 * math.pi * direction_index / direction_count
            direction = torch.tensor(
                (math.cos(angle), math.sin(angle), 0.0),
                dtype=dtype,
                device=device,
            )
            seeds.append(magnitude * waveform[:, None] * direction[None])
            labels.append(
                f"recoil-{fraction:.3g}-dir-{math.degrees(angle):.1f}deg"
            )
    return torch.stack(seeds), tuple(labels)


def _select_periodic_swing_seed(
    simulator: WhipSimulator,
    current_state: DroneCableState,
    experiment_initial_state: DroneCableState,
    reference: Figure8Reference,
    start_path_progress_cycles: float,
    mppi_settings: MppiSettings,
    cost_settings: TrackingCostSettings,
    previous_acceleration_m_s2: torch.Tensor | None,
) -> tuple[torch.Tensor, float, str]:
    seeds, labels = periodic_swing_seed_bank(
        mppi_settings.knot_count,
        simulator.settings.horizon_s,
        simulator.settings.maximum_acceleration_m_s2,
        dtype=simulator.dtype,
        device=simulator.device,
    )
    controls = interpolate_control_knots(
        seeds,
        simulator.settings.control_count,
        simulator.settings.maximum_acceleration_m_s2,
    )
    with torch.no_grad():
        rollout = simulator.rollout(current_state, controls, create_graph=False)
        costs, _diagnostics = tracking_rollout_objective(
            rollout,
            experiment_initial_state,
            reference,
            start_path_progress_cycles,
            simulator,
            cost_settings,
            previous_acceleration_m_s2,
        )
    index = int(torch.argmin(costs).detach().cpu())
    return (
        seeds[index].detach().clone(),
        float(costs[index].detach().cpu()),
        labels[index],
    )


def optimize_tracking_mppi(
    simulator: WhipSimulator,
    current_state: DroneCableState,
    experiment_initial_state: DroneCableState,
    reference: Figure8Reference,
    start_path_progress_cycles: float,
    mppi_settings: MppiSettings,
    cost_settings: TrackingCostSettings,
    *,
    warm_start_knots_m_s2: np.ndarray | torch.Tensor | None = None,
    previous_acceleration_m_s2: torch.Tensor | None = None,
    periodic_swing_bootstrap: bool = False,
    progress: ProgressCallback | None = None,
) -> TrackingPlan:
    """Run the existing stochastic MPPI update against the tracking objective."""

    if mppi_settings.gradient_guidance_fraction != 0.0:
        raise ValueError("DDER gradient guidance is not part of figure-eight tracking.")
    report = progress if progress is not None else (lambda _message: None)
    count = simulator.settings.control_count
    if mppi_settings.knot_count > count + 1:
        raise ValueError("MPPI knot count cannot exceed control count plus one.")
    dtype = simulator.dtype
    device = simulator.device
    maximum_acceleration = simulator.settings.maximum_acceleration_m_s2
    initialization = "shifted warm start"
    bootstrap_cost = math.inf
    if warm_start_knots_m_s2 is None and periodic_swing_bootstrap:
        nominal, bootstrap_cost, initialization = _select_periodic_swing_seed(
            simulator,
            current_state,
            experiment_initial_state,
            reference,
            start_path_progress_cycles,
            mppi_settings,
            cost_settings,
            previous_acceleration_m_s2,
        )
        report(
            f"periodic bootstrap: {initialization}, cost={bootstrap_cost:.4g}"
        )
    elif warm_start_knots_m_s2 is None:
        nominal = torch.zeros(
            (mppi_settings.knot_count, 3), dtype=dtype, device=device
        )
        initialization = "zero"
    else:
        nominal = torch.tensor(
            warm_start_knots_m_s2, dtype=dtype, device=device
        ).reshape(mppi_settings.knot_count, 3)
        nominal = _bound_vectors(nominal, maximum_acceleration)

    generator = torch.Generator(device=device)
    generator.manual_seed(mppi_settings.seed)
    best_cost = bootstrap_cost
    best_knots: torch.Tensor | None = (
        nominal.detach().clone() if math.isfinite(bootstrap_cost) else None
    )
    for iteration in range(mppi_settings.iterations):
        sigma = mppi_settings.acceleration_noise_sigma_m_s2 * (
            mppi_settings.noise_decay**iteration
        )
        perturbations = _sample_perturbations(
            mppi_settings.samples,
            mppi_settings.knot_count,
            sigma,
            generator=generator,
            dtype=dtype,
            device=device,
        )
        candidates = _bound_vectors(
            nominal[None] + perturbations, maximum_acceleration
        )
        applied_delta = candidates - nominal[None]
        cost_parts: list[torch.Tensor] = []
        rmse_parts: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(
                0, mppi_settings.samples, mppi_settings.rollout_batch_size
            ):
                stop = min(
                    start + mppi_settings.rollout_batch_size,
                    mppi_settings.samples,
                )
                controls = interpolate_control_knots(
                    candidates[start:stop], count, maximum_acceleration
                )
                rollout = simulator.rollout(
                    current_state, controls, create_graph=False
                )
                costs, diagnostics = tracking_rollout_objective(
                    rollout,
                    experiment_initial_state,
                    reference,
                    start_path_progress_cycles,
                    simulator,
                    cost_settings,
                    previous_acceleration_m_s2,
                )
                cost_parts.append(costs)
                rmse_parts.append(diagnostics["tracking_rmse_m"])
            costs = torch.cat(cost_parts)
            rmses = torch.cat(rmse_parts)
            if not bool(torch.all(torch.isfinite(costs)).detach().cpu()):
                raise RuntimeError("Tracking MPPI produced a non-finite objective.")
            minimum = torch.min(costs)
            importance = torch.exp(
                -(costs - minimum) / mppi_settings.temperature
            )
            weights = importance / torch.clamp(torch.sum(importance), min=1.0e-30)
            nominal = _bound_vectors(
                nominal
                + torch.sum(weights[:, None, None] * applied_delta, dim=0),
                maximum_acceleration,
            )
            candidate_index = int(torch.argmin(costs).detach().cpu())
            candidate_cost = float(costs[candidate_index].detach().cpu())
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_knots = candidates[candidate_index].detach().clone()
            report(
                f"tracking MPPI {iteration + 1}/{mppi_settings.iterations}: "
                f"best={best_cost:.4g}, sampled RMSE="
                f"{1000.0 * float(rmses[candidate_index].detach().cpu()):.1f}mm"
            )

    if best_knots is None:
        raise RuntimeError("Tracking MPPI produced no candidate plan.")
    with torch.no_grad():
        best_controls = interpolate_control_knots(
            best_knots[None], count, maximum_acceleration
        )
        best_rollout = simulator.rollout(
            current_state, best_controls, create_graph=False
        )
        objective, diagnostics = tracking_rollout_objective(
            best_rollout,
            experiment_initial_state,
            reference,
            start_path_progress_cycles,
            simulator,
            cost_settings,
            previous_acceleration_m_s2,
        )
    terms = {
        name: float(value[0].detach().cpu())
        for name, value in diagnostics.items()
    }
    controls_array = best_controls[0].detach().cpu().numpy().copy()
    knots_array = best_knots.detach().cpu().numpy().copy()
    predicted_time = best_rollout.time_s.detach().cpu().numpy().copy()
    predicted_drone = (
        best_rollout.drone_positions_m[0].detach().cpu().numpy().copy()
    )
    predicted_tip = (
        best_rollout.cable_positions_m[0, :, -1].detach().cpu().numpy().copy()
    )
    with torch.no_grad():
        _error, _progress, predicted_reference_tensor = path_following_metrics(
            best_rollout.cable_positions_m[:, :, -1],
            reference,
            start_path_progress_cycles,
        )
    predicted_reference = (
        predicted_reference_tensor[0].detach().cpu().numpy().copy()
    )
    for value in (
        controls_array,
        knots_array,
        predicted_time,
        predicted_drone,
        predicted_tip,
        predicted_reference,
    ):
        value.setflags(write=False)
    return TrackingPlan(
        controls_m_s2=controls_array,
        control_knots_m_s2=knots_array,
        cost=float(objective[0].detach().cpu()),
        terms=terms,
        predicted_time_s=predicted_time,
        predicted_drone_positions_m=predicted_drone,
        predicted_tip_positions_m=predicted_tip,
        predicted_reference_positions_m=predicted_reference,
        initialization=initialization,
    )


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy()
    result.setflags(write=False)
    return result


def run_figure8_tracking(
    planner: WhipSimulator,
    plant: WhipSimulator,
    initial_state: DroneCableState,
    reference: Figure8Reference,
    mppi_settings: MppiSettings,
    cost_settings: TrackingCostSettings,
    execution_settings: Figure8ExecutionSettings,
    *,
    live_update: LiveUpdateCallback | None = None,
    progress: ProgressCallback | None = None,
    cancelled: CancellationCallback | None = None,
    wait_if_paused: PauseCallback | None = None,
    observation_mode_provider: ObservationModeProvider | None = None,
    history_observer: DderHistoryObserver | None = None,
    controller_initial_state: DroneCableState | None = None,
    plant_state_hook: PlantStateHook | None = None,
) -> Figure8Execution:
    """Run matched-model receding-horizon tracking until duration or cancellation."""

    if planner.snapshot.node_count != 11 or plant.snapshot.node_count != 11:
        raise ValueError("The baseline figure-eight task requires exactly 11 DDER nodes.")
    if planner.snapshot.sha256 != plant.snapshot.sha256:
        raise ValueError("Figure-eight baseline requires matched controller/plant physics.")
    if initial_state.batch_size != 1:
        raise ValueError("Figure-eight execution expects one initial state.")
    ratio = execution_settings.replan_interval_s / plant.settings.control_interval_s
    if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("Replan interval must be a multiple of control interval.")
    controls_per_replan = max(1, int(round(ratio)))
    report = progress if progress is not None else (lambda _message: None)

    plant_state = initial_state
    observer_state = (
        initial_state if controller_initial_state is None else controller_initial_state
    )
    if observer_state.batch_size != 1:
        raise ValueError("Figure-eight controller prior expects one initial state.")
    experiment_initial_state = initial_state
    # ``None`` is semantically important: it identifies the first solve, when
    # the optional structured bootstrap is allowed to select a maneuver family.
    # Every later update receives the shifted previous solution.
    warm_knots: np.ndarray | None = None
    elapsed = 0.0
    path_progress_cycles = 0.0
    update_index = 0
    previous_acceleration: torch.Tensor | None = None
    planning_times: list[float] = []
    sequential_update_times: list[float] = []

    initial_tip = _numpy(initial_state.cable.positions_m[0, -1])
    initial_tip_velocity = _numpy(initial_state.cable.velocities_m_s[0, -1])
    initial_drone = _numpy(initial_state.drone_position_m[0])
    initial_drone_velocity = _numpy(initial_state.drone_velocity_m_s[0])
    initial_cable = _numpy(initial_state.cable.positions_m[0])
    initial_cable_velocity = _numpy(initial_state.cable.velocities_m_s[0])
    if execution_settings.observation_mode == "full":
        # The full-state baseline observes the plant at t=0 as well as at all
        # later replans.  Keeping a deliberately different controller prior is
        # useful for hidden-state ablations, but it must not contaminate the
        # evaluator's full-state estimate log.
        initial_estimated_cable = initial_cable.copy()
        initial_estimated_velocity = initial_cable_velocity.copy()
    elif execution_settings.observation_mode == "history":
        if history_observer is None:
            raise ValueError("History observation mode requires a DDER history observer.")
        initial_estimated_cable = _numpy(
            history_observer.current_state.positions_m[0]
        )
        initial_estimated_velocity = _numpy(
            history_observer.current_state.velocities_m_s[0]
        )
    else:
        initial_estimated_cable = _numpy(observer_state.cable.positions_m[0])
        initial_estimated_velocity = _numpy(observer_state.cable.velocities_m_s[0])
    time_values = [0.0]
    reference_positions = [reference.point_numpy(0.0)]
    path_progress_values = [path_progress_cycles]
    actual_tips = [initial_tip]
    actual_tip_velocities = [initial_tip_velocity]
    drone_positions = [initial_drone]
    drone_velocities = [initial_drone_velocity]
    commands = [np.zeros(3, dtype=np.float64)]
    cable_positions = [initial_cable]
    cable_velocities = [initial_cable_velocity]
    estimated_cable_positions = [initial_estimated_cable]
    estimated_cable_velocities = [initial_estimated_velocity]
    tracking_errors = [
        float(np.linalg.norm(initial_tip - reference_positions[0]))
    ]
    observation_modes: list[ObservationMode] = [
        execution_settings.observation_mode
    ]
    observer_update_times: list[float] = []
    observer_rmse_before: list[float] = []
    observer_rmse_after: list[float] = []
    observer_accepted: list[bool] = []

    while True:
        loop_wall_started = time.perf_counter()
        if cancelled is not None and cancelled():
            break
        if execution_settings.duration_s is not None and (
            elapsed + 1.0e-9 >= execution_settings.duration_s
        ):
            break
        if wait_if_paused is not None:
            wait_if_paused()
            if cancelled is not None and cancelled():
                break

        observation_mode = (
            execution_settings.observation_mode
            if observation_mode_provider is None
            else observation_mode_provider()
        )
        if observation_mode not in {"full", "endpoint", "history"}:
            raise ValueError(
                "Live observation mode provider must return full, endpoint, or history."
            )
        if plant_state_hook is not None:
            hooked_state = plant_state_hook(elapsed, plant_state)
            if hooked_state.batch_size != 1:
                raise ValueError("Figure-eight plant-state hook must return one state.")
            if hooked_state is not plant_state:
                plant_state = hooked_state
                cable_positions[-1] = _numpy(plant_state.cable.positions_m[0])
                cable_velocities[-1] = _numpy(plant_state.cable.velocities_m_s[0])
                actual_tips[-1] = _numpy(plant_state.cable.positions_m[0, -1])
                actual_tip_velocities[-1] = _numpy(
                    plant_state.cable.velocities_m_s[0, -1]
                )
                if observation_mode == "full":
                    estimated_cable_positions[-1] = cable_positions[-1].copy()
                    estimated_cable_velocities[-1] = cable_velocities[-1].copy()
        sequential_started = time.perf_counter()
        observer_update: HistoryObserverUpdate | None = None
        if observation_mode == "history":
            if history_observer is None:
                raise ValueError("History observation mode requires a DDER history observer.")
            observer_update = history_observer.correct()
            estimated_cable_positions[-1] = _numpy(
                history_observer.current_state.positions_m[0]
            )
            estimated_cable_velocities[-1] = _numpy(
                history_observer.current_state.velocities_m_s[0]
            )
        observer_update_times.append(
            0.0 if observer_update is None else observer_update.total_time_s
        )
        observer_rmse_before.append(
            math.nan
            if observer_update is None
            else observer_update.measurement_rmse_before_m
        )
        observer_rmse_after.append(
            math.nan
            if observer_update is None
            else observer_update.measurement_rmse_after_m
        )
        observer_accepted.append(
            False if observer_update is None else observer_update.accepted
        )
        if observation_mode == "full":
            controller_state = plant_state
        elif observation_mode == "endpoint":
            endpoint_observation = endpoint_only_observation_state(
                observer_state, plant_state
            )
            controller_state = endpoint_conditioned_state(
                observer_state,
                endpoint_observation,
                planner.settings.attachment_drop_m,
            )
        else:
            assert history_observer is not None
            controller_state = DroneCableState(
                plant_state.drone_position_m,
                plant_state.drone_velocity_m_s,
                history_observer.current_state,
            )
        settings_this_update = replace(
            mppi_settings, seed=mppi_settings.seed + update_index
        )
        if planner.device.type == "cuda":
            torch.cuda.synchronize(planner.device)
        started = time.perf_counter()
        plan = optimize_tracking_mppi(
            planner,
            controller_state,
            experiment_initial_state,
            reference,
            path_progress_cycles,
            settings_this_update,
            cost_settings,
            warm_start_knots_m_s2=warm_knots,
            previous_acceleration_m_s2=previous_acceleration,
            periodic_swing_bootstrap=(
                execution_settings.periodic_swing_bootstrap and warm_knots is None
            ),
        )
        if planner.device.type == "cuda":
            torch.cuda.synchronize(planner.device)
        planning_wall = time.perf_counter() - started
        planning_times.append(planning_wall)
        sequential_update_times.append(time.perf_counter() - sequential_started)

        apply_count = controls_per_replan
        if execution_settings.duration_s is not None:
            remaining = max(
                1,
                int(
                    math.ceil(
                        (execution_settings.duration_s - elapsed)
                        / plant.settings.control_interval_s
                        - 1.0e-9
                    )
                ),
            )
            apply_count = min(apply_count, remaining)
        controls_array = np.asarray(
            plan.controls_m_s2[:apply_count], dtype=np.float32
        ).copy()
        controls = torch.as_tensor(
            controls_array, dtype=plant.dtype, device=plant.device
        )[None]
        with torch.no_grad():
            plant_segment = plant.rollout(
                plant_state, controls, create_graph=False
            )
            observer_segment = planner.rollout(
                controller_state,
                controls.to(planner.device),
                create_graph=False,
            )
        plant_state = plant_segment.final_state()
        observer_state = observer_segment.final_state()

        segment_time = _numpy(plant_segment.time_s[1:]) + elapsed
        segment_tip = _numpy(plant_segment.cable_positions_m[0, 1:, -1])
        segment_tip_velocity = _numpy(
            plant_segment.cable_velocities_m_s[0, 1:, -1]
        )
        segment_drone = _numpy(plant_segment.drone_positions_m[0, 1:])
        segment_drone_velocity = _numpy(
            plant_segment.drone_velocities_m_s[0, 1:]
        )
        segment_cable = _numpy(plant_segment.cable_positions_m[0, 1:])
        segment_cable_velocity = _numpy(
            plant_segment.cable_velocities_m_s[0, 1:]
        )
        if history_observer is not None:
            segment_estimated_position: list[np.ndarray] = []
            segment_estimated_velocity: list[np.ndarray] = []
            for local_index, sample_time in enumerate(segment_time):
                estimate = history_observer.ingest(
                    EndpointHistoryObservation(
                        timestamp_s=float(sample_time),
                        root_position_m=segment_cable[local_index, 0],
                        root_velocity_m_s=segment_cable_velocity[local_index, 0],
                        tip_position_m=segment_tip[local_index],
                    )
                )
                segment_estimated_position.append(_numpy(estimate.positions_m[0]))
                segment_estimated_velocity.append(_numpy(estimate.velocities_m_s[0]))
            estimated_cable_positions.extend(segment_estimated_position)
            estimated_cable_velocities.extend(segment_estimated_velocity)
        else:
            estimate_segment = (
                plant_segment if observation_mode == "full" else observer_segment
            )
            estimated_cable_positions.extend(
                _numpy(estimate_segment.cable_positions_m[0, 1:])
            )
            estimated_cable_velocities.extend(
                _numpy(estimate_segment.cable_velocities_m_s[0, 1:])
            )
        (
            segment_reference,
            segment_progress,
            segment_error,
        ) = reference.project_sequence_numpy(
            segment_tip,
            path_progress_cycles,
        )
        path_progress_cycles = float(segment_progress[-1])
        step_commands = np.repeat(
            controls_array,
            plant.settings.steps_per_control,
            axis=0,
        )[: len(segment_time)]
        time_values.extend(segment_time.tolist())
        reference_positions.extend(segment_reference)
        path_progress_values.extend(segment_progress.tolist())
        actual_tips.extend(segment_tip)
        actual_tip_velocities.extend(segment_tip_velocity)
        drone_positions.extend(segment_drone)
        drone_velocities.extend(segment_drone_velocity)
        commands.extend(step_commands)
        cable_positions.extend(segment_cable)
        cable_velocities.extend(segment_cable_velocity)
        tracking_errors.extend(segment_error.tolist())
        observation_modes.extend([observation_mode] * len(segment_time))

        segment_duration = apply_count * plant.settings.control_interval_s
        elapsed += segment_duration
        warm_knots = shift_control_knots(
            plan.control_knots_m_s2,
            segment_duration,
            planner.settings.horizon_s,
            planner.settings.maximum_acceleration_m_s2,
        )
        previous_acceleration = torch.as_tensor(
            controls_array[-1], dtype=planner.dtype, device=planner.device
        )
        error_array = np.asarray(tracking_errors, dtype=np.float64)
        planning_array = np.asarray(planning_times, dtype=np.float64)
        recent_count = min(len(error_array), int(round(8.0 / plant.settings.simulation_dt_s)))
        if live_update is not None:
            drone_delta = (
                np.asarray(drone_positions, dtype=np.float64)
                - np.asarray(drone_positions[0], dtype=np.float64)
            )
            drone_displacement = np.linalg.norm(drone_delta, axis=1)
            horizontal_displacement = np.linalg.norm(drone_delta[:, :2], axis=1)
            live_update(
                Figure8LiveUpdate(
                    update_index=update_index,
                    time_s=elapsed,
                    planning_wall_time_s=planning_wall,
                    drone_position_m=_readonly(drone_positions[-1]),
                    drone_velocity_m_s=_readonly(drone_velocities[-1]),
                    cable_positions_m=_readonly(cable_positions[-1]),
                    cable_velocities_m_s=_readonly(cable_velocities[-1]),
                    reference_tip_position_m=_readonly(reference_positions[-1]),
                    path_progress_cycles=path_progress_cycles,
                    command_m_s2=_readonly(commands[-1]),
                    tracking_error_m=float(error_array[-1]),
                    rmse_m=float(np.sqrt(np.mean(np.square(error_array)))),
                    mean_error_m=float(np.mean(error_array)),
                    maximum_error_m=float(np.max(error_array)),
                    p95_error_m=float(np.percentile(error_array, 95.0)),
                    mean_planning_wall_time_s=float(np.mean(planning_array)),
                    p95_planning_wall_time_s=float(
                        np.percentile(planning_array, 95.0)
                    ),
                    drone_displacement_from_start_m=float(drone_displacement[-1]),
                    maximum_drone_displacement_from_start_m=float(
                        np.max(drone_displacement)
                    ),
                    drone_horizontal_displacement_from_start_m=float(
                        horizontal_displacement[-1]
                    ),
                    maximum_drone_horizontal_displacement_from_start_m=float(
                        np.max(horizontal_displacement)
                    ),
                    drone_vertical_displacement_m=float(drone_delta[-1, 2]),
                    recent_time_s=_readonly(np.asarray(time_values[-recent_count:])),
                    recent_actual_tip_positions_m=_readonly(
                        np.asarray(actual_tips[-recent_count:])
                    ),
                    recent_reference_tip_positions_m=_readonly(
                        np.asarray(reference_positions[-recent_count:])
                    ),
                    recent_tracking_errors_m=_readonly(error_array[-recent_count:]),
                    observation_mode=observation_mode,
                    observer_ready=(
                        False if observer_update is None else observer_update.ready
                    ),
                    observer_accepted=(
                        False if observer_update is None else observer_update.accepted
                    ),
                    observer_history_rmse_m=(
                        math.nan
                        if observer_update is None
                        else observer_update.measurement_rmse_after_m
                    ),
                    observer_wall_time_s=(
                        0.0 if observer_update is None else observer_update.total_time_s
                    ),
                )
            )
        report(
            f"t={elapsed:.2f}s mode={observation_mode} "
            f"progress={path_progress_cycles:.3f} cycles "
            f"error={1000.0 * error_array[-1]:.1f}mm "
            f"RMSE={1000.0 * math.sqrt(float(np.mean(np.square(error_array)))):.1f}mm "
            f"plan={planning_wall:.3f}s"
        )
        update_index += 1
        if execution_settings.realtime_pacing:
            remaining_wall = segment_duration - (
                time.perf_counter() - loop_wall_started
            )
            if remaining_wall > 0.0 and not (
                cancelled is not None and cancelled()
            ):
                time.sleep(remaining_wall)

    arrays = [
        np.asarray(time_values),
        np.asarray(reference_positions),
        np.asarray(path_progress_values),
        np.asarray(actual_tips),
        np.asarray(actual_tip_velocities),
        np.asarray(drone_positions),
        np.asarray(drone_velocities),
        np.asarray(commands),
        np.asarray(cable_positions),
        np.asarray(cable_velocities),
        np.asarray(estimated_cable_positions),
        np.asarray(estimated_cable_velocities),
        np.asarray(tracking_errors),
        np.asarray(planning_times),
        np.asarray(sequential_update_times),
        np.asarray(observer_update_times),
        np.asarray(observer_rmse_before),
        np.asarray(observer_rmse_after),
        np.asarray(observer_accepted, dtype=bool),
    ]
    readonly = [_readonly(value) for value in arrays]
    readonly_observation_modes = _readonly(
        np.asarray(observation_modes, dtype="<U8")
    )
    unique_observation_modes = set(observation_modes)
    execution_observation_mode: ExecutionObservationMode = (
        observation_modes[-1]
        if len(unique_observation_modes) == 1
        else "mixed"
    )
    return Figure8Execution(
        time_s=readonly[0],
        reference_tip_positions_m=readonly[1],
        path_progress_cycles=readonly[2],
        actual_tip_positions_m=readonly[3],
        actual_tip_velocities_m_s=readonly[4],
        drone_positions_m=readonly[5],
        drone_velocities_m_s=readonly[6],
        commands_m_s2=readonly[7],
        cable_positions_m=readonly[8],
        cable_velocities_m_s=readonly[9],
        estimated_cable_positions_m=readonly[10],
        estimated_cable_velocities_m_s=readonly[11],
        tracking_errors_m=readonly[12],
        planning_wall_times_s=readonly[13],
        sequential_update_wall_times_s=readonly[14],
        observer_update_times_s=readonly[15],
        observer_history_rmse_before_m=readonly[16],
        observer_history_rmse_after_m=readonly[17],
        observer_update_accepted=readonly[18],
        observation_modes=readonly_observation_modes,
        observation_mode=execution_observation_mode,
        reference=reference,
        controller_model_sha256=planner.snapshot.sha256,
        plant_model_sha256=plant.snapshot.sha256,
    )


def save_figure8_execution(
    directory: str | Path,
    execution: Figure8Execution,
    simulation_settings: object,
    mppi_settings: MppiSettings,
    cost_settings: TrackingCostSettings,
    execution_settings: Figure8ExecutionSettings,
) -> tuple[Path, Path, Path]:
    """Save compact CSV, full truth NPZ, and reproducibility JSON."""

    output = Path(directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"figure8_{execution.observation_mode}_{stamp}"
    csv_path = output / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "time_s",
                "reference_x_m", "reference_y_m", "reference_z_m",
                "actual_tip_x_m", "actual_tip_y_m", "actual_tip_z_m",
                "tracking_error_m",
                "path_progress_cycles",
                "actual_tip_vx_m_s", "actual_tip_vy_m_s", "actual_tip_vz_m_s",
                "drone_x_m", "drone_y_m", "drone_z_m",
                "drone_vx_m_s", "drone_vy_m_s", "drone_vz_m_s",
                "command_ax_m_s2", "command_ay_m_s2", "command_az_m_s2",
                "observation_mode",
            )
        )
        for index, sample_time in enumerate(execution.time_s):
            writer.writerow(
                (
                    float(sample_time),
                    *execution.reference_tip_positions_m[index],
                    *execution.actual_tip_positions_m[index],
                    float(execution.tracking_errors_m[index]),
                    float(execution.path_progress_cycles[index]),
                    *execution.actual_tip_velocities_m_s[index],
                    *execution.drone_positions_m[index],
                    *execution.drone_velocities_m_s[index],
                    *execution.commands_m_s2[index],
                    execution.observation_modes[index],
                )
            )
    npz_path = output / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        time_s=execution.time_s,
        reference_tip_positions_m=execution.reference_tip_positions_m,
        path_progress_cycles=execution.path_progress_cycles,
        actual_tip_positions_m=execution.actual_tip_positions_m,
        actual_tip_velocities_m_s=execution.actual_tip_velocities_m_s,
        drone_positions_m=execution.drone_positions_m,
        drone_velocities_m_s=execution.drone_velocities_m_s,
        commands_m_s2=execution.commands_m_s2,
        cable_positions_m=execution.cable_positions_m,
        cable_velocities_m_s=execution.cable_velocities_m_s,
        estimated_cable_positions_m=execution.estimated_cable_positions_m,
        estimated_cable_velocities_m_s=execution.estimated_cable_velocities_m_s,
        tracking_errors_m=execution.tracking_errors_m,
        planning_wall_times_s=execution.planning_wall_times_s,
        sequential_update_wall_times_s=execution.sequential_update_wall_times_s,
        observer_update_times_s=execution.observer_update_times_s,
        observer_history_rmse_before_m=execution.observer_history_rmse_before_m,
        observer_history_rmse_after_m=execution.observer_history_rmse_after_m,
        observer_update_accepted=execution.observer_update_accepted,
        observation_modes=execution.observation_modes,
        observation_mode=np.asarray(execution.observation_mode),
    )
    json_path = output / f"{stem}.json"
    json_path.write_text(
        json.dumps(
            {
                "schema": "geometric_figure8_path_following_dder_mppi_v2",
                "observation_mode": execution.observation_mode,
                "matched_physics": (
                    execution.controller_model_sha256
                    == execution.plant_model_sha256
                ),
                "controller_model_sha256": execution.controller_model_sha256,
                "plant_model_sha256": execution.plant_model_sha256,
                "reference": asdict(execution.reference),
                "simulation_settings": asdict(simulation_settings),
                "mppi_settings": asdict(mppi_settings),
                "tracking_cost_settings": asdict(cost_settings),
                "execution_settings": asdict(execution_settings),
                "summary": execution.summary,
                "reference_has_prescribed_timing": False,
                "flat_path": True,
                "truth_state_saved_for_evaluation_only": True,
                "estimated_distributed_state_saved": True,
                "history_observer_active": bool(
                    np.any(execution.observation_modes == "history")
                ),
                "parameter_adaptation": False,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return csv_path, npz_path, json_path
