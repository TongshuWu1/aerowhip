"""CUDA-batched finite-horizon and receding-horizon drone-whip control."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable

import casadi as ca
import numpy as np
import torch

from .simulator import (
    DroneCableState,
    SimulationResult,
    TensorRollout,
    WhipSimulator,
    tensor_rollout_to_result,
)


ProgressCallback = Callable[[str], None]
CancellationCallback = Callable[[], bool]


class MpcCancelled(RuntimeError):
    """Raised after a cooperative stop request."""


@dataclass(frozen=True, slots=True)
class MpcProblem:
    target_position_m: tuple[float, float, float]
    impact_direction: tuple[float, float, float]
    minimum_impact_speed_m_s: float = 1.5
    drone_keepout_radius_m: float = 0.30
    maximum_drone_excursion_m: float = 0.15
    minimum_forward_stroke_m: float = 0.0
    minimum_recoil_stroke_m: float = 0.0
    drone_workspace_center_m: tuple[float, float, float] | None = None
    maximum_tip_error_m: float = 0.05
    planning_tip_error_margin_m: float = 0.0
    maximum_impact_angle_deg: float = 20.0
    planning_impact_angle_margin_deg: float = 0.0

    def __post_init__(self) -> None:
        target = np.asarray(self.target_position_m, dtype=np.float64)
        direction = np.asarray(self.impact_direction, dtype=np.float64)
        workspace_center = (
            None
            if self.drone_workspace_center_m is None
            else np.asarray(self.drone_workspace_center_m, dtype=np.float64)
        )
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("Target position must be a finite XYZ vector.")
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("Impact direction must be a finite XYZ vector.")
        if workspace_center is not None and (
            workspace_center.shape != (3,) or not np.all(np.isfinite(workspace_center))
        ):
            raise ValueError("Drone workspace center must be a finite XYZ vector.")
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-9:
            raise ValueError("Impact direction cannot be zero.")
        if (
            not math.isfinite(self.minimum_impact_speed_m_s)
            or self.minimum_impact_speed_m_s <= 0.0
            or not math.isfinite(self.drone_keepout_radius_m)
            or self.drone_keepout_radius_m <= 0.0
            or not math.isfinite(self.maximum_drone_excursion_m)
            or self.maximum_drone_excursion_m <= 0.0
            or not math.isfinite(self.minimum_forward_stroke_m)
            or not 0.0 <= self.minimum_forward_stroke_m
            <= self.maximum_drone_excursion_m
            or not math.isfinite(self.minimum_recoil_stroke_m)
            or not 0.0 <= self.minimum_recoil_stroke_m
            <= 2.0 * self.maximum_drone_excursion_m
            or not math.isfinite(self.maximum_tip_error_m)
            or self.maximum_tip_error_m <= 0.0
            or not math.isfinite(self.planning_tip_error_margin_m)
            or self.planning_tip_error_margin_m < 0.0
            or self.planning_tip_error_margin_m >= self.maximum_tip_error_m
            or not math.isfinite(self.maximum_impact_angle_deg)
            or not 0.0 < self.maximum_impact_angle_deg < 90.0
            or not math.isfinite(self.planning_impact_angle_margin_deg)
            or self.planning_impact_angle_margin_deg < 0.0
            or self.planning_impact_angle_margin_deg
            >= self.maximum_impact_angle_deg
        ):
            raise ValueError(
                "Impact speed, hit tolerance, keepout radius, drone excursion, and "
                "forward/recoil stroke thresholds must "
                "be non-negative; the planning margin must be non-negative and smaller "
                "than the hit tolerance; the impact angle must be between 0 and 90 "
                "degrees and its planning margin must leave a positive cone."
            )
        object.__setattr__(self, "target_position_m", tuple(float(v) for v in target))
        object.__setattr__(
            self,
            "impact_direction",
            tuple(float(v) for v in direction / norm),
        )
        if workspace_center is not None:
            object.__setattr__(
                self,
                "drone_workspace_center_m",
                tuple(float(v) for v in workspace_center),
            )

    @property
    def planning_tip_error_limit_m(self) -> float:
        """Tightened internal constraint; physical success uses the full tolerance."""

        return self.maximum_tip_error_m - self.planning_tip_error_margin_m

    @property
    def planning_impact_angle_limit_deg(self) -> float:
        """Tightened internal cone; physical success uses the full angle."""

        return (
            self.maximum_impact_angle_deg
            - self.planning_impact_angle_margin_deg
        )


@dataclass(frozen=True, slots=True)
class CostWeights:
    """Secondary regularization only; task and safety requirements are constraints."""

    acceleration_effort: float = 1.0
    acceleration_smoothness: float = 1.0

    def __post_init__(self) -> None:
        values = (self.acceleration_effort, self.acceleration_smoothness)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("MPC regularization weights must be finite and non-negative.")
        if sum(values) <= 0.0:
            raise ValueError("At least one MPC regularization weight must be positive.")


@dataclass(frozen=True, slots=True)
class OptimizerSettings:
    """Numerical settings for the bounded IPOPT casting NLP."""

    iterations: int = 20
    tolerance: float = 1.0e-4
    acceptable_tolerance: float = 1.0e-3
    maximum_wall_time_s: float = 1.0
    finite_difference_step: float = 1.0e-3
    replan_interval_s: float = 0.20
    initial_samples: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.iterations <= 500:
            raise ValueError("IPOPT iterations must be between 1 and 500.")
        if not 0 <= self.initial_samples <= 4096:
            raise ValueError("IPOPT initial samples must be between 0 and 4096.")
        if (
            not math.isfinite(self.tolerance)
            or self.tolerance <= 0.0
            or not math.isfinite(self.acceptable_tolerance)
            or self.acceptable_tolerance < self.tolerance
            or not math.isfinite(self.maximum_wall_time_s)
            or self.maximum_wall_time_s <= 0.0
            or not math.isfinite(self.finite_difference_step)
            or not 1.0e-6 <= self.finite_difference_step <= 0.1
            or not math.isfinite(self.replan_interval_s)
            or self.replan_interval_s <= 0.0
        ):
            raise ValueError("IPOPT tolerances, timing, or derivative step are invalid.")


@dataclass(frozen=True, slots=True)
class MpcPlan:
    controls_m_s2: np.ndarray
    prediction: SimulationResult
    cost: float
    cost_terms: dict[str, float]
    history: tuple[float, ...]
    impact_time_s: float
    feasible: bool
    constraint_violation: float
    casting_action: "CastingAction"


@dataclass(frozen=True, slots=True)
class CastingAction:
    """Low-dimensional target-aligned forward stroke and backward recoil."""

    forward_elevation_deg: float
    recoil_deflection_deg: float
    forward_excursion_m: float
    recoil_excursion_m: float
    reversal_fraction: float
    motion_fraction: float

    def __post_init__(self) -> None:
        values = (
            self.forward_elevation_deg,
            self.recoil_deflection_deg,
            self.forward_excursion_m,
            self.recoil_excursion_m,
            self.reversal_fraction,
            self.motion_fraction,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Casting-action parameters must be finite.")
        if self.forward_excursion_m < 0.0 or self.recoil_excursion_m < 0.0:
            raise ValueError("Forward and recoil excursions must be non-negative.")
        if not 0.0 < self.reversal_fraction < 1.0:
            raise ValueError("Casting reversal fraction must lie between zero and one.")
        if not 0.0 < self.motion_fraction <= 1.0:
            raise ValueError("Casting motion fraction must lie in (0, 1].")


@dataclass(frozen=True, slots=True)
class CastingPhaseSchedule:
    """Absolute timing for one injection/release maneuver.

    ``elapsed_s`` is measured from maneuver start at the state for which the
    new plan will activate.  Keeping the two phase boundaries absolute stops a
    receding-horizon update from restarting the injection phase.
    """

    elapsed_s: float
    injection_end_s: float
    motion_end_s: float
    lock_to_target_plane: bool = True

    def __post_init__(self) -> None:
        values = (self.elapsed_s, self.injection_end_s, self.motion_end_s)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Casting phase times must be finite.")
        if self.elapsed_s < 0.0:
            raise ValueError("Casting phase elapsed time cannot be negative.")
        if not 0.0 < self.injection_end_s < self.motion_end_s:
            raise ValueError(
                "Casting injection must end strictly inside the maneuver duration."
            )


@dataclass(frozen=True, slots=True)
class MpcExecution:
    result: SimulationResult
    executed_controls_m_s2: np.ndarray
    replanning_costs: tuple[float, ...]


def _bounded_acceleration(controls: torch.Tensor, maximum_m_s2: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(controls, dim=2, keepdim=True)
    scale = torch.clamp(maximum_m_s2 / torch.clamp(norm, min=1.0e-12), max=1.0)
    return controls * scale


_CASTING_PARAMETER_COUNT = 6
_MINIMUM_STROKE_FRACTION = 0.0
_MAXIMUM_FORWARD_ELEVATION_RAD = math.radians(45.0)
_MAXIMUM_RECOIL_DEFLECTION_RAD = math.radians(30.0)
_MINIMUM_REVERSAL_FRACTION = 0.25
_MAXIMUM_REVERSAL_FRACTION = 0.60
_MINIMUM_MOTION_FRACTION = 0.15


def _decode_casting_actions(
    normalized: torch.Tensor,
    maximum_excursion_m: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Map unit-cube optimizer variables to physical casting parameters."""

    if normalized.ndim != 2 or normalized.shape[1] != _CASTING_PARAMETER_COUNT:
        raise ValueError("Normalized casting actions must have shape Bx6.")
    unit = torch.clamp(normalized, 0.0, 1.0)
    forward_elevation = (
        (2.0 * unit[:, 0] - 1.0) * _MAXIMUM_FORWARD_ELEVATION_RAD
    )
    recoil_deflection = (
        (2.0 * unit[:, 1] - 1.0) * _MAXIMUM_RECOIL_DEFLECTION_RAD
    )
    minimum_stroke = _MINIMUM_STROKE_FRACTION * maximum_excursion_m
    stroke_span = maximum_excursion_m - minimum_stroke
    forward = minimum_stroke + stroke_span * unit[:, 2]
    recoil = minimum_stroke + stroke_span * unit[:, 3]
    reversal = _MINIMUM_REVERSAL_FRACTION + (
        _MAXIMUM_REVERSAL_FRACTION - _MINIMUM_REVERSAL_FRACTION
    ) * unit[:, 4]
    motion = _MINIMUM_MOTION_FRACTION + (
        1.0 - _MINIMUM_MOTION_FRACTION
    ) * unit[:, 5]
    return (
        forward_elevation,
        recoil_deflection,
        forward,
        recoil,
        reversal,
        motion,
    )


def _casting_action_from_normalized(
    normalized: torch.Tensor,
    maximum_excursion_m: float,
) -> CastingAction:
    values = normalized.detach().reshape(1, -1)
    elevation, deflection, forward, recoil, reversal, motion = _decode_casting_actions(
        values, maximum_excursion_m
    )
    return CastingAction(
        forward_elevation_deg=math.degrees(float(elevation[0].cpu())),
        recoil_deflection_deg=math.degrees(float(deflection[0].cpu())),
        forward_excursion_m=float(forward[0].cpu()),
        recoil_excursion_m=float(recoil[0].cpu()),
        reversal_fraction=float(reversal[0].cpu()),
        motion_fraction=float(motion[0].cpu()),
    )


def _normalized_casting_action(
    action: CastingAction,
    maximum_excursion_m: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    minimum_stroke = _MINIMUM_STROKE_FRACTION * maximum_excursion_m
    stroke_span = maximum_excursion_m - minimum_stroke
    values = torch.tensor(
        (
            (
                math.radians(action.forward_elevation_deg)
                / _MAXIMUM_FORWARD_ELEVATION_RAD
                + 1.0
            )
            / 2.0,
            (
                math.radians(action.recoil_deflection_deg)
                / _MAXIMUM_RECOIL_DEFLECTION_RAD
                + 1.0
            )
            / 2.0,
            (action.forward_excursion_m - minimum_stroke) / stroke_span,
            (action.recoil_excursion_m - minimum_stroke) / stroke_span,
            (action.reversal_fraction - _MINIMUM_REVERSAL_FRACTION)
            / (_MAXIMUM_REVERSAL_FRACTION - _MINIMUM_REVERSAL_FRACTION),
            (action.motion_fraction - _MINIMUM_MOTION_FRACTION)
            / (1.0 - _MINIMUM_MOTION_FRACTION),
        ),
        dtype=dtype,
        device=device,
    )
    return torch.clamp(values, 0.0, 1.0)


def _quintic_acceleration(
    time_s: torch.Tensor,
    duration_s: torch.Tensor,
    start_position_m: torch.Tensor,
    start_velocity_m_s: torch.Tensor,
    end_position_m: torch.Tensor,
) -> torch.Tensor:
    """Acceleration of a quintic segment with zero terminal velocity/acceleration."""

    duration = duration_s[:, None]
    displacement = end_position_m - start_position_m
    c3 = (10.0 * displacement - 6.0 * start_velocity_m_s * duration) / duration**3
    c4 = (-15.0 * displacement + 8.0 * start_velocity_m_s * duration) / duration**4
    c5 = (6.0 * displacement - 3.0 * start_velocity_m_s * duration) / duration**5
    time = time_s[:, :, None]
    return (
        6.0 * c3[:, None] * time
        + 12.0 * c4[:, None] * time**2
        + 20.0 * c5[:, None] * time**3
    )


def casting_controls(
    normalized_actions: torch.Tensor,
    initial_state: DroneCableState,
    problem: MpcProblem,
    simulator: WhipSimulator,
    control_count: int,
    phase_schedule: CastingPhaseSchedule | None = None,
) -> torch.Tensor:
    """Generate a smooth target-aligned forward stroke and backward recoil.

    The desired drone path starts at the measured state, moves toward the target
    to a forward waypoint, then recoils through the workspace centre to a backward waypoint,
    and stops. Both waypoints lie inside the hard excursion sphere. The sampled
    feed-forward accelerations are bounded by the vehicle model; the normal MPC
    safety test still evaluates the resulting state trajectory exactly.
    """

    if control_count < 1:
        raise ValueError("A casting action requires at least one control interval.")
    batch = normalized_actions.shape[0]
    if initial_state.batch_size not in (1, batch):
        raise ValueError("Casting actions and initial-state batches are incompatible.")
    elevation, deflection, forward_stroke, recoil_stroke, reversal, motion = (
        _decode_casting_actions(
            normalized_actions, problem.maximum_drone_excursion_m
        )
    )
    if phase_schedule is not None and phase_schedule.lock_to_target_plane:
        deflection = torch.zeros_like(deflection)
    reference = initial_state.drone_position_m
    start_position = reference.expand(batch, -1)
    start_velocity = initial_state.drone_velocity_m_s.expand(batch, -1)
    if problem.drone_workspace_center_m is None:
        workspace_center = start_position
    else:
        workspace_center = torch.tensor(
            problem.drone_workspace_center_m,
            dtype=reference.dtype,
            device=reference.device,
        )[None].expand(batch, -1)
    target = torch.tensor(
        problem.target_position_m,
        dtype=reference.dtype,
        device=reference.device,
    )
    horizontal = target - workspace_center[0]
    horizontal = horizontal.clone()
    horizontal[2] = 0.0
    horizontal_norm = torch.linalg.vector_norm(horizontal)
    if float(horizontal_norm.detach().cpu()) <= 1.0e-9:
        horizontal = torch.tensor(
            problem.impact_direction,
            dtype=reference.dtype,
            device=reference.device,
        )
        horizontal = horizontal.clone()
        horizontal[2] = 0.0
        horizontal_norm = torch.linalg.vector_norm(horizontal)
    if float(horizontal_norm.detach().cpu()) <= 1.0e-9:
        horizontal = torch.tensor(
            (1.0, 0.0, 0.0), dtype=reference.dtype, device=reference.device
        )
        horizontal_norm = torch.tensor(1.0, dtype=reference.dtype, device=reference.device)
    forward = horizontal / horizontal_norm
    vertical = torch.tensor(
        (0.0, 0.0, 1.0), dtype=reference.dtype, device=reference.device
    )
    forward_direction = (
        torch.cos(elevation)[:, None] * forward[None]
        + torch.sin(elevation)[:, None] * vertical[None]
    )
    transverse_direction = (
        -torch.sin(elevation)[:, None] * forward[None]
        + torch.cos(elevation)[:, None] * vertical[None]
    )
    recoil_direction = (
        torch.cos(deflection)[:, None] * forward_direction
        + torch.sin(deflection)[:, None] * transverse_direction
    )
    forward_waypoint = (
        workspace_center + forward_stroke[:, None] * forward_direction
    )
    recoil_waypoint = (
        workspace_center - recoil_stroke[:, None] * recoil_direction
    )

    interval = simulator.settings.control_interval_s
    horizon = control_count * interval
    midpoints = (
        torch.arange(
            control_count,
            dtype=reference.dtype,
            device=reference.device,
        )
        + 0.5
    ) * interval
    absolute_time = midpoints[None].expand(batch, -1)

    if phase_schedule is not None:
        remaining_motion = max(
            0.0,
            phase_schedule.motion_end_s - phase_schedule.elapsed_s,
        )
        if remaining_motion <= 1.0e-12:
            return torch.zeros(
                (batch, control_count, 3),
                dtype=reference.dtype,
                device=reference.device,
            )
        remaining_injection = max(
            0.0,
            phase_schedule.injection_end_s - phase_schedule.elapsed_s,
        )
        if remaining_injection > 1.0e-12:
            first_duration = torch.full(
                (batch,),
                remaining_injection,
                dtype=reference.dtype,
                device=reference.device,
            )
            second_duration = torch.full(
                (batch,),
                phase_schedule.motion_end_s - phase_schedule.injection_end_s,
                dtype=reference.dtype,
                device=reference.device,
            )
            first_time = torch.minimum(absolute_time, first_duration[:, None])
            first_acceleration = _quintic_acceleration(
                first_time,
                first_duration,
                start_position,
                start_velocity,
                forward_waypoint,
            )
            second_time = torch.clamp(
                absolute_time - first_duration[:, None], min=0.0
            )
            second_acceleration = _quintic_acceleration(
                torch.minimum(second_time, second_duration[:, None]),
                second_duration,
                forward_waypoint,
                torch.zeros_like(start_velocity),
                recoil_waypoint,
            )
            in_first = absolute_time <= first_duration[:, None]
            in_motion = absolute_time <= remaining_motion
            controls = torch.where(
                in_first[:, :, None], first_acceleration, second_acceleration
            )
            controls = torch.where(
                in_motion[:, :, None], controls, torch.zeros_like(controls)
            )
        else:
            release_duration = torch.full(
                (batch,),
                remaining_motion,
                dtype=reference.dtype,
                device=reference.device,
            )
            release_time = torch.minimum(
                absolute_time, release_duration[:, None]
            )
            controls = _quintic_acceleration(
                release_time,
                release_duration,
                start_position,
                start_velocity,
                recoil_waypoint,
            )
            in_motion = absolute_time <= remaining_motion
            controls = torch.where(
                in_motion[:, :, None], controls, torch.zeros_like(controls)
            )
        return _bounded_acceleration(
            controls, simulator.settings.maximum_acceleration_m_s2
        )

    total_duration = torch.clamp(
        motion * horizon,
        min=min(horizon, interval),
    )
    first_duration = torch.clamp(reversal * total_duration, min=0.25 * interval)
    second_duration = torch.clamp(
        total_duration - first_duration, min=0.25 * interval
    )
    first_time = torch.minimum(absolute_time, first_duration[:, None])
    first_acceleration = _quintic_acceleration(
        first_time,
        first_duration,
        start_position,
        start_velocity,
        forward_waypoint,
    )
    second_time = torch.clamp(
        absolute_time - first_duration[:, None], min=0.0
    )
    second_acceleration = _quintic_acceleration(
        torch.minimum(second_time, second_duration[:, None]),
        second_duration,
        forward_waypoint,
        torch.zeros_like(start_velocity),
        recoil_waypoint,
    )
    in_first = absolute_time <= first_duration[:, None]
    in_motion = absolute_time <= total_duration[:, None]
    controls = torch.where(
        in_first[:, :, None], first_acceleration, second_acceleration
    )
    controls = torch.where(in_motion[:, :, None], controls, torch.zeros_like(controls))
    return _bounded_acceleration(
        controls, simulator.settings.maximum_acceleration_m_s2
    )


@dataclass(frozen=True, slots=True)
class _ImpactEventTable:
    impact_frames: torch.Tensor
    terms: dict[str, torch.Tensor]


def _row_lexicographic_order(*keys: torch.Tensor) -> torch.Tensor:
    """Return per-row column indices sorted by the supplied priority keys."""

    if not keys or any(key.ndim != 2 for key in keys):
        raise ValueError("Row-wise lexicographic keys must be non-empty matrices.")
    shape = keys[0].shape
    if any(key.shape != shape for key in keys):
        raise ValueError("Row-wise lexicographic keys must have identical shapes.")
    order = torch.arange(shape[1], device=keys[0].device)[None].expand(shape[0], -1)
    for key in reversed(keys):
        values = torch.gather(key, 1, order)
        permutation = torch.argsort(values, dim=1, stable=True)
        order = torch.gather(order, 1, permutation)
    return order


def _vector_lexicographic_order(*keys: torch.Tensor) -> torch.Tensor:
    """Return candidate indices sorted by the supplied priority keys."""

    if not keys or any(key.ndim != 1 for key in keys):
        raise ValueError("Lexicographic candidate keys must be non-empty vectors.")
    length = keys[0].shape[0]
    if any(key.shape != (length,) for key in keys):
        raise ValueError("Lexicographic candidate keys must have identical lengths.")
    order = torch.arange(length, device=keys[0].device)
    for key in reversed(keys):
        order = order[torch.argsort(key[order], stable=True)]
    return order


def _impact_event_table(
    rollout: TensorRollout,
    initial_state: DroneCableState,
    problem: MpcProblem,
    simulator: WhipSimulator,
    weights: CostWeights,
    previous_acceleration_m_s2: torch.Tensor | None = None,
) -> _ImpactEventTable:
    """Evaluate task, safety, and regularization metrics at every physics frame."""

    reference = rollout.drone_positions_m
    batch = reference.shape[0]
    impact_frames = torch.arange(
        1,
        rollout.frame_count,
        dtype=torch.long,
        device=reference.device,
    )
    if impact_frames.numel() == 0:
        raise ValueError("Whip MPC requires at least one simulated physics step.")
    target = torch.tensor(
        problem.target_position_m,
        dtype=reference.dtype,
        device=reference.device,
    )
    direction = torch.tensor(
        problem.impact_direction,
        dtype=reference.dtype,
        device=reference.device,
    )

    tip_position = rollout.cable_positions_m[:, impact_frames, -1]
    # Impact is defined against the target.  For the stationary target used by
    # this task, collision velocity is the free-tip world velocity, not motion
    # relative to the drone.
    tip_velocity = rollout.cable_velocities_m_s[:, impact_frames, -1]
    position_error = torch.linalg.vector_norm(
        tip_position - target[None, None], dim=2
    )
    tip_speed = torch.linalg.vector_norm(tip_velocity, dim=2)
    directional_speed = torch.sum(tip_velocity * direction[None, None], dim=2)
    lateral_velocity = torch.linalg.vector_norm(
        tip_velocity - directional_speed[:, :, None] * direction[None, None],
        dim=2,
    )
    direction_cosine = directional_speed / torch.clamp(tip_speed, min=1.0e-9)
    direction_error_deg = torch.rad2deg(
        torch.acos(torch.clamp(direction_cosine, -1.0, 1.0))
    )

    drone_target_distance = torch.linalg.vector_norm(
        rollout.drone_positions_m - target[None, None], dim=2
    )
    if problem.drone_workspace_center_m is None:
        workspace_center = initial_state.drone_position_m
    else:
        workspace_center = torch.tensor(
            problem.drone_workspace_center_m,
            dtype=reference.dtype,
            device=reference.device,
        )[None].expand(batch, -1)
    drone_displacement = torch.linalg.vector_norm(
        rollout.drone_positions_m - workspace_center[:, None], dim=2
    )
    drone_speed = torch.linalg.vector_norm(rollout.drone_velocities_m_s, dim=2)
    maximum_drone_excursion = torch.cummax(drone_displacement, dim=1).values[
        :, impact_frames
    ]
    minimum_drone_clearance = torch.cummin(drone_target_distance, dim=1).values[
        :, impact_frames
    ]
    maximum_drone_speed = torch.cummax(drone_speed, dim=1).values[:, impact_frames]

    horizontal_forward = target - workspace_center[0]
    horizontal_forward = horizontal_forward.clone()
    horizontal_forward[2] = 0.0
    horizontal_norm = torch.linalg.vector_norm(horizontal_forward)
    if float(horizontal_norm.detach().cpu()) <= 1.0e-9:
        horizontal_forward = direction.clone()
        horizontal_forward[2] = 0.0
        horizontal_norm = torch.linalg.vector_norm(horizontal_forward)
    if float(horizontal_norm.detach().cpu()) <= 1.0e-9:
        horizontal_forward = torch.tensor(
            (1.0, 0.0, 0.0), dtype=reference.dtype, device=reference.device
        )
        horizontal_norm = torch.tensor(
            1.0, dtype=reference.dtype, device=reference.device
        )
    horizontal_forward = horizontal_forward / horizontal_norm
    forward_progress = torch.sum(
        (rollout.drone_positions_m - workspace_center[:, None])
        * horizontal_forward[None, None],
        dim=2,
    )
    maximum_forward_progress = torch.cummax(forward_progress, dim=1).values
    recoil_distance = maximum_forward_progress - forward_progress
    event_forward_progress = maximum_forward_progress[:, impact_frames]
    event_recoil_distance = recoil_distance[:, impact_frames]

    planning_tip_error_limit = problem.planning_tip_error_limit_m
    position_violation = (
        torch.relu(position_error - planning_tip_error_limit)
        / planning_tip_error_limit
    ) ** 2
    speed_violation = (
        torch.relu(problem.minimum_impact_speed_m_s - directional_speed)
        / problem.minimum_impact_speed_m_s
    ) ** 2
    planning_impact_angle_limit = problem.planning_impact_angle_limit_deg
    tangent = math.tan(math.radians(planning_impact_angle_limit))
    lateral_limit = tangent * directional_speed
    direction_scale = max(problem.minimum_impact_speed_m_s * tangent, 1.0e-9)
    direction_violation = (
        torch.relu(lateral_velocity - lateral_limit) / direction_scale
    ) ** 2
    if problem.minimum_forward_stroke_m > 0.0:
        forward_stroke_violation = (
            torch.relu(problem.minimum_forward_stroke_m - event_forward_progress)
            / problem.minimum_forward_stroke_m
        ) ** 2
    else:
        forward_stroke_violation = torch.zeros_like(event_forward_progress)
    if problem.minimum_recoil_stroke_m > 0.0:
        recoil_stroke_violation = (
            torch.relu(problem.minimum_recoil_stroke_m - event_recoil_distance)
            / problem.minimum_recoil_stroke_m
        ) ** 2
    else:
        recoil_stroke_violation = torch.zeros_like(event_recoil_distance)
    hit_violation = (
        position_violation
        + speed_violation
        + direction_violation
        + forward_stroke_violation
        + recoil_stroke_violation
    )

    excursion_violation = (
        torch.relu(maximum_drone_excursion - problem.maximum_drone_excursion_m)
        / problem.maximum_drone_excursion_m
    ) ** 2
    keepout_violation = (
        torch.relu(problem.drone_keepout_radius_m - minimum_drone_clearance)
        / problem.drone_keepout_radius_m
    ) ** 2
    speed_limit_violation = (
        torch.relu(maximum_drone_speed - simulator.settings.maximum_speed_m_s)
        / simulator.settings.maximum_speed_m_s
    ) ** 2
    safety_violation = (
        excursion_violation + keepout_violation + speed_limit_violation
    )

    acceleration = rollout.accelerations_m_s2
    acceleration_norm = torch.linalg.vector_norm(acceleration, dim=2)
    control_denominator = torch.arange(
        1,
        acceleration.shape[1] + 1,
        dtype=reference.dtype,
        device=reference.device,
    )
    effort_by_control = torch.cumsum(
        (acceleration_norm / simulator.settings.maximum_acceleration_m_s2) ** 2,
        dim=1,
    ) / control_denominator[None]
    if previous_acceleration_m_s2 is not None:
        previous = torch.as_tensor(
            previous_acceleration_m_s2,
            dtype=acceleration.dtype,
            device=acceleration.device,
        )
        if previous.ndim == 1:
            previous = previous[None].expand(batch, -1)
        acceleration_change = torch.linalg.vector_norm(
            torch.cat(
                (
                    acceleration[:, :1] - previous[:, None],
                    acceleration[:, 1:] - acceleration[:, :-1],
                ),
                dim=1,
            ),
            dim=2,
        )
        change_cost = (
            acceleration_change / simulator.settings.maximum_acceleration_m_s2
        ) ** 2
        change_denominator = control_denominator
    else:
        acceleration_change = torch.linalg.vector_norm(
            acceleration[:, 1:] - acceleration[:, :-1], dim=2
        )
        zero = torch.zeros((batch, 1), dtype=reference.dtype, device=reference.device)
        change_cost = torch.cat(
            (
                zero,
                (
                    acceleration_change
                    / simulator.settings.maximum_acceleration_m_s2
                )
                ** 2,
            ),
            dim=1,
        )
        change_denominator = torch.clamp(control_denominator - 1.0, min=1.0)
    smoothness_by_control = torch.cumsum(change_cost, dim=1) / change_denominator[None]
    event_control_indices = torch.div(
        impact_frames - 1,
        simulator.settings.steps_per_control,
        rounding_mode="floor",
    ).clamp(max=acceleration.shape[1] - 1)
    effort = effort_by_control[:, event_control_indices]
    smoothness = smoothness_by_control[:, event_control_indices]
    weight_sum = weights.acceleration_effort + weights.acceleration_smoothness
    regularization = (
        weights.acceleration_effort * effort
        + weights.acceleration_smoothness * smoothness
    ) / weight_sum
    tip_mass_kg = float(simulator.snapshot.model.parameters.vertex_masses_kg[-1])
    directional_energy_j = (
        0.5 * tip_mass_kg * torch.relu(directional_speed) ** 2
    )
    negative_directional_energy_j = -directional_energy_j
    impact_time_s = rollout.time_s[impact_frames][None].expand(batch, -1)
    impact_time = impact_time_s / torch.clamp(rollout.time_s[-1], min=1.0e-9)
    feasible = (safety_violation == 0.0) & (hit_violation == 0.0)

    return _ImpactEventTable(
        impact_frames=impact_frames,
        terms={
            "feasible": feasible,
            "safety_violation": safety_violation,
            "hit_violation": hit_violation,
            "constraint_violation": safety_violation + hit_violation,
            "impact_time": impact_time,
            "impact_time_s": impact_time_s,
            "regularization": regularization,
            "position_error_m": position_error,
            "planning_tip_error_limit_m": torch.full_like(
                position_error, planning_tip_error_limit
            ),
            "direction_error_deg": direction_error_deg,
            "planning_impact_angle_limit_deg": torch.full_like(
                direction_error_deg, planning_impact_angle_limit
            ),
            "directional_speed_m_s": directional_speed,
            "directional_tip_energy_j": directional_energy_j,
            "negative_directional_tip_energy_j": negative_directional_energy_j,
            "maximum_drone_excursion_m": maximum_drone_excursion,
            "minimum_drone_clearance_m": minimum_drone_clearance,
            "maximum_drone_speed_m_s": maximum_drone_speed,
            "maximum_forward_stroke_m": event_forward_progress,
            "recoil_stroke_m": event_recoil_distance,
            "acceleration_effort": effort,
            "acceleration_smoothness": smoothness,
            "position_violation": position_violation,
            "direction_violation": direction_violation,
            "speed_violation": speed_violation,
            "forward_stroke_violation": forward_stroke_violation,
            "recoil_stroke_violation": recoil_stroke_violation,
            "excursion_violation": excursion_violation,
            "keepout_violation": keepout_violation,
            "speed_limit_violation": speed_limit_violation,
        },
    )


def _select_impact_events(
    table: _ImpactEventTable,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    terms = table.terms
    order = _row_lexicographic_order(
        terms["safety_violation"],
        terms["hit_violation"],
        terms["negative_directional_tip_energy_j"],
        terms["impact_time"],
        terms["regularization"],
    )
    selected_column = order[:, 0]
    rows = torch.arange(selected_column.shape[0], device=selected_column.device)
    selected_terms = {
        name: value[rows, selected_column] for name, value in terms.items()
    }
    return selected_terms, table.impact_frames[selected_column]


def rollout_cost_terms(
    rollout: TensorRollout,
    initial_state: DroneCableState,
    problem: MpcProblem,
    simulator: WhipSimulator,
    previous_acceleration_m_s2: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluate the final frame using the same constrained event metrics."""

    table = _impact_event_table(
        rollout,
        initial_state,
        problem,
        simulator,
        CostWeights(),
        previous_acceleration_m_s2,
    )
    return {name: value[:, -1] for name, value in table.terms.items()}


def total_rollout_cost(
    terms: dict[str, torch.Tensor], weights: CostWeights
) -> torch.Tensor:
    """A readable scalar summary; selection itself is lexicographic."""

    del weights
    feasible = terms["feasible"]
    regularization = terms["regularization"] / (1.0 + terms["regularization"])
    energy_scale = torch.clamp(
        terms["directional_tip_energy_j"].detach().new_tensor(1.0e-3),
        min=1.0e-12,
    )
    feasible_score = (
        1.0 / (1.0 + terms["directional_tip_energy_j"] / energy_scale)
        + 1.0e-3 * terms["impact_time"]
        + 1.0e-6 * regularization
    )
    safe_infeasible = 2.0 + terms["hit_violation"] + 1.0e-3 * regularization
    unsafe = (
        4.0
        + terms["safety_violation"]
        + 1.0e-3 * terms["hit_violation"] / (1.0 + terms["hit_violation"])
    )
    return torch.where(
        feasible,
        feasible_score,
        torch.where(terms["safety_violation"] == 0.0, safe_infeasible, unsafe),
    )


def variable_impact_rollout_cost_terms(
    rollout: TensorRollout,
    initial_state: DroneCableState,
    problem: MpcProblem,
    simulator: WhipSimulator,
    weights: CostWeights,
    previous_acceleration_m_s2: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Select the safest, valid, highest-energy event for each rollout."""

    return _select_impact_events(
        _impact_event_table(
            rollout,
            initial_state,
            problem,
            simulator,
            weights,
            previous_acceleration_m_s2,
        )
    )


def _repeat_initial_state(state: DroneCableState, count: int) -> DroneCableState:
    return WhipSimulator._repeat_state(state, count)


def _select_rollout(rollout: TensorRollout, index: int) -> TensorRollout:
    """Detach one candidate before the static CUDA buffers are replayed."""

    return TensorRollout(
        time_s=rollout.time_s.detach().clone(),
        drone_positions_m=rollout.drone_positions_m[index : index + 1].detach().clone(),
        drone_velocities_m_s=rollout.drone_velocities_m_s[index : index + 1].detach().clone(),
        attachment_positions_m=rollout.attachment_positions_m[index : index + 1]
        .detach()
        .clone(),
        cable_positions_m=rollout.cable_positions_m[index : index + 1].detach().clone(),
        cable_velocities_m_s=rollout.cable_velocities_m_s[index : index + 1]
        .detach()
        .clone(),
        accelerations_m_s2=rollout.accelerations_m_s2[index : index + 1]
        .detach()
        .clone(),
    )


def _truncate_rollout(
    rollout: TensorRollout,
    frame_count: int,
    control_count: int,
) -> TensorRollout:
    return TensorRollout(
        time_s=rollout.time_s[:frame_count].detach().clone(),
        drone_positions_m=rollout.drone_positions_m[:, :frame_count].detach().clone(),
        drone_velocities_m_s=rollout.drone_velocities_m_s[:, :frame_count].detach().clone(),
        attachment_positions_m=rollout.attachment_positions_m[:, :frame_count]
        .detach()
        .clone(),
        cable_positions_m=rollout.cable_positions_m[:, :frame_count].detach().clone(),
        cable_velocities_m_s=rollout.cable_velocities_m_s[:, :frame_count]
        .detach()
        .clone(),
        accelerations_m_s2=rollout.accelerations_m_s2[:, :control_count]
        .detach()
        .clone(),
    )


class _IpoptCastingEvaluation:
    """Numerical IPOPT oracle backed by the deployed DDER rollout.

    CasADi supplies IPOPT, while this object keeps the already validated
    PyTorch cable dynamics as the single source of truth.  Jacobians are
    central finite differences in the small normalized casting space.  All
    perturbations are evaluated as one CUDA batch, so IPOPT does not require a
    symbolic reimplementation of DDER and does not pay one rollout per partial
    derivative.
    """

    _CONSTRAINT_COUNT = 8

    def __init__(
        self,
        simulator: WhipSimulator,
        initial_state: DroneCableState,
        problem: MpcProblem,
        control_count: int,
        weights: CostWeights,
        previous_acceleration_m_s2: np.ndarray | torch.Tensor | None,
        phase_schedule: CastingPhaseSchedule | None,
        optimize_impact_time: bool,
        finite_difference_step: float,
        cancelled: CancellationCallback | None,
    ) -> None:
        self.simulator = simulator
        self.initial_state = initial_state
        self.problem = problem
        self.control_count = control_count
        self.weights = weights
        self.phase_schedule = phase_schedule
        self.optimize_impact_time = optimize_impact_time
        self.finite_difference_step = finite_difference_step
        self.cancelled = cancelled
        self.previous_acceleration = (
            None
            if previous_acceleration_m_s2 is None
            else torch.as_tensor(
                previous_acceleration_m_s2,
                dtype=simulator.dtype,
                device=simulator.device,
            ).reshape(1, 3)
        )
        if phase_schedule is None:
            self.active_casting_indices = tuple(range(_CASTING_PARAMETER_COUNT))
        elif phase_schedule.elapsed_s < phase_schedule.injection_end_s - 1.0e-12:
            # The absolute phase schedule fixes the plane and both timing
            # variables. Elevation and the two stroke lengths remain active.
            self.active_casting_indices = (0, 2, 3)
        else:
            # Once injection has ended, only the recoil waypoint remains in
            # the decoded trajectory.
            self.active_casting_indices = (0, 3)
        self.variable_count = len(self.active_casting_indices) + int(
            optimize_impact_time
        )
        self.output_count = 1 + self._CONSTRAINT_COUNT
        self.best_x: np.ndarray | None = None
        self.best_rank: tuple[float, float, float] | None = None
        self.history: list[float] = []
        self._cached_value_x: np.ndarray | None = None
        self._cached_value: np.ndarray | None = None
        self._cached_jacobian_x: np.ndarray | None = None
        self._cached_jacobian: np.ndarray | None = None

    def initial_guess(
        self,
        warm_start_action: CastingAction | None,
    ) -> np.ndarray:
        if warm_start_action is None:
            # Start from a decisive but symmetric forward/recoil motion.  A
            # local constrained solver needs a non-degenerate cable response;
            # zero or tiny strokes provide almost no useful target gradient.
            normalized = np.asarray((0.5, 0.5, 0.9, 0.9, 0.5, 1.0))
        else:
            normalized = (
                _normalized_casting_action(
                    warm_start_action,
                    self.problem.maximum_drone_excursion_m,
                    dtype=self.simulator.dtype,
                    device=self.simulator.device,
                )
                .detach()
                .cpu()
                .numpy()
            )
        values = [float(normalized[index]) for index in self.active_casting_indices]
        if self.optimize_impact_time:
            values.append(0.80)
        return np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)

    def normalized_actions(self, x: np.ndarray) -> torch.Tensor:
        values = np.asarray(x, dtype=np.float64)
        if values.ndim == 1:
            values = values[None]
        if values.ndim != 2 or values.shape[1] != self.variable_count:
            raise ValueError("IPOPT casting variables have the wrong shape.")
        batch = values.shape[0]
        normalized = torch.full(
            (batch, _CASTING_PARAMETER_COUNT),
            0.5,
            dtype=self.simulator.dtype,
            device=self.simulator.device,
        )
        active = torch.as_tensor(
            values[:, : len(self.active_casting_indices)],
            dtype=self.simulator.dtype,
            device=self.simulator.device,
        )
        for column, casting_index in enumerate(self.active_casting_indices):
            normalized[:, casting_index] = active[:, column]
        return normalized

    def select_seed_impact_times(
        self,
        x: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Pair each casting seed with its best physics-frame impact event.

        A seven-dimensional Sobol sample is too sparse if impact time is
        sampled independently of a rapidly moving cable tip.  The event time
        is not a control, so it is legitimate and much more informative to
        scan all rollout frames for each six-dimensional casting seed before
        asking IPOPT for continuous-time refinement.
        """

        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.variable_count:
            raise ValueError("IPOPT seed candidates have the wrong shape.")
        if not self.optimize_impact_time:
            return values.copy(), {}
        normalized = self.normalized_actions(values)
        controls = casting_controls(
            normalized,
            self.initial_state,
            self.problem,
            self.simulator,
            self.control_count,
            self.phase_schedule,
        )
        with torch.no_grad():
            rollout = self.simulator.rollout(
                self.initial_state,
                controls,
                create_graph=False,
                cancelled=self.cancelled,
            )
            terms, impact_frames = variable_impact_rollout_cost_terms(
                rollout,
                self.initial_state,
                self.problem,
                self.simulator,
                self.weights,
                self.previous_acceleration,
            )
        first_time = self.simulator.settings.simulation_dt_s
        horizon = float(rollout.time_s[-1].detach().cpu())
        impact_times = rollout.time_s[impact_frames]
        fractions = torch.clamp(
            (impact_times - first_time) / max(horizon - first_time, 1.0e-12),
            0.0,
            1.0,
        )
        selected = values.copy()
        selected[:, -1] = fractions.detach().cpu().numpy()
        arrays = {
            name: value.detach().cpu().numpy().copy()
            for name, value in terms.items()
        }
        return selected, arrays

    @staticmethod
    def _interpolate(
        values: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        rows = torch.arange(values.shape[0], device=values.device)
        low = values[rows, lower]
        high = values[rows, upper]
        shape = (values.shape[0],) + (1,) * (values.ndim - 2)
        blend = alpha.reshape(shape)
        return low + blend * (high - low)

    def _evaluate_batch(self, x: np.ndarray, *, record: bool) -> np.ndarray:
        if self.cancelled is not None and self.cancelled():
            raise MpcCancelled("MPC stopped by user.")
        values = np.asarray(x, dtype=np.float64)
        if values.ndim == 1:
            values = values[None]
        normalized = self.normalized_actions(values)
        controls = casting_controls(
            normalized,
            self.initial_state,
            self.problem,
            self.simulator,
            self.control_count,
            self.phase_schedule,
        )
        with torch.no_grad():
            rollout = self.simulator.rollout(
                self.initial_state,
                controls,
                create_graph=False,
                cancelled=self.cancelled,
            )
            batch = values.shape[0]
            last_frame = rollout.frame_count - 1
            if self.optimize_impact_time:
                fraction = torch.as_tensor(
                    values[:, -1],
                    dtype=self.simulator.dtype,
                    device=self.simulator.device,
                )
                first_time = self.simulator.settings.simulation_dt_s
                horizon = float(rollout.time_s[-1].detach().cpu())
                impact_time = first_time + fraction * (horizon - first_time)
                frame_coordinate = impact_time / first_time
                lower = torch.floor(frame_coordinate).to(torch.long).clamp(1, last_frame)
                upper = (lower + 1).clamp(max=last_frame)
                alpha = torch.clamp(frame_coordinate - lower, 0.0, 1.0)
            else:
                lower = torch.full(
                    (batch,), last_frame, dtype=torch.long, device=self.simulator.device
                )
                upper = lower
                alpha = torch.zeros(
                    batch, dtype=self.simulator.dtype, device=self.simulator.device
                )
                impact_time = torch.full(
                    (batch,),
                    float(rollout.time_s[-1].detach().cpu()),
                    dtype=self.simulator.dtype,
                    device=self.simulator.device,
                )

            tip_position = self._interpolate(
                rollout.cable_positions_m[:, :, -1], lower, upper, alpha
            )
            tip_velocity = self._interpolate(
                rollout.cable_velocities_m_s[:, :, -1], lower, upper, alpha
            )
            drone_position = rollout.drone_positions_m
            drone_velocity = rollout.drone_velocities_m_s
            target = torch.tensor(
                self.problem.target_position_m,
                dtype=self.simulator.dtype,
                device=self.simulator.device,
            )
            direction = torch.tensor(
                self.problem.impact_direction,
                dtype=self.simulator.dtype,
                device=self.simulator.device,
            )
            if self.problem.drone_workspace_center_m is None:
                workspace_center = self.initial_state.drone_position_m.expand(batch, -1)
            else:
                workspace_center = torch.tensor(
                    self.problem.drone_workspace_center_m,
                    dtype=self.simulator.dtype,
                    device=self.simulator.device,
                )[None].expand(batch, -1)

            directional_speed = torch.sum(tip_velocity * direction[None], dim=1)
            lateral_velocity = tip_velocity - directional_speed[:, None] * direction[None]
            lateral_speed_squared = torch.sum(lateral_velocity**2, dim=1)
            position_error_squared = torch.sum((tip_position - target[None]) ** 2, dim=1)

            drone_displacement_squared = torch.sum(
                (drone_position - workspace_center[:, None]) ** 2, dim=2
            )
            drone_target_distance_squared = torch.sum(
                (drone_position - target[None, None]) ** 2, dim=2
            )
            drone_speed_squared = torch.sum(drone_velocity**2, dim=2)

            horizontal_forward = target - workspace_center[0]
            horizontal_forward = horizontal_forward.clone()
            horizontal_forward[2] = 0.0
            norm = torch.linalg.vector_norm(horizontal_forward)
            if float(norm.detach().cpu()) <= 1.0e-9:
                horizontal_forward = direction.clone()
                horizontal_forward[2] = 0.0
                norm = torch.linalg.vector_norm(horizontal_forward)
            if float(norm.detach().cpu()) <= 1.0e-9:
                horizontal_forward = direction.new_tensor((1.0, 0.0, 0.0))
                norm = direction.new_tensor(1.0)
            horizontal_forward = horizontal_forward / norm
            forward_progress = torch.sum(
                (drone_position - workspace_center[:, None])
                * horizontal_forward[None, None],
                dim=2,
            )
            frame_numbers = torch.arange(
                rollout.frame_count, device=self.simulator.device
            )[None]
            prefix_mask = frame_numbers <= upper[:, None]
            prefix_progress = torch.where(
                prefix_mask,
                forward_progress,
                torch.full_like(forward_progress, -torch.inf),
            )
            maximum_forward = torch.max(prefix_progress, dim=1).values
            impact_forward = self._interpolate(
                forward_progress[:, :, None], lower, upper, alpha
            )[:, 0]
            recoil = maximum_forward - impact_forward

            maximum_acceleration = self.simulator.settings.maximum_acceleration_m_s2
            effort = torch.mean(
                torch.sum(controls**2, dim=2) / maximum_acceleration**2,
                dim=1,
            )
            if controls.shape[1] > 1:
                changes = controls[:, 1:] - controls[:, :-1]
                if self.previous_acceleration is not None:
                    first = controls[:, :1] - self.previous_acceleration[:, None]
                    changes = torch.cat((first, changes), dim=1)
                smoothness = torch.mean(
                    torch.sum(changes**2, dim=2) / maximum_acceleration**2,
                    dim=1,
                )
            elif self.previous_acceleration is not None:
                smoothness = torch.sum(
                    (controls[:, 0] - self.previous_acceleration) ** 2,
                    dim=1,
                ) / maximum_acceleration**2
            else:
                smoothness = torch.zeros_like(effort)
            weight_sum = self.weights.acceleration_effort + self.weights.acceleration_smoothness
            regularization = (
                self.weights.acceleration_effort * effort
                + self.weights.acceleration_smoothness * smoothness
            ) / weight_sum

            speed_scale = self.problem.minimum_impact_speed_m_s
            tangent = math.tan(
                math.radians(self.problem.planning_impact_angle_limit_deg)
            )
            direction_scale_squared = max((speed_scale * tangent) ** 2, 1.0e-12)
            objective = (
                -(directional_speed / speed_scale) ** 2
                + 1.0e-3 * impact_time / max(float(rollout.time_s[-1]), 1.0e-9)
                + 1.0e-4 * regularization
            )
            constraints = torch.stack(
                (
                    position_error_squared / self.problem.planning_tip_error_limit_m**2 - 1.0,
                    (speed_scale - directional_speed) / speed_scale,
                    (
                        lateral_speed_squared
                        - tangent**2 * directional_speed**2
                    )
                    / direction_scale_squared,
                    torch.max(drone_displacement_squared, dim=1).values
                    / self.problem.maximum_drone_excursion_m**2
                    - 1.0,
                    1.0
                    - torch.min(drone_target_distance_squared, dim=1).values
                    / self.problem.drone_keepout_radius_m**2,
                    torch.max(drone_speed_squared, dim=1).values
                    / self.simulator.settings.maximum_speed_m_s**2
                    - 1.0,
                    (
                        self.problem.minimum_forward_stroke_m - maximum_forward
                    )
                    / max(self.problem.maximum_drone_excursion_m, 1.0e-9),
                    (self.problem.minimum_recoil_stroke_m - recoil)
                    / max(self.problem.maximum_drone_excursion_m, 1.0e-9),
                ),
                dim=1,
            )
            output = torch.cat((objective[:, None], constraints), dim=1)
            result = output.detach().cpu().numpy().astype(np.float64, copy=False)

        if record:
            for row, candidate in zip(result, values, strict=True):
                hit = float(np.sum(np.maximum(row[[1, 2, 3, 7, 8]], 0.0) ** 2))
                safety = float(np.sum(np.maximum(row[[4, 5, 6]], 0.0) ** 2))
                rank = (safety, hit, float(row[0]))
                if self.best_rank is None or rank < self.best_rank:
                    self.best_rank = rank
                    self.best_x = candidate.copy()
                    score = 1.0e6 * safety + 1.0e3 * hit + float(row[0])
                    self.history.append(
                        score if not self.history else min(self.history[-1], score)
                    )
        return result

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        point = np.asarray(x, dtype=np.float64).reshape(-1)
        if self._cached_value_x is not None and np.array_equal(
            point, self._cached_value_x
        ):
            assert self._cached_value is not None
            return self._cached_value.copy()
        value = self._evaluate_batch(point, record=True)[0]
        self._cached_value_x = point.copy()
        self._cached_value = value.copy()
        return value

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        nominal = np.asarray(x, dtype=np.float64).reshape(-1)
        if self._cached_jacobian_x is not None and np.array_equal(
            nominal, self._cached_jacobian_x
        ):
            assert self._cached_jacobian is not None
            return self._cached_jacobian.copy()
        step = self.finite_difference_step
        points: list[np.ndarray] = []
        denominators: list[float] = []
        for index in range(self.variable_count):
            lower = nominal.copy()
            upper = nominal.copy()
            lower[index] = max(0.0, nominal[index] - step)
            upper[index] = min(1.0, nominal[index] + step)
            points.extend((lower, upper))
            denominators.append(upper[index] - lower[index])
        values = self._evaluate_batch(np.stack(points), record=False)
        jacobian = np.empty(
            (self.output_count, self.variable_count), dtype=np.float64
        )
        for index, denominator in enumerate(denominators):
            jacobian[:, index] = (
                values[2 * index + 1] - values[2 * index]
            ) / max(denominator, 1.0e-12)
        self._cached_jacobian_x = nominal.copy()
        self._cached_jacobian = jacobian.copy()
        return jacobian


class _IpoptJacobianCallback(ca.Callback):
    def __init__(self, name: str, evaluation: _IpoptCastingEvaluation) -> None:
        self.evaluation = evaluation
        super().__init__()
        self.construct(name, {})

    def get_n_in(self) -> int:
        return 2

    def get_n_out(self) -> int:
        return 1

    def get_sparsity_in(self, index: int) -> ca.Sparsity:
        if index == 0:
            return ca.Sparsity.dense(self.evaluation.variable_count, 1)
        return ca.Sparsity.dense(self.evaluation.output_count, 1)

    def get_sparsity_out(self, _index: int) -> ca.Sparsity:
        return ca.Sparsity.dense(
            self.evaluation.output_count, self.evaluation.variable_count
        )

    def eval(self, arguments: list[ca.DM]) -> list[ca.DM]:
        x = np.asarray(arguments[0], dtype=np.float64).reshape(-1)
        return [ca.DM(self.evaluation.jacobian(x))]


class _IpoptEvaluationCallback(ca.Callback):
    def __init__(self, name: str, evaluation: _IpoptCastingEvaluation) -> None:
        self.evaluation = evaluation
        self.jacobian_callback: _IpoptJacobianCallback | None = None
        super().__init__()
        self.construct(name, {})

    def get_n_in(self) -> int:
        return 1

    def get_n_out(self) -> int:
        return 1

    def get_sparsity_in(self, _index: int) -> ca.Sparsity:
        return ca.Sparsity.dense(self.evaluation.variable_count, 1)

    def get_sparsity_out(self, _index: int) -> ca.Sparsity:
        return ca.Sparsity.dense(self.evaluation.output_count, 1)

    def eval(self, arguments: list[ca.DM]) -> list[ca.DM]:
        x = np.asarray(arguments[0], dtype=np.float64).reshape(-1)
        return [ca.DM(self.evaluation.evaluate(x))]

    def has_jacobian(self) -> bool:
        return True

    def get_jacobian(
        self,
        name: str,
        _input_names: list[str],
        _output_names: list[str],
        _options: dict[str, object],
    ) -> ca.Function:
        self.jacobian_callback = _IpoptJacobianCallback(
            name, self.evaluation
        )
        return self.jacobian_callback


def optimize_controls(
    simulator: WhipSimulator,
    initial_state: DroneCableState,
    problem: MpcProblem,
    optimizer_settings: OptimizerSettings,
    *,
    control_count: int | None = None,
    warm_start_action: CastingAction | None = None,
    weights: CostWeights = CostWeights(),
    previous_acceleration_m_s2: np.ndarray | torch.Tensor | None = None,
    progress: ProgressCallback | None = None,
    cancelled: CancellationCallback | None = None,
    optimize_impact_time: bool = False,
    truncate_at_impact: bool = True,
    phase_schedule: CastingPhaseSchedule | None = None,
) -> MpcPlan:
    """Solve the constrained casting problem with IPOPT.

    The nonlinear program retains the low-dimensional, target-aligned casting
    primitive.  IPOPT chooses its bounded geometric parameters and, when
    requested, the continuous impact time.  The objective maximizes directed
    free-tip speed; hit, direction, vehicle, workspace, and two-stroke
    requirements enter as explicit nonlinear constraints.
    """

    report = progress if progress is not None else (lambda _text: None)
    count = simulator.settings.control_count if control_count is None else int(control_count)
    if count < 1:
        raise ValueError("MPC horizon must contain at least one control interval.")
    evaluation = _IpoptCastingEvaluation(
        simulator,
        initial_state,
        problem,
        count,
        weights,
        previous_acceleration_m_s2,
        phase_schedule,
        optimize_impact_time,
        optimizer_settings.finite_difference_step,
        cancelled,
    )
    x0 = evaluation.initial_guess(warm_start_action)
    evaluation.evaluate(x0)
    if optimizer_settings.initial_samples > 0 and warm_start_action is None:
        # Whip dynamics are strongly non-convex in stroke duration and impact
        # time.  A single slow-motion initial guess can have a useless local
        # derivative even when a fast feasible cast exists.  A deterministic
        # low-discrepancy scan supplies IPOPT with a physically meaningful
        # basin while preserving IPOPT as the constrained local solver.
        sobol = torch.quasirandom.SobolEngine(
            dimension=evaluation.variable_count,
            scramble=True,
            seed=17,
        )
        candidates = sobol.draw(optimizer_settings.initial_samples).to(
            dtype=torch.float64
        ).numpy()
        candidates = np.concatenate((x0[None], candidates), axis=0)
        candidates, scan_terms = evaluation.select_seed_impact_times(candidates)
        seed_values = evaluation._evaluate_batch(candidates, record=True)
        hit_violation = np.sum(
            np.maximum(seed_values[:, [1, 2, 3, 7, 8]], 0.0) ** 2,
            axis=1,
        )
        safety_violation = np.sum(
            np.maximum(seed_values[:, [4, 5, 6]], 0.0) ** 2,
            axis=1,
        )
        order = np.lexsort((seed_values[:, 0], hit_violation, safety_violation))
        x0 = candidates[int(order[0])]
        feasible_seeds = int(
            np.count_nonzero((safety_violation == 0.0) & (hit_violation == 0.0))
        )
        safe_seeds = int(np.count_nonzero(safety_violation == 0.0))
        minimum_position_error = float(
            np.min(scan_terms.get("position_error_m", np.asarray((math.nan,))))
        )
        report(
            f"Deterministic seed scan: {len(candidates)} candidates, "
            f"safe={safe_seeds}, feasible={feasible_seeds}, "
            f"minimum sampled tip error={1000.0 * minimum_position_error:.1f}mm, "
            f"best hit violation={hit_violation[int(order[0])]:.4g}"
        )
    callback_name = f"mpc_eval_{id(evaluation):x}"
    callback = _IpoptEvaluationCallback(callback_name, evaluation)
    variables = ca.MX.sym("casting", evaluation.variable_count)
    outputs = callback(variables)
    solver = ca.nlpsol(
        f"mpc_ipopt_{id(evaluation):x}",
        "ipopt",
        {"x": variables, "f": outputs[0], "g": outputs[1:]},
        {
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": optimizer_settings.iterations,
            "ipopt.max_wall_time": optimizer_settings.maximum_wall_time_s,
            "ipopt.tol": optimizer_settings.tolerance,
            "ipopt.acceptable_tol": optimizer_settings.acceptable_tolerance,
            "ipopt.acceptable_iter": 3,
            "ipopt.constr_viol_tol": optimizer_settings.tolerance,
            "ipopt.acceptable_constr_viol_tol": optimizer_settings.acceptable_tolerance,
            "ipopt.hessian_approximation": "limited-memory",
            "ipopt.bound_relax_factor": 0.0,
        },
    )
    report(
        f"IPOPT: variables={evaluation.variable_count}, constraints="
        f"{evaluation._CONSTRAINT_COUNT}, max_iter={optimizer_settings.iterations}"
    )
    status = "not run"
    iteration_count = 0
    try:
        solution = solver(
            x0=x0,
            lbx=np.zeros(evaluation.variable_count),
            ubx=np.ones(evaluation.variable_count),
            lbg=np.full(evaluation._CONSTRAINT_COUNT, -np.inf),
            ubg=np.zeros(evaluation._CONSTRAINT_COUNT),
        )
        solved_x = np.asarray(solution["x"], dtype=np.float64).reshape(-1)
        evaluation.evaluate(solved_x)
        statistics = solver.stats()
        status = str(statistics.get("return_status", "unknown"))
        iteration_count = int(statistics.get("iter_count", 0))
    except MpcCancelled:
        raise
    except RuntimeError as error:
        if cancelled is not None and cancelled():
            raise MpcCancelled("MPC stopped by user.") from error
        status = f"callback/solver stopped: {error}"
    if evaluation.best_x is None:
        raise RuntimeError("IPOPT produced no finite control plan.")
    best_normalized_action = evaluation.normalized_actions(evaluation.best_x)[0]
    best_controls = casting_controls(
        best_normalized_action[None],
        initial_state,
        problem,
        simulator,
        count,
        phase_schedule,
    )[0]
    with torch.no_grad():
        best_rollout = simulator.rollout(
            initial_state,
            best_controls[None],
            create_graph=False,
            cancelled=cancelled,
        )
        previous = None
        if previous_acceleration_m_s2 is not None:
            previous = torch.as_tensor(
                previous_acceleration_m_s2,
                dtype=simulator.dtype,
                device=simulator.device,
            )
        if optimize_impact_time:
            tensor_terms, impact_frames = variable_impact_rollout_cost_terms(
                best_rollout,
                initial_state,
                problem,
                simulator,
                weights,
                previous,
            )
        else:
            tensor_terms = rollout_cost_terms(
                best_rollout,
                initial_state,
                problem,
                simulator,
                previous,
            )
            impact_frames = torch.full(
                (1,),
                best_rollout.frame_count - 1,
                dtype=torch.long,
                device=simulator.device,
            )
        costs = total_rollout_cost(tensor_terms, weights)
    if not bool(torch.all(torch.isfinite(costs)).detach().cpu()):
        raise RuntimeError("IPOPT MPC rollout produced a non-finite cost.")
    best_cost = float(costs[0].detach().cpu())
    best_impact_frame = int(impact_frames[0].detach().cpu())
    best_terms = {
        name: float(value[0].detach().cpu())
        for name, value in tensor_terms.items()
    }
    report(
        f"IPOPT {status} after {iteration_count} iteration(s): "
        f"feasible={'yes' if bool(best_terms['feasible']) else 'no'} "
        f"violation={best_terms['constraint_violation']:.4g} "
        f"impact={best_terms['impact_time_s']:.3f}s"
    )
    selected_impact_time_s = float(
        best_rollout.time_s[best_impact_frame].detach().cpu()
    )
    if truncate_at_impact:
        used_control_count = int(
            math.ceil(best_impact_frame / simulator.settings.steps_per_control)
        )
        best_controls = best_controls[:used_control_count]
        best_rollout = _truncate_rollout(
            best_rollout,
            best_impact_frame + 1,
            used_control_count,
        )
    target = torch.tensor(problem.target_position_m, dtype=simulator.dtype)
    direction = torch.tensor(problem.impact_direction, dtype=simulator.dtype)
    result = tensor_rollout_to_result(
        best_rollout,
        batch_index=0,
        target_position_m=target,
        impact_direction=direction,
        model_sha256=simulator.snapshot.sha256,
    )
    controls_array = best_controls.detach().cpu().numpy().copy()
    controls_array.setflags(write=False)
    casting_action = _casting_action_from_normalized(
        best_normalized_action,
        problem.maximum_drone_excursion_m,
    )
    if phase_schedule is not None:
        casting_action = replace(
            casting_action,
            recoil_deflection_deg=(
                0.0
                if phase_schedule.lock_to_target_plane
                else casting_action.recoil_deflection_deg
            ),
            reversal_fraction=(
                phase_schedule.injection_end_s / phase_schedule.motion_end_s
            ),
            motion_fraction=min(
                1.0,
                phase_schedule.motion_end_s
                / (count * simulator.settings.control_interval_s),
            ),
        )
    return MpcPlan(
        controls_m_s2=controls_array,
        prediction=result,
        cost=best_cost,
        cost_terms=best_terms,
        history=tuple(evaluation.history),
        impact_time_s=selected_impact_time_s,
        feasible=bool(best_terms["feasible"]),
        constraint_violation=best_terms["constraint_violation"],
        casting_action=casting_action,
    )


def run_receding_horizon_mpc(
    simulator: WhipSimulator,
    initial_state: DroneCableState,
    problem: MpcProblem,
    optimizer_settings: OptimizerSettings,
    *,
    weights: CostWeights = CostWeights(),
    progress: ProgressCallback | None = None,
    cancelled: CancellationCallback | None = None,
) -> MpcExecution:
    """Replan to one fixed impact time and execute only the leading controls."""

    report = progress if progress is not None else (lambda _text: None)
    ratio = optimizer_settings.replan_interval_s / simulator.settings.control_interval_s
    if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("MPC replan interval must be a multiple of control interval.")
    controls_per_replan = max(1, int(round(ratio)))
    remaining = simulator.settings.control_count
    state = initial_state
    warm_action: CastingAction | None = None
    elapsed = 0.0
    time_parts: list[np.ndarray] = []
    drone_position_parts: list[np.ndarray] = []
    drone_velocity_parts: list[np.ndarray] = []
    attachment_parts: list[np.ndarray] = []
    cable_position_parts: list[np.ndarray] = []
    cable_velocity_parts: list[np.ndarray] = []
    executed_controls: list[np.ndarray] = []
    replanning_costs: list[float] = []
    replan_index = 0
    while remaining > 0:
        if cancelled is not None and cancelled():
            raise MpcCancelled("MPC stopped by user.")
        replan_index += 1
        report(
            f"Replan {replan_index}: remaining horizon="
            f"{remaining * simulator.settings.control_interval_s:.2f}s"
        )
        plan = optimize_controls(
            simulator,
            state,
            problem,
            optimizer_settings,
            control_count=remaining,
            warm_start_action=warm_action,
            weights=weights,
            progress=report,
            cancelled=cancelled,
        )
        apply_count = min(controls_per_replan, remaining)
        controls = torch.tensor(
            plan.controls_m_s2[:apply_count],
            dtype=simulator.dtype,
            device=simulator.device,
        )[None]
        with torch.no_grad():
            segment = simulator.rollout(
                state, controls, create_graph=False, cancelled=cancelled
            )
        segment_result = tensor_rollout_to_result(
            segment,
            batch_index=0,
            target_position_m=torch.tensor(problem.target_position_m),
            impact_direction=torch.tensor(problem.impact_direction),
            model_sha256=simulator.snapshot.sha256,
        )
        start = 0 if not time_parts else 1
        time_parts.append(segment_result.time_s[start:] + elapsed)
        drone_position_parts.append(segment_result.drone_positions_m[start:])
        drone_velocity_parts.append(segment_result.drone_velocities_m_s[start:])
        attachment_parts.append(segment_result.attachment_positions_m[start:])
        cable_position_parts.append(segment_result.cable_positions_m[start:])
        cable_velocity_parts.append(segment_result.cable_velocities_m_s[start:])
        executed_controls.append(plan.controls_m_s2[:apply_count])
        replanning_costs.append(plan.cost)
        state = segment.final_state()
        elapsed += apply_count * simulator.settings.control_interval_s
        remaining -= apply_count
        warm_action = plan.casting_action if remaining else None

    def joined(parts: list[np.ndarray]) -> np.ndarray:
        value = np.concatenate(parts, axis=0)
        value.setflags(write=False)
        return value

    control_array = np.concatenate(executed_controls, axis=0)
    control_array.setflags(write=False)
    target = np.asarray(problem.target_position_m, dtype=np.float64)
    direction = np.asarray(problem.impact_direction, dtype=np.float64)
    target.setflags(write=False)
    direction.setflags(write=False)
    result = SimulationResult(
        time_s=joined(time_parts),
        drone_positions_m=joined(drone_position_parts),
        drone_velocities_m_s=joined(drone_velocity_parts),
        attachment_positions_m=joined(attachment_parts),
        cable_positions_m=joined(cable_position_parts),
        cable_velocities_m_s=joined(cable_velocity_parts),
        accelerations_m_s2=control_array,
        target_position_m=target,
        impact_direction=direction,
        model_sha256=simulator.snapshot.sha256,
    )
    return MpcExecution(result, control_array, tuple(replanning_costs))
