"""Low-frequency MPPI for the matched-model drone-whip experiment.

Candidate trajectories are always ranked by the unchanged hard strike and
safety objective.  The default controller treats DDER as a black-box forward
model.  An opt-in research mode uses one exact differentiable DDER rollout to
shift half of the stochastic proposals along a smooth local strike gradient;
it does not replace MPPI weighting or introduce a new task reward.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import time
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as functional

from .problem import MpcProblem
from .simulator import (
    DroneCableState,
    SimulationResult,
    WhipSimulator,
    tensor_rollout_to_result,
)


ProgressCallback = Callable[[str], None]
CancellationCallback = Callable[[], bool]


# One authoritative objective for the supported receding-horizon application.
# Historical studies that need another objective must pass their values
# explicitly rather than inheriting an accidental set of class defaults.
PUBLIC_MPPI_OBJECTIVE: dict[str, float | str | bool] = {
    "objective_stage": "full",
    "position_sigma_m": 0.18,
    "velocity_gate_sigma_m": 0.12,
    "position_weight": 40.0,
    "speed_weight": 25.0,
    "predictive_speed_weight": 0.0,
    "direction_weight": 20.0,
    "success_cost": 600.0,
    "drone_displacement_weight": 2.0,
    "safety_weight": 180.0,
    "enforce_workspace_limit": False,
    "control_effort_weight": 1.0e-5,
    "control_smoothness_weight": 1.0e-5,
    "gradient_guidance_fraction": 0.0,
}


@dataclass(frozen=True, slots=True)
class MppiSettings:
    """Numerical settings for low-frequency path-integral control."""

    iterations: int = 20
    samples: int = 128
    rollout_batch_size: int = 128
    knot_count: int = 12
    temperature: float = 1.0
    acceleration_noise_sigma_m_s2: float = 8.0
    noise_decay: float = 0.92
    seed: int = 17
    objective_stage: str = "full"
    position_sigma_m: float = float(PUBLIC_MPPI_OBJECTIVE["position_sigma_m"])
    velocity_gate_sigma_m: float = float(
        PUBLIC_MPPI_OBJECTIVE["velocity_gate_sigma_m"]
    )
    position_weight: float = float(PUBLIC_MPPI_OBJECTIVE["position_weight"])
    speed_weight: float = float(PUBLIC_MPPI_OBJECTIVE["speed_weight"])
    predictive_speed_weight: float = float(
        PUBLIC_MPPI_OBJECTIVE["predictive_speed_weight"]
    )
    predictive_velocity_gate_sigma_m: float = 0.45
    predictive_speed_ratio: float = 0.25
    direction_weight: float = float(PUBLIC_MPPI_OBJECTIVE["direction_weight"])
    success_cost: float = float(PUBLIC_MPPI_OBJECTIVE["success_cost"])
    drone_displacement_weight: float = float(
        PUBLIC_MPPI_OBJECTIVE["drone_displacement_weight"]
    )
    safety_weight: float = float(PUBLIC_MPPI_OBJECTIVE["safety_weight"])
    enforce_workspace_limit: bool = bool(
        PUBLIC_MPPI_OBJECTIVE["enforce_workspace_limit"]
    )
    control_effort_weight: float = float(
        PUBLIC_MPPI_OBJECTIVE["control_effort_weight"]
    )
    control_smoothness_weight: float = float(
        PUBLIC_MPPI_OBJECTIVE["control_smoothness_weight"]
    )
    ground_height_m: float = 0.0
    ground_clearance_m: float = 0.03
    maximum_altitude_m: float = 3.0
    cable_drone_clearance_m: float = 0.05
    gradient_guidance_fraction: float = float(
        PUBLIC_MPPI_OBJECTIVE["gradient_guidance_fraction"]
    )
    gradient_step_sigma_ratio: float = 0.025
    smooth_softmin_temperature_m: float = 0.05
    gradient_norm_epsilon: float = 1.0e-9

    def __post_init__(self) -> None:
        if not 1 <= self.iterations <= 1000:
            raise ValueError("MPPI iterations must be between 1 and 1000.")
        if not 4 <= self.samples <= 65536:
            raise ValueError("MPPI samples must be between 4 and 65536.")
        if not 1 <= self.rollout_batch_size <= self.samples:
            raise ValueError("MPPI rollout batch size must lie within the sample count.")
        if not 2 <= self.knot_count <= 256:
            raise ValueError("MPPI knot count must be between 2 and 256.")
        if self.objective_stage not in {"position", "speed", "full"}:
            raise ValueError("MPPI objective stage must be position, speed, or full.")
        if not math.isfinite(self.noise_decay) or not 0.5 <= self.noise_decay <= 1.0:
            raise ValueError("MPPI noise decay must lie between 0.5 and 1.0.")
        positive_scales_and_limits = (
            self.temperature,
            self.acceleration_noise_sigma_m_s2,
            self.position_sigma_m,
            self.velocity_gate_sigma_m,
            self.predictive_velocity_gate_sigma_m,
            self.ground_clearance_m,
            self.maximum_altitude_m,
            self.cable_drone_clearance_m,
        )
        nonnegative = (
            self.position_weight,
            self.speed_weight,
            self.direction_weight,
            self.success_cost,
            self.drone_displacement_weight,
            self.safety_weight,
            self.control_effort_weight,
            self.control_smoothness_weight,
            self.predictive_speed_weight,
            self.gradient_guidance_fraction,
        )
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in positive_scales_and_limits
        ):
            raise ValueError(
                "MPPI scales, safety distances, temperature, and noise must be "
                "finite and positive."
            )
        if any(not math.isfinite(value) or value < 0.0 for value in nonnegative):
            raise ValueError("MPPI objective weights must be finite and non-negative.")
        if (
            not math.isfinite(self.predictive_speed_ratio)
            or not 0.0 < self.predictive_speed_ratio <= 1.0
        ):
            raise ValueError("MPPI predictive speed ratio must lie in (0, 1].")
        if self.gradient_guidance_fraction >= 1.0:
            raise ValueError(
                "Gradient guidance must leave a nonzero standard-MPPI sample fraction."
            )
        if (
            not math.isfinite(self.gradient_step_sigma_ratio)
            or self.gradient_step_sigma_ratio <= 0.0
        ):
            raise ValueError("Gradient step/noise ratio must be positive.")
        if (
            not math.isfinite(self.smooth_softmin_temperature_m)
            or self.smooth_softmin_temperature_m <= 0.0
        ):
            raise ValueError("Smooth temporal-softmin temperature must be positive.")
        if (
            not math.isfinite(self.gradient_norm_epsilon)
            or self.gradient_norm_epsilon <= 0.0
        ):
            raise ValueError("Gradient norm epsilon must be positive.")


@dataclass(frozen=True, slots=True)
class MppiPlan:
    """Best sampled plan and research diagnostics from one MPPI solve."""

    controls_m_s2: np.ndarray
    control_knots_m_s2: np.ndarray
    prediction: SimulationResult
    cost: float
    cost_terms: dict[str, float]
    history: tuple[float, ...]
    effective_sample_size_history: tuple[float, ...]
    sample_success_rate_history: tuple[float, ...]
    minimum_tip_error_history_m: tuple[float, ...]
    gradient_valid_history: tuple[bool, ...]
    gradient_reason_history: tuple[str, ...]
    smooth_surrogate_cost_history: tuple[float, ...]
    gradient_norm_history: tuple[float, ...]
    gradient_computation_time_s_history: tuple[float, ...]
    impact_time_s: float
    feasible: bool
    constraint_violation: float


@dataclass(frozen=True, slots=True)
class DderGuidance:
    """One local action-space direction obtained from exact DDER autograd."""

    negative_gradient_direction: torch.Tensor | None
    smooth_cost: float
    gradient_norm: float
    computation_time_s: float
    valid: bool
    reason: str


def _bound_vectors(values: torch.Tensor, maximum_norm: float) -> torch.Tensor:
    norms = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    scale = torch.clamp(maximum_norm / torch.clamp(norms, min=1.0e-12), max=1.0)
    return values * scale


def interpolate_control_knots(
    knots_m_s2: torch.Tensor,
    control_count: int,
    maximum_acceleration_m_s2: float,
) -> torch.Tensor:
    """Linearly interpolate low-frequency 3-D knots to actuator intervals."""

    knots = torch.as_tensor(knots_m_s2)
    if knots.ndim != 3 or knots.shape[1] < 2 or knots.shape[2] != 3:
        raise ValueError("MPPI control knots must have shape BxMx3 with M >= 2.")
    if control_count < 1:
        raise ValueError("MPPI requires at least one control interval.")
    if control_count == 1:
        controls = knots[:, :1]
    else:
        controls = functional.interpolate(
            knots.transpose(1, 2),
            size=control_count,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
    return _bound_vectors(controls, maximum_acceleration_m_s2)


def _sample_perturbations(
    sample_count: int,
    knot_count: int,
    sigma: float,
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Antithetic low-frequency perturbations with an exact nominal sample."""

    pair_count = (sample_count - 1 + 1) // 2
    base = torch.randn(
        (pair_count, knot_count, 3),
        generator=generator,
        dtype=dtype,
        device=device,
    ) * sigma
    paired = torch.cat((base, -base), dim=0)[: sample_count - 1]
    zero = torch.zeros((1, knot_count, 3), dtype=dtype, device=device)
    return torch.cat((zero, paired), dim=0)


def _sample_nonzero_perturbations(
    sample_count: int,
    knot_count: int,
    sigma: float,
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Antithetic Gaussian perturbations without an exact nominal member."""

    if sample_count < 1:
        return torch.empty((0, knot_count, 3), dtype=dtype, device=device)
    pair_count = (sample_count + 1) // 2
    base = torch.randn(
        (pair_count, knot_count, 3),
        generator=generator,
        dtype=dtype,
        device=device,
    ) * sigma
    return torch.cat((base, -base), dim=0)[:sample_count]


def smooth_strike_surrogate(
    rollout,
    problem: MpcProblem,
    settings: MppiSettings,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Smooth strike-only objective used solely to obtain a local gradient.

    A temporal softmin over tip distance produces smooth approximate impact
    position and velocity.  Velocity terms are deliberately not proximity
    gated here: otherwise a gradient can improve the surrogate merely by
    moving farther away and switching those terms off.  This excludes contact
    identity, success bonuses, safety, displacement, control, cable shape, and energy.
    Candidate selection still uses :func:`_mppi_event_objective` exclusively.
    """

    dtype = rollout.cable_positions_m.dtype
    device = rollout.cable_positions_m.device
    target = torch.as_tensor(problem.target_position_m, dtype=dtype, device=device)
    direction = torch.as_tensor(problem.impact_direction, dtype=dtype, device=device)
    tip_position = rollout.cable_positions_m[:, 1:, -1]
    tip_velocity = rollout.cable_velocities_m_s[:, 1:, -1]
    tip_distance = torch.linalg.vector_norm(tip_position - target[None, None], dim=2)
    temporal_weights = torch.softmax(
        -tip_distance / settings.smooth_softmin_temperature_m,
        dim=1,
    )
    approximate_position = torch.sum(
        temporal_weights[:, :, None] * tip_position, dim=1
    )
    approximate_velocity = torch.sum(
        temporal_weights[:, :, None] * tip_velocity, dim=1
    )
    distance = torch.linalg.vector_norm(approximate_position - target[None], dim=1)
    total_speed = torch.linalg.vector_norm(approximate_velocity, dim=1)
    directed_speed = torch.sum(approximate_velocity * direction[None], dim=1)
    direction_cosine = torch.clamp(
        directed_speed / torch.clamp(total_speed, min=1.0e-9), -1.0, 1.0
    )
    cone_cosine = math.cos(math.radians(problem.maximum_impact_angle_deg))

    position_cost = settings.position_weight * distance.square() / (
        distance.square() + settings.position_sigma_m**2
    )
    speed_cost = (
        settings.speed_weight
        * torch.relu(problem.minimum_impact_speed_m_s - directed_speed).square()
    )
    direction_cost = (
        settings.direction_weight
        * torch.relu(cone_cosine - direction_cosine).square()
    )
    if settings.objective_stage == "position":
        speed_cost = torch.zeros_like(speed_cost)
        direction_cost = torch.zeros_like(direction_cost)
    elif settings.objective_stage == "speed":
        direction_cost = torch.zeros_like(direction_cost)
    total = position_cost + speed_cost + direction_cost
    approximate_time = torch.sum(
        temporal_weights * rollout.time_s[None, 1:], dim=1
    )
    return total, {
        "position_error_m": distance,
        "directional_speed_m_s": directed_speed,
        "tip_speed_m_s": total_speed,
        "direction_cosine": direction_cosine,
        "approximate_impact_time_s": approximate_time,
        "position_cost": position_cost,
        "speed_cost": speed_cost,
        "direction_cost": direction_cost,
        "smooth_objective": total,
    }


def compute_dder_guidance(
    simulator: WhipSimulator,
    initial_state: DroneCableState,
    problem: MpcProblem,
    settings: MppiSettings,
    nominal_knots_m_s2: torch.Tensor,
    control_count: int,
    *,
    cancelled: CancellationCallback | None = None,
) -> DderGuidance:
    """Differentiate one smooth surrogate through the exact full DDER rollout."""

    if simulator.device.type == "cuda":
        torch.cuda.synchronize(simulator.device)
    started = time.perf_counter()
    direction: torch.Tensor | None = None
    smooth_value = math.nan
    gradient_norm = math.nan
    reason = "valid"
    valid = False
    try:
        nominal = nominal_knots_m_s2.detach().clone().requires_grad_(True)
        controls = interpolate_control_knots(
            nominal[None],
            control_count,
            simulator.settings.maximum_acceleration_m_s2,
        )
        rollout = simulator.rollout(
            initial_state,
            controls,
            create_graph=True,
            cancelled=cancelled,
        )
        smooth_cost, _ = smooth_strike_surrogate(rollout, problem, settings)
        gradient = torch.autograd.grad(
            smooth_cost.sum(), nominal, create_graph=False, retain_graph=False
        )[0]
        smooth_value = float(smooth_cost[0].detach().cpu())
        gradient_norm_tensor = torch.linalg.vector_norm(gradient)
        gradient_norm = float(gradient_norm_tensor.detach().cpu())
        finite = bool(torch.all(torch.isfinite(gradient)).detach().cpu())
        if not math.isfinite(smooth_value):
            reason = "non-finite smooth objective"
        elif not finite or not math.isfinite(gradient_norm):
            reason = "non-finite gradient"
        elif gradient_norm <= settings.gradient_norm_epsilon:
            reason = "near-zero gradient"
        else:
            # Store the predicted improving direction directly so callers do
            # not need to repeat a sign convention.
            direction = (-gradient / (gradient_norm_tensor + settings.gradient_norm_epsilon)).detach()
            valid = True
        del rollout, controls, gradient, smooth_cost
    except (RuntimeError, FloatingPointError) as error:
        reason = f"gradient rollout failed: {error}"
    if simulator.device.type == "cuda":
        torch.cuda.synchronize(simulator.device)
    elapsed = time.perf_counter() - started
    return DderGuidance(
        negative_gradient_direction=direction,
        smooth_cost=smooth_value,
        gradient_norm=gradient_norm,
        computation_time_s=elapsed,
        valid=valid,
        reason=reason,
    )


def _mppi_event_objective(
    rollout,
    initial_state: DroneCableState,
    problem: MpcProblem,
    simulator: WhipSimulator,
    settings: MppiSettings,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Evaluate the task at first tip contact or closest approach.

    The cable trajectory determines the maneuver.  No cable-shape, energy,
    wind-up, release-time, or manually prescribed whip term appears here.

    Contact is evaluated on the piecewise-linear trajectory between stored
    physics frames.  ``impact_frame`` remains the upper bracketing frame for
    compatibility with the controller, while the event time, position, and
    velocity are interpolated at the continuous entry event.
    """

    fixed_cuda_objective = (
        rollout.cable_positions_m.is_cuda
        and rollout.cable_positions_m.dtype == torch.float32
        and os.environ.get("DRONE_MPPI_FUSED_COST", "1") != "0"
    )
    if fixed_cuda_objective:
        from .cuda_mppi_cost import evaluate_cuda_mppi_cost

        return evaluate_cuda_mppi_cost(
            rollout, initial_state, problem, simulator, settings
        )

    dtype = rollout.cable_positions_m.dtype
    device = rollout.cable_positions_m.device
    batch = rollout.batch_size
    target = torch.as_tensor(problem.target_position_m, dtype=dtype, device=device)
    direction = torch.as_tensor(problem.impact_direction, dtype=dtype, device=device)

    tip_position = rollout.cable_positions_m[:, :, -1]
    tip_velocity = rollout.cable_velocities_m_s[:, :, -1]
    tip_start = tip_position[:, :-1]
    tip_step = tip_position[:, 1:] - tip_start
    relative_start = tip_start - target[None, None]
    quadratic_a = torch.sum(tip_step.square(), dim=2)
    quadratic_b = torch.sum(relative_start * tip_step, dim=2)
    quadratic_c = (
        torch.sum(relative_start.square(), dim=2)
        - problem.maximum_tip_error_m**2
    )
    discriminant = quadratic_b.square() - quadratic_a * quadratic_c
    nondegenerate = quadratic_a > torch.finfo(dtype).eps
    entry_fraction = (
        -quadratic_b - torch.sqrt(torch.clamp(discriminant, min=0.0))
    ) / torch.clamp(quadratic_a, min=torch.finfo(dtype).eps)
    starts_inside = quadratic_c <= 0.0
    swept_contact = starts_inside | (
        nondegenerate
        & (discriminant >= 0.0)
        & (entry_fraction >= 0.0)
        & (entry_fraction <= 1.0)
    )
    entry_fraction = torch.where(
        starts_inside,
        torch.zeros_like(entry_fraction),
        torch.clamp(entry_fraction, 0.0, 1.0),
    )
    has_contact = torch.any(swept_contact, dim=1)
    first_contact_interval = torch.argmax(swept_contact.to(torch.int64), dim=1)

    closest_fraction = torch.clamp(
        -quadratic_b / torch.clamp(quadratic_a, min=torch.finfo(dtype).eps),
        0.0,
        1.0,
    )
    closest_fraction = torch.where(
        nondegenerate, closest_fraction, torch.zeros_like(closest_fraction)
    )
    closest_position = tip_start + closest_fraction[:, :, None] * tip_step
    continuous_tip_distance = torch.linalg.vector_norm(
        closest_position - target[None, None], dim=2
    )
    minimum_tip_target_center_distance = torch.min(
        continuous_tip_distance, dim=1
    ).values
    closest_interval = torch.argmin(continuous_tip_distance, dim=1)
    selected_interval = torch.where(
        has_contact, first_contact_interval, closest_interval
    )
    rows = torch.arange(batch, device=device)
    selected_fraction = torch.where(
        has_contact,
        entry_fraction[rows, selected_interval],
        closest_fraction[rows, selected_interval],
    )
    impact_frames = selected_interval + 1
    event_position = tip_position[rows, selected_interval] + selected_fraction[:, None] * (
        tip_position[rows, impact_frames] - tip_position[rows, selected_interval]
    )
    distance = torch.linalg.vector_norm(event_position - target[None], dim=1)
    # ``distance`` is necessarily the target radius at first sphere entry, so
    # it is a contact/event metric rather than a useful measure of strike
    # placement.  Measure placement against the ideal point on the incoming
    # face of the target sphere instead.  A perfectly centred strike travelling
    # along ``direction`` reaches ``target - radius * direction`` first.
    ideal_impact_position = (
        target[None]
        - problem.maximum_tip_error_m * direction[None]
    )
    impact_surface_placement_error = torch.linalg.vector_norm(
        event_position - ideal_impact_position, dim=1
    )
    impact_surface_placement_error = torch.where(
        has_contact,
        impact_surface_placement_error,
        torch.full_like(impact_surface_placement_error, torch.inf),
    )
    velocity = tip_velocity[rows, selected_interval] + selected_fraction[:, None] * (
        tip_velocity[rows, impact_frames] - tip_velocity[rows, selected_interval]
    )
    event_time = rollout.time_s[selected_interval] + selected_fraction * (
        rollout.time_s[impact_frames] - rollout.time_s[selected_interval]
    )
    total_tip_speed = torch.linalg.vector_norm(velocity, dim=1)
    directed_speed = torch.sum(velocity * direction[None], dim=1)
    direction_cosine = directed_speed / torch.clamp(total_tip_speed, min=1.0e-9)
    direction_cosine = torch.clamp(direction_cosine, -1.0, 1.0)
    direction_error_deg = torch.rad2deg(torch.acos(direction_cosine))
    cone_cosine = math.cos(math.radians(problem.maximum_impact_angle_deg))

    position_cost = settings.position_weight * distance.square() / (
        distance.square() + settings.position_sigma_m**2
    )
    proximity = torch.exp(
        -distance.square() / (2.0 * settings.velocity_gate_sigma_m**2)
    )
    speed_deficit = torch.relu(problem.minimum_impact_speed_m_s - directed_speed)
    speed_cost = settings.speed_weight * proximity * speed_deficit.square()
    predictive_proximity = torch.exp(
        -distance.square()
        / (2.0 * settings.predictive_velocity_gate_sigma_m**2)
    )
    predictive_speed_deficit = torch.relu(
        settings.predictive_speed_ratio * problem.minimum_impact_speed_m_s
        - directed_speed
    )
    predictive_speed_cost = (
        settings.predictive_speed_weight
        * predictive_proximity
        * predictive_speed_deficit.square()
    )
    direction_deficit = torch.relu(cone_cosine - direction_cosine)
    direction_cost = settings.direction_weight * proximity * direction_deficit.square()
    if settings.objective_stage == "position":
        speed_cost = torch.zeros_like(speed_cost)
        predictive_speed_cost = torch.zeros_like(predictive_speed_cost)
        direction_cost = torch.zeros_like(direction_cost)
    elif settings.objective_stage == "speed":
        direction_cost = torch.zeros_like(direction_cost)

    initial_drone = initial_state.drone_position_m
    if initial_drone.shape[0] == 1:
        initial_drone = initial_drone.expand(batch, -1)
    event_drone = rollout.drone_positions_m[rows, selected_interval] + selected_fraction[:, None] * (
        rollout.drone_positions_m[rows, impact_frames]
        - rollout.drone_positions_m[rows, selected_interval]
    )
    event_drone_velocity = (
        rollout.drone_velocities_m_s[rows, selected_interval]
        + selected_fraction[:, None]
        * (
            rollout.drone_velocities_m_s[rows, impact_frames]
            - rollout.drone_velocities_m_s[rows, selected_interval]
        )
    )
    event_cable = (
        rollout.cable_positions_m[rows, selected_interval]
        + selected_fraction[:, None, None]
        * (
            rollout.cable_positions_m[rows, impact_frames]
            - rollout.cable_positions_m[rows, selected_interval]
        )
    )
    drone_displacement = torch.linalg.vector_norm(event_drone - initial_drone, dim=1)
    displacement_cost = settings.drone_displacement_weight * drone_displacement.square()

    all_drone_displacement = torch.linalg.vector_norm(
        rollout.drone_positions_m - initial_drone[:, None], dim=2
    )
    maximum_drone_excursion = torch.cummax(
        all_drone_displacement, dim=1
    ).values[rows, selected_interval]
    maximum_drone_excursion = torch.maximum(
        maximum_drone_excursion, drone_displacement
    )
    if settings.enforce_workspace_limit:
        workspace_violation = (
            torch.relu(maximum_drone_excursion - problem.maximum_drone_excursion_m)
            / problem.maximum_drone_excursion_m
        ).square()
    else:
        workspace_violation = torch.zeros_like(maximum_drone_excursion)
    drone_target_distance = torch.linalg.vector_norm(
        rollout.drone_positions_m - target[None, None], dim=2
    )
    minimum_drone_clearance = torch.cummin(
        drone_target_distance, dim=1
    ).values[rows, selected_interval]
    minimum_drone_clearance = torch.minimum(
        minimum_drone_clearance,
        torch.linalg.vector_norm(event_drone - target[None], dim=1),
    )
    keepout_violation = (
        torch.relu(problem.drone_keepout_radius_m - minimum_drone_clearance)
        / problem.drone_keepout_radius_m
    ).square()
    drone_speed = torch.linalg.vector_norm(rollout.drone_velocities_m_s, dim=2)
    maximum_drone_speed = torch.cummax(drone_speed, dim=1).values[
        rows, selected_interval
    ]
    maximum_drone_speed = torch.maximum(
        maximum_drone_speed,
        torch.linalg.vector_norm(event_drone_velocity, dim=1),
    )
    speed_limit_violation = (
        torch.relu(maximum_drone_speed - simulator.settings.maximum_speed_m_s)
        / simulator.settings.maximum_speed_m_s
    ).square()

    scene_height_by_frame = torch.minimum(
        rollout.drone_positions_m[:, :, 2],
        torch.amin(rollout.cable_positions_m[:, :, :, 2], dim=2),
    )
    minimum_scene_height = torch.cummin(
        scene_height_by_frame, dim=1
    ).values[rows, selected_interval]
    event_scene_height = torch.minimum(
        event_drone[:, 2], torch.amin(event_cable[:, :, 2], dim=1)
    )
    minimum_scene_height = torch.minimum(minimum_scene_height, event_scene_height)
    ground_violation = (
        torch.relu(
            settings.ground_height_m + settings.ground_clearance_m
            - minimum_scene_height
        )
        / settings.ground_clearance_m
    ).square()
    maximum_drone_altitude = torch.cummax(
        rollout.drone_positions_m[:, :, 2], dim=1
    ).values[rows, selected_interval]
    maximum_drone_altitude = torch.maximum(
        maximum_drone_altitude, event_drone[:, 2]
    )
    altitude_violation = (
        torch.relu(maximum_drone_altitude - settings.maximum_altitude_m)
        / settings.maximum_altitude_m
    ).square()

    cable_drone_distance = torch.linalg.vector_norm(
        rollout.cable_positions_m[:, :, 1:] - rollout.drone_positions_m[:, :, None],
        dim=3,
    )
    cable_drone_clearance_by_frame = torch.amin(cable_drone_distance, dim=2)
    minimum_cable_drone_clearance = torch.cummin(
        cable_drone_clearance_by_frame, dim=1
    ).values[rows, selected_interval]
    event_cable_drone_clearance = torch.amin(
        torch.linalg.vector_norm(
            event_cable[:, 1:] - event_drone[:, None], dim=2
        ),
        dim=1,
    )
    minimum_cable_drone_clearance = torch.minimum(
        minimum_cable_drone_clearance, event_cable_drone_clearance
    )
    cable_drone_violation = (
        torch.relu(
            settings.cable_drone_clearance_m - minimum_cable_drone_clearance
        )
        / settings.cable_drone_clearance_m
    ).square()

    # Non-tip contact uses the complete piecewise-linear cable, rather than
    # only its material vertices.  Between physics frames, each cable segment
    # is treated as a bilinear moving segment.  Conservative advancement uses
    # the maximum endpoint displacement as a Lipschitz bound, so a segment
    # cannot tunnel through the target sphere between samples.  Its reported
    # entry time is numerical (64 iterations, one-micrometre tolerance), not an
    # analytic rigid-body CCD solution.
    cable_a = rollout.cable_positions_m[:, :, :-1]
    cable_b = rollout.cable_positions_m[:, :, 1:]
    segment_vector = cable_b - cable_a
    target_from_a = target[None, None, None] - cable_a
    segment_denominator = torch.sum(segment_vector.square(), dim=3)
    spatial_fraction = torch.clamp(
        torch.sum(target_from_a * segment_vector, dim=3)
        / torch.clamp(segment_denominator, min=torch.finfo(dtype).eps),
        0.0,
        1.0,
    )
    spatial_fraction = torch.where(
        segment_denominator > torch.finfo(dtype).eps,
        spatial_fraction,
        torch.zeros_like(spatial_fraction),
    )
    segment_closest = cable_a + spatial_fraction[:, :, :, None] * segment_vector
    segment_distance_by_frame = torch.linalg.vector_norm(
        segment_closest - target[None, None, None], dim=3
    )

    moving_a0 = cable_a[:, :-1]
    moving_b0 = cable_b[:, :-1]
    moving_da = cable_a[:, 1:] - moving_a0
    moving_db = cable_b[:, 1:] - moving_b0
    speed_bound = torch.maximum(
        torch.linalg.vector_norm(moving_da, dim=3),
        torch.linalg.vector_norm(moving_db, dim=3),
    )

    # Conservative advancement can converge slowly when a slowly moving
    # endpoint is the contact feature but the opposite endpoint moves much
    # faster (the latter sets the global Lipschitz bound).  Endpoint motion is
    # linear, so solve those two point/sphere sweeps analytically and retain
    # the earlier result.  Conservative advancement remains responsible for
    # contacts in the segment interior.
    endpoint_start = torch.stack((moving_a0, moving_b0), dim=3)
    endpoint_step = torch.stack((moving_da, moving_db), dim=3)
    endpoint_relative = endpoint_start - target[None, None, None, None]
    endpoint_a = torch.sum(endpoint_step.square(), dim=4)
    endpoint_b = torch.sum(endpoint_relative * endpoint_step, dim=4)
    endpoint_c = (
        torch.sum(endpoint_relative.square(), dim=4)
        - problem.maximum_tip_error_m**2
    )
    endpoint_discriminant = endpoint_b.square() - endpoint_a * endpoint_c
    endpoint_nondegenerate = endpoint_a > torch.finfo(dtype).eps
    endpoint_entry = (
        -endpoint_b
        - torch.sqrt(torch.clamp(endpoint_discriminant, min=0.0))
    ) / torch.clamp(endpoint_a, min=torch.finfo(dtype).eps)
    endpoint_starts_inside = endpoint_c <= 0.0
    endpoint_hit = endpoint_starts_inside | (
        endpoint_nondegenerate
        & (endpoint_discriminant >= 0.0)
        & (endpoint_entry >= 0.0)
        & (endpoint_entry <= 1.0)
    )
    endpoint_entry = torch.where(
        endpoint_starts_inside,
        torch.zeros_like(endpoint_entry),
        torch.clamp(endpoint_entry, 0.0, 1.0),
    )
    analytic_endpoint_hit = torch.any(endpoint_hit, dim=3)
    analytic_endpoint_entry = torch.amin(
        torch.where(endpoint_hit, endpoint_entry, torch.inf), dim=3
    )
    endpoint_closest_fraction = torch.clamp(
        -endpoint_b / torch.clamp(endpoint_a, min=torch.finfo(dtype).eps),
        0.0,
        1.0,
    )
    endpoint_closest_fraction = torch.where(
        endpoint_nondegenerate,
        endpoint_closest_fraction,
        torch.zeros_like(endpoint_closest_fraction),
    )
    endpoint_closest_position = (
        endpoint_start
        + endpoint_closest_fraction[:, :, :, :, None] * endpoint_step
    )
    endpoint_minimum_distance = torch.amin(
        torch.linalg.vector_norm(
            endpoint_closest_position
            - target[None, None, None, None],
            dim=4,
        ),
        dim=3,
    )
    sweep_fraction = torch.zeros_like(speed_bound)
    sweep_hit = torch.zeros_like(speed_bound, dtype=torch.bool)
    sweep_hit_fraction = torch.full_like(speed_bound, torch.inf)
    sweep_minimum_distance = endpoint_minimum_distance
    contact_tolerance = 1.0e-6
    for _ in range(64):
        moving_a = moving_a0 + sweep_fraction[:, :, :, None] * moving_da
        moving_b = moving_b0 + sweep_fraction[:, :, :, None] * moving_db
        moving_edge = moving_b - moving_a
        moving_denominator = torch.sum(moving_edge.square(), dim=3)
        moving_fraction = torch.clamp(
            torch.sum((target[None, None, None] - moving_a) * moving_edge, dim=3)
            / torch.clamp(moving_denominator, min=torch.finfo(dtype).eps),
            0.0,
            1.0,
        )
        moving_fraction = torch.where(
            moving_denominator > torch.finfo(dtype).eps,
            moving_fraction,
            torch.zeros_like(moving_fraction),
        )
        moving_closest = moving_a + moving_fraction[:, :, :, None] * moving_edge
        moving_distance = torch.linalg.vector_norm(
            moving_closest - target[None, None, None], dim=3
        )
        sweep_minimum_distance = torch.minimum(
            sweep_minimum_distance, moving_distance
        )
        newly_hit = (~sweep_hit) & (
            moving_distance <= problem.maximum_tip_error_m + contact_tolerance
        )
        sweep_hit_fraction = torch.where(
            newly_hit, sweep_fraction, sweep_hit_fraction
        )
        sweep_hit = sweep_hit | newly_hit
        active = (~sweep_hit) & (sweep_fraction < 1.0)
        safe_step = (
            (moving_distance - problem.maximum_tip_error_m)
            / torch.clamp(speed_bound, min=torch.finfo(dtype).eps)
        )
        safe_step = torch.where(
            speed_bound > torch.finfo(dtype).eps,
            torch.clamp(safe_step, min=0.0),
            torch.ones_like(safe_step),
        )
        next_fraction = torch.clamp(sweep_fraction + safe_step, max=1.0)
        sweep_fraction = torch.where(active, next_fraction, sweep_fraction)

    moving_a = moving_a0 + sweep_fraction[:, :, :, None] * moving_da
    moving_b = moving_b0 + sweep_fraction[:, :, :, None] * moving_db
    moving_edge = moving_b - moving_a
    moving_denominator = torch.sum(moving_edge.square(), dim=3)
    moving_fraction = torch.clamp(
        torch.sum((target[None, None, None] - moving_a) * moving_edge, dim=3)
        / torch.clamp(moving_denominator, min=torch.finfo(dtype).eps),
        0.0,
        1.0,
    )
    moving_closest = moving_a + moving_fraction[:, :, :, None] * moving_edge
    moving_distance = torch.linalg.vector_norm(
        moving_closest - target[None, None, None], dim=3
    )
    sweep_minimum_distance = torch.minimum(sweep_minimum_distance, moving_distance)
    newly_hit = (~sweep_hit) & (
        moving_distance <= problem.maximum_tip_error_m + contact_tolerance
    )
    sweep_hit_fraction = torch.where(newly_hit, sweep_fraction, sweep_hit_fraction)
    sweep_hit = sweep_hit | newly_hit
    sweep_hit_fraction = torch.minimum(
        sweep_hit_fraction, analytic_endpoint_entry
    )
    sweep_hit = sweep_hit | analytic_endpoint_hit

    interval_index = torch.arange(
        rollout.frame_count - 1, dtype=dtype, device=device
    )[None, :, None]
    non_tip_event_key = interval_index + sweep_hit_fraction
    earliest_non_tip_key = torch.amin(
        torch.where(sweep_hit, non_tip_event_key, torch.inf), dim=(1, 2)
    )
    tip_event_key = selected_interval.to(dtype) + selected_fraction
    # The final segment contains the free tip.  Contact at the same continuous
    # instant as tip entry is therefore allowed; only strictly earlier contact
    # is a non-tip-first failure.
    non_tip_contact_before_tip = earliest_non_tip_key < tip_event_key - 1.0e-5

    frame_time = rollout.time_s[None, :, None]
    strictly_before_event = frame_time < event_time[:, None, None] - 1.0e-9
    minimum_non_tip_target_distance = torch.amin(
        torch.where(strictly_before_event, segment_distance_by_frame, torch.inf),
        dim=(1, 2),
    )
    completed_before_event = interval_index < selected_interval[:, None, None]
    minimum_swept_non_tip_distance = torch.amin(
        torch.where(completed_before_event, sweep_minimum_distance, torch.inf),
        dim=(1, 2),
    )
    minimum_non_tip_target_distance = torch.minimum(
        minimum_non_tip_target_distance, minimum_swept_non_tip_distance
    )
    minimum_non_tip_target_distance = torch.where(
        non_tip_contact_before_tip,
        torch.minimum(
            minimum_non_tip_target_distance,
            torch.full_like(
                minimum_non_tip_target_distance, problem.maximum_tip_error_m
            ),
        ),
        minimum_non_tip_target_distance,
    )
    penetration_violation = (
        torch.relu(
            problem.maximum_tip_error_m - minimum_non_tip_target_distance
        )
        / problem.maximum_tip_error_m
    ).square()
    non_tip_contact_violation = torch.where(
        non_tip_contact_before_tip,
        torch.maximum(penetration_violation, torch.ones_like(penetration_violation)),
        penetration_violation,
    )

    acceleration = rollout.accelerations_m_s2
    acceleration_norm = torch.linalg.vector_norm(acceleration, dim=2)
    event_control_index = torch.div(
        impact_frames - 1,
        simulator.settings.steps_per_control,
        rounding_mode="floor",
    ).clamp(max=acceleration.shape[1] - 1)
    maximum_acceleration = torch.cummax(acceleration_norm, dim=1).values[
        rows, event_control_index
    ]
    actuator_violation = (
        torch.relu(
            maximum_acceleration - simulator.settings.maximum_acceleration_m_s2
        )
        / simulator.settings.maximum_acceleration_m_s2
    ).square()
    effort_by_control = torch.cumsum(
        torch.sum(acceleration.square(), dim=2), dim=1
    )
    effort = effort_by_control[rows, event_control_index]
    if acceleration.shape[1] > 1:
        changes = acceleration[:, 1:] - acceleration[:, :-1]
        change_cost = torch.sum(changes.square(), dim=2)
        change_cost = torch.cat(
            (torch.zeros_like(change_cost[:, :1]), change_cost), dim=1
        )
        smoothness_by_control = torch.cumsum(change_cost, dim=1)
        smoothness = smoothness_by_control[rows, event_control_index]
    else:
        smoothness = torch.zeros_like(effort)
    control_cost = (
        settings.control_effort_weight * effort
        + settings.control_smoothness_weight * smoothness
    )

    safety_violation = (
        workspace_violation
        + keepout_violation
        + speed_limit_violation
        + ground_violation
        + altitude_violation
        + cable_drone_violation
        + non_tip_contact_violation
        + actuator_violation
    )
    safety_cost = settings.safety_weight * safety_violation

    position_success = has_contact
    speed_success = directed_speed >= problem.minimum_impact_speed_m_s
    direction_success = direction_cosine >= cone_cosine
    if settings.objective_stage == "position":
        task_success = position_success
    elif settings.objective_stage == "speed":
        task_success = position_success & speed_success
    else:
        task_success = position_success & speed_success & direction_success
    safe = safety_violation == 0.0
    success_cost = torch.where(
        task_success & safe,
        torch.full_like(distance, -settings.success_cost),
        torch.zeros_like(distance),
    )
    total_cost = (
        position_cost
        + speed_cost
        + predictive_speed_cost
        + direction_cost
        + success_cost
        + displacement_cost
        + safety_cost
        + control_cost
    )

    position_violation = (
        torch.relu(distance - problem.maximum_tip_error_m)
        / problem.maximum_tip_error_m
    ).square()
    speed_violation = (
        torch.relu(problem.minimum_impact_speed_m_s - directed_speed)
        / problem.minimum_impact_speed_m_s
    ).square()
    direction_violation = direction_deficit.square()
    task_violation = position_violation
    if settings.objective_stage in {"speed", "full"}:
        task_violation = task_violation + speed_violation
    if settings.objective_stage == "full":
        task_violation = task_violation + direction_violation

    diagnostics = {
        "feasible": task_success & safe,
        "constraint_violation": task_violation + safety_violation,
        "impact_frame": impact_frames,
        "impact_time_s": event_time,
        "position_error_m": distance,
        "minimum_tip_target_center_distance_m": minimum_tip_target_center_distance,
        "impact_surface_placement_error_m": impact_surface_placement_error,
        "directional_speed_m_s": directed_speed,
        "tip_speed_m_s": total_tip_speed,
        "direction_cosine": direction_cosine,
        "direction_error_deg": direction_error_deg,
        "geometric_tip_contact": has_contact,
        "physical_tip_contact": torch.zeros_like(has_contact),
        "drone_displacement_at_impact_m": drone_displacement,
        "maximum_drone_excursion_m": maximum_drone_excursion,
        "minimum_drone_clearance_m": minimum_drone_clearance,
        "maximum_drone_speed_m_s": maximum_drone_speed,
        "minimum_scene_height_m": minimum_scene_height,
        "maximum_drone_altitude_m": maximum_drone_altitude,
        "minimum_cable_drone_clearance_m": minimum_cable_drone_clearance,
        "minimum_non_tip_target_distance_m": minimum_non_tip_target_distance,
        "maximum_acceleration_m_s2": maximum_acceleration,
        "acceleration_effort": effort,
        "acceleration_smoothness": smoothness,
        "position_cost": position_cost,
        "speed_cost": speed_cost,
        "predictive_speed_cost": predictive_speed_cost,
        "direction_cost": direction_cost,
        "success_cost": success_cost,
        "drone_displacement_cost": displacement_cost,
        "safety_cost": safety_cost,
        "control_cost": control_cost,
        "workspace_violation": workspace_violation,
        "keepout_violation": keepout_violation,
        "speed_limit_violation": speed_limit_violation,
        "ground_violation": ground_violation,
        "altitude_violation": altitude_violation,
        "cable_drone_violation": cable_drone_violation,
        "non_tip_contact_violation": non_tip_contact_violation,
        "actuator_violation": actuator_violation,
        "mppi_objective": total_cost,
    }
    return total_cost, diagnostics


def evaluate_mppi_rollout(
    rollout,
    initial_state: DroneCableState,
    problem: MpcProblem,
    simulator: WhipSimulator,
    settings: MppiSettings,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Evaluate one or more trajectories without a hard excursion constraint."""

    objective, diagnostics = _mppi_event_objective(
        rollout, initial_state, problem, simulator, settings
    )
    impact_frames = diagnostics.pop("impact_frame")
    return objective, diagnostics, impact_frames


def optimize_mppi(
    simulator: WhipSimulator,
    initial_state: DroneCableState,
    problem: MpcProblem,
    settings: MppiSettings = MppiSettings(),
    *,
    control_count: int | None = None,
    warm_start_knots_m_s2: np.ndarray | torch.Tensor | None = None,
    objective_reference_state: DroneCableState | None = None,
    progress: ProgressCallback | None = None,
    cancelled: CancellationCallback | None = None,
) -> MppiPlan:
    """Optimize a smooth open-loop whip using batched low-frequency MPPI.

    When guidance is enabled, its differentiable surrogate changes only where
    half of the stochastic proposals are centered.  The hard real objective is
    still the sole source of MPPI weights and best-trajectory selection.
    """

    report = progress if progress is not None else (lambda _message: None)
    # Receding-horizon rollouts must start from the current measured cable
    # state, while displacement and excursion retain the original maneuver
    # reference.  One-shot callers keep the historical behavior by default.
    objective_state = (
        initial_state
        if objective_reference_state is None
        else objective_reference_state
    )
    count = simulator.settings.control_count if control_count is None else int(control_count)
    if count < 1:
        raise ValueError("MPPI horizon must contain at least one control interval.")
    if settings.knot_count > count + 1:
        raise ValueError("MPPI knot count cannot exceed control count plus one.")

    dtype = simulator.dtype
    device = simulator.device
    maximum_acceleration = simulator.settings.maximum_acceleration_m_s2
    if warm_start_knots_m_s2 is None:
        nominal = torch.zeros(
            (settings.knot_count, 3), dtype=dtype, device=device
        )
    else:
        # ``MppiPlan`` arrays are deliberately read-only.  Copy the warm start
        # so PyTorch never wraps a non-writable NumPy buffer.
        nominal = torch.tensor(
            warm_start_knots_m_s2, dtype=dtype, device=device
        ).reshape(settings.knot_count, 3)
        nominal = _bound_vectors(nominal, maximum_acceleration)

    generator = torch.Generator(device=device)
    generator.manual_seed(settings.seed)
    history: list[float] = []
    ess_history: list[float] = []
    success_history: list[float] = []
    error_history: list[float] = []
    gradient_valid_history: list[bool] = []
    gradient_reason_history: list[str] = []
    surrogate_history: list[float] = []
    gradient_norm_history: list[float] = []
    gradient_time_history: list[float] = []
    best_cost = math.inf
    best_knots: torch.Tensor | None = None

    report(
        f"Low-frequency MPPI: samples={settings.samples}, batch="
        f"{settings.rollout_batch_size}, knots={settings.knot_count}, "
        f"iterations={settings.iterations}, temperature={settings.temperature:g}, "
        f"noise={settings.acceleration_noise_sigma_m_s2:g}m/s^2, "
        f"DDER-guided={100.0 * settings.gradient_guidance_fraction:.0f}%"
    )
    for iteration in range(settings.iterations):
        if cancelled is not None and cancelled():
            raise RuntimeError("MPPI stopped by user.")
        sigma = settings.acceleration_noise_sigma_m_s2 * (
            settings.noise_decay**iteration
        )
        if settings.gradient_guidance_fraction > 0.0:
            guidance = compute_dder_guidance(
                simulator,
                initial_state,
                problem,
                settings,
                nominal,
                count,
                cancelled=cancelled,
            )
        else:
            guidance = DderGuidance(
                negative_gradient_direction=None,
                smooth_cost=math.nan,
                gradient_norm=math.nan,
                computation_time_s=0.0,
                valid=False,
                reason="disabled",
            )
        gradient_valid_history.append(guidance.valid)
        gradient_reason_history.append(guidance.reason)
        surrogate_history.append(guidance.smooth_cost)
        gradient_norm_history.append(guidance.gradient_norm)
        gradient_time_history.append(guidance.computation_time_s)

        with torch.no_grad():
            if guidance.valid and guidance.negative_gradient_direction is not None:
                guided_count = int(
                    round(settings.samples * settings.gradient_guidance_fraction)
                )
                guided_count = min(max(guided_count, 1), settings.samples - 1)
                standard_count = settings.samples - guided_count
                standard_noise = _sample_perturbations(
                    standard_count,
                    settings.knot_count,
                    sigma,
                    generator=generator,
                    dtype=dtype,
                    device=device,
                )
                guided_noise = _sample_nonzero_perturbations(
                    guided_count,
                    settings.knot_count,
                    sigma,
                    generator=generator,
                    dtype=dtype,
                    device=device,
                )
                action_dimension = settings.knot_count * 3
                gradient_step_l2 = (
                    settings.gradient_step_sigma_ratio
                    * sigma
                    * math.sqrt(action_dimension)
                )
                guided_center = (
                    gradient_step_l2 * guidance.negative_gradient_direction
                )
                candidates = torch.cat(
                    (
                        nominal[None] + standard_noise,
                        nominal[None] + guided_center[None] + guided_noise,
                    ),
                    dim=0,
                )
            else:
                perturbations = _sample_perturbations(
                    settings.samples,
                    settings.knot_count,
                    sigma,
                    generator=generator,
                    dtype=dtype,
                    device=device,
                )
                candidates = nominal[None] + perturbations
            candidates = _bound_vectors(candidates, maximum_acceleration)
            applied_delta = candidates - nominal[None]
            costs_parts: list[torch.Tensor] = []
            error_parts: list[torch.Tensor] = []
            speed_parts: list[torch.Tensor] = []
            feasible_parts: list[torch.Tensor] = []
            for start in range(0, settings.samples, settings.rollout_batch_size):
                if cancelled is not None and cancelled():
                    raise RuntimeError("MPPI stopped by user.")
                stop = min(start + settings.rollout_batch_size, settings.samples)
                controls = interpolate_control_knots(
                    candidates[start:stop], count, maximum_acceleration
                )
                rollout = simulator.rollout(
                    initial_state,
                    controls,
                    create_graph=False,
                    cancelled=cancelled,
                )
                costs, diagnostics = _mppi_event_objective(
                    rollout,
                    objective_state,
                    problem,
                    simulator,
                    settings,
                )
                costs_parts.append(costs)
                error_parts.append(diagnostics["position_error_m"])
                speed_parts.append(diagnostics["directional_speed_m_s"])
                feasible_parts.append(diagnostics["feasible"])
                del rollout, controls

            costs = torch.cat(costs_parts)
            errors = torch.cat(error_parts)
            speeds = torch.cat(speed_parts)
            feasible = torch.cat(feasible_parts)
            if not bool(torch.all(torch.isfinite(costs)).detach().cpu()):
                raise RuntimeError("MPPI produced a non-finite rollout objective.")

            minimum = torch.min(costs)
            importance = torch.exp(
                -(costs - minimum) / settings.temperature
            )
            weight_sum = torch.sum(importance)
            weights = importance / torch.clamp(weight_sum, min=1.0e-30)
            ess = 1.0 / torch.sum(weights.square())
            nominal = _bound_vectors(
                nominal + torch.sum(weights[:, None, None] * applied_delta, dim=0),
                maximum_acceleration,
            )

            candidate_index = int(torch.argmin(costs).detach().cpu())
            candidate_cost = float(costs[candidate_index].detach().cpu())
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_knots = candidates[candidate_index].detach().clone()
            history.append(best_cost)
            ess_value = float(ess.detach().cpu())
            success_rate = float(torch.mean(feasible.to(dtype)).detach().cpu())
            minimum_error = float(torch.min(errors).detach().cpu())
            ess_history.append(ess_value)
            success_history.append(success_rate)
            error_history.append(minimum_error)
            report(
                f"MPPI {iteration + 1}/{settings.iterations}: "
                f"best={best_cost:.4g}, min tip error={1000.0 * minimum_error:.1f}mm, "
                f"selected speed={float(speeds[candidate_index].cpu()):.2f}m/s, "
                f"success={100.0 * success_rate:.1f}%, ESS={ess_value:.1f}/{settings.samples}, "
                f"sigma={sigma:.2f}m/s^2, "
                f"gradient={'yes' if guidance.valid else 'no'}"
            )

    if best_knots is None:
        raise RuntimeError("MPPI produced no candidate plan.")
    with torch.no_grad():
        best_controls = interpolate_control_knots(
            best_knots[None], count, maximum_acceleration
        )
        best_rollout = simulator.rollout(
            initial_state,
            best_controls,
            create_graph=False,
            cancelled=cancelled,
        )
        objective, final_terms, impact_frames = evaluate_mppi_rollout(
            best_rollout, objective_state, problem, simulator, settings
        )


    impact_frame = int(impact_frames[0].detach().cpu())
    terms = {
        name: float(value[0].detach().cpu())
        for name, value in final_terms.items()
    }
    target = torch.as_tensor(problem.target_position_m, dtype=dtype)
    direction = torch.as_tensor(problem.impact_direction, dtype=dtype)
    prediction = tensor_rollout_to_result(
        best_rollout,
        batch_index=0,
        target_position_m=target,
        impact_direction=direction,
        model_sha256=simulator.snapshot.sha256,
    )

    controls_array = best_controls[0].detach().cpu().numpy().copy()
    knots_array = best_knots.detach().cpu().numpy().copy()
    controls_array.setflags(write=False)
    knots_array.setflags(write=False)
    return MppiPlan(
        controls_m_s2=controls_array,
        control_knots_m_s2=knots_array,
        prediction=prediction,
        cost=float(objective[0].detach().cpu()),
        cost_terms=terms,
        history=tuple(history),
        effective_sample_size_history=tuple(ess_history),
        sample_success_rate_history=tuple(success_history),
        minimum_tip_error_history_m=tuple(error_history),
        gradient_valid_history=tuple(gradient_valid_history),
        gradient_reason_history=tuple(gradient_reason_history),
        smooth_surrogate_cost_history=tuple(surrogate_history),
        gradient_norm_history=tuple(gradient_norm_history),
        gradient_computation_time_s_history=tuple(gradient_time_history),
        impact_time_s=terms["impact_time_s"],
        feasible=bool(final_terms["feasible"][0].detach().cpu()),
        constraint_violation=terms["constraint_violation"],
    )
