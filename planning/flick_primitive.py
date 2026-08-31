"""Compact smooth flick primitives and their bounded CEM search.

The primitive deliberately separates motion representation from physics.  It
decodes physical primitive parameters into the existing 16-knot normalized
production action, and every candidate is then evaluated by the unchanged
fixed-2048 production rollout contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import torch

from learning.policy_action import encode_physical_action
from learning.policy_context import PolicyContext
from simulator.simulator import CoupledSimulator

from .cem_task import VariableDurationWhipTask
from .production_cem import (
    ProductionCemSettings,
    evaluate_normalized_actions_fixed_batch,
    event_segment,
)
FLICK_PARAMETER_NAMES = (
    "azimuth_offset_rad",
    "elevation_rad",
    "first_pulse_m_s2",
    "reverse_pulse_m_s2",
    "maneuver_duration_s",
)
FLICK_PARAMETER_DIM = len(FLICK_PARAMETER_NAMES)
DIRECTED_FLICK_PARAMETER_NAMES = (
    "first_azimuth_offset_rad",
    "first_elevation_rad",
    "reverse_azimuth_offset_rad",
    "reverse_elevation_rad",
    "first_pulse_m_s2",
    "reverse_pulse_m_s2",
    "maneuver_duration_s",
)
DIRECTED_FLICK_PARAMETER_DIM = len(DIRECTED_FLICK_PARAMETER_NAMES)


@dataclass(frozen=True, slots=True)
class FlickPrimitiveBounds:
    """Hard physical envelope for the target-relative two-pulse primitive."""

    azimuth_offset_min_rad: float = -math.pi
    azimuth_offset_max_rad: float = math.pi
    elevation_min_rad: float = -0.5 * math.pi
    elevation_max_rad: float = 0.5 * math.pi
    pulse_min_m_s2: float = 0.0
    pulse_max_m_s2: float = 20.0
    duration_min_s: float = 0.45
    duration_max_s: float = 1.80
    switch_fraction: float = 0.50

    def __post_init__(self) -> None:
        if self.azimuth_offset_min_rad >= self.azimuth_offset_max_rad:
            raise ValueError("Invalid flick azimuth bounds.")
        if self.elevation_min_rad >= self.elevation_max_rad:
            raise ValueError("Invalid flick elevation bounds.")
        if self.pulse_min_m_s2 < 0.0 or self.pulse_min_m_s2 >= self.pulse_max_m_s2:
            raise ValueError("Invalid flick pulse bounds.")
        if self.duration_min_s >= self.duration_max_s:
            raise ValueError("Invalid flick duration bounds.")
        if not 0.0 < self.switch_fraction < 1.0:
            raise ValueError("The flick switch fraction must lie in (0,1).")


@dataclass(frozen=True, slots=True)
class FlickCemSettings:
    """Small CEM used only to test support of an audited primitive family."""

    population: int = 2048
    elite_fraction: float = 0.05
    maximum_iterations: int = 16
    minimum_iterations: int = 5
    polish_iterations_after_success: int = 2
    authoritative_top_n: int = 32
    old_distribution_weight: float = 0.30
    elite_distribution_weight: float = 0.70
    covariance_jitter: float = 1.0e-8
    initial_mean: tuple[float, ...] = (
        0.0,
        0.0,
        12.0,
        12.0,
        1.05,
    )
    initial_std: tuple[float, ...] = (
        1.0,
        0.65,
        5.0,
        5.0,
        0.25,
    )
    std_floor: tuple[float, ...] = (
        0.05,
        0.04,
        0.20,
        0.20,
        0.008,
    )

    def __post_init__(self) -> None:
        if self.population != 2048:
            raise ValueError("The flick audit uses one exact fixed-2048 population.")
        if not 0.0 < self.elite_fraction < 1.0:
            raise ValueError("Elite fraction must lie in (0,1).")
        if self.authoritative_top_n < 1 or self.authoritative_top_n > self.population:
            raise ValueError("Invalid authoritative top-N.")
        if self.minimum_iterations < 1 or self.maximum_iterations < self.minimum_iterations:
            raise ValueError("Invalid flick CEM iteration bounds.")
        if not (
            len(self.initial_mean) == len(self.initial_std) == len(self.std_floor)
        ):
            raise ValueError("Flick CEM vectors must have identical dimensions.")
        if len(self.initial_mean) not in (FLICK_PARAMETER_DIM, DIRECTED_FLICK_PARAMETER_DIM):
            raise ValueError("Only the audited five- or seven-parameter flick is supported.")
        if any(value <= 0.0 for value in self.initial_std + self.std_floor):
            raise ValueError("Flick CEM standard deviations must be positive.")


@dataclass(frozen=True, slots=True)
class FlickCemResult:
    context_id: str
    seed: int
    success: bool
    parameters: torch.Tensor
    normalized_action: torch.Tensor
    authoritative_metrics: dict[str, Any]
    iterations: int
    population_rollouts: int
    authoritative_rollouts: int
    runtime_s: float
    first_success_iteration: int | None
    history: tuple[dict[str, Any], ...]


def project_flick_parameters(
    parameters: torch.Tensor,
    bounds: FlickPrimitiveBounds,
) -> torch.Tensor:
    """Project physical primitive parameters without altering their meaning."""

    value = torch.as_tensor(parameters)
    one_row = value.ndim == 1
    if one_row:
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape[1] != FLICK_PARAMETER_DIM:
        raise ValueError("Flick parameters must have shape 5 or Bx5.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Flick parameters must be finite.")
    result = value.clone()
    # Azimuth is periodic.  Wrapping avoids artificial probability mass at a
    # clipped +/-pi boundary while retaining one canonical representation.
    result[:, 0] = torch.remainder(result[:, 0] + math.pi, 2.0 * math.pi) - math.pi
    result[:, 1] = result[:, 1].clamp(
        bounds.elevation_min_rad, bounds.elevation_max_rad
    )
    result[:, 2:4] = result[:, 2:4].clamp(
        bounds.pulse_min_m_s2, bounds.pulse_max_m_s2
    )
    result[:, 4] = result[:, 4].clamp(
        bounds.duration_min_s, bounds.duration_max_s
    )
    return result[0] if one_row else result


def project_directed_flick_parameters(
    parameters: torch.Tensor,
    bounds: FlickPrimitiveBounds,
) -> torch.Tensor:
    """Project the seven-parameter independent-pulse-direction primitive."""

    value = torch.as_tensor(parameters)
    one_row = value.ndim == 1
    if one_row:
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape[1] != DIRECTED_FLICK_PARAMETER_DIM:
        raise ValueError("Directed flick parameters must have shape 7 or Bx7.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Directed flick parameters must be finite.")
    result = value.clone()
    for index in (0, 2):
        result[:, index] = torch.remainder(
            result[:, index] + math.pi, 2.0 * math.pi
        ) - math.pi
    for index in (1, 3):
        result[:, index] = result[:, index].clamp(
            bounds.elevation_min_rad, bounds.elevation_max_rad
        )
    result[:, 4:6] = result[:, 4:6].clamp(
        bounds.pulse_min_m_s2, bounds.pulse_max_m_s2
    )
    result[:, 6] = result[:, 6].clamp(
        bounds.duration_min_s, bounds.duration_max_s
    )
    return result[0] if one_row else result


def _target_relative_axes(
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    target_direction_local: torch.Tensor,
) -> torch.Tensor:
    batch = int(azimuth.shape[0])
    direction = torch.as_tensor(
        target_direction_local, dtype=azimuth.dtype, device=azimuth.device
    )
    if direction.ndim == 1:
        direction = direction.unsqueeze(0)
    if direction.shape == (1, 3) and batch > 1:
        direction = direction.expand(batch, 3)
    if direction.shape != (batch, 3) or not bool(torch.isfinite(direction).all()):
        raise ValueError("Target directions must have shape 3, 1x3, or Bx3.")
    horizontal = direction.clone()
    horizontal[:, 2] = 0.0
    horizontal_norm = torch.linalg.vector_norm(horizontal, dim=-1, keepdim=True)
    if not bool((horizontal_norm > torch.finfo(azimuth.dtype).eps).all()):
        raise ValueError("Flick primitive requires a non-degenerate horizontal target direction.")
    forward = horizontal / horizontal_norm
    lateral = torch.stack(
        (-forward[:, 1], forward[:, 0], torch.zeros_like(forward[:, 0])), dim=-1
    )
    vertical = torch.zeros_like(forward)
    vertical[:, 2] = 1.0
    in_plane = (
        torch.cos(azimuth)[:, None] * forward
        + torch.sin(azimuth)[:, None] * lateral
    )
    axis = (
        torch.cos(elevation)[:, None] * in_plane
        + torch.sin(elevation)[:, None] * vertical
    )
    return axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True)


def _pulse_shapes(
    *,
    knot_count: int,
    bounds: FlickPrimitiveBounds,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    phase = torch.linspace(0.0, 1.0, knot_count, dtype=dtype, device=device)
    first_coordinate = torch.clamp(phase / bounds.switch_fraction, 0.0, 1.0)
    reverse_coordinate = torch.clamp(
        (phase - bounds.switch_fraction) / (1.0 - bounds.switch_fraction),
        0.0,
        1.0,
    )
    first_shape = torch.where(
        phase <= bounds.switch_fraction,
        torch.sin(math.pi * first_coordinate).square(),
        torch.zeros_like(phase),
    )
    reverse_shape = torch.where(
        phase >= bounds.switch_fraction,
        torch.sin(math.pi * reverse_coordinate).square(),
        torch.zeros_like(phase),
    )
    return first_shape, reverse_shape


def flick_parameters_to_knots(
    parameters: torch.Tensor,
    target_direction_local: torch.Tensor,
    *,
    knot_count: int = 16,
    bounds: FlickPrimitiveBounds | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode a target-relative 3-D flick into smooth acceleration knots.

    The first and reverse lobes share a learned spatial axis and use a fixed
    half-time switch.  Both lobes are ``sin^2`` profiles, so acceleration is
    exactly zero at the start, reversal, and end in continuous time.
    """

    envelope = FlickPrimitiveBounds() if bounds is None else bounds
    value = project_flick_parameters(parameters, envelope)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    azimuth, elevation = value[:, 0], value[:, 1]
    axis = _target_relative_axes(azimuth, elevation, target_direction_local)
    first_shape, reverse_shape = _pulse_shapes(
        knot_count=knot_count,
        bounds=envelope,
        dtype=value.dtype,
        device=value.device,
    )
    scalar = (
        value[:, 2:3] * first_shape[None]
        - value[:, 3:4] * reverse_shape[None]
    )
    knots = scalar[..., None] * axis[:, None, :]
    return knots, value[:, 4]


def directed_flick_parameters_to_knots(
    parameters: torch.Tensor,
    target_direction_local: torch.Tensor,
    *,
    knot_count: int = 16,
    bounds: FlickPrimitiveBounds | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode two independently directed smooth acceleration pulses."""

    envelope = FlickPrimitiveBounds() if bounds is None else bounds
    value = project_directed_flick_parameters(parameters, envelope)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    first_axis = _target_relative_axes(value[:, 0], value[:, 1], target_direction_local)
    reverse_axis = _target_relative_axes(value[:, 2], value[:, 3], target_direction_local)
    first_shape, reverse_shape = _pulse_shapes(
        knot_count=knot_count,
        bounds=envelope,
        dtype=value.dtype,
        device=value.device,
    )
    knots = (
        value[:, 4:5, None] * first_shape[None, :, None] * first_axis[:, None, :]
        + value[:, 5:6, None] * reverse_shape[None, :, None] * reverse_axis[:, None, :]
    )
    return knots, value[:, 6]


def encode_flick_as_production_action(
    parameters: torch.Tensor,
    target_direction_local: torch.Tensor,
    task: VariableDurationWhipTask,
    production_settings: ProductionCemSettings,
    *,
    bounds: FlickPrimitiveBounds | None = None,
) -> torch.Tensor:
    """Return the one authoritative normalized 49-D action representation."""

    knots, durations = flick_parameters_to_knots(
        parameters,
        target_direction_local,
        knot_count=task.cem.knot_count,
        bounds=bounds,
    )
    return encode_physical_action(
        knots,
        durations,
        task,
        duration_max_s=production_settings.duration_max_s,
    )


def encode_directed_flick_as_production_action(
    parameters: torch.Tensor,
    target_direction_local: torch.Tensor,
    task: VariableDurationWhipTask,
    production_settings: ProductionCemSettings,
    *,
    bounds: FlickPrimitiveBounds | None = None,
) -> torch.Tensor:
    """Encode the seven-parameter repair through the same production codec."""

    knots, durations = directed_flick_parameters_to_knots(
        parameters,
        target_direction_local,
        knot_count=task.cem.knot_count,
        bounds=bounds,
    )
    return encode_physical_action(
        knots,
        durations,
        task,
        duration_max_s=production_settings.duration_max_s,
    )


def _regularize_covariance(
    covariance: torch.Tensor,
    settings: FlickCemSettings,
) -> torch.Tensor:
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    floor = torch.as_tensor(settings.std_floor, dtype=symmetric.dtype).square()
    # Full covariance is preserved; diagonal floors only prevent a coordinate
    # from disappearing numerically during this bounded feasibility search.
    rebuilt = eigenvectors @ torch.diag(torch.clamp(eigenvalues, min=settings.covariance_jitter)) @ eigenvectors.T
    diagonal = torch.maximum(torch.diag(rebuilt), floor)
    rebuilt = rebuilt - torch.diag(torch.diag(rebuilt)) + torch.diag(diagonal)
    return 0.5 * (rebuilt + rebuilt.T) + torch.eye(
        rebuilt.shape[0], dtype=rebuilt.dtype
    ) * settings.covariance_jitter


def _metrics_with_segment(
    metrics: dict[str, Any],
    duration_s: float,
    production_settings: ProductionCemSettings,
) -> dict[str, Any]:
    result = dict(metrics)
    result["maneuver_duration_s"] = float(duration_s)
    result["hit_segment"] = event_segment(
        result["first_entry_time_s"],
        duration_s,
        production_settings.settle_duration_s,
    )
    return result


def _scientific_support_key(
    metrics: dict[str, Any], task: VariableDurationWhipTask
) -> tuple[float, float, float, float, float]:
    """Lexicographic key for a representation-support search.

    The production planner's legacy shaped objective is intentionally left
    unchanged.  This local audit key instead preserves safe tip entries and
    improves their missing hard gates, because the scientific question is
    whether the compact family contains a valid strike at all.
    """

    finite = bool(metrics["finite"])
    feasible = bool(metrics["feasible"])
    success = bool(metrics["success"])
    marker = metrics["first_entry_marker"]
    tip_entry = marker == 10 and math.isfinite(
        float(metrics["first_entry_tip_distance_m"])
    )
    speed_deficit = max(
        0.0,
        task.minimum_directed_speed_m_s
        - float(metrics["first_entry_directed_speed_m_s"]),
    ) / task.minimum_directed_speed_m_s
    direction_deficit = max(
        0.0,
        float(metrics["first_entry_direction_angle_deg"])
        - task.maximum_direction_error_deg,
    ) / task.maximum_direction_error_deg
    gate_deficit = speed_deficit * speed_deficit + direction_deficit * direction_deficit
    if success:
        return (
            0.0,
            float(metrics["task_cost"]),
            float(metrics["first_entry_tip_distance_m"]),
            float(metrics["first_entry_direction_angle_deg"]),
            -float(metrics["first_entry_directed_speed_m_s"]),
        )
    if finite and feasible and tip_entry:
        return (
            1.0,
            gate_deficit,
            float(metrics["first_entry_tip_distance_m"]),
            float(metrics["first_entry_direction_angle_deg"]),
            -float(metrics["first_entry_directed_speed_m_s"]),
        )
    if finite and feasible:
        return (
            2.0,
            float(metrics["minimum_tip_target_distance_m"]),
            float(metrics["best_event_direction_angle_deg"]),
            -float(metrics["best_event_directed_speed_m_s"]),
            float(metrics["task_cost"]),
        )
    if finite:
        return (
            3.0,
            float(metrics["feasibility_violation"]),
            float(metrics["minimum_tip_target_distance_m"]),
            float(metrics["task_cost"]),
            0.0,
        )
    return (4.0, float("inf"), float("inf"), float("inf"), 0.0)


def scientific_support_elite_order(
    result: Any, task: VariableDurationWhipTask
) -> np.ndarray:
    """Order a population by unchanged hard-gate support semantics."""

    def array(name: str) -> np.ndarray:
        return getattr(result, name).detach().cpu().numpy()

    success = array("success").astype(bool)
    finite = array("finite").astype(bool)
    feasible = array("feasible").astype(bool)
    marker = array("first_entry_marker")
    first_distance = np.nan_to_num(
        array("first_entry_tip_distance_m"), nan=np.inf, posinf=np.inf
    )
    first_speed = np.nan_to_num(
        array("first_entry_directed_speed_m_s"), nan=-np.inf
    )
    first_angle = np.nan_to_num(
        array("first_entry_direction_angle_deg"), nan=np.inf, posinf=np.inf
    )
    minimum_distance = np.nan_to_num(
        array("minimum_tip_target_distance_m"), nan=np.inf, posinf=np.inf
    )
    best_speed = np.nan_to_num(
        array("best_event_directed_speed_m_s"), nan=-np.inf
    )
    best_angle = np.nan_to_num(
        array("best_event_direction_angle_deg"), nan=np.inf, posinf=np.inf
    )
    task_cost = np.nan_to_num(array("task_cost"), nan=np.inf, posinf=np.inf)
    violation = np.nan_to_num(
        array("feasibility_violation"), nan=np.inf, posinf=np.inf
    )
    tip_entry = (marker == 10) & np.isfinite(first_distance)
    speed_deficit = np.maximum(
        0.0, task.minimum_directed_speed_m_s - first_speed
    ) / task.minimum_directed_speed_m_s
    direction_deficit = np.maximum(
        0.0, first_angle - task.maximum_direction_error_deg
    ) / task.maximum_direction_error_deg
    gate_deficit = speed_deficit**2 + direction_deficit**2
    category = np.where(
        success,
        0,
        np.where(
            finite & feasible & tip_entry,
            1,
            np.where(finite & feasible, 2, np.where(finite, 3, 4)),
        ),
    )
    key1 = np.where(
        success,
        task_cost,
        np.where(
            category == 1,
            gate_deficit,
            np.where(category == 2, minimum_distance, violation),
        ),
    )
    key2 = np.where(
        success,
        first_distance,
        np.where(category == 1, first_distance, np.where(category == 2, best_angle, minimum_distance)),
    )
    key3 = np.where(
        success,
        first_angle,
        np.where(category == 1, first_angle, np.where(category == 2, -best_speed, task_cost)),
    )
    key4 = np.where(success | (category == 1), -first_speed, task_cost)
    return np.lexsort((key4, key3, key2, key1, category)).astype(
        np.int64, copy=False
    )


def optimize_flick_cem(
    simulator: CoupledSimulator,
    repeated_context: PolicyContext,
    task: VariableDurationWhipTask,
    *,
    context_id: str,
    seed: int,
    production_settings: ProductionCemSettings,
    flick_settings: FlickCemSettings,
    bounds: FlickPrimitiveBounds,
    checkpoint_directory: Path,
    parameter_projector: Callable[[torch.Tensor, FlickPrimitiveBounds], torch.Tensor] = project_flick_parameters,
    action_encoder: Callable[..., torch.Tensor] = encode_flick_as_production_action,
    duration_index: int = 4,
) -> FlickCemResult:
    """Search only the selected primitive parameters under production physics."""

    if repeated_context.batch_size != flick_settings.population:
        raise ValueError("Flick CEM requires exactly one fixed-2048 context batch.")
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    dimension = len(flick_settings.initial_mean)
    if duration_index < 0:
        duration_index += dimension
    if not 0 <= duration_index < dimension:
        raise ValueError("Flick duration index is outside the parameter vector.")
    mean = parameter_projector(
        torch.tensor(flick_settings.initial_mean, dtype=torch.float64), bounds
    )
    covariance = torch.diag(
        torch.tensor(flick_settings.initial_std, dtype=torch.float64).square()
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    history: list[dict[str, Any]] = []
    best_parameters = mean.clone()
    best_action: torch.Tensor | None = None
    best_metrics: dict[str, Any] | None = None
    first_success: int | None = None
    population_rollouts = 0
    authoritative_rollouts = 0
    started = time.perf_counter()

    for iteration in range(1, flick_settings.maximum_iterations + 1):
        iteration_started = time.perf_counter()
        cholesky = torch.linalg.cholesky(_regularize_covariance(covariance, flick_settings))
        noise = torch.randn(
            (flick_settings.population, dimension),
            generator=generator,
            dtype=torch.float64,
        )
        parameters = parameter_projector(mean[None] + noise @ cholesky.T, bounds)
        parameters[0] = mean
        parameters[1] = best_parameters
        actions = action_encoder(
            parameters.float(),
            repeated_context.target_direction_local,
            task,
            production_settings,
            bounds=bounds,
        ).double()
        population = evaluate_normalized_actions_fixed_batch(
            simulator,
            repeated_context,
            actions,
            task,
            production_settings,
        )
        population_rollouts += flick_settings.population
        order = scientific_support_elite_order(population, task)
        elite_count = max(1, int(round(flick_settings.population * flick_settings.elite_fraction)))
        elite_indices = torch.from_numpy(order[:elite_count].copy()).long()
        elites = parameters[elite_indices]
        elite_mean = elites.mean(dim=0)
        centered = elites - elite_mean
        elite_covariance = centered.T @ centered / max(elite_count - 1, 1)
        mean = parameter_projector(
            flick_settings.old_distribution_weight * mean
            + flick_settings.elite_distribution_weight * elite_mean,
            bounds,
        )
        covariance = _regularize_covariance(
            flick_settings.old_distribution_weight * covariance
            + flick_settings.elite_distribution_weight * elite_covariance,
            flick_settings,
        )

        top_indices = torch.from_numpy(
            order[: flick_settings.authoritative_top_n].copy()
        ).long()
        top_parameters = parameters[top_indices]
        top_actions = actions[top_indices]
        authoritative = evaluate_normalized_actions_fixed_batch(
            simulator,
            repeated_context,
            top_actions,
            task,
            production_settings,
        )
        authoritative_rollouts += flick_settings.authoritative_top_n
        authoritative_order = scientific_support_elite_order(authoritative, task)
        selected = int(authoritative_order[0])
        selected_parameters = top_parameters[selected].clone()
        selected_action = top_actions[selected].clone()
        selected_metrics = _metrics_with_segment(
            authoritative.row(selected),
            float(selected_parameters[duration_index]),
            production_settings,
        )
        if best_metrics is None:
            better = True
        else:
            candidate_key = _scientific_support_key(selected_metrics, task)
            best_key = _scientific_support_key(best_metrics, task)
            better = candidate_key < best_key
        if better:
            best_parameters = selected_parameters
            best_action = selected_action
            best_metrics = selected_metrics
        if bool(selected_metrics["success"]) and first_success is None:
            first_success = iteration

        history.append(
            {
                "iteration": iteration,
                "population_success_count": int(population.success.sum().detach().cpu()),
                "population_feasible_count": int(population.feasible.sum().detach().cpu()),
                "authoritative_success_count": int(authoritative.success.sum().detach().cpu()),
                "selected_parameters": selected_parameters.tolist(),
                "selected_metrics": selected_metrics,
                "mean": mean.tolist(),
                "marginal_std": torch.sqrt(torch.diag(covariance)).tolist(),
                "runtime_s": time.perf_counter() - iteration_started,
            }
        )
        np.savez_compressed(
            checkpoint_directory / f"iteration_{iteration:03d}.npz",
            mean=mean.numpy(),
            covariance=covariance.numpy(),
            best_parameters=best_parameters.numpy(),
            rng_state=generator.get_state().numpy(),
        )
        if (
            first_success is not None
            and iteration >= flick_settings.minimum_iterations
            and iteration >= first_success + flick_settings.polish_iterations_after_success
        ):
            break

    if best_action is None or best_metrics is None:
        raise RuntimeError("Flick CEM produced no authoritative candidate.")
    # One final batch-one logical replay through the same padded production
    # evaluator prevents a population-only success claim.
    final = evaluate_normalized_actions_fixed_batch(
        simulator,
        repeated_context,
        best_action[None],
        task,
        production_settings,
    )
    authoritative_rollouts += 1
    final_metrics = _metrics_with_segment(
        final.row(0), float(best_parameters[duration_index]), production_settings
    )
    return FlickCemResult(
        context_id=context_id,
        seed=int(seed),
        success=bool(final_metrics["success"]),
        parameters=best_parameters.detach().cpu(),
        normalized_action=best_action.detach().cpu(),
        authoritative_metrics=final_metrics,
        iterations=len(history),
        population_rollouts=population_rollouts,
        authoritative_rollouts=authoritative_rollouts,
        runtime_s=time.perf_counter() - started,
        first_success_iteration=first_success,
        history=tuple(history),
    )
