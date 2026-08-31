"""Derivative-free full-horizon MPPI over acceleration knots."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import torch

from simulator.simulator import CoupledSimulator
from simulator.state import SimulatorState

from .command_parameterization import project_acceleration_knots
from .rollout import run_population_rollout
from .task import CanonicalWhipTask


@dataclass(frozen=True, slots=True)
class MppiIterationRecord:
    iteration: int
    current_best_cost: float
    best_ever_cost: float
    effective_sample_size: float
    target_effective_sample_size: float
    selected_temperature: float
    minimum_target_error_m: float
    best_overall_tip_error_m: float
    best_feasible_tip_error_m: float | None
    best_feasible_cost: float | None
    best_directed_tip_speed_m_s: float
    best_direction_error_deg: float
    success_count: int
    feasible_count: int
    best_row_index: int
    best_is_nominal: bool
    iteration_runtime_s: float
    cumulative_runtime_s: float


@dataclass(frozen=True, slots=True)
class MppiResult:
    best_knots_m_s2: torch.Tensor
    best_metrics: dict[str, object]
    best_feasible_knots_m_s2: torch.Tensor | None
    best_feasible_metrics: dict[str, object] | None
    best_overall_knots_m_s2: torch.Tensor
    best_overall_metrics: dict[str, object]
    best_iteration: int
    best_row_index: int
    final_nominal_knots_m_s2: torch.Tensor
    iteration_history: tuple[MppiIterationRecord, ...]
    total_runtime_s: float
    candidate_rollouts_evaluated: int
    stopped_after_success: bool
    hard_stop_reached: bool


def adaptive_temperature_weights(
    costs: torch.Tensor,
    *,
    eligible: torch.Tensor | None = None,
    target_fraction: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose one temperature from a cost vector to reach a target ESS.

    The fixed bisection uses only already-computed candidate costs.  Invalid
    candidates receive exactly zero weight.  The returned tuple is
    ``(weights, temperature, achieved_ess, target_ess)``.
    """

    vector = torch.as_tensor(costs)
    if vector.ndim != 1 or vector.numel() < 1:
        raise ValueError("Adaptive temperature requires a non-empty cost vector.")
    if not 0.0 < target_fraction <= 1.0:
        raise ValueError("ESS target fraction must lie in (0, 1].")
    mask = torch.isfinite(vector) if eligible is None else (
        torch.as_tensor(eligible, device=vector.device, dtype=torch.bool)
        & torch.isfinite(vector)
    )
    if mask.shape != vector.shape:
        raise ValueError("Adaptive-temperature eligibility must match costs.")
    if not bool(torch.any(mask).detach().cpu()):
        mask = torch.ones_like(mask)

    work = vector.to(torch.float64)
    minimum = torch.amin(
        torch.where(mask, work, torch.full_like(work, float("inf")))
    )
    shifted = torch.where(mask, work - minimum, torch.full_like(work, float("inf")))
    eligible_count = torch.sum(mask).to(torch.float64)
    requested_target = vector.new_tensor(
        float(vector.numel()) * target_fraction, dtype=torch.float64
    )
    target = torch.clamp(requested_target, min=1.0, max=eligible_count)

    def normalized_for_temperature(
        temperature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = torch.where(
            mask,
            torch.exp(-shifted / torch.clamp(temperature, min=1.0e-15)),
            torch.zeros_like(work),
        )
        normalized = raw / torch.clamp(torch.sum(raw), min=torch.finfo(work.dtype).tiny)
        ess = torch.reciprocal(torch.sum(normalized.square()))
        return normalized, ess

    scale = torch.clamp(
        torch.amax(torch.where(mask, shifted, torch.zeros_like(shifted))),
        min=1.0,
    )
    lower = torch.clamp(scale * 1.0e-10, min=1.0e-12)
    upper = scale.clone()
    for _ in range(16):
        _weights, upper_ess = normalized_for_temperature(upper)
        upper = torch.where(upper_ess < target, upper * 2.0, upper)
    for _ in range(48):
        middle = torch.sqrt(lower * upper)
        _weights, middle_ess = normalized_for_temperature(middle)
        lower = torch.where(middle_ess < target, middle, lower)
        upper = torch.where(middle_ess < target, upper, middle)
    weights64, achieved = normalized_for_temperature(upper)
    return weights64.to(vector.dtype), upper, achieved, target


def optimize_full_horizon_mppi(
    simulator: CoupledSimulator,
    initial_state: SimulatorState,
    initial_nominal_knots_m_s2: torch.Tensor,
    task: CanonicalWhipTask,
    iteration_callback: Callable[[MppiIterationRecord], None] | None = None,
) -> MppiResult:
    """Run the one authorized seed-42 canonical full-horizon solve."""

    settings = task.mppi
    # The frozen residual MLP uses float32 GEMMs whose arithmetic path depends
    # on row count.  Standardize only that small upstream evaluation shape;
    # DDER still advances the true candidate batch (1, 8, or 2048).  This makes
    # a logical batch-one replay use the same UAV/residual arithmetic as the
    # sampled population without changing the model equations or parameters.
    simulator.uav_model.set_fixed_evaluation_batch_size(
        settings.fixed_uav_evaluation_batch_size
    )
    nominal = project_acceleration_knots(
        torch.as_tensor(
            initial_nominal_knots_m_s2,
            dtype=simulator.dtype,
            device=simulator.device,
        ),
        task.maximum_command_acceleration_m_s2,
    )
    if nominal.shape != (settings.knot_count, 3):
        raise ValueError("Initial nominal must have shape Kx3.")
    generator = torch.Generator(device=simulator.device)
    generator.manual_seed(settings.random_seed)
    global_best_ranked_cost = float("inf")
    global_best_feasible_cost = float("inf")
    global_best_feasible_knots: torch.Tensor | None = None
    global_best_feasible_metrics: dict[str, object] | None = None
    global_best_feasible_iteration = -1
    global_best_feasible_row = -1
    global_best_infeasible_violation = float("inf")
    global_best_infeasible_task_cost = float("inf")
    global_best_infeasible_knots = nominal.clone()
    global_best_infeasible_metrics: dict[str, object] = {}
    global_best_infeasible_iteration = -1
    global_best_infeasible_row = -1
    global_best_overall_cost = float("inf")
    global_best_overall_knots = nominal.clone()
    global_best_overall_metrics: dict[str, object] = {}
    history: list[MppiIterationRecord] = []
    success_iteration: int | None = None
    hard_stop_reached = False
    if simulator.device.type == "cuda":
        torch.cuda.synchronize(simulator.device)
    solve_start = time.perf_counter()

    for iteration in range(settings.maximum_iterations):
        iteration_start = time.perf_counter()
        epsilon = torch.randn(
            (settings.sample_count, settings.knot_count, 3),
            dtype=simulator.dtype,
            device=simulator.device,
            generator=generator,
        ) * settings.sigma_m_s2
        epsilon[0].zero_()
        candidates = project_acceleration_knots(
            nominal[None] + epsilon,
            task.maximum_command_acceleration_m_s2,
        )
        result = run_population_rollout(simulator, initial_state, candidates, task)
        costs = result.cost
        current_best_index_tensor = torch.argmin(costs)
        current_best_index = int(current_best_index_tensor.detach().cpu())
        current_best_cost = float(costs[current_best_index].detach().cpu())

        (
            normalized_weights,
            selected_temperature_tensor,
            effective_sample_size_tensor,
            target_ess_tensor,
        ) = adaptive_temperature_weights(
            costs,
            eligible=result.finite,
            target_fraction=settings.adaptive_ess_target_fraction,
        )
        nominal = project_acceleration_knots(
            nominal
            + torch.sum(normalized_weights[:, None, None] * epsilon, dim=0),
            task.maximum_command_acceleration_m_s2,
        )
        success_count_tensor = torch.sum(result.success)
        feasible_count_tensor = torch.sum(result.feasible)
        best_overall_tip_error_tensor = torch.amin(
            result.minimum_tip_target_distance_m
        )
        feasible_tip_errors = torch.where(
            result.feasible,
            result.minimum_tip_target_distance_m,
            torch.full_like(result.minimum_tip_target_distance_m, float("inf")),
        )
        best_feasible_tip_error_tensor = torch.amin(feasible_tip_errors)
        feasible_task_costs = torch.where(
            result.feasible,
            result.task_cost,
            torch.full_like(result.task_cost, float("inf")),
        )
        best_feasible_index_tensor = torch.argmin(feasible_task_costs)
        best_feasible_cost_tensor = feasible_task_costs[best_feasible_index_tensor]
        overall_task_costs = torch.where(
            result.finite,
            result.task_cost,
            torch.full_like(result.task_cost, float("inf")),
        )
        best_overall_index_tensor = torch.argmin(overall_task_costs)
        best_overall_cost_tensor = overall_task_costs[best_overall_index_tensor]
        if simulator.device.type == "cuda":
            torch.cuda.synchronize(simulator.device)
        iteration_runtime = time.perf_counter() - iteration_start
        cumulative_runtime = time.perf_counter() - solve_start
        success_count = int(success_count_tensor.detach().cpu())
        feasible_count = int(feasible_count_tensor.detach().cpu())
        ess = float(effective_sample_size_tensor.detach().cpu())
        target_ess = float(target_ess_tensor.detach().cpu())
        selected_temperature = float(selected_temperature_tensor.detach().cpu())
        current_metrics = result.row(current_best_index)
        global_best_ranked_cost = min(global_best_ranked_cost, current_best_cost)
        best_overall_cost = float(best_overall_cost_tensor.detach().cpu())
        best_overall_index = int(best_overall_index_tensor.detach().cpu())
        if best_overall_cost < global_best_overall_cost:
            global_best_overall_cost = best_overall_cost
            global_best_overall_knots = candidates[best_overall_index].clone()
            global_best_overall_metrics = result.row(best_overall_index)
        best_feasible_cost: float | None = None
        best_feasible_tip_error: float | None = None
        if feasible_count > 0:
            best_feasible_cost = float(best_feasible_cost_tensor.detach().cpu())
            best_feasible_tip_error = float(
                best_feasible_tip_error_tensor.detach().cpu()
            )
            best_feasible_index = int(best_feasible_index_tensor.detach().cpu())
            if best_feasible_cost < global_best_feasible_cost:
                global_best_feasible_cost = best_feasible_cost
                global_best_feasible_knots = candidates[best_feasible_index].clone()
                global_best_feasible_metrics = result.row(best_feasible_index)
                global_best_feasible_iteration = iteration
                global_best_feasible_row = best_feasible_index
        infeasible_violation = torch.where(
            result.finite & ~result.feasible,
            result.feasibility_violation,
            torch.full_like(result.feasibility_violation, float("inf")),
        )
        best_infeasible_index_tensor = torch.argmin(infeasible_violation)
        best_infeasible_violation_tensor = infeasible_violation[
            best_infeasible_index_tensor
        ]
        best_infeasible_violation = float(
            best_infeasible_violation_tensor.detach().cpu()
        )
        if torch.isfinite(best_infeasible_violation_tensor):
            best_infeasible_index = int(best_infeasible_index_tensor.detach().cpu())
            best_infeasible_task_cost = float(
                result.task_cost[best_infeasible_index].detach().cpu()
            )
            better_infeasible = (
                best_infeasible_violation < global_best_infeasible_violation
                or (
                    best_infeasible_violation == global_best_infeasible_violation
                    and best_infeasible_task_cost
                    < global_best_infeasible_task_cost
                )
            )
            if better_infeasible:
                global_best_infeasible_violation = best_infeasible_violation
                global_best_infeasible_task_cost = best_infeasible_task_cost
                global_best_infeasible_knots = candidates[
                    best_infeasible_index
                ].clone()
                global_best_infeasible_metrics = result.row(best_infeasible_index)
                global_best_infeasible_iteration = iteration
                global_best_infeasible_row = best_infeasible_index
        record = MppiIterationRecord(
                iteration=iteration + 1,
                current_best_cost=current_best_cost,
                best_ever_cost=global_best_ranked_cost,
                effective_sample_size=ess,
                target_effective_sample_size=target_ess,
                selected_temperature=selected_temperature,
                minimum_target_error_m=float(
                    current_metrics["minimum_tip_target_distance_m"]
                ),
                best_overall_tip_error_m=float(
                    best_overall_tip_error_tensor.detach().cpu()
                ),
                best_feasible_tip_error_m=best_feasible_tip_error,
                best_feasible_cost=best_feasible_cost,
                best_directed_tip_speed_m_s=float(
                    current_metrics["best_event_directed_speed_m_s"]
                ),
                best_direction_error_deg=float(
                    current_metrics["best_event_direction_angle_deg"]
                ),
                success_count=success_count,
                feasible_count=feasible_count,
                best_row_index=current_best_index,
                best_is_nominal=current_best_index == 0,
                iteration_runtime_s=iteration_runtime,
                cumulative_runtime_s=cumulative_runtime,
            )
        history.append(record)
        if iteration_callback is not None:
            iteration_callback(record)
        if success_count > 0 and success_iteration is None:
            success_iteration = iteration
        if (
            success_iteration is not None
            and iteration >= success_iteration + settings.success_polishing_iterations
        ):
            break
        if cumulative_runtime >= settings.hard_stop_s:
            hard_stop_reached = True
            break

    total_runtime = time.perf_counter() - solve_start
    if global_best_feasible_knots is not None:
        selected_knots = global_best_feasible_knots
        selected_metrics = global_best_feasible_metrics or {}
        selected_iteration = global_best_feasible_iteration
        selected_row = global_best_feasible_row
    else:
        selected_knots = global_best_infeasible_knots
        selected_metrics = global_best_infeasible_metrics
        selected_iteration = global_best_infeasible_iteration
        selected_row = global_best_infeasible_row
    return MppiResult(
        best_knots_m_s2=selected_knots,
        best_metrics=selected_metrics,
        best_feasible_knots_m_s2=global_best_feasible_knots,
        best_feasible_metrics=global_best_feasible_metrics,
        best_overall_knots_m_s2=global_best_overall_knots,
        best_overall_metrics=global_best_overall_metrics,
        best_iteration=selected_iteration + 1,
        best_row_index=selected_row,
        final_nominal_knots_m_s2=nominal,
        iteration_history=tuple(history),
        total_runtime_s=total_runtime,
        candidate_rollouts_evaluated=len(history) * settings.sample_count,
        stopped_after_success=(
            success_iteration is not None
            and len(history)
            >= success_iteration + 1 + settings.success_polishing_iterations
        ),
        hard_stop_reached=hard_stop_reached,
    )
