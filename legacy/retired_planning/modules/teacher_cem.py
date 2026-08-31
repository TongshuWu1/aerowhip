"""Consistent local CEM teacher for amortized trajectory optimization.

This module is deliberately narrower than the Milestone 4C campaign runner.
It solves one supplied policy context, starts from one maneuver-family center,
and evaluates a separate post-maneuver hold window.  It never trains a policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import time

import numpy as np
import torch

from learning.policy_context import PolicyContext
from simulator.simulator import CoupledSimulator

from .cem import feasibility_elite_order
from .cem_task import VariableDurationWhipTask
from .metrics import PopulationRolloutResult
from .variable_duration import project_decisions, run_variable_population_rollout


@dataclass(frozen=True, slots=True)
class TeacherCemSettings:
    population: int = 2048
    elite_fraction: float = 0.05
    maximum_iterations: int = 15
    minimum_iterations_after_success: int = 3
    initial_acceleration_std_m_s2: float = 1.0
    initial_duration_std_s: float = 0.025
    acceleration_std_floor_m_s2: float = 0.15
    duration_std_floor_s: float = 0.005
    old_distribution_weight: float = 0.30
    elite_distribution_weight: float = 0.70
    covariance_jitter: float = 1.0e-8
    duration_max_s: float = 1.20
    evaluation_time_s: float = 1.50

    def __post_init__(self) -> None:
        if self.population != 2048:
            raise ValueError("Teacher CEM preserves the canonical 2,048-row numerical shape.")
        if not (0.0 < self.elite_fraction < 1.0):
            raise ValueError("Teacher elite fraction must lie in (0,1).")
        if self.evaluation_time_s < self.duration_max_s:
            raise ValueError("Teacher evaluation must include the full maneuver interval.")


@dataclass(frozen=True, slots=True)
class TeacherContextResult:
    context_id: str
    seed: int
    success: bool
    decision_local: torch.Tensor | None
    metrics: dict[str, object]
    iterations: int
    rollouts: int
    runtime_s: float
    first_success_iteration: int | None
    history: tuple[dict[str, object], ...]


def _cpu_result(result: PopulationRolloutResult) -> PopulationRolloutResult:
    return PopulationRolloutResult(
        **{
            item.name: (
                None
                if getattr(result, item.name) is None
                else getattr(result, item.name).detach().cpu()
            )
            for item in fields(PopulationRolloutResult)
        }
    )


def _regularize(covariance: torch.Tensor, settings: TeacherCemSettings) -> torch.Tensor:
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    covariance = (
        eigenvectors
        * torch.clamp(eigenvalues, min=settings.covariance_jitter)
    ) @ eigenvectors.T
    floors = torch.full(
        (49,), settings.acceleration_std_floor_m_s2**2, dtype=torch.float64
    )
    floors[-1] = settings.duration_std_floor_s**2
    deficit = torch.clamp(floors - torch.diagonal(covariance), min=0.0)
    return covariance + torch.diag(deficit) + settings.covariance_jitter * torch.eye(49, dtype=torch.float64)


def _sample(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    generator: torch.Generator,
    count: int,
) -> torch.Tensor:
    identity = torch.eye(mean.numel(), dtype=mean.dtype)
    jitter = 0.0
    for _ in range(6):
        try:
            factor = torch.linalg.cholesky(covariance + jitter * identity)
            break
        except RuntimeError:
            jitter = 1.0e-8 if jitter == 0.0 else 10.0 * jitter
    else:
        raise RuntimeError("Teacher covariance is not positive definite.")
    noise = torch.randn((count, mean.numel()), dtype=mean.dtype, generator=generator)
    return mean[None] + noise @ factor.T


def _better_success(candidate: dict[str, object], previous: dict[str, object] | None) -> bool:
    if previous is None:
        return True
    return (
        float(candidate["task_cost"]),
        float(candidate["first_entry_tip_distance_m"]),
        float(candidate["first_entry_direction_angle_deg"]),
    ) < (
        float(previous["task_cost"]),
        float(previous["first_entry_tip_distance_m"]),
        float(previous["first_entry_direction_angle_deg"]),
    )


def optimize_teacher_context(
    simulator: CoupledSimulator,
    context: PolicyContext,
    task: VariableDurationWhipTask,
    warm_decision_local: torch.Tensor,
    *,
    context_id: str,
    seed: int,
    settings: TeacherCemSettings,
    artifact_directory: Path,
) -> TeacherContextResult:
    """Optimize one context while remaining in one warm-started maneuver family."""

    if context.batch_size != 1:
        raise ValueError("Teacher CEM accepts exactly one scientific context per solve.")
    warm = project_decisions(
        torch.as_tensor(warm_decision_local, dtype=torch.float64, device="cpu"),
        task,
        settings.duration_max_s,
    )
    mean = warm.clone()
    std = torch.full_like(mean, settings.initial_acceleration_std_m_s2)
    std[-1] = settings.initial_duration_std_s
    covariance = torch.diag(std.square())
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_decision: torch.Tensor | None = None
    best_metrics: dict[str, object] | None = None
    first_success_iteration: int | None = None
    history: list[dict[str, object]] = []
    start = time.perf_counter()
    rotation = context.frame.rotation_world_from_local[0].to(
        device=simulator.device, dtype=simulator.dtype
    )

    artifact_directory.mkdir(parents=True, exist_ok=True)
    for iteration in range(1, settings.maximum_iterations + 1):
        iteration_start = time.perf_counter()
        decisions = _sample(mean, covariance, generator, settings.population)
        decisions[0] = warm
        if best_decision is not None:
            decisions[1] = best_decision
        decisions = project_decisions(decisions, task, settings.duration_max_s)
        knots_local = decisions[:, :-1].reshape(-1, task.cem.knot_count, 3).to(
            device=simulator.device, dtype=simulator.dtype
        )
        knots_world = torch.einsum("ij,bkj->bki", rotation, knots_local)
        durations = decisions[:, -1].to(device=simulator.device, dtype=simulator.dtype)
        result = _cpu_result(
            run_variable_population_rollout(
                simulator,
                context.initial_state_world,
                knots_world,
                durations,
                task,
                maximum_time_s=settings.evaluation_time_s,
                evaluation_time_s=settings.evaluation_time_s,
                command_initial_positions_m=context.command_initial_position_world_m.expand(
                    settings.population, -1
                ),
                command_initial_velocities_m_s=context.command_initial_velocity_world_m_s.expand(
                    settings.population, -1
                ),
                command_yaws_rad=context.command_yaw_world_rad.expand(settings.population),
                target_positions_m=context.target_position_world_m().expand(
                    settings.population, -1
                ),
                desired_directions=context.target_direction_world().expand(
                    settings.population, -1
                ),
                initial_uav_positions_m=context.command_initial_position_world_m.expand(
                    settings.population, -1
                ),
            )
        )
        order = feasibility_elite_order(result, task)
        elite_count = max(1, int(round(settings.population * settings.elite_fraction)))
        elite_indices = torch.from_numpy(order[:elite_count].copy()).to(torch.int64)
        elites = decisions[elite_indices]
        elite_mean = elites.mean(dim=0)
        centered = elites - elite_mean
        elite_covariance = centered.T @ centered / max(elite_count - 1, 1)
        mean = settings.old_distribution_weight * mean + settings.elite_distribution_weight * elite_mean
        covariance = _regularize(
            settings.old_distribution_weight * covariance
            + settings.elite_distribution_weight * elite_covariance,
            settings,
        )
        successful = np.asarray(result.success, dtype=bool)
        success_index = next((int(index) for index in order if successful[index]), None)
        if success_index is not None:
            metrics = result.row(success_index)
            metrics["optimized_duration_s"] = float(decisions[success_index, -1])
            metrics["maneuver_duration_s"] = float(decisions[success_index, -1])
            metrics["evaluation_time_s"] = settings.evaluation_time_s
            metrics["hit_after_maneuver"] = bool(
                float(metrics["first_entry_time_s"]) > float(decisions[success_index, -1]) + 1.0e-7
            )
            if _better_success(metrics, best_metrics):
                best_metrics = metrics
                best_decision = decisions[success_index].clone()
            if first_success_iteration is None:
                first_success_iteration = iteration
        record = {
            "iteration": iteration,
            "successful_candidates": int(result.success.sum()),
            "feasible_candidates": int(result.feasible.sum()),
            "elite_successes": int(result.success[elite_indices].sum()),
            "mean_duration_s": float(mean[-1]),
            "duration_std_s": float(torch.sqrt(covariance[-1, -1])),
            "best_success_tip_error_m": None if best_metrics is None else float(best_metrics["first_entry_tip_distance_m"]),
            "best_success_directed_speed_m_s": None if best_metrics is None else float(best_metrics["first_entry_directed_speed_m_s"]),
            "best_success_direction_error_deg": None if best_metrics is None else float(best_metrics["first_entry_direction_angle_deg"]),
            "iteration_runtime_s": time.perf_counter() - iteration_start,
        }
        history.append(record)
        np.savez_compressed(
            artifact_directory / f"iteration_{iteration:03d}.npz",
            mean=mean.numpy(), covariance=covariance.numpy(),
            rng_state=generator.get_state().numpy(),
            best_decision=np.asarray([]) if best_decision is None else best_decision.numpy(),
        )
        (artifact_directory / "history.json").write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if (
            first_success_iteration is not None
            and iteration >= first_success_iteration + settings.minimum_iterations_after_success - 1
        ):
            break

    metrics = best_metrics or {
        "success": False,
        "failure_reason": "NO_SCIENTIFIC_SUCCESS_IN_TEACHER_BUDGET",
        "evaluation_time_s": settings.evaluation_time_s,
    }
    summary = {
        "context_id": context_id,
        "seed": seed,
        "success": best_decision is not None,
        "iterations": len(history),
        "rollouts": len(history) * settings.population,
        "runtime_s": time.perf_counter() - start,
        "first_success_iteration": first_success_iteration,
        "metrics": metrics,
        "settings": asdict(settings),
    }
    (artifact_directory / "result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return TeacherContextResult(
        context_id=context_id,
        seed=seed,
        success=best_decision is not None,
        decision_local=best_decision,
        metrics=metrics,
        iterations=len(history),
        rollouts=len(history) * settings.population,
        runtime_s=time.perf_counter() - start,
        first_success_iteration=first_success_iteration,
        history=tuple(history),
    )
