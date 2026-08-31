"""Checkpointed feasibility-first CEM for variable-duration cable whipping."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch

from simulator.simulator import CoupledSimulator
from simulator.state import SimulatorState

from .cem_task import VariableDurationWhipTask
from .metrics import PopulationRolloutResult
from .variable_duration import project_decisions, run_variable_population_rollout


@dataclass(frozen=True, slots=True)
class CemIterationRecord:
    seed: int
    iteration: int
    covariance_type: str
    mean_duration_s: float
    duration_std_s: float
    minimum_sampled_duration_s: float
    maximum_sampled_duration_s: float
    feasible_candidate_count: int
    successful_candidate_count: int
    best_feasible_tip_error_m: float | None
    best_successful_tip_error_m: float | None
    best_directed_tip_speed_m_s: float
    best_direction_error_deg: float
    best_event_time_s: float
    best_uav_displacement_m: float
    best_uav_speed_m_s: float
    elite_success_count: int
    elite_feasible_non_success_count: int
    elite_infeasible_count: int
    best_ever_objective: float
    iteration_runtime_s: float
    cumulative_runtime_s: float


@dataclass(frozen=True, slots=True)
class CemSeedResult:
    seed: int
    duration_max_s: float
    iterations: int
    stopped_reason: str
    total_runtime_s: float
    population_rollouts_evaluated: int
    best_success_decision: torch.Tensor | None
    best_success_metrics: dict[str, object] | None
    best_feasible_decision: torch.Tensor | None
    best_feasible_metrics: dict[str, object] | None
    best_overall_decision: torch.Tensor
    best_overall_metrics: dict[str, object]
    final_mean: torch.Tensor
    final_covariance: torch.Tensor
    history: tuple[CemIterationRecord, ...]


def _cpu_result(result: PopulationRolloutResult) -> PopulationRolloutResult:
    return PopulationRolloutResult(
        **{
            item.name: getattr(result, item.name).detach().cpu()
            for item in fields(PopulationRolloutResult)
        }
    )


def _concatenate(results: list[PopulationRolloutResult]) -> PopulationRolloutResult:
    return PopulationRolloutResult(
        **{
            item.name: torch.cat([getattr(result, item.name) for result in results], dim=0)
            for item in fields(PopulationRolloutResult)
        }
    )


def _metric_arrays(result: PopulationRolloutResult) -> dict[str, np.ndarray]:
    names = (
        "task_cost",
        "feasible",
        "feasibility_violation",
        "success",
        "finite",
        "best_event_tip_distance_m",
        "best_event_directed_speed_m_s",
        "best_event_direction_angle_deg",
        "best_event_time_s",
        "first_entry_tip_distance_m",
        "first_entry_directed_speed_m_s",
        "first_entry_direction_angle_deg",
        "first_entry_time_s",
        "maximum_uav_displacement_m",
        "maximum_uav_speed_m_s",
        "maximum_command_acceleration_m_s2",
    )
    return {name: getattr(result, name).numpy() for name in names}


def _reward_ranked_successes(task: VariableDurationWhipTask | None) -> bool:
    objective = None if task is None else task.legacy_run_online_objective
    return bool(
        objective is not None
        and objective.profile == "legacy_run_online_strike_margin_tuned_v4"
    )


def feasibility_elite_order(
    result: PopulationRolloutResult,
    task: VariableDurationWhipTask | None = None,
) -> np.ndarray:
    """Return a global lexicographic order for success/feasible/infeasible rows."""

    data = _metric_arrays(result)
    success = data["success"].astype(bool)
    feasible = data["feasible"].astype(bool)
    finite = data["finite"].astype(bool)
    category = np.where(success, 0, np.where(feasible, 1, 2)).astype(np.int64)
    category = np.where(finite, category, 3)

    tip = np.where(success, data["first_entry_tip_distance_m"], data["best_event_tip_distance_m"])
    direction = np.where(
        success,
        data["first_entry_direction_angle_deg"],
        data["best_event_direction_angle_deg"],
    )
    task_cost = np.nan_to_num(data["task_cost"], nan=1e30, posinf=1e30, neginf=-1e30)
    violation = np.nan_to_num(data["feasibility_violation"], nan=1e30, posinf=1e30)
    slack = np.minimum(
        1.0 - data["maximum_uav_displacement_m"] / 0.5,
        1.0 - data["maximum_uav_speed_m_s"] / 3.0,
    )
    if _reward_ranked_successes(task):
        # Reward tuning must continue to matter after the first valid strike.
        # The task cost contains the configured center-hit and interior-angle
        # shaping terms; hard feasibility remains the primary category.
        key1 = np.where(success, task_cost, np.where(feasible, task_cost, violation))
        key2 = np.where(success, tip, np.where(feasible, tip, task_cost))
        key3 = np.where(success, direction, direction)
        key4 = np.where(success, -slack, np.zeros_like(slack))
        return np.lexsort((key4, key3, key2, key1, category))
    key1 = np.where(success, tip, np.where(feasible, task_cost, violation))
    key2 = np.where(success, direction, np.where(feasible, tip, task_cost))
    key3 = np.where(success, -slack, direction)
    return np.lexsort((key3, key2, key1, category))


def _best_indices(
    result: PopulationRolloutResult,
    task: VariableDurationWhipTask | None = None,
) -> tuple[int | None, int | None, int]:
    order = feasibility_elite_order(result, task)
    success = result.success.numpy().astype(bool)
    feasible = result.feasible.numpy().astype(bool)
    finite = result.finite.numpy().astype(bool)
    best_success = next((int(index) for index in order if success[index]), None)
    best_feasible = next((int(index) for index in order if feasible[index]), None)
    task = result.task_cost.numpy().copy()
    task[~finite] = np.inf
    best_overall = int(np.argmin(task))
    return best_success, best_feasible, best_overall


def _success_key(
    metrics: dict[str, object],
    task: VariableDurationWhipTask | None = None,
) -> tuple[float, ...]:
    margin = min(
        0.5 - float(metrics["maximum_uav_displacement_m"]),
        (3.0 - float(metrics["maximum_uav_speed_m_s"])) / 3.0,
    )
    physical_key = (
        float(metrics["first_entry_tip_distance_m"]),
        float(metrics["first_entry_direction_angle_deg"]),
        -margin,
        float(metrics.get("optimized_duration_s", math.inf)),
    )
    if _reward_ranked_successes(task):
        return (float(metrics["task_cost"]),) + physical_key
    return physical_key


def _feasible_key(metrics: dict[str, object]) -> tuple[float, float]:
    return (float(metrics["task_cost"]), float(metrics["best_event_tip_distance_m"]))


def _overall_key(metrics: dict[str, object]) -> float:
    return float(metrics["task_cost"])


def _regularize_covariance(
    covariance: torch.Tensor,
    task: VariableDurationWhipTask,
) -> torch.Tensor:
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    covariance = (eigenvectors * torch.clamp(eigenvalues, min=task.cem.covariance_jitter)) @ eigenvectors.T
    floors = torch.full(
        (covariance.shape[0],),
        task.cem.acceleration_std_floor_m_s2**2,
        dtype=covariance.dtype,
    )
    floors[-1] = task.cem.duration_std_floor_s**2
    diagonal = torch.diagonal(covariance)
    covariance = covariance + torch.diag(torch.clamp(floors - diagonal, min=0.0))
    return covariance + torch.eye(covariance.shape[0], dtype=covariance.dtype) * task.cem.covariance_jitter


def initial_distribution(
    task: VariableDurationWhipTask,
) -> tuple[torch.Tensor, torch.Tensor]:
    nominal = task.nominal_knots(device=torch.device("cpu"), dtype=torch.float64).reshape(-1)
    mean = torch.cat((nominal, torch.tensor([task.cem.initial_duration_mean_s], dtype=torch.float64)))
    std = torch.full_like(mean, task.cem.initial_acceleration_std_m_s2)
    std[-1] = task.cem.initial_duration_std_s
    return mean, torch.diag(std.square())


def save_cem_checkpoint(
    checkpoint_directory: Path,
    *,
    seed: int,
    iteration: int,
    mean: torch.Tensor,
    covariance: torch.Tensor,
    generator: torch.Generator,
    global_best_decision: torch.Tensor | None,
    global_best_metrics: dict[str, object] | None,
    duration_max_s: float,
    task: VariableDurationWhipTask,
    record: CemIterationRecord,
) -> tuple[Path, Path]:
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    stem = checkpoint_directory / f"iteration_{iteration:03d}"
    npz = stem.with_suffix(".npz")
    metadata = stem.with_suffix(".json")
    np.savez_compressed(
        npz,
        mean=mean.numpy(),
        covariance=covariance.numpy(),
        rng_state=generator.get_state().numpy(),
        global_best_decision=(
            np.asarray([], dtype=np.float64)
            if global_best_decision is None
            else global_best_decision.numpy()
        ),
    )
    payload = {
        "schema": "variable_duration_cem_checkpoint_v1",
        "seed": seed,
        "completed_iteration": iteration,
        "next_iteration": iteration + 1,
        "duration_bounds_s": [task.cem.duration_min_s, duration_max_s],
        "model_freeze": task.model_freeze,
        "task_id": task.task_id,
        "task_config_sha256": hashlib.sha256(task.source_path.read_bytes()).hexdigest(),
        "cem_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "global_best_metrics": global_best_metrics,
        "iteration_record": asdict(record),
        "npz": npz.name,
    }
    metadata.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return npz, metadata


def load_cem_checkpoint(path: str | Path) -> dict[str, object]:
    metadata_path = Path(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(metadata_path.with_name(metadata["npz"]), allow_pickle=False) as archive:
        return {
            "metadata": metadata,
            "mean": torch.from_numpy(np.asarray(archive["mean"]).copy()),
            "covariance": torch.from_numpy(np.asarray(archive["covariance"]).copy()),
            "rng_state": torch.from_numpy(np.asarray(archive["rng_state"]).copy()),
            "global_best_decision": torch.from_numpy(
                np.asarray(archive["global_best_decision"]).copy()
            ),
        }


def _sample_population(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    generator: torch.Generator,
    population: int,
) -> torch.Tensor:
    jitter = 0.0
    identity = torch.eye(mean.numel(), dtype=mean.dtype)
    for _ in range(6):
        try:
            factor = torch.linalg.cholesky(covariance + jitter * identity)
            break
        except RuntimeError:
            jitter = 1e-8 if jitter == 0.0 else jitter * 10.0
    else:
        raise RuntimeError("CEM covariance was not positive definite after jitter retries.")
    noise = torch.randn((population, mean.numel()), dtype=mean.dtype, generator=generator)
    return mean[None] + noise @ factor.T


def optimize_variable_duration_cem(
    simulator: CoupledSimulator,
    initial_state: SimulatorState,
    task: VariableDurationWhipTask,
    *,
    seed: int,
    duration_max_s: float,
    seed_directory: Path,
    campaign_deadline: float,
    iteration_callback: Callable[[CemIterationRecord], None] | None = None,
) -> CemSeedResult:
    """Run one fixed-configuration seed with global 8192-row elite updates."""

    config = task.cem
    mean, covariance = initial_distribution(task)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    history: list[CemIterationRecord] = []
    best_success_decision: torch.Tensor | None = None
    best_success_metrics: dict[str, object] | None = None
    best_feasible_decision: torch.Tensor | None = None
    best_feasible_metrics: dict[str, object] | None = None
    best_overall_decision: torch.Tensor | None = None
    best_overall_metrics: dict[str, object] | None = None
    strong_stale = 0
    previous_strong_error = math.inf
    run_start = time.perf_counter()
    stopped_reason = "maximum_iterations"

    for iteration in range(1, config.maximum_iterations + 1):
        if time.perf_counter() >= campaign_deadline:
            stopped_reason = "campaign_hard_stop"
            break
        iteration_start = time.perf_counter()
        decisions = _sample_population(mean, covariance, generator, config.population)
        decisions[0] = mean
        if best_success_decision is not None:
            decisions[1] = best_success_decision
        elif best_feasible_decision is not None:
            decisions[1] = best_feasible_decision
        decisions = project_decisions(decisions, task, duration_max_s)

        chunks: list[PopulationRolloutResult] = []
        for start in range(0, config.population, config.logical_chunk_size):
            chunk = decisions[start : start + config.logical_chunk_size]
            knots = chunk[:, :-1].reshape(-1, config.knot_count, 3).to(
                device=simulator.device, dtype=simulator.dtype
            )
            durations = chunk[:, -1].to(device=simulator.device, dtype=simulator.dtype)
            chunks.append(
                _cpu_result(
                    run_variable_population_rollout(
                        simulator,
                        initial_state,
                        knots,
                        durations,
                        task,
                        maximum_time_s=duration_max_s,
                    )
                )
            )
        if simulator.device.type == "cuda":
            torch.cuda.synchronize(simulator.device)
        result = _concatenate(chunks)
        order = feasibility_elite_order(result, task)
        elite_count = max(1, int(round(config.elite_fraction * config.population)))
        elite_indices = torch.from_numpy(order[:elite_count].copy()).to(torch.int64)
        elites = decisions[elite_indices]
        elite_mean = elites.mean(dim=0)
        centered = elites - elite_mean
        elite_covariance = centered.T @ centered / max(elite_count - 1, 1)
        mean = config.old_distribution_weight * mean + config.elite_distribution_weight * elite_mean
        covariance = (
            config.old_distribution_weight * covariance
            + config.elite_distribution_weight * elite_covariance
        )
        covariance = _regularize_covariance(covariance, task)

        success_index, feasible_index, overall_index = _best_indices(result, task)
        for index, kind in (
            (success_index, "success"),
            (feasible_index, "feasible"),
            (overall_index, "overall"),
        ):
            if index is None:
                continue
            metrics = result.row(index)
            metrics["optimized_duration_s"] = float(decisions[index, -1])
            decision = decisions[index].clone()
            if kind == "success" and (
                best_success_metrics is None
                or _success_key(metrics, task) < _success_key(best_success_metrics, task)
            ):
                best_success_metrics, best_success_decision = metrics, decision
            elif kind == "feasible" and (
                best_feasible_metrics is None
                or _feasible_key(metrics) < _feasible_key(best_feasible_metrics)
            ):
                best_feasible_metrics, best_feasible_decision = metrics, decision
            elif kind == "overall" and (
                best_overall_metrics is None
                or _overall_key(metrics) < _overall_key(best_overall_metrics)
            ):
                best_overall_metrics, best_overall_decision = metrics, decision

        elite_success = int(result.success[elite_indices].sum())
        elite_feasible = int(result.feasible[elite_indices].sum())
        selected = best_success_metrics or best_feasible_metrics or best_overall_metrics
        assert selected is not None
        record = CemIterationRecord(
            seed=seed,
            iteration=iteration,
            covariance_type=config.covariance,
            mean_duration_s=float(mean[-1]),
            duration_std_s=float(torch.sqrt(covariance[-1, -1])),
            minimum_sampled_duration_s=float(decisions[:, -1].min()),
            maximum_sampled_duration_s=float(decisions[:, -1].max()),
            feasible_candidate_count=int(result.feasible.sum()),
            successful_candidate_count=int(result.success.sum()),
            best_feasible_tip_error_m=(
                None if feasible_index is None else float(result.best_event_tip_distance_m[feasible_index])
            ),
            best_successful_tip_error_m=(
                None if success_index is None else float(result.first_entry_tip_distance_m[success_index])
            ),
            best_directed_tip_speed_m_s=float(
                selected["first_entry_directed_speed_m_s"]
                if bool(selected["success"])
                else selected["best_event_directed_speed_m_s"]
            ),
            best_direction_error_deg=float(
                selected["first_entry_direction_angle_deg"]
                if bool(selected["success"])
                else selected["best_event_direction_angle_deg"]
            ),
            best_event_time_s=float(
                selected["first_entry_time_s"]
                if bool(selected["success"])
                else selected["best_event_time_s"]
            ),
            best_uav_displacement_m=float(selected["maximum_uav_displacement_m"]),
            best_uav_speed_m_s=float(selected["maximum_uav_speed_m_s"]),
            elite_success_count=elite_success,
            elite_feasible_non_success_count=elite_feasible - elite_success,
            elite_infeasible_count=elite_count - elite_feasible,
            best_ever_objective=float(selected["task_cost"]),
            iteration_runtime_s=time.perf_counter() - iteration_start,
            cumulative_runtime_s=time.perf_counter() - run_start,
        )
        history.append(record)
        seed_directory.mkdir(parents=True, exist_ok=True)
        (seed_directory / "cem_iteration_history.json").write_text(
            json.dumps([asdict(item) for item in history], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checkpoint_best = (
            best_success_decision
            if best_success_decision is not None
            else best_feasible_decision
            if best_feasible_decision is not None
            else best_overall_decision
        )
        checkpoint_metrics = (
            best_success_metrics
            if best_success_metrics is not None
            else best_feasible_metrics
            if best_feasible_metrics is not None
            else best_overall_metrics
        )
        save_cem_checkpoint(
            seed_directory / "cem_checkpoints",
            seed=seed,
            iteration=iteration,
            mean=mean,
            covariance=covariance,
            generator=generator,
            global_best_decision=checkpoint_best,
            global_best_metrics=checkpoint_metrics,
            duration_max_s=duration_max_s,
            task=task,
            record=record,
        )
        if iteration_callback is not None:
            iteration_callback(record)

        if best_success_metrics is not None:
            strong = (
                float(best_success_metrics["first_entry_tip_distance_m"]) <= config.strong_tip_error_m
                and float(best_success_metrics["first_entry_directed_speed_m_s"]) >= config.strong_directed_speed_m_s
                and float(best_success_metrics["first_entry_direction_angle_deg"]) <= config.strong_direction_error_deg
                and float(best_success_metrics["maximum_uav_displacement_m"]) <= config.strong_uav_displacement_m
                and float(best_success_metrics["maximum_uav_speed_m_s"]) <= config.strong_uav_speed_m_s
            )
            error = float(best_success_metrics["first_entry_tip_distance_m"])
            if strong:
                if previous_strong_error - error >= 0.0005:
                    strong_stale = 0
                    previous_strong_error = error
                else:
                    strong_stale += 1
                if strong_stale >= config.strong_stale_iterations:
                    stopped_reason = "strong_solution_stable"
                    break

    if best_overall_decision is None or best_overall_metrics is None:
        raise RuntimeError("CEM completed without any finite population result.")
    return CemSeedResult(
        seed=seed,
        duration_max_s=duration_max_s,
        iterations=len(history),
        stopped_reason=stopped_reason,
        total_runtime_s=time.perf_counter() - run_start,
        population_rollouts_evaluated=len(history) * config.population,
        best_success_decision=best_success_decision,
        best_success_metrics=best_success_metrics,
        best_feasible_decision=best_feasible_decision,
        best_feasible_metrics=best_feasible_metrics,
        best_overall_decision=best_overall_decision,
        best_overall_metrics=best_overall_metrics,
        final_mean=mean,
        final_covariance=covariance,
        history=tuple(history),
    )


def is_marginal_success(metrics: dict[str, object] | None) -> bool:
    if metrics is None:
        return True
    return bool(
        float(metrics["first_entry_tip_distance_m"]) > 0.030
        or float(metrics["first_entry_direction_angle_deg"]) > 25.0
        or float(metrics["maximum_uav_displacement_m"]) > 0.48
        or float(metrics["maximum_uav_speed_m_s"]) > 2.9
    )


def select_final_seed(
    results: list[CemSeedResult],
    task: VariableDurationWhipTask | None = None,
) -> tuple[CemSeedResult, torch.Tensor, dict[str, object], str]:
    successes = [item for item in results if item.best_success_decision is not None]
    if successes:
        selected = min(
            successes,
            key=lambda item: _success_key(item.best_success_metrics or {}, task),
        )
        assert selected.best_success_decision is not None and selected.best_success_metrics is not None
        return selected, selected.best_success_decision, selected.best_success_metrics, "successful_feasible"
    feasible = [item for item in results if item.best_feasible_decision is not None]
    if feasible:
        selected = min(feasible, key=lambda item: _feasible_key(item.best_feasible_metrics or {}))
        assert selected.best_feasible_decision is not None and selected.best_feasible_metrics is not None
        return selected, selected.best_feasible_decision, selected.best_feasible_metrics, "best_feasible_near_miss"
    selected = min(results, key=lambda item: _overall_key(item.best_overall_metrics))
    return selected, selected.best_overall_decision, selected.best_overall_metrics, "best_overall_infeasible"
