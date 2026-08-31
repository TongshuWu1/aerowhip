"""Production-normalized CEM and fixed-2048 authoritative evaluation.

The only optimized/saved/deployed action is a normalized 49-D complete action.
Every physics path decodes it through ``decode_policy_action``.  Logical batches
smaller than 2048 are padded by cyclic repetition before *all* UAV and DDER
physics, then sliced back to their logical size.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from learning.policy_action import decode_policy_action, encode_physical_action
from learning.policy_context import PolicyContext
from simulator.simulator import CoupledSimulator

from .cem_task import VariableDurationWhipTask
from .metrics import PopulationRolloutResult
from .rollout import clone_state_batch
from .selection import feasibility_elite_order
from .variable_duration import (
    VariableWhipAccumulator,
    run_variable_population_rollout,
    variable_duration_fullstate,
)


FIXED_NUMERICAL_BATCH_SIZE = 2048


@dataclass(frozen=True, slots=True)
class ProductionCemSettings:
    population: int = 4096
    elite_fraction: float = 0.05
    maximum_iterations: int = 20
    minimum_iterations: int = 4
    polish_iterations_after_success: int = 2
    authoritative_top_n: int = 32
    initial_acceleration_std_m_s2: float = 1.0
    initial_duration_std_s: float = 0.05
    acceleration_std_floor_m_s2: float = 0.15
    duration_std_floor_s: float = 0.005
    old_distribution_weight: float = 0.30
    elite_distribution_weight: float = 0.70
    covariance_jitter: float = 1.0e-8
    duration_min_s: float = 0.45
    duration_max_s: float = 1.80
    settle_duration_s: float = 0.30
    evaluation_time_s: float = 2.40

    def __post_init__(self) -> None:
        if self.population < FIXED_NUMERICAL_BATCH_SIZE or self.population % FIXED_NUMERICAL_BATCH_SIZE:
            raise ValueError("Production CEM population must be a multiple of 2048.")
        if not (0.0 < self.elite_fraction < 1.0):
            raise ValueError("Elite fraction must lie in (0,1).")
        if self.duration_min_s >= self.duration_max_s:
            raise ValueError("Invalid maneuver-duration range.")
        if self.evaluation_time_s < self.duration_max_s + self.settle_duration_s:
            raise ValueError("Evaluation must include maneuver plus complete settle.")


@dataclass(frozen=True, slots=True)
class ProductionCemResult:
    context_id: str
    seed: int
    success: bool
    normalized_action: torch.Tensor
    authoritative_metrics: dict[str, Any]
    population_metrics: dict[str, Any]
    iterations: int
    population_rollouts: int
    authoritative_rollouts: int
    runtime_s: float
    first_authoritative_success_iteration: int | None
    history: tuple[dict[str, Any], ...]
    authoritative_top_actions: torch.Tensor
    authoritative_top_metrics: tuple[dict[str, Any], ...]
    scorer_actions: torch.Tensor
    scorer_metrics: tuple[dict[str, Any], ...]
    scorer_provenance: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class FixedBatchTrajectory:
    times_s: torch.Tensor
    command_positions_m: torch.Tensor
    command_velocities_m_s: torch.Tensor
    command_accelerations_m_s2: torch.Tensor
    uav_positions_m: torch.Tensor
    uav_velocities_m_s: torch.Tensor
    cable_positions_m: torch.Tensor
    cable_velocities_m_s: torch.Tensor
    metrics: PopulationRolloutResult


def _slice_result(result: PopulationRolloutResult, count: int) -> PopulationRolloutResult:
    values: dict[str, Any] = {}
    for item in fields(PopulationRolloutResult):
        value = getattr(result, item.name)
        if value is None:
            values[item.name] = None
        elif item.name == "rl_reward_components":
            values[item.name] = type(value)(
                **{field.name: getattr(value, field.name)[:count] for field in fields(value)}
            )
        else:
            values[item.name] = value[:count]
    return PopulationRolloutResult(**values)


def _index_result(
    result: PopulationRolloutResult, indices: torch.Tensor
) -> PopulationRolloutResult:
    """Select arbitrary rows while preserving the complete rollout-result schema."""

    index = torch.as_tensor(indices, dtype=torch.int64, device="cpu").reshape(-1)
    values: dict[str, Any] = {}
    for item in fields(PopulationRolloutResult):
        value = getattr(result, item.name)
        if value is None:
            values[item.name] = None
        elif item.name == "rl_reward_components":
            values[item.name] = type(value)(
                **{field.name: getattr(value, field.name)[index] for field in fields(value)}
            )
        else:
            values[item.name] = value[index]
    return PopulationRolloutResult(**values)


def _cpu_result(result: PopulationRolloutResult) -> PopulationRolloutResult:
    values: dict[str, Any] = {}
    for item in fields(PopulationRolloutResult):
        value = getattr(result, item.name)
        if value is None:
            values[item.name] = None
        elif item.name == "rl_reward_components":
            values[item.name] = type(value)(
                **{
                    field.name: getattr(value, field.name).detach().cpu()
                    for field in fields(value)
                }
            )
        else:
            values[item.name] = value.detach().cpu()
    return PopulationRolloutResult(**values)


def _concatenate(results: list[PopulationRolloutResult]) -> PopulationRolloutResult:
    values: dict[str, Any] = {}
    for item in fields(PopulationRolloutResult):
        members = [getattr(result, item.name) for result in results]
        if members[0] is None:
            values[item.name] = None
        elif item.name == "rl_reward_components":
            values[item.name] = type(members[0])(
                **{
                    field.name: torch.cat([getattr(value, field.name) for value in members])
                    for field in fields(members[0])
                }
            )
        else:
            values[item.name] = torch.cat(members)
    return PopulationRolloutResult(**values)


def _pad_actions(actions: torch.Tensor, size: int = FIXED_NUMERICAL_BATCH_SIZE) -> torch.Tensor:
    value = torch.as_tensor(actions)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape[1] != 49 or value.shape[0] < 1 or value.shape[0] > size:
        raise ValueError("Logical normalized actions must have shape Nx49 with 1<=N<=2048.")
    if value.shape[0] == size:
        return value
    source = torch.arange(size - value.shape[0], device=value.device) % value.shape[0]
    return torch.cat((value, value[source]), dim=0)


def evaluate_normalized_actions_fixed_batch(
    simulator: CoupledSimulator,
    repeated_context: PolicyContext,
    normalized_actions: torch.Tensor,
    task: VariableDurationWhipTask,
    settings: ProductionCemSettings,
) -> PopulationRolloutResult:
    """Evaluate logical actions after padding all coupled physics to 2048 rows."""

    logical = torch.as_tensor(normalized_actions)
    if logical.ndim == 1:
        logical = logical.unsqueeze(0)
    logical_count = int(logical.shape[0])
    if repeated_context.batch_size != FIXED_NUMERICAL_BATCH_SIZE:
        raise ValueError("The production context must already contain exactly 2048 repeats.")
    padded = _pad_actions(logical.to(device=simulator.device, dtype=simulator.dtype))
    decoded = decode_policy_action(padded, task, duration_max_s=settings.duration_max_s)
    knots_world = repeated_context.frame.vectors_to_world(
        decoded.acceleration_knots_local_m_s2
    )
    result = run_variable_population_rollout(
        simulator,
        repeated_context.initial_state_world,
        knots_world,
        decoded.duration_s,
        task,
        maximum_time_s=settings.evaluation_time_s,
        evaluation_time_s=settings.evaluation_time_s,
        command_initial_positions_m=repeated_context.command_initial_position_world_m,
        command_initial_velocities_m_s=repeated_context.command_initial_velocity_world_m_s,
        command_yaws_rad=repeated_context.command_yaw_world_rad,
        target_positions_m=repeated_context.target_position_world_m(),
        desired_directions=repeated_context.target_direction_world(),
        initial_uav_positions_m=repeated_context.command_initial_position_world_m,
    )
    return _slice_result(result, logical_count)


def record_normalized_actions_fixed_batch(
    simulator: CoupledSimulator,
    repeated_context: PolicyContext,
    normalized_actions: torch.Tensor,
    task: VariableDurationWhipTask,
    settings: ProductionCemSettings,
    *,
    record_count: int | None = None,
) -> FixedBatchTrajectory:
    """Run fixed-2048 physics while retaining only requested logical rows."""

    logical = torch.as_tensor(normalized_actions)
    if logical.ndim == 1:
        logical = logical.unsqueeze(0)
    logical_count = int(logical.shape[0])
    keep = logical_count if record_count is None else int(record_count)
    if keep < 1 or keep > logical_count:
        raise ValueError("record_count must select at least one logical row.")
    if repeated_context.batch_size != FIXED_NUMERICAL_BATCH_SIZE:
        raise ValueError("Trajectory recording requires the fixed 2048-row context.")
    padded = _pad_actions(logical.to(device=simulator.device, dtype=simulator.dtype))
    decoded = decode_policy_action(padded, task, duration_max_s=settings.duration_max_s)
    knots_world = repeated_context.frame.vectors_to_world(
        decoded.acceleration_knots_local_m_s2
    )
    command = variable_duration_fullstate(
        knots_world,
        decoded.duration_s,
        initial_position_m=repeated_context.command_initial_position_world_m,
        initial_velocity_m_s=repeated_context.command_initial_velocity_world_m_s,
        yaw_rad=repeated_context.command_yaw_world_rad,
        maximum_time_s=settings.evaluation_time_s,
        dt_s=simulator.dt_s,
        settle_duration_s=settings.settle_duration_s,
    )
    state = clone_state_batch(repeated_context.initial_state_world, FIXED_NUMERICAL_BATCH_SIZE)
    evaluation_durations = torch.full_like(decoded.duration_s, settings.evaluation_time_s)
    accumulator = VariableWhipAccumulator(
        task,
        decoded.duration_s,
        knots_world,
        evaluation_durations_s=evaluation_durations,
        target_positions_m=repeated_context.target_position_world_m(),
        desired_directions=repeated_context.target_direction_world(),
        initial_uav_positions_m=repeated_context.command_initial_position_world_m,
    )
    accumulator.observe(
        0.0,
        uav_position_m=state.uav.position_m,
        uav_velocity_m_s=state.uav.velocity_m_s,
        cable_positions_m=state.cable.positions_m,
        cable_velocities_m_s=state.cable.velocities_m_s,
    )
    # Retain selected rows on the GPU and transfer once after propagation.
    # Per-step host copies would synchronize CUDA hundreds of times and make
    # the 256-action population/batch-one diagnostic unnecessarily slow.
    uav_position = [state.uav.position_m[:keep].detach().clone()]
    uav_velocity = [state.uav.velocity_m_s[:keep].detach().clone()]
    cable_position = [state.cable.positions_m[:keep].detach().clone()]
    cable_velocity = [state.cable.velocities_m_s[:keep].detach().clone()]
    sequence = command.simulator_sequence()
    with torch.no_grad():
        for index in range(sequence.step_count):
            state = simulator._propagate(  # noqa: SLF001
                state, sequence.command_at(index), simulator.parameters, create_graph=False
            )
            accumulator.observe(
                (index + 1) * simulator.dt_s,
                uav_position_m=state.uav.position_m,
                uav_velocity_m_s=state.uav.velocity_m_s,
                cable_positions_m=state.cable.positions_m,
                cable_velocities_m_s=state.cable.velocities_m_s,
            )
            uav_position.append(state.uav.position_m[:keep].detach().clone())
            uav_velocity.append(state.uav.velocity_m_s[:keep].detach().clone())
            cable_position.append(state.cable.positions_m[:keep].detach().clone())
            cable_velocity.append(state.cable.velocities_m_s[:keep].detach().clone())
    metrics = _cpu_result(_slice_result(accumulator.finalize(command), logical_count))
    return FixedBatchTrajectory(
        times_s=command.times_s.detach().cpu(),
        command_positions_m=command.positions_m[:, :keep].detach().cpu(),
        command_velocities_m_s=command.velocities_m_s[:, :keep].detach().cpu(),
        command_accelerations_m_s2=command.accelerations_m_s2[:, :keep].detach().cpu(),
        uav_positions_m=torch.stack(uav_position).cpu(),
        uav_velocities_m_s=torch.stack(uav_velocity).cpu(),
        cable_positions_m=torch.stack(cable_position).cpu(),
        cable_velocities_m_s=torch.stack(cable_velocity).cpu(),
        metrics=metrics,
    )


def event_segment(hit_time_s: float | None, maneuver_duration_s: float, settle_duration_s: float) -> str | None:
    if hit_time_s is None or not math.isfinite(hit_time_s):
        return None
    if hit_time_s <= maneuver_duration_s + 1.0e-7:
        return "ACTIVE"
    if hit_time_s <= maneuver_duration_s + settle_duration_s + 1.0e-7:
        return "SETTLE"
    return "HOLD"


def _project_normalized(actions: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(actions).clone()
    one_row = value.ndim == 1
    if one_row:
        value = value.unsqueeze(0)
    knots = value[:, :48].reshape(-1, 16, 3)
    norms = torch.linalg.vector_norm(knots, dim=-1, keepdim=True)
    knots = knots * torch.clamp(1.0 / torch.clamp(norms, min=1.0e-12), max=1.0)
    result = torch.cat((knots.reshape(value.shape[0], 48), value[:, 48:49].clamp(-1.0, 1.0)), dim=-1)
    return result[0] if one_row else result


def _normalized_std(settings: ProductionCemSettings) -> torch.Tensor:
    value = torch.full((49,), settings.initial_acceleration_std_m_s2 / 20.0, dtype=torch.float64)
    value[-1] = 2.0 * settings.initial_duration_std_s / (
        settings.duration_max_s - settings.duration_min_s
    )
    return value


def _normalized_std_floor(settings: ProductionCemSettings) -> torch.Tensor:
    value = torch.full((49,), settings.acceleration_std_floor_m_s2 / 20.0, dtype=torch.float64)
    value[-1] = 2.0 * settings.duration_std_floor_s / (
        settings.duration_max_s - settings.duration_min_s
    )
    return value


def _regularize(covariance: torch.Tensor, settings: ProductionCemSettings) -> torch.Tensor:
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    covariance = (
        eigenvectors * torch.clamp(eigenvalues, min=settings.covariance_jitter)
    ) @ eigenvectors.T
    floors = _normalized_std_floor(settings).square()
    covariance += torch.diag(torch.clamp(floors - torch.diagonal(covariance), min=0.0))
    return covariance + settings.covariance_jitter * torch.eye(49, dtype=torch.float64)


def _sample(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    generator: torch.Generator,
    count: int,
) -> torch.Tensor:
    identity = torch.eye(49, dtype=torch.float64)
    jitter = 0.0
    for _ in range(7):
        try:
            factor = torch.linalg.cholesky(covariance + jitter * identity)
            break
        except RuntimeError:
            jitter = 1.0e-9 if jitter == 0.0 else 10.0 * jitter
    else:
        raise RuntimeError("Normalized CEM covariance is not positive definite.")
    noise = torch.randn((count, 49), generator=generator, dtype=torch.float64)
    return _project_normalized(mean[None] + noise @ factor.T)


def _selection_key(metrics: dict[str, Any]) -> tuple[float, ...]:
    if bool(metrics["success"]):
        return (
            0.0,
            float(metrics["task_cost"]),
            float(metrics["first_entry_tip_distance_m"]),
            float(metrics["first_entry_direction_angle_deg"]),
        )
    if bool(metrics["feasible"]):
        return (1.0, float(metrics["task_cost"]), float(metrics["best_event_tip_distance_m"]), 0.0)
    return (2.0, float(metrics["feasibility_violation"]), float(metrics["task_cost"]), 0.0)


def _deterministic_scorer_indices(
    population: PopulationRolloutResult,
    *,
    excluded_indices: torch.Tensor,
    maximum_count: int,
    seed: int,
) -> tuple[list[int], list[str]]:
    """Choose broad final-population outcomes without retaining all 4,096 rows."""

    count = int(population.success.shape[0])
    excluded = {int(value) for value in excluded_indices.tolist()}
    available = [index for index in range(count) if index not in excluded]
    success = np.asarray(population.success, dtype=bool)
    feasible = np.asarray(population.feasible, dtype=bool)
    task_cost = np.asarray(population.task_cost, dtype=np.float64)
    violation = np.asarray(population.feasibility_violation, dtype=np.float64)
    rng = np.random.default_rng(seed)
    buckets: list[tuple[str, list[int]]] = [
        ("population_success", sorted((i for i in available if success[i]), key=lambda i: task_cost[i])),
        (
            "feasible_near_miss",
            sorted((i for i in available if feasible[i] and not success[i]), key=lambda i: task_cost[i]),
        ),
        (
            "unsafe_near_miss",
            sorted((i for i in available if not feasible[i]), key=lambda i: (violation[i], task_cost[i])),
        ),
        ("low_reward", sorted(available, key=lambda i: task_cost[i], reverse=True)),
        ("uniform_random", rng.permutation(np.asarray(available, dtype=np.int64)).tolist()),
    ]
    selected: list[int] = []
    categories: list[str] = []
    used: set[int] = set()
    quota = max(1, maximum_count // len(buckets))
    for name, indices in buckets:
        taken = 0
        for index in indices:
            if index in used:
                continue
            selected.append(index)
            categories.append(name)
            used.add(index)
            taken += 1
            if taken >= quota or len(selected) >= maximum_count:
                break
    if len(selected) < maximum_count:
        for index in rng.permutation(np.asarray(available, dtype=np.int64)).tolist():
            if index in used:
                continue
            selected.append(int(index))
            categories.append("uniform_fill")
            used.add(int(index))
            if len(selected) >= maximum_count:
                break
    return selected, categories


def _metrics_for_actions(
    result: PopulationRolloutResult,
    actions: torch.Tensor,
    task: VariableDurationWhipTask,
    settings: ProductionCemSettings,
) -> tuple[dict[str, Any], ...]:
    durations = decode_policy_action(
        actions.float(), task, duration_max_s=settings.duration_max_s
    ).duration_s.detach().cpu()
    rows: list[dict[str, Any]] = []
    for index in range(actions.shape[0]):
        metrics = result.row(index)
        duration = float(durations[index])
        metrics["maneuver_duration_s"] = duration
        metrics["hit_segment"] = event_segment(
            metrics["first_entry_time_s"], duration, settings.settle_duration_s
        )
        rows.append(metrics)
    return tuple(rows)


def canonical_family_action_for_context(
    historical_knots_world_m_s2: torch.Tensor,
    historical_duration_s: float,
    repeated_context: PolicyContext,
    task: VariableDurationWhipTask,
    settings: ProductionCemSettings,
) -> torch.Tensor:
    """Rotate the canonical +X maneuver family toward the local target direction."""

    knots = torch.as_tensor(historical_knots_world_m_s2, dtype=torch.float64).reshape(16, 3)
    direction = repeated_context.target_direction_local[0].detach().cpu().double()
    angle = torch.atan2(direction[1], direction[0])
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotation = torch.tensor(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    local = torch.einsum("ij,kj->ki", rotation, knots)
    return encode_physical_action(
        local,
        historical_duration_s,
        task,
        duration_max_s=settings.duration_max_s,
    )[0].double()


def optimize_production_cem(
    simulator: CoupledSimulator,
    repeated_context: PolicyContext,
    task: VariableDurationWhipTask,
    initial_normalized_action: torch.Tensor,
    *,
    context_id: str,
    seed: int,
    settings: ProductionCemSettings,
    checkpoint_directory: Path,
) -> ProductionCemResult:
    """Optimize one context and select only from authoritative top-32 replay."""

    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    mean = _project_normalized(torch.as_tensor(initial_normalized_action, dtype=torch.float64)).reshape(49)
    covariance = torch.diag(_normalized_std(settings).square())
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history: list[dict[str, Any]] = []
    best_action = mean.clone()
    best_authoritative: dict[str, Any] | None = None
    best_population: dict[str, Any] | None = None
    first_success: int | None = None
    final_actions: torch.Tensor | None = None
    final_population: PopulationRolloutResult | None = None
    population_rollouts = 0
    authoritative_rollouts = 0
    started = time.perf_counter()

    for iteration in range(1, settings.maximum_iterations + 1):
        iteration_start = time.perf_counter()
        actions = _sample(mean, covariance, generator, settings.population)
        actions[0] = mean
        actions[1] = best_action
        chunks: list[PopulationRolloutResult] = []
        for start in range(0, settings.population, FIXED_NUMERICAL_BATCH_SIZE):
            chunks.append(
                _cpu_result(
                    evaluate_normalized_actions_fixed_batch(
                        simulator,
                        repeated_context,
                        actions[start : start + FIXED_NUMERICAL_BATCH_SIZE],
                        task,
                        settings,
                    )
                )
            )
        population = _concatenate(chunks)
        population_rollouts += settings.population
        order = feasibility_elite_order(population, task)
        elite_count = max(1, int(round(settings.population * settings.elite_fraction)))
        elite_indices = torch.from_numpy(order[:elite_count].copy()).long()
        elites = actions[elite_indices]
        elite_mean = elites.mean(dim=0)
        centered = elites - elite_mean
        elite_covariance = centered.T @ centered / max(elite_count - 1, 1)
        mean = _project_normalized(
            settings.old_distribution_weight * mean
            + settings.elite_distribution_weight * elite_mean
        )
        covariance = _regularize(
            settings.old_distribution_weight * covariance
            + settings.elite_distribution_weight * elite_covariance,
            settings,
        )

        top_indices = torch.from_numpy(order[: settings.authoritative_top_n].copy()).long()
        top_actions = actions[top_indices]
        authoritative = _cpu_result(
            evaluate_normalized_actions_fixed_batch(
                simulator, repeated_context, top_actions, task, settings
            )
        )
        authoritative_rollouts += settings.authoritative_top_n
        authoritative_order = feasibility_elite_order(authoritative, task)
        selected_index = int(authoritative_order[0])
        selected_action = top_actions[selected_index].clone()
        authoritative_metrics = authoritative.row(selected_index)
        population_metrics = population.row(int(top_indices[selected_index]))
        duration = float(
            decode_policy_action(
                selected_action.float(), task, duration_max_s=settings.duration_max_s
            ).duration_s[0]
        )
        for metrics in (authoritative_metrics, population_metrics):
            metrics["maneuver_duration_s"] = duration
            metrics["hit_segment"] = event_segment(
                metrics["first_entry_time_s"], duration, settings.settle_duration_s
            )
        if best_authoritative is None or _selection_key(authoritative_metrics) < _selection_key(best_authoritative):
            best_action = selected_action
            best_authoritative = authoritative_metrics
            best_population = population_metrics
        if bool(authoritative_metrics["success"]) and first_success is None:
            first_success = iteration

        record = {
            "iteration": iteration,
            "population_success_count": int(population.success.sum()),
            "population_feasible_count": int(population.feasible.sum()),
            "authoritative_top_n_success_count": int(authoritative.success.sum()),
            "selected_authoritative": authoritative_metrics,
            "selected_population": population_metrics,
            "mean_duration_s": float(
                decode_policy_action(mean.float(), task, duration_max_s=settings.duration_max_s).duration_s[0]
            ),
            "duration_std_normalized": float(torch.sqrt(covariance[-1, -1])),
            "runtime_s": time.perf_counter() - iteration_start,
        }
        history.append(record)
        np.savez_compressed(
            checkpoint_directory / f"iteration_{iteration:03d}.npz",
            mean=mean.numpy(),
            covariance=covariance.numpy(),
            selected_action=selected_action.numpy(),
            rng_state=generator.get_state().numpy(),
        )
        final_actions, final_population = actions, population
        if (
            first_success is not None
            and iteration >= settings.minimum_iterations
            and iteration >= first_success + settings.polish_iterations_after_success
        ):
            break

    if final_actions is None or final_population is None:
        raise RuntimeError("Production CEM executed no iterations.")
    final_order = feasibility_elite_order(final_population, task)
    final_indices = torch.from_numpy(final_order[: settings.authoritative_top_n].copy()).long()
    final_top = final_actions[final_indices]
    final_authoritative = _cpu_result(
        evaluate_normalized_actions_fixed_batch(
            simulator, repeated_context, final_top, task, settings
        )
    )
    authoritative_rollouts += settings.authoritative_top_n
    final_auth_order = feasibility_elite_order(final_authoritative, task)
    final_index = int(final_auth_order[0])
    final_action = final_top[final_index].clone()
    authoritative_metrics = final_authoritative.row(final_index)
    population_metrics = final_population.row(int(final_indices[final_index]))
    duration = float(
        decode_policy_action(final_action.float(), task, duration_max_s=settings.duration_max_s).duration_s[0]
    )
    for metrics in (authoritative_metrics, population_metrics):
        metrics["maneuver_duration_s"] = duration
        metrics["hit_segment"] = event_segment(
            metrics["first_entry_time_s"], duration, settings.settle_duration_s
        )
    top_metrics = _metrics_for_actions(
        final_authoritative, final_top, task, settings
    )
    non_top_count = max(0, 512 - settings.authoritative_top_n)
    scorer_indices, scorer_categories = _deterministic_scorer_indices(
        final_population,
        excluded_indices=final_indices,
        maximum_count=non_top_count,
        seed=seed + 97_531,
    )
    scorer_index_tensor = torch.tensor(scorer_indices, dtype=torch.int64)
    sampled_actions = final_actions[scorer_index_tensor]
    sampled_result = _index_result(final_population, scorer_index_tensor)
    sampled_metrics = _metrics_for_actions(sampled_result, sampled_actions, task, settings)
    scorer_actions = torch.cat((final_top, sampled_actions), dim=0)
    scorer_metrics = top_metrics + sampled_metrics
    scorer_provenance = tuple(
        {
            "source": "authoritative_final_top32",
            "iteration": len(history),
            "population_index": int(final_indices[index]),
            "rank": index,
            "seed": seed,
        }
        for index in range(final_top.shape[0])
    ) + tuple(
        {
            "source": category,
            "iteration": len(history),
            "population_index": int(index),
            "rank": None,
            "seed": seed,
        }
        for index, category in zip(scorer_indices, scorer_categories, strict=True)
    )
    return ProductionCemResult(
        context_id=context_id,
        seed=seed,
        success=bool(authoritative_metrics["success"]),
        normalized_action=final_action,
        authoritative_metrics=authoritative_metrics,
        population_metrics=population_metrics,
        iterations=len(history),
        population_rollouts=population_rollouts,
        authoritative_rollouts=authoritative_rollouts,
        runtime_s=time.perf_counter() - started,
        first_authoritative_success_iteration=first_success,
        history=tuple(history),
        authoritative_top_actions=final_top.detach().cpu().double(),
        authoritative_top_metrics=top_metrics,
        scorer_actions=scorer_actions.detach().cpu().double(),
        scorer_metrics=scorer_metrics,
        scorer_provenance=scorer_provenance,
    )
