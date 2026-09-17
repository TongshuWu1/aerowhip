"""Identify pivot-attached DDER stiffness and damping from measured-root rollouts."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from simulator.cable import (
    START_PINNED_FREE_END,
    CableConfiguration,
    DderModel,
    DderState,
)

from .io import atomic_json, canonical_json_hash, sha256_file


PIVOT_FIT_SCHEMA = "pivot_cable_fit_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "config" / "model.json"
DEFAULT_FIT_CONFIG_PATH = PROJECT_ROOT / "config" / "cable_fit.json"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "force_takes"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "cable_fit_run"
INVALID_OBJECTIVE = 1.0e6


@dataclass(frozen=True, slots=True)
class PreparedTake:
    take_id: str
    role: str
    initial_positions_m: torch.Tensor
    initial_velocities_m_s: torch.Tensor
    root_positions_m: torch.Tensor
    measured_marker_positions_m: torch.Tensor
    starts: tuple[int, ...]
    initialization_marker_rmse_m: float
    dt_s: float

    @property
    def window_count(self) -> int:
        return int(self.initial_positions_m.shape[0])

    @property
    def horizon_steps(self) -> int:
        return int(self.root_positions_m.shape[1] - 1)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def contiguous_window_starts(
    valid: np.ndarray, *, horizon_steps: int, stride_steps: int
) -> tuple[int, ...]:
    """Return fixed-stride windows that never cross an invalid frame."""

    mask = np.asarray(valid, dtype=bool)
    if mask.ndim != 1 or horizon_steps < 1 or stride_steps < 1:
        raise ValueError("Window selection requires a 1D mask and positive steps.")
    padded = np.concatenate(([False], mask, [False]))
    transitions = np.diff(padded.astype(np.int8))
    run_starts = np.flatnonzero(transitions == 1)
    run_stops = np.flatnonzero(transitions == -1)
    starts: list[int] = []
    for run_start, run_stop in zip(run_starts, run_stops, strict=True):
        last_start = int(run_stop) - horizon_steps - 1
        if last_start >= int(run_start):
            starts.extend(range(int(run_start), last_start + 1, stride_steps))
    return tuple(starts)


def _stack_windows(values: np.ndarray, starts: tuple[int, ...], steps: int) -> np.ndarray:
    return np.stack([values[start : start + steps + 1] for start in starts])


def _prepare_take(
    take_id: str,
    role: str,
    take_path: Path,
    *,
    model: DderModel,
    cable: CableConfiguration,
    device: torch.device,
    dtype: torch.dtype,
    horizon_s: float,
    stride_s: float,
    projection_passes: int,
    initialization: str = "offline_centered",
    initialization_samples: int = 11,
) -> PreparedTake:
    with np.load(take_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    time_s = np.asarray(arrays["time_s"], dtype=np.float64)
    dt_s = float(np.median(np.diff(time_s)))
    horizon_steps = int(round(horizon_s / dt_s))
    stride_steps = int(round(stride_s / dt_s))
    if initialization not in {"offline_centered", "causal_polynomial"}:
        raise ValueError(f"Unknown fitting initialization: {initialization}")
    if initialization_samples < 3:
        raise ValueError("Initialization needs at least three samples.")
    history_steps = initialization_samples - 1 if initialization == "causal_polynomial" else 0
    starts = contiguous_window_starts(
        arrays["state_valid"],
        horizon_steps=horizon_steps + history_steps,
        stride_steps=stride_steps,
    )
    starts = tuple(start + history_steps for start in starts)
    if not starts:
        raise ValueError(f"{take_id} has no complete valid fitting windows.")

    positions_np = np.asarray(arrays["cable_node_position_world_m"], dtype=np.float64)
    velocities_np = np.asarray(arrays["cable_node_velocity_world_m_s"], dtype=np.float64)
    root_positions_np = np.asarray(arrays["root_position_world_m"], dtype=np.float64)
    root_velocities_np = np.asarray(arrays["root_velocity_world_m_s"], dtype=np.float64)
    window_positions = _stack_windows(positions_np, starts, horizon_steps)
    window_roots = _stack_windows(root_positions_np, starts, horizon_steps)
    initial_positions = torch.as_tensor(
        window_positions[:, 0], dtype=dtype, device=device
    )
    initial_velocities = torch.as_tensor(
        np.stack([velocities_np[start] for start in starts]),
        dtype=dtype,
        device=device,
    )
    initial_root_velocity = torch.as_tensor(
        np.stack([root_velocities_np[start] for start in starts]),
        dtype=dtype,
        device=device,
    )[:, None]
    if initialization == "causal_polynomial":
        from .state_initialization import endpoint_velocity
        history = torch.as_tensor(np.stack([
            positions_np[start - history_steps:start + 1] for start in starts
        ]), dtype=dtype, device=device)
        initial_velocities = endpoint_velocity(history, dt_s, initialization_samples)
        initial_root_velocity = initial_velocities[:, :1].clone()
    initial_boundary = initial_positions[:, :1].clone()
    for _ in range(projection_passes):
        initial_positions = model.project_lengths(
            initial_positions,
            initial_boundary,
            pinned_endpoints=START_PINNED_FREE_END,
        )
    initial_velocities = model.project_velocities(
        initial_positions,
        initial_velocities,
        initial_root_velocity,
        pinned_endpoints=START_PINNED_FREE_END,
    )
    marker_nodes = torch.as_tensor(
        cable.marker_node_indices[1:], dtype=torch.long, device=device
    )
    measured_initial = torch.as_tensor(
        window_positions[:, 0, cable.marker_node_indices[1:], :],
        dtype=dtype,
        device=device,
    )
    projected_initial = initial_positions.index_select(1, marker_nodes)
    initialization_rmse = float(
        torch.sqrt(
            torch.mean(
                torch.sum((projected_initial - measured_initial).square(), dim=2)
            )
        )
        .detach()
        .cpu()
    )
    return PreparedTake(
        take_id=take_id,
        role=role,
        initial_positions_m=initial_positions,
        initial_velocities_m_s=initial_velocities,
        root_positions_m=torch.as_tensor(window_roots, dtype=dtype, device=device),
        measured_marker_positions_m=torch.as_tensor(
            window_positions[:, :, cable.marker_node_indices[1:], :],
            dtype=dtype,
            device=device,
        ),
        starts=starts,
        initialization_marker_rmse_m=initialization_rmse,
        dt_s=dt_s,
    )


def _finite_state(state: DderState) -> torch.Tensor:
    return (
        torch.isfinite(state.positions_m).all(dim=(1, 2))
        & torch.isfinite(state.velocities_m_s).all(dim=(1, 2))
        & (state.positions_m.abs().amax(dim=(1, 2)) < 1.0e3)
        & (state.velocities_m_s.abs().amax(dim=(1, 2)) < 1.0e4)
    )


def _pseudo_huber(distance_m: torch.Tensor, scale_m: float) -> torch.Tensor:
    scale = torch.as_tensor(scale_m, dtype=distance_m.dtype, device=distance_m.device)
    return scale.square() * (torch.sqrt(1.0 + (distance_m / scale).square()) - 1.0)


def evaluate_population(
    population: np.ndarray | torch.Tensor,
    takes: tuple[PreparedTake, ...],
    *,
    model: DderModel,
    cable: CableConfiguration,
    robust_scale_m: float,
    lead_times_s: tuple[float, ...],
    use_optimized_cuda: bool,
) -> dict[str, Any]:
    """Evaluate EI/Cb candidates without using validation to select a candidate."""

    reference = takes[0].initial_positions_m
    candidates = torch.as_tensor(
        population, dtype=reference.dtype, device=reference.device
    )
    if candidates.ndim != 2 or candidates.shape[1] != 2:
        raise ValueError("population must have shape Px2 [EI, Cb].")
    candidate_count = int(candidates.shape[0])
    marker_nodes = torch.as_tensor(
        cable.marker_node_indices[1:], dtype=torch.long, device=reference.device
    )
    per_take: dict[str, dict[str, torch.Tensor]] = {}
    overall_invalid = torch.zeros(
        candidate_count, dtype=torch.bool, device=reference.device
    )

    for take in takes:
        windows = take.window_count
        batch = windows * candidate_count
        positions = (
            take.initial_positions_m[:, None]
            .expand(windows, candidate_count, -1, -1)
            .reshape(batch, cable.node_count, 3)
            .contiguous()
        )
        velocities = (
            take.initial_velocities_m_s[:, None]
            .expand(windows, candidate_count, -1, -1)
            .reshape(batch, cable.node_count, 3)
            .contiguous()
        )
        state = model.initial_state(positions, velocities)
        fallback = state
        repeated_candidates = candidates.repeat(windows, 1)
        constants = replace(
            model.runtime_constants(positions),
            bending_stiffness_n_m2=repeated_candidates[:, 0],
            bending_damping_n_m2_s=repeated_candidates[:, 1],
        )
        dt = torch.full(
            (batch,), take.dt_s, dtype=reference.dtype, device=reference.device
        )
        invalid_instances = torch.zeros(
            (windows, candidate_count), dtype=torch.bool, device=reference.device
        )
        robust_sum = torch.zeros(candidate_count, dtype=reference.dtype, device=reference.device)
        square_sum = torch.zeros_like(robust_sum)
        tip_square_sum = torch.zeros_like(robust_sum)
        sample_count = 0
        tip_count = 0
        lead_square: dict[float, torch.Tensor] = {
            lead: torch.zeros_like(robust_sum) for lead in lead_times_s
        }
        lead_tip_square: dict[float, torch.Tensor] = {
            lead: torch.zeros_like(robust_sum) for lead in lead_times_s
        }
        lead_step = {
            int(round(lead / take.dt_s)): lead for lead in lead_times_s
        }

        with torch.no_grad():
            for step in range(1, take.horizon_steps + 1):
                boundary = (
                    take.root_positions_m[:, step, None, :][:, None]
                    .expand(windows, candidate_count, 1, 3)
                    .reshape(batch, 1, 3)
                    .contiguous()
                )
                state = model.step_runtime(
                    state,
                    boundary,
                    dt,
                    constants,
                    iterative_damping=use_optimized_cuda,
                    damping_backend=(
                        "pcg32_experimental" if use_optimized_cuda else "pcg60_reference"
                    ),
                    pinned_endpoints=START_PINNED_FREE_END,
                    create_graph=False,
                )
                newly_invalid = (~_finite_state(state)).reshape(windows, candidate_count)
                invalid_instances |= newly_invalid
                invalid_flat = invalid_instances.reshape(batch)
                state = DderState(
                    torch.where(
                        invalid_flat[:, None, None], fallback.positions_m, state.positions_m
                    ),
                    torch.where(
                        invalid_flat[:, None, None], fallback.velocities_m_s, state.velocities_m_s
                    ),
                )
                predicted = state.positions_m.index_select(1, marker_nodes).reshape(
                    windows, candidate_count, cable.moving_marker_count, 3
                )
                measured = take.measured_marker_positions_m[:, step, None]
                distance = torch.linalg.vector_norm(predicted - measured, dim=-1)
                robust_sum += _pseudo_huber(distance, robust_scale_m).sum(dim=(0, 2))
                square = distance.square()
                square_sum += square.sum(dim=(0, 2))
                tip_square_sum += square[:, :, -1].sum(dim=0)
                sample_count += windows * cable.moving_marker_count
                tip_count += windows
                if step in lead_step:
                    lead = lead_step[step]
                    lead_square[lead] += square.sum(dim=(0, 2))
                    lead_tip_square[lead] += square[:, :, -1].sum(dim=0)

        invalid_candidates = invalid_instances.any(dim=0)
        overall_invalid |= invalid_candidates
        take_result: dict[str, torch.Tensor] = {
            "objective": robust_sum / sample_count,
            "marker_mse": square_sum / sample_count,
            "tip_mse": tip_square_sum / tip_count,
            "invalid": invalid_candidates,
        }
        lead_count = windows * cable.moving_marker_count
        for lead in lead_times_s:
            take_result[f"marker_mse_at_{lead:g}s"] = lead_square[lead] / lead_count
            take_result[f"tip_mse_at_{lead:g}s"] = lead_tip_square[lead] / windows
        per_take[take.take_id] = take_result

    objective = torch.stack([row["objective"] for row in per_take.values()]).mean(dim=0)
    marker_mse = torch.stack([row["marker_mse"] for row in per_take.values()]).mean(dim=0)
    tip_mse = torch.stack([row["tip_mse"] for row in per_take.values()]).mean(dim=0)
    objective = torch.where(
        overall_invalid | ~torch.isfinite(objective),
        torch.full_like(objective, INVALID_OBJECTIVE),
        objective,
    )
    return {
        "objective": objective,
        "marker_mse": marker_mse,
        "tip_mse": tip_mse,
        "invalid": overall_invalid,
        "per_take": per_take,
    }


def _candidate_report(
    evaluation: dict[str, Any], index: int, lead_times_s: tuple[float, ...]
) -> dict[str, Any]:
    per_take: dict[str, Any] = {}
    for take_id, values in evaluation["per_take"].items():
        row = {
            "objective": float(values["objective"][index].detach().cpu()),
            "marker_rmse_m": math.sqrt(
                float(values["marker_mse"][index].detach().cpu())
            ),
            "tip_rmse_m": math.sqrt(float(values["tip_mse"][index].detach().cpu())),
        }
        for lead in lead_times_s:
            row[f"marker_rmse_at_{lead:g}s_m"] = math.sqrt(
                float(values[f"marker_mse_at_{lead:g}s"][index].detach().cpu())
            )
            row[f"tip_rmse_at_{lead:g}s_m"] = math.sqrt(
                float(values[f"tip_mse_at_{lead:g}s"][index].detach().cpu())
            )
        per_take[take_id] = row
    return {
        "objective": float(evaluation["objective"][index].detach().cpu()),
        "equal_take_marker_rmse_m": math.sqrt(
            float(evaluation["marker_mse"][index].detach().cpu())
        ),
        "equal_take_tip_rmse_m": math.sqrt(
            float(evaluation["tip_mse"][index].detach().cpu())
        ),
        "per_take": per_take,
    }


def _refined_log_axis(
    axis: np.ndarray,
    best_index: int,
    *,
    floor: float,
    ceiling: float,
) -> np.ndarray:
    if 0 < best_index < len(axis) - 1:
        lower, upper = axis[best_index - 1], axis[best_index + 1]
    elif best_index == 0:
        step = float(axis[1] - axis[0])
        lower, upper = max(math.log10(floor), axis[0] - 2.0 * step), axis[1]
    else:
        step = float(axis[-1] - axis[-2])
        lower, upper = axis[-2], min(math.log10(ceiling), axis[-1] + 2.0 * step)
    return np.linspace(lower, upper, len(axis))


def _write_grid(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_pivot_cable(
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    fit_config_path: Path = DEFAULT_FIT_CONFIG_PATH,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    progress: Any = print,
) -> dict[str, Any]:
    """Fit training takes, then evaluate the frozen winner on validation takes."""

    model_payload = _load_json(Path(model_path))
    config = _load_json(Path(fit_config_path))
    data_manifest = _load_json(Path(data_root) / "manifest.json")
    requested_device = str(config["device"])
    device = torch.device(
        "cuda" if requested_device == "auto" and torch.cuda.is_available()
        else "cpu" if requested_device == "auto"
        else requested_device
    )
    dtype = torch.float32 if config["dtype"] == "float32" else torch.float64
    use_optimized_cuda = device.type == "cuda" and dtype == torch.float32
    cable = CableConfiguration.from_mapping(model_payload["cable"])
    parameters = cable.dder_parameters(
        EI=float(model_payload["cable"]["EI_n_m2"]),
        Cb=float(model_payload["cable"]["Cb_n_m2_s"]),
    )
    dder = DderModel(parameters)
    objective_config = config["objective"]

    prepared: list[PreparedTake] = []
    for take_id, row in data_manifest["takes"].items():
        role = str(row["role"])
        if role not in {"training", "validation"}:
            continue
        item = _prepare_take(
            take_id,
            role,
            Path(data_root) / str(row["path"]),
            model=dder,
            cable=cable,
            device=device,
            dtype=dtype,
            horizon_s=float(objective_config["horizon_s"]),
            stride_s=float(objective_config["stride_s"]),
            projection_passes=int(objective_config["initial_projection_passes"]),
            initialization=objective_config.get("initialization", "offline_centered"),
            initialization_samples=int(objective_config.get("initialization_samples", 11)),
        )
        prepared.append(item)
        progress(
            f"Prepared {take_id}: {item.window_count} windows, "
            f"initialization RMSE {1000.0 * item.initialization_marker_rmse_m:.2f} mm"
        )
    training = tuple(item for item in prepared if item.role == "training")
    validation = tuple(item for item in prepared if item.role == "validation")
    if not training or not validation:
        raise ValueError("The fit requires both training and validation takes.")

    search = config["search"]
    grid_size = int(search["grid_size_per_axis"])
    passes = int(search["passes"])
    stable_ei = dder.maximum_stable_bending_stiffness(
        float(model_payload["simulation"]["dt_s"]),
        pinned_endpoints=START_PINNED_FREE_END,
    )
    ei_bounds = tuple(float(value) for value in search["EI_n_m2"])
    cb_bounds = tuple(float(value) for value in search["Cb_n_m2_s"])
    if ei_bounds[1] >= stable_ei:
        raise ValueError(
            f"EI search upper bound {ei_bounds[1]:g} exceeds stability limit {stable_ei:g}."
        )
    ei_axis = np.linspace(math.log10(ei_bounds[0]), math.log10(ei_bounds[1]), grid_size)
    cb_axis = np.linspace(math.log10(cb_bounds[0]), math.log10(cb_bounds[1]), grid_size)
    lead_times = tuple(float(value) for value in config["validation_lead_times_s"])
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    pass_summaries: list[dict[str, Any]] = []
    best_parameters: np.ndarray | None = None
    best_objective = math.inf
    started = time.perf_counter()

    for pass_index in range(1, passes + 1):
        mesh_ei, mesh_cb = np.meshgrid(ei_axis, cb_axis, indexing="ij")
        population = np.column_stack((10.0**mesh_ei.ravel(), 10.0**mesh_cb.ravel()))
        pass_started = time.perf_counter()
        evaluation = evaluate_population(
            population,
            training,
            model=dder,
            cable=cable,
            robust_scale_m=float(objective_config["pseudo_huber_scale_m"]),
            lead_times_s=lead_times,
            use_optimized_cuda=use_optimized_cuda,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        objectives = evaluation["objective"].detach().cpu().numpy()
        invalid = evaluation["invalid"].detach().cpu().numpy()
        best_flat = int(np.argmin(objectives))
        if bool(invalid[best_flat]):
            raise RuntimeError("Every EI/Cb candidate was invalid.")
        best_i, best_j = np.unravel_index(best_flat, (grid_size, grid_size))
        pass_best = population[best_flat]
        pass_best_objective = float(objectives[best_flat])
        if pass_best_objective < best_objective:
            best_objective = pass_best_objective
            best_parameters = pass_best.copy()
        rows: list[dict[str, Any]] = []
        for index, (ei, cb) in enumerate(population):
            row: dict[str, Any] = {
                "pass": pass_index,
                "candidate": index,
                "EI_n_m2": float(ei),
                "Cb_n_m2_s": float(cb),
                "objective": float(objectives[index]),
                "marker_rmse_m": math.sqrt(
                    float(evaluation["marker_mse"][index].detach().cpu())
                ),
                "tip_rmse_m": math.sqrt(
                    float(evaluation["tip_mse"][index].detach().cpu())
                ),
                "invalid": bool(invalid[index]),
            }
            for take_id, values in evaluation["per_take"].items():
                row[f"{take_id}_objective"] = float(
                    values["objective"][index].detach().cpu()
                )
            rows.append(row)
        all_rows.extend(rows)
        _write_grid(output / f"grid_pass_{pass_index}.csv", rows)
        pass_summary = {
            "pass": pass_index,
            "runtime_s": time.perf_counter() - pass_started,
            "best_EI_n_m2": float(pass_best[0]),
            "best_Cb_n_m2_s": float(pass_best[1]),
            "best_objective": pass_best_objective,
            "best_grid_index": [int(best_i), int(best_j)],
            "invalid_candidates": int(np.count_nonzero(invalid)),
            "EI_range_n_m2": [float(10.0**ei_axis[0]), float(10.0**ei_axis[-1])],
            "Cb_range_n_m2_s": [float(10.0**cb_axis[0]), float(10.0**cb_axis[-1])],
        }
        pass_summaries.append(pass_summary)
        atomic_json(output / "progress.json", {"passes": pass_summaries})
        progress(
            f"Pass {pass_index}/{passes}: EI={pass_best[0]:.7g}, "
            f"Cb={pass_best[1]:.7g}, objective={pass_best_objective:.7g}, "
            f"runtime={pass_summary['runtime_s']:.1f}s"
        )
        if pass_index < passes:
            ei_axis = _refined_log_axis(
                ei_axis,
                int(best_i),
                floor=1.0e-8,
                ceiling=0.95 * stable_ei,
            )
            cb_axis = _refined_log_axis(
                cb_axis,
                int(best_j),
                floor=1.0e-7,
                ceiling=0.2,
            )

    assert best_parameters is not None
    comparison_baseline = config["comparison_baseline"]
    baseline_parameters = np.asarray(
        [[
            float(comparison_baseline["EI_n_m2"]),
            float(comparison_baseline["Cb_n_m2_s"]),
        ]],
        dtype=np.float64,
    )
    winner = best_parameters[None]
    training_baseline = evaluate_population(
        baseline_parameters,
        training,
        model=dder,
        cable=cable,
        robust_scale_m=float(objective_config["pseudo_huber_scale_m"]),
        lead_times_s=lead_times,
        use_optimized_cuda=use_optimized_cuda,
    )
    training_winner = evaluate_population(
        winner,
        training,
        model=dder,
        cable=cable,
        robust_scale_m=float(objective_config["pseudo_huber_scale_m"]),
        lead_times_s=lead_times,
        use_optimized_cuda=use_optimized_cuda,
    )
    validation_baseline = evaluate_population(
        baseline_parameters,
        validation,
        model=dder,
        cable=cable,
        robust_scale_m=float(objective_config["pseudo_huber_scale_m"]),
        lead_times_s=lead_times,
        use_optimized_cuda=use_optimized_cuda,
    )
    validation_winner = evaluate_population(
        winner,
        validation,
        model=dder,
        cable=cable,
        robust_scale_m=float(objective_config["pseudo_huber_scale_m"]),
        lead_times_s=lead_times,
        use_optimized_cuda=use_optimized_cuda,
    )
    train_before = _candidate_report(training_baseline, 0, lead_times)
    train_after = _candidate_report(training_winner, 0, lead_times)
    validation_before = _candidate_report(validation_baseline, 0, lead_times)
    validation_after = _candidate_report(validation_winner, 0, lead_times)

    def improvement(before: dict[str, Any], after: dict[str, Any], key: str) -> float:
        return 100.0 * (float(before[key]) - float(after[key])) / float(before[key])

    final_i = int(np.argmin(np.abs(10.0**ei_axis - best_parameters[0])))
    final_j = int(np.argmin(np.abs(10.0**cb_axis - best_parameters[1])))
    result = {
        "schema": PIVOT_FIT_SCHEMA,
        "status": "fit_complete_validation_not_used_for_selection",
        "fitted_parameters": {
            "EI_n_m2": float(best_parameters[0]),
            "Cb_n_m2_s": float(best_parameters[1]),
        },
        "replaced_parameters": {
            "EI_n_m2": float(baseline_parameters[0, 0]),
            "Cb_n_m2_s": float(baseline_parameters[0, 1]),
            "reason": str(comparison_baseline["boundary"]),
        },
        "boundary_model": {
            "type": "one_position_node_pivot",
            "pinned_nodes": 1,
            "attitude_used": False,
            "root_motion": "measured attachment trajectory",
        },
        "selection": {
            "uses_roles": ["training"],
            "validation_used_for_selection": False,
            "untouched_test_used": False,
            "passes": pass_summaries,
            "best_training_objective": best_objective,
            "final_grid_boundary": {
                "EI": final_i in {0, grid_size - 1},
                "Cb": final_j in {0, grid_size - 1},
            },
        },
        "data": {
            "training_takes": [item.take_id for item in training],
            "validation_takes": [item.take_id for item in validation],
            "protected_takes": data_manifest.get("excluded_takes", {}),
            "training_windows": sum(item.window_count for item in training),
            "validation_windows": sum(item.window_count for item in validation),
            "window_horizon_s": float(objective_config["horizon_s"]),
            "window_stride_s": float(objective_config["stride_s"]),
            "initialization": objective_config.get("initialization", "offline_centered"),
            "initialization_samples": int(objective_config.get("initialization_samples", 11)),
            "initialization_marker_rmse_m": {
                item.take_id: item.initialization_marker_rmse_m for item in prepared
            },
        },
        "training": {
            "before": train_before,
            "after": train_after,
            "marker_rmse_improvement_percent": improvement(
                train_before, train_after, "equal_take_marker_rmse_m"
            ),
            "tip_rmse_improvement_percent": improvement(
                train_before, train_after, "equal_take_tip_rmse_m"
            ),
        },
        "validation": {
            "before": validation_before,
            "after": validation_after,
            "marker_rmse_improvement_percent": improvement(
                validation_before, validation_after, "equal_take_marker_rmse_m"
            ),
            "tip_rmse_improvement_percent": improvement(
                validation_before, validation_after, "equal_take_tip_rmse_m"
            ),
        },
        "solver": {
            "device": str(device),
            "dtype": str(dtype).removeprefix("torch."),
            "substeps": cable.substeps,
            "constraint_iterations": cable.constraint_iterations,
            "damping_backend": (
                "pcg32_experimental" if use_optimized_cuda else "direct"
            ),
            "maximum_stable_EI_n_m2": stable_ei,
        },
        "provenance": {
            "model_config_sha256": canonical_json_hash(model_payload),
            "fit_config_sha256": sha256_file(fit_config_path),
            "force_dataset_manifest_sha256": sha256_file(Path(data_root) / "manifest.json"),
            "pivot_fitter_sha256": sha256_file(Path(__file__)),
            "dder_source_sha256": sha256_file(
                PROJECT_ROOT / "simulator" / "cable" / "dder.py"
            ),
            "cuda_mechanics_source_sha256": sha256_file(
                PROJECT_ROOT / "simulator" / "cable" / "cuda_fixed_pcg.py"
            ),
        },
        "runtime_s": time.perf_counter() - started,
    }
    atomic_json(output / "fit_result.json", result)
    _write_grid(output / "all_candidates.csv", all_rows)
    record_validation_window(output / 'validation_window.npz', validation[0], model=dder, cable=cable,
                             before=baseline_parameters[0], after=best_parameters,
                             use_optimized_cuda=use_optimized_cuda)
    return result


@torch.no_grad()
def record_validation_window(path, take, *, model, cable, before, after, use_optimized_cuda):
    """Diagnostic first held-out window with the same prescribed-root dynamics."""
    reference = take.initial_positions_m[:1]
    marker_nodes = torch.as_tensor(cable.marker_node_indices[1:], device=reference.device)
    arrays = {'time_s': np.arange(take.horizon_steps + 1) * take.dt_s,
              'measured_markers_m': take.measured_marker_positions_m[0].cpu().numpy()}
    for name, parameters in (('active_markers_m', before), ('candidate_markers_m', after)):
        state = model.initial_state(reference.clone(), take.initial_velocities_m_s[:1].clone())
        constants = replace(model.runtime_constants(reference),
            bending_stiffness_n_m2=reference.new_tensor([float(parameters[0])]),
            bending_damping_n_m2_s=reference.new_tensor([float(parameters[1])]))
        frames = [state.positions_m[0, marker_nodes].clone()]
        for step in range(1, take.horizon_steps + 1):
            state = model.step_runtime(state, take.root_positions_m[:1, step, None, :],
                reference.new_full((1,), take.dt_s), constants, iterative_damping=use_optimized_cuda,
                damping_backend='pcg32_experimental' if use_optimized_cuda else 'pcg60_reference',
                pinned_endpoints=START_PINNED_FREE_END, create_graph=False)
            frames.append(state.positions_m[0, marker_nodes].clone())
        arrays[name] = torch.stack(frames).cpu().numpy()
    from simulator.validation_preview import write_recording
    write_recording(Path(path), arrays, {'take': take.take_id, 'start_frame': take.starts[0],
        'role': take.role, 'boundary': 'measured_attachment_position',
        'selection': 'first valid window of first validation take'})
