"""Planar trajectory preparation and two-parameter cable identification."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
import uuid
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .config import DEFAULT_CONFIG_PATH, OfflineSettings, load_settings
from .planar_data import (
    PLANAR_OBSERVATION_METHOD,
    PlanarSequence,
    load_planar_sequence,
    save_planar_trajectory,
)
from ..shared.capture_manifest import load_capture_manifest
from ..shared.dder import DderModel, DderParameters, DderState
from ..shared.observation_data import sha256_file
from ..shared.routes import resample_polyline


MODEL_SCHEMA = "planar_rgb_reference_dder_v2"


@dataclass(frozen=True, slots=True)
class PreparedSequence:
    svo_path: Path
    observation_path: Path
    timestamp_ns: np.ndarray
    observation_positions_m: np.ndarray
    observation_velocities_m_s: np.ndarray
    initial_positions_m: np.ndarray
    initial_velocities_m_s: np.ndarray
    valid: np.ndarray
    initialization_shift_m: float
    pixel_scale_m: float


@dataclass(frozen=True, slots=True)
class FitResult:
    bending_stiffness_n_m2: float
    bending_damping_n_m2_s: float
    train_curve_residual_m: float
    elapsed_s: float


def _bounded_log_parameter(
    raw: torch.Tensor,
    minimum: float,
    maximum: float,
) -> torch.Tensor:
    lower, upper = math.log(minimum), math.log(maximum)
    return torch.exp(lower + (upper - lower) * torch.sigmoid(raw))


def _raw_for_value(value: float, minimum: float, maximum: float) -> float:
    fraction = (math.log(value) - math.log(minimum)) / (
        math.log(maximum) - math.log(minimum)
    )
    fraction = min(max(fraction, 1.0e-6), 1.0 - 1.0e-6)
    return math.log(fraction / (1.0 - fraction))


def _contiguous_runs(
    valid: np.ndarray,
    timestamp_ns: np.ndarray,
    maximum_dt_s: float,
) -> list[np.ndarray]:
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        return []
    runs: list[list[int]] = [[int(indices[0])]]
    for index in indices[1:]:
        previous = runs[-1][-1]
        dt = (int(timestamp_ns[index]) - int(timestamp_ns[previous])) * 1.0e-9
        if int(index) == previous + 1 and 0.0 < dt <= maximum_dt_s:
            runs[-1].append(int(index))
        else:
            runs.append([int(index)])
    return [np.asarray(run, dtype=np.int64) for run in runs]


def _local_quadratic(
    values: np.ndarray,
    timestamp_ns: np.ndarray,
    valid: np.ndarray,
    *,
    window: int,
    maximum_dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Local polynomial position/velocity estimate over contiguous measurements."""

    source = np.asarray(values, dtype=np.float64)
    smoothed = np.full_like(source, np.nan)
    derivative = np.full_like(source, np.nan)
    half = window // 2
    for run in _contiguous_runs(valid, timestamp_ns, maximum_dt_s):
        if len(run) < 3:
            continue
        for offset, frame in enumerate(run):
            start = max(0, offset - half)
            stop = min(len(run), offset + half + 1)
            if stop - start < 3:
                if start == 0:
                    stop = min(len(run), 3)
                else:
                    start = max(0, len(run) - 3)
            selected = run[start:stop]
            time_s = (timestamp_ns[selected] - timestamp_ns[frame]) * 1.0e-9
            design = np.column_stack((np.ones(len(selected)), time_s, np.square(time_s)))
            target = source[selected].reshape(len(selected), -1)
            coefficient, *_ = np.linalg.lstsq(design, target, rcond=None)
            smoothed[frame] = coefficient[0].reshape(source.shape[1:])
            derivative[frame] = coefficient[1].reshape(source.shape[1:])
    return smoothed, derivative


def _make_model(settings: OfflineSettings, *, ei: float, cb: float) -> DderModel:
    return DderModel(
        DderParameters(
            node_count=settings.cable.node_count,
            cable_length_m=settings.cable.length_m,
            cable_mass_kg=settings.cable.mass_kg,
            cable_diameter_m=settings.cable.diameter_m,
            bending_stiffness_n_m2=ei,
            bending_damping_n_m2_s=cb,
            gravity_camera_m_s2=(0.0, 0.0, -9.81),
            substeps=settings.solver.substeps,
            constraint_iterations=settings.solver.constraint_iterations,
        )
    )


def _project_fixed_length(
    positions_m: np.ndarray,
    valid: np.ndarray,
    model: DderModel,
    device: torch.device,
) -> np.ndarray:
    output = np.full_like(positions_m, np.nan, dtype=np.float64)
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        return output
    value = torch.as_tensor(positions_m[indices], dtype=torch.float64, device=device)
    boundary = value[:, (0, -1)]
    for _ in range(12):
        value = model.project_lengths(value, boundary)
        if float(model.maximum_segment_error_m(value).max().detach().cpu()) <= 1.0e-8:
            break
    else:
        raise RuntimeError("Fixed-length planar curve projection did not converge.")
    output[indices] = value.detach().cpu().numpy()
    return output


def _prepare_sequence(
    sequence: PlanarSequence,
    svo_path: Path,
    settings: OfflineSettings,
    device: torch.device,
) -> PreparedSequence:
    mapping_length = float(sequence.metadata["image_plane_mapping"]["cable_length_m"])
    if not math.isclose(
        mapping_length,
        settings.cable.length_m,
        rel_tol=1.0e-9,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            f"{svo_path.name} was scaled using cable length {mapping_length:g} m, "
            f"but the active cable length is {settings.cable.length_m:g} m. "
            "Re-extract the 2D trajectory."
        )
    frame_count = sequence.frame_count
    node_count = settings.cable.node_count
    raw_xz = np.full((frame_count, node_count, 2), np.nan, dtype=np.float64)
    valid = np.asarray(sequence.complete, dtype=bool).copy()
    for frame in np.flatnonzero(valid):
        route = resample_polyline(sequence.route_xz_m[frame], node_count)
        route[0], route[-1] = sequence.endpoint_xz_m[frame]
        raw_xz[frame] = route

    observed_xz, observed_velocity_xz = _local_quadratic(
        raw_xz,
        sequence.timestamp_ns,
        valid,
        window=settings.optimization.velocity_window_frames,
        maximum_dt_s=settings.optimization.maximum_dt_s,
    )
    valid &= np.all(np.isfinite(observed_xz), axis=(1, 2))
    valid &= np.all(np.isfinite(observed_velocity_xz), axis=(1, 2))
    separation = np.linalg.norm(observed_xz[:, -1] - observed_xz[:, 0], axis=1)
    valid &= separation <= settings.cable.length_m
    if np.count_nonzero(valid) < settings.optimization.window_frames:
        raise ValueError(
            f"{svo_path.name} has too few complete, metrically feasible planar frames."
        )

    observations = np.full((frame_count, node_count, 3), np.nan, dtype=np.float64)
    observations[:, :, 0] = observed_xz[:, :, 0]
    observations[:, :, 1] = 0.0
    observations[:, :, 2] = observed_xz[:, :, 1]
    observation_velocity = np.full_like(observations, np.nan)
    observation_velocity[:, :, 0] = observed_velocity_xz[:, :, 0]
    observation_velocity[:, :, 1] = 0.0
    observation_velocity[:, :, 2] = observed_velocity_xz[:, :, 1]
    nominal_ei = math.sqrt(
        settings.optimization.ei_min_n_m2 * settings.optimization.ei_max_n_m2
    )
    nominal_cb = math.sqrt(
        settings.optimization.cb_min_n_m2_s
        * settings.optimization.cb_max_n_m2_s
    )
    model = _make_model(settings, ei=nominal_ei, cb=nominal_cb)
    initial_positions = _project_fixed_length(observations, valid, model, device)
    indices = np.flatnonzero(valid)
    q = torch.as_tensor(initial_positions[indices], dtype=torch.float64, device=device)
    v = torch.as_tensor(observation_velocity[indices], dtype=torch.float64, device=device)
    boundary_velocity = v[:, (0, -1)]
    v = model.project_velocities(q, v, boundary_velocity)
    initial_velocity = np.full_like(observation_velocity, np.nan)
    initial_velocity[indices] = v.detach().cpu().numpy()

    residual = np.linalg.norm(
        initial_positions[valid] - observations[valid],
        axis=2,
    )
    initialization_shift = float(np.sqrt(np.mean(np.square(residual))))
    pixel_scale_m = float(sequence.metadata["image_plane_mapping"]["scale_m_per_px"])
    save_planar_trajectory(
        sequence.path,
        positions_xz_m=observations[:, :, (0, 2)],
        velocities_xz_m_s=observation_velocity[:, :, (0, 2)],
        valid=valid,
        timestamp_ns=sequence.timestamp_ns,
        metadata={
            "method": "pidnet_route_nodes_without_position_filter",
            "node_count": node_count,
            "cable_length_m": settings.cable.length_m,
            "velocity_window_frames": settings.optimization.velocity_window_frames,
            "initialization_shift_m": initialization_shift,
            "pixel_scale_m": pixel_scale_m,
        },
    )
    return PreparedSequence(
        svo_path,
        sequence.path,
        sequence.timestamp_ns,
        observations,
        observation_velocity,
        initial_positions,
        initial_velocity,
        valid,
        initialization_shift,
        pixel_scale_m,
    )


def _validate_recording(svo_path: Path) -> None:
    manifest = load_capture_manifest(svo_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Recording manifest is missing for {svo_path.name}.")
    experiment = manifest.get("experiment")
    if not isinstance(experiment, dict) or int(experiment.get("cable_identity", 0)) not in (1, 2):
        raise ValueError(f"Recording metadata is incomplete for {svo_path.name}.")


def prepare_recording(
    svo_path: str | Path,
    observation_path: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> PreparedSequence:
    """Create and save unfiltered PIDNet node observations for one recording."""

    settings = load_settings(config_path)
    svo = Path(svo_path).expanduser().resolve()
    observation = Path(observation_path).expanduser().resolve()
    _validate_recording(svo)
    return _prepare_sequence(
        load_planar_sequence(observation),
        svo,
        settings,
        torch.device(settings.optimization.device),
    )


def _windows(
    sequences: Sequence[PreparedSequence],
    settings: OfflineSettings,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    observation_windows: list[np.ndarray] = []
    position_starts: list[np.ndarray] = []
    velocity_starts: list[np.ndarray] = []
    timestamp_windows: list[np.ndarray] = []
    length = settings.optimization.window_frames
    stride = settings.optimization.window_stride
    for sequence in sequences:
        for run in _contiguous_runs(
            sequence.valid,
            sequence.timestamp_ns,
            settings.optimization.maximum_dt_s,
        ):
            for offset in range(0, len(run) - length + 1, stride):
                selected = run[offset : offset + length]
                observation_windows.append(sequence.observation_positions_m[selected])
                position_starts.append(sequence.initial_positions_m[selected[0]])
                velocity_starts.append(sequence.initial_velocities_m_s[selected[0]])
                timestamp_windows.append(sequence.timestamp_ns[selected])
    if not observation_windows:
        names = ", ".join(item.svo_path.name for item in sequences)
        raise ValueError(f"No complete trajectory window is available in {names}.")
    return (
        torch.as_tensor(np.asarray(observation_windows), dtype=torch.float64),
        torch.as_tensor(np.asarray(position_starts), dtype=torch.float64),
        torch.as_tensor(np.asarray(velocity_starts), dtype=torch.float64),
        torch.as_tensor(np.asarray(timestamp_windows), dtype=torch.int64),
    )


def _rollout_loss(
    model: DderModel,
    observations: torch.Tensor,
    initial_position: torch.Tensor,
    initial_velocity: torch.Tensor,
    timestamps: torch.Tensor,
    *,
    ei: torch.Tensor,
    cb: torch.Tensor,
    robust_scale_m: float,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = DderState(initial_position, initial_velocity)
    squared: list[torch.Tensor] = []
    robust: list[torch.Tensor] = []
    for frame in range(1, observations.shape[1]):
        dt = (timestamps[:, frame] - timestamps[:, frame - 1]).to(
            dtype=observations.dtype
        ) * 1.0e-9
        state = model.step(
            state,
            observations[:, frame, (0, -1)],
            dt,
            create_graph=create_graph,
            bending_stiffness_n_m2=ei,
            bending_damping_n_m2_s=cb,
        )
        error = state.positions_m[:, 1:-1] - observations[:, frame, 1:-1]
        distance = torch.linalg.vector_norm(error, dim=-1)
        normalized = distance / robust_scale_m
        robust.append(torch.sqrt(1.0 + normalized.square()) - 1.0)
        squared.append(distance.square())
    return torch.cat([value.reshape(-1) for value in robust]).mean(), torch.cat(
        [value.reshape(-1) for value in squared]
    ).mean()


def _fit(
    sequences: Sequence[PreparedSequence],
    settings: OfflineSettings,
) -> FitResult:
    device = torch.device(settings.optimization.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for planar DDER fitting but is unavailable.")
    if not sequences:
        raise ValueError("Select at least one parameter-fit recording.")
    observations, initial_positions, initial_velocities, timestamps = (
        value.to(device) for value in _windows(sequences, settings)
    )
    nominal_ei = math.sqrt(
        settings.optimization.ei_min_n_m2 * settings.optimization.ei_max_n_m2
    )
    nominal_cb = math.sqrt(
        settings.optimization.cb_min_n_m2_s
        * settings.optimization.cb_max_n_m2_s
    )
    model = _make_model(settings, ei=nominal_ei, cb=nominal_cb)
    if settings.optimization.ei_max_n_m2 > model.maximum_stable_bending_stiffness(
        settings.optimization.maximum_dt_s
    ):
        raise ValueError("Configured EI range exceeds the explicit solver stability limit.")
    if settings.optimization.cb_max_n_m2_s > model.maximum_stable_bending_damping(
        settings.optimization.maximum_dt_s
    ):
        raise ValueError("Configured Cb range exceeds the explicit solver stability limit.")

    raw_ei = torch.tensor(
        _raw_for_value(
            nominal_ei,
            settings.optimization.ei_min_n_m2,
            settings.optimization.ei_max_n_m2,
        ),
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    raw_cb = torch.tensor(
        _raw_for_value(
            nominal_cb,
            settings.optimization.cb_min_n_m2_s,
            settings.optimization.cb_max_n_m2_s,
        ),
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    maximum_evaluations = max(20, 2 * settings.optimization.optimizer_iterations)
    optimizer = torch.optim.LBFGS(
        (raw_ei, raw_cb),
        lr=1.0,
        max_iter=settings.optimization.optimizer_iterations,
        max_eval=maximum_evaluations,
        tolerance_grad=1.0e-10,
        tolerance_change=1.0e-12,
        line_search_fn="strong_wolfe",
    )
    started = time.perf_counter()
    best_loss = math.inf
    best_ei = float("nan")
    best_cb = float("nan")
    best_rmse = float("nan")
    evaluation = 0

    def closure() -> torch.Tensor:
        nonlocal best_loss, best_ei, best_cb, best_rmse, evaluation
        optimizer.zero_grad(set_to_none=True)
        ei = _bounded_log_parameter(
            raw_ei,
            settings.optimization.ei_min_n_m2,
            settings.optimization.ei_max_n_m2,
        )
        cb = _bounded_log_parameter(
            raw_cb,
            settings.optimization.cb_min_n_m2_s,
            settings.optimization.cb_max_n_m2_s,
        )
        loss, square = _rollout_loss(
            model,
            observations,
            initial_positions,
            initial_velocities,
            timestamps,
            ei=ei,
            cb=cb,
            robust_scale_m=settings.optimization.robust_scale_m,
            create_graph=True,
        )
        if not bool(torch.isfinite(loss).detach().cpu()):
            raise RuntimeError("Cable parameter optimization produced a non-finite loss.")
        loss.backward()
        loss_value = float(loss.detach().cpu())
        rmse = float(torch.sqrt(square).detach().cpu())
        ei_value = float(ei.detach().cpu())
        cb_value = float(cb.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_ei = ei_value
            best_cb = cb_value
            best_rmse = rmse
        evaluation += 1
        print(
            f"fit evaluation={evaluation}/{maximum_evaluations} "
            f"loss={loss_value:.6g} "
            f"curve_residual={1000.0 * rmse:.3f}mm "
            f"EI={ei_value:.6g}Nm2 "
            f"Cb={cb_value:.6g}Nm2s",
            flush=True,
        )
        return loss

    optimizer.step(closure)
    if not all(math.isfinite(value) for value in (best_ei, best_cb, best_rmse)):
        raise RuntimeError("Cable parameter optimization produced no finite result.")
    return FitResult(
        best_ei,
        best_cb,
        best_rmse,
        time.perf_counter() - started,
    )


def _median_positive_scale(values: Iterable[float], minimum: float) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float(minimum)
    return max(float(minimum), float(np.median(np.abs(array))))


def _endpoint_acceleration_scale(sequences: Sequence[PreparedSequence]) -> float:
    samples: list[float] = []
    for sequence in sequences:
        for run in _contiguous_runs(sequence.valid, sequence.timestamp_ns, 0.05):
            if len(run) < 3:
                continue
            time_s = sequence.timestamp_ns[run] * 1.0e-9
            velocity = sequence.observation_velocities_m_s[run][:, (0, -1)]
            dt = np.diff(time_s)
            acceleration = np.diff(velocity, axis=0) / dt[:, None, None]
            samples.extend(np.linalg.norm(acceleration, axis=2).ravel().tolist())
    return _median_positive_scale(samples, 1.0e-3)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(arguments: argparse.Namespace) -> Path:
    settings = load_settings(arguments.config)
    if getattr(arguments, "iterations", None) is not None:
        settings = replace(
            settings,
            optimization=replace(
                settings.optimization,
                optimizer_iterations=int(arguments.iterations),
            ),
        )
    if len(arguments.svo) != len(arguments.observation):
        raise ValueError("Each SVO must have one planar observation archive.")
    device = torch.device(settings.optimization.device)
    prepared: list[PreparedSequence] = []
    for svo_value, observation_value in zip(arguments.svo, arguments.observation, strict=True):
        svo = Path(svo_value).expanduser().resolve()
        observation = Path(observation_value).expanduser().resolve()
        sequence = load_planar_sequence(observation)
        _validate_recording(svo)
        item = _prepare_sequence(sequence, svo, settings, device)
        prepared.append(item)
        print(
            f"prepared {svo.name} "
            f"valid={np.count_nonzero(item.valid)}/{len(item.valid)} "
            f"DDER_initialization_shift={1000.0 * item.initialization_shift_m:.3f}mm",
            flush=True,
        )
    result = _fit(prepared, settings)
    curve_noise = _median_positive_scale((item.pixel_scale_m for item in prepared), 1.0e-5)
    initialization_shift = _median_positive_scale(
        (item.initialization_shift_m for item in prepared),
        0.0,
    )
    endpoint_acceleration = _endpoint_acceleration_scale(prepared)
    process_acceleration = max(
        1.0e-3,
        2.0 * result.train_curve_residual_m / settings.optimization.maximum_dt_s**2,
    )
    reference_sequence = prepared[0]
    payload = {
        "schema": MODEL_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cable_identity": int(load_planar_sequence(reference_sequence.observation_path).metadata["cable_identity"]),
        "measured": {
            "length_m": settings.cable.length_m,
            "mass_kg": settings.cable.mass_kg,
            "diameter_m": settings.cable.diameter_m,
            "node_count": settings.cable.node_count,
            "rest_shape": "straight",
            "torsion": "omitted",
        },
        "optimized": {
            "bending_stiffness_n_m2": result.bending_stiffness_n_m2,
            "bending_damping_n_m2_s": result.bending_damping_n_m2_s,
        },
        "solver": {
            "method": "explicit_forces_with_mass_weighted_rattle_projection",
            "substeps": settings.solver.substeps,
            "constraint_iterations": settings.solver.constraint_iterations,
        },
        "fit": {
            "status": "completed",
            "train_window_curve_residual_m": result.train_curve_residual_m,
            "observation_curve_residual_m": curve_noise,
            "initialization_constraint_shift_m": initialization_shift,
            "optimizer_iterations": settings.optimization.optimizer_iterations,
            "window_frames": settings.optimization.window_frames,
            "elapsed_s": result.elapsed_s,
        },
        "observation_model": {
            "method": PLANAR_OBSERVATION_METHOD,
            "scale_basis": "one_metric_image_pixel",
            "degrees_of_freedom": 4.0,
            "unresolved_curve_scale_m": curve_noise,
            "unresolved_endpoint_scale_m": curve_noise,
        },
        "process_model": {
            "interior_acceleration_sigma_m_s2": process_acceleration,
            "unobserved_endpoint_acceleration_sigma_m_s2": endpoint_acceleration,
        },
        "sources": [
            {
                "svo": str(item.svo_path),
                "observation": str(item.observation_path),
                "observation_sha256": sha256_file(item.observation_path),
            }
            for item in prepared
        ],
    }
    output = Path(arguments.output).expanduser().resolve()
    _atomic_json(output, payload)
    print(
        f"Saved planar cable model: {output} | "
        f"EI={result.bending_stiffness_n_m2:.6g}Nm2 "
        f"Cb={result.bending_damping_n_m2_s:.6g}Nm2s",
        flush=True,
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit EI and Cb from planar PIDNet trajectories.")
    parser.add_argument("--svo", nargs="+", type=Path, required=True)
    parser.add_argument("--observation", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--iterations", type=int, default=None)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
