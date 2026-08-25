"""Differentiable EI/Cb identification for a one-attached, free-tip cable."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Callable, Sequence
import uuid

import numpy as np
import torch

from cable_twin.shared.dder import (
    DderModel,
    DderParameters,
    DderState,
    START_PINNED_FREE_END,
)
from cable_twin.shared.observation_data import sha256_file

from .config import CableSpecification, OptitrackFitConfig
from .data import MotiveCableTake, load_motive_cable_csv


MODEL_SCHEMA = "optitrack_one_attached_free_rod_v1"
MAX_PARALLEL_INITIALIZER_ROLLOUTS = 512
ProgressCallback = Callable[[str], None]
CancellationCallback = Callable[[], bool]


class FitCancelled(RuntimeError):
    """Raised when the user requests a clean cooperative fit stop."""


def _check_cancelled(cancelled: CancellationCallback | None) -> None:
    if cancelled is not None and cancelled():
        raise FitCancelled("Fit stopped by user; no model was saved.")


@dataclass(frozen=True, slots=True)
class PreparedTake:
    take: MotiveCableTake
    positions_m: np.ndarray
    velocities_m_s: np.ndarray
    valid: np.ndarray
    quality_rejections: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class TakeWindowAudit:
    """Strict preflight result for one take before parameter fitting."""

    source_path: Path
    valid_frame_count: int
    candidate_window_count: int
    accepted_window_count: int
    initialization_rejected_count: int
    quality_rejections: tuple[tuple[str, int], ...]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class WindowBatch:
    observations_m: torch.Tensor
    initial_positions_m: torch.Tensor
    initial_velocities_m_s: torch.Tensor
    timestamps_s: torch.Tensor
    initialization_rmse_m: float
    marker_node_indices: torch.Tensor | None = None
    take_indices: torch.Tensor | None = None
    window_weights: torch.Tensor | None = None

    @property
    def count(self) -> int:
        return int(self.observations_m.shape[0])


@dataclass(frozen=True, slots=True)
class OptitrackFitResult:
    bending_stiffness_n_m2: float
    bending_damping_n_m2_s: float
    train_rmse_m: float
    validation_rmse_m: float | None
    train_window_count: int
    validation_window_count: int
    elapsed_s: float
    model_path: Path


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    bending_stiffness_n_m2: float
    bending_damping_n_m2_s: float
    objective: float
    rmse_m: float
    best_evaluation: int
    completed_evaluations: int
    history: tuple[tuple[int, float, float, float, float, float], ...] = ()
    initialization_history: tuple[tuple[int, float, float, float, float], ...] = ()


@dataclass(frozen=True, slots=True)
class OptimizationInitialization:
    bending_stiffness_n_m2: float
    bending_damping_n_m2_s: float
    method: str


def _contiguous_runs(
    valid: np.ndarray,
    timestamps_s: np.ndarray,
    maximum_dt_s: float,
) -> list[np.ndarray]:
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        return []
    runs: list[list[int]] = [[int(indices[0])]]
    for value in indices[1:]:
        index = int(value)
        previous = runs[-1][-1]
        dt = float(timestamps_s[index] - timestamps_s[previous])
        if index == previous + 1 and 0.0 < dt <= maximum_dt_s:
            runs[-1].append(index)
        else:
            runs.append([index])
    return [np.asarray(run, dtype=np.int64) for run in runs]


def _local_quadratic_velocity(
    positions_m: np.ndarray,
    timestamps_s: np.ndarray,
    complete: np.ndarray,
    *,
    window_frames: int,
    maximum_dt_s: float,
) -> np.ndarray:
    """Estimate initial velocity only; measured positions remain unchanged."""

    derivative = np.full_like(positions_m, np.nan, dtype=np.float64)
    half = window_frames // 2
    for run in _contiguous_runs(complete, timestamps_s, maximum_dt_s):
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
            relative_time = timestamps_s[selected] - timestamps_s[frame]
            design = np.column_stack(
                (np.ones(len(selected)), relative_time, np.square(relative_time))
            )
            target = positions_m[selected].reshape(len(selected), -1)
            coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
            derivative[frame] = coefficients[1].reshape(positions_m.shape[1:])
    return derivative


def _causal_quadratic_velocity(
    positions_m: np.ndarray,
    timestamps_s: np.ndarray,
    complete: np.ndarray,
    *,
    window_frames: int,
    maximum_dt_s: float,
) -> np.ndarray:
    """Estimate velocity from the current and preceding frames only."""

    derivative = np.full_like(positions_m, np.nan, dtype=np.float64)
    for run in _contiguous_runs(complete, timestamps_s, maximum_dt_s):
        for offset in range(2, len(run)):
            start = max(0, offset - window_frames + 1)
            selected = run[start : offset + 1]
            frame = int(run[offset])
            relative_time = timestamps_s[selected] - timestamps_s[frame]
            design = np.column_stack(
                (np.ones(len(selected)), relative_time, np.square(relative_time))
            )
            target = positions_m[selected].reshape(len(selected), -1)
            coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
            derivative[frame] = coefficients[1].reshape(positions_m.shape[1:])
    return derivative


def _make_model(
    config: OptitrackFitConfig,
    *,
    ei: float,
    cb: float,
) -> DderModel:
    cable, fit = config.cable, config.fit
    return DderModel(
        DderParameters(
            node_count=cable.node_count,
            cable_length_m=cable.length_m,
            cable_mass_kg=cable.total_dynamic_mass_kg,
            cable_diameter_m=cable.diameter_m,
            bending_stiffness_n_m2=ei,
            bending_damping_n_m2_s=cb,
            torsional_stiffness_n_m2=0.0,
            external_drag_s_inv=0.0,
            gravity_camera_m_s2=(0.0, 0.0, -9.80665),
            rest_lengths_m=cable.rod_rest_lengths_m,
            vertex_masses_kg=cable.vertex_masses_kg,
            substeps=fit.substeps,
            constraint_iterations=fit.constraint_iterations,
        )
    )


def marker_positions_to_rod(
    marker_positions_m: np.ndarray,
    cable: CableSpecification | OptitrackFitConfig,
) -> np.ndarray:
    """Interpolate measured material markers onto the DER material grid.

    A cubic Hermite curve in material coordinate supplies only the unobserved
    initial nodes.  Every measured marker remains exact before the subsequent
    inextensibility projection.
    """

    specification = cable.cable if isinstance(cable, OptitrackFitConfig) else cable
    markers = np.asarray(marker_positions_m, dtype=np.float64)
    if markers.ndim < 2 or markers.shape[-2:] != (specification.marker_count, 3):
        raise ValueError(
            f"marker_positions_m must end in {specification.marker_count}x3."
        )
    if not np.all(np.isfinite(markers)):
        raise ValueError("Marker positions must be finite before rod initialization.")
    return _interpolate_marker_field(markers, specification)


def marker_vectors_to_rod(
    marker_vectors: np.ndarray,
    cable: CableSpecification | OptitrackFitConfig,
) -> np.ndarray:
    """Interpolate measured marker vectors onto the DER material grid."""

    specification = cable.cable if isinstance(cable, OptitrackFitConfig) else cable
    values = np.asarray(marker_vectors, dtype=np.float64)
    if values.ndim < 2 or values.shape[-2:] != (specification.marker_count, 3):
        raise ValueError(f"marker_vectors must end in {specification.marker_count}x3.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Marker vectors must be finite before rod initialization.")
    return _interpolate_marker_field(values, specification)


def marker_positions_to_feasible_rod(
    marker_positions_m: np.ndarray,
    cable: CableSpecification | OptitrackFitConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an inextensible refined rod while preserving measured sites.

    Interior marker positions are adjusted only when measurement noise makes a
    chord longer than its material interval.  Latent nodes then form the
    smoothest locally seeded equal-edge circular arc through each conditioned
    marker pair.  The returned per-sample RMSE reports the complete observation
    correction; physical vertex mass is deliberately not used as observation
    confidence.
    """

    specification = cable.cable if isinstance(cable, OptitrackFitConfig) else cable
    if specification.rod_segments_per_marker_interval < 2:
        raise ValueError(
            "Marker-preserving identification requires at least two DER edges "
            "per measured interval."
        )
    markers = np.asarray(marker_positions_m, dtype=np.float64)
    if markers.ndim < 2 or markers.shape[-2:] != (specification.marker_count, 3):
        raise ValueError(
            f"marker_positions_m must end in {specification.marker_count}x3."
        )
    if not np.all(np.isfinite(markers)):
        raise ValueError("Marker positions must be finite before rod initialization.")

    original_shape = markers.shape
    conditioned = markers.reshape(-1, specification.marker_count, 3).copy()
    rest = np.asarray(specification.rest_lengths_m, dtype=np.float64)
    # Active-set Gauss-Newton projection onto adjacent-chord inequalities.
    # Only material site zero is a prescribed boundary.  Every measured moving
    # marker, including the distal tip, may receive the minimum correction
    # required to make noisy chords physically feasible.
    edge_count = specification.marker_count - 1
    diagonal = np.full((len(conditioned), edge_count), 2.0, dtype=np.float64)
    diagonal[:, 0] = 1.0
    edge_indices = np.arange(edge_count)
    for _ in range(32):
        difference = conditioned[:, 1:] - conditioned[:, :-1]
        distance = np.linalg.norm(difference, axis=2)
        excess = np.maximum(distance - rest[None], 0.0)
        largest = float(np.max(excess, initial=0.0))
        if largest <= 2.0e-9:
            break
        direction = difference / np.maximum(distance[..., None], 1.0e-15)
        active = excess > 1.0e-12
        normal = np.zeros(
            (len(conditioned), edge_count, edge_count), dtype=np.float64
        )
        normal[:, edge_indices, edge_indices] = np.where(active, diagonal, 1.0)
        coupling = -np.sum(direction[:, :-1] * direction[:, 1:], axis=2)
        coupling *= active[:, :-1] & active[:, 1:]
        normal[:, edge_indices[:-1], edge_indices[1:]] = coupling
        normal[:, edge_indices[1:], edge_indices[:-1]] = coupling
        multiplier = np.linalg.solve(normal, excess[..., None])[..., 0]
        correction = np.zeros_like(conditioned)
        correction[:, :-1] -= multiplier[..., None] * direction
        correction[:, 1:] += multiplier[..., None] * direction
        conditioned[:, 1:] += 0.8 * correction[:, 1:]
    else:
        raise RuntimeError("Measured marker chain conditioning did not converge.")

    conditioned_view = conditioned.reshape(original_shape)
    seed = _interpolate_marker_field(conditioned_view, specification).reshape(
        -1, specification.node_count, 3
    )
    rod = np.empty_like(seed)
    subdivision = specification.rod_segments_per_marker_interval
    gravity = np.asarray((0.0, 0.0, -1.0), dtype=np.float64)
    for interval, material_length in enumerate(rest):
        first = conditioned[:, interval]
        last = conditioned[:, interval + 1]
        chord = last - first
        chord_length = np.linalg.norm(chord, axis=1)
        if np.any(chord_length > material_length + 3.0e-9):
            raise RuntimeError("Conditioned marker chord remains longer than material length.")
        tangent = chord / np.maximum(chord_length[:, None], 1.0e-12)
        midpoint = 0.5 * (first + last)
        seed_mid = seed[:, interval * subdivision + subdivision // 2]
        normal = seed_mid - midpoint
        normal -= tangent * np.sum(normal * tangent, axis=1, keepdims=True)
        normal_length = np.linalg.norm(normal, axis=1)
        fallback = gravity[None] - tangent * np.sum(
            gravity[None] * tangent, axis=1, keepdims=True
        )
        fallback_length = np.linalg.norm(fallback, axis=1)
        second_axis = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        second = second_axis[None] - tangent * np.sum(
            second_axis[None] * tangent, axis=1, keepdims=True
        )
        use_second = fallback_length <= 1.0e-9
        fallback[use_second] = second[use_second]
        fallback_length = np.linalg.norm(fallback, axis=1)
        use_fallback = normal_length <= 1.0e-9
        normal[use_fallback] = fallback[use_fallback]
        normal_length = np.linalg.norm(normal, axis=1)
        normal /= np.maximum(normal_length[:, None], 1.0e-12)

        edge_length = material_length / subdivision
        low = np.zeros_like(chord_length)
        high = np.full_like(chord_length, 2.0 * math.pi - 1.0e-7)
        straight = np.abs(chord_length - material_length) <= 1.0e-10
        for _ in range(56):
            angle = 0.5 * (low + high)
            ratio = np.sin(angle / (2.0 * subdivision)) / np.maximum(
                np.sin(angle / 2.0), 1.0e-15
            )
            value = chord_length * ratio - edge_length
            high = np.where(value > 0.0, angle, high)
            low = np.where(value > 0.0, low, angle)
        total_angle = np.where(straight, 0.0, 0.5 * (low + high))
        small = total_angle <= 1.0e-8
        radius = np.empty_like(total_angle)
        radius[small] = np.inf
        radius[~small] = edge_length / (
            2.0 * np.sin(total_angle[~small] / (2.0 * subdivision))
        )
        center_offset = np.zeros_like(total_angle)
        center_offset[~small] = radius[~small] * np.cos(total_angle[~small] / 2.0)
        for local in range(subdivision + 1):
            node = interval * subdivision + local
            if local == 0:
                rod[:, node] = first
            elif local == subdivision:
                rod[:, node] = last
            else:
                phase = -0.5 * total_angle + local * total_angle / subdivision
                safe_radius = np.where(small, 0.0, radius)
                curved = (
                    midpoint
                    - center_offset[:, None] * normal
                    + safe_radius[:, None]
                    * (
                        np.sin(phase)[:, None] * tangent
                        + np.cos(phase)[:, None] * normal
                    )
                )
                linear = first + (local / subdivision) * chord
                rod[:, node] = np.where(small[:, None], linear, curved)

    correction = conditioned - markers.reshape(conditioned.shape)
    rmse = np.sqrt(np.mean(np.sum(np.square(correction), axis=2), axis=1))
    return rod.reshape(original_shape[:-2] + (specification.node_count, 3)), rmse.reshape(
        original_shape[:-2]
    )


def _measurement_quality_mask(
    take: MotiveCableTake,
    config: OptitrackFitConfig,
    complete: np.ndarray,
) -> tuple[np.ndarray, tuple[tuple[str, int], ...]]:
    """Reject corrupt measurements and split trajectories at impossible jumps."""

    fit = config.fit
    quality = complete.copy()
    rejected: dict[str, np.ndarray] = {}

    rigid_error = take.attachment_error_m > fit.maximum_rigid_body_error_m
    rejected["rigid_body_error"] = complete & rigid_error

    chords = np.linalg.vector_norm(np.diff(take.positions_m, axis=1), axis=2)
    chord_excess = np.any(
        chords
        > np.asarray(config.cable.rest_lengths_m, dtype=np.float64)[None]
        + fit.maximum_chord_excess_m,
        axis=1,
    )
    rejected["overlong_marker_chord"] = complete & chord_excess

    dt = np.diff(take.timestamps_s)
    adjacent_complete = complete[:-1] & complete[1:]
    valid_dt = (dt > 0.0) & (dt <= 1.5 / take.export_rate_hz)
    transition = adjacent_complete & valid_dt
    safe_dt = np.maximum(dt, np.finfo(np.float64).tiny)

    marker_speed = np.linalg.vector_norm(
        np.diff(take.positions_m, axis=0), axis=2
    ) / safe_dt[:, None]
    bad_marker_transition = transition & np.any(
        marker_speed > fit.maximum_marker_speed_m_s,
        axis=1,
    )
    attachment_speed = marker_speed[:, 0]
    bad_attachment_transition = transition & (
        attachment_speed > fit.maximum_attachment_speed_m_s
    )

    def transition_frames(values: np.ndarray) -> np.ndarray:
        frames = np.zeros(take.frame_count, dtype=bool)
        indices = np.flatnonzero(values)
        frames[indices] = True
        frames[indices + 1] = True
        return frames

    rejected["marker_speed"] = transition_frames(bad_marker_transition)
    rejected["attachment_speed"] = transition_frames(bad_attachment_transition)
    for frames in rejected.values():
        quality &= ~frames
    counts = tuple(
        (name, int(np.count_nonzero(frames)))
        for name, frames in rejected.items()
    )
    return quality, counts


def _interpolate_marker_field(
    marker_values: np.ndarray,
    cable: CableSpecification,
) -> np.ndarray:
    """C1 material-coordinate interpolation shared by position and velocity."""

    values = np.asarray(marker_values, dtype=np.float64)
    marker_s = np.asarray(
        cable.marker_material_coordinates_m,
        dtype=np.float64,
    )
    subdivision = cable.rod_segments_per_marker_interval
    tangents = np.empty_like(values)
    tangents[..., 0, :] = (
        values[..., 1, :] - values[..., 0, :]
    ) / (marker_s[1] - marker_s[0])
    tangents[..., -1, :] = (
        values[..., -1, :] - values[..., -2, :]
    ) / (marker_s[-1] - marker_s[-2])
    tangents[..., 1:-1, :] = (
        values[..., 2:, :] - values[..., :-2, :]
    ) / (marker_s[2:] - marker_s[:-2])[..., None]

    rod = np.empty(values.shape[:-2] + (cable.node_count, 3), dtype=np.float64)
    for interval, length in enumerate(cable.rest_lengths_m):
        p0 = values[..., interval, :]
        p1 = values[..., interval + 1, :]
        m0 = tangents[..., interval, :]
        m1 = tangents[..., interval + 1, :]
        for local in range(subdivision):
            alpha = local / subdivision
            alpha2 = alpha * alpha
            alpha3 = alpha2 * alpha
            h00 = 2.0 * alpha3 - 3.0 * alpha2 + 1.0
            h10 = alpha3 - 2.0 * alpha2 + alpha
            h01 = -2.0 * alpha3 + 3.0 * alpha2
            h11 = alpha3 - alpha2
            node = interval * subdivision + local
            rod[..., node, :] = (
                h00 * p0
                + h10 * length * m0
                + h01 * p1
                + h11 * length * m1
            )
    rod[..., -1, :] = values[..., -1, :]
    return rod


def prepare_take(
    path: str | Path,
    config: OptitrackFitConfig,
    *,
    causal_velocity: bool = False,
) -> PreparedTake:
    take = load_motive_cable_csv(path)
    if not take.has_attachment:
        raise ValueError(f"{take.source_path.name} has no rigid attachment pivot.")
    if take.marker_count != config.cable.marker_count:
        raise ValueError(
            f"{take.source_path.name} has {take.marker_count} measured cable sites; "
            f"the fitted cable requires {config.cable.marker_count} "
            "(one rigid attachment plus c1-c10)."
        )
    complete = np.all(take.observed, axis=1) & take.attachment_observed
    quality, quality_rejections = _measurement_quality_mask(take, config, complete)
    usable = complete & quality
    maximum_dt_s = 1.5 / take.export_rate_hz
    velocity_estimator = (
        _causal_quadratic_velocity if causal_velocity else _local_quadratic_velocity
    )
    velocities = velocity_estimator(
        take.positions_m,
        take.timestamps_s,
        usable,
        window_frames=config.fit.velocity_window_frames,
        maximum_dt_s=maximum_dt_s,
    )
    valid = (
        usable
        & np.all(np.isfinite(velocities), axis=(1, 2))
    )
    if not any(
        len(run) >= config.fit.window_frames
        for run in _contiguous_runs(valid, take.timestamps_s, maximum_dt_s)
    ):
        raise ValueError(
            f"{take.source_path.name} has no complete, physically feasible run of "
            f"{config.fit.window_frames} frames."
        )
    valid.setflags(write=False)
    velocities.setflags(write=False)
    return PreparedTake(
        take=take,
        positions_m=take.positions_m,
        velocities_m_s=velocities,
        valid=valid,
        quality_rejections=quality_rejections,
    )


def _window_arrays(
    prepared: Sequence[PreparedTake],
    config: OptitrackFitConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    observations: list[np.ndarray] = []
    initial_positions: list[np.ndarray] = []
    initial_velocities: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    take_indices: list[int] = []
    window = config.fit.window_frames
    stride = config.fit.window_stride
    for take_index, item in enumerate(prepared):
        maximum_dt_s = 1.5 / item.take.export_rate_hz
        for run in _contiguous_runs(item.valid, item.take.timestamps_s, maximum_dt_s):
            for offset in range(0, len(run) - window + 1, stride):
                selected = run[offset : offset + window]
                observations.append(item.positions_m[selected])
                initial_positions.append(item.positions_m[selected[0]])
                initial_velocities.append(item.velocities_m_s[selected[0]])
                timestamps.append(item.take.timestamps_s[selected])
                take_indices.append(take_index)
    if not observations:
        raise ValueError("The selected takes contain no valid fitting windows.")
    return (
        np.asarray(observations),
        np.asarray(initial_positions),
        np.asarray(initial_velocities),
        np.asarray(timestamps),
        np.asarray(take_indices, dtype=np.int64),
    )


def _make_windows(
    prepared: Sequence[PreparedTake],
    config: OptitrackFitConfig,
    model: DderModel,
    device: torch.device,
) -> WindowBatch:
    (
        observation_array,
        position_array,
        velocity_array,
        timestamp_array,
        take_index_array,
    ) = (
        _window_arrays(prepared, config)
    )
    feasible_start, correction_rmse = marker_positions_to_feasible_rod(
        position_array, config
    )
    accepted = correction_rmse <= config.fit.maximum_initialization_rmse_m
    if not np.any(accepted):
        raise ValueError(
            "Every fitting window exceeds the marker-conditioning limit of "
            f"{1000.0 * config.fit.maximum_initialization_rmse_m:.3f} mm."
        )
    if not np.all(accepted):
        observation_array = observation_array[accepted]
        position_array = position_array[accepted]
        velocity_array = velocity_array[accepted]
        timestamp_array = timestamp_array[accepted]
        take_index_array = take_index_array[accepted]
        feasible_start = feasible_start[accepted]
        correction_rmse = correction_rmse[accepted]

    observations = torch.as_tensor(observation_array, dtype=torch.float64, device=device)
    projected = torch.as_tensor(
        feasible_start,
        dtype=torch.float64,
        device=device,
    )
    maximum_error = float(model.maximum_segment_error_m(projected).max().detach().cpu())
    if maximum_error > 1.0e-8:
        raise RuntimeError(
            "Marker-conditioned initialization did not satisfy cable lengths "
            f"(maximum residual {maximum_error:g} m)."
        )
    velocity = torch.as_tensor(
        marker_vectors_to_rod(velocity_array, config),
        dtype=torch.float64,
        device=device,
    )
    velocity = model.project_velocities(
        projected,
        velocity,
        velocity[:, :1],
        pinned_endpoints=START_PINNED_FREE_END,
    )
    initialization_rmse = float(np.sqrt(np.mean(np.square(correction_rmse))))
    take_indices = torch.as_tensor(take_index_array, dtype=torch.long, device=device)
    window_weight_array = np.full(
        len(take_index_array),
        1.0 / len(take_index_array),
        dtype=np.float64,
    )
    return WindowBatch(
        observations_m=observations,
        initial_positions_m=projected,
        initial_velocities_m_s=velocity,
        timestamps_s=torch.as_tensor(timestamp_array, dtype=torch.float64, device=device),
        initialization_rmse_m=initialization_rmse,
        marker_node_indices=torch.tensor(
            config.cable.marker_node_indices,
            dtype=torch.long,
            device=device,
        ),
        take_indices=take_indices,
        window_weights=torch.as_tensor(
            window_weight_array,
            dtype=torch.float64,
            device=device,
        ),
    )


def audit_optitrack_takes(
    take_roles: Sequence[tuple[str | Path, str]],
    config: OptitrackFitConfig,
    *,
    progress: ProgressCallback | None = None,
) -> tuple[TakeWindowAudit, ...]:
    """Apply the fitting preflight to every take without fitting a model.

    The audit deliberately uses only measurement integrity, physical feasibility,
    continuity and initialization consistency.  It never selects windows using a
    fitted residual, which would bias the identification data.
    """

    report = progress if progress is not None else (lambda _text: None)
    audits: list[TakeWindowAudit] = []
    for path_value, role in take_roles:
        path = Path(path_value).expanduser().resolve()
        try:
            prepared = prepare_take(
                path,
                config,
                causal_velocity=role == "validation",
            )
            arrays = _window_arrays((prepared,), config)
            candidate_count = int(arrays[1].shape[0])
            _feasible, correction_rmse = marker_positions_to_feasible_rod(
                arrays[1], config
            )
            accepted = correction_rmse <= config.fit.maximum_initialization_rmse_m
            accepted_count = int(np.count_nonzero(accepted))
            audit = TakeWindowAudit(
                source_path=prepared.take.source_path.resolve(),
                valid_frame_count=int(np.count_nonzero(prepared.valid)),
                candidate_window_count=candidate_count,
                accepted_window_count=accepted_count,
                initialization_rejected_count=candidate_count - accepted_count,
                quality_rejections=prepared.quality_rejections,
            )
        except (OSError, RuntimeError, ValueError) as error:
            audit = TakeWindowAudit(
                source_path=path,
                valid_frame_count=0,
                candidate_window_count=0,
                accepted_window_count=0,
                initialization_rejected_count=0,
                quality_rejections=(),
                error=str(error),
            )
        audits.append(audit)
        if audit.error is None:
            rejected = ", ".join(
                f"{name}={count}"
                for name, count in audit.quality_rejections
                if count
            )
            report(
                f"{role} {path.name}: clean windows="
                f"{audit.accepted_window_count}/{audit.candidate_window_count}, "
                f"initialization rejected={audit.initialization_rejected_count}"
                + (f"; {rejected}" if rejected else "")
            )
        else:
            report(f"{role} {path.name}: REJECTED - {audit.error}")
    return tuple(audits)


def _rollout_error_fields(
    model: DderModel,
    batch: WindowBatch,
    *,
    ei: torch.Tensor | float,
    cb: torch.Tensor | float,
    robust_scale_m: float,
    create_graph: bool,
    cancelled: CancellationCallback | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_cancelled(cancelled)
    state = DderState(batch.initial_positions_m, batch.initial_velocities_m_s)
    robust_values: list[torch.Tensor] = []
    squared_values: list[torch.Tensor] = []
    observations = batch.observations_m
    marker_node_indices = batch.marker_node_indices
    if marker_node_indices is None:
        if observations.shape[2] != state.positions_m.shape[1]:
            raise ValueError(
                "WindowBatch must provide marker_node_indices when observations "
                "and DER state use different spatial discretizations."
            )
        marker_node_indices = torch.arange(
            observations.shape[2],
            dtype=torch.long,
            device=observations.device,
        )
    for frame in range(1, observations.shape[1]):
        _check_cancelled(cancelled)
        dt = batch.timestamps_s[:, frame] - batch.timestamps_s[:, frame - 1]
        state = model.step(
            state,
            observations[:, frame, :1],
            dt,
            create_graph=create_graph,
            bending_stiffness_n_m2=ei,
            bending_damping_n_m2_s=cb,
            _validate=False,
            _dense_constraint_solve=True,
            pinned_endpoints=START_PINNED_FREE_END,
        )
        predicted_markers = torch.index_select(
            state.positions_m,
            1,
            marker_node_indices,
        )
        distance = torch.linalg.vector_norm(
            predicted_markers[:, 1:] - observations[:, frame, 1:], dim=-1
        )
        normalized = distance / robust_scale_m
        robust_values.append(torch.sqrt(1.0 + normalized.square()) - 1.0)
        squared_values.append(
            distance.square() if not create_graph else distance.detach().square()
        )
    return torch.stack(robust_values, dim=1), torch.stack(squared_values, dim=1)


def _rollout_metrics(
    model: DderModel,
    batch: WindowBatch,
    *,
    ei: torch.Tensor | float,
    cb: torch.Tensor | float,
    robust_scale_m: float,
    create_graph: bool,
    cancelled: CancellationCallback | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    robust, squared = _rollout_error_fields(
        model,
        batch,
        ei=ei,
        cb=cb,
        robust_scale_m=robust_scale_m,
        create_graph=create_graph,
        cancelled=cancelled,
    )
    robust_per_window = robust.mean(dim=(1, 2))
    squared_per_window = squared.mean(dim=(1, 2))
    if batch.window_weights is None:
        return robust_per_window.mean(), squared_per_window.mean()
    weights = batch.window_weights / batch.window_weights.sum()
    return (
        torch.sum(weights * robust_per_window),
        torch.sum(weights * squared_per_window),
    )


def _per_take_rmse(
    model: DderModel,
    batch: WindowBatch,
    *,
    ei: float,
    cb: float,
    robust_scale_m: float,
    cancelled: CancellationCallback | None = None,
) -> list[float]:
    """Evaluate each take separately without changing fitted parameters."""

    _robust, squared = _rollout_error_fields(
        model,
        batch,
        ei=ei,
        cb=cb,
        robust_scale_m=robust_scale_m,
        create_graph=False,
        cancelled=cancelled,
    )
    if batch.take_indices is None:
        return [float(torch.sqrt(squared.mean()).detach().cpu())]
    values: list[float] = []
    take_count = int(torch.max(batch.take_indices).detach().cpu()) + 1
    for take_index in range(take_count):
        selected = batch.take_indices == take_index
        if not bool(torch.any(selected).detach().cpu()):
            values.append(math.nan)
            continue
        values.append(
            float(torch.sqrt(squared[selected].mean()).detach().cpu())
        )
    return values


def _local_identifiability_profile(
    model: DderModel,
    batch: WindowBatch,
    config: OptitrackFitConfig,
    *,
    ei: float,
    cb: float,
    base_objective: float,
    cancelled: CancellationCallback | None = None,
) -> dict[str, dict[str, float]]:
    """Profile each log parameter locally while holding the others fixed."""

    specifications = {
        "bending_stiffness_n_m2": (ei, config.fit.ei_min_n_m2, config.fit.ei_max_n_m2),
        "bending_damping_n_m2_s": (
            cb,
            config.fit.cb_min_n_m2_s,
            config.fit.cb_max_n_m2_s,
        ),
    }
    result: dict[str, dict[str, float]] = {}
    for name, (value, minimum, maximum) in specifications.items():
        _check_cancelled(cancelled)
        samples: dict[str, float] = {}
        for direction, factor in (("lower", math.exp(-0.1)), ("upper", math.exp(0.1))):
            candidate = min(max(value * factor, minimum), maximum)
            parameters = {
                "ei": ei,
                "cb": cb,
            }
            parameters[
                {
                    "bending_stiffness_n_m2": "ei",
                    "bending_damping_n_m2_s": "cb",
                }[name]
            ] = candidate
            objective, _squared = _rollout_metrics(
                model,
                batch,
                ei=parameters["ei"],
                cb=parameters["cb"],
                robust_scale_m=config.fit.robust_scale_m,
                create_graph=False,
                cancelled=cancelled,
            )
            samples[f"{direction}_value"] = candidate
            samples[f"{direction}_objective"] = float(objective.detach().cpu())
        samples["selected_value"] = value
        samples["selected_objective"] = base_objective
        samples["minimum_relative_objective_change"] = min(
            abs(samples["lower_objective"] - base_objective),
            abs(samples["upper_objective"] - base_objective),
        ) / max(abs(base_objective), 1.0e-15)
        result[name] = samples
    return result


def _differentiable_fit(
    model: DderModel,
    batch: WindowBatch,
    config: OptitrackFitConfig,
    report: ProgressCallback,
    initialization: OptimizationInitialization,
    cancelled: CancellationCallback | None = None,
) -> OptimizationResult:
    """Fit bounded log-parameters with deterministic search and batched Adam."""

    fit = config.fit
    device = batch.observations_m.device
    log_lower = torch.tensor(
        (
            math.log(fit.ei_min_n_m2),
            math.log(fit.cb_min_n_m2_s),
        ),
        dtype=torch.float64,
        device=device,
    )
    log_upper = torch.tensor(
        (
            math.log(fit.ei_max_n_m2),
            math.log(fit.cb_max_n_m2_s),
        ),
        dtype=torch.float64,
        device=device,
    )
    log_span = log_upper - log_lower

    def select_windows(source: WindowBatch, indices: torch.Tensor) -> WindowBatch:
        weights = None
        if source.window_weights is not None:
            weights = torch.index_select(source.window_weights, 0, indices)
            weights = weights / weights.sum()
        return WindowBatch(
            observations_m=torch.index_select(source.observations_m, 0, indices),
            initial_positions_m=torch.index_select(
                source.initial_positions_m, 0, indices
            ),
            initial_velocities_m_s=torch.index_select(
                source.initial_velocities_m_s, 0, indices
            ),
            timestamps_s=torch.index_select(source.timestamps_s, 0, indices),
            initialization_rmse_m=source.initialization_rmse_m,
            marker_node_indices=source.marker_node_indices,
            take_indices=(
                None
                if source.take_indices is None
                else torch.index_select(source.take_indices, 0, indices)
            ),
            window_weights=weights,
        )

    def repeat_windows(source: WindowBatch, copies: int) -> WindowBatch:
        def repeated(value: torch.Tensor) -> torch.Tensor:
            return value.repeat((copies,) + (1,) * (value.ndim - 1))

        return WindowBatch(
            observations_m=repeated(source.observations_m),
            initial_positions_m=repeated(source.initial_positions_m),
            initial_velocities_m_s=repeated(source.initial_velocities_m_s),
            timestamps_s=repeated(source.timestamps_s),
            initialization_rmse_m=source.initialization_rmse_m,
            marker_node_indices=source.marker_node_indices,
            take_indices=(
                None
                if source.take_indices is None
                else source.take_indices.repeat(copies)
            ),
            window_weights=(
                None
                if source.window_weights is None
                else source.window_weights.repeat(copies)
            ),
        )

    def initializer_indices() -> torch.Tensor:
        count = min(fit.initializer_windows, batch.count)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(fit.optimizer_seed)
        if batch.take_indices is None:
            order = torch.randperm(batch.count, generator=generator)
            return order[:count].to(device=device)
        take_indices_cpu = batch.take_indices.detach().cpu()
        groups: list[list[int]] = []
        for take_index in torch.unique(take_indices_cpu, sorted=True).tolist():
            members = torch.nonzero(
                take_indices_cpu == take_index,
                as_tuple=False,
            ).flatten().tolist()
            permutation = torch.randperm(len(members), generator=generator).tolist()
            groups.append([members[index] for index in permutation])
        selected: list[int] = []
        depth = 0
        while len(selected) < count:
            added = False
            for group in groups:
                if depth < len(group):
                    selected.append(group[depth])
                    added = True
                    if len(selected) == count:
                        break
            if not added:
                break
            depth += 1
        return torch.tensor(selected, dtype=torch.long, device=device)

    initial_log = torch.tensor(
        (
            math.log(initialization.bending_stiffness_n_m2),
            math.log(initialization.bending_damping_n_m2_s),
        ),
        dtype=torch.float64,
        device=device,
    )
    midpoint_coordinate = torch.clamp(
        (initial_log - log_lower) / log_span, min=0.0, max=1.0
    )
    sobol = torch.quasirandom.SobolEngine(
        dimension=2,
        scramble=True,
        seed=fit.optimizer_seed,
    )
    candidate_coordinates = torch.cat(
        (
            midpoint_coordinate[None],
            sobol.draw(fit.initializer_candidates - 1).to(
                dtype=torch.float64, device=device
            ),
        ),
        dim=0,
    )
    candidate_parameters = torch.exp(
        log_lower[None] + candidate_coordinates * log_span[None]
    )
    initializer_batch = select_windows(batch, initializer_indices())
    initializer_history: list[tuple[int, float, float, float, float]] = []
    best_candidate = 0
    best_candidate_objective = math.inf
    report(
        "deterministic bounded initialization: "
        f"{fit.initializer_candidates} log-space candidates on "
        f"{initializer_batch.count} balanced training windows"
    )
    # Forward-only candidates share one CUDA batch.  The state is small and no
    # autograd graph is retained, so this uses substantially less time than
    # launching one sequential 100-frame rollout per candidate.
    # Forward-only search is cheap enough to vectorize, but the UI permits
    # large candidate/window counts.  Bound the number of simultaneous
    # float64 trajectories so an exploratory setting cannot exhaust the GPU.
    candidate_chunk = max(
        1,
        min(
            fit.initializer_candidates,
            MAX_PARALLEL_INITIALIZER_ROLLOUTS // initializer_batch.count,
        ),
    )
    for first in range(0, fit.initializer_candidates, candidate_chunk):
        _check_cancelled(cancelled)
        last = min(first + candidate_chunk, fit.initializer_candidates)
        parameters = candidate_parameters[first:last]
        copies = last - first
        repeated_batch = repeat_windows(initializer_batch, copies)
        repeated_parameters = parameters.repeat_interleave(
            initializer_batch.count, dim=0
        )
        started = time.perf_counter()
        with torch.no_grad():
            robust, squared = _rollout_error_fields(
                model,
                repeated_batch,
                ei=repeated_parameters[:, 0],
                cb=repeated_parameters[:, 1],
                robust_scale_m=fit.robust_scale_m,
                create_graph=False,
                cancelled=cancelled,
            )
            robust_per_window = robust.mean(dim=(1, 2)).reshape(
                copies, initializer_batch.count
            )
            squared_per_window = squared.mean(dim=(1, 2)).reshape(
                copies, initializer_batch.count
            )
            if initializer_batch.window_weights is None:
                weights = torch.full(
                    (initializer_batch.count,),
                    1.0 / initializer_batch.count,
                    dtype=torch.float64,
                    device=device,
                )
            else:
                weights = initializer_batch.window_weights
                weights = weights / weights.sum()
            objectives = torch.sum(robust_per_window * weights[None], dim=1)
            rmses = torch.sqrt(torch.sum(squared_per_window * weights[None], dim=1))
        for local in range(copies):
            candidate = first + local
            values = parameters[local].detach().cpu().tolist()
            objective = float(objectives[local].detach().cpu())
            rmse = float(rmses[local].detach().cpu())
            ei, cb = (float(value) for value in values)
            initializer_history.append((candidate + 1, objective, rmse, ei, cb))
            if math.isfinite(objective) and objective < best_candidate_objective:
                best_candidate_objective = objective
                best_candidate = candidate
            report(
                f"candidate {candidate + 1}/{fit.initializer_candidates}: "
                f"objective={objective:.6g}  error={1000.0 * rmse:.3f} mm  "
                f"EI={ei:.6g}  Cb={cb:.6g}"
            )
        report(
            f"candidate batch {first // candidate_chunk + 1}: "
            f"time={time.perf_counter() - started:.1f}s"
        )
    if not math.isfinite(best_candidate_objective):
        raise RuntimeError("Bounded initialization produced no finite physical model.")

    coordinate = torch.nn.Parameter(candidate_coordinates[best_candidate].clone())

    def physical_parameters() -> torch.Tensor:
        return torch.exp(log_lower + torch.clamp(coordinate, 0.0, 1.0) * log_span)

    optimizer = torch.optim.Adam(
        (coordinate,),
        lr=fit.optimizer_learning_rate,
        betas=(0.9, 0.99),
    )
    history: list[tuple[int, float, float, float, float, float]] = []
    best = OptimizationResult(math.nan, math.nan, math.inf, math.inf, 0, 0)
    full_evaluation_count = 0

    def full_evaluation(
        update: int,
        gradient_norm: float,
        *,
        parameter_values: torch.Tensor | None = None,
        objective_value: torch.Tensor | None = None,
        squared_value: torch.Tensor | None = None,
    ) -> None:
        nonlocal best, full_evaluation_count
        _check_cancelled(cancelled)
        started = time.perf_counter()
        current_parameters = (
            physical_parameters().detach()
            if parameter_values is None
            else parameter_values.detach()
        )
        values = current_parameters.cpu().tolist()
        ei, cb = (float(value) for value in values)
        if objective_value is None or squared_value is None:
            with torch.no_grad():
                objective_tensor, squared = _rollout_metrics(
                    model,
                    batch,
                    ei=ei,
                    cb=cb,
                    robust_scale_m=fit.robust_scale_m,
                    create_graph=False,
                    cancelled=cancelled,
                )
        else:
            objective_tensor = objective_value.detach()
            squared = squared_value.detach()
        objective = float(objective_tensor.detach().cpu())
        rmse = float(torch.sqrt(squared).detach().cpu())
        full_evaluation_count += 1
        history.append((update, objective, rmse, ei, cb, gradient_norm))
        if math.isfinite(objective) and objective < best.objective:
            best = OptimizationResult(
                ei,
                cb,
                objective,
                rmse,
                update,
                update,
            )
        report(
            f"full evaluation {full_evaluation_count}: update={update}  "
            f"objective={objective:.6g}  error={1000.0 * rmse:.3f} mm  "
            f"EI={ei:.6g} N m^2  Cb={cb:.6g} N m^2 s  "
            f"time={time.perf_counter() - started:.1f}s"
        )

    report(
        "projected CUDA-batched Adam refinement: "
        f"{fit.optimizer_iterations} updates, {fit.optimizer_batch_windows} "
        "windows/update; saved parameters are selected only by full-training evaluations"
    )
    uses_full_training_batch = fit.optimizer_batch_windows >= batch.count
    if not uses_full_training_batch:
        full_evaluation(0, math.nan)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(fit.optimizer_seed + 1)
    order = torch.randperm(batch.count, generator=generator)
    cursor = 0
    last_full_update = 0
    last_gradient_norm = math.nan
    for update in range(1, fit.optimizer_iterations + 1):
        _check_cancelled(cancelled)
        if cursor == batch.count:
            order = torch.randperm(batch.count, generator=generator)
            cursor = 0
        end = min(cursor + fit.optimizer_batch_windows, batch.count)
        indices = order[cursor:end].to(device=device)
        mini_batch = select_windows(batch, indices)
        completes_pass = end == batch.count
        cursor = end
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        parameters = physical_parameters()
        objective, squared = _rollout_metrics(
            model,
            mini_batch,
            ei=parameters[0],
            cb=parameters[1],
            robust_scale_m=fit.robust_scale_m,
            create_graph=True,
            cancelled=cancelled,
        )
        if not bool(torch.isfinite(objective).detach().cpu()):
            raise RuntimeError("A differentiable mini-batch rollout became non-finite.")
        objective.backward()
        _check_cancelled(cancelled)
        if coordinate.grad is None or not bool(
            torch.all(torch.isfinite(coordinate.grad)).detach().cpu()
        ):
            raise RuntimeError(
                "The physical rollout produced a non-finite EI/Cb gradient."
            )
        unclipped_norm = torch.linalg.vector_norm(coordinate.grad)
        last_gradient_norm = float(unclipped_norm.detach().cpu())
        if uses_full_training_batch:
            # This differentiated objective already is the exact full-Training
            # score at the pre-update parameter vector; do not launch a second
            # identical forward pass merely for model selection.
            full_evaluation(
                update - 1,
                last_gradient_norm,
                parameter_values=parameters,
                objective_value=objective,
                squared_value=squared,
            )
            last_full_update = update - 1
        torch.nn.utils.clip_grad_norm_((coordinate,), fit.optimizer_gradient_clip)
        optimizer.step()
        with torch.no_grad():
            coordinate.clamp_(0.0, 1.0)
        values = physical_parameters().detach().cpu().tolist()
        ei, cb = (float(value) for value in values)
        report(
            f"update {update}/{fit.optimizer_iterations}: "
            f"batch objective={float(objective.detach().cpu()):.6g}  "
            f"error={1000.0 * float(torch.sqrt(squared).detach().cpu()):.3f} mm  "
            f"EI={ei:.6g}  Cb={cb:.6g}  "
            f"|grad|={last_gradient_norm:.3g}  "
            f"time={time.perf_counter() - started:.1f}s"
        )
        if completes_pass and not uses_full_training_batch:
            full_evaluation(update, last_gradient_norm)
            last_full_update = update
    if last_full_update != fit.optimizer_iterations:
        full_evaluation(fit.optimizer_iterations, last_gradient_norm)

    if not all(
        math.isfinite(value)
        for value in (
            best.bending_stiffness_n_m2,
            best.bending_damping_n_m2_s,
            best.objective,
            best.rmse_m,
        )
    ):
        raise RuntimeError("Differentiable optimization produced no finite model.")
    return OptimizationResult(
        best.bending_stiffness_n_m2,
        best.bending_damping_n_m2_s,
        best.objective,
        best.rmse_m,
        best.best_evaluation,
        fit.optimizer_iterations,
        tuple(history),
        tuple(initializer_history),
    )


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _bound_status(value: float, minimum: float, maximum: float) -> str:
    if value <= minimum * (1.0 + 1.0e-9):
        return "lower"
    if value >= maximum * (1.0 - 1.0e-9):
        return "upper"
    return "interior"


def _parameter_initialization(
    config: OptitrackFitConfig,
) -> OptimizationInitialization:
    """Return the deterministic cold-start point for a clean independent fit."""

    fit = config.fit
    return OptimizationInitialization(
        math.sqrt(fit.ei_min_n_m2 * fit.ei_max_n_m2),
        math.sqrt(fit.cb_min_n_m2_s * fit.cb_max_n_m2_s),
        "cold start at geometric midpoint of fixed logarithmic parameter bounds",
    )


def fit_optitrack_takes(
    training_paths: Sequence[str | Path],
    validation_paths: Sequence[str | Path],
    config: OptitrackFitConfig,
    *,
    progress: ProgressCallback | None = None,
    cancelled: CancellationCallback | None = None,
) -> OptitrackFitResult:
    """Fit all Training takes and report independent fit-window Validation error."""

    report = progress if progress is not None else (lambda _text: None)
    _check_cancelled(cancelled)
    if not training_paths:
        raise ValueError("Select at least one training take.")
    role_sources = {
        "training": [Path(path).expanduser().resolve() for path in training_paths],
        "validation": [Path(path).expanduser().resolve() for path in validation_paths],
    }
    hashes_by_role = {
        role: [sha256_file(path) for path in paths]
        for role, paths in role_sources.items()
    }
    all_hashes = [
        (role, digest)
        for role, digests in hashes_by_role.items()
        for digest in digests
    ]
    if len({digest for _role, digest in all_hashes}) != len(all_hashes):
        duplicates = sorted(
            digest
            for digest in {value for _role, value in all_hashes}
            if sum(value == candidate for _role, candidate in all_hashes) > 1
        )
        raise ValueError(
            "Training and Validation inputs must be content-disjoint; "
            f"duplicate SHA-256: {', '.join(duplicates)}"
        )
    device = torch.device(config.fit.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the OptiTrack fit configuration.")

    training: list[PreparedTake] = []
    for path in training_paths:
        _check_cancelled(cancelled)
        item = prepare_take(path, config)
        training.append(item)
        report(
            f"training {item.take.source_path.name}: "
            f"valid={int(np.count_nonzero(item.valid))}/{item.take.frame_count}; "
            + ", ".join(
                f"{name}={count}" for name, count in item.quality_rejections if count
            )
        )
    validation: list[PreparedTake] = []
    for path in validation_paths:
        _check_cancelled(cancelled)
        item = prepare_take(path, config, causal_velocity=True)
        validation.append(item)
        report(
            f"validation {item.take.source_path.name}: "
            f"valid={int(np.count_nonzero(item.valid))}/{item.take.frame_count}; "
            + ", ".join(
                f"{name}={count}" for name, count in item.quality_rejections if count
            )
        )

    nominal_ei = math.sqrt(config.fit.ei_min_n_m2 * config.fit.ei_max_n_m2)
    nominal_cb = math.sqrt(config.fit.cb_min_n_m2_s * config.fit.cb_max_n_m2_s)
    model = _make_model(
        config,
        ei=nominal_ei,
        cb=nominal_cb,
    )
    train_batch = _make_windows(
        training,
        config,
        model,
        device,
    )
    validation_batch = (
        _make_windows(validation, config, model, device) if validation else None
    )
    maximum_dt_s = float(
        torch.max(
            train_batch.timestamps_s[:, 1:] - train_batch.timestamps_s[:, :-1]
        )
        .detach()
        .cpu()
    )
    stable_ei = model.maximum_stable_bending_stiffness(
        maximum_dt_s,
        pinned_endpoints=START_PINNED_FREE_END,
    )
    if config.fit.ei_max_n_m2 > stable_ei:
        raise ValueError(
            f"EI upper bound {config.fit.ei_max_n_m2:g} exceeds the solver's "
            f"stable limit {stable_ei:g} N m^2 for this sampling interval."
        )
    report(
        f"windows: training={train_batch.count}, "
        f"validation={0 if validation_batch is None else validation_batch.count}, "
        f"horizon={config.fit.window_frames} frames, "
        f"stride={config.fit.window_stride}, device={device}"
    )
    assert train_batch.take_indices is not None
    report(
        "training windows per take: "
        + ", ".join(
            f"{item.take.source_path.name}="
            f"{int(torch.count_nonzero(train_batch.take_indices == index).cpu())}"
            for index, item in enumerate(training)
        )
        + "; objective weight is equal per clean window"
    )
    report(
        f"model: 11 measured material points -> {config.cable.node_count} DER nodes; "
        "only the attachment position is prescribed and the distal tip is dynamic"
    )
    initialization = _parameter_initialization(config)
    report(
        f"initialization: {initialization.method}; "
        f"EI={initialization.bending_stiffness_n_m2:.6g} N m^2, "
        f"Cb={initialization.bending_damping_n_m2_s:.6g} N m^2 s"
    )

    started = time.perf_counter()
    selected = _differentiable_fit(
        model,
        train_batch,
        config,
        report,
        initialization,
        cancelled,
    )
    _check_cancelled(cancelled)
    best_ei = selected.bending_stiffness_n_m2
    best_cb = selected.bending_damping_n_m2_s
    best_rmse = selected.rmse_m
    training_rmse_by_take = _per_take_rmse(
        model,
        train_batch,
        ei=best_ei,
        cb=best_cb,
        robust_scale_m=config.fit.robust_scale_m,
        cancelled=cancelled,
    )
    report(
        "training RMSE by take: "
        + ", ".join(
            f"{item.take.source_path.name}={1000.0 * value:.3f}mm"
            for item, value in zip(training, training_rmse_by_take)
        )
    )
    identifiability_profile = _local_identifiability_profile(
        model,
        train_batch,
        config,
        ei=best_ei,
        cb=best_cb,
        base_objective=selected.objective,
        cancelled=cancelled,
    )
    report(
        "local profile (10% log perturbation, minimum relative objective change): "
        + ", ".join(
            f"{name}={100.0 * values['minimum_relative_objective_change']:.4g}%"
            for name, values in identifiability_profile.items()
        )
    )

    validation_rmse: float | None = None
    validation_rmse_by_take: list[float] = []
    if validation_batch is not None:
        _validation_loss, validation_squared = _rollout_metrics(
            model,
            validation_batch,
            ei=best_ei,
            cb=best_cb,
            robust_scale_m=config.fit.robust_scale_m,
            create_graph=False,
            cancelled=cancelled,
        )
        validation_rmse = float(torch.sqrt(validation_squared).detach().cpu())
        validation_rmse_by_take = _per_take_rmse(
            model,
            validation_batch,
            ei=best_ei,
            cb=best_cb,
            robust_scale_m=config.fit.robust_scale_m,
            cancelled=cancelled,
        )

    _check_cancelled(cancelled)
    elapsed = time.perf_counter() - started
    ei_bound_status = _bound_status(
        best_ei, config.fit.ei_min_n_m2, config.fit.ei_max_n_m2
    )
    cb_bound_status = _bound_status(
        best_cb, config.fit.cb_min_n_m2_s, config.fit.cb_max_n_m2_s
    )
    if (
        ei_bound_status != "interior"
        or cb_bound_status != "interior"
    ):
        report(
            "IDENTIFIABILITY WARNING: selected parameter at a fixed bound; "
            "the bound will not be expanded automatically"
        )
    payload: dict[str, object] = {
        "schema": MODEL_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "measured": {
            "marker_count": config.cable.marker_count,
            "node_count": config.cable.node_count,
            "rod_segments_per_marker_interval": (
                config.cable.rod_segments_per_marker_interval
            ),
            "marker_node_indices": list(config.cable.marker_node_indices),
            "marker_interval_lengths_m": list(config.cable.rest_lengths_m),
            "rest_lengths_m": list(config.cable.rod_rest_lengths_m),
            "marker_material_coordinates_m": list(
                config.cable.marker_material_coordinates_m
            ),
            "rod_material_coordinates_m": list(
                config.cable.rod_material_coordinates_m
            ),
            "length_m": config.cable.length_m,
            "bare_cable_mass_kg": config.cable.bare_cable_mass_kg,
            "moving_marker_count": config.cable.moving_marker_count,
            "moving_marker_masses_kg": list(config.cable.moving_marker_masses_kg),
            "vertex_masses_kg": list(config.cable.vertex_masses_kg),
            "total_dynamic_mass_kg": config.cable.total_dynamic_mass_kg,
            "diameter_m": config.cable.diameter_m,
            "rest_shape": "straight",
            "torsion": (
                "analytically eliminated: a homogeneous circular isotropic cable "
                "with a torque-free tip has zero relaxed material twist strain"
            ),
        },
        "optimized": {
            "bending_stiffness_n_m2": best_ei,
            "bending_damping_n_m2_s": best_cb,
        },
        "fit": {
            "training_rollout_rmse_m": best_rmse,
            "validation_rollout_rmse_m": validation_rmse,
            "training_window_count": train_batch.count,
            "validation_window_count": (
                0 if validation_batch is None else validation_batch.count
            ),
            "training_initialization_rmse_m": train_batch.initialization_rmse_m,
            "validation_initialization_rmse_m": (
                None
                if validation_batch is None
                else validation_batch.initialization_rmse_m
            ),
            "training_rmse_by_take_m": {
                item.take.source_path.name: value
                for item, value in zip(training, training_rmse_by_take)
            },
            "validation_rmse_by_take_m": {
                item.take.source_path.name: value
                for item, value in zip(validation, validation_rmse_by_take)
            },
            "training_windows_by_take": {
                item.take.source_path.name: int(
                    torch.count_nonzero(train_batch.take_indices == index).cpu()
                )
                for index, item in enumerate(training)
            },
            "optimizer": "sobol_initialization_then_projected_cuda_batched_adam",
            "parameterization": "dimensionless projected logarithmic EI and Cb",
            "initializer_candidate_count": config.fit.initializer_candidates,
            "initializer_window_count": min(
                config.fit.initializer_windows, train_batch.count
            ),
            "optimizer_maximum_updates": config.fit.optimizer_iterations,
            "optimizer_batch_windows": config.fit.optimizer_batch_windows,
            "optimizer_learning_rate": config.fit.optimizer_learning_rate,
            "optimizer_gradient_clip": config.fit.optimizer_gradient_clip,
            "optimizer_seed": config.fit.optimizer_seed,
            "optimizer_batch": (
                "deterministically shuffled batches of clean training windows; the "
                "default batch holds every current window concurrently on CUDA; the "
                "final model is selected only by full-training evaluation"
            ),
            "optimizer_stopping": (
                "fixed accepted-update budget with cooperative user cancellation; "
                "no line-search extrapolation"
            ),
            "optimizer_initialization": {
                "method": (
                    "fixed-seed scrambled Sobol search in bounded logarithmic "
                    "parameter space; the geometric midpoint is candidate one"
                ),
                "seed_point_method": initialization.method,
                "bending_stiffness_n_m2": initialization.bending_stiffness_n_m2,
                "bending_damping_n_m2_s": (
                    initialization.bending_damping_n_m2_s
                ),
            },
            "initialization_search_history": [
                {
                    "candidate": candidate,
                    "subset_objective": objective,
                    "subset_rmse_m": rmse,
                    "bending_stiffness_n_m2": ei,
                    "bending_damping_n_m2_s": cb,
                }
                for candidate, objective, rmse, ei, cb
                in selected.initialization_history
            ],
            "offline_constraint_solver": (
                "batched dense exact solve of the symmetric tridiagonal "
                "RATTLE multiplier system"
            ),
            "bending_damping_integrator": (
                "backward Euler solve of the exact fixed-geometry Kelvin-Voigt "
                "objective corotational curvature-rate operator at every dynamics substep"
            ),
            "best_update": selected.best_evaluation,
            "completed_updates": selected.completed_evaluations,
            "selected_training_objective": selected.objective,
            "optimizer_history": [
                {
                    "update": iteration,
                    "scope": "full_training_set",
                    "objective": objective,
                    "rmse_m": rmse,
                    "bending_stiffness_n_m2": ei,
                    "bending_damping_n_m2_s": cb,
                    "bounded_log_coordinate_gradient_norm": gradient_norm,
                }
                for iteration, objective, rmse, ei, cb, gradient_norm
                in selected.history
            ],
            "parameter_bound_status": {
                "bending_stiffness_n_m2": ei_bound_status,
                "bending_damping_n_m2_s": cb_bound_status,
            },
            "local_identifiability_profile": identifiability_profile,
            "window_frames": config.fit.window_frames,
            "window_stride": config.fit.window_stride,
            "robust_scale_m": config.fit.robust_scale_m,
            "ei_bounds_n_m2": [config.fit.ei_min_n_m2, config.fit.ei_max_n_m2],
            "cb_bounds_n_m2_s": [
                config.fit.cb_min_n_m2_s,
                config.fit.cb_max_n_m2_s,
            ],
            "elapsed_s": elapsed,
        },
        "measurement_quality": {
            "maximum_rigid_body_error_m": config.fit.maximum_rigid_body_error_m,
            "maximum_attachment_speed_m_s": config.fit.maximum_attachment_speed_m_s,
            "maximum_marker_speed_m_s": config.fit.maximum_marker_speed_m_s,
            "maximum_chord_excess_m": config.fit.maximum_chord_excess_m,
            "maximum_initialization_rmse_m": (
                config.fit.maximum_initialization_rmse_m
            ),
            "training_rejections": {
                item.take.source_path.name: dict(item.quality_rejections)
                for item in training
            },
            "validation_rejections": {
                item.take.source_path.name: dict(item.quality_rejections)
                for item in validation
            },
        },
        "solver": {
            "method": (
                "explicit_elastic_der_forces, implicit_kelvin_voigt_damping, "
                "mass_weighted_rattle_projection"
            ),
            "ambient_drag": "not fitted; external drag is fixed to zero",
            "substeps": config.fit.substeps,
            "constraint_iterations": config.fit.constraint_iterations,
            "gravity_m_s2": [0.0, 0.0, -9.80665],
        },
        "boundary_condition": {
            "current_dataset": "one rigid attachment pivot plus markers c1-c10",
            "prescribed_vertices": [0],
            "attachment": "prescribed position with free tangent and free material roll",
            "distal_terminal": (
                "not kinematically prescribed; terminal marker mass is dynamic, "
                "with no applied contact moment or externally imposed tip motion"
            ),
            "terminal_orientation_observed": False,
        },
        "observation": {
            "source": "Motive reconstructed labeled marker trajectories",
            "mapping": (
                "eleven measured material points select marker_node_indices from "
                "the refined DER state"
            ),
            "latent_node_initialization": (
                "measured-site-conditioned equal-edge circular arcs; only the "
                "attachment remains exact when noisy chords require conditioning"
            ),
            "training_take_weighting": (
                "mean over time and all dynamic markers c1-c10 within each window, "
                "then equal mean over clean windows"
            ),
            "position_filter": "none",
            "training_velocity_estimator": "centered local quadratic",
            "validation_velocity_estimator": "causal local quadratic",
            "velocity_window_frames": config.fit.velocity_window_frames,
            "coordinate_transform": "Motive Y-up to project Z-up: (x,-z,y)",
        },
        "sources": [
            {
                "path": str(item.take.source_path),
                "sha256": sha256_file(item.take.source_path),
                "role": role,
                "take_name": item.take.take_name,
                "capture_rate_hz": item.take.capture_rate_hz,
                "export_rate_hz": item.take.export_rate_hz,
                "attachment_rigid_body_name": item.take.attachment_rigid_body_name,
            }
            for role, items in (("training", training), ("validation", validation))
            for item in items
        ],
    }
    model_path = config.model_path.expanduser().resolve()
    _check_cancelled(cancelled)
    _atomic_json(model_path, payload)
    report(
        f"completed: train={1000.0 * best_rmse:.3f} mm"
        + (
            ""
            if validation_rmse is None
            else f", validation={1000.0 * validation_rmse:.3f} mm"
        )
    )
    return OptitrackFitResult(
        best_ei,
        best_cb,
        best_rmse,
        validation_rmse,
        train_batch.count,
        0 if validation_batch is None else validation_batch.count,
        elapsed,
        model_path,
    )
