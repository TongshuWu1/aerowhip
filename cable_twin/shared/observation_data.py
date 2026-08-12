"""Registered-depth cable observations and fixed-length trajectory storage."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
import uuid

import numpy as np

from .saving import validate_observation_archive


TRAJECTORY_SCHEMA_VERSION = 2
RECONSTRUCTION_METHOD = "uncertainty_weighted_fixed_length_curve_v2"
MEDIAN_STANDARD_ERROR_FACTOR = 1.2533141373155
RECONSTRUCTION_KINDS = frozenset(
    {
        "fixed_length_partial_observation_reconstruction",
        "frozen_pointcloud_reconstruction",
    }
)


@dataclass(frozen=True, slots=True)
class ObservationSummary:
    frame_count: int
    duration_s: float
    metric_curve_count: int
    metric_fraction: float
    cable_identity: int
    fitted: bool


@dataclass(frozen=True, slots=True)
class MetricCurveSequence:
    path: Path
    metadata: dict[str, Any]
    timestamps_ns: np.ndarray
    source_positions: np.ndarray
    route_xy: np.ndarray
    route_valid: np.ndarray
    route_segment_id: np.ndarray
    endpoint_centers_xy: np.ndarray
    endpoint_valid: np.ndarray
    metric_points_m: np.ndarray
    metric_valid: np.ndarray
    metric_endpoint_points_m: np.ndarray
    metric_endpoint_valid: np.ndarray
    depth_spread_m: np.ndarray
    depth_support: np.ndarray
    metric_endpoint_depth_spread_m: np.ndarray
    metric_endpoint_support: np.ndarray
    gravity_camera_m_s2: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(len(self.timestamps_ns))

    @property
    def cable_identity(self) -> int:
        return int(self.metadata["settings"]["cable_identity"])

@dataclass(frozen=True, slots=True)
class StoredTrajectory:
    positions_m: np.ndarray
    velocities_m_s: np.ndarray
    metrics: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def registered_depth_standard_deviation(
    points_m: np.ndarray,
    valid: np.ndarray,
    depth_spread_m: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    """Propagate robust depth dispersion through median lifting along the camera ray."""

    points = np.asarray(points_m, dtype=np.float64)
    active = np.asarray(valid, dtype=bool) & np.all(np.isfinite(points), axis=-1)
    depth = np.abs(points[..., 2])
    ray_norm = np.linalg.norm(points, axis=-1) / np.maximum(
        depth, np.finfo(np.float64).tiny
    )
    sigma = (
        MEDIAN_STANDARD_ERROR_FACTOR
        * np.asarray(depth_spread_m, dtype=np.float64)
        / np.sqrt(np.maximum(np.asarray(support, dtype=np.float64), 1.0))
        * ray_norm
    )
    positive = active & np.isfinite(sigma) & (sigma > 0.0)
    if not np.any(positive):
        raise ValueError("Registered-depth evidence has no measurable dispersion.")
    typical = float(np.median(sigma[positive]))
    return np.where(positive, sigma, typical)


def trajectory_path(observation_path: str | Path) -> Path:
    path = Path(observation_path).expanduser().resolve()
    return path.with_name(path.stem + ".trajectory.npz")


def reconstruction_path(observation_path: str | Path) -> Path:
    """Immutable state-estimation cache used as DDER identification data."""

    path = Path(observation_path).expanduser().resolve()
    return path.with_name(path.stem + ".reconstruction.npz")


def load_stored_trajectory(
    path: str | Path,
    *,
    sequence: MetricCurveSequence,
    node_count: int,
    accepted_kinds: frozenset[str] | None = None,
) -> StoredTrajectory:
    """Load a trajectory only when it exactly matches its source observation."""

    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as data:
        try:
            schema = int(np.asarray(data["schema_version"]).item())
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
            timestamps = np.asarray(data["timestamp_ns"], dtype=np.int64)
            source_positions = np.asarray(data["source_position"], dtype=np.int64)
            positions = np.asarray(data["positions_m"], dtype=np.float32)
            velocities = np.asarray(data["velocities_m_s"], dtype=np.float32)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid DDER trajectory archive: {source}") from error
    expected_shape = (sequence.frame_count, int(node_count), 3)
    if schema != TRAJECTORY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported DDER trajectory schema {schema}: {source}")
    if metadata.get("observation_sha256") != sha256_file(sequence.path):
        raise ValueError(f"DDER trajectory is stale for {sequence.path.name}.")
    if not np.array_equal(timestamps, sequence.timestamps_ns) or not np.array_equal(
        source_positions, sequence.source_positions
    ):
        raise ValueError(f"DDER trajectory frame identity does not match {sequence.path.name}.")
    if positions.shape != expected_shape or velocities.shape != expected_shape:
        raise ValueError(
            f"DDER trajectory has shape {positions.shape}, expected {expected_shape}."
        )
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
        raise ValueError(f"DDER trajectory contains non-finite state: {source}")
    metrics = metadata.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"DDER trajectory metrics are invalid: {source}")
    kind = metrics.get("trajectory_kind")
    if accepted_kinds is not None and kind not in accepted_kinds:
        raise ValueError(f"DDER trajectory kind {kind!r} is not a reconstruction cache.")
    return StoredTrajectory(positions, velocities, dict(metrics))


def load_observation(path: str | Path) -> MetricCurveSequence:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as data:
        metadata = validate_observation_archive(data)
        metric_valid = np.asarray(data["metric_valid"], dtype=bool)
        active = np.flatnonzero(np.count_nonzero(metric_valid, axis=1) >= 8)
        if len(active) == 0:
            frame_slice = slice(None)
        else:
            frame_slice = slice(int(active[0]), int(active[-1]) + 1)
        return MetricCurveSequence(
            path=source,
            metadata=metadata,
            timestamps_ns=np.asarray(data["timestamp_ns"], dtype=np.int64)[frame_slice],
            source_positions=np.asarray(data["source_position"], dtype=np.int64)[frame_slice],
            route_xy=np.asarray(data["route_xy"], dtype=np.float64)[frame_slice],
            route_valid=np.asarray(data["route_valid"], dtype=bool)[frame_slice],
            route_segment_id=np.asarray(data["route_segment_id"], dtype=np.int16)[frame_slice],
            endpoint_centers_xy=np.asarray(data["endpoint_centers_xy"], dtype=np.float64)[frame_slice],
            endpoint_valid=np.asarray(data["endpoint_valid"], dtype=bool)[frame_slice],
            metric_points_m=np.asarray(data["metric_points_m"], dtype=np.float64)[frame_slice],
            metric_valid=metric_valid[frame_slice],
            metric_endpoint_points_m=np.asarray(data["metric_endpoint_points_m"], dtype=np.float64)[frame_slice],
            metric_endpoint_valid=np.asarray(data["metric_endpoint_valid"], dtype=bool)[frame_slice],
            depth_spread_m=np.asarray(data["depth_spread_m"], dtype=np.float64)[frame_slice],
            depth_support=np.asarray(data["depth_support"], dtype=np.int16)[frame_slice],
            metric_endpoint_depth_spread_m=np.asarray(
                data["metric_endpoint_depth_spread_m"], dtype=np.float64
            )[frame_slice],
            metric_endpoint_support=np.asarray(
                data["metric_endpoint_support"], dtype=np.int16
            )[frame_slice],
            gravity_camera_m_s2=np.asarray(data["gravity_camera_m_s2"], dtype=np.float64)[frame_slice],
        )


def usable_metric_frames(sequence: MetricCurveSequence, minimum_samples: int = 12) -> np.ndarray:
    finite = np.all(np.isfinite(sequence.metric_points_m), axis=2)
    valid = sequence.metric_valid & finite
    counts = np.count_nonzero(valid, axis=1)
    return counts >= int(minimum_samples)


def summarize_observation(path: str | Path) -> ObservationSummary:
    sequence = load_observation(path)
    usable = usable_metric_frames(sequence)
    duration = float(sequence.timestamps_ns[-1] - sequence.timestamps_ns[0]) * 1.0e-9
    return ObservationSummary(
        frame_count=sequence.frame_count,
        duration_s=duration,
        metric_curve_count=int(np.count_nonzero(usable)),
        metric_fraction=float(np.mean(usable)),
        cable_identity=sequence.cable_identity,
        fitted=trajectory_path(sequence.path).is_file(),
    )


def _arc_resample(points: np.ndarray, count: int) -> np.ndarray:
    delta = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], delta > 1.0e-6))
    points = points[keep]
    if len(points) < 2:
        raise ValueError("Metric curve has fewer than two distinct points.")
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    targets = np.linspace(0.0, arc[-1], count)
    return np.column_stack([np.interp(targets, arc, points[:, axis]) for axis in range(3)])


def initial_metric_trajectory(
    sequence: MetricCurveSequence,
    node_count: int,
    cable_length_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a smooth initializer from ordered point-cloud evidence.

    Observations are resampled by metric arc length. Missing frames are filled
    only for initialization; they carry no measurement weight in fitting.
    """

    frame_count = sequence.frame_count
    nodes = np.full((frame_count, node_count, 3), np.nan, dtype=np.float64)
    usable = usable_metric_frames(sequence)
    calibration = sequence.metadata["source"]["calibration"]
    intrinsics = np.asarray(calibration["left_intrinsics"], dtype=np.float64)
    fx, fy, cx, cy = (
        intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    )
    previous_candidate: np.ndarray | None = None
    for frame in np.flatnonzero(usable):
        valid = sequence.metric_valid[frame]
        segment_ids = sequence.route_segment_id[frame]
        candidates = [
            (np.count_nonzero(valid & (segment_ids == segment_id)), int(segment_id))
            for segment_id in np.unique(segment_ids[valid])
        ]
        if not candidates:
            usable[frame] = False
            continue
        _count, selected_segment = max(candidates)
        selected = valid & (segment_ids == selected_segment)
        measured = sequence.metric_points_m[frame, selected]
        finite = np.all(np.isfinite(measured), axis=1)
        measured = measured[finite]
        route = sequence.route_xy[frame, selected][finite]
        if len(measured) < 4:
            usable[frame] = False
            continue
        coordinate = np.linspace(0.0, 1.0, len(measured))
        degree = min(3, len(measured) - 1)
        weights = np.sqrt(np.maximum(sequence.depth_support[frame, selected][finite], 1))
        coefficients = np.polyfit(coordinate, measured[:, 2], degree, w=weights)
        depth = np.polyval(coefficients, coordinate)
        points = np.column_stack(
            (
                (route[:, 0] - cx) * depth / fx,
                (route[:, 1] - cy) * depth / fy,
                depth,
            )
        )
        padded = np.pad(points, ((2, 2), (0, 0)), mode="edge")
        smooth = sum(padded[offset : offset + len(points)] for offset in range(5)) / 5.0
        endpoints = sequence.metric_endpoint_points_m[frame, sequence.metric_endpoint_valid[frame]]
        if len(endpoints) == 2:
            direct = np.linalg.norm(endpoints[0] - smooth[0]) + np.linalg.norm(endpoints[1] - smooth[-1])
            flipped = np.linalg.norm(endpoints[1] - smooth[0]) + np.linalg.norm(endpoints[0] - smooth[-1])
            ordered = endpoints if direct <= flipped else endpoints[::-1]
            smooth = np.vstack((ordered[0], smooth, ordered[1]))
        elif len(endpoints) == 1:
            if np.linalg.norm(endpoints[0] - smooth[0]) <= np.linalg.norm(endpoints[0] - smooth[-1]):
                smooth = np.vstack((endpoints[0], smooth))
            else:
                smooth = np.vstack((smooth, endpoints[0]))
        try:
            candidate = _arc_resample(smooth, node_count)
        except ValueError:
            usable[frame] = False
            continue
        # A centerline is geometrically unchanged when its node order is reversed,
        # but temporal dynamics are not.  Preserve material-node identity by choosing
        # the orientation closest to the preceding accepted observation.
        if previous_candidate is not None:
            direct_motion = np.mean(np.linalg.norm(candidate - previous_candidate, axis=1))
            reversed_motion = np.mean(
                np.linalg.norm(candidate[::-1] - previous_candidate, axis=1)
            )
            if reversed_motion < direct_motion:
                candidate = candidate[::-1].copy()
        nodes[frame] = candidate
        previous_candidate = candidate
    valid_indices = np.flatnonzero(usable)
    if len(valid_indices) < 3:
        raise ValueError(
            f"{sequence.path.name} has fewer than three usable ZED point-cloud curves."
        )
    frames = np.arange(frame_count)
    for node in range(node_count):
        for axis in range(3):
            nodes[:, node, axis] = np.interp(frames, valid_indices, nodes[valid_indices, node, axis])
    return nodes, usable


def save_trajectory(
    path: str | Path,
    *,
    sequence: MetricCurveSequence,
    positions_m: np.ndarray,
    velocities_m_s: np.ndarray,
    metrics: Mapping[str, Any],
    model_path: str | Path | None,
) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": "zed_pointcloud_dder_trajectory_v2",
        "observation": str(sequence.path),
        "observation_sha256": sha256_file(sequence.path),
        "model": None if model_path is None else str(Path(model_path).expanduser().resolve()),
        "metrics": dict(metrics),
    }
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            schema_version=np.asarray(TRAJECTORY_SCHEMA_VERSION, dtype=np.int32),
            metadata_json=np.asarray(json.dumps(metadata, separators=(",", ":"))),
            timestamp_ns=sequence.timestamps_ns,
            source_position=sequence.source_positions,
            positions_m=np.asarray(positions_m, dtype=np.float32),
            velocities_m_s=np.asarray(velocities_m_s, dtype=np.float32),
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
