"""Image-plane PIDNet route and node-observation archives."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping
import uuid

import numpy as np

from ..shared.contracts import StereoFrame
from ..shared.observation_data import sha256_file, trajectory_path
from ..shared.planar_observer import PlanarFrameObservation


PLANAR_OBSERVATION_SCHEMA = 2
PLANAR_OBSERVATION_METHOD = "pidnet_image_plane_complete_route_v2"
PLANAR_TRAJECTORY_SCHEMA = "pidnet_image_plane_node_observations_v4"
PLANAR_TIMING_NAMES = (
    "pidnet_ms",
    "mask_postprocess_ms",
    "route_ms",
    "total_ms",
)


@dataclass(frozen=True, slots=True)
class PlanarSequence:
    path: Path
    metadata: dict[str, Any]
    source_position: np.ndarray
    timestamp_ns: np.ndarray
    route_xy_px: np.ndarray
    route_xz_m: np.ndarray
    endpoint_xy_px: np.ndarray
    endpoint_xz_m: np.ndarray
    complete: np.ndarray
    failure: np.ndarray
    timings_ms: np.ndarray

    @property
    def frame_count(self) -> int:
        return len(self.timestamp_ns)


@dataclass(frozen=True, slots=True)
class PlanarSummary:
    frame_count: int
    duration_s: float
    complete_fraction: float
    fitted: bool


@dataclass(frozen=True, slots=True)
class PlanarTrajectory:
    positions_xz_m: np.ndarray
    velocities_xz_m_s: np.ndarray
    valid: np.ndarray
    timestamp_ns: np.ndarray
    metadata: dict[str, Any]


def _metadata(data: Mapping[str, np.ndarray]) -> dict[str, Any]:
    try:
        payload = json.loads(str(np.asarray(data["metadata_json"]).item()))
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("Planar observation metadata is missing or invalid.") from error
    if payload.get("method") != PLANAR_OBSERVATION_METHOD:
        raise ValueError("Observation archive is not a current planar PIDNet sequence.")
    return payload


def validate_planar_observation(data: Mapping[str, np.ndarray]) -> dict[str, Any]:
    try:
        schema = int(np.asarray(data["schema_version"]).item())
        frame_count = int(np.asarray(data["frame_count"]).item())
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Planar observation header is invalid.") from error
    if schema != PLANAR_OBSERVATION_SCHEMA or frame_count < 1:
        raise ValueError("Planar observation schema is unsupported or empty.")
    required = {
        "source_position": (frame_count,),
        "timestamp_ns": (frame_count,),
        "endpoint_xy_px": (frame_count, 2, 2),
        "endpoint_xz_m": (frame_count, 2, 2),
        "complete": (frame_count,),
        "failure": (frame_count,),
        "timings_ms": (frame_count, len(PLANAR_TIMING_NAMES)),
    }
    for name, shape in required.items():
        if name not in data or np.asarray(data[name]).shape != shape:
            raise ValueError(f"Planar observation field {name} has an invalid shape.")
    route_px = np.asarray(data.get("route_xy_px"))
    route_xz = np.asarray(data.get("route_xz_m"))
    if (
        route_px.ndim != 3
        or route_px.shape[0] != frame_count
        or route_px.shape[2] != 2
        or route_px.shape != route_xz.shape
        or route_px.shape[1] < 24
    ):
        raise ValueError("Planar routes must have shape TxMx2 with M >= 24.")
    complete = np.asarray(data["complete"], dtype=bool)
    if np.any(complete & ~np.all(np.isfinite(route_px), axis=(1, 2))):
        raise ValueError("Complete planar routes must contain finite image points.")
    if np.any(complete & ~np.all(np.isfinite(route_xz), axis=(1, 2))):
        raise ValueError("Complete planar routes must contain finite metric points.")
    endpoints = np.asarray(data["endpoint_xz_m"])
    if np.any(complete & ~np.all(np.isfinite(endpoints), axis=(1, 2))):
        raise ValueError("Complete planar routes must contain finite endpoints.")
    timestamps = np.asarray(data["timestamp_ns"], dtype=np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Planar observation timestamps must increase strictly.")
    metadata = _metadata(data)
    mapping = metadata.get("image_plane_mapping")
    if not isinstance(mapping, dict):
        raise ValueError("Image-plane metric mapping is missing.")
    if mapping.get("method") != "constant_scale_from_median_complete_route_length":
        raise ValueError("Image-plane metric mapping method is unsupported.")
    origin = np.asarray(mapping.get("image_origin_px"), dtype=np.float64)
    image_size = np.asarray(mapping.get("image_size_px"), dtype=np.int64)
    if origin.shape != (2,) or not np.all(np.isfinite(origin)):
        raise ValueError("Image-plane origin is invalid.")
    if image_size.shape != (2,) or np.any(image_size < 1):
        raise ValueError("Image-plane dimensions are invalid.")
    scale = mapping.get("scale_m_per_px")
    if np.any(complete) and (
        not isinstance(scale, (int, float))
        or not np.isfinite(float(scale))
        or float(scale) <= 0.0
    ):
        raise ValueError("Image-plane pixel-to-metre scale is invalid.")
    return metadata


class PlanarSequenceWriter:
    def __init__(
        self,
        path: str | Path,
        *,
        metadata: Mapping[str, Any],
        route_samples: int,
        cable_length_m: float,
        image_width_px: int,
        image_height_px: int,
        replace_existing: bool = False,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if self.path.exists() and not replace_existing:
            raise FileExistsError(f"Refusing to overwrite planar observation: {self.path}")
        self.metadata = dict(metadata)
        self.route_samples = int(route_samples)
        self.cable_length_m = float(cable_length_m)
        self.image_width_px = int(image_width_px)
        self.image_height_px = int(image_height_px)
        if self.cable_length_m <= 0.0:
            raise ValueError("Cable length must be positive for image-plane scaling.")
        if min(self.image_width_px, self.image_height_px) < 1:
            raise ValueError("Image dimensions must be positive.")
        self.replace_existing = bool(replace_existing)
        self.rows: list[dict[str, Any]] = []

    def append(self, frame: StereoFrame, observed: PlanarFrameObservation) -> None:
        if observed.view.curve.points_xy.shape != (self.route_samples, 2):
            raise ValueError("Planar route sample count changed during extraction.")
        self.rows.append(
            {
                "source_position": frame.source_position,
                "timestamp_ns": frame.timestamp_ns,
                "route_xy_px": observed.view.curve.points_xy,
                "endpoint_xy_px": observed.view.curve.endpoint_centers_xy,
                "complete": observed.complete,
                "failure": str(observed.view.failure or ""),
                "timings_ms": np.asarray(
                    [observed.timings_ms[name] for name in PLANAR_TIMING_NAMES],
                    dtype=np.float32,
                ),
            }
        )

    def close(self) -> Path:
        if not self.rows:
            raise ValueError("Cannot save an empty planar observation sequence.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        route_xy_px = np.asarray(
            [row["route_xy_px"] for row in self.rows], dtype=np.float64
        )
        endpoint_xy_px = np.asarray(
            [row["endpoint_xy_px"] for row in self.rows], dtype=np.float64
        )
        complete = np.asarray([row["complete"] for row in self.rows], dtype=bool)
        route_xz_m = np.full_like(route_xy_px, np.nan)
        endpoint_xz_m = np.full_like(endpoint_xy_px, np.nan)
        scale_m_per_px: float | None = None
        if np.any(complete):
            arc_length_px = np.linalg.norm(
                np.diff(route_xy_px[complete], axis=1), axis=2
            ).sum(axis=1)
            if np.any(~np.isfinite(arc_length_px) | (arc_length_px <= 0.0)):
                raise ValueError("Complete PIDNet routes must have positive pixel length.")
            scale_m_per_px = self.cable_length_m / float(np.median(arc_length_px))
            center = np.asarray(
                (0.5 * self.image_width_px, 0.5 * self.image_height_px),
                dtype=np.float64,
            )
            route_xz_m[complete, :, 0] = (
                route_xy_px[complete, :, 0] - center[0]
            ) * scale_m_per_px
            route_xz_m[complete, :, 1] = (
                center[1] - route_xy_px[complete, :, 1]
            ) * scale_m_per_px
            endpoint_xz_m[complete, :, 0] = (
                endpoint_xy_px[complete, :, 0] - center[0]
            ) * scale_m_per_px
            endpoint_xz_m[complete, :, 1] = (
                center[1] - endpoint_xy_px[complete, :, 1]
            ) * scale_m_per_px
        metadata = {
            **self.metadata,
            "image_plane_mapping": {
                "method": "constant_scale_from_median_complete_route_length",
                "cable_length_m": self.cable_length_m,
                "scale_m_per_px": scale_m_per_px,
                "image_origin_px": [
                    0.5 * self.image_width_px,
                    0.5 * self.image_height_px,
                ],
                "image_size_px": [self.image_width_px, self.image_height_px],
                "axes": "x_right_z_up",
                "assumption": "fixed_level_camera_cable_motion_parallel_to_image_plane",
            },
        }
        payload = {
            "schema_version": np.asarray(PLANAR_OBSERVATION_SCHEMA, dtype=np.int32),
            "metadata_json": np.asarray(json.dumps(metadata, separators=(",", ":"))),
            "timing_names": np.asarray(PLANAR_TIMING_NAMES),
            "source_position": np.asarray([row["source_position"] for row in self.rows], dtype=np.int64),
            "timestamp_ns": np.asarray([row["timestamp_ns"] for row in self.rows], dtype=np.int64),
            "route_xy_px": route_xy_px.astype(np.float32),
            "route_xz_m": route_xz_m.astype(np.float32),
            "endpoint_xy_px": endpoint_xy_px.astype(np.float32),
            "endpoint_xz_m": endpoint_xz_m.astype(np.float32),
            "complete": complete,
            "failure": np.asarray([row["failure"] for row in self.rows]),
            "timings_ms": np.asarray([row["timings_ms"] for row in self.rows], dtype=np.float32),
            "frame_count": np.asarray(len(self.rows), dtype=np.int64),
        }
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp.npz")
        try:
            np.savez_compressed(temporary, **payload)
            with np.load(temporary, allow_pickle=False) as data:
                validate_planar_observation(data)
            if self.path.exists() and not self.replace_existing:
                raise FileExistsError(f"Refusing to overwrite planar observation: {self.path}")
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
        return self.path


def load_planar_sequence(path: str | Path) -> PlanarSequence:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as data:
        metadata = validate_planar_observation(data)
        return PlanarSequence(
            source,
            metadata,
            np.asarray(data["source_position"], dtype=np.int64),
            np.asarray(data["timestamp_ns"], dtype=np.int64),
            np.asarray(data["route_xy_px"], dtype=np.float64),
            np.asarray(data["route_xz_m"], dtype=np.float64),
            np.asarray(data["endpoint_xy_px"], dtype=np.float64),
            np.asarray(data["endpoint_xz_m"], dtype=np.float64),
            np.asarray(data["complete"], dtype=bool),
            np.asarray(data["failure"]),
            np.asarray(data["timings_ms"], dtype=np.float64),
        )


def summarize_planar_observation(path: str | Path) -> PlanarSummary:
    sequence = load_planar_sequence(path)
    duration = float((sequence.timestamp_ns[-1] - sequence.timestamp_ns[0]) * 1.0e-9)
    return PlanarSummary(
        sequence.frame_count,
        duration,
        float(np.mean(sequence.complete)),
        trajectory_path(sequence.path).is_file(),
    )


def save_planar_trajectory(
    observation_path: str | Path,
    *,
    positions_xz_m: np.ndarray,
    velocities_xz_m_s: np.ndarray,
    valid: np.ndarray,
    timestamp_ns: np.ndarray,
    metadata: Mapping[str, Any],
) -> Path:
    observation = Path(observation_path).expanduser().resolve()
    target = trajectory_path(observation)
    payload_metadata = {
        **dict(metadata),
        "schema": PLANAR_TRAJECTORY_SCHEMA,
        "observation_sha256": sha256_file(observation),
    }
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp.npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.savez_compressed(
            temporary,
            positions_xz_m=np.asarray(positions_xz_m, dtype=np.float32),
            velocities_xz_m_s=np.asarray(velocities_xz_m_s, dtype=np.float32),
            valid=np.asarray(valid, dtype=bool),
            timestamp_ns=np.asarray(timestamp_ns, dtype=np.int64),
            metadata_json=np.asarray(json.dumps(payload_metadata, separators=(",", ":"))),
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_planar_trajectory(
    path: str | Path,
    *,
    observation_path: str | Path | None = None,
) -> PlanarTrajectory:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        if metadata.get("schema") != PLANAR_TRAJECTORY_SCHEMA:
            raise ValueError("Trajectory is not a current planar cable trajectory.")
        if observation_path is not None and metadata.get("observation_sha256") != sha256_file(
            observation_path
        ):
            raise ValueError("Planar trajectory does not match its observation archive.")
        positions = np.asarray(data["positions_xz_m"], dtype=np.float64)
        velocities = np.asarray(data["velocities_xz_m_s"], dtype=np.float64)
        valid = np.asarray(data["valid"], dtype=bool)
        timestamps = np.asarray(data["timestamp_ns"], dtype=np.int64)
    if positions.ndim != 3 or positions.shape[-1] != 2 or velocities.shape != positions.shape:
        raise ValueError("Planar trajectory positions and velocities have invalid shapes.")
    if valid.shape != (len(positions),) or timestamps.shape != valid.shape:
        raise ValueError("Planar trajectory frame arrays have invalid shapes.")
    return PlanarTrajectory(positions, velocities, valid, timestamps, metadata)
