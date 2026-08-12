from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping
import uuid

import numpy as np

from .contracts import (
    OBSERVATION_SCHEMA_VERSION,
    POINT_CLOUD_CURVE_METHOD,
    StereoFrame,
    ViewObservation,
)
from .metric_curve import MetricCurveObservation


TIMING_NAMES = (
    "pidnet_ms",
    "mask_postprocess_ms",
    "route_ms",
    "depth_lift_ms",
    "total_ms",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} in observation metadata.")


def _metadata(data: Mapping[str, np.ndarray]) -> dict[str, Any]:
    try:
        value = json.loads(str(np.asarray(data["metadata_json"]).item()))
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("Point-cloud observation metadata is missing or invalid.") from error
    if value.get("method") != POINT_CLOUD_CURVE_METHOD:
        raise ValueError("Observation archive uses a superseded reconstruction method.")
    return value


def validate_observation_archive(data: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Validate ordered PIDNet curves lifted by registered ZED depth."""

    try:
        schema = int(np.asarray(data["schema_version"]).item())
        frame_count = int(np.asarray(data["frame_count"]).item())
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError("Observation archive header is invalid.") from error
    if schema != OBSERVATION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported observation schema {schema}; analyze the original SVO2 again."
        )
    if frame_count < 1:
        raise ValueError("Observation archive is empty.")
    required = {
        "sequence_index": (frame_count,),
        "source_position": (frame_count,),
        "timestamp_ns": (frame_count,),
        "route_xy": None,
        "route_valid": None,
        "route_segment_id": None,
        "endpoint_centers_xy": (frame_count, 2, 2),
        "endpoint_valid": (frame_count, 2),
        "component_counts": (frame_count, 2),
        "metric_points_m": None,
        "metric_valid": None,
        "depth_spread_m": None,
        "depth_support": None,
        "metric_endpoint_points_m": (frame_count, 2, 3),
        "metric_endpoint_valid": (frame_count, 2),
        "metric_endpoint_depth_spread_m": (frame_count, 2),
        "metric_endpoint_support": (frame_count, 2),
        "gravity_camera_m_s2": (frame_count, 3),
        "angular_velocity_camera_deg_s": (frame_count, 3),
        "failure": (frame_count,),
        "timings_ms": (frame_count, len(TIMING_NAMES)),
    }
    for name, shape in required.items():
        if name not in data:
            raise ValueError(f"Observation archive is missing {name}.")
        array = np.asarray(data[name])
        if shape is not None and array.shape != shape:
            raise ValueError(f"Observation field {name} has shape {array.shape}, expected {shape}.")
    routes = np.asarray(data["route_xy"])
    if routes.ndim != 3 or routes.shape[0] != frame_count or routes.shape[-1] != 2:
        raise ValueError("route_xy must have shape TxMx2.")
    if routes.shape[1] < 24:
        raise ValueError("Observation routes contain fewer than 24 samples.")
    sample_count = routes.shape[1]
    expected_metric_shapes = {
        "route_valid": (frame_count, sample_count),
        "route_segment_id": (frame_count, sample_count),
        "metric_points_m": (frame_count, sample_count, 3),
        "metric_valid": (frame_count, sample_count),
        "depth_spread_m": (frame_count, sample_count),
        "depth_support": (frame_count, sample_count),
    }
    for name, shape in expected_metric_shapes.items():
        if np.asarray(data[name]).shape != shape:
            raise ValueError(f"Observation field {name} has an invalid shape.")
    metric_points = np.asarray(data["metric_points_m"])
    metric_valid = np.asarray(data["metric_valid"], dtype=bool)
    if np.any(metric_valid & ~np.all(np.isfinite(metric_points), axis=2)):
        raise ValueError("Valid metric curve samples must be finite.")
    endpoint_points = np.asarray(data["metric_endpoint_points_m"])
    endpoint_valid = np.asarray(data["metric_endpoint_valid"], dtype=bool)
    if np.any(endpoint_valid & ~np.all(np.isfinite(endpoint_points), axis=2)):
        raise ValueError("Valid metric endpoint samples must be finite.")
    timestamps = np.asarray(data["timestamp_ns"], dtype=np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Observation timestamps must be strictly increasing.")
    return _metadata(data)


class ObservationSequenceWriter:
    """Atomic writer for registered-depth metric curve evidence."""

    def __init__(
        self,
        path: str | Path,
        *,
        metadata: Mapping[str, Any],
        route_samples: int,
        replace_existing: bool = False,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if self.path.exists() and not replace_existing:
            raise FileExistsError(f"Refusing to overwrite observation archive: {self.path}")
        self.metadata = dict(metadata)
        self.route_samples = int(route_samples)
        self.replace_existing = bool(replace_existing)
        self._rows: list[dict[str, Any]] = []
        self._closed = False

    @staticmethod
    def _view(view: ViewObservation, sample_count: int) -> tuple[Any, ...]:
        curve = view.curve
        if len(curve.points_xy) != sample_count:
            raise ValueError("Visible-curve sample count changed during extraction.")
        counts = np.asarray(
            (curve.endpoint_component_count, curve.body_component_count), dtype=np.int32
        )
        return (
            curve.points_xy, curve.point_valid, curve.segment_ids,
            curve.endpoint_centers_xy, curve.endpoint_valid, counts,
            str(view.failure or ""),
        )

    def append(
        self,
        frame: StereoFrame,
        left: ViewObservation,
        metric: MetricCurveObservation | None,
        *,
        timings_ms: Mapping[str, float],
    ) -> None:
        if self._closed:
            raise RuntimeError("Observation writer is closed.")
        left_values = self._view(left, self.route_samples)
        if metric is None:
            metric_points = np.full((self.route_samples, 3), np.nan, dtype=np.float32)
            metric_valid = np.zeros(self.route_samples, dtype=bool)
            depth_spread = np.full(self.route_samples, np.nan, dtype=np.float32)
            depth_support = np.zeros(self.route_samples, dtype=np.int16)
            metric_endpoints = np.full((2, 3), np.nan, dtype=np.float32)
            metric_endpoint_valid = np.zeros(2, dtype=bool)
            metric_endpoint_spread = np.full(2, np.nan, dtype=np.float32)
            metric_endpoint_support = np.zeros(2, dtype=np.int16)
        else:
            if len(metric.points_camera_m) != self.route_samples:
                raise ValueError("Metric sample count changed during observation extraction.")
            metric_points = metric.points_camera_m
            metric_valid = metric.valid
            depth_spread = metric.depth_spread_m
            depth_support = metric.support
            metric_endpoints = metric.endpoint_points_camera_m
            metric_endpoint_valid = metric.endpoint_valid
            metric_endpoint_spread = metric.endpoint_depth_spread_m
            metric_endpoint_support = metric.endpoint_support
        gravity = (
            np.full(3, np.nan, dtype=np.float32)
            if frame.gravity_camera_m_s2 is None
            else np.asarray(frame.gravity_camera_m_s2, dtype=np.float32)
        )
        angular = (
            np.full(3, np.nan, dtype=np.float32)
            if frame.angular_velocity_camera_deg_s is None
            else np.asarray(frame.angular_velocity_camera_deg_s, dtype=np.float32)
        )
        self._rows.append(
            {
                "sequence_index": frame.sequence_index,
                "source_position": frame.source_position,
                "timestamp_ns": frame.timestamp_ns,
                "route": left_values[0],
                "valid": left_values[1],
                "segment_ids": left_values[2],
                "endpoints": left_values[3],
                "endpoint_valid": left_values[4],
                "counts": left_values[5],
                "metric_points": metric_points,
                "metric_valid": metric_valid,
                "depth_spread": depth_spread,
                "depth_support": depth_support,
                "metric_endpoints": metric_endpoints,
                "metric_endpoint_valid": metric_endpoint_valid,
                "metric_endpoint_spread": metric_endpoint_spread,
                "metric_endpoint_support": metric_endpoint_support,
                "gravity": gravity,
                "angular": angular,
                "failure": left_values[6],
                "timings": np.asarray(
                    [float(timings_ms.get(name, np.nan)) for name in TIMING_NAMES],
                    dtype=np.float32,
                ),
            }
        )

    def close(self) -> Path:
        if self._closed:
            return self.path
        if not self._rows:
            raise ValueError("Cannot save an empty observation archive.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": np.asarray(OBSERVATION_SCHEMA_VERSION, dtype=np.int32),
            "metadata_json": np.asarray(
                json.dumps(self.metadata, separators=(",", ":"), default=_json_default)
            ),
            "timing_names": np.asarray(TIMING_NAMES),
            "sequence_index": np.asarray([row["sequence_index"] for row in self._rows], dtype=np.int64),
            "source_position": np.asarray([row["source_position"] for row in self._rows], dtype=np.int64),
            "timestamp_ns": np.asarray([row["timestamp_ns"] for row in self._rows], dtype=np.int64),
            "route_xy": np.asarray([row["route"] for row in self._rows], dtype=np.float32),
            "route_valid": np.asarray([row["valid"] for row in self._rows], dtype=bool),
            "route_segment_id": np.asarray([row["segment_ids"] for row in self._rows], dtype=np.int16),
            "endpoint_centers_xy": np.asarray([row["endpoints"] for row in self._rows], dtype=np.float32),
            "endpoint_valid": np.asarray([row["endpoint_valid"] for row in self._rows], dtype=bool),
            "component_counts": np.asarray([row["counts"] for row in self._rows], dtype=np.int32),
            "metric_points_m": np.asarray([row["metric_points"] for row in self._rows], dtype=np.float32),
            "metric_valid": np.asarray([row["metric_valid"] for row in self._rows], dtype=bool),
            "depth_spread_m": np.asarray([row["depth_spread"] for row in self._rows], dtype=np.float32),
            "depth_support": np.asarray([row["depth_support"] for row in self._rows], dtype=np.int16),
            "metric_endpoint_points_m": np.asarray([row["metric_endpoints"] for row in self._rows], dtype=np.float32),
            "metric_endpoint_valid": np.asarray([row["metric_endpoint_valid"] for row in self._rows], dtype=bool),
            "metric_endpoint_depth_spread_m": np.asarray([row["metric_endpoint_spread"] for row in self._rows], dtype=np.float32),
            "metric_endpoint_support": np.asarray([row["metric_endpoint_support"] for row in self._rows], dtype=np.int16),
            "gravity_camera_m_s2": np.asarray([row["gravity"] for row in self._rows], dtype=np.float32),
            "angular_velocity_camera_deg_s": np.asarray([row["angular"] for row in self._rows], dtype=np.float32),
            "failure": np.asarray([row["failure"] for row in self._rows]),
            "timings_ms": np.asarray([row["timings"] for row in self._rows], dtype=np.float32),
            "frame_count": np.asarray(len(self._rows), dtype=np.int64),
        }
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp.npz")
        try:
            np.savez_compressed(temporary, **payload)
            with np.load(temporary, allow_pickle=False) as data:
                validate_observation_archive(data)
            if self.path.exists() and not self.replace_existing:
                raise FileExistsError(f"Refusing to overwrite observation archive: {self.path}")
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
        self._closed = True
        return self.path

    def abort(self) -> None:
        self._rows.clear()
        self._closed = True

    def __enter__(self) -> "ObservationSequenceWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()
