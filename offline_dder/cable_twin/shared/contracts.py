from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


OBSERVATION_SCHEMA_VERSION = 6
POINT_CLOUD_CURVE_METHOD = "zed_registered_depth_endpoint_capped_curves_v3"


def _readonly(array: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    value = np.ascontiguousarray(array, dtype=dtype)
    value.setflags(write=False)
    return value


@dataclass(frozen=True, slots=True)
class StereoCalibration:
    """Rectified ZED calibration in the left-camera metric frame."""

    width_px: int
    height_px: int
    fps: float
    left_intrinsics: np.ndarray
    right_intrinsics: np.ndarray
    baseline_m: float
    serial_number: int = 0
    camera_model: str = "ZED"
    imu_to_camera_transform: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        if self.width_px <= 0 or self.height_px <= 0 or self.fps <= 0.0:
            raise ValueError("Stereo image dimensions and FPS must be positive.")
        if self.baseline_m <= 0.0 or not np.isfinite(self.baseline_m):
            raise ValueError("Stereo baseline must be finite and positive.")
        for name in ("left_intrinsics", "right_intrinsics"):
            matrix = np.asarray(getattr(self, name), dtype=np.float64)
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                raise ValueError(f"{name} must be a finite 3x3 matrix.")
            if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
                raise ValueError(f"{name} must contain positive focal lengths.")
            object.__setattr__(self, name, _readonly(matrix, np.float64))
        transform = np.asarray(self.imu_to_camera_transform, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("imu_to_camera_transform must be a finite 4x4 matrix.")
        object.__setattr__(
            self, "imu_to_camera_transform", _readonly(transform, np.float64)
        )

    @property
    def projection_left(self) -> np.ndarray:
        projection = np.zeros((3, 4), dtype=np.float64)
        projection[:, :3] = self.left_intrinsics
        return projection

    @property
    def projection_right(self) -> np.ndarray:
        extrinsic = np.eye(3, 4, dtype=np.float64)
        extrinsic[0, 3] = -self.baseline_m
        return self.right_intrinsics @ extrinsic


@dataclass(frozen=True, slots=True)
class StereoFrame:
    sequence_index: int
    source_position: int
    timestamp_ns: int
    left_bgr: np.ndarray
    depth_m: np.ndarray | None
    imu_timestamp_ns: int | None = None
    gravity_camera_m_s2: np.ndarray | None = None
    angular_velocity_camera_deg_s: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.sequence_index < 0 or self.source_position < 0:
            raise ValueError("Frame indices must be nonnegative.")
        if self.timestamp_ns <= 0:
            raise ValueError("Frame timestamp must be positive.")
        left = np.asarray(self.left_bgr, dtype=np.uint8)
        if left.ndim != 3 or left.shape[2] != 3:
            raise ValueError("The registered left image must be a uint8 HxWx3 array.")
        object.__setattr__(self, "left_bgr", _readonly(left, np.uint8))
        if self.depth_m is not None:
            depth = np.asarray(self.depth_m, dtype=np.float32)
            if depth.shape != left.shape[:2]:
                raise ValueError("Registered ZED depth must match the left image size.")
            object.__setattr__(self, "depth_m", _readonly(depth, np.float32))
        for name in ("gravity_camera_m_s2", "angular_velocity_camera_deg_s"):
            value = getattr(self, name)
            if value is None:
                continue
            vector = np.asarray(value, dtype=np.float64)
            if vector.shape != (3,) or not np.all(np.isfinite(vector)):
                raise ValueError(f"{name} must be a finite 3-vector when present.")
            object.__setattr__(self, name, _readonly(vector, np.float64))


@dataclass(frozen=True, slots=True)
class PartialCurveObservation:
    """Disconnected visible cable segments plus independent cable endpoints."""

    points_xy: np.ndarray
    point_valid: np.ndarray
    segment_ids: np.ndarray
    endpoint_centers_xy: np.ndarray
    endpoint_valid: np.ndarray
    endpoint_component_count: int
    body_component_count: int

    def __post_init__(self) -> None:
        points = np.asarray(self.points_xy, dtype=np.float64)
        valid = np.asarray(self.point_valid, dtype=bool)
        segment_ids = np.asarray(self.segment_ids, dtype=np.int16)
        endpoints = np.asarray(self.endpoint_centers_xy, dtype=np.float64)
        endpoint_valid = np.asarray(self.endpoint_valid, dtype=bool)
        if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 2:
            raise ValueError("Partial curve points must have shape Nx2 with N >= 2.")
        if valid.shape != (len(points),) or segment_ids.shape != valid.shape:
            raise ValueError("Point validity and segment IDs must match partial curve points.")
        if np.any(valid & ~np.all(np.isfinite(points), axis=1)):
            raise ValueError("Valid partial curve points must be finite.")
        if np.any(valid & (segment_ids < 0)) or np.any(~valid & (segment_ids != -1)):
            raise ValueError("Segment IDs must be nonnegative exactly at valid points.")
        if endpoints.shape != (2, 2):
            raise ValueError("Endpoint observations must have shape 2x2.")
        if endpoint_valid.shape != (2,):
            raise ValueError("Endpoint validity must contain two flags.")
        if np.any(endpoint_valid & ~np.all(np.isfinite(endpoints), axis=1)):
            raise ValueError("Valid endpoint coordinates must be finite.")
        object.__setattr__(self, "points_xy", _readonly(points, np.float64))
        object.__setattr__(self, "point_valid", _readonly(valid, bool))
        object.__setattr__(self, "segment_ids", _readonly(segment_ids, np.int16))
        object.__setattr__(
            self, "endpoint_centers_xy", _readonly(endpoints, np.float64)
        )
        object.__setattr__(self, "endpoint_valid", _readonly(endpoint_valid, bool))


@dataclass(frozen=True, slots=True)
class ViewObservation:
    body_mask: np.ndarray
    endpoint_mask: np.ndarray
    curve: PartialCurveObservation
    failure: str | None = None

    def __post_init__(self) -> None:
        body = np.asarray(self.body_mask, dtype=np.uint8)
        endpoints = np.asarray(self.endpoint_mask, dtype=np.uint8)
        if body.ndim != 2 or endpoints.shape != body.shape:
            raise ValueError("Body and endpoint masks must be same-sized 2-D arrays.")
        object.__setattr__(self, "body_mask", _readonly(body, np.uint8))
        object.__setattr__(self, "endpoint_mask", _readonly(endpoints, np.uint8))


@dataclass(frozen=True, slots=True)
class SourceDescriptor:
    label: str
    kind: str
    calibration: StereoCalibration
    total_frames: int | None = None
    source_path: Path | None = None
