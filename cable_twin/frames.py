"""Camera-independent contracts shared by live capture and offline replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class FrameKey:
    """Exact identity of one RGB-D frame."""

    source_id: str
    sequence_index: int
    source_position: int
    timestamp_ns: int

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must be non-empty")
        if self.sequence_index < 0:
            raise ValueError("sequence_index must be nonnegative")
        if self.source_position < 0:
            raise ValueError("source_position must be nonnegative")
        if self.timestamp_ns <= 0:
            raise ValueError("timestamp_ns must be positive")


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    """Rectified left-camera calibration for registered RGB and depth."""

    serial_number: int
    camera_model: str
    width_px: int
    height_px: int
    fps: float
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    coordinate_system: str = "RIGHT_HANDED_Y_UP"
    depth_unit: str = "METER"

    def __post_init__(self) -> None:
        if self.serial_number < 0:
            raise ValueError("serial_number must be nonnegative")
        if not self.camera_model.strip():
            raise ValueError("camera_model must be non-empty")
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("camera dimensions must be positive")
        if self.fps <= 0.0:
            raise ValueError("fps must be positive")
        if self.fx_px <= 0.0 or self.fy_px <= 0.0:
            raise ValueError("camera focal lengths must be positive")


@dataclass(frozen=True, slots=True)
class SourceDescriptor:
    source_id: str
    kind: str
    label: str
    calibration: CameraCalibration
    total_frames: int | None = None

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.label.strip():
            raise ValueError("source descriptor strings must be non-empty")
        if self.kind not in {"live", "svo"}:
            raise ValueError("source kind must be 'live' or 'svo'")
        if self.total_frames is not None and self.total_frames < 0:
            raise ValueError("total_frames must be nonnegative")


@dataclass(frozen=True, slots=True)
class RgbdFrame:
    """One immutable rectified BGR image and pixel-registered metric depth."""

    key: FrameKey
    host_received_ns: int
    bgr_u8: np.ndarray
    depth_m_f32: np.ndarray

    def __post_init__(self) -> None:
        if self.host_received_ns <= 0:
            raise ValueError("host_received_ns must be positive")
        if self.bgr_u8.dtype != np.uint8 or self.bgr_u8.ndim != 3:
            raise ValueError("bgr_u8 must be a uint8 HxWx3 array")
        if self.bgr_u8.shape[2] != 3:
            raise ValueError("bgr_u8 must have exactly three channels")
        if self.depth_m_f32.dtype != np.float32 or self.depth_m_f32.ndim != 2:
            raise ValueError("depth_m_f32 must be a float32 HxW array")
        if self.depth_m_f32.shape != self.bgr_u8.shape[:2]:
            raise ValueError("RGB and registered depth dimensions must match")
        if not self.bgr_u8.flags.c_contiguous:
            raise ValueError("bgr_u8 must be C-contiguous")
        if not self.depth_m_f32.flags.c_contiguous:
            raise ValueError("depth_m_f32 must be C-contiguous")
        self.bgr_u8.setflags(write=False)
        self.depth_m_f32.setflags(write=False)


@dataclass(frozen=True, slots=True)
class RecordingStatus:
    active: bool
    path: Path | None
    compression: str | None
    frame_count: int
    first_timestamp_ns: int | None
    last_timestamp_ns: int | None


class FrameSource(Protocol):
    @property
    def descriptor(self) -> SourceDescriptor:
        ...

    def read(self, timeout_s: float | None = None) -> RgbdFrame | None:
        ...

    def close(self) -> None:
        ...
