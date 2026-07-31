"""Small CPU-only OpenCV diagnostic viewer for synchronized perception.

This module deliberately knows nothing about the ZED SDK, CUDA, inference, or
recording.  The application supplies immutable CPU snapshots and interprets
the actions returned by :class:`OpenCvViewer`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


BODY_BGR = (20, 148, 255)
ENDPOINT_1_BGR = (255, 89, 13)
ENDPOINT_2_BGR = (77, 255, 38)
MASK_COLORS_BGR = (BODY_BGR, ENDPOINT_1_BGR, ENDPOINT_2_BGR)
MASK_LABELS = ("BODY", "END 1", "END 2")

_STATUS_HEIGHT_PX = 52
_PANE_GAP_PX = 4
_BACKGROUND_BGR = (12, 15, 18)
_STATUS_BGR = (24, 28, 33)
_TEXT_BGR = (235, 240, 245)
_MUTED_BGR = (148, 158, 168)
_WARNING_BGR = (70, 190, 255)
_RECORDING_BGR = (45, 55, 235)


class ViewerAction(Enum):
    """A request for the application; the viewer never performs the action."""

    NONE = "none"
    QUIT = "quit"
    TOGGLE_RECORDING = "toggle_recording"
    TOGGLE_PAUSE = "toggle_pause"
    STEP = "step"


@dataclass(frozen=True, slots=True)
class ViewerSnapshot:
    """One synchronized CPU-only preview and its compact runtime status.

    The mask order is cable body, endpoint set 1, and endpoint set 2.  Any
    nonzero mask pixel is foreground.  Arrays are made read-only so a queued
    preview cannot be changed underneath the compositor.
    """

    bgr_u8: np.ndarray
    depth_m_f32: np.ndarray
    masks_u8: tuple[np.ndarray, np.ndarray, np.ndarray]
    source_kind: str
    source_label: str
    frame_index: int
    source_position: int
    timestamp_ns: int
    total_frames: int | None = None
    pipeline_fps: float | None = None
    inference_ms: float | None = None
    latency_ms: float | None = None
    recording: bool = False
    recording_elapsed_s: float | None = None
    paused: bool = False
    skipped_frames: int = 0
    message: str = ""

    def __post_init__(self) -> None:
        _validate_image(
            "bgr_u8",
            self.bgr_u8,
            dtype=np.uint8,
            dimensions=3,
        )
        if self.bgr_u8.shape[2] != 3:
            raise ValueError("bgr_u8 must have shape HxWx3")
        _validate_image(
            "depth_m_f32",
            self.depth_m_f32,
            dtype=np.float32,
            dimensions=2,
        )
        image_shape = self.bgr_u8.shape[:2]
        if self.depth_m_f32.shape != image_shape:
            raise ValueError("registered depth must have the same HxW as bgr_u8")

        if not isinstance(self.masks_u8, tuple) or len(self.masks_u8) != 3:
            raise ValueError("masks_u8 must be a tuple of exactly three masks")
        for index, mask in enumerate(self.masks_u8):
            _validate_image(
                f"masks_u8[{index}]",
                mask,
                dtype=np.uint8,
                dimensions=2,
            )
            if mask.shape != image_shape:
                raise ValueError(
                    f"masks_u8[{index}] must have the same HxW as bgr_u8"
                )

        kind = self.source_kind.strip().lower()
        if kind not in {"live", "svo"}:
            raise ValueError("source_kind must be 'live' or 'svo'")
        if not self.source_label.strip():
            raise ValueError("source_label must be non-empty")
        if self.frame_index < 0 or self.source_position < 0:
            raise ValueError("frame indices must be nonnegative")
        if self.timestamp_ns <= 0:
            raise ValueError("timestamp_ns must be positive")
        if self.total_frames is not None and self.total_frames < 0:
            raise ValueError("total_frames must be nonnegative")
        _validate_optional_metric("pipeline_fps", self.pipeline_fps, positive=True)
        _validate_optional_metric("inference_ms", self.inference_ms)
        _validate_optional_metric("latency_ms", self.latency_ms)
        _validate_optional_metric("recording_elapsed_s", self.recording_elapsed_s)
        if self.skipped_frames < 0:
            raise ValueError("skipped_frames must be nonnegative")

        object.__setattr__(self, "source_kind", kind)
        self.bgr_u8.setflags(write=False)
        self.depth_m_f32.setflags(write=False)
        for mask in self.masks_u8:
            mask.setflags(write=False)


def _validate_image(
    name: str,
    value: object,
    *,
    dtype: np.dtype,
    dimensions: int,
) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a CPU numpy.ndarray")
    if value.dtype != dtype or value.ndim != dimensions:
        raise ValueError(
            f"{name} must be a {np.dtype(dtype).name} {dimensions}-D array"
        )
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")


def _validate_optional_metric(
    name: str,
    value: float | None,
    *,
    positive: bool = False,
) -> None:
    if value is None:
        return
    number = float(value)
    valid = np.isfinite(number) and (number > 0.0 if positive else number >= 0.0)
    if not valid:
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")


def _validate_composition_settings(
    width_px: int,
    depth_min_m: float,
    depth_max_m: float,
    overlay_alpha: float,
) -> None:
    if int(width_px) < 320:
        raise ValueError("width_px must be at least 320")
    if not 0.0 < float(depth_min_m) < float(depth_max_m):
        raise ValueError("depth range must satisfy 0 < minimum < maximum")
    if not 0.0 <= float(overlay_alpha) <= 1.0:
        raise ValueError("overlay_alpha must be in [0, 1]")


def compose_viewer_frame(
    snapshot: ViewerSnapshot,
    *,
    width_px: int = 1280,
    depth_min_m: float = 0.20,
    depth_max_m: float = 3.00,
    overlay_alpha: float = 0.58,
) -> np.ndarray:
    """Compose a deterministic two-pane BGR image without GUI side effects."""

    if not isinstance(snapshot, ViewerSnapshot):
        raise TypeError("snapshot must be a ViewerSnapshot")
    _validate_composition_settings(
        width_px,
        depth_min_m,
        depth_max_m,
        overlay_alpha,
    )

    output_width = int(width_px)
    left_width = (output_width - _PANE_GAP_PX) // 2
    right_width = output_width - _PANE_GAP_PX - left_width
    source_height, source_width = snapshot.bgr_u8.shape[:2]
    content_height = max(
        1,
        int(round(source_height * min(left_width, right_width) / source_width)),
    )

    left_masks = _resize_masks(snapshot.masks_u8, left_width, content_height)
    right_masks = (
        left_masks
        if right_width == left_width
        else _resize_masks(snapshot.masks_u8, right_width, content_height)
    )
    rgb_panel = cv2.resize(
        snapshot.bgr_u8,
        (left_width, content_height),
        interpolation=cv2.INTER_AREA,
    )
    rgb_panel = np.ascontiguousarray(rgb_panel)
    _blend_masks(rgb_panel, left_masks, float(overlay_alpha))
    _draw_mask_contours(rgb_panel, left_masks, thickness=1)

    # Depth is diagnostic display data here.  Resize the scalar image before
    # colorization to avoid allocating and processing a full-resolution BGR
    # depth view only to immediately discard most of its pixels.
    display_depth = cv2.resize(
        snapshot.depth_m_f32,
        (right_width, content_height),
        interpolation=cv2.INTER_NEAREST,
    )
    depth_panel = _colorize_depth(
        display_depth,
        float(depth_min_m),
        float(depth_max_m),
    )
    depth_panel = np.ascontiguousarray(depth_panel)
    _draw_mask_contours(depth_panel, right_masks, thickness=2)

    _draw_panel_label(rgb_panel, "RGB + PIDNET")
    _draw_panel_label(
        depth_panel,
        f"REGISTERED DEPTH  {depth_min_m:.2f}-{depth_max_m:.2f} m",
    )

    canvas = np.full(
        (_STATUS_HEIGHT_PX + content_height, output_width, 3),
        _BACKGROUND_BGR,
        dtype=np.uint8,
    )
    canvas[_STATUS_HEIGHT_PX:, :left_width] = rgb_panel
    right_x = left_width + _PANE_GAP_PX
    canvas[_STATUS_HEIGHT_PX:, right_x:] = depth_panel
    _draw_status(canvas, snapshot)
    return np.ascontiguousarray(canvas)


def _resize_masks(
    masks: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(
        np.ascontiguousarray(
            cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        )
        for mask in masks
    )


def _blend_masks(
    panel: np.ndarray,
    masks: tuple[np.ndarray, np.ndarray, np.ndarray],
    alpha: float,
) -> None:
    if alpha <= 0.0:
        return
    for mask, color in zip(masks, MASK_COLORS_BGR):
        foreground = mask != 0
        if not np.any(foreground):
            continue
        source = panel[foreground].astype(np.float32)
        blended = source * (1.0 - alpha) + np.asarray(color) * alpha
        panel[foreground] = np.clip(blended + 0.5, 0.0, 255.0).astype(np.uint8)


def _draw_mask_contours(
    panel: np.ndarray,
    masks: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    thickness: int,
) -> None:
    for mask, color in zip(masks, MASK_COLORS_BGR):
        contours, _hierarchy = cv2.findContours(
            np.ascontiguousarray(mask),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if contours:
            cv2.drawContours(
                panel,
                contours,
                -1,
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )


def _colorize_depth(
    depth_m_f32: np.ndarray,
    depth_min_m: float,
    depth_max_m: float,
) -> np.ndarray:
    valid = np.isfinite(depth_m_f32) & (depth_m_f32 > 0.0)
    normalized = np.zeros(depth_m_f32.shape, dtype=np.float32)
    np.subtract(depth_m_f32, depth_min_m, out=normalized, where=valid)
    normalized[valid] /= depth_max_m - depth_min_m
    np.clip(normalized, 0.0, 1.0, out=normalized)
    depth_u8 = np.rint(normalized * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return np.ascontiguousarray(color)


def _draw_panel_label(panel: np.ndarray, label: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.43
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        font,
        scale,
        thickness,
    )
    cv2.rectangle(
        panel,
        (0, 0),
        (text_width + 16, text_height + baseline + 10),
        (8, 10, 12),
        -1,
    )
    cv2.putText(
        panel,
        label,
        (8, text_height + 5),
        font,
        scale,
        _TEXT_BGR,
        thickness,
        cv2.LINE_AA,
    )


def _draw_status(canvas: np.ndarray, snapshot: ViewerSnapshot) -> None:
    canvas[:_STATUS_HEIGHT_PX] = _STATUS_BGR
    font = cv2.FONT_HERSHEY_SIMPLEX
    source = f"{snapshot.source_kind.upper()}  {_shorten(snapshot.source_label, 30)}"
    frame = f"frame {snapshot.frame_index}"
    if snapshot.total_frames is not None:
        frame += f"  [{snapshot.source_position + 1}/{snapshot.total_frames}]"
    elif snapshot.source_position != snapshot.frame_index:
        frame += f"  [source {snapshot.source_position}]"
    metrics = [source, frame, f"ts {snapshot.timestamp_ns} ns"]
    if snapshot.pipeline_fps is not None:
        metrics.append(f"{snapshot.pipeline_fps:.1f} FPS")
    if snapshot.inference_ms is not None:
        metrics.append(f"PID {snapshot.inference_ms:.1f} ms")
    if snapshot.latency_ms is not None:
        metrics.append(f"age {snapshot.latency_ms:.1f} ms")
    cv2.putText(
        canvas,
        "  |  ".join(metrics),
        (10, 18),
        font,
        0.42,
        _TEXT_BGR,
        1,
        cv2.LINE_AA,
    )

    cursor_x = 10
    if snapshot.recording:
        recording = "REC"
        if snapshot.recording_elapsed_s is not None:
            recording += f" {_format_duration(snapshot.recording_elapsed_s)}"
        cursor_x = _draw_status_item(
            canvas,
            cursor_x,
            recording,
            _RECORDING_BGR,
        )
    else:
        cursor_x = _draw_status_item(canvas, cursor_x, "REC OFF", _MUTED_BGR)
    if snapshot.paused:
        cursor_x = _draw_status_item(canvas, cursor_x, "PAUSED", _WARNING_BGR)
    if snapshot.skipped_frames:
        cursor_x = _draw_status_item(
            canvas,
            cursor_x,
            f"skipped {snapshot.skipped_frames}",
            _WARNING_BGR,
        )
    if snapshot.message.strip():
        cursor_x = _draw_status_item(
            canvas,
            cursor_x,
            _shorten(snapshot.message.replace("\n", " "), 48),
            _TEXT_BGR,
        )

    legend_width = 0
    legend_sizes: list[tuple[int, str, tuple[int, int, int]]] = []
    for label, color in zip(MASK_LABELS, MASK_COLORS_BGR):
        (text_width, _height), _baseline = cv2.getTextSize(label, font, 0.38, 1)
        item_width = 15 + text_width + 14
        legend_sizes.append((item_width, label, color))
        legend_width += item_width
    legend_x = max(cursor_x + 12, canvas.shape[1] - legend_width - 6)
    if legend_x + legend_width <= canvas.shape[1]:
        for item_width, label, color in legend_sizes:
            cv2.rectangle(canvas, (legend_x, 32), (legend_x + 9, 41), color, -1)
            cv2.putText(
                canvas,
                label,
                (legend_x + 14, 42),
                font,
                0.38,
                _MUTED_BGR,
                1,
                cv2.LINE_AA,
            )
            legend_x += item_width


def _draw_status_item(
    canvas: np.ndarray,
    cursor_x: int,
    text: str,
    color: tuple[int, int, int],
) -> int:
    cv2.circle(canvas, (cursor_x + 4, 37), 3, color, -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        text,
        (cursor_x + 12, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        color,
        1,
        cv2.LINE_AA,
    )
    (text_width, _height), _baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        1,
    )
    return cursor_x + 12 + text_width + 14


def _format_duration(seconds: float) -> str:
    whole_seconds = max(0, int(seconds))
    minutes, remaining = divmod(whole_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{remaining:02d}"
    return f"{minutes:02d}:{remaining:02d}"


def _shorten(value: str, limit: int) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


def viewer_action_from_key(key: int) -> ViewerAction:
    """Translate one OpenCV key code without changing application state."""

    if key < 0:
        return ViewerAction.NONE
    key = int(key) & 0xFF
    if key in (27, ord("q"), ord("Q")):
        return ViewerAction.QUIT
    if key in (ord("r"), ord("R")):
        return ViewerAction.TOGGLE_RECORDING
    if key == ord(" "):
        return ViewerAction.TOGGLE_PAUSE
    if key in (ord("n"), ord("N")):
        return ViewerAction.STEP
    return ViewerAction.NONE


class OpenCvViewer:
    """Thin window sink with no queue, worker, recorder, or source ownership."""

    def __init__(
        self,
        *,
        window_name: str = "Cable Twin Perception",
        width_px: int = 1280,
        depth_min_m: float = 0.20,
        depth_max_m: float = 3.00,
        overlay_alpha: float = 0.58,
    ) -> None:
        if not str(window_name).strip():
            raise ValueError("window_name must be non-empty")
        _validate_composition_settings(
            width_px,
            depth_min_m,
            depth_max_m,
            overlay_alpha,
        )
        self.window_name = str(window_name)
        self.width_px = int(width_px)
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)
        self.overlay_alpha = float(overlay_alpha)
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def show(self, snapshot: ViewerSnapshot) -> ViewerAction:
        """Display the latest supplied snapshot and return one requested action."""

        canvas = compose_viewer_frame(
            snapshot,
            width_px=self.width_px,
            depth_min_m=self.depth_min_m,
            depth_max_m=self.depth_max_m,
            overlay_alpha=self.overlay_alpha,
        )
        if not self._open:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(self.window_name, canvas.shape[1], canvas.shape[0])
            self._open = True
        cv2.imshow(self.window_name, canvas)
        return self.poll(delay_ms=1)

    def poll(self, delay_ms: int = 10) -> ViewerAction:
        """Process window events without recomposing the current frame."""

        if not self._open:
            return ViewerAction.NONE
        try:
            action = viewer_action_from_key(cv2.waitKey(max(1, int(delay_ms))))
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1.0:
                self._open = False
                return ViewerAction.QUIT
            return action
        except cv2.error:
            self._open = False
            return ViewerAction.QUIT

    def close(self) -> None:
        if not self._open:
            return
        try:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)
        except cv2.error:
            pass
        finally:
            self._open = False

    def __enter__(self) -> OpenCvViewer:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


__all__ = [
    "OpenCvViewer",
    "ViewerAction",
    "ViewerSnapshot",
    "compose_viewer_frame",
    "viewer_action_from_key",
]
