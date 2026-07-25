"""Asynchronous RGB-depth tracking with best-effort full-rate visualization."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
import sys
import threading
import time
import tomllib
import traceback

import cv2
import numpy as np
import pyzed.sl as sl


SOURCE_DIR = Path(__file__).resolve().parent
APPLICATION_DIR = SOURCE_DIR.parent
REPOSITORY_DIR = APPLICATION_DIR.parent
NN_SOURCE_DIR = REPOSITORY_DIR / "NN_collection_training" / "source"
if str(NN_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(NN_SOURCE_DIR))

from cable_pidnet import PidNetSegmenter  # noqa: E402
from observation import (  # noqa: E402
    CameraModel,
    FrameObservation,
    ObservationBuilder,
    ObservationConfig,
    ObservationProfile,
)
from particle_filter import (  # noqa: E402
    BatchedCableParticleFilter,
    ParticleFeatures,
    ParticleFilterConfig,
    ParticleFilterFrame,
    ParticleFilterProfile,
)
from split_viewer import SplitPointCloudViewer  # noqa: E402


RESOLUTIONS = {
    "HD2K": sl.RESOLUTION.HD2K,
    "HD1200": sl.RESOLUTION.HD1200,
    "HD1080": sl.RESOLUTION.HD1080,
    "HD720": sl.RESOLUTION.HD720,
    "SVGA": sl.RESOLUTION.SVGA,
    "VGA": sl.RESOLUTION.VGA,
}
DEPTH_MODES = {
    "NEURAL": sl.DEPTH_MODE.NEURAL,
    "NEURAL_LIGHT": sl.DEPTH_MODE.NEURAL_LIGHT,
    "ULTRA": sl.DEPTH_MODE.ULTRA,
    "QUALITY": sl.DEPTH_MODE.QUALITY,
    "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
}

# Float RGB for the OpenGL point cloud and uint8 BGR for the camera panel.
CABLE_RGB = np.asarray((1.00, 0.58, 0.08), dtype=np.float32)
ENDPOINT_1_RGB = np.asarray((0.05, 0.35, 1.00), dtype=np.float32)
ENDPOINT_2_RGB = np.asarray((0.15, 1.00, 0.30), dtype=np.float32)
CROSSING_RGB = np.asarray((1.00, 1.00, 0.00), dtype=np.float32)
CABLE_BGR = np.asarray((20, 148, 255), dtype=np.uint8)
ENDPOINT_1_BGR = np.asarray((255, 89, 13), dtype=np.uint8)
ENDPOINT_2_BGR = np.asarray((77, 255, 38), dtype=np.uint8)
CROSSING_BGR = np.asarray((0, 255, 255), dtype=np.uint8)
SKELETON_MASK_RGB = np.asarray((38, 43, 49), dtype=np.uint8)
SKELETON_LINE_RGB = np.asarray((238, 244, 250), dtype=np.uint8)
SKELETON_NODE_RGB = np.asarray((55, 205, 255), dtype=np.uint8)
SKELETON_BRANCH_RGB = np.asarray((255, 72, 88), dtype=np.uint8)


def _packed_rgb_float(rgb: np.ndarray) -> np.float32:
    channels = np.rint(np.asarray(rgb) * 255.0).astype(np.uint32)
    packed = channels[0] | (channels[1] << 8) | (channels[2] << 16)
    return np.asarray(packed, dtype=np.uint32).view(np.float32)[()]


CABLE_PACKED = _packed_rgb_float(CABLE_RGB)
ENDPOINT_1_PACKED = _packed_rgb_float(ENDPOINT_1_RGB)
ENDPOINT_2_PACKED = _packed_rgb_float(ENDPOINT_2_RGB)
CROSSING_PACKED = _packed_rgb_float(CROSSING_RGB)


@dataclass(frozen=True)
class CaptureProfile:
    grab_ms: float
    image_retrieve_ms: float
    image_copy_ms: float
    depth_retrieve_ms: float
    depth_copy_ms: float
    cloud_retrieve_ms: float
    cloud_copy_ms: float
    submit_interval_ms: float


@dataclass(frozen=True)
class TrackingInput:
    """One immutable frame ownership token passed through the latest-frame mailbox."""

    buffer_index: int
    frame_index: int
    captured_at: float
    published_at: float
    bgr: np.ndarray
    depth: np.ndarray
    render_cloud: np.ndarray | None
    render_depth: np.ndarray | None
    render_captured_at: float | None
    requested_features: ParticleFeatures
    capture_profile: CaptureProfile


@dataclass(frozen=True)
class TrackingResult:
    """Algorithm output retained only until visualization accepts or drops it."""

    buffer_index: int
    frame_index: int
    captured_at: float
    completed_at: float
    bgr: np.ndarray
    render_cloud: np.ndarray | None
    render_depth: np.ndarray | None
    render_captured_at: float | None
    masks: tuple[np.ndarray, ...]
    observation: FrameObservation
    particle_filter: ParticleFilterFrame
    inference_ms: float
    tracking_ms: float
    schedule_wait_ms: float
    capture_profile: CaptureProfile


@dataclass(frozen=True)
class SegmentationResult:
    frame_index: int
    captured_at: float
    rgb_image: np.ndarray
    skeleton_image: np.ndarray
    vertices: np.ndarray
    cloud_captured_at: float
    stats: dict[str, int | float]
    scene_center: np.ndarray
    scene_radius: float
    inference_ms: float
    processing_ms: float
    cloud_processing_ms: float
    overlay_processing_ms: float
    latency_ms: float
    source_age_ms: float
    observation: FrameObservation
    particle_filter: ParticleFilterFrame


@dataclass(frozen=True)
class PreparedPointCloud:
    vertices: np.ndarray
    stats: dict[str, int | float]
    scene_center: np.ndarray
    scene_radius: float
    captured_at: float
    processing_ms: float


def load_config(path: Path) -> dict:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    for section in ("camera", "viewer", "pidnet"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Configuration is missing [{section}].")
    return config


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_DIR / path
    return path.resolve()


def _boolean_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(mask)
    if array.shape != shape:
        array = cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(array > 0)


def _scene_bounds(xyz: np.ndarray) -> tuple[np.ndarray, float]:
    """Estimate only the view framing; this never reduces rendered points."""

    if len(xyz) == 0:
        return np.asarray((0.0, 0.0, -2.0), dtype=np.float32), 2.0
    sample = xyz
    if len(sample) > 20_000:
        indices = np.linspace(0, len(sample) - 1, 20_000, dtype=np.int64)
        sample = sample[indices]
    center = np.median(sample, axis=0).astype(np.float32)
    distance = np.linalg.norm(sample - center, axis=1)
    radius = float(np.percentile(distance, 95.0)) if len(distance) else 2.0
    return center, float(np.clip(radius, 0.25, 20.0))


def full_point_cloud_vertices(
    point_data: np.ndarray,
    masks: tuple[np.ndarray, ...],
    stride: int = 1,
) -> tuple[np.ndarray, dict[str, int], np.ndarray, float]:
    """Color a regular viewer-only sample of the ZED XYZRGBA cloud."""

    cloud = np.asarray(point_data)
    if cloud.ndim != 3 or cloud.shape[2] < 4:
        raise ValueError(f"ZED point cloud must have shape HxWx4; got {cloud.shape}.")
    if len(masks) < 4:
        raise ValueError(
            f"PIDNet must return cable, two endpoint, and crossing masks; got {len(masks)}."
        )

    height, width = cloud.shape[:2]
    source_shape = (height, width)
    sample_stride = max(1, int(stride))
    cable = _boolean_mask(masks[0], source_shape)[
        ::sample_stride, ::sample_stride
    ].reshape(-1)
    endpoint_1 = _boolean_mask(masks[1], source_shape)[
        ::sample_stride, ::sample_stride
    ].reshape(-1)
    endpoint_2 = _boolean_mask(masks[2], source_shape)[
        ::sample_stride, ::sample_stride
    ].reshape(-1)
    crossing = _boolean_mask(masks[3], source_shape)[
        ::sample_stride, ::sample_stride
    ].reshape(-1)

    flat = cloud[::sample_stride, ::sample_stride, :4].reshape(-1, 4)
    xyz_all = flat[:, :3]
    finite = np.all(np.isfinite(xyz_all), axis=1)
    valid_count = int(np.count_nonzero(finite))
    # Keep ZED's packed RGB float intact. The OpenGL vertex shader decodes it,
    # avoiding three CPU color floats and a full-frame RGB expansion.
    vertices = np.empty((valid_count, 4), dtype=np.float32)
    all_valid = valid_count == len(flat)
    if all_valid:
        vertices[:] = flat[:, :4]
        valid_cable = cable
        valid_endpoint_1 = endpoint_1
        valid_endpoint_2 = endpoint_2
        valid_crossing = crossing
    else:
        vertices[:, :3] = xyz_all[finite]
        vertices[:, 3] = flat[finite, 3]
        valid_cable = cable[finite]
        valid_endpoint_1 = endpoint_1[finite]
        valid_endpoint_2 = endpoint_2[finite]
        valid_crossing = crossing[finite]

    # Lowest-to-highest priority: cable, endpoint 2, endpoint 1, crossing.
    vertices[valid_cable, 3] = CABLE_PACKED
    vertices[valid_endpoint_2, 3] = ENDPOINT_2_PACKED
    vertices[valid_endpoint_1, 3] = ENDPOINT_1_PACKED
    vertices[valid_crossing, 3] = CROSSING_PACKED
    stats = {
        "total_pixels": int(len(flat)),
        "source_pixels": int(height * width),
        "point_cloud_stride": sample_stride,
        "valid_points": valid_count,
        "invalid_points": int(finite.size - valid_count),
        "cable_points": int(np.count_nonzero(valid_cable)),
        "endpoint_1_points": int(np.count_nonzero(valid_endpoint_1)),
        "endpoint_2_points": int(np.count_nonzero(valid_endpoint_2)),
        "crossing_points": int(np.count_nonzero(valid_crossing)),
    }
    center, radius = _scene_bounds(vertices[:, :3])
    return vertices, stats, center, radius


def depth_point_cloud_vertices(
    depth: np.ndarray,
    bgr: np.ndarray,
    masks: tuple[np.ndarray, ...],
    camera_model: CameraModel,
    stride: int = 1,
) -> tuple[np.ndarray, dict[str, int], np.ndarray, float]:
    """Unproject the existing depth frame for a viewer-only point cloud."""

    depth_array = np.asarray(depth, dtype=np.float32)
    image = np.asarray(bgr, dtype=np.uint8)
    if depth_array.ndim != 2 or image.shape[:2] != depth_array.shape:
        raise ValueError(
            f"Depth/RGB viewer inputs disagree: {depth_array.shape} vs {image.shape}."
        )
    if len(masks) < 4:
        raise ValueError(
            f"PIDNet must return cable, two endpoint, and crossing masks; got {len(masks)}."
        )

    height, width = depth_array.shape
    source_shape = (height, width)
    sample_stride = max(1, int(stride))
    cable = _boolean_mask(masks[0], source_shape)[
        ::sample_stride, ::sample_stride
    ].reshape(-1)
    endpoint_1 = _boolean_mask(masks[1], source_shape)[
        ::sample_stride, ::sample_stride
    ].reshape(-1)
    endpoint_2 = _boolean_mask(masks[2], source_shape)[
        ::sample_stride, ::sample_stride
    ].reshape(-1)
    crossing = _boolean_mask(masks[3], source_shape)[
        ::sample_stride, ::sample_stride
    ].reshape(-1)
    sampled_depth = depth_array[::sample_stride, ::sample_stride]
    sampled_image = image[::sample_stride, ::sample_stride]
    flat_depth = sampled_depth.reshape(-1)
    finite = np.isfinite(flat_depth) & (flat_depth > 0.0)
    pixel_y, pixel_x = np.meshgrid(
        np.arange(0, height, sample_stride, dtype=np.float32),
        np.arange(0, width, sample_stride, dtype=np.float32),
        indexing="ij",
    )
    selected_depth = flat_depth[finite]
    selected_x = pixel_x.reshape(-1)[finite]
    selected_y = pixel_y.reshape(-1)[finite]

    vertices = np.empty((len(selected_depth), 4), dtype=np.float32)
    vertices[:, 0] = (
        (selected_x - float(camera_model.cx))
        * selected_depth
        / float(camera_model.fx)
    )
    vertices[:, 1] = (
        (selected_y - float(camera_model.cy))
        * selected_depth
        / float(camera_model.image_y_sign * camera_model.fy)
    )
    vertices[:, 2] = selected_depth / float(camera_model.forward_sign)
    selected_bgr = sampled_image.reshape(-1, 3)[finite].astype(np.uint32)
    packed_rgb = (
        selected_bgr[:, 2]
        | (selected_bgr[:, 1] << 8)
        | (selected_bgr[:, 0] << 16)
    )
    vertices[:, 3] = packed_rgb.view(np.float32)
    vertices[cable[finite], 3] = CABLE_PACKED
    vertices[endpoint_2[finite], 3] = ENDPOINT_2_PACKED
    vertices[endpoint_1[finite], 3] = ENDPOINT_1_PACKED
    vertices[crossing[finite], 3] = CROSSING_PACKED
    stats = {
        "total_pixels": int(flat_depth.size),
        "source_pixels": int(height * width),
        "point_cloud_stride": sample_stride,
        "valid_points": int(np.count_nonzero(finite)),
        "invalid_points": int(finite.size - np.count_nonzero(finite)),
        "cable_points": int(np.count_nonzero(finite & cable)),
        "endpoint_1_points": int(np.count_nonzero(finite & endpoint_1)),
        "endpoint_2_points": int(np.count_nonzero(finite & endpoint_2)),
        "crossing_points": int(np.count_nonzero(finite & crossing)),
    }
    center, radius = _scene_bounds(vertices[:, :3])
    return vertices, stats, center, radius


def segmentation_overlay(
    bgr: np.ndarray,
    masks: tuple[np.ndarray, ...],
    alpha: float,
    maximum_width: int,
) -> np.ndarray:
    """Render the same deterministic class colors over the left camera image."""

    image = np.ascontiguousarray(np.asarray(bgr, dtype=np.uint8)[:, :, :3])
    source_shape = image.shape[:2]
    target_width = min(source_shape[1], max(1, int(maximum_width)))
    target_height = max(
        1,
        int(round(source_shape[0] * target_width / source_shape[1])),
    )
    target_size = (target_width, target_height)
    if target_size != (source_shape[1], source_shape[0]):
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    def display_mask(mask: np.ndarray) -> np.ndarray:
        selected = _boolean_mask(mask, source_shape)
        if target_size != (source_shape[1], source_shape[0]):
            selected = cv2.resize(
                selected.astype(np.uint8),
                target_size,
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        return selected

    cable = display_mask(masks[0])
    endpoint_1 = display_mask(masks[1])
    endpoint_2 = display_mask(masks[2])
    crossing = display_mask(masks[3])
    output = image.copy()
    blend = float(np.clip(alpha, 0.0, 1.0))
    for selected, color in (
        (cable, CABLE_BGR),
        (endpoint_2, ENDPOINT_2_BGR),
        (endpoint_1, ENDPOINT_1_BGR),
        (crossing, CROSSING_BGR),
    ):
        if not np.any(selected):
            continue
        output[selected] = np.rint(
            (1.0 - blend) * image[selected].astype(np.float32)
            + blend * color.astype(np.float32)
        ).astype(np.uint8)
    return np.ascontiguousarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))


def skeleton_diagnostic_image(
    masks: tuple[np.ndarray, ...],
    observation: FrameObservation,
    maximum_width: int,
) -> np.ndarray:
    """Render the exact thinned mask and compressed graph used by tracking."""

    if len(masks) < 4:
        raise ValueError(
            f"Skeleton diagnostics require four PIDNet masks; got {len(masks)}."
        )
    source_shape = np.asarray(masks[0]).shape[:2]
    if len(source_shape) != 2:
        raise ValueError(
            f"Cable mask must be two-dimensional; got {source_shape}."
        )
    target_width = min(source_shape[1], max(1, int(maximum_width)))
    target_height = max(
        1,
        int(round(source_shape[0] * target_width / source_shape[1])),
    )
    target_size = (target_width, target_height)

    def display_mask(mask: np.ndarray) -> np.ndarray:
        selected = _boolean_mask(mask, source_shape)
        if target_size != (source_shape[1], source_shape[0]):
            selected = cv2.resize(
                selected.astype(np.uint8),
                target_size,
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        return selected

    cable = display_mask(masks[0])
    crossing = display_mask(masks[3])
    output = np.full((target_height, target_width, 3), 8, dtype=np.uint8)
    output[cable] = SKELETON_MASK_RGB

    debug = observation.skeleton_debug
    if debug is None:
        return np.ascontiguousarray(output)

    height, width = target_height, target_width
    scale_x = target_width / source_shape[1]
    scale_y = target_height / source_shape[0]
    skeleton_pixels = np.asarray(
        debug.pixels_xy,
        dtype=np.float32,
    ).reshape(-1, 2)
    if len(skeleton_pixels) > 0:
        skeleton_x = np.rint(skeleton_pixels[:, 0] * scale_x).astype(np.int32)
        skeleton_y = np.rint(skeleton_pixels[:, 1] * scale_y).astype(np.int32)
        valid = (
            (skeleton_x >= 0)
            & (skeleton_x < width)
            & (skeleton_y >= 0)
            & (skeleton_y < height)
        )
        output[skeleton_y[valid], skeleton_x[valid]] = SKELETON_LINE_RGB

    def draw_points(points_xy: np.ndarray, color: np.ndarray, radius: int) -> None:
        for point in np.asarray(points_xy, dtype=np.float32).reshape(-1, 2):
            if not np.all(np.isfinite(point)):
                continue
            x = int(round(float(point[0]) * scale_x))
            y = int(round(float(point[1]) * scale_y))
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(
                    output,
                    (x, y),
                    radius,
                    tuple(int(value) for value in color),
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

    draw_points(debug.node_pixels_xy, SKELETON_NODE_RGB, 5)
    draw_points(debug.branch_pixels_xy, SKELETON_BRANCH_RGB, 3)
    for cable_index, cable_observation in enumerate(observation.cables):
        endpoint_color = (
            np.rint(ENDPOINT_1_RGB * 255.0).astype(np.uint8)
            if cable_index == 0
            else np.rint(ENDPOINT_2_RGB * 255.0).astype(np.uint8)
        )
        draw_points(
            cable_observation.endpoint_pixels_xy[
                cable_observation.endpoint_visible
            ],
            endpoint_color,
            8,
        )

    crossing_u8 = crossing.astype(np.uint8)
    if np.any(crossing_u8):
        contours, _hierarchy = cv2.findContours(
            crossing_u8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            output,
            contours,
            contourIdx=-1,
            color=tuple(
                int(value)
                for value in np.rint(CROSSING_RGB * 255.0).astype(np.uint8)
            ),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    return np.ascontiguousarray(output)


def open_zed(camera_config: dict) -> tuple[sl.Camera, sl.RuntimeParameters]:
    resolution_name = str(camera_config.get("resolution", "HD720")).upper()
    depth_name = str(camera_config.get("depth_mode", "NEURAL")).upper()
    if resolution_name not in RESOLUTIONS:
        raise ValueError(f"Unsupported ZED resolution: {resolution_name}")
    if depth_name not in DEPTH_MODES:
        raise ValueError(f"Unsupported ZED depth mode: {depth_name}")

    init = sl.InitParameters()
    init.camera_resolution = RESOLUTIONS[resolution_name]
    init.camera_fps = int(camera_config.get("fps", 60))
    init.depth_mode = DEPTH_MODES[depth_name]
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(
            f"Could not open ZED camera: {status}. Close any other ZED viewer or "
            "collection application before launching this viewer."
        )

    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = int(camera_config.get("confidence", 100))
    runtime.texture_confidence_threshold = int(camera_config.get("texture_confidence", 100))
    runtime.remove_saturated_areas = False
    if hasattr(runtime, "enable_fill_mode"):
        runtime.enable_fill_mode = bool(camera_config.get("fill", True))
    return zed, runtime


def configure_viewer_from_zed(zed: sl.Camera, viewer: SplitPointCloudViewer) -> None:
    try:
        information = zed.get_camera_information()
        left = information.camera_configuration.calibration_parameters.left_cam
        viewer.set_camera_fov(float(left.v_fov))
    except Exception as exc:
        print(f"Using default viewer FOV: {exc}")


def camera_frame_shape(zed: sl.Camera) -> tuple[int, int]:
    information = zed.get_camera_information()
    resolution = information.camera_configuration.resolution
    height, width = int(resolution.height), int(resolution.width)
    if height <= 0 or width <= 0:
        raise RuntimeError(f"ZED returned an invalid camera resolution: {width}x{height}.")
    return height, width


def camera_model_from_zed(zed: sl.Camera) -> CameraModel:
    information = zed.get_camera_information()
    resolution = information.camera_configuration.resolution
    left = information.camera_configuration.calibration_parameters.left_cam
    return CameraModel(
        fx=float(left.fx),
        fy=float(left.fy),
        cx=float(left.cx),
        cy=float(left.cy),
        width=int(resolution.width),
        height=int(resolution.height),
        # RIGHT_HANDED_Y_UP uses -Z as camera-forward and +Y upward.
        forward_sign=-1.0,
        image_y_sign=-1.0,
    )


class AsyncSegmentationPipeline:
    """Independent capture, tracking, and best-effort visualization stages."""

    def __init__(
        self,
        zed: sl.Camera,
        runtime: sl.RuntimeParameters,
        segmenter: PidNetSegmenter,
        thresholds: tuple[float, ...],
        overlay_alpha: float,
        observation_builder: ObservationBuilder,
        particle_filter: BatchedCableParticleFilter,
        visualization_max_fps: float,
        visualization_image_width: int,
        visualization_point_cloud_stride: int,
        visualization_enabled: bool = True,
        visualization_source: str = "zed",
    ):
        self.zed = zed
        self.runtime = runtime
        self.segmenter = segmenter
        self.thresholds = thresholds
        self.overlay_alpha = float(overlay_alpha)
        self.observation_builder = observation_builder
        self.particle_filter = particle_filter
        self.visualization_image_width = max(
            1,
            int(visualization_image_width),
        )
        self.visualization_point_cloud_stride = max(
            1,
            int(visualization_point_cloud_stride),
        )
        self.visualization_enabled = bool(visualization_enabled)
        self.visualization_source = str(visualization_source).strip().lower()
        if self.visualization_source not in {"zed", "depth"}:
            raise ValueError(
                "viewer.point_cloud_source must be either 'zed' or 'depth'"
            )
        self.visualization_period = 1.0 / max(0.1, float(visualization_max_fps))
        self.next_visualization_time = 0.0
        self.image = sl.Mat()
        self.depth = sl.Mat()
        self.point_cloud = sl.Mat()
        self.tracking_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tracking")
        self.visualization_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="visualization",
        )
        self.point_cloud_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="point-cloud",
        )
        # Tracking owns at most one active frame and one replaceable pending
        # frame. Four reusable host buffers cover active, pending, and one
        # visualization frame without allocating full RGB/depth arrays at 60 Hz.
        self.tracking_gate = threading.Lock()
        self.tracking_future: Future | None = None
        self.pending_tracking_input: TrackingInput | None = None
        self.completed_tracking_results: deque[TrackingResult] = deque()
        self.tracking_buffers: list[tuple[np.ndarray, np.ndarray]] = []
        self.free_tracking_buffers: deque[int] = deque()
        self.tracking_buffer_count = 4
        self.visualization_buffer_index: int | None = None
        self.visualization_future: Future | None = None
        self.pending_visualization_result: TrackingResult | None = None
        self.cached_point_cloud: PreparedPointCloud | None = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._capture_loop, name="zed-capture", daemon=True)
        self.started = False
        self.lock = threading.Lock()
        self.latest_result: SegmentationResult | None = None
        self.error: str | None = None
        self.frame_count = 0
        self.tracking_published = 0
        self.tracking_submitted = 0
        self.tracking_completed = 0
        self.tracking_replaced = 0
        self.tracking_buffer_starved = 0
        self.tracking_busy_frames = 0
        self.visualization_submitted = 0
        self.visualization_completed = 0
        self.visualization_dropped = 0
        self.point_cloud_retrievals = 0
        self.capture_fps = 0.0
        self.tracking_fps = 0.0
        self.visualization_fps = 0.0
        self.last_inference_ms = 0.0
        self.last_tracking_ms = 0.0
        self.last_visualization_ms = 0.0
        self.last_observation_ms = 0.0
        self.last_particle_filter_ms = 0.0
        self.last_particle_filter_gpu_ms = 0.0
        self._last_fps_time = time.perf_counter()
        self._last_fps_frame = 0
        self._last_rate_time = self._last_fps_time
        self._last_tracking_completed = 0
        self._last_visualization_completed = 0
        self._last_tracking_submit_time: float | None = None
        self._last_tracking_frame_index = -1
        self._last_tracking_capture_time = float("-inf")
        self.pending_features = particle_filter.features
        self.observation_profile_history = {
            name: deque(maxlen=120)
            for name in (
                "total",
                "geometry",
                "components",
                "endpoints",
                "skeleton",
                "graph",
                "routes",
            )
        }
        self.observation_profile_stats: dict[str, float | int] = {}
        self.performance_history = {
            name: deque(maxlen=120)
            for name in (
                "grab",
                "image_retrieve",
                "image_copy",
                "depth_retrieve",
                "depth_copy",
                "cloud_retrieve",
                "cloud_copy",
                "submit_interval",
                "schedule_wait",
                "inference",
                "observation",
                "pf_wall",
                "pf_gpu",
                "tracking",
                "viewer_prep",
                "viewer_cloud",
                "viewer_overlay",
                "viewer_latency",
                "viewer_source_age",
                "pf_cpu_route_input",
                "pf_gpu_inputs",
                "pf_gpu_prediction",
                "pf_gpu_prediction_constraint",
                "pf_gpu_proposal",
                "pf_gpu_measurement_constraint",
                "pf_gpu_velocity_correction",
                "pf_gpu_route_score",
                "pf_gpu_partial_score",
                "pf_gpu_weights",
                "pf_gpu_posterior",
                "pf_gpu_estimate_constraint",
                "pf_gpu_diagnostics",
                "pf_cpu_readback",
            )
        }
        self.performance_stats: dict[str, float | int | str] = {
            "viewer_mode": (
                self.visualization_source if self.visualization_enabled else "off"
            )
        }
        self.observation_profile_console_period_s = float(
            observation_builder.config.profile_console_period_s
        )
        self.next_observation_profile_console_time = (
            time.perf_counter() + self.observation_profile_console_period_s
        )

    def start(self) -> None:
        if self.started:
            raise RuntimeError("The asynchronous pipeline has already been started.")
        self.started = True
        self.thread.start()

    def warmup(self, image_shape: tuple[int, int]) -> float:
        """Initialize CUDA on the persistent tracking thread without timing it as a frame."""

        height, width = (int(value) for value in image_shape)
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        started = time.perf_counter()
        future = self.tracking_executor.submit(
            self.segmenter.tracking_channels,
            blank,
            self.thresholds,
        )
        future.result()
        return float((time.perf_counter() - started) * 1000.0)

    def stop(self) -> None:
        self.stop_event.set()
        with self.tracking_gate:
            if self.pending_tracking_input is not None:
                self._release_tracking_buffer_locked(
                    self.pending_tracking_input.buffer_index
                )
                self.pending_tracking_input = None
        if self.started:
            self.thread.join(timeout=5.0)
            if self.thread.is_alive():
                raise RuntimeError("ZED capture thread did not stop within five seconds.")
        self.tracking_executor.shutdown(wait=True, cancel_futures=True)
        self.visualization_executor.shutdown(wait=True, cancel_futures=True)
        self.point_cloud_executor.shutdown(wait=True, cancel_futures=True)
        with self.tracking_gate:
            if self.pending_visualization_result is not None:
                self._release_tracking_buffer_locked(
                    self.pending_visualization_result.buffer_index
                )
                self.pending_visualization_result = None
            if self.visualization_buffer_index is not None:
                self._release_tracking_buffer_locked(
                    self.visualization_buffer_index
                )
                self.visualization_buffer_index = None
            self.visualization_future = None

    def free(self) -> None:
        self.image.free()
        self.depth.free()
        self.point_cloud.free()
        self.tracking_buffers.clear()
        self.free_tracking_buffers.clear()

    def snapshot(self) -> tuple[SegmentationResult | None, dict[str, float | int]]:
        with self.lock:
            if self.error is not None:
                raise RuntimeError(f"Asynchronous ZED pipeline failed:\n{self.error}")
            stats: dict[str, float | int] = {
                "capture_fps": float(self.capture_fps),
                "tracking_fps": float(self.tracking_fps),
                "visualization_fps": float(self.visualization_fps),
                "inference_ms": float(self.last_inference_ms),
                "tracking_ms": float(self.last_tracking_ms),
                "visualization_ms": float(self.last_visualization_ms),
                "observation_ms": float(self.last_observation_ms),
                "particle_filter_ms": float(self.last_particle_filter_ms),
                "particle_filter_gpu_ms": float(self.last_particle_filter_gpu_ms),
                "captured": int(self.frame_count),
                "tracking_published": int(self.tracking_published),
                "tracking_submitted": int(self.tracking_submitted),
                "tracking_completed": int(self.tracking_completed),
                "tracking_replaced": int(self.tracking_replaced),
                "tracking_buffer_starved": int(self.tracking_buffer_starved),
                "tracking_busy_frames": int(self.tracking_busy_frames),
                "visualization_submitted": int(self.visualization_submitted),
                "visualization_completed": int(self.visualization_completed),
                "visualization_dropped": int(self.visualization_dropped),
                "point_cloud_retrievals": int(self.point_cloud_retrievals),
            }
            stats.update(self.observation_profile_stats)
            stats.update(self.performance_stats)
            return self.latest_result, stats

    def _ensure_tracking_buffers(
        self,
        bgr_shape: tuple[int, ...],
        depth_shape: tuple[int, ...],
    ) -> None:
        with self.tracking_gate:
            if self.tracking_buffers:
                if (
                    self.tracking_buffers[0][0].shape != bgr_shape
                    or self.tracking_buffers[0][1].shape != depth_shape
                ):
                    raise RuntimeError("ZED frame shape changed while tracking")
                return
            self.tracking_buffers = [
                (
                    np.empty(bgr_shape, dtype=np.uint8),
                    np.empty(depth_shape, dtype=np.float32),
                )
                for _ in range(self.tracking_buffer_count)
            ]
            self.free_tracking_buffers.extend(range(self.tracking_buffer_count))

    def _acquire_tracking_buffer(
        self,
    ) -> tuple[int, np.ndarray, np.ndarray] | None:
        with self.tracking_gate:
            if not self.free_tracking_buffers:
                return None
            index = int(self.free_tracking_buffers.popleft())
            bgr, depth = self.tracking_buffers[index]
            return index, bgr, depth

    def _release_tracking_buffer_locked(self, index: int) -> None:
        if index not in self.free_tracking_buffers:
            self.free_tracking_buffers.append(int(index))

    def _release_tracking_buffer(self, index: int) -> None:
        with self.tracking_gate:
            self._release_tracking_buffer_locked(index)

    def _publish_tracking_input(self, tracking_input: TrackingInput) -> None:
        """Publish one frame, replacing only an unconsumed older frame."""

        future_to_arm: Future | None = None
        input_for_callback: TrackingInput | None = None
        replaced_pending = False
        active_busy = False
        with self.tracking_gate:
            if self.stop_event.is_set():
                self._release_tracking_buffer_locked(tracking_input.buffer_index)
                return
            active_busy = self.tracking_future is not None
            if not active_busy:
                future_to_arm = self.tracking_executor.submit(
                    self._execute_tracking_input,
                    tracking_input,
                )
                self.tracking_future = future_to_arm
                input_for_callback = tracking_input
            else:
                previous = self.pending_tracking_input
                if previous is not None:
                    replaced_pending = True
                    # A viewer reservation must not force the tracker
                    # to consume an old frame. Carry only its diagnostic payload
                    # while replacing RGB/depth with the newest observation.
                    if (
                        tracking_input.render_cloud is None
                        and tracking_input.render_depth is None
                    ):
                        if previous.render_cloud is not None:
                            tracking_input = replace(
                                tracking_input,
                                render_cloud=previous.render_cloud,
                                render_captured_at=previous.render_captured_at,
                            )
                        elif previous.render_depth is not None:
                            tracking_input = replace(
                                tracking_input,
                                render_depth=tracking_input.depth,
                                render_captured_at=tracking_input.captured_at,
                            )
                    self._release_tracking_buffer_locked(previous.buffer_index)
                self.pending_tracking_input = tracking_input

        if future_to_arm is not None and input_for_callback is not None:
            future_to_arm.add_done_callback(
                lambda completed, item=input_for_callback: self._tracking_done(
                    completed, item
                )
            )
            with self.lock:
                self.tracking_submitted += 1
        with self.lock:
            self.tracking_published += 1
            if active_busy:
                self.tracking_busy_frames += 1
            if replaced_pending:
                self.tracking_replaced += 1

    def _execute_tracking_input(self, tracking_input: TrackingInput) -> TrackingResult:
        if (
            tracking_input.frame_index <= self._last_tracking_frame_index
            or tracking_input.captured_at <= self._last_tracking_capture_time
        ):
            raise RuntimeError(
                "Latest-frame mailbox delivered non-monotonic camera input: "
                f"frame={tracking_input.frame_index}, "
                f"previous={self._last_tracking_frame_index}"
            )
        self._last_tracking_frame_index = int(tracking_input.frame_index)
        self._last_tracking_capture_time = float(tracking_input.captured_at)
        if tracking_input.requested_features != self.particle_filter.features:
            self.particle_filter.set_features(tracking_input.requested_features)
        started_at = time.perf_counter()
        submit_interval_ms = (
            0.0
            if self._last_tracking_submit_time is None
            else (started_at - self._last_tracking_submit_time) * 1000.0
        )
        self._last_tracking_submit_time = started_at
        return self._run_tracking(
            tracking_input.buffer_index,
            tracking_input.frame_index,
            tracking_input.captured_at,
            tracking_input.published_at,
            tracking_input.bgr,
            tracking_input.depth,
            tracking_input.render_cloud,
            tracking_input.render_depth,
            tracking_input.render_captured_at,
            replace(
                tracking_input.capture_profile,
                submit_interval_ms=float(submit_interval_ms),
            ),
        )

    def _tracking_done(
        self,
        completed_future: Future,
        completed_input: TrackingInput,
    ) -> None:
        try:
            result = completed_future.result()
        except Exception:
            with self.tracking_gate:
                self.tracking_future = None
                self._release_tracking_buffer_locked(completed_input.buffer_index)
                if self.pending_tracking_input is not None:
                    self._release_tracking_buffer_locked(
                        self.pending_tracking_input.buffer_index
                    )
                    self.pending_tracking_input = None
            with self.lock:
                self.error = traceback.format_exc()
            self.stop_event.set()
            return

        next_future: Future | None = None
        next_input: TrackingInput | None = None
        with self.tracking_gate:
            self.completed_tracking_results.append(result)
            self.tracking_future = None
            if self.stop_event.is_set():
                if self.pending_tracking_input is not None:
                    self._release_tracking_buffer_locked(
                        self.pending_tracking_input.buffer_index
                    )
                    self.pending_tracking_input = None
            elif self.pending_tracking_input is not None:
                next_input = self.pending_tracking_input
                self.pending_tracking_input = None
                next_future = self.tracking_executor.submit(
                    self._execute_tracking_input,
                    next_input,
                )
                self.tracking_future = next_future

        if next_future is not None and next_input is not None:
            next_future.add_done_callback(
                lambda completed, item=next_input: self._tracking_done(
                    completed, item
                )
            )
            with self.lock:
                self.tracking_submitted += 1

    def _append_performance(self, name: str, value: float) -> None:
        value = float(value)
        if name in self.performance_history and np.isfinite(value) and value >= 0.0:
            self.performance_history[name].append(value)

    def _refresh_performance_stats(self) -> None:
        for name, history in self.performance_history.items():
            if not history:
                continue
            samples = np.asarray(history, dtype=np.float64)
            self.performance_stats[f"{name}_median_ms"] = float(
                np.median(samples)
            )
            self.performance_stats[f"{name}_p95_ms"] = float(
                np.percentile(samples, 95.0)
            )
        self.performance_stats["performance_profile_samples"] = int(
            len(self.performance_history["tracking"])
        )
        self.performance_stats.update(
            {
                "tracking_published": int(self.tracking_published),
                "tracking_consumed": int(self.tracking_submitted),
                "tracking_replaced": int(self.tracking_replaced),
                "tracking_buffer_starved": int(self.tracking_buffer_starved),
            }
        )

    def _record_particle_filter_profile(
        self, profile: ParticleFilterProfile | None
    ) -> None:
        if profile is None:
            return
        values = {
            "pf_cpu_route_input": profile.route_input_cpu_ms,
            "pf_gpu_inputs": profile.inputs_gpu_ms,
            "pf_gpu_prediction": profile.prediction_gpu_ms,
            "pf_gpu_prediction_constraint": profile.prediction_constraint_gpu_ms,
            "pf_gpu_proposal": profile.proposal_gpu_ms,
            "pf_gpu_measurement_constraint": profile.measurement_constraint_gpu_ms,
            "pf_gpu_velocity_correction": profile.velocity_correction_gpu_ms,
            "pf_gpu_route_score": profile.route_score_gpu_ms,
            "pf_gpu_partial_score": profile.partial_score_gpu_ms,
            "pf_gpu_weights": profile.weight_update_gpu_ms,
            "pf_gpu_posterior": profile.posterior_gpu_ms,
            "pf_gpu_estimate_constraint": profile.estimate_constraint_gpu_ms,
            "pf_gpu_diagnostics": profile.diagnostics_gpu_ms,
            "pf_cpu_readback": profile.readback_cpu_ms,
        }
        for name, value in values.items():
            self._append_performance(name, value)

    def _record_observation_profile(self, profile: ObservationProfile) -> None:
        values = {
            "total": profile.total_ms,
            "geometry": profile.geometry_ms,
            "components": profile.connected_components_ms,
            "endpoints": profile.endpoints_ms,
            "skeleton": profile.skeleton_ms,
            "graph": profile.graph_ms,
            "routes": profile.route_search_ms + profile.route_assembly_ms,
        }
        for name, value in values.items():
            history = self.observation_profile_history[name]
            history.append(float(value))
            samples = np.asarray(history, dtype=np.float64)
            self.observation_profile_stats[
                f"observation_{name}_median_ms"
            ] = float(np.median(samples))
            self.observation_profile_stats[
                f"observation_{name}_p95_ms"
            ] = float(np.percentile(samples, 95.0))
        self.observation_profile_stats.update(
            {
                "observation_profile_samples": int(
                    len(self.observation_profile_history["total"])
                ),
                "observation_maps_cuda_ms": float(profile.maps_cuda_ms),
                "observation_unprojection_cuda_ms": float(
                    profile.unprojection_cuda_ms
                ),
                "observation_readback_wall_ms": float(
                    profile.readback_wall_ms
                ),
                "observation_masks_latest_ms": float(profile.masks_ms),
                "observation_component_preparation_latest_ms": float(
                    profile.component_preparation_ms
                ),
                "observation_fragments_latest_ms": float(
                    profile.global_fragments_ms + profile.cable_fragments_ms
                ),
                "observation_tangents_latest_ms": float(profile.tangents_ms),
                "observation_route_search_latest_ms": float(
                    profile.route_search_ms
                ),
                "observation_route_assembly_latest_ms": float(
                    profile.route_assembly_ms
                ),
                "observation_finalization_latest_ms": float(
                    profile.finalization_ms
                ),
                "observation_unaccounted_latest_ms": float(
                    profile.unaccounted_ms
                ),
                "observation_cable_pixels": int(profile.cable_pixels),
                "observation_relevant_pixels": int(profile.relevant_pixels),
                "observation_valid_3d_points": int(profile.valid_3d_points),
                "observation_component_count": int(profile.component_count),
                "observation_processed_component_count": int(
                    profile.processed_component_count
                ),
                "observation_endpoint_count": int(profile.endpoint_count),
                "observation_skeleton_pixels": int(profile.skeleton_pixels),
                "observation_graph_nodes": int(profile.graph_nodes),
                "observation_graph_edges": int(profile.graph_edges),
                "observation_branch_pixels": int(profile.branch_pixels),
                "observation_route_candidates": int(profile.route_candidates),
                "observation_fragments": int(profile.fragments),
            }
        )

    def set_feature(self, name: str, enabled: bool) -> None:
        """Queue one ablation change; the tracking worker applies it between frames."""

        if name not in ParticleFeatures.__dataclass_fields__ or name == "crossing":
            raise ValueError(f"Feature is not available in phase 1: {name}")
        with self.lock:
            self.pending_features = replace(
                self.pending_features,
                **{name: bool(enabled)},
            )

    def _drain_visualization(self) -> None:
        if self.visualization_future is None or not self.visualization_future.done():
            return
        buffer_index = self.visualization_buffer_index
        try:
            result = self.visualization_future.result()
        finally:
            self.visualization_future = None
            self.visualization_buffer_index = None
            if buffer_index is not None:
                self._release_tracking_buffer(buffer_index)
        with self.lock:
            self.latest_result = result
            self.visualization_completed += 1
            self.last_visualization_ms = float(result.processing_ms)
            self._append_performance("viewer_prep", result.processing_ms)
            self._append_performance(
                "viewer_cloud", result.cloud_processing_ms
            )
            self._append_performance(
                "viewer_overlay", result.overlay_processing_ms
            )
            self._append_performance("viewer_latency", result.latency_ms)
            self._append_performance("viewer_source_age", result.source_age_ms)

        pending = self.pending_visualization_result
        self.pending_visualization_result = None
        if pending is not None:
            if self.stop_event.is_set():
                self._release_tracking_buffer(pending.buffer_index)
            else:
                self._start_visualization(pending)

    def _start_visualization(self, result: TrackingResult) -> None:
        """Start one viewer job; callers keep at most one newer pending frame."""

        if self.visualization_future is not None:
            raise RuntimeError("Visualization worker already has an active frame")
        self.visualization_future = self.visualization_executor.submit(
            self._prepare_visualization,
            result,
        )
        self.visualization_buffer_index = result.buffer_index
        with self.lock:
            self.visualization_submitted += 1

    def _drain_tracking(self) -> None:
        with self.tracking_gate:
            if not self.completed_tracking_results:
                return
            completed = tuple(self.completed_tracking_results)
            self.completed_tracking_results.clear()

        for result in completed:
            with self.lock:
                self.tracking_completed += 1
                self.last_inference_ms = float(result.inference_ms)
                self.last_tracking_ms = float(result.tracking_ms)
                self.last_observation_ms = float(result.observation.processing_ms)
                self.last_particle_filter_ms = float(result.particle_filter.processing_ms)
                self.last_particle_filter_gpu_ms = float(result.particle_filter.gpu_ms)
                capture = result.capture_profile
                for name, value in (
                    ("grab", capture.grab_ms),
                    ("image_retrieve", capture.image_retrieve_ms),
                    ("image_copy", capture.image_copy_ms),
                    ("depth_retrieve", capture.depth_retrieve_ms),
                    ("depth_copy", capture.depth_copy_ms),
                    ("submit_interval", capture.submit_interval_ms),
                    ("schedule_wait", result.schedule_wait_ms),
                    ("inference", result.inference_ms),
                    ("observation", result.observation.processing_ms),
                    ("pf_wall", result.particle_filter.processing_ms),
                    ("pf_gpu", result.particle_filter.gpu_ms),
                    ("tracking", result.tracking_ms),
                ):
                    self._append_performance(name, value)
                if capture.cloud_retrieve_ms > 0.0:
                    self._append_performance(
                        "cloud_retrieve", capture.cloud_retrieve_ms
                    )
                if capture.cloud_copy_ms > 0.0:
                    self._append_performance("cloud_copy", capture.cloud_copy_ms)
                self._record_particle_filter_profile(result.particle_filter.profile)
                if result.observation.profile is not None:
                    self._record_observation_profile(result.observation.profile)

            # Raw RGB/depth storage is retained only while the diagnostic viewer
            # actually consumes this result. It is returned to the fixed pool
            # immediately for ordinary tracking-only frames.
            has_visualization = self.visualization_enabled
            if not has_visualization:
                self._release_tracking_buffer(result.buffer_index)
            else:
                if self.visualization_future is None:
                    self._start_visualization(result)
                else:
                    replaced = self.pending_visualization_result
                    self.pending_visualization_result = result
                    if replaced is not None:
                        self._release_tracking_buffer(replaced.buffer_index)
                        with self.lock:
                            self.visualization_dropped += 1

    def _capture_loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                grab_started = time.perf_counter()
                status = self.zed.grab(self.runtime)
                grab_finished = time.perf_counter()
                if status != sl.ERROR_CODE.SUCCESS:
                    continue
                captured_at = grab_finished
                with self.lock:
                    self.frame_count += 1
                    frame_index = int(self.frame_count)
                self._drain_visualization()
                self._drain_tracking()
                self._update_capture_fps(captured_at)
                self._update_stage_rates(captured_at)

                image_retrieve_started = time.perf_counter()
                image_status = self.zed.retrieve_image(
                    self.image, sl.VIEW.LEFT, sl.MEM.CPU
                )
                image_retrieve_finished = time.perf_counter()
                depth_retrieve_started = image_retrieve_finished
                depth_status = self.zed.retrieve_measure(
                    self.depth,
                    sl.MEASURE.DEPTH,
                    sl.MEM.CPU,
                )
                depth_retrieve_finished = time.perf_counter()
                if image_status != sl.ERROR_CODE.SUCCESS or depth_status != sl.ERROR_CODE.SUCCESS:
                    raise RuntimeError(
                        f"ZED retrieval failed: image={image_status}, depth={depth_status}"
                    )
                image_copy_started = time.perf_counter()
                bgra = np.asarray(self.image.get_data())
                depth_source = np.asarray(
                    self.depth.get_data(), dtype=np.float32
                ).squeeze()
                if depth_source.ndim != 2:
                    raise RuntimeError(
                        f"ZED returned an invalid depth shape: {depth_source.shape}"
                    )
                self._ensure_tracking_buffers(
                    tuple(bgra.shape[:2]) + (3,),
                    tuple(depth_source.shape),
                )
                acquired = self._acquire_tracking_buffer()
                if acquired is None:
                    with self.lock:
                        self.tracking_buffer_starved += 1
                    continue
                buffer_index, bgr, depth = acquired
                np.copyto(bgr, bgra[:, :, :3], casting="no")
                image_copy_finished = time.perf_counter()
                depth_copy_started = image_copy_finished
                np.copyto(depth, depth_source, casting="no")
                depth_copy_finished = time.perf_counter()

                render_cloud = None
                render_depth = None
                render_captured_at = None
                cloud_retrieve_ms = 0.0
                cloud_copy_ms = 0.0
                if (
                    self.visualization_enabled
                    and captured_at >= self.next_visualization_time
                ):
                    if self.visualization_source == "zed":
                        cloud_retrieve_started = time.perf_counter()
                        cloud_status = self.zed.retrieve_measure(
                            self.point_cloud,
                            sl.MEASURE.XYZRGBA,
                            sl.MEM.CPU,
                        )
                        cloud_retrieve_finished = time.perf_counter()
                        if cloud_status != sl.ERROR_CODE.SUCCESS:
                            raise RuntimeError(
                                "ZED visualization cloud retrieval failed: "
                                f"{cloud_status}"
                            )
                        cloud_copy_started = time.perf_counter()
                        render_cloud = np.ascontiguousarray(
                            self.point_cloud.get_data().copy(),
                            dtype=np.float32,
                        )
                        cloud_copy_finished = time.perf_counter()
                        cloud_retrieve_ms = float(
                            (cloud_retrieve_finished - cloud_retrieve_started)
                            * 1000.0
                        )
                        cloud_copy_ms = float(
                            (cloud_copy_finished - cloud_copy_started) * 1000.0
                        )
                        with self.lock:
                            self.point_cloud_retrievals += 1
                    else:
                        # The fixed tracking buffer remains owned by this frame
                        # until the visualization worker has finished with it.
                        render_depth = depth
                    render_captured_at = captured_at
                    self.next_visualization_time = captured_at + self.visualization_period

                with self.lock:
                    requested_features = self.pending_features
                published_at = time.perf_counter()
                capture_profile = CaptureProfile(
                    grab_ms=float((grab_finished - grab_started) * 1000.0),
                    image_retrieve_ms=float(
                        (image_retrieve_finished - image_retrieve_started)
                        * 1000.0
                    ),
                    image_copy_ms=float(
                        (image_copy_finished - image_copy_started) * 1000.0
                    ),
                    depth_retrieve_ms=float(
                        (depth_retrieve_finished - depth_retrieve_started)
                        * 1000.0
                    ),
                    depth_copy_ms=float(
                        (depth_copy_finished - depth_copy_started) * 1000.0
                    ),
                    cloud_retrieve_ms=cloud_retrieve_ms,
                    cloud_copy_ms=cloud_copy_ms,
                    # Filled when the persistent worker actually consumes this
                    # frame; publication occurs independently at camera rate.
                    submit_interval_ms=0.0,
                )
                self._publish_tracking_input(
                    TrackingInput(
                        buffer_index=buffer_index,
                        frame_index=frame_index,
                        captured_at=captured_at,
                        published_at=published_at,
                        bgr=bgr,
                        depth=depth,
                        render_cloud=render_cloud,
                        render_depth=render_depth,
                        render_captured_at=render_captured_at,
                        requested_features=requested_features,
                        capture_profile=capture_profile,
                    )
                )
            self._drain_visualization()
            self._drain_tracking()
        except Exception:
            with self.lock:
                self.error = traceback.format_exc()

    def _update_capture_fps(self, now: float) -> None:
        elapsed = now - self._last_fps_time
        if elapsed < 0.5:
            return
        frames = self.frame_count - self._last_fps_frame
        with self.lock:
            self.capture_fps = float(frames / max(elapsed, 1e-6))
        self._last_fps_time = now
        self._last_fps_frame = self.frame_count

    def _update_stage_rates(self, now: float) -> None:
        elapsed = now - self._last_rate_time
        if elapsed >= 0.5:
            with self.lock:
                self.tracking_fps = float(
                    (self.tracking_completed - self._last_tracking_completed)
                    / max(elapsed, 1e-6)
                )
                self.visualization_fps = float(
                    (
                        self.visualization_completed
                        - self._last_visualization_completed
                    )
                    / max(elapsed, 1e-6)
                )
                self._last_tracking_completed = int(self.tracking_completed)
                self._last_visualization_completed = int(
                    self.visualization_completed
                )
            self._last_rate_time = now
        self._print_observation_profile(now)

    def _print_observation_profile(self, now: float) -> None:
        period = self.observation_profile_console_period_s
        if (
            period <= 0.0
            or now < self.next_observation_profile_console_time
        ):
            return
        with self.lock:
            self._refresh_performance_stats()
            stats = dict(self.observation_profile_stats)
            performance = dict(self.performance_stats)
            frame_count = int(self.frame_count)
            tracking_fps = float(self.tracking_fps)
            visualization_fps = float(self.visualization_fps)
        if (
            "observation_total_median_ms" not in stats
            and "tracking_median_ms" not in performance
        ):
            return
        self.next_observation_profile_console_time = now + period

        if "tracking_median_ms" in performance:
            self._print_tracking_profile(
                performance,
                frame_count,
                tracking_fps,
                visualization_fps,
            )
        if "observation_total_median_ms" not in stats:
            return

        def pair(name: str) -> str:
            return (
                f"{float(stats.get(f'observation_{name}_median_ms', 0.0)):.2f}/"
                f"{float(stats.get(f'observation_{name}_p95_ms', 0.0)):.2f}"
            )

        print(
            f"OBS_PROFILE frame={frame_count} "
            f"samples={int(stats.get('observation_profile_samples', 0))} "
            f"tracking={tracking_fps:.1f}fps\n"
            f"  median/p95_ms total={pair('total')} geometry={pair('geometry')} "
            f"components={pair('components')} endpoints={pair('endpoints')} "
            f"skeleton={pair('skeleton')} graph={pair('graph')} "
            f"routes={pair('routes')}\n"
            f"  latest_ms maps_cuda="
            f"{float(stats.get('observation_maps_cuda_ms', 0.0)):.2f} "
            f"unproject_cuda="
            f"{float(stats.get('observation_unprojection_cuda_ms', 0.0)):.2f} "
            f"readback_block="
            f"{float(stats.get('observation_readback_wall_ms', 0.0)):.2f}\n"
            f"  latest_other_ms masks="
            f"{float(stats.get('observation_masks_latest_ms', 0.0)):.2f} "
            f"component_prep="
            f"{float(stats.get('observation_component_preparation_latest_ms', 0.0)):.2f} "
            f"fragments="
            f"{float(stats.get('observation_fragments_latest_ms', 0.0)):.2f} "
            f"tangents="
            f"{float(stats.get('observation_tangents_latest_ms', 0.0)):.2f} "
            f"route_search/build="
            f"{float(stats.get('observation_route_search_latest_ms', 0.0)):.2f}/"
            f"{float(stats.get('observation_route_assembly_latest_ms', 0.0)):.2f} "
            f"final="
            f"{float(stats.get('observation_finalization_latest_ms', 0.0)):.2f} "
            f"unaccounted="
            f"{float(stats.get('observation_unaccounted_latest_ms', 0.0)):.2f}\n"
            f"  topology cable/relevant/valid3D="
            f"{int(stats.get('observation_cable_pixels', 0))}/"
            f"{int(stats.get('observation_relevant_pixels', 0))}/"
            f"{int(stats.get('observation_valid_3d_points', 0))} "
            f"components="
            f"{int(stats.get('observation_processed_component_count', 0))}/"
            f"{int(stats.get('observation_component_count', 0))} "
            f"endpoints={int(stats.get('observation_endpoint_count', 0))}/4 "
            f"skeleton_px={int(stats.get('observation_skeleton_pixels', 0))} "
            f"nodes/edges/branches="
            f"{int(stats.get('observation_graph_nodes', 0))}/"
            f"{int(stats.get('observation_graph_edges', 0))}/"
            f"{int(stats.get('observation_branch_pixels', 0))} "
            f"routes={int(stats.get('observation_route_candidates', 0))} "
            f"fragments={int(stats.get('observation_fragments', 0))}",
            flush=True,
        )

    @staticmethod
    def _print_tracking_profile(
        stats: dict[str, float | int | str],
        frame_count: int,
        tracking_fps: float,
        visualization_fps: float,
    ) -> None:
        def pair(name: str) -> str:
            return (
                f"{float(stats.get(f'{name}_median_ms', 0.0)):.2f}/"
                f"{float(stats.get(f'{name}_p95_ms', 0.0)):.2f}"
            )

        print(
            f"TRACK_PROFILE frame={frame_count} "
            f"samples={int(stats.get('performance_profile_samples', 0))} "
            f"tracking={tracking_fps:.1f}fps viewer={visualization_fps:.1f}fps "
            f"mode={stats.get('viewer_mode', 'unknown')}\n"
            f"  median/p95_ms total={pair('tracking')} "
            f"schedule={pair('schedule_wait')} NN={pair('inference')} "
            f"observation={pair('observation')} PFwall={pair('pf_wall')} "
            f"PFgpu={pair('pf_gpu')} submit_interval={pair('submit_interval')}\n"
            f"  mailbox published/consumed/replaced/starved="
            f"{int(stats.get('tracking_published', 0))}/"
            f"{int(stats.get('tracking_consumed', 0))}/"
            f"{int(stats.get('tracking_replaced', 0))}/"
            f"{int(stats.get('tracking_buffer_starved', 0))} "
            f"handoff_age={pair('schedule_wait')}\n"
            f"  capture_ms grab={pair('grab')} "
            f"image_retrieve/copy={pair('image_retrieve')}/{pair('image_copy')} "
            f"depth_retrieve/copy={pair('depth_retrieve')}/{pair('depth_copy')} "
            f"cloud_retrieve/copy={pair('cloud_retrieve')}/{pair('cloud_copy')}\n"
            f"  viewer_ms prep={pair('viewer_prep')} "
            f"cloud={pair('viewer_cloud')} overlay={pair('viewer_overlay')} "
            f"latency={pair('viewer_latency')} "
            f"source_age={pair('viewer_source_age')}\n"
            f"PF_PROFILE gpu_median/p95_ms inputs={pair('pf_gpu_inputs')} "
            f"prediction={pair('pf_gpu_prediction')} "
            f"predict_constraint={pair('pf_gpu_prediction_constraint')} "
            f"proposal={pair('pf_gpu_proposal')} "
            f"measure_constraint={pair('pf_gpu_measurement_constraint')} "
            f"velocity_update={pair('pf_gpu_velocity_correction')}\n"
            f"  route={pair('pf_gpu_route_score')} "
            f"partial={pair('pf_gpu_partial_score')} "
            f"weights={pair('pf_gpu_weights')} "
            f"posterior={pair('pf_gpu_posterior')} "
            f"estimate_constraint={pair('pf_gpu_estimate_constraint')} "
            f"diagnostics={pair('pf_gpu_diagnostics')} "
            f"cpu_route_input={pair('pf_cpu_route_input')} "
            f"cpu_readback={pair('pf_cpu_readback')}",
            flush=True,
        )

    def _run_tracking(
        self,
        buffer_index: int,
        frame_index: int,
        captured_at: float,
        submitted_at: float,
        bgr: np.ndarray,
        depth: np.ndarray,
        render_cloud: np.ndarray | None,
        render_depth: np.ndarray | None,
        render_captured_at: float | None,
        capture_profile: CaptureProfile,
    ) -> TrackingResult:
        tracking_start = time.perf_counter()
        inference_start = tracking_start
        masks, cable_probability = self.segmenter.tracking_channels(
            bgr, self.thresholds
        )
        inference_finished = time.perf_counter()
        observation = self.observation_builder.build(
            masks,
            depth,
            cable_probability,
            include_skeleton_debug=self.visualization_enabled,
        )
        particle_filter = self.particle_filter.update(
            observation,
            captured_at,
            refresh_diagnostics=(
                render_cloud is not None or render_depth is not None
            ),
        )
        display_observation = replace(
            observation,
            cable_probability=None,
            scene_depth=None,
        )
        completed_at = time.perf_counter()
        return TrackingResult(
            buffer_index=buffer_index,
            frame_index=frame_index,
            captured_at=captured_at,
            completed_at=completed_at,
            bgr=bgr,
            render_cloud=render_cloud,
            render_depth=render_depth,
            render_captured_at=render_captured_at,
            masks=masks,
            observation=display_observation,
            particle_filter=particle_filter,
            inference_ms=float((inference_finished - inference_start) * 1000.0),
            tracking_ms=float((completed_at - tracking_start) * 1000.0),
            schedule_wait_ms=float((tracking_start - submitted_at) * 1000.0),
            capture_profile=capture_profile,
        )

    def _prepare_point_cloud(
        self,
        tracked: TrackingResult,
    ) -> PreparedPointCloud:
        started = time.perf_counter()
        if tracked.render_cloud is not None:
            vertices, stats, center, radius = full_point_cloud_vertices(
                tracked.render_cloud,
                tracked.masks,
                self.visualization_point_cloud_stride,
            )
        elif tracked.render_depth is not None:
            vertices, stats, center, radius = depth_point_cloud_vertices(
                tracked.render_depth,
                tracked.bgr,
                tracked.masks,
                self.observation_builder.camera_model,
                self.visualization_point_cloud_stride,
            )
        else:
            raise ValueError("Visualization requires a reserved cloud or depth frame")
        return PreparedPointCloud(
            vertices=vertices,
            stats=stats,
            scene_center=center,
            scene_radius=radius,
            captured_at=float(
                tracked.render_captured_at
                if tracked.render_captured_at is not None
                else tracked.captured_at
            ),
            processing_ms=float((time.perf_counter() - started) * 1000.0),
        )

    def _prepare_visualization(self, tracked: TrackingResult) -> SegmentationResult:
        visualization_start = time.perf_counter()
        has_fresh_cloud = (
            tracked.render_cloud is not None or tracked.render_depth is not None
        )
        cloud_future = (
            self.point_cloud_executor.submit(self._prepare_point_cloud, tracked)
            if has_fresh_cloud
            else None
        )
        overlay_started = time.perf_counter()
        rgb_image = segmentation_overlay(
            tracked.bgr,
            tracked.masks,
            self.overlay_alpha,
            self.visualization_image_width,
        )
        skeleton_image = skeleton_diagnostic_image(
            tracked.masks,
            tracked.observation,
            self.visualization_image_width,
        )
        overlay_finished = time.perf_counter()
        if cloud_future is not None:
            self.cached_point_cloud = cloud_future.result()
        cloud = self.cached_point_cloud
        if cloud is None:
            raise RuntimeError("The viewer has not received its initial point cloud")
        finished = time.perf_counter()
        return SegmentationResult(
            frame_index=tracked.frame_index,
            captured_at=tracked.captured_at,
            rgb_image=rgb_image,
            skeleton_image=skeleton_image,
            vertices=cloud.vertices,
            cloud_captured_at=cloud.captured_at,
            stats=cloud.stats,
            scene_center=cloud.scene_center,
            scene_radius=cloud.scene_radius,
            inference_ms=float(tracked.inference_ms),
            processing_ms=float((finished - visualization_start) * 1000.0),
            cloud_processing_ms=(
                cloud.processing_ms if has_fresh_cloud else 0.0
            ),
            overlay_processing_ms=float(
                (overlay_finished - overlay_started) * 1000.0
            ),
            latency_ms=float((finished - tracked.captured_at) * 1000.0),
            source_age_ms=float(
                max(
                    0.0,
                    tracked.captured_at - cloud.captured_at,
                )
                * 1000.0
            ),
            observation=tracked.observation,
            particle_filter=tracked.particle_filter,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=SOURCE_DIR / "config.toml",
        help="Viewer TOML configuration.",
    )
    parser.add_argument("--checkpoint", type=Path, help="Override the PIDNet checkpoint.")
    parser.add_argument(
        "--viewer-mode",
        choices=("off", "depth", "zed"),
        help="Override viewer isolation mode without editing config.toml.",
    )
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=0.0,
        help="Stop automatically after this many seconds; zero runs until closed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    camera_config = config["camera"]
    viewer_config = config["viewer"]
    pidnet_config = config["pidnet"]
    observation_config = ObservationConfig.from_mapping(config.get("observation"))
    particle_filter_config = ParticleFilterConfig.from_mapping(config.get("particle_filter"))
    particle_features = ParticleFeatures.from_mapping(config.get("features"))
    checkpoint = resolve_project_path(args.checkpoint or pidnet_config["checkpoint"])

    configured_source = str(viewer_config.get("point_cloud_source", "zed"))
    viewer_enabled = bool(viewer_config.get("enabled", True))
    if args.viewer_mode is not None:
        viewer_enabled = args.viewer_mode != "off"
        if viewer_enabled:
            configured_source = args.viewer_mode
    viewer: SplitPointCloudViewer | None = None
    if viewer_enabled:
        width = int(viewer_config.get("rgb_width", 620)) + int(
            viewer_config.get("cloud_width", 1180)
        )
        viewer = SplitPointCloudViewer(
            width=width,
            height=int(viewer_config.get("height", 900)),
            left_panel_width=int(viewer_config.get("rgb_width", 620)),
            point_size=float(viewer_config.get("point_size", 1.0)),
            recording_directory=resolve_project_path(
                viewer_config.get(
                    "recording_directory",
                    "diagnostics/recordings",
                )
            ),
            recording_fps=float(viewer_config.get("recording_fps", 30.0)),
            recording_codec=str(viewer_config.get("recording_codec", "avc1")),
            recording_queue_frames=int(
                viewer_config.get("recording_queue_frames", 8)
            ),
        )
        viewer.init()
        viewer.update_status("Loading PIDNet on CUDA...")
        viewer.poll()

    pipeline: AsyncSegmentationPipeline | None = None
    zed: sl.Camera | None = None
    try:
        segmenter = PidNetSegmenter(
            checkpoint,
            device=str(pidnet_config.get("device", "cuda")),
            amp=bool(pidnet_config.get("amp", True)),
            channels_last=bool(pidnet_config.get("channels_last", True)),
        )
        thresholds = (
            float(pidnet_config.get("threshold", 0.85)),
            *tuple(float(value) for value in pidnet_config.get("endpoint_thresholds", (0.90, 0.75))),
            float(pidnet_config.get("crossing_threshold", 0.95)),
        )
        if len(thresholds) != 4:
            raise ValueError("PIDNet requires one cable, two endpoint, and one crossing threshold.")

        if viewer is not None:
            viewer.update_status("Opening ZED camera...")
            viewer.poll()
        zed, runtime = open_zed(camera_config)
        if viewer is not None:
            configure_viewer_from_zed(zed, viewer)
        camera_model = camera_model_from_zed(zed)
        pipeline = AsyncSegmentationPipeline(
            zed,
            runtime,
            segmenter,
            thresholds,
            float(pidnet_config.get("overlay_alpha", 0.62)),
            ObservationBuilder(
                observation_config,
                particle_filter_config.cable_lengths_m,
                camera_model,
                particle_filter_config.device,
            ),
            BatchedCableParticleFilter(
                particle_filter_config,
                particle_features,
                camera_model,
            ),
            float(viewer_config.get("point_cloud_fps", 5.0)),
            int(viewer_config.get("rgb_width", 620)),
            int(viewer_config.get("point_cloud_stride", 1)),
            visualization_enabled=viewer_enabled,
            visualization_source=configured_source,
        )
        if viewer is not None:
            viewer.set_feature_controls(particle_features, pipeline.set_feature)
            viewer.update_status("Warming the persistent CUDA tracking worker...")
            viewer.poll()
        warmup_ms = pipeline.warmup(camera_frame_shape(zed))
        print(f"CUDA tracking worker warm-up: {warmup_ms:.1f} ms (excluded from frame timing)")
        pipeline.start()
        last_frame = -1
        display_fps = max(1.0, float(viewer_config.get("max_fps", 30.0)))
        display_period = 1.0 / display_fps
        next_display = time.perf_counter()
        run_deadline = (
            time.perf_counter() + float(args.run_seconds)
            if args.run_seconds > 0.0
            else float("inf")
        )
        while (
            time.perf_counter() < run_deadline
            and (viewer is None or viewer.is_available())
        ):
            result, pipeline_stats = pipeline.snapshot()
            now = time.perf_counter()
            if viewer is None:
                time.sleep(0.05)
            elif now >= next_display:
                if result is not None and result.frame_index != last_frame:
                    viewer.update_frame(result, pipeline_stats)
                    last_frame = result.frame_index
                viewer.poll()
                next_display = now + display_period
            else:
                time.sleep(min(0.002, next_display - now))
    finally:
        try:
            if pipeline is not None:
                try:
                    pipeline.stop()
                finally:
                    pipeline.free()
        finally:
            try:
                if zed is not None:
                    zed.close()
            finally:
                if viewer is not None:
                    viewer.close()


if __name__ == "__main__":
    main()
