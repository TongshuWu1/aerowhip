"""Asynchronous RGB-depth tracking with best-effort full-rate visualization."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
import hashlib
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

from cable_detection import (  # noqa: E402
    PidNetMaskConfig,
    postprocess_pidnet_masks,
)
from cable_pidnet import PidNetSegmenter  # noqa: E402
from contact_estimator import (  # noqa: E402
    CableCubeContactEstimator,
    ContactEstimatorConfig,
    ContactFrame,
)
from cube_tracker import (  # noqa: E402
    CameraModel as CubeCameraModel,
    CubeTrackerConfig,
    CubeTrackingResult,
    KnownCubeTracker,
    draw_cube_wireframe_overlay,
)
from cube_state_filter import (  # noqa: E402
    CubeStateEstimate,
    CubeStateFilterConfig,
    RigidCubeStateFilter,
)
from live_evaluation import LiveEvaluation  # noqa: E402
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
CABLE_BGR = np.asarray((20, 148, 255), dtype=np.uint8)
ENDPOINT_1_BGR = np.asarray((255, 89, 13), dtype=np.uint8)
ENDPOINT_2_BGR = np.asarray((77, 255, 38), dtype=np.uint8)
SKELETON_MASK_RGB = np.asarray((38, 43, 49), dtype=np.uint8)
SKELETON_LINE_RGB = np.asarray((238, 244, 250), dtype=np.uint8)
SKELETON_NODE_RGB = np.asarray((55, 205, 255), dtype=np.uint8)
SKELETON_BRANCH_RGB = np.asarray((255, 72, 88), dtype=np.uint8)
SKELETON_AMBIGUOUS_RGB = np.asarray((255, 166, 48), dtype=np.uint8)
SKELETON_UNASSIGNED_RGB = np.asarray((150, 156, 166), dtype=np.uint8)


def _packed_rgb_float(rgb: np.ndarray) -> np.float32:
    channels = np.rint(np.asarray(rgb) * 255.0).astype(np.uint32)
    packed = channels[0] | (channels[1] << 8) | (channels[2] << 16)
    return np.asarray(packed, dtype=np.uint32).view(np.float32)[()]


CABLE_PACKED = _packed_rgb_float(CABLE_RGB)
ENDPOINT_1_PACKED = _packed_rgb_float(ENDPOINT_1_RGB)
ENDPOINT_2_PACKED = _packed_rgb_float(ENDPOINT_2_RGB)


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
    host_captured_at: float
    published_at: float
    bgr: np.ndarray
    depth: np.ndarray
    render_cloud: np.ndarray | None
    render_depth: np.ndarray | None
    render_captured_at: float | None
    requested_features: ParticleFeatures
    include_top_particles: bool
    capture_profile: CaptureProfile


@dataclass(frozen=True)
class TrackingResult:
    """Algorithm output retained only until visualization accepts or drops it."""

    buffer_index: int
    frame_index: int
    captured_at: float
    host_captured_at: float
    bgr: np.ndarray
    render_cloud: np.ndarray | None
    render_depth: np.ndarray | None
    render_captured_at: float | None
    masks: tuple[np.ndarray, ...]
    observation: FrameObservation
    particle_filter: ParticleFilterFrame
    cube_tracking: CubeTrackingResult | None
    cube_state: CubeStateEstimate | None
    contact: ContactFrame | None
    inference_ms: float
    tracking_ms: float
    schedule_wait_ms: float
    evaluation_sampled: bool
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
    cube_tracking: CubeTrackingResult | None
    cube_state: CubeStateEstimate | None
    contact: ContactFrame | None


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


def _boolean_mask(
    mask: np.ndarray,
    shape: tuple[int, int],
    stride: int = 1,
) -> np.ndarray:
    array = np.asarray(mask)
    if array.shape != shape:
        array = cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    sample_stride = max(1, int(stride))
    return np.ascontiguousarray(
        array[::sample_stride, ::sample_stride] != 0
    )


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
    if len(masks) != 3:
        raise ValueError(
            f"PIDNet must return one cable and two endpoint masks; got {len(masks)}."
        )

    height, width = cloud.shape[:2]
    source_shape = (height, width)
    sample_stride = max(1, int(stride))
    cable = _boolean_mask(masks[0], source_shape, sample_stride).reshape(-1)
    endpoint_1 = _boolean_mask(masks[1], source_shape, sample_stride).reshape(-1)
    endpoint_2 = _boolean_mask(masks[2], source_shape, sample_stride).reshape(-1)
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
    else:
        vertices[:, :3] = xyz_all[finite]
        vertices[:, 3] = flat[finite, 3]
        valid_cable = cable[finite]
        valid_endpoint_1 = endpoint_1[finite]
        valid_endpoint_2 = endpoint_2[finite]

    # Lowest-to-highest priority: cable, endpoint 2, endpoint 1.
    vertices[valid_cable, 3] = CABLE_PACKED
    vertices[valid_endpoint_2, 3] = ENDPOINT_2_PACKED
    vertices[valid_endpoint_1, 3] = ENDPOINT_1_PACKED
    stats = {
        "total_pixels": int(len(flat)),
        "source_pixels": int(height * width),
        "point_cloud_stride": sample_stride,
        "valid_points": valid_count,
        "invalid_points": int(finite.size - valid_count),
        "cable_points": int(np.count_nonzero(valid_cable)),
        "endpoint_1_points": int(np.count_nonzero(valid_endpoint_1)),
        "endpoint_2_points": int(np.count_nonzero(valid_endpoint_2)),
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
    if len(masks) != 3:
        raise ValueError(
            f"PIDNet must return one cable and two endpoint masks; got {len(masks)}."
        )

    height, width = depth_array.shape
    source_shape = (height, width)
    sample_stride = max(1, int(stride))
    cable = _boolean_mask(masks[0], source_shape, sample_stride).reshape(-1)
    endpoint_1 = _boolean_mask(masks[1], source_shape, sample_stride).reshape(-1)
    endpoint_2 = _boolean_mask(masks[2], source_shape, sample_stride).reshape(-1)
    sampled_depth = depth_array[::sample_stride, ::sample_stride]
    sampled_image = image[::sample_stride, ::sample_stride]
    flat_depth = sampled_depth.reshape(-1)
    finite = np.isfinite(flat_depth) & (flat_depth > 0.0)
    valid_indices = np.flatnonzero(finite)
    sampled_width = sampled_depth.shape[1]
    selected_depth = flat_depth[finite]
    selected_x = np.ascontiguousarray(
        (valid_indices % sampled_width) * sample_stride,
        dtype=np.float32,
    )
    selected_y = np.ascontiguousarray(
        (valid_indices // sampled_width) * sample_stride,
        dtype=np.float32,
    )

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
    stats = {
        "total_pixels": int(flat_depth.size),
        "source_pixels": int(height * width),
        "point_cloud_stride": sample_stride,
        "valid_points": int(np.count_nonzero(finite)),
        "invalid_points": int(finite.size - np.count_nonzero(finite)),
        "cable_points": int(np.count_nonzero(finite & cable)),
        "endpoint_1_points": int(np.count_nonzero(finite & endpoint_1)),
        "endpoint_2_points": int(np.count_nonzero(finite & endpoint_2)),
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
        return _boolean_mask(mask, (target_height, target_width))

    cable = display_mask(masks[0])
    endpoint_1 = display_mask(masks[1])
    endpoint_2 = display_mask(masks[2])
    output = image.copy()
    blend = float(np.clip(alpha, 0.0, 1.0))
    for selected, color in (
        (cable, CABLE_BGR),
        (endpoint_2, ENDPOINT_2_BGR),
        (endpoint_1, ENDPOINT_1_BGR),
    ):
        if not np.any(selected):
            continue
        output[selected] = np.rint(
            (1.0 - blend) * image[selected].astype(np.float32)
            + blend * color.astype(np.float32)
        ).astype(np.uint8)
    return np.ascontiguousarray(output)


def skeleton_diagnostic_image(
    masks: tuple[np.ndarray, ...],
    observation: FrameObservation,
    particle_filter: ParticleFilterFrame,
    maximum_width: int,
) -> np.ndarray:
    """Render the exact thinned mask and compressed graph used by tracking."""

    if len(masks) != 3:
        raise ValueError(
            f"Skeleton diagnostics require three PIDNet masks; got {len(masks)}."
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
        return _boolean_mask(mask, (target_height, target_width))

    cable = display_mask(masks[0])
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

    def draw_edge(
        points_xy: np.ndarray,
        color: np.ndarray,
        thickness: int = 2,
    ) -> tuple[int, int] | None:
        points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
        points = points[np.all(np.isfinite(points), axis=1)]
        if len(points) < 2:
            return None
        displayed = np.column_stack(
            (
                np.rint(points[:, 0] * scale_x),
                np.rint(points[:, 1] * scale_y),
            )
        ).astype(np.int32)
        valid = (
            (displayed[:, 0] >= 0)
            & (displayed[:, 0] < width)
            & (displayed[:, 1] >= 0)
            & (displayed[:, 1] < height)
        )
        displayed = displayed[valid]
        if len(displayed) < 2:
            return None
        cv2.polylines(
            output,
            [displayed.reshape(-1, 1, 2)],
            isClosed=False,
            color=tuple(int(value) for value in color),
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        center = displayed[len(displayed) // 2]
        return int(center[0]), int(center[1])

    attribution_by_edge = {}
    attribution = particle_filter.graph_attribution
    if attribution is not None:
        attribution_by_edge = {
            (int(item.component_label), int(item.edge_id)): item
            for item in attribution.edges
        }
    endpoint_colors = (
        np.rint(ENDPOINT_1_RGB * 255.0).astype(np.uint8),
        np.rint(ENDPOINT_2_RGB * 255.0).astype(np.uint8),
    )
    for edge in observation.graph_edges:
        assigned = attribution_by_edge.get(
            (int(edge.component_label), int(edge.edge_id))
        )
        if assigned is None:
            draw_edge(edge.pixels_xy, SKELETON_LINE_RGB, thickness=2)
            continue
        cable_probability = np.asarray(
            assigned.cable_probabilities,
            dtype=np.float32,
        )
        if assigned.selected_cable in (0, 1):
            color = endpoint_colors[assigned.selected_cable]
        elif (
            assigned.unassigned_probability
            >= float(np.max(cable_probability))
        ):
            color = SKELETON_UNASSIGNED_RGB
        else:
            color = SKELETON_AMBIGUOUS_RGB
        center = draw_edge(edge.pixels_xy, color, thickness=3)
        if center is None:
            continue
        if assigned.selected_cable in (0, 1):
            identity = f"C{assigned.selected_cable + 1}"
            arc_text = (
                f" {assigned.arc_start_m:.2f}>{assigned.arc_end_m:.2f}m"
                if np.isfinite(assigned.arc_start_m)
                and np.isfinite(assigned.arc_end_m)
                else ""
            )
        else:
            identity = "UNASSIGNED"
            arc_text = ""
        cv2.putText(
            output,
            (
                f"{identity} p={assigned.confidence:.2f}{arc_text} "
                f"r={assigned.mean_residual_m * 1000.0:.0f}mm"
                + (
                    f" T={assigned.temporal_match_distance_m * 1000.0:.0f}mm"
                    if assigned.temporal_prior_applied
                    and np.isfinite(assigned.temporal_match_distance_m)
                    else ""
                )
                if np.isfinite(assigned.mean_residual_m)
                else f"{identity} p={assigned.confidence:.2f}{arc_text}"
            ),
            (center[0] + 3, center[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            tuple(int(value) for value in color),
            1,
            cv2.LINE_AA,
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

    if attribution is not None:
        cv2.putText(
            output,
            (
                "ORDERED EDGE ASSOCIATION "
                f"edges={len(attribution.edges)} "
                f"assigned/ambiguous/unassigned="
                f"{attribution.attributed_count}/"
                f"{attribution.ambiguous_count}/"
                f"{attribution.unassigned_count} "
                f"temporal={attribution.temporally_matched_count} "
                f"cpu/gpu={attribution.processing_ms:.2f}/"
                f"{attribution.gpu_ms:.2f}ms"
            ),
            (8, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            tuple(int(value) for value in SKELETON_AMBIGUOUS_RGB),
            1,
            cv2.LINE_AA,
        )
    return np.ascontiguousarray(output)


def open_zed(camera_config: dict) -> tuple[sl.Camera, sl.RuntimeParameters]:
    resolution_name = str(camera_config.get("resolution", "HD1080")).upper()
    depth_name = str(camera_config.get("depth_mode", "NEURAL")).upper()
    if resolution_name not in RESOLUTIONS:
        raise ValueError(f"Unsupported ZED resolution: {resolution_name}")
    if depth_name not in DEPTH_MODES:
        raise ValueError(f"Unsupported ZED depth mode: {depth_name}")

    init = sl.InitParameters()
    init.camera_resolution = RESOLUTIONS[resolution_name]
    init.camera_fps = int(camera_config.get("fps", 30))
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
        mask_config: PidNetMaskConfig,
        overlay_alpha: float,
        observation_builder: ObservationBuilder,
        particle_filter: BatchedCableParticleFilter,
        cube_tracker: KnownCubeTracker | None,
        cube_state_filter: RigidCubeStateFilter | None,
        contact_estimator: CableCubeContactEstimator | None,
        visualization_max_fps: float,
        visualization_image_width: int,
        visualization_point_cloud_stride: int,
        visualization_enabled: bool = True,
        visualization_source: str = "zed",
        live_evaluation: LiveEvaluation | None = None,
    ):
        self.zed = zed
        self.runtime = runtime
        self.segmenter = segmenter
        self.mask_config = mask_config
        self.thresholds = mask_config.thresholds
        self.overlay_alpha = float(overlay_alpha)
        self.observation_builder = observation_builder
        self.particle_filter = particle_filter
        self.cube_tracker = cube_tracker
        self.cube_state_filter = cube_state_filter
        self.contact_estimator = contact_estimator
        self.visualization_image_width = max(
            1,
            int(visualization_image_width),
        )
        self.visualization_point_cloud_stride = max(
            1,
            int(visualization_point_cloud_stride),
        )
        self.visualization_enabled = bool(visualization_enabled)
        self.live_evaluation = live_evaluation
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
        self.cube_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="cube-tracking")
            if self.cube_tracker is not None
            else None
        )
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
        # visualization frame without allocating duplicate full RGB/depth arrays.
        self.tracking_gate = threading.Lock()
        self.tracking_future: Future | None = None
        self.pending_tracking_input: TrackingInput | None = None
        self.completed_tracking_results: deque[TrackingResult] = deque()
        self.tracking_buffers: list[tuple[np.ndarray, np.ndarray]] = []
        self.free_tracking_buffers: deque[int] = deque()
        self.tracking_buffer_count = 4
        frame_height = int(self.observation_builder.camera_model.height)
        frame_width = int(self.observation_builder.camera_model.width)
        self._ensure_tracking_buffers(
            (frame_height, frame_width, 3),
            (frame_height, frame_width),
        )
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
        self.last_cube_tracking_ms = 0.0
        self.last_cube_state_ms = 0.0
        self.last_contact_ms = 0.0
        self.last_contact_gpu_ms = 0.0
        self._last_fps_time = time.perf_counter()
        self._last_fps_frame = 0
        self._last_rate_time = self._last_fps_time
        self._last_tracking_completed = 0
        self._last_visualization_completed = 0
        self._last_tracking_submit_time: float | None = None
        self._last_tracking_frame_index = -1
        self._last_tracking_capture_time = float("-inf")
        self.pending_features = particle_filter.features
        self.top_particles_enabled = True
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
                "cube",
                "cube_state",
                "contact",
                "contact_gpu",
                "tracking",
                "viewer_prep",
                "viewer_cloud",
                "viewer_overlay",
                "viewer_latency",
                "viewer_source_age",
                "pf_cpu_route_input",
                "pf_gpu_inputs",
                "pf_gpu_graph_attribution",
                "pf_gpu_prediction",
                "pf_gpu_prediction_constraint",
                "pf_gpu_proposal",
                "pf_gpu_measurement_constraint",
                "pf_gpu_local_motion",
                "pf_gpu_route_score",
                "pf_gpu_visible_edge_score",
                "pf_gpu_regularization",
                "pf_gpu_weights",
                "pf_gpu_posterior",
                "pf_gpu_estimate_constraint",
                "pf_gpu_diagnostics",
                "pf_cpu_readback",
                "pf_graph_attribution_wall",
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
        """Warm PIDNet, cube observation, and contact tensor paths."""

        height, width = (int(value) for value in image_shape)
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        blank_mask = np.zeros((height, width), dtype=np.uint8)
        started = time.perf_counter()
        future = self.tracking_executor.submit(
            self._segment_frame,
            blank,
        )
        cube_future = (
            self.cube_executor.submit(
                self._track_cube,
                blank,
                np.full((height, width), np.nan, dtype=np.float32),
                (blank_mask, blank_mask, blank_mask),
            )
            if self.cube_executor is not None and self.cube_tracker is not None
            else None
        )
        future.result()
        if cube_future is not None:
            cube_future.result()
        if self.contact_estimator is not None:
            self.contact_estimator.warmup()
        return float((time.perf_counter() - started) * 1000.0)

    def _segment_frame(self, bgr: np.ndarray):
        raw_masks = self.segmenter.mask_channels(
            bgr,
            range(len(self.thresholds)),
            self.thresholds,
        )
        masks, _component_count = postprocess_pidnet_masks(
            raw_masks,
            self.mask_config,
        )
        return masks

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
        self.observation_builder.close()
        if self.cube_executor is not None:
            self.cube_executor.shutdown(wait=True, cancel_futures=True)
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
                "cube_tracking_ms": float(self.last_cube_tracking_ms),
                "cube_state_ms": float(self.last_cube_state_ms),
                "contact_ms": float(self.last_contact_ms),
                "contact_gpu_ms": float(self.last_contact_gpu_ms),
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
            tracking_input.host_captured_at,
            tracking_input.published_at,
            tracking_input.bgr,
            tracking_input.depth,
            tracking_input.render_cloud,
            tracking_input.render_depth,
            tracking_input.render_captured_at,
            tracking_input.include_top_particles,
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
            "pf_gpu_graph_attribution": profile.graph_attribution_gpu_ms,
            "pf_gpu_prediction": profile.prediction_gpu_ms,
            "pf_gpu_prediction_constraint": profile.prediction_constraint_gpu_ms,
            "pf_gpu_proposal": profile.proposal_gpu_ms,
            "pf_gpu_measurement_constraint": profile.measurement_constraint_gpu_ms,
            "pf_gpu_local_motion": profile.local_motion_gpu_ms,
            "pf_gpu_route_score": profile.route_score_gpu_ms,
            "pf_gpu_visible_edge_score": profile.visible_edge_score_gpu_ms,
            "pf_gpu_regularization": profile.regularization_gpu_ms,
            "pf_gpu_weights": profile.weight_update_gpu_ms,
            "pf_gpu_posterior": profile.posterior_gpu_ms,
            "pf_gpu_estimate_constraint": profile.estimate_constraint_gpu_ms,
            "pf_gpu_diagnostics": profile.diagnostics_gpu_ms,
            "pf_cpu_readback": profile.readback_cpu_ms,
            "pf_graph_attribution_wall": profile.graph_attribution_wall_ms,
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
                "observation_graph_edges_latest_ms": float(
                    profile.graph_edges_ms
                ),
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
                "observation_graph_edge_count": int(
                    profile.graph_edge_count
                ),
            }
        )

    def set_feature(self, name: str, enabled: bool) -> None:
        """Queue one ablation change; the tracking worker applies it between frames."""

        if name not in ParticleFeatures.__dataclass_fields__:
            raise ValueError(f"Unknown particle-filter feature: {name}")
        with self.lock:
            self.pending_features = replace(
                self.pending_features,
                **{name: bool(enabled)},
            )

    def set_top_particles_enabled(self, enabled: bool) -> None:
        with self.lock:
            self.top_particles_enabled = bool(enabled)

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
                self.last_cube_tracking_ms = float(
                    result.cube_tracking.processing_ms
                    if result.cube_tracking is not None
                    else 0.0
                )
                self.last_cube_state_ms = float(
                    result.cube_state.processing_ms
                    if result.cube_state is not None
                    else 0.0
                )
                self.last_contact_ms = float(
                    result.contact.processing_ms
                    if result.contact is not None
                    else 0.0
                )
                self.last_contact_gpu_ms = float(
                    result.contact.gpu_ms
                    if result.contact is not None
                    else 0.0
                )
                if result.cube_tracking is not None:
                    cube = result.cube_tracking
                    self.performance_stats["cube_valid"] = bool(cube.valid)
                    self.performance_stats["cube_faces"] = int(cube.face_count)
                    self.performance_stats["cube_surface_rms_mm"] = float(
                        cube.surface_rms_m * 1000.0
                    )
                    self.performance_stats["cube_refinement_valid"] = bool(
                        cube.refinement_valid
                    )
                    self.performance_stats["cube_refined_surface_rms_mm"] = float(
                        cube.refined_surface_rms_m * 1000.0
                    )
                    self.performance_stats["cube_reason"] = str(cube.reason)
                if result.cube_state is not None:
                    state = result.cube_state
                    self.performance_stats["cube_state_valid"] = bool(state.valid)
                    self.performance_stats["cube_state_age_ms"] = float(
                        state.measurement_age_s * 1000.0
                    )
                    self.performance_stats["cube_state_speed_mps"] = float(
                        np.linalg.norm(state.linear_velocity_mps)
                        if state.linear_velocity_mps is not None
                        else float("nan")
                    )
                if result.contact is not None:
                    for cable_index, contact in enumerate(result.contact.cables):
                        prefix = f"contact{cable_index + 1}"
                        self.performance_stats[f"{prefix}_initialized"] = bool(
                            contact.initialized
                        )
                        self.performance_stats[f"{prefix}_probability"] = float(
                            contact.contact_probability
                        )
                        self.performance_stats[f"{prefix}_gap_mm"] = float(
                            contact.minimum_gap_m * 1000.0
                        )
                        self.performance_stats[f"{prefix}_support_mass"] = float(
                            contact.local_support_mass
                        )
                        self.performance_stats[f"{prefix}_evidence"] = bool(
                            contact.evidence_used
                        )
                for cable_index, cable in enumerate(
                    result.particle_filter.cables
                ):
                    prefix = f"pf{cable_index + 1}"
                    diagnostic = cable.diagnostics
                    self.performance_stats[f"{prefix}_state"] = (
                        diagnostic.tracking_state
                    )
                    self.performance_stats[f"{prefix}_source"] = (
                        diagnostic.measurement_source
                    )
                    self.performance_stats[f"{prefix}_edges"] = int(
                        diagnostic.attributed_edge_count
                    )
                    self.performance_stats[f"{prefix}_trace_mm"] = float(
                        diagnostic.trace_mean_mm
                    )
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
                    (
                        "cube",
                        result.cube_tracking.processing_ms
                        if result.cube_tracking is not None
                        else 0.0,
                    ),
                    (
                        "cube_state",
                        result.cube_state.processing_ms
                        if result.cube_state is not None
                        else 0.0,
                    ),
                    (
                        "contact",
                        result.contact.processing_ms
                        if result.contact is not None
                        else 0.0,
                    ),
                    (
                        "contact_gpu",
                        result.contact.gpu_ms
                        if result.contact is not None
                        else 0.0,
                    ),
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

            if self.live_evaluation is not None and result.evaluation_sampled:
                self.live_evaluation.append(
                    result.frame_index,
                    result.captured_at,
                    result.particle_filter,
                    observation=result.observation,
                    cube_tracking=result.cube_tracking,
                    contact=result.contact,
                )

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
                timestamp_ns = int(
                    self.zed.get_timestamp(
                        sl.TIME_REFERENCE.IMAGE
                    ).get_nanoseconds()
                )
                if timestamp_ns <= 0:
                    raise RuntimeError(
                        f"ZED returned an invalid image timestamp: {timestamp_ns}"
                    )
                captured_at = float(timestamp_ns * 1.0e-9)
                host_captured_at = grab_finished
                with self.lock:
                    self.frame_count += 1
                    frame_index = int(self.frame_count)
                self._drain_visualization()
                self._drain_tracking()
                self._update_capture_fps(host_captured_at)
                self._update_stage_rates(host_captured_at)

                acquired = self._acquire_tracking_buffer()
                if acquired is None:
                    with self.lock:
                        self.tracking_buffer_starved += 1
                    continue
                buffer_index, bgr, depth = acquired

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
                    self._release_tracking_buffer(buffer_index)
                    raise RuntimeError(
                        f"ZED retrieval failed: image={image_status}, depth={depth_status}"
                    )
                image_copy_started = time.perf_counter()
                bgra = np.asarray(self.image.get_data())
                depth_source = np.asarray(
                    self.depth.get_data(), dtype=np.float32
                ).squeeze()
                if depth_source.ndim != 2:
                    self._release_tracking_buffer(buffer_index)
                    raise RuntimeError(
                        f"ZED returned an invalid depth shape: {depth_source.shape}"
                    )
                expected_bgr_shape = tuple(bgr.shape)
                actual_bgr_shape = tuple(bgra.shape[:2]) + (3,)
                if (
                    actual_bgr_shape != expected_bgr_shape
                    or tuple(depth_source.shape) != tuple(depth.shape)
                ):
                    self._release_tracking_buffer(buffer_index)
                    raise RuntimeError(
                        "ZED frame shape changed while tracking: "
                        f"image={actual_bgr_shape}, depth={depth_source.shape}, "
                        f"expected image={expected_bgr_shape}, depth={depth.shape}"
                    )
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
                    and host_captured_at >= self.next_visualization_time
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
                    self.next_visualization_time = (
                        host_captured_at + self.visualization_period
                    )

                with self.lock:
                    requested_features = self.pending_features
                    include_top_particles = self.top_particles_enabled
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
                        host_captured_at=host_captured_at,
                        published_at=published_at,
                        bgr=bgr,
                        depth=depth,
                        render_cloud=render_cloud,
                        render_depth=render_depth,
                        render_captured_at=render_captured_at,
                        requested_features=requested_features,
                        include_top_particles=include_top_particles,
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
            f"graph_edges="
            f"{float(stats.get('observation_graph_edges_latest_ms', 0.0)):.2f} "
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
            f"observed_edges="
            f"{int(stats.get('observation_graph_edge_count', 0))}",
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
            f"PFgpu={pair('pf_gpu')} cube={pair('cube')} "
            f"contact={pair('contact')} contactGPU={pair('contact_gpu')} "
            f"submit_interval={pair('submit_interval')}\n"
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
            f"edge_attr={pair('pf_gpu_graph_attribution')} "
            f"prediction={pair('pf_gpu_prediction')} "
            f"predict_constraint={pair('pf_gpu_prediction_constraint')} "
            f"proposal={pair('pf_gpu_proposal')} "
            f"measure_constraint={pair('pf_gpu_measurement_constraint')} "
            f"local_motion={pair('pf_gpu_local_motion')}\n"
            f"  route={pair('pf_gpu_route_score')} "
            f"visible_edges={pair('pf_gpu_visible_edge_score')} "
            f"regularization={pair('pf_gpu_regularization')} "
            f"weights={pair('pf_gpu_weights')} "
            f"posterior={pair('pf_gpu_posterior')} "
            f"estimate_constraint={pair('pf_gpu_estimate_constraint')} "
            f"diagnostics={pair('pf_gpu_diagnostics')} "
            f"cpu_route_input={pair('pf_cpu_route_input')} "
            f"cpu_readback={pair('pf_cpu_readback')} "
            f"edge_attr_readback={pair('pf_graph_attribution_wall')}\n"
            f"  measurement PF1="
            f"{stats.get('pf1_state', 'unknown')}/"
            f"{stats.get('pf1_source', 'none')}/"
            f"{int(stats.get('pf1_edges', 0))}edges/"
            f"{float(stats.get('pf1_trace_mm', float('nan'))):.1f}mm "
            f"PF2={stats.get('pf2_state', 'unknown')}/"
            f"{stats.get('pf2_source', 'none')}/"
            f"{int(stats.get('pf2_edges', 0))}edges/"
            f"{float(stats.get('pf2_trace_mm', float('nan'))):.1f}mm "
            f"Cube={'valid' if stats.get('cube_valid', False) else 'invalid'}/"
            f"{int(stats.get('cube_faces', 0))}faces/"
            f"raw={float(stats.get('cube_surface_rms_mm', float('nan'))):.1f}mm/"
            f"ref="
            f"{float(stats.get('cube_refined_surface_rms_mm', float('nan'))):.1f}mm "
            f"state={'valid' if stats.get('cube_state_valid', False) else 'invalid'}/"
            f"{float(stats.get('cube_state_age_ms', float('nan'))):.0f}ms/"
            f"{float(stats.get('cube_state_speed_mps', float('nan'))):.3f}mps\n"
            f"  contact probability/gap/support/source C1="
            f"{float(stats.get('contact1_probability', float('nan'))):.3f}/"
            f"{float(stats.get('contact1_gap_mm', float('nan'))):+.1f}mm/"
            f"{float(stats.get('contact1_support_mass', float('nan'))):.2f}/"
            f"{'M' if stats.get('contact1_evidence', False) else 'P'} "
            f"C2={float(stats.get('contact2_probability', float('nan'))):.3f}/"
            f"{float(stats.get('contact2_gap_mm', float('nan'))):+.1f}mm/"
            f"{float(stats.get('contact2_support_mass', float('nan'))):.2f}/"
            f"{'M' if stats.get('contact2_evidence', False) else 'P'}",
            flush=True,
        )

    def _track_cube(
        self,
        bgr: np.ndarray,
        depth: np.ndarray,
        masks: tuple[np.ndarray, ...],
    ) -> CubeTrackingResult:
        if self.cube_tracker is None:
            raise RuntimeError("Cube tracking is disabled.")
        if len(masks) != 3:
            raise ValueError(
                f"Cube exclusion requires three PIDNet masks; got {len(masks)}."
            )
        cable_mask = cv2.bitwise_or(
            cv2.bitwise_or(masks[0], masks[1]),
            masks[2],
        )
        return self.cube_tracker.track(bgr, depth, cable_mask)

    def _run_tracking(
        self,
        buffer_index: int,
        frame_index: int,
        captured_at: float,
        host_captured_at: float,
        submitted_at: float,
        bgr: np.ndarray,
        depth: np.ndarray,
        render_cloud: np.ndarray | None,
        render_depth: np.ndarray | None,
        render_captured_at: float | None,
        include_top_particles: bool,
        capture_profile: CaptureProfile,
    ) -> TrackingResult:
        tracking_start = time.perf_counter()
        viewer_diagnostics = bool(
            render_cloud is not None or render_depth is not None
        )
        evaluation_sampled = bool(
            self.live_evaluation is not None
            and (viewer_diagnostics or not self.visualization_enabled)
            and self.live_evaluation.should_sample(captured_at)
        )
        inference_start = tracking_start
        masks = self._segment_frame(bgr)
        inference_finished = time.perf_counter()
        # All mask union/exclusion work stays in the cube worker. Launch it as
        # soon as PIDNet finishes so its CPU fit overlaps the cable observation
        # and PF instead of becoming a serial frame-rate limiter.
        cube_future = (
            self.cube_executor.submit(
                self._track_cube,
                bgr,
                depth,
                masks,
            )
            if self.cube_executor is not None and self.cube_tracker is not None
            else None
        )
        observation = self.observation_builder.build(
            masks,
            depth,
            include_skeleton_debug=self.visualization_enabled,
            include_graph_edges=bool(
                (
                    self.particle_filter.features.edge_attribution_diagnostics
                    and viewer_diagnostics
                )
                or self.particle_filter.features.visible_edge_scoring
                or self.particle_filter.features.visible_edge_transport
                or self.particle_filter.features.visible_edge_exploration
            ),
        )
        particle_filter = self.particle_filter.update(
            observation,
            captured_at,
            refresh_diagnostics=(
                viewer_diagnostics or evaluation_sampled
            ),
            refresh_edge_attribution_diagnostics=viewer_diagnostics,
            include_top_particles=(
                include_top_particles and viewer_diagnostics
            ),
            require_contact_support=(self.contact_estimator is not None),
        )
        cube_tracking = cube_future.result() if cube_future is not None else None
        cube_state = (
            self.cube_state_filter.update(cube_tracking, captured_at)
            if self.cube_state_filter is not None
            else None
        )
        contact = (
            self.contact_estimator.update(
                self.particle_filter,
                particle_filter,
                cube_state,
                cube_tracking,
                captured_at,
            )
            if self.contact_estimator is not None
            else None
        )
        completed_at = time.perf_counter()
        return TrackingResult(
            buffer_index=buffer_index,
            frame_index=frame_index,
            captured_at=captured_at,
            host_captured_at=host_captured_at,
            bgr=bgr,
            render_cloud=render_cloud,
            render_depth=render_depth,
            render_captured_at=render_captured_at,
            masks=masks,
            observation=observation,
            particle_filter=particle_filter,
            cube_tracking=cube_tracking,
            cube_state=cube_state,
            contact=contact,
            inference_ms=float((inference_finished - inference_start) * 1000.0),
            tracking_ms=float((completed_at - tracking_start) * 1000.0),
            schedule_wait_ms=float((tracking_start - submitted_at) * 1000.0),
            evaluation_sampled=evaluation_sampled,
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
        overlay_bgr = segmentation_overlay(
            tracked.bgr,
            tracked.masks,
            self.overlay_alpha,
            self.visualization_image_width,
        )
        skeleton_image = skeleton_diagnostic_image(
            tracked.masks,
            tracked.observation,
            tracked.particle_filter,
            self.visualization_image_width,
        )
        if tracked.cube_tracking is not None and self.cube_tracker is not None:
            source_camera = self.cube_tracker.camera
            scale_x = overlay_bgr.shape[1] / float(source_camera.width)
            scale_y = overlay_bgr.shape[0] / float(source_camera.height)
            display_camera = CubeCameraModel(
                fx=source_camera.fx * scale_x,
                fy=source_camera.fy * scale_y,
                cx=source_camera.cx * scale_x,
                cy=source_camera.cy * scale_y,
                width=overlay_bgr.shape[1],
                height=overlay_bgr.shape[0],
            )
            overlay_bgr = draw_cube_wireframe_overlay(
                overlay_bgr,
                tracked.cube_tracking,
                display_camera,
                self.cube_tracker.config.cube_side_m,
            )
        rgb_image = np.ascontiguousarray(
            cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
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
            latency_ms=float((finished - tracked.host_captured_at) * 1000.0),
            source_age_ms=float(
                max(
                    0.0,
                    tracked.captured_at - cloud.captured_at,
                )
                * 1000.0
            ),
            observation=tracked.observation,
            particle_filter=tracked.particle_filter,
            cube_tracking=tracked.cube_tracking,
            cube_state=tracked.cube_state,
            contact=tracked.contact,
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
    pidnet_viewer_config = config["pidnet"]
    if "runtime_config" not in pidnet_viewer_config:
        raise ValueError("[pidnet].runtime_config is required.")
    pidnet_runtime_path = resolve_project_path(
        pidnet_viewer_config["runtime_config"]
    )
    with pidnet_runtime_path.open("rb") as stream:
        pidnet_runtime = tomllib.load(stream)
    runtime_pidnet_config = pidnet_runtime.get("pidnet")
    if not isinstance(runtime_pidnet_config, dict):
        raise ValueError(
            f"Shared PIDNet runtime config is missing [pidnet]: {pidnet_runtime_path}"
        )
    mask_config = PidNetMaskConfig.from_mapping(pidnet_runtime)
    evaluation_config = config.get("evaluation") or {}
    cube_tracking_values = config.get("cube_tracking") or {}
    cube_tracking_enabled = bool(cube_tracking_values.get("enabled", False))
    cube_tracker_config = (
        CubeTrackerConfig.from_mapping(cube_tracking_values)
        if cube_tracking_enabled
        else None
    )
    cube_state_values = config.get("cube_state_filter") or {}
    cube_state_enabled = bool(
        cube_state_values.get("enabled", cube_tracking_enabled)
    )
    cube_state_config = (
        CubeStateFilterConfig.from_mapping(cube_state_values)
        if cube_tracking_enabled and cube_state_enabled
        else None
    )
    contact_values = config.get("contact") or {}
    contact_enabled = bool(contact_values.get("enabled", False))
    if contact_enabled and cube_state_config is None:
        raise ValueError(
            "Passive contact inference requires cube tracking and the temporal "
            "cube-state filter."
        )
    contact_config = (
        ContactEstimatorConfig.from_mapping(contact_values)
        if contact_enabled
        else None
    )
    observation_config = ObservationConfig.from_mapping(config.get("observation"))
    particle_filter_config = ParticleFilterConfig.from_mapping(config.get("particle_filter"))
    particle_features = ParticleFeatures.from_mapping(config.get("features"))
    checkpoint = resolve_project_path(
        args.checkpoint or runtime_pidnet_config["checkpoint"]
    )

    configured_source = str(viewer_config.get("point_cloud_source", "depth"))
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
    live_evaluation: LiveEvaluation | None = None
    try:
        if bool(evaluation_config.get("enabled", True)):
            live_evaluation = LiveEvaluation(
                resolve_project_path(
                    evaluation_config.get(
                        "directory",
                        "diagnostics/evaluation_runs",
                    )
                ),
                sample_hz=float(evaluation_config.get("sample_hz", 10.0)),
                history_seconds=float(
                    evaluation_config.get("history_seconds", 30.0)
                ),
            )
            live_evaluation.init_window()
            print(
                f"LIVE_EVALUATION_STARTED path={live_evaluation.run_directory}",
                flush=True,
            )
        segmenter = PidNetSegmenter(
            checkpoint,
            device=str(runtime_pidnet_config["device"]),
            amp=bool(runtime_pidnet_config["amp"]),
            channels_last=bool(runtime_pidnet_config["channels_last"]),
        )
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        print(
            "PIDNET_RUNTIME "
            f"checkpoint={checkpoint} sha256={checkpoint_sha256} "
            f"thresholds={'/'.join(f'{value:.8g}' for value in mask_config.thresholds)} "
            f"cleanup=open{mask_config.open_kernel}/close{mask_config.close_kernel}/"
            f"min{mask_config.min_area_px} config={pidnet_runtime_path}",
            flush=True,
        )

        if viewer is not None:
            viewer.update_status("Opening ZED camera...")
            viewer.poll()
        zed, runtime = open_zed(camera_config)
        if viewer is not None:
            configure_viewer_from_zed(zed, viewer)
        camera_model = camera_model_from_zed(zed)
        cube_tracker = (
            KnownCubeTracker(
                CubeCameraModel(
                    fx=camera_model.fx,
                    fy=camera_model.fy,
                    cx=camera_model.cx,
                    cy=camera_model.cy,
                    width=camera_model.width,
                    height=camera_model.height,
                ),
                cube_tracker_config,
            )
            if cube_tracker_config is not None
            else None
        )
        cube_state_filter = (
            RigidCubeStateFilter(cube_state_config)
            if cube_state_config is not None
            else None
        )
        particle_filter = BatchedCableParticleFilter(
            particle_filter_config,
            particle_features,
        )
        contact_estimator = (
            CableCubeContactEstimator(
                contact_config,
                cable_lengths_m=particle_filter_config.cable_lengths_m,
                particle_count=particle_filter_config.particle_count,
                node_count=particle_filter_config.node_count,
                dense_samples_per_segment=(
                    particle_filter_config.dense_samples_per_segment
                ),
                device=particle_filter_config.device,
            )
            if contact_config is not None
            else None
        )
        pipeline = AsyncSegmentationPipeline(
            zed,
            runtime,
            segmenter,
            mask_config,
            float(pidnet_viewer_config.get("overlay_alpha", 0.62)),
            ObservationBuilder(
                observation_config,
                particle_filter_config.cable_lengths_m,
                camera_model,
                particle_filter_config.device,
            ),
            particle_filter,
            cube_tracker,
            cube_state_filter,
            contact_estimator,
            float(viewer_config.get("point_cloud_fps", 5.0)),
            int(viewer_config.get("rgb_width", 620)),
            int(viewer_config.get("point_cloud_stride", 1)),
            visualization_enabled=viewer_enabled,
            visualization_source=configured_source,
            live_evaluation=live_evaluation,
        )
        if viewer is not None:
            viewer.set_feature_controls(particle_features, pipeline.set_feature)
            viewer.set_top_particle_control(
                pipeline.set_top_particles_enabled
            )
            viewer.update_status("Warming PIDNet, cube, and contact paths...")
            viewer.poll()
        warmup_ms = pipeline.warmup(camera_frame_shape(zed))
        print(
            "PIDNet/cube/contact warm-up: "
            f"{warmup_ms:.1f} ms (excluded from frame timing)"
        )
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
            if live_evaluation is not None:
                live_evaluation.poll()
    finally:
        try:
            if pipeline is not None:
                try:
                    pipeline.stop()
                finally:
                    pipeline.free()
        finally:
            try:
                if live_evaluation is not None:
                    live_evaluation.close()
                    print(
                        "LIVE_EVALUATION_SAVED "
                        f"csv={live_evaluation.csv_path} "
                        f"plot={live_evaluation.plot_path}",
                        flush=True,
                    )
            finally:
                try:
                    if zed is not None:
                        zed.close()
                finally:
                    if viewer is not None:
                        viewer.close()


if __name__ == "__main__":
    main()
