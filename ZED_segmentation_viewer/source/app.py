"""Asynchronous ZED capture, PIDNet inference, and full point-cloud coloring."""

from __future__ import annotations

import argparse
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
from observation import FrameObservation, ObservationBuilder, ObservationConfig  # noqa: E402
from particle_filter import (  # noqa: E402
    BatchedCableParticleFilter,
    ParticleFeatures,
    ParticleFilterConfig,
    ParticleFilterFrame,
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


@dataclass(frozen=True)
class TrackingResult:
    """Algorithm output retained only until visualization accepts or drops it."""

    frame_index: int
    captured_at: float
    completed_at: float
    bgr: np.ndarray
    cloud: np.ndarray
    masks: tuple[np.ndarray, ...]
    observation: FrameObservation
    particle_filter: ParticleFilterFrame
    inference_ms: float
    tracking_ms: float


@dataclass(frozen=True)
class SegmentationResult:
    frame_index: int
    captured_at: float
    rgb_image: np.ndarray
    vertices: np.ndarray
    stats: dict[str, int | float]
    scene_center: np.ndarray
    scene_radius: float
    inference_ms: float
    processing_ms: float
    latency_ms: float
    observation: FrameObservation
    particle_filter: ParticleFilterFrame


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


def decode_zed_rgba(color_values: np.ndarray) -> np.ndarray:
    """Decode ZED's packed RGBA float without changing dark RGB values."""

    packed_float = np.ascontiguousarray(color_values, dtype=np.float32).reshape(-1)
    packed = packed_float.view(np.uint32)
    rgb = np.empty((packed.size, 3), dtype=np.float32)
    rgb[:, 0] = (packed & 0x000000FF).astype(np.float32) / 255.0
    rgb[:, 1] = ((packed & 0x0000FF00) >> 8).astype(np.float32) / 255.0
    rgb[:, 2] = ((packed & 0x00FF0000) >> 16).astype(np.float32) / 255.0
    return rgb


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
) -> tuple[np.ndarray, dict[str, int], np.ndarray, float]:
    """Color every renderable ZED point; apply no stride, cap, or range filter."""

    cloud = np.asarray(point_data)
    if cloud.ndim != 3 or cloud.shape[2] < 4:
        raise ValueError(f"ZED point cloud must have shape HxWx4; got {cloud.shape}.")
    if len(masks) < 4:
        raise ValueError(
            f"PIDNet must return cable, two endpoint, and crossing masks; got {len(masks)}."
        )

    height, width = cloud.shape[:2]
    shape = (height, width)
    cable = _boolean_mask(masks[0], shape).reshape(-1)
    endpoint_1 = _boolean_mask(masks[1], shape).reshape(-1)
    endpoint_2 = _boolean_mask(masks[2], shape).reshape(-1)
    crossing = _boolean_mask(masks[3], shape).reshape(-1)

    flat = cloud[:, :, :4].reshape(-1, 4)
    xyz_all = np.asarray(flat[:, :3], dtype=np.float32)
    finite = np.all(np.isfinite(xyz_all), axis=1)
    valid_count = int(np.count_nonzero(finite))
    xyz = np.ascontiguousarray(xyz_all[finite], dtype=np.float32)
    rgb = decode_zed_rgba(flat[finite, 3])

    # Lowest-to-highest priority: cable, endpoint 2, endpoint 1, crossing.
    rgb[cable[finite]] = CABLE_RGB
    rgb[endpoint_2[finite]] = ENDPOINT_2_RGB
    rgb[endpoint_1[finite]] = ENDPOINT_1_RGB
    rgb[crossing[finite]] = CROSSING_RGB

    vertices = np.empty((len(xyz), 6), dtype=np.float32)
    vertices[:, :3] = xyz
    vertices[:, 3:] = rgb
    vertices = np.ascontiguousarray(vertices)
    stats = {
        "total_pixels": int(height * width),
        "valid_points": valid_count,
        "invalid_points": int(finite.size - valid_count),
        "cable_points": int(np.count_nonzero(finite & cable)),
        "endpoint_1_points": int(np.count_nonzero(finite & endpoint_1)),
        "endpoint_2_points": int(np.count_nonzero(finite & endpoint_2)),
        "crossing_points": int(np.count_nonzero(finite & crossing)),
    }
    center, radius = _scene_bounds(xyz)
    return vertices, stats, center, radius


def segmentation_overlay(
    bgr: np.ndarray,
    masks: tuple[np.ndarray, ...],
    alpha: float,
) -> np.ndarray:
    """Render the same deterministic class colors over the left camera image."""

    image = np.ascontiguousarray(np.asarray(bgr, dtype=np.uint8)[:, :, :3])
    shape = image.shape[:2]
    cable = _boolean_mask(masks[0], shape)
    endpoint_1 = _boolean_mask(masks[1], shape)
    endpoint_2 = _boolean_mask(masks[2], shape)
    crossing = _boolean_mask(masks[3], shape)
    labels = np.zeros(shape, dtype=np.uint8)
    labels[cable] = 1
    labels[endpoint_2] = 3
    labels[endpoint_1] = 2
    labels[crossing] = 4

    overlay = image.copy()
    overlay[labels == 1] = CABLE_BGR
    overlay[labels == 3] = ENDPOINT_2_BGR
    overlay[labels == 2] = ENDPOINT_1_BGR
    overlay[labels == 4] = CROSSING_BGR
    selected = labels > 0
    output = image.copy()
    if np.any(selected):
        blend = float(np.clip(alpha, 0.0, 1.0))
        output[selected] = np.rint(
            (1.0 - blend) * image[selected].astype(np.float32)
            + blend * overlay[selected].astype(np.float32)
        ).astype(np.uint8)
    return np.ascontiguousarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))


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
    ):
        self.zed = zed
        self.runtime = runtime
        self.segmenter = segmenter
        self.thresholds = thresholds
        self.overlay_alpha = float(overlay_alpha)
        self.observation_builder = observation_builder
        self.particle_filter = particle_filter
        self.visualization_period = 1.0 / max(0.1, float(visualization_max_fps))
        self.next_visualization_time = 0.0
        self.image = sl.Mat()
        self.point_cloud = sl.Mat()
        self.tracking_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tracking")
        self.visualization_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="visualization",
        )
        self.tracking_future: Future | None = None
        self.visualization_future: Future | None = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._capture_loop, name="zed-capture", daemon=True)
        self.started = False
        self.lock = threading.Lock()
        self.latest_result: SegmentationResult | None = None
        self.error: str | None = None
        self.frame_count = 0
        self.tracking_submitted = 0
        self.tracking_completed = 0
        self.tracking_busy_frames = 0
        self.visualization_submitted = 0
        self.visualization_completed = 0
        self.visualization_dropped = 0
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
        self.pending_features = particle_filter.features

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
            self.segmenter.observation_channels,
            blank,
            self.thresholds,
        )
        future.result()
        return float((time.perf_counter() - started) * 1000.0)

    def stop(self) -> None:
        self.stop_event.set()
        if self.started:
            self.thread.join(timeout=5.0)
            if self.thread.is_alive():
                raise RuntimeError("ZED capture thread did not stop within five seconds.")
        self.tracking_executor.shutdown(wait=True, cancel_futures=True)
        self.visualization_executor.shutdown(wait=True, cancel_futures=True)

    def free(self) -> None:
        self.image.free()
        self.point_cloud.free()

    def snapshot(self) -> tuple[SegmentationResult | None, dict[str, float | int]]:
        with self.lock:
            if self.error is not None:
                raise RuntimeError(f"Asynchronous ZED pipeline failed:\n{self.error}")
            return self.latest_result, {
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
                "tracking_submitted": int(self.tracking_submitted),
                "tracking_completed": int(self.tracking_completed),
                "tracking_busy_frames": int(self.tracking_busy_frames),
                "visualization_submitted": int(self.visualization_submitted),
                "visualization_completed": int(self.visualization_completed),
                "visualization_dropped": int(self.visualization_dropped),
            }

    def set_feature(self, name: str, enabled: bool) -> None:
        """Queue one ablation change; the capture thread applies it between frames."""

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
        result = self.visualization_future.result()
        self.visualization_future = None
        with self.lock:
            self.latest_result = result
            self.visualization_completed += 1
            self.last_visualization_ms = float(result.processing_ms)

    def _drain_tracking(self) -> None:
        if self.tracking_future is None or not self.tracking_future.done():
            return
        result = self.tracking_future.result()
        self.tracking_future = None
        with self.lock:
            self.tracking_completed += 1
            self.last_inference_ms = float(result.inference_ms)
            self.last_tracking_ms = float(result.tracking_ms)
            self.last_observation_ms = float(result.observation.processing_ms)
            self.last_particle_filter_ms = float(result.particle_filter.processing_ms)
            self.last_particle_filter_gpu_ms = float(result.particle_filter.gpu_ms)

        # Visualization is best effort: never queue it and never hold tracking.
        now = time.perf_counter()
        if self.visualization_future is None and now >= self.next_visualization_time:
            self.visualization_future = self.visualization_executor.submit(
                self._prepare_visualization,
                result,
            )
            self.next_visualization_time = now + self.visualization_period
            with self.lock:
                self.visualization_submitted += 1
        else:
            with self.lock:
                self.visualization_dropped += 1

    def _capture_loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                status = self.zed.grab(self.runtime)
                if status != sl.ERROR_CODE.SUCCESS:
                    continue
                captured_at = time.perf_counter()
                with self.lock:
                    self.frame_count += 1
                    frame_index = int(self.frame_count)
                self._drain_visualization()
                self._drain_tracking()
                self._update_capture_fps(captured_at)
                self._update_stage_rates(captured_at)

                if self.tracking_future is not None:
                    with self.lock:
                        self.tracking_busy_frames += 1
                    continue

                with self.lock:
                    requested_features = self.pending_features
                if requested_features != self.particle_filter.features:
                    self.particle_filter.set_features(requested_features)

                image_status = self.zed.retrieve_image(self.image, sl.VIEW.LEFT, sl.MEM.CPU)
                cloud_status = self.zed.retrieve_measure(
                    self.point_cloud,
                    sl.MEASURE.XYZRGBA,
                    sl.MEM.CPU,
                )
                if image_status != sl.ERROR_CODE.SUCCESS or cloud_status != sl.ERROR_CODE.SUCCESS:
                    raise RuntimeError(
                        f"ZED retrieval failed: image={image_status}, cloud={cloud_status}"
                    )
                bgra = np.asarray(self.image.get_data())
                bgr = np.ascontiguousarray(bgra[:, :, :3].copy())
                cloud = np.ascontiguousarray(self.point_cloud.get_data().copy(), dtype=np.float32)
                self.tracking_future = self.tracking_executor.submit(
                    self._run_tracking,
                    frame_index,
                    captured_at,
                    bgr,
                    cloud,
                )
                with self.lock:
                    self.tracking_submitted += 1
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
        if elapsed < 0.5:
            return
        with self.lock:
            self.tracking_fps = float(
                (self.tracking_completed - self._last_tracking_completed) / max(elapsed, 1e-6)
            )
            self.visualization_fps = float(
                (self.visualization_completed - self._last_visualization_completed)
                / max(elapsed, 1e-6)
            )
            self._last_tracking_completed = int(self.tracking_completed)
            self._last_visualization_completed = int(self.visualization_completed)
        self._last_rate_time = now

    def _run_tracking(
        self,
        frame_index: int,
        captured_at: float,
        bgr: np.ndarray,
        cloud: np.ndarray,
    ) -> TrackingResult:
        tracking_start = time.perf_counter()
        inference_start = tracking_start
        masks, _crossing_probability = self.segmenter.observation_channels(bgr, self.thresholds)
        inference_finished = time.perf_counter()
        observation = self.observation_builder.build(masks, cloud)
        particle_filter = self.particle_filter.update(observation, captured_at)
        completed_at = time.perf_counter()
        return TrackingResult(
            frame_index=frame_index,
            captured_at=captured_at,
            completed_at=completed_at,
            bgr=bgr,
            cloud=cloud,
            masks=masks,
            observation=observation,
            particle_filter=particle_filter,
            inference_ms=float((inference_finished - inference_start) * 1000.0),
            tracking_ms=float((completed_at - tracking_start) * 1000.0),
        )

    def _prepare_visualization(self, tracked: TrackingResult) -> SegmentationResult:
        visualization_start = time.perf_counter()
        vertices, stats, center, radius = full_point_cloud_vertices(tracked.cloud, tracked.masks)
        rgb_image = segmentation_overlay(tracked.bgr, tracked.masks, self.overlay_alpha)
        finished = time.perf_counter()
        return SegmentationResult(
            frame_index=tracked.frame_index,
            captured_at=tracked.captured_at,
            rgb_image=rgb_image,
            vertices=vertices,
            stats=stats,
            scene_center=center,
            scene_radius=radius,
            inference_ms=float(tracked.inference_ms),
            processing_ms=float((finished - visualization_start) * 1000.0),
            latency_ms=float((finished - tracked.captured_at) * 1000.0),
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

    width = int(viewer_config.get("rgb_width", 620)) + int(viewer_config.get("cloud_width", 1180))
    viewer = SplitPointCloudViewer(
        width=width,
        height=int(viewer_config.get("height", 900)),
        left_panel_width=int(viewer_config.get("rgb_width", 620)),
        point_size=float(viewer_config.get("point_size", 1.0)),
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

        viewer.update_status("Opening ZED camera...")
        viewer.poll()
        zed, runtime = open_zed(camera_config)
        configure_viewer_from_zed(zed, viewer)
        pipeline = AsyncSegmentationPipeline(
            zed,
            runtime,
            segmenter,
            thresholds,
            float(pidnet_config.get("overlay_alpha", 0.62)),
            ObservationBuilder(
                observation_config,
                particle_filter_config.cable_lengths_m,
            ),
            BatchedCableParticleFilter(particle_filter_config, particle_features),
            float(viewer_config.get("point_cloud_fps", 5.0)),
        )
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
        while viewer.is_available():
            result, pipeline_stats = pipeline.snapshot()
            now = time.perf_counter()
            if now >= next_display:
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
                viewer.close()


if __name__ == "__main__":
    main()
