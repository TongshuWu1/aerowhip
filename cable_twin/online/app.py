"""Live single-cable tracking with the identified DDER particle filter."""

from __future__ import annotations

import argparse
from dataclasses import replace
import math
from pathlib import Path
import time

import numpy as np

from ..shared.config import load_settings as load_observation_settings
from ..shared.diagnostics import diagnostic_panels
from ..shared.metric_curve import MetricCurveObservation
from ..shared.model_artifact import DderArtifact, load_dder_artifact
from ..shared.observer import CableFrameObservation, CableObserver
from ..shared.pidnet_runtime import PidnetRuntime
from ..shared.pointcloud_viewer import AsyncZedPointCloudViewer, PointCloudSnapshot
from ..shared.zed_source import ZedStereoSource
from .config import load_settings as load_filter_settings
from .filter import DderParticleFilter, ParticleFilterEstimate


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "offline_dder" / "models" / "cable1_dder.json"


def _measurement_sigma(
    points_m: np.ndarray,
    valid: np.ndarray,
    depth_spread_m: np.ndarray,
    support: np.ndarray,
    unresolved_noise_m: float,
) -> np.ndarray:
    """Combine propagated registered-depth uncertainty with learned residual noise."""

    points = np.asarray(points_m, dtype=np.float64)
    active = np.asarray(valid, dtype=bool) & np.all(np.isfinite(points), axis=-1)
    sigma = np.ones(active.shape, dtype=np.float64)
    if not np.any(active):
        return sigma
    depth = np.abs(points[..., 2])
    ray_norm = np.linalg.norm(points, axis=-1) / np.maximum(
        depth, np.finfo(np.float64).tiny
    )
    sensor = (
        1.2533141373155
        * np.asarray(depth_spread_m, dtype=np.float64)
        / np.sqrt(np.maximum(np.asarray(support, dtype=np.float64), 1.0))
        * ray_norm
    )
    combined = np.sqrt(np.square(sensor) + float(unresolved_noise_m) ** 2)
    if np.any(active & (~np.isfinite(combined) | (combined <= 0.0))):
        raise ValueError("Observed cable samples have invalid measurement uncertainty.")
    sigma[active] = combined[active]
    return sigma


def _metric_arrays(
    observed: CableFrameObservation,
    artifact: DderArtifact,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    curve = observed.view.curve
    metric = observed.metric
    if metric is None:
        count = len(curve.points_xy)
        return (
            np.full((count, 3), np.nan, dtype=np.float32),
            np.zeros(count, dtype=bool),
            np.ones(count, dtype=np.float64),
            np.full((2, 3), np.nan, dtype=np.float32),
            np.ones(2, dtype=np.float64),
        )
    metric_sigma = _measurement_sigma(
        metric.points_camera_m,
        metric.valid,
        metric.depth_spread_m,
        metric.support,
        artifact.unresolved_curve_noise_m,
    )
    endpoint_sigma = _measurement_sigma(
        metric.endpoint_points_camera_m,
        metric.endpoint_valid,
        metric.endpoint_depth_spread_m,
        metric.endpoint_support,
        artifact.unresolved_endpoint_noise_m,
    )
    return (
        metric.points_camera_m,
        metric.valid,
        metric_sigma,
        metric.endpoint_points_camera_m,
        endpoint_sigma,
    )


def _visible_segments(observed: CableFrameObservation) -> tuple[np.ndarray, ...] | None:
    metric = observed.metric
    if metric is None:
        return None
    segment_ids = observed.view.curve.segment_ids
    segments = tuple(
        metric.points_camera_m[metric.valid & (segment_ids == segment)]
        for segment in np.unique(segment_ids[metric.valid])
        if np.count_nonzero(metric.valid & (segment_ids == segment)) >= 2
    )
    return segments or None


def _gravity_ready(samples: list[np.ndarray]) -> tuple[float, float, float] | None:
    if len(samples) < 10:
        return None
    value = np.median(np.asarray(samples[-30:]), axis=0)
    magnitude = float(np.linalg.norm(value))
    if not 8.5 <= magnitude <= 10.8:
        raise ValueError(f"Implausible ZED gravity magnitude {magnitude:.3f} m/s^2.")
    return tuple(float(component) for component in value)


def _status(
    estimate: ParticleFilterEstimate | None,
    observed: CableFrameObservation,
    particle_count: int,
    fps: float,
) -> str:
    metric_count = 0 if observed.metric is None else int(np.count_nonzero(observed.metric.valid))
    endpoint_count = (
        0 if observed.metric is None else int(np.count_nonzero(observed.metric.endpoint_valid))
    )
    if estimate is None:
        return f"WAITING FOR INITIALIZATION | observed {metric_count} | endpoints {endpoint_count}/2 | {fps:.1f} FPS"
    residual = "prediction" if not math.isfinite(estimate.residual_m) else f"{1000.0 * estimate.residual_m:.1f} mm"
    return (
        f"PF + DDER | ESS {estimate.ess:.0f}/{particle_count} | "
        f"residual {residual} | {fps:.1f} FPS"
    )


def run(model_path: str | Path, *, max_frames: int | None = None) -> int:
    artifact = load_dder_artifact(model_path)
    observation_settings = load_observation_settings()
    observation_settings = replace(
        observation_settings,
        cable_identity=artifact.cable_identity,
    )
    filter_settings = load_filter_settings()
    pidnet = PidnetRuntime(observation_settings.pidnet_runtime_config)
    warmup_ms = pidnet.warm_up(1080, 1920)
    print(
        f"Online model: {artifact.path.name} | cable {artifact.cable_identity} | "
        f"PIDNet warm-up {warmup_ms:.1f} ms",
        flush=True,
    )

    source: ZedStereoSource | None = None
    viewer: AsyncZedPointCloudViewer | None = None
    try:
        source = ZedStereoSource()
        observer = CableObserver(
            observation_settings,
            source.descriptor.calibration,
            pidnet=pidnet,
        )
        viewer = AsyncZedPointCloudViewer(
            width_px=observation_settings.viewer_width_px,
            height_px=observation_settings.viewer_height_px,
            stride=observation_settings.depth.viewer_stride_px,
            depth_min_m=observation_settings.depth.minimum_m,
            depth_max_m=observation_settings.depth.maximum_m,
        )
        tracker: DderParticleFilter | None = None
        gravity_samples: list[np.ndarray] = []
        smoothed_fps = 0.0
        processed = 0
        print("Online PF + DDER tracking ready.", flush=True)
        while not viewer.quit_requested:
            if max_frames is not None and processed >= max_frames:
                break
            loop_started = time.perf_counter()
            frame = source.read()
            if frame is None:
                raise RuntimeError("Live ZED capture ended unexpectedly.")
            observed = observer.process(frame)

            if frame.gravity_camera_m_s2 is not None:
                gravity_samples.append(np.asarray(frame.gravity_camera_m_s2, dtype=np.float64))
            if tracker is None:
                gravity = _gravity_ready(gravity_samples)
                if gravity is not None:
                    tracker = DderParticleFilter(
                        artifact.make_model(gravity),
                        filter_settings,
                        artifact,
                    )

            estimate = None
            if tracker is not None:
                points, valid, point_sigma, endpoints, endpoint_sigma = _metric_arrays(
                    observed, artifact
                )
                endpoint_valid = (
                    np.zeros(2, dtype=bool)
                    if observed.metric is None
                    else observed.metric.endpoint_valid
                )
                if tracker.initialized:
                    estimate = tracker.update(
                        timestamp_ns=frame.timestamp_ns,
                        metric_points_m=points,
                        metric_valid=valid,
                        segment_ids=observed.view.curve.segment_ids,
                        metric_sigma_m=point_sigma,
                        endpoints_m=endpoints,
                        endpoint_valid=endpoint_valid,
                        endpoint_sigma_m=endpoint_sigma,
                    )
                else:
                    try:
                        estimate = tracker.initialize(
                            timestamp_ns=frame.timestamp_ns,
                            metric_points_m=points,
                            metric_valid=valid,
                            segment_ids=observed.view.curve.segment_ids,
                            endpoints_m=endpoints,
                            endpoint_valid=endpoint_valid,
                            metric_sigma_m=point_sigma,
                            endpoint_sigma_m=endpoint_sigma,
                        )
                        print(f"PF initialized at ZED frame {frame.source_position}.", flush=True)
                    except ValueError:
                        estimate = None

            elapsed = time.perf_counter() - loop_started
            instantaneous_fps = 1.0 / max(elapsed, 1.0e-6)
            smoothed_fps = (
                instantaneous_fps
                if smoothed_fps == 0.0
                else 0.9 * smoothed_fps + 0.1 * instantaneous_fps
            )
            segmentation_rgb, skeleton_rgb = diagnostic_panels(
                frame.left_bgr, observed.view, observed.skeleton
            )
            viewer.publish(
                PointCloudSnapshot(
                    frame=frame,
                    calibration=source.descriptor.calibration,
                    observed_curve_camera_m=_visible_segments(observed),
                    estimated_centerline_camera_m=(
                        None if estimate is None else estimate.positions_m
                    ),
                    status=_status(
                        estimate,
                        observed,
                        filter_settings.particle_count,
                        smoothed_fps,
                    ),
                    segmentation_rgb=segmentation_rgb,
                    skeleton_graph_rgb=skeleton_rgb,
                )
            )
            processed += 1
            if estimate is not None and (processed == 1 or processed % 30 == 0):
                print(
                    f"online frame={frame.source_position} ESS={estimate.ess:.1f} "
                    f"PF={estimate.timing_ms[-1]:.1f}ms FPS={smoothed_fps:.1f}",
                    flush=True,
                )
        return 0
    finally:
        if source is not None:
            source.close()
        if viewer is not None:
            viewer.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run live ZED cable tracking with the identified DDER particle filter."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-frames", type=int, default=None, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    try:
        code = run(arguments.model, max_frames=arguments.max_frames)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"ONLINE_TRACKING_FAILED: {error}", flush=True)
        raise SystemExit(1) from None
    raise SystemExit(code)


if __name__ == "__main__":
    main()
