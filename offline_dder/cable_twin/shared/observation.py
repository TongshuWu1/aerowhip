from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .capture_manifest import load_capture_manifest
from .config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_settings
from .contracts import POINT_CLOUD_CURVE_METHOD
from .diagnostics import diagnostic_panels
from .observer import CableObserver
from .pidnet_runtime import PidnetRuntime
from .pointcloud_viewer import AsyncZedPointCloudViewer, PointCloudSnapshot
from .saving import ObservationSequenceWriter
from .zed_source import ZedStereoSource


def _default_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return PROJECT_ROOT / "data" / "offline_dder" / "observations" / f"pointcloud_{timestamp}.npz"


def _source_metadata(source: ZedStereoSource) -> dict[str, Any]:
    descriptor = source.descriptor
    calibration = descriptor.calibration
    metadata = {
        "kind": descriptor.kind,
        "label": descriptor.label,
        "source_path": None if descriptor.source_path is None else str(descriptor.source_path),
        "total_frames": descriptor.total_frames,
        "calibration": {
            "width_px": calibration.width_px,
            "height_px": calibration.height_px,
            "fps": calibration.fps,
            "left_intrinsics": calibration.left_intrinsics.tolist(),
            "right_intrinsics": calibration.right_intrinsics.tolist(),
            "baseline_m": calibration.baseline_m,
            "serial_number": calibration.serial_number,
            "camera_model": calibration.camera_model,
            "imu_to_camera_transform": calibration.imu_to_camera_transform.tolist(),
        },
        "registered_depth": {
            "sdk_mode": "NEURAL_PLUS",
            "coordinate_system": "IMAGE",
            "units": "METER",
        },
    }
    if descriptor.source_path is not None:
        metadata["capture_manifest"] = load_capture_manifest(descriptor.source_path)
    return metadata


def _observation_metadata(
    settings: Any,
    pidnet: PidnetRuntime,
    source: ZedStereoSource,
) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": POINT_CLOUD_CURVE_METHOD,
        "evidence": (
            "independent PIDNet endpoints and disconnected visible cable segments "
            "lifted from registered ZED depth; no binary end-to-end route gate"
        ),
        "settings": asdict(settings),
        "pidnet": pidnet.identity,
        "source": _source_metadata(source),
    }


def analyze_svo(
    svo_path: str | Path,
    output_path: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    cable_identity: int | None = None,
    replace_existing: bool = False,
    headless: bool = True,
    max_frames: int | None = None,
) -> Path:
    arguments = argparse.Namespace(
        svo=Path(svo_path),
        save=Path(output_path),
        config=Path(config_path),
        cable=cable_identity,
        replace_output=replace_existing,
        headless=headless,
        max_frames=max_frames,
    )
    if run(arguments) != 0:
        raise RuntimeError("Registered-depth PIDNet analysis did not complete.")
    return Path(output_path).expanduser().resolve()


def run(arguments: argparse.Namespace) -> int:
    settings = load_settings(arguments.config)
    if arguments.cable is not None:
        settings = replace(settings, cable_identity=int(arguments.cable))
    output_path = Path(arguments.save or _default_output_path()).expanduser().resolve()

    source: ZedStereoSource | None = None
    viewer: AsyncZedPointCloudViewer | None = None
    writer: ObservationSequenceWriter | None = None
    processed = 0
    metric_routes = 0
    complete = False
    try:
        pidnet = PidnetRuntime(settings.pidnet_runtime_config)
        warmup_ms = pidnet.warm_up(1080, 1920)
        thresholds = "/".join(f"{value:.2f}" for value in pidnet.mask_config.thresholds)
        print(
            f"PIDNet {pidnet.checkpoint_path.name} "
            f"sha256={pidnet.identity['checkpoint_sha256'][:12]} thresholds={thresholds}"
        )
        print(f"PIDNet CUDA warm-up {warmup_ms:.1f} ms.")
        source = ZedStereoSource(arguments.svo)
        observer = CableObserver(
            settings,
            source.descriptor.calibration,
            pidnet=pidnet,
        )
        if not arguments.headless:
            viewer = AsyncZedPointCloudViewer(
                width_px=settings.viewer_width_px,
                height_px=settings.viewer_height_px,
                stride=settings.depth.viewer_stride_px,
                depth_min_m=settings.depth.minimum_m,
                depth_max_m=settings.depth.maximum_m,
            )
        writer = ObservationSequenceWriter(
            output_path,
            metadata=_observation_metadata(settings, pidnet, source),
            route_samples=settings.route.dense_samples,
            replace_existing=arguments.replace_output,
        )
        print(
            f"Analyzing registered ZED depth: {source.descriptor.label} | "
            f"cable {settings.cable_identity}"
        )
        while viewer is None or not viewer.quit_requested:
            if arguments.max_frames is not None and processed >= arguments.max_frames:
                break
            frame = source.read()
            if frame is None:
                break
            observed = observer.process(frame)
            left, metric = observed.view, observed.metric
            if metric is not None and np.count_nonzero(metric.valid) >= 12:
                metric_routes += 1
            writer.append(frame, left, metric, timings_ms=observed.timings_ms)
            if viewer is not None:
                segmentation_rgb, skeleton_rgb = diagnostic_panels(
                    frame.left_bgr, left, observed.skeleton
                )
                visible_segments: tuple[np.ndarray, ...] = ()
                if metric is not None:
                    visible_segments = tuple(
                        metric.points_camera_m[
                            metric.valid & (left.curve.segment_ids == segment_id)
                        ]
                        for segment_id in np.unique(left.curve.segment_ids[metric.valid])
                        if np.count_nonzero(
                            metric.valid & (left.curve.segment_ids == segment_id)
                        ) >= 2
                    )
                viewer.publish(
                    PointCloudSnapshot(
                        frame=frame,
                        calibration=source.descriptor.calibration,
                        observed_curve_camera_m=visible_segments or None,
                        status=(
                            f"visible metric {0 if metric is None else np.count_nonzero(metric.valid)}"
                            f"/{settings.route.dense_samples} | endpoints "
                            f"{0 if metric is None else np.count_nonzero(metric.endpoint_valid)}/2"
                        ),
                        segmentation_rgb=segmentation_rgb,
                        skeleton_graph_rgb=skeleton_rgb,
                    )
                )
            processed += 1
            total = source.descriptor.total_frames
            if total is not None and (processed == 1 or processed % 30 == 0):
                print(f"PROGRESS {processed} {total}", flush=True)
            if processed == 1 or processed % 30 == 0:
                print(
                    f"analyze frame={frame.source_position} "
                    f"pidnet={observed.pidnet.inference_ms:.1f}ms "
                    f"route={observed.timings_ms['route_ms']:.1f}ms "
                    f"depth={observed.timings_ms['depth_lift_ms']:.1f}ms "
                    f"metric={0 if metric is None else np.count_nonzero(metric.valid)}/{settings.route.dense_samples}",
                    flush=True,
                )
        total = source.descriptor.total_frames
        if arguments.max_frames is None and total is not None and processed != total:
            raise RuntimeError(f"SVO analysis ended early: {processed}/{total} frames.")
        complete = True
    finally:
        if source is not None:
            source.close()
        if viewer is not None:
            viewer.close()
        if writer is not None:
            if complete:
                writer.close()
            else:
                writer.abort()
    print(
        f"Saved {processed} synchronized frames with {metric_routes} usable metric curves: {output_path}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract PIDNet-selected metric cable curves from registered ZED depth."
    )
    parser.add_argument("--svo", type=Path, required=True)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--cable", type=int, choices=(1, 2), default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--replace-output", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))
