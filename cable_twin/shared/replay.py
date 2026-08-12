from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .observation_data import load_observation, sha256_file, trajectory_path

from .config import load_settings
from .diagnostics import diagnostic_panels
from .metric_curve import orient_view_to_previous
from .pidnet_runtime import PidnetRuntime
from .pointcloud_viewer import AsyncZedPointCloudViewer, PointCloudSnapshot
from .routes import extract_partial_curve, skeletonize_body
from .zed_source import ZedStereoSource


def run(svo_path: Path, observation_path: Path, fitted_path: Path | None = None) -> int:
    sequence = load_observation(observation_path)
    trajectory = trajectory_path(observation_path) if fitted_path is None else fitted_path.resolve()
    nodes = None
    if trajectory.is_file():
        with np.load(trajectory, allow_pickle=False) as data:
            try:
                metadata = json.loads(str(data["metadata_json"].item()))
            except (KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
                raise ValueError("Fitted trajectory metadata is invalid.") from error
            nodes = np.asarray(data["positions_m"], dtype=np.float64)
            timestamps = np.asarray(data["timestamp_ns"], dtype=np.int64)
        if metadata.get("observation_sha256") != sha256_file(observation_path):
            raise ValueError("Fitted trajectory does not belong to this observation archive.")
        if nodes.ndim != 3 or nodes.shape[0] != sequence.frame_count or nodes.shape[2] != 3:
            raise ValueError("Fitted trajectory shape does not match the observation archive.")
        if not np.array_equal(timestamps, sequence.timestamps_ns):
            raise ValueError("Fitted trajectory timestamps do not match the observation archive.")
        kind = metadata.get("metrics", {}).get("trajectory_kind")
        status = {
            "held_out_recursive_prediction": "held-out DDER prediction",
            "causal_dder_particle_filter_map": "causal DDER particle filter",
        }.get(kind, "fixed-length pointcloud reconstruction")
    else:
        status = "raw registered-depth curve; fit not run"

    settings = load_settings()
    pidnet = PidnetRuntime(settings.pidnet_runtime_config)
    pidnet.warm_up(1080, 1920)
    source = ZedStereoSource(svo_path)
    viewer = AsyncZedPointCloudViewer(
        width_px=settings.viewer_width_px,
        height_px=settings.viewer_height_px,
        stride=settings.depth.viewer_stride_px,
        depth_min_m=settings.depth.minimum_m,
        depth_max_m=settings.depth.maximum_m,
    )
    try:
        index = 0
        replay_wall_start = time.perf_counter()
        replay_timestamp_start = None
        previous_endpoints = None
        while index < sequence.frame_count and not viewer.quit_requested:
            frame = source.read()
            if frame is None:
                break
            if frame.source_position < sequence.source_positions[index]:
                continue
            if frame.source_position != sequence.source_positions[index]:
                raise ValueError("SVO frame positions do not match the observation archive.")
            valid = sequence.metric_valid[index]
            observed_segments = tuple(
                sequence.metric_points_m[
                    index,
                    valid & (sequence.route_segment_id[index] == segment_id),
                ]
                for segment_id in np.unique(sequence.route_segment_id[index, valid])
                if np.count_nonzero(
                    valid & (sequence.route_segment_id[index] == segment_id)
                ) >= 2
            ) or None
            estimate = None if nodes is None else nodes[index]
            result = pidnet.infer(frame.left_bgr)
            skeleton = skeletonize_body(result.masks[0])
            view = extract_partial_curve(
                result.masks[0],
                result.masks[sequence.cable_identity],
                result.body_component_count,
                settings.route,
                skeleton=skeleton,
            )
            view, previous_endpoints = orient_view_to_previous(view, previous_endpoints)
            segmentation_rgb, skeleton_rgb = diagnostic_panels(frame.left_bgr, view, skeleton)
            viewer.publish(
                PointCloudSnapshot(
                    frame=frame,
                    calibration=source.descriptor.calibration,
                    observed_curve_camera_m=observed_segments,
                    estimated_centerline_camera_m=estimate,
                    status=status,
                    segmentation_rgb=segmentation_rgb,
                    skeleton_graph_rgb=skeleton_rgb,
                )
            )
            if replay_timestamp_start is None:
                replay_timestamp_start = frame.timestamp_ns
            target_elapsed = (frame.timestamp_ns - replay_timestamp_start) * 1.0e-9
            remaining = target_elapsed - (time.perf_counter() - replay_wall_start)
            if remaining > 0.0:
                time.sleep(min(0.10, remaining))
            index += 1
        if index != sequence.frame_count and not viewer.quit_requested:
            raise RuntimeError(f"Replay ended early: {index}/{sequence.frame_count} frames.")
    finally:
        source.close()
        viewer.close()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay ZED point cloud and reconstructed DDER cable.")
    parser.add_argument("--svo", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, default=None)
    arguments = parser.parse_args()
    raise SystemExit(run(arguments.svo, arguments.observation, arguments.trajectory))


if __name__ == "__main__":
    main()
