"""View RGB observations beside the metric-scaled image-plane trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .planar_data import load_planar_sequence, load_planar_trajectory
from ..shared.observation_data import trajectory_path
from ..shared.zed_source import ZedStereoSource


def _metric_panel(
    route_xz: np.ndarray,
    nodes_xz: np.ndarray,
    *,
    complete: bool,
    bounds_xz: tuple[float, float, float, float],
    width_px: int,
    height_px: int,
) -> np.ndarray:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - interactive runtime
        raise RuntimeError("OpenCV is required for the 2D trajectory viewer.") from error
    panel = np.full((height_px, width_px, 3), (13, 20, 29), dtype=np.uint8)
    margin = 35
    x_min, x_max, z_min, z_max = bounds_xz

    def pixels(points: np.ndarray) -> np.ndarray:
        x = margin + (points[:, 0] - x_min) / (x_max - x_min) * (width_px - 2 * margin)
        y = height_px - margin - (points[:, 1] - z_min) / (z_max - z_min) * (height_px - 2 * margin)
        return np.rint(np.column_stack((x, y))).astype(np.int32)

    cv2.rectangle(panel, (margin, margin), (width_px - margin, height_px - margin), (70, 85, 100), 1)
    if complete:
        cv2.polylines(panel, [pixels(route_xz)[:, None]], False, (0, 210, 255), 2, cv2.LINE_AA)
    if np.all(np.isfinite(nodes_xz)):
        node_pixels = pixels(nodes_xz)
        cv2.polylines(panel, [node_pixels[:, None]], False, (70, 245, 95), 3, cv2.LINE_AA)
        for point in node_pixels:
            cv2.circle(panel, tuple(point), 3, (240, 245, 250), -1, cv2.LINE_AA)
    cv2.putText(panel, "PIDNET ROUTE / 24 OBSERVATION NODES", (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (230, 235, 240), 1, cv2.LINE_AA)
    return panel


def _metric_bounds(
    points_xz: np.ndarray,
    *,
    cable_length_m: float,
    panel_aspect: float,
) -> tuple[float, float, float, float]:
    points = np.asarray(points_xz, dtype=np.float64).reshape(-1, 2)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) == 0:
        half = 0.55 * cable_length_m
        return -half, half, -half, half
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = 0.5 * (lower + upper)
    span = np.maximum(upper - lower, 0.10 * cable_length_m)
    span += 0.10 * cable_length_m
    if span[0] / span[1] < panel_aspect:
        span[0] = span[1] * panel_aspect
    else:
        span[1] = span[0] / panel_aspect
    return (
        float(center[0] - 0.5 * span[0]),
        float(center[0] + 0.5 * span[0]),
        float(center[1] - 0.5 * span[1]),
        float(center[1] + 0.5 * span[1]),
    )


def run(svo_path: Path, observation_path: Path, trajectory_value: Path | None = None) -> int:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - interactive runtime
        raise RuntimeError("OpenCV is required for the 2D trajectory viewer.") from error
    sequence = load_planar_sequence(observation_path)
    trajectory_file = trajectory_path(observation_path) if trajectory_value is None else trajectory_value
    trajectory = load_planar_trajectory(
        trajectory_file,
        observation_path=observation_path,
    )
    mapping = sequence.metadata["image_plane_mapping"]
    scale_m_per_px = float(mapping["scale_m_per_px"])
    image_origin_px = np.asarray(mapping["image_origin_px"], dtype=np.float64)
    cable_length_m = float(mapping["cable_length_m"])
    metric_width_px = 520
    image_size_px = np.asarray(mapping["image_size_px"], dtype=np.int64)
    display_width = 900
    display_height = int(round(display_width * image_size_px[1] / image_size_px[0]))
    bounds = _metric_bounds(
        trajectory.positions_xz_m[trajectory.valid],
        cable_length_m=cable_length_m,
        panel_aspect=(metric_width_px - 70) / (display_height - 70),
    )
    source = ZedStereoSource(svo_path, enable_depth=False)
    paused = False
    frame_index = 0
    try:
        while frame_index < sequence.frame_count:
            frame = source.read()
            if frame is None:
                break
            image = np.array(frame.left_bgr, copy=True)
            complete = bool(sequence.complete[frame_index])
            if complete:
                route_px = np.rint(sequence.route_xy_px[frame_index]).astype(np.int32)
                cv2.polylines(image, [route_px[:, None]], False, (0, 220, 255), 3, cv2.LINE_AA)
            nodes_xz = trajectory.positions_xz_m[frame_index]
            if bool(trajectory.valid[frame_index]):
                nodes_px = np.empty_like(nodes_xz)
                nodes_px[:, 0] = nodes_xz[:, 0] / scale_m_per_px + image_origin_px[0]
                nodes_px[:, 1] = image_origin_px[1] - nodes_xz[:, 1] / scale_m_per_px
                nodes_px = np.rint(nodes_px).astype(np.int32)
                cv2.polylines(image, [nodes_px[:, None]], False, (70, 245, 95), 3, cv2.LINE_AA)
                for point in nodes_px:
                    cv2.circle(image, tuple(point), 3, (245, 245, 245), -1, cv2.LINE_AA)
            rgb_panel = cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_AREA)
            metric_panel = _metric_panel(
                sequence.route_xz_m[frame_index],
                nodes_xz,
                complete=complete,
                bounds_xz=bounds,
                width_px=metric_width_px,
                height_px=display_height,
            )
            panel = np.hstack((rgb_panel, metric_panel))
            cv2.rectangle(panel, (0, 0), (panel.shape[1], 35), (10, 18, 28), -1)
            cv2.putText(
                panel,
                f"frame {frame_index + 1}/{sequence.frame_count} | "
                f"{'VALID' if trajectory.valid[frame_index] else 'NO COMPLETE ROUTE'} | SPACE pause | Q close",
                (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (95, 235, 130) if trajectory.valid[frame_index] else (50, 180, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Image-plane 2D cable trajectory", panel)
            while True:
                key = cv2.waitKey(0 if paused else 1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    return 0
                if key == 32:
                    paused = not paused
                    if paused:
                        continue
                break
            frame_index += 1
    finally:
        source.close()
        cv2.destroyAllWindows()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="View a metric-scaled image-plane cable trajectory.")
    parser.add_argument("--svo", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, default=None)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    raise SystemExit(run(arguments.svo, arguments.observation, arguments.trajectory))


if __name__ == "__main__":
    main()
