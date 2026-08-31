from __future__ import annotations

import numpy as np

from .contracts import ViewObservation
from .routes import skeletonize_body


def diagnostic_panels(
    bgr: np.ndarray,
    observation: ViewObservation,
    skeleton: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return RGB segmentation and skeleton-graph panels for the viewer."""

    try:
        import cv2
    except ImportError as error:  # pragma: no cover - interactive runtime
        raise RuntimeError("OpenCV is required for viewer diagnostics.") from error

    source_full = np.asarray(bgr, dtype=np.uint8)
    display_width = min(640, source_full.shape[1])
    display_height = max(1, int(round(display_width * source_full.shape[0] / source_full.shape[1])))
    source = cv2.resize(source_full, (display_width, display_height), interpolation=cv2.INTER_AREA)
    scale = np.asarray(
        (display_width / source_full.shape[1], display_height / source_full.shape[0]),
        dtype=np.float64,
    )
    body = cv2.resize(
        observation.body_mask,
        (display_width, display_height),
        interpolation=cv2.INTER_NEAREST,
    ) != 0
    endpoints = cv2.resize(
        observation.endpoint_mask,
        (display_width, display_height),
        interpolation=cv2.INTER_NEAREST,
    ) != 0
    segmentation = np.array(source, copy=True)
    overlay = np.zeros_like(segmentation)
    overlay[body] = (45, 220, 65)
    overlay[endpoints] = (0, 155, 255)
    selected = body | endpoints
    segmentation[selected] = np.rint(
        0.52 * segmentation[selected] + 0.48 * overlay[selected]
    ).astype(np.uint8)
    curve = observation.curve
    for segment_id in np.unique(curve.segment_ids[curve.point_valid]):
        selected_points = curve.points_xy[curve.point_valid & (curve.segment_ids == segment_id)]
        route = np.rint(selected_points * scale).astype(np.int32)
        cv2.polylines(segmentation, [route.reshape(-1, 1, 2)], False, (255, 255, 0), 3, cv2.LINE_AA)
    for point in np.rint(curve.endpoint_centers_xy[curve.endpoint_valid] * scale).astype(np.int32):
        cv2.circle(segmentation, tuple(point), 7, (0, 60, 255), -1, cv2.LINE_AA)

    skeleton_full = skeletonize_body(observation.body_mask) if skeleton is None else np.asarray(skeleton, dtype=bool)
    skeleton = cv2.resize(
        skeleton_full.astype(np.uint8),
        (display_width, display_height),
        interpolation=cv2.INTER_NEAREST,
    ) != 0
    graph = np.full(source.shape, (13, 18, 24), dtype=np.uint8)
    graph[skeleton] = (235, 210, 35)
    if np.any(skeleton):
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0
        degree = cv2.filter2D(skeleton.astype(np.uint8), cv2.CV_16U, kernel)
        terminal = skeleton & (degree == 1)
        junction = skeleton & (degree >= 3)
        graph[terminal] = (0, 180, 255)
        graph[junction] = (40, 40, 255)
    colors = ((45, 245, 80), (255, 180, 30), (220, 80, 255), (70, 210, 255))
    for segment_id in np.unique(curve.segment_ids[curve.point_valid]):
        selected_points = curve.points_xy[curve.point_valid & (curve.segment_ids == segment_id)]
        route = np.rint(selected_points * scale).astype(np.int32)
        cv2.polylines(
            graph, [route.reshape(-1, 1, 2)], False,
            colors[int(segment_id) % len(colors)], 2, cv2.LINE_AA,
        )
    for point in np.rint(curve.endpoint_centers_xy[curve.endpoint_valid] * scale).astype(np.int32):
        cv2.circle(graph, tuple(point), 8, (255, 80, 20), 2, cv2.LINE_AA)

    return (
        np.ascontiguousarray(segmentation[:, :, ::-1]),
        np.ascontiguousarray(graph[:, :, ::-1]),
    )
