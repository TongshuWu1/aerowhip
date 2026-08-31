from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import numpy as np

from .config import RouteSettings
from .contracts import PartialCurveObservation, ViewObservation


def _require_cv2():
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - interactive runtime
        raise RuntimeError("OpenCV is required for PIDNet curve extraction.") from error
    return cv2


def resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] not in (2, 3) or len(values) < 2:
        raise ValueError("Polyline must have shape Nx2 or Nx3 with N >= 2.")
    segment = np.linalg.norm(np.diff(values, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment)))
    if cumulative[-1] <= 1.0e-12:
        raise ValueError("Cannot resample a zero-length polyline.")
    target = np.linspace(0.0, cumulative[-1], int(count))
    return np.stack(
        [np.interp(target, cumulative, values[:, axis]) for axis in range(values.shape[1])],
        axis=1,
    )


def skeletonize_body(mask: np.ndarray) -> np.ndarray:
    """Connectivity-preserving Guo-Hall thinning in a tight foreground ROI."""

    cv2 = _require_cv2()
    binary = np.where(np.asarray(mask, dtype=np.uint8) > 0, 255, 0).astype(np.uint8)
    foreground_y, foreground_x = np.nonzero(binary)
    output = np.zeros(binary.shape, dtype=bool)
    if len(foreground_x) == 0:
        return output
    y0, y1 = max(0, int(foreground_y.min()) - 2), min(
        binary.shape[0], int(foreground_y.max()) + 3
    )
    x0, x1 = max(0, int(foreground_x.min()) - 2), min(
        binary.shape[1], int(foreground_x.max()) + 3
    )
    work = np.ascontiguousarray(binary[y0:y1, x0:x1])
    output[y0:y1, x0:x1] = cv2.ximgproc.thinning(
        work, thinningType=cv2.ximgproc.THINNING_GUOHALL
    ) > 0
    return output


@dataclass(frozen=True, slots=True)
class _Graph:
    pixels_yx: np.ndarray
    index_image: np.ndarray
    neighbour_indices: np.ndarray
    y0: int
    x0: int


@dataclass(frozen=True, slots=True)
class _VisiblePath:
    length_px: float
    points_xy: np.ndarray
    start_endpoint_xy: np.ndarray | None = None
    end_endpoint_xy: np.ndarray | None = None


def _graph(pixels_yx: np.ndarray) -> _Graph:
    pixels = np.asarray(pixels_yx, dtype=np.int32)
    y0, y1 = int(pixels[:, 0].min()) - 1, int(pixels[:, 0].max()) + 2
    x0, x1 = int(pixels[:, 1].min()) - 1, int(pixels[:, 1].max()) + 2
    index = np.full((y1 - y0, x1 - x0), -1, dtype=np.int32)
    index[pixels[:, 0] - y0, pixels[:, 1] - x0] = np.arange(len(pixels))
    neighbours = np.full((len(pixels), 8), -1, dtype=np.int32)
    for column, (dy, dx) in enumerate(
        ((-1, -1), (-1, 0), (-1, 1), (0, -1),
         (0, 1), (1, -1), (1, 0), (1, 1))
    ):
        neighbours[:, column] = index[
            pixels[:, 0] + dy - y0,
            pixels[:, 1] + dx - x0,
        ]
    return _Graph(pixels, index, neighbours, y0, x0)


def _shortest_tree(graph: _Graph, start: int) -> tuple[np.ndarray, np.ndarray]:
    distance = np.full(len(graph.pixels_yx), np.inf, dtype=np.float64)
    predecessor = np.full(len(graph.pixels_yx), -1, dtype=np.int32)
    distance[start] = 0.0
    queue: list[tuple[float, int]] = [(0.0, start)]
    costs = (math.sqrt(2.0), 1.0, math.sqrt(2.0), 1.0,
             1.0, math.sqrt(2.0), 1.0, math.sqrt(2.0))
    while queue:
        current_distance, current = heapq.heappop(queue)
        if current_distance != distance[current]:
            continue
        for target_value, cost in zip(
            graph.neighbour_indices[current], costs, strict=True
        ):
            target = int(target_value)
            if target < 0:
                continue
            candidate = current_distance + cost
            if candidate < distance[target]:
                distance[target], predecessor[target] = candidate, current
                heapq.heappush(queue, (candidate, target))
    return distance, predecessor


def _diameter_path(pixels_yx: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the longest geodesic supported by one connected skeleton component."""

    graph = _graph(pixels_yx)
    first_distance, _ = _shortest_tree(graph, 0)
    first = int(np.argmax(first_distance))
    distance, predecessor = _shortest_tree(graph, first)
    last = int(np.argmax(distance))
    path: list[tuple[float, float]] = []
    current = last
    while current >= 0:
        y, x = graph.pixels_yx[current]
        path.append((float(x), float(y)))
        if current == first:
            break
        current = int(predecessor[current])
    path.reverse()
    return np.asarray(path, dtype=np.float64), float(distance[last])


def _refine_normal_centroid(points_xy: np.ndarray, body_mask: np.ndarray) -> np.ndarray:
    cv2 = _require_cv2()
    points = np.asarray(points_xy, dtype=np.float64)
    tangent = np.empty_like(points)
    tangent[0], tangent[-1] = points[1] - points[0], points[-1] - points[-2]
    if len(points) > 2:
        tangent[1:-1] = points[2:] - points[:-2]
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1.0e-12)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    offsets = np.arange(-6.0, 6.0001, 0.25)
    samples = points[:, None] + offsets[None, :, None] * normal[:, None]
    weights = cv2.remap(
        np.asarray(body_mask, dtype=np.uint8),
        samples[:, :, 0].astype(np.float32), samples[:, :, 1].astype(np.float32),
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    ).astype(np.float64) / 255.0
    weight_sum = weights.sum(axis=1)
    displacement = np.zeros(len(points), dtype=np.float64)
    supported = weight_sum > 1.0e-6
    displacement[supported] = (weights[supported] * offsets).sum(axis=1) / weight_sum[supported]
    return points + np.clip(displacement, -2.5, 2.5)[:, None] * normal


def _endpoint_observations(mask: np.ndarray, minimum_area: int) -> tuple[np.ndarray, np.ndarray, int]:
    cv2 = _require_cv2()
    count, _labels, statistics, centroids = cv2.connectedComponentsWithStats(
        np.asarray(mask, dtype=np.uint8), connectivity=8
    )
    candidates = [
        (int(statistics[label, cv2.CC_STAT_AREA]), np.asarray(centroids[label], dtype=np.float64))
        for label in range(1, int(count))
        if int(statistics[label, cv2.CC_STAT_AREA]) >= int(minimum_area)
    ]
    centers = np.full((2, 2), np.nan, dtype=np.float64)
    valid = np.zeros(2, dtype=bool)
    if len(candidates) == 1:
        centers[0], valid[0] = candidates[0][1], True
    elif len(candidates) >= 2:
        best = max(
            (
                ((min(a[0], b[0]), float(np.linalg.norm(a[1] - b[1])), a[0] + b[0]), a[1], b[1])
                for index, a in enumerate(candidates)
                for b in candidates[index + 1 :]
            ),
            key=lambda item: item[0],
        )
        pair = sorted((best[1], best[2]), key=lambda value: (value[0], value[1]))
        centers[:] = pair
        valid[:] = True
    return centers, valid, len(candidates)


def _path_length(points_xy: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points_xy, axis=0), axis=1).sum())


def _trim_paths_at_endpoints(
    paths: list[_VisiblePath],
    endpoint_centers_xy: np.ndarray,
    endpoint_valid: np.ndarray,
    image_shape: tuple[int, int],
) -> list[_VisiblePath]:
    """Cap visible paths at endpoint centres and remove skeleton beyond them."""

    if not paths or not np.any(endpoint_valid):
        return paths
    maximum_projection_distance = 0.020 * math.hypot(image_shape[1], image_shape[0])
    assignments: list[list[tuple[int, np.ndarray]]] = [[] for _ in paths]
    for center in endpoint_centers_xy[endpoint_valid]:
        best: tuple[float, int, int] | None = None
        for path_index, visible in enumerate(paths):
            distance = np.linalg.norm(visible.points_xy - center, axis=1)
            point_index = int(np.argmin(distance))
            candidate = (float(distance[point_index]), path_index, point_index)
            if best is None or candidate < best:
                best = candidate
        if best is not None and best[0] <= maximum_projection_distance:
            assignments[best[1]].append((best[2], np.asarray(center, dtype=np.float64)))

    result: list[_VisiblePath] = []
    for visible, anchors in zip(paths, assignments, strict=True):
        path = visible.points_xy
        start_endpoint = None
        end_endpoint = None
        if len(anchors) >= 2:
            anchors.sort(key=lambda value: value[0])
            first_index, first_center = anchors[0]
            last_index, last_center = anchors[-1]
            if first_index < last_index:
                path = np.vstack((first_center, path[first_index + 1 : last_index], last_center))
                start_endpoint, end_endpoint = first_center, last_center
        elif len(anchors) == 1:
            point_index, center = anchors[0]
            cumulative = np.concatenate(
                ([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
            )
            if cumulative[point_index] <= 0.5 * cumulative[-1]:
                path = np.vstack((center, path[point_index + 1 :]))
                start_endpoint = center
            else:
                path = np.vstack((path[:point_index], center))
                end_endpoint = center
        if len(path) < 2 or _path_length(path) <= 1.0e-6:
            continue
        result.append(
            _VisiblePath(_path_length(path), path, start_endpoint, end_endpoint)
        )
    return result


def _sample_segments(
    skeleton: np.ndarray,
    body_mask: np.ndarray,
    settings: RouteSettings,
    endpoint_centers_xy: np.ndarray,
    endpoint_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cv2 = _require_cv2()
    component_count, labels = cv2.connectedComponents(
        np.asarray(skeleton, dtype=np.uint8), connectivity=8
    )
    paths: list[_VisiblePath] = []
    for label in range(1, int(component_count)):
        pixels = np.argwhere(labels == label)
        if len(pixels) < 2:
            continue
        path, length = _diameter_path(pixels)
        if length >= settings.minimum_segment_length_px:
            paths.append(_VisiblePath(length, path))
    paths.sort(key=lambda item: item.length_px, reverse=True)
    paths = paths[: settings.maximum_segments]
    paths = _trim_paths_at_endpoints(
        paths, endpoint_centers_xy, endpoint_valid, body_mask.shape
    )
    points = np.full((settings.dense_samples, 2), np.nan, dtype=np.float64)
    valid = np.zeros(settings.dense_samples, dtype=bool)
    segment_ids = np.full(settings.dense_samples, -1, dtype=np.int16)
    if not paths:
        return points, valid, segment_ids
    remaining = settings.dense_samples - 2 * len(paths)
    if remaining < 0:
        raise ValueError("dense_samples must provide at least two samples per visible segment.")
    lengths = np.asarray([item.length_px for item in paths], dtype=np.float64)
    quota = remaining * lengths / lengths.sum()
    allocation = np.full(len(paths), 2, dtype=np.int32) + np.floor(quota).astype(np.int32)
    for index in np.argsort(-(quota - np.floor(quota)))[: settings.dense_samples - int(allocation.sum())]:
        allocation[index] += 1
    cursor = 0
    for segment_id, (visible, count) in enumerate(zip(paths, allocation, strict=True)):
        sampled = _refine_normal_centroid(
            resample_polyline(visible.points_xy, int(count)), body_mask
        )
        if visible.start_endpoint_xy is not None:
            sampled[0] = visible.start_endpoint_xy
        if visible.end_endpoint_xy is not None:
            sampled[-1] = visible.end_endpoint_xy
        points[cursor : cursor + count] = sampled
        valid[cursor : cursor + count] = True
        segment_ids[cursor : cursor + count] = segment_id
        cursor += int(count)
    return points, valid, segment_ids


def extract_partial_curve(
    body_mask: np.ndarray,
    endpoint_mask: np.ndarray,
    body_component_count: int,
    settings: RouteSettings,
    *,
    skeleton: np.ndarray | None = None,
) -> ViewObservation:
    body = np.asarray(body_mask, dtype=np.uint8)
    endpoints = np.asarray(endpoint_mask, dtype=np.uint8)
    if body.ndim != 2 or endpoints.shape != body.shape:
        raise ValueError("Body and endpoint masks must be same-sized 2-D arrays.")
    skeleton = skeletonize_body(body) if skeleton is None else np.asarray(skeleton, dtype=bool)
    endpoint_centers, endpoint_valid, endpoint_count = _endpoint_observations(
        endpoints, settings.endpoint_min_area_px
    )
    points, point_valid, segment_ids = _sample_segments(
        skeleton, body, settings, endpoint_centers, endpoint_valid
    )
    curve = PartialCurveObservation(
        points, point_valid, segment_ids, endpoint_centers, endpoint_valid,
        endpoint_count, int(body_component_count),
    )
    failure = None if np.any(point_valid) else "No visible cable segment passed the length gate."
    return ViewObservation(body, endpoints, curve, failure)


def extract_skeleton_evidence(
    body_mask: np.ndarray,
    endpoint_mask: np.ndarray,
    body_component_count: int,
    settings: RouteSettings,
    *,
    skeleton: np.ndarray,
) -> ViewObservation:
    """Sample visible skeleton pixels when curve order is not required.

    The online posterior uses a one-way observed-point-to-projected-curve
    likelihood.  It is invariant to observation order, so running the full
    geodesic graph extraction after PF initialization would add computation
    without adding information.
    """

    body = np.asarray(body_mask, dtype=np.uint8)
    endpoints = np.asarray(endpoint_mask, dtype=np.uint8)
    skeleton_value = np.asarray(skeleton, dtype=bool)
    if body.ndim != 2 or endpoints.shape != body.shape or skeleton_value.shape != body.shape:
        raise ValueError("Body, endpoints and skeleton must be same-sized 2-D arrays.")
    endpoint_centers, endpoint_valid, endpoint_count = _endpoint_observations(
        endpoints, settings.endpoint_min_area_px
    )
    pixels_yx = np.argwhere(skeleton_value)
    points = np.full((settings.dense_samples, 2), np.nan, dtype=np.float64)
    valid = np.zeros(settings.dense_samples, dtype=bool)
    segment_ids = np.full(settings.dense_samples, -1, dtype=np.int16)
    if len(pixels_yx) >= settings.minimum_segment_length_px:
        count = min(settings.dense_samples, len(pixels_yx))
        indices = np.linspace(0, len(pixels_yx) - 1, count).round().astype(np.int32)
        points[:count] = pixels_yx[indices, ::-1]
        valid[:count] = True
        segment_ids[:count] = 0
    curve = PartialCurveObservation(
        points,
        valid,
        segment_ids,
        endpoint_centers,
        endpoint_valid,
        endpoint_count,
        int(body_component_count),
    )
    failure = None if np.any(valid) else "No visible cable skeleton passed the length gate."
    return ViewObservation(body, endpoints, curve, failure)


def extract_complete_endpoint_route(
    body_mask: np.ndarray,
    endpoint_mask: np.ndarray,
    body_component_count: int,
    settings: RouteSettings,
    *,
    skeleton: np.ndarray | None = None,
) -> ViewObservation:
    """Extract one complete endpoint-to-endpoint route for planar experiments.

    The planar identification protocol deliberately excludes occlusion and
    crossings.  Both endpoint components must attach to one connected skeleton
    component; the geodesic between those attachments is the observed cable.
    """

    cv2 = _require_cv2()
    body = np.asarray(body_mask, dtype=np.uint8)
    endpoints = np.asarray(endpoint_mask, dtype=np.uint8)
    if body.ndim != 2 or endpoints.shape != body.shape:
        raise ValueError("Body and endpoint masks must be same-sized 2-D arrays.")
    skeleton_value = (
        skeletonize_body(body)
        if skeleton is None
        else np.asarray(skeleton, dtype=bool)
    )
    centers, endpoint_valid, endpoint_count = _endpoint_observations(
        endpoints,
        settings.endpoint_min_area_px,
    )
    points = np.full((settings.dense_samples, 2), np.nan, dtype=np.float64)
    valid = np.zeros(settings.dense_samples, dtype=bool)
    segment_ids = np.full(settings.dense_samples, -1, dtype=np.int16)
    failure: str | None = None
    if endpoint_count != 2 or np.count_nonzero(endpoint_valid) != 2:
        failure = "Planar observation requires exactly two endpoint components."
    elif not np.any(skeleton_value):
        failure = "PIDNet cable body produced no skeleton."
    else:
        component_count, labels = cv2.connectedComponents(
            skeleton_value.astype(np.uint8), connectivity=8
        )
        best: tuple[float, np.ndarray, int, int] | None = None
        for label in range(1, int(component_count)):
            pixels = np.argwhere(labels == label)
            if len(pixels) < 2:
                continue
            xy = pixels[:, ::-1].astype(np.float64)
            start = int(np.argmin(np.linalg.norm(xy - centers[0], axis=1)))
            end = int(np.argmin(np.linalg.norm(xy - centers[1], axis=1)))
            if start == end:
                continue
            score = float(
                np.linalg.norm(xy[start] - centers[0])
                + np.linalg.norm(xy[end] - centers[1])
            )
            candidate = (score, pixels, start, end)
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best is None:
            failure = "No connected skeleton component spans both endpoints."
        else:
            attachment_distance, pixels, start, end = best
            maximum_attachment_distance = 0.020 * math.hypot(
                body.shape[1], body.shape[0]
            )
            if attachment_distance > 2.0 * maximum_attachment_distance:
                failure = "Endpoint components do not attach to the cable skeleton."
            else:
                graph = _graph(pixels)
                distance, predecessor = _shortest_tree(graph, start)
                if not np.isfinite(distance[end]):
                    failure = "The selected skeleton endpoints are disconnected."
                else:
                    indices: list[int] = []
                    current = end
                    while current >= 0:
                        indices.append(current)
                        if current == start:
                            break
                        current = int(predecessor[current])
                    indices.reverse()
                    path = graph.pixels_yx[
                        np.asarray(indices, dtype=np.int32)
                    ][:, ::-1]
                    path = np.vstack(
                        (centers[0], path.astype(np.float64), centers[1])
                    )
                    if _path_length(path) < settings.minimum_segment_length_px:
                        failure = (
                            "Endpoint route is shorter than the configured cable route."
                        )
                    else:
                        sampled = _refine_normal_centroid(
                            resample_polyline(path, settings.dense_samples), body
                        )
                        sampled[0], sampled[-1] = centers[0], centers[1]
                        points[:] = sampled
                        valid[:] = True
                        segment_ids[:] = 0
    curve = PartialCurveObservation(
        points,
        valid,
        segment_ids,
        centers,
        endpoint_valid,
        endpoint_count,
        int(body_component_count),
    )
    return ViewObservation(body, endpoints, curve, failure)
