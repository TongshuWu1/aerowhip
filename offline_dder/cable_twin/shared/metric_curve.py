from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import PartialCurveObservation, ViewObservation


@dataclass(frozen=True, slots=True)
class MetricCurveObservation:
    """Partial metric curve segments and independent metric endpoints."""

    points_camera_m: np.ndarray
    valid: np.ndarray
    depth_spread_m: np.ndarray
    support: np.ndarray
    endpoint_points_camera_m: np.ndarray
    endpoint_valid: np.ndarray
    endpoint_depth_spread_m: np.ndarray
    endpoint_support: np.ndarray

    def __post_init__(self) -> None:
        points = np.ascontiguousarray(self.points_camera_m, dtype=np.float32)
        valid = np.ascontiguousarray(self.valid, dtype=bool)
        spread = np.ascontiguousarray(self.depth_spread_m, dtype=np.float32)
        support = np.ascontiguousarray(self.support, dtype=np.int16)
        endpoints = np.ascontiguousarray(self.endpoint_points_camera_m, dtype=np.float32)
        endpoint_valid = np.ascontiguousarray(self.endpoint_valid, dtype=bool)
        endpoint_spread = np.ascontiguousarray(self.endpoint_depth_spread_m, dtype=np.float32)
        endpoint_support = np.ascontiguousarray(self.endpoint_support, dtype=np.int16)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Metric curve points must have shape Mx3.")
        if valid.shape != (len(points),) or spread.shape != valid.shape or support.shape != valid.shape:
            raise ValueError("Metric curve quality arrays must contain one value per point.")
        if endpoints.shape != (2, 3) or endpoint_valid.shape != (2,):
            raise ValueError("Metric endpoints must have shapes 2x3 and 2.")
        if endpoint_spread.shape != (2,) or endpoint_support.shape != (2,):
            raise ValueError("Metric endpoint quality arrays must contain two values.")
        if np.any(valid & ~np.all(np.isfinite(points), axis=1)):
            raise ValueError("Valid metric curve points must be finite.")
        if np.any(endpoint_valid & ~np.all(np.isfinite(endpoints), axis=1)):
            raise ValueError("Valid metric endpoints must be finite.")
        for value in (
            points, valid, spread, support, endpoints, endpoint_valid,
            endpoint_spread, endpoint_support,
        ):
            value.setflags(write=False)
        object.__setattr__(self, "points_camera_m", points)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "depth_spread_m", spread)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "endpoint_points_camera_m", endpoints)
        object.__setattr__(self, "endpoint_valid", endpoint_valid)
        object.__setattr__(self, "endpoint_depth_spread_m", endpoint_spread)
        object.__setattr__(self, "endpoint_support", endpoint_support)


def orient_view_to_previous(
    view: ViewObservation,
    previous_endpoints_xy: np.ndarray | None,
) -> tuple[ViewObservation, np.ndarray | None]:
    curve = view.curve
    centers = np.array(curve.endpoint_centers_xy, copy=True)
    valid = np.array(curve.endpoint_valid, copy=True)
    if not np.any(valid):
        return view, previous_endpoints_xy
    if previous_endpoints_xy is not None:
        previous = np.asarray(previous_endpoints_xy, dtype=np.float64)
        if np.all(valid) and np.all(np.isfinite(previous)):
            direct = np.linalg.norm(centers - previous, axis=1).sum()
            flipped = np.linalg.norm(centers[::-1] - previous, axis=1).sum()
            if flipped < direct:
                centers, valid = centers[::-1].copy(), valid[::-1].copy()
        elif np.count_nonzero(valid) == 1:
            observed = centers[valid][0]
            finite_previous = np.all(np.isfinite(previous), axis=1)
            if np.any(finite_previous):
                slots = np.flatnonzero(finite_previous)
                target = int(slots[np.argmin(np.linalg.norm(previous[slots] - observed, axis=1))])
                centers[:] = np.nan
                valid[:] = False
                centers[target], valid[target] = observed, True
    oriented = PartialCurveObservation(
        curve.points_xy, curve.point_valid, curve.segment_ids,
        centers, valid, curve.endpoint_component_count, curve.body_component_count,
    )
    next_previous = (
        np.array(centers, copy=True)
        if previous_endpoints_xy is None
        else np.array(previous_endpoints_xy, copy=True)
    )
    next_previous[valid] = centers[valid]
    return ViewObservation(view.body_mask, view.endpoint_mask, oriented, view.failure), next_previous


def _lift_samples(
    points_xy: np.ndarray,
    requested: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    *,
    radius_px: int,
    minimum_support: int,
    maximum_cluster_span_m: float,
    depth_min_m: float,
    depth_max_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k = np.asarray(intrinsics, dtype=np.float64)
    fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    count = len(points_xy)
    points = np.full((count, 3), np.nan, dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    spread = np.full(count, np.nan, dtype=np.float32)
    support = np.zeros(count, dtype=np.int16)
    height, width = depth.shape
    offsets_y, offsets_x = np.mgrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    disk = offsets_x * offsets_x + offsets_y * offsets_y <= radius_px * radius_px
    offsets_x, offsets_y = offsets_x[disk], offsets_y[disk]
    for index in np.flatnonzero(requested):
        x, y = points_xy[index]
        columns, rows = int(round(float(x))) + offsets_x, int(round(float(y))) + offsets_y
        inside = (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
        columns, rows = columns[inside], rows[inside]
        selected = mask[rows, columns]
        columns, rows = columns[selected], rows[selected]
        values = depth[rows, columns]
        finite = np.isfinite(values) & (values >= depth_min_m) & (values <= depth_max_m)
        columns, rows, values = columns[finite], rows[finite], values[finite]
        if len(values) < minimum_support:
            continue
        order = np.argsort(values)
        values, columns, rows = values[order], columns[order], rows[order]
        chosen = None
        for start in range(len(values) - minimum_support + 1):
            if float(values[start + minimum_support - 1] - values[start]) <= maximum_cluster_span_m:
                stop = start + minimum_support
                while stop < len(values) and float(values[stop] - values[start]) <= maximum_cluster_span_m:
                    stop += 1
                chosen = slice(start, stop)
                break
        if chosen is None:
            continue
        z, u, v = values[chosen].astype(np.float64), columns[chosen], rows[chosen]
        xyz = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
        points[index] = np.median(xyz, axis=0)
        median_z = float(np.median(z))
        spread[index] = 1.4826 * np.median(np.abs(z - median_z))
        support[index] = min(len(z), np.iinfo(np.int16).max)
        valid[index] = True
    return points, valid, spread, support


def _reject_depth_islands(
    points: np.ndarray,
    valid: np.ndarray,
    spread: np.ndarray,
    support: np.ndarray,
    segment_ids: np.ndarray,
    maximum_cluster_span_m: float,
) -> None:
    generator = np.random.default_rng(0)
    for segment_id in np.unique(segment_ids[valid]):
        indices = np.flatnonzero(valid & (segment_ids == segment_id))
        if len(indices) < 4:
            continue
        coordinate = np.linspace(0.0, 1.0, len(indices))
        measured_z = points[indices, 2].astype(np.float64)
        degree, subset_size = min(3, len(indices) - 1), min(4, len(indices))
        candidates = [np.linspace(0, len(indices) - 1, subset_size).round().astype(int)]
        candidates.extend(
            np.sort(generator.choice(len(indices), subset_size, replace=False)) for _ in range(96)
        )
        best = np.zeros(len(indices), dtype=bool)
        best_score = -np.inf
        threshold = max(0.020, maximum_cluster_span_m)
        for subset in candidates:
            coefficients = np.polyfit(coordinate[subset], measured_z[subset], degree)
            inlier = np.abs(np.polyval(coefficients, coordinate) - measured_z) <= threshold
            score = float(np.count_nonzero(inlier)) + 0.002 * float(support[indices[inlier]].sum())
            if score > best_score:
                best, best_score = inlier, score
        rejected = indices[~best]
        valid[rejected] = False
        points[rejected] = np.nan
        spread[rejected] = np.nan
        support[rejected] = 0


def lift_partial_curve_from_registered_depth(
    view: ViewObservation,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    *,
    radius_px: int,
    minimum_support: int,
    maximum_cluster_span_m: float,
    depth_min_m: float,
    depth_max_m: float,
) -> MetricCurveObservation:
    depth = np.asarray(depth_m, dtype=np.float32)
    mask = np.asarray(view.body_mask, dtype=bool) | np.asarray(view.endpoint_mask, dtype=bool)
    if depth.shape != mask.shape:
        raise ValueError("Registered depth and PIDNet mask dimensions differ.")
    curve = view.curve
    points, valid, spread, support = _lift_samples(
        curve.points_xy, curve.point_valid, depth, mask, intrinsics,
        radius_px=radius_px, minimum_support=minimum_support,
        maximum_cluster_span_m=maximum_cluster_span_m,
        depth_min_m=depth_min_m, depth_max_m=depth_max_m,
    )
    _reject_depth_islands(
        points, valid, spread, support, curve.segment_ids, maximum_cluster_span_m
    )
    endpoints, endpoint_valid, endpoint_spread, endpoint_support = _lift_samples(
        curve.endpoint_centers_xy, curve.endpoint_valid, depth, mask, intrinsics,
        radius_px=radius_px, minimum_support=minimum_support,
        maximum_cluster_span_m=maximum_cluster_span_m,
        depth_min_m=depth_min_m, depth_max_m=depth_max_m,
    )
    return MetricCurveObservation(
        points, valid, spread, support,
        endpoints, endpoint_valid, endpoint_spread, endpoint_support,
    )
