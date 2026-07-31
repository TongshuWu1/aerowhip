"""Endpoint-conditioned graph observations for indistinguishable cable pixels."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import heapq
import itertools
import time

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class CameraModel:
    """Calibrated pinhole projection for the ZED left camera."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    forward_sign: float = -1.0
    image_y_sign: float = -1.0


@dataclass(frozen=True)
class EndpointMeasurement:
    pixel_xy: np.ndarray
    xyz: np.ndarray
    area_px: int


@dataclass(frozen=True)
class RouteHypothesis:
    """One edge-simple trail through the complete observed skeleton graph."""

    route_xyz: np.ndarray
    length_m: float


@dataclass(frozen=True)
class GraphEndpointAnchor:
    """Known endpoint identity attached to one end of an observed graph edge."""

    edge_end: int
    cable_index: int
    endpoint_index: int


@dataclass(frozen=True)
class GraphEdgeObservation:
    """One ordered, directly observed 3D skeleton edge."""

    component_label: int
    edge_id: int
    xyz: np.ndarray
    pixels_xy: np.ndarray
    length_m: float
    endpoint_anchors: tuple[GraphEndpointAnchor, ...] = ()


@dataclass(frozen=True)
class ObservationProfile:
    """Stage timings and input complexity for one observation build."""

    total_ms: float
    masks_ms: float
    geometry_ms: float
    maps_cuda_ms: float
    unprojection_cuda_ms: float
    readback_wall_ms: float
    connected_components_ms: float
    endpoints_ms: float
    component_preparation_ms: float
    skeleton_ms: float
    graph_ms: float
    graph_edges_ms: float
    route_search_ms: float
    route_assembly_ms: float
    finalization_ms: float
    unaccounted_ms: float
    cable_pixels: int
    relevant_pixels: int
    valid_3d_points: int
    component_count: int
    processed_component_count: int
    endpoint_count: int
    skeleton_pixels: int
    graph_nodes: int
    graph_edges: int
    branch_pixels: int
    route_candidates: int
    graph_edge_count: int


@dataclass(frozen=True)
class CableObservation:
    valid: bool
    reason: str
    endpoints_xyz: np.ndarray
    endpoint_pixels_xy: np.ndarray
    endpoint_visible: np.ndarray
    endpoint_component_labels: np.ndarray
    routes: tuple[RouteHypothesis, ...]


@dataclass(frozen=True)
class SkeletonDebug:
    """Viewer-rate image-space topology produced by the observation graph."""

    pixels_xy: np.ndarray
    node_pixels_xy: np.ndarray
    branch_pixels_xy: np.ndarray


@dataclass(frozen=True)
class FrameObservation:
    cables: tuple[CableObservation, CableObservation]
    processing_ms: float
    graph_edges: tuple[GraphEdgeObservation, ...] = ()
    profile: ObservationProfile | None = None
    skeleton_debug: SkeletonDebug | None = None


@dataclass(frozen=True)
class ObservationConfig:
    endpoint_min_area_px: int = 24
    endpoint_label_search_px: int = 18
    route_depth_radius_px: int = 2
    route_samples: int = 128
    graph_node_dilation_px: int = 3
    route_candidate_limit: int = 16
    route_search_state_limit: int = 4096
    route_edge_limit: int = 24
    route_length_margin_m: float = 0.18
    graph_edge_samples: int = 24
    component_min_area_px: int = 8
    endpoint_association_gate_m: float = 0.12
    profile_stages: bool = True
    profile_console_period_s: float = 1.0

    @classmethod
    def from_mapping(cls, values: dict | None) -> "ObservationConfig":
        values = values or {}
        return cls(
            endpoint_min_area_px=max(1, int(values.get("endpoint_min_area_px", 24))),
            endpoint_label_search_px=max(1, int(values.get("endpoint_label_search_px", 18))),
            route_depth_radius_px=max(0, int(values.get("route_depth_radius_px", 2))),
            route_samples=max(16, int(values.get("route_samples", 128))),
            graph_node_dilation_px=max(
                0, int(values.get("graph_node_dilation_px", 3))
            ),
            route_candidate_limit=max(1, int(values.get("route_candidate_limit", 16))),
            route_search_state_limit=max(
                64, int(values.get("route_search_state_limit", 4096))
            ),
            route_edge_limit=max(2, int(values.get("route_edge_limit", 24))),
            route_length_margin_m=max(
                0.01, float(values.get("route_length_margin_m", 0.18))
            ),
            graph_edge_samples=max(4, int(values.get("graph_edge_samples", 24))),
            component_min_area_px=max(
                2,
                int(values.get("component_min_area_px", 8)),
            ),
            endpoint_association_gate_m=max(
                0.01, float(values.get("endpoint_association_gate_m", 0.12))
            ),
            profile_stages=bool(values.get("profile_stages", True)),
            profile_console_period_s=max(
                0.0, float(values.get("profile_console_period_s", 1.0))
            ),
        )


@dataclass(frozen=True)
class _SegmentedGeometry:
    """Compact CPU view of XYZ values unprojected together on the GPU."""

    xyz: np.ndarray
    index_image: np.ndarray
    valid: np.ndarray

    def points_at(self, pixel_y: np.ndarray, pixel_x: np.ndarray) -> np.ndarray:
        indices = np.asarray(self.index_image[pixel_y, pixel_x], dtype=np.int32)
        flat_indices = indices.reshape(-1)
        points = np.full((flat_indices.size, 3), np.nan, dtype=np.float32)
        selected = flat_indices >= 0
        if np.any(selected):
            points[selected] = self.xyz[flat_indices[selected]]
        return np.ascontiguousarray(points.reshape(indices.shape + (3,)))


@dataclass(frozen=True)
class _GeometryProfile:
    wall_ms: float
    maps_cuda_ms: float
    unprojection_cuda_ms: float
    readback_wall_ms: float
    relevant_pixels: int
    valid_3d_points: int


@dataclass(frozen=True)
class _CableAssemblyProfile:
    route_search_ms: float
    route_assembly_ms: float
    finalization_ms: float


@dataclass(frozen=True)
class _GraphEdge:
    edge_id: int
    node_a: int
    node_b: int
    pixels_xy: np.ndarray
    xyz: np.ndarray
    length_m: float


@dataclass(frozen=True)
class _SkeletonGraph:
    edges: tuple[_GraphEdge, ...]
    endpoint_nodes: tuple[int, ...]
    node_count: int
    branch_pixels: int
    node_pixels_xy: np.ndarray
    branch_pixels_xy: np.ndarray


def _empty_cable(reason: str) -> CableObservation:
    return CableObservation(
        valid=False,
        reason=str(reason),
        endpoints_xyz=np.full((2, 3), np.nan, dtype=np.float32),
        endpoint_pixels_xy=np.full((2, 2), np.nan, dtype=np.float32),
        endpoint_visible=np.zeros(2, dtype=bool),
        endpoint_component_labels=np.zeros(2, dtype=np.int32),
        routes=(),
    )


def _mask_u8(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(mask)
    if array.shape != shape:
        array = cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    return np.ascontiguousarray(array > 0, dtype=np.uint8)


def _component_roi(
    labels: np.ndarray,
    stats: np.ndarray,
    component_label: int,
    padding: int = 2,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Return one exact component crop without scanning the full label image."""

    left = int(stats[component_label, cv2.CC_STAT_LEFT])
    top = int(stats[component_label, cv2.CC_STAT_TOP])
    width = int(stats[component_label, cv2.CC_STAT_WIDTH])
    height = int(stats[component_label, cv2.CC_STAT_HEIGHT])
    x0 = max(0, left - padding)
    x1 = min(labels.shape[1], left + width + padding)
    y0 = max(0, top - padding)
    y1 = min(labels.shape[0], top + height + padding)
    return labels[y0:y1, x0:x1] == int(component_label), (y0, x0)


def _resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or len(points) < 2:
        dimensions = points.shape[1] if points.ndim == 2 else 0
        return np.empty((0, dimensions), dtype=np.float32)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], segment_lengths > 1e-9))
    points = points[keep]
    if len(points) < 2:
        return np.repeat(points[:1], count, axis=0).astype(np.float32)
    cumulative = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    total = float(cumulative[-1])
    if not np.isfinite(total) or total <= 1e-9:
        return np.repeat(points[:1], count, axis=0).astype(np.float32)
    targets = np.linspace(0.0, total, int(count), dtype=np.float64)
    output = np.empty((len(targets), points.shape[1]), dtype=np.float64)
    for axis in range(points.shape[1]):
        output[:, axis] = np.interp(targets, cumulative, points[:, axis])
    return np.ascontiguousarray(output, dtype=np.float32)


def _thinning_lookup_tables() -> tuple[np.ndarray, np.ndarray]:
    tables = []
    for first_step in (True, False):
        table = np.zeros(256, dtype=np.uint8)
        for code in range(256):
            neighbors = [(code >> bit) & 1 for bit in range(8)]
            count = sum(neighbors)
            transitions = sum(
                neighbors[index] == 0 and neighbors[(index + 1) % 8] == 1
                for index in range(8)
            )
            p2, _, p4, _, p6, _, p8, _ = neighbors
            removable = 2 <= count <= 6 and transitions == 1
            if first_step:
                removable &= p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0
            else:
                removable &= p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0
            table[code] = int(removable)
        tables.append(table)
    return tables[0], tables[1]


_THINNING_LUTS = _thinning_lookup_tables()
_OFFSETS = (
    (-1, 0),
    (0, 1),
    (1, 0),
    (0, -1),
    (-1, -1),
    (-1, 1),
    (1, 1),
    (1, -1),
)


def _morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    """Exact Zhang-Suen thinning evaluated only at foreground pixels."""

    source = np.asarray(mask, dtype=np.uint8)
    if source.ndim != 2:
        raise ValueError(f"Skeleton mask must be two-dimensional, got {source.shape}")
    if not np.any(source):
        return np.zeros_like(source, dtype=bool)

    # Padding provides the same zero boundary condition as the former full-crop
    # OpenCV convolution. Zhang-Suen only removes pixels, so the active set can
    # be compacted after every sub-iteration instead of rescanning empty crop
    # pixels. The LUT and simultaneous deletion rule are unchanged.
    image = np.pad(source > 0, 1, mode="constant").astype(np.uint8, copy=False)
    flat = image.reshape(-1)
    stride = image.shape[1]
    active = np.flatnonzero(flat)
    neighbor_offsets = np.asarray(
        (
            -stride,
            -stride + 1,
            1,
            stride + 1,
            stride,
            stride - 1,
            -1,
            -stride - 1,
        ),
        dtype=np.int64,
    )
    neighbor_weights = np.asarray((1, 2, 4, 8, 16, 32, 64, 128), dtype=np.uint16)
    while True:
        removed = 0
        for table in _THINNING_LUTS:
            codes = np.sum(
                flat[active[:, None] + neighbor_offsets[None, :]]
                * neighbor_weights[None, :],
                axis=1,
                dtype=np.uint16,
            )
            delete = table[codes] > 0
            count = int(np.count_nonzero(delete))
            if count:
                flat[active[delete]] = 0
                active = active[~delete]
                removed += count
        if removed == 0:
            break
    return image[1:-1, 1:-1] > 0


def _nearest_pixel(mask: np.ndarray, pixel_xy: np.ndarray) -> tuple[int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    dx = xs.astype(np.float64) - float(pixel_xy[0])
    dy = ys.astype(np.float64) - float(pixel_xy[1])
    index = int(np.argmin(dx * dx + dy * dy))
    return int(ys[index]), int(xs[index])


def _nearest_component_label(
    labels: np.ndarray,
    pixel_xy: np.ndarray,
    radius: int,
) -> int:
    x = int(round(float(pixel_xy[0])))
    y = int(round(float(pixel_xy[1])))
    height, width = labels.shape
    if 0 <= x < width and 0 <= y < height and labels[y, x] > 0:
        return int(labels[y, x])
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    window = labels[y0:y1, x0:x1]
    ys, xs = np.nonzero(window > 0)
    if len(xs) == 0:
        return 0
    dx = (xs + x0).astype(np.float64) - float(pixel_xy[0])
    dy = (ys + y0).astype(np.float64) - float(pixel_xy[1])
    index = int(np.argmin(dx * dx + dy * dy))
    return int(window[ys[index], xs[index]])


def _lift_route_samples(
    route_yx: np.ndarray,
    geometry: _SegmentedGeometry,
    component_labels: np.ndarray,
    component_label: int,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift pixels with the exact median of valid same-component neighbors."""

    route_yx = np.asarray(route_yx, dtype=np.int32)
    sample_count = len(route_yx)
    if sample_count == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty(0, dtype=bool),
        )
    if len(geometry.xyz) == 0:
        return (
            np.full((sample_count, 3), np.nan, dtype=np.float32),
            np.zeros(sample_count, dtype=bool),
        )
    height, width = component_labels.shape
    offsets = np.arange(-radius, radius + 1, dtype=np.int32)
    offset_y, offset_x = np.meshgrid(offsets, offsets, indexing="ij")
    sample_y = route_yx[:, 0, None] + offset_y.reshape(1, -1)
    sample_x = route_yx[:, 1, None] + offset_x.reshape(1, -1)
    in_bounds = (
        (sample_y >= 0) & (sample_y < height) & (sample_x >= 0) & (sample_x < width)
    )
    safe_y = np.clip(sample_y, 0, height - 1)
    safe_x = np.clip(sample_x, 0, width - 1)
    selected = (
        in_bounds
        & (component_labels[safe_y, safe_x] == int(component_label))
        & geometry.valid[safe_y, safe_x]
    )
    point_indices = geometry.index_image[safe_y, safe_x]
    local_xyz = geometry.xyz[np.maximum(point_indices, 0)]

    # All three coordinates share one validity mask. Sorting finite values
    # before +inf reproduces np.nanmedian exactly while avoiding its masked
    # array construction and per-axis NaN bookkeeping for this small window.
    ordered = np.sort(
        np.where(selected[..., None], local_xyz, np.float32(np.inf)),
        axis=1,
    )
    valid_counts = np.sum(selected, axis=1)
    rows = np.arange(sample_count)
    lower = ordered[rows, np.maximum(0, (valid_counts - 1) // 2)]
    upper = ordered[rows, valid_counts // 2]
    points = (lower + upper) * np.float32(0.5)
    valid = valid_counts > 0
    points[~valid] = np.nan
    return np.ascontiguousarray(points, dtype=np.float32), valid


def _pixel_adjacency(
    skeleton: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the corner-pruned pixel graph in compact array form."""

    coordinates = np.column_stack(np.nonzero(skeleton)).astype(np.int32)
    height, width = skeleton.shape
    index_image = np.full((height, width), -1, dtype=np.int32)
    if len(coordinates) == 0:
        return coordinates, np.empty((0, len(_OFFSETS)), dtype=np.int32), index_image
    index_image[coordinates[:, 0], coordinates[:, 1]] = np.arange(
        len(coordinates), dtype=np.int32
    )
    adjacency = np.full((len(coordinates), len(_OFFSETS)), -1, dtype=np.int32)
    y = coordinates[:, 0]
    x = coordinates[:, 1]
    for slot, (dy, dx) in enumerate(_OFFSETS):
        neighbor_y = y + dy
        neighbor_x = x + dx
        in_bounds = (
            (neighbor_y >= 0)
            & (neighbor_y < height)
            & (neighbor_x >= 0)
            & (neighbor_x < width)
        )
        rows = np.flatnonzero(in_bounds)
        if len(rows) == 0:
            continue
        neighbor = index_image[neighbor_y[rows], neighbor_x[rows]]
        present = neighbor >= 0
        if dy != 0 and dx != 0:
            # Match the original corner-pruning rule: a diagonal edge is
            # omitted when either intervening axial skeleton pixel exists.
            blocked = (
                (index_image[neighbor_y[rows], x[rows]] >= 0)
                | (index_image[y[rows], neighbor_x[rows]] >= 0)
            )
            present &= ~blocked
        selected = rows[present]
        adjacency[selected, slot] = neighbor[present]
    return coordinates, adjacency, index_image


def _order_nonbranching_chain(
    component_indices: np.ndarray,
    adjacency: np.ndarray,
    start_index: int,
    end_index: int,
) -> list[int]:
    """Walk one compressed edge, choosing the shortest direction for a loop."""

    if start_index == end_index:
        return [start_index]
    membership = np.zeros(len(adjacency), dtype=bool)
    membership[component_indices] = True
    if not membership[start_index] or not membership[end_index]:
        return []

    component_adjacency = adjacency[component_indices]
    neighbor_in_component = (component_adjacency >= 0) & membership[
        np.maximum(component_adjacency, 0)
    ]
    if np.any(np.sum(neighbor_in_component, axis=1) > 2):
        raise ValueError("compressed graph edge contains an internal branch")

    paths = []
    for first_value in adjacency[start_index]:
        first = int(first_value)
        if first < 0 or not membership[first]:
            continue
        path = [start_index, first]
        previous, current = start_index, first
        while current != end_index and len(path) <= len(component_indices) + 1:
            next_index = -1
            for candidate_value in adjacency[current]:
                candidate = int(candidate_value)
                if (
                    candidate >= 0
                    and candidate != previous
                    and membership[candidate]
                ):
                    next_index = candidate
                    break
            if next_index < 0 or next_index == start_index:
                break
            path.append(next_index)
            previous, current = current, next_index
        if current == end_index:
            paths.append(path)
    return min(paths, key=len) if paths else []


def _build_skeleton_graph(
    skeleton: np.ndarray,
    anchors_yx: tuple[tuple[int, int], ...],
    crop_origin_yx: tuple[int, int],
    component_labels: np.ndarray,
    component_label: int,
    geometry: _SegmentedGeometry,
    depth_radius: int,
    node_dilation: int,
) -> tuple[_SkeletonGraph | None, str]:
    coordinates, adjacency, index_image = _pixel_adjacency(skeleton)
    if len(coordinates) < 2:
        return None, "skeleton contains fewer than two pixels"
    anchor_indices = []
    for anchor in anchors_yx:
        y, x = int(anchor[0]), int(anchor[1])
        if not (0 <= y < skeleton.shape[0] and 0 <= x < skeleton.shape[1]):
            return None, "endpoint anchor is absent from skeleton"
        index = int(index_image[y, x])
        if index < 0:
            return None, "endpoint anchor is absent from skeleton"
        anchor_indices.append(index)

    node_seed = np.zeros_like(skeleton, dtype=np.uint8)
    degree = np.count_nonzero(adjacency >= 0, axis=1)
    node_indices = np.flatnonzero(degree != 2)
    node_seed[
        coordinates[node_indices, 0], coordinates[node_indices, 1]
    ] = 1
    for index in anchor_indices:
        y, x = coordinates[index]
        node_seed[y, x] = 1
    if not np.any(node_seed):
        # A closed skeleton loop has degree two everywhere. One explicit graph
        # node turns it into a self-loop edge without inventing a break or path.
        y, x = coordinates[0]
        node_seed[y, x] = 1
    if node_dilation:
        node_region = cv2.dilate(
            node_seed,
            np.ones((3, 3), dtype=np.uint8),
            iterations=node_dilation,
        )
        node_region = (node_region > 0) & skeleton
    else:
        node_region = (node_seed > 0) & skeleton
    node_count, node_labels = cv2.connectedComponents(
        node_region.astype(np.uint8), connectivity=8
    )
    if node_count <= 1:
        return None, "skeleton graph has no nodes"

    node_label_by_index = node_labels[coordinates[:, 0], coordinates[:, 1]]
    node_representatives: dict[int, int] = {}
    for node_label in range(1, node_count):
        member_indices = np.flatnonzero(node_label_by_index == node_label)
        node_pixels = coordinates[member_indices]
        center = np.mean(node_pixels, axis=0)
        distance = np.sum((node_pixels - center[None, :]) ** 2, axis=1)
        node_representatives[node_label] = int(
            member_indices[int(np.argmin(distance))]
        )

    endpoint_nodes = tuple(int(node_label_by_index[index]) for index in anchor_indices)
    if any(node <= 0 for node in endpoint_nodes):
        return None, "endpoint anchor was not assigned to a graph node"

    chain_mask = skeleton & ~node_region
    chain_count, chain_labels = cv2.connectedComponents(
        chain_mask.astype(np.uint8), connectivity=8
    )
    chain_label_by_index = chain_labels[coordinates[:, 0], coordinates[:, 1]]
    crop_y, crop_x = crop_origin_yx
    global_coordinates_yx = coordinates + np.asarray(
        (crop_y, crop_x), dtype=np.int32
    )
    # Every edge is assembled from this same skeleton pixel set. Lifting once
    # preserves the exact local-median observation while avoiding one full
    # neighborhood construction and nanmedian call per graph edge.
    lifted_xyz, lifted_valid = _lift_route_samples(
        global_coordinates_yx,
        geometry,
        component_labels,
        component_label,
        depth_radius,
    )
    edges: list[_GraphEdge] = []
    for chain_label in range(1, chain_count):
        member_indices = np.flatnonzero(chain_label_by_index == chain_label)
        if len(member_indices) == 0:
            continue
        neighbor_indices = adjacency[member_indices].reshape(-1)
        source_indices = np.repeat(member_indices, adjacency.shape[1])
        valid_neighbors = neighbor_indices >= 0
        neighbor_indices = neighbor_indices[valid_neighbors]
        source_indices = source_indices[valid_neighbors]
        neighbor_node_labels = node_label_by_index[neighbor_indices]
        touches_node = neighbor_node_labels > 0
        neighbor_node_labels = neighbor_node_labels[touches_node]
        source_indices = source_indices[touches_node]
        contacts = {
            int(node_label): np.unique(source_indices[neighbor_node_labels == node_label])
            for node_label in np.unique(neighbor_node_labels)
        }
        contact_nodes = sorted(contacts)
        if len(contact_nodes) > 2 or len(contact_nodes) == 0:
            continue
        if len(contact_nodes) == 2:
            node_a, node_b = contact_nodes
            start_candidates = contacts[node_a]
            end_candidates = contacts[node_b]
        else:
            node_a = node_b = contact_nodes[0]
            boundary = contacts[node_a]
            if len(boundary) < 2:
                continue
            boundary_yx = coordinates[boundary].astype(np.float64)
            pair_distance = np.sum(
                (boundary_yx[:, None, :] - boundary_yx[None, :, :]) ** 2,
                axis=-1,
            )
            first, second = np.unravel_index(np.argmax(pair_distance), pair_distance.shape)
            start_candidates = {boundary[int(first)]}
            end_candidates = {boundary[int(second)]}

        representative_a = coordinates[node_representatives[node_a]]
        representative_b = coordinates[node_representatives[node_b]]
        start_index = min(
            (int(index) for index in start_candidates),
            key=lambda index: float(
                np.sum((coordinates[index] - representative_a) ** 2)
            ),
        )
        end_index = min(
            (int(index) for index in end_candidates),
            key=lambda index: float(
                np.sum((coordinates[index] - representative_b) ** 2)
            ),
        )
        ordered_indices = _order_nonbranching_chain(
            member_indices,
            adjacency,
            start_index,
            end_index,
        )
        if not ordered_indices:
            continue
        route_indices = np.concatenate(
            (
                np.asarray((node_representatives[node_a],), dtype=np.int32),
                np.asarray(ordered_indices, dtype=np.int32),
                np.asarray((node_representatives[node_b],), dtype=np.int32),
            )
        )
        valid_route_indices = route_indices[lifted_valid[route_indices]]
        edge_xyz = np.ascontiguousarray(
            lifted_xyz[valid_route_indices], dtype=np.float32
        )
        edge_pixels = np.ascontiguousarray(
            global_coordinates_yx[valid_route_indices, ::-1], dtype=np.float32
        )
        if len(edge_xyz) < 2:
            continue
        length_m = float(np.sum(np.linalg.norm(np.diff(edge_xyz, axis=0), axis=1)))
        if not np.isfinite(length_m) or length_m <= 1e-6:
            continue
        edges.append(
            _GraphEdge(
                edge_id=len(edges),
                node_a=node_a - 1,
                node_b=node_b - 1,
                pixels_xy=edge_pixels,
                xyz=edge_xyz,
                length_m=length_m,
            )
        )

    if not edges:
        return None, "skeleton graph has no finite 3D edges"
    graph = _SkeletonGraph(
        edges=tuple(edges),
        endpoint_nodes=tuple(node - 1 for node in endpoint_nodes),
        node_count=node_count - 1,
        branch_pixels=int(np.count_nonzero(degree > 2)),
        node_pixels_xy=np.ascontiguousarray(
            global_coordinates_yx[
                np.asarray(
                    [
                        node_representatives[node_label]
                        for node_label in range(1, node_count)
                    ],
                    dtype=np.int32,
                )
            ][:, ::-1],
            dtype=np.float32,
        ),
        branch_pixels_xy=np.ascontiguousarray(
            global_coordinates_yx[degree > 2][:, ::-1],
            dtype=np.float32,
        ),
    )
    return graph, ""


def _enumerate_graph_trails(
    graph: _SkeletonGraph,
    start_node: int,
    end_node: int,
    target_length_m: float,
    candidate_limit: int,
    state_limit: int,
    edge_limit: int,
    length_margin_m: float,
) -> list[tuple[tuple[int, bool], ...]]:
    """Enumerate several length-compatible edge-simple trails, never one shortest path."""

    adjacency: list[list[tuple[int, int, bool]]] = [
        [] for _ in range(graph.node_count)
    ]
    for edge in graph.edges:
        adjacency[edge.node_a].append((edge.edge_id, edge.node_b, True))
        if edge.node_b != edge.node_a:
            adjacency[edge.node_b].append((edge.edge_id, edge.node_a, False))
    counter = itertools.count()
    queue: list[tuple[float, int, int, float, int, tuple[tuple[int, bool], ...]]] = []
    heapq.heappush(queue, (target_length_m, next(counter), start_node, 0.0, 0, ()))
    completed: list[tuple[float, tuple[tuple[int, bool], ...]]] = []
    expanded = 0
    maximum_length = target_length_m + length_margin_m
    while queue and expanded < state_limit:
        _priority, _order, node, length_m, used_mask, trail = heapq.heappop(queue)
        expanded += 1
        if node == end_node and trail:
            completed.append((length_m, trail))
            continue
        if len(trail) >= edge_limit:
            continue
        for edge_id, next_node, forward in adjacency[node]:
            edge_bit = 1 << edge_id
            if used_mask & edge_bit:
                continue
            new_length = length_m + graph.edges[edge_id].length_m
            if new_length > maximum_length:
                continue
            new_trail = trail + ((edge_id, forward),)
            priority = abs(target_length_m - new_length)
            heapq.heappush(
                queue,
                (
                    priority,
                    next(counter),
                    next_node,
                    new_length,
                    used_mask | edge_bit,
                    new_trail,
                ),
            )
    completed.sort(key=lambda item: (abs(item[0] - target_length_m), len(item[1])))
    return [trail for _length, trail in completed[:candidate_limit]]


def _assemble_route(
    graph: _SkeletonGraph,
    trail: tuple[tuple[int, bool], ...],
    endpoints: tuple[EndpointMeasurement, EndpointMeasurement],
    sample_count: int,
) -> RouteHypothesis | None:
    xyz_parts = []
    for edge_id, forward in trail:
        edge = graph.edges[edge_id]
        xyz = edge.xyz if forward else edge.xyz[::-1]
        if xyz_parts:
            xyz = xyz[1:]
        xyz_parts.append(xyz)
    if not xyz_parts:
        return None
    raw_xyz = np.concatenate(xyz_parts, axis=0).astype(np.float32, copy=False)
    if len(raw_xyz) < 2:
        return None
    raw_xyz[0] = endpoints[0].xyz
    raw_xyz[-1] = endpoints[1].xyz
    route_xyz = _resample_polyline(raw_xyz, sample_count)
    if len(route_xyz) != sample_count:
        return None
    route_xyz[0] = endpoints[0].xyz
    route_xyz[-1] = endpoints[1].xyz
    length_m = float(np.sum(np.linalg.norm(np.diff(route_xyz, axis=0), axis=1)))
    return RouteHypothesis(
        route_xyz=np.ascontiguousarray(route_xyz, dtype=np.float32),
        length_m=length_m,
    )


def _graph_edge_observation(
    component_label: int,
    edge: _GraphEdge,
    sample_count: int,
    node_by_owner: dict[tuple[int, int], int],
) -> GraphEdgeObservation | None:
    sampled_xyz = _resample_polyline(edge.xyz, sample_count)
    sampled_pixels = _resample_polyline(edge.pixels_xy, sample_count)
    if len(sampled_xyz) != sample_count:
        return None
    length_m = float(np.sum(np.linalg.norm(np.diff(sampled_xyz, axis=0), axis=1)))
    if not np.isfinite(length_m) or length_m <= 1e-6:
        return None
    anchors = []
    for owner, node_id in sorted(node_by_owner.items()):
        if int(node_id) == int(edge.node_a):
            edge_end = 0
        elif int(node_id) == int(edge.node_b):
            edge_end = 1
        else:
            continue
        anchors.append(
            GraphEndpointAnchor(
                edge_end=edge_end,
                cable_index=int(owner[0]),
                endpoint_index=int(owner[1]),
            )
        )
    return GraphEdgeObservation(
        component_label=int(component_label),
        edge_id=int(edge.edge_id),
        xyz=np.ascontiguousarray(sampled_xyz, dtype=np.float32),
        pixels_xy=np.ascontiguousarray(sampled_pixels, dtype=np.float32),
        length_m=length_m,
        endpoint_anchors=tuple(anchors),
    )


class ObservationBuilder:
    """Build complete routes and expose every directly observed graph edge."""

    def __init__(
        self,
        config: ObservationConfig,
        cable_lengths_m: tuple[float, float] = (0.515, 0.515),
        camera_model: CameraModel | None = None,
        device: str | torch.device | None = None,
    ):
        self.config = config
        self.cable_lengths_m = tuple(float(value) for value in cable_lengths_m)
        if len(self.cable_lengths_m) != 2 or any(
            not np.isfinite(value) or value <= 0.0 for value in self.cable_lengths_m
        ):
            raise ValueError("ObservationBuilder requires two positive cable lengths")
        if camera_model is None:
            raise ValueError("ObservationBuilder requires calibrated camera intrinsics")
        if (
            not np.isfinite(camera_model.fx)
            or not np.isfinite(camera_model.fy)
            or camera_model.fx <= 0.0
            or camera_model.fy <= 0.0
            or camera_model.width <= 0
            or camera_model.height <= 0
            or camera_model.forward_sign == 0.0
            or camera_model.image_y_sign == 0.0
        ):
            raise ValueError("ObservationBuilder received invalid camera intrinsics")
        if device is None:
            raise ValueError("ObservationBuilder requires an explicit compute device")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA observation processing was requested but unavailable")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.camera_model = camera_model
        self.previous_endpoints = np.full((2, 2, 3), np.nan, dtype=np.float32)
        self._cpu_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="cable-observation",
        )

    def close(self) -> None:
        self._cpu_executor.shutdown(wait=True, cancel_futures=True)

    def _prepare_geometry(
        self,
        depth: np.ndarray,
        relevant_pixels: np.ndarray,
    ) -> tuple[_SegmentedGeometry, _GeometryProfile]:
        """Upload and unproject only pixels selected by the three PIDNet masks."""

        started = time.perf_counter()
        shape = depth.shape
        pixel_y, pixel_x = np.nonzero(relevant_pixels)
        index_image = np.full(shape, -1, dtype=np.int32)
        valid_image = np.zeros(shape, dtype=bool)
        if len(pixel_x) == 0:
            geometry = _SegmentedGeometry(
                xyz=np.empty((0, 3), dtype=np.float32),
                index_image=index_image,
                valid=valid_image,
            )
            return (
                geometry,
                _GeometryProfile(
                    wall_ms=float((time.perf_counter() - started) * 1000.0),
                    maps_cuda_ms=0.0,
                    unprojection_cuda_ms=0.0,
                    readback_wall_ms=0.0,
                    relevant_pixels=0,
                    valid_3d_points=0,
                ),
            )

        selected_depth_cpu = np.ascontiguousarray(
            depth[pixel_y, pixel_x],
            dtype=np.float32,
        )
        cuda_events = None
        if self.config.profile_stages and self.device.type == "cuda":
            cuda_events = tuple(
                torch.cuda.Event(enable_timing=True) for _ in range(3)
            )
            cuda_events[0].record()
        y_cuda = torch.as_tensor(pixel_y, device=self.device, dtype=torch.float32)
        x_cuda = torch.as_tensor(pixel_x, device=self.device, dtype=torch.float32)
        selected_depth = torch.as_tensor(
            selected_depth_cpu,
            device=self.device,
            dtype=torch.float32,
        )
        selected_valid = torch.isfinite(selected_depth) & (selected_depth > 0.0)
        if cuda_events is not None:
            cuda_events[1].record()
        point_x = (
            (x_cuda - self.camera_model.cx)
            * selected_depth
            / self.camera_model.fx
        )
        point_y = (
            (y_cuda - self.camera_model.cy)
            * selected_depth
            / (self.camera_model.image_y_sign * self.camera_model.fy)
        )
        point_z = selected_depth / self.camera_model.forward_sign
        xyz_cuda = torch.stack((point_x, point_y, point_z), dim=-1)
        xyz_cuda = xyz_cuda.masked_fill(~selected_valid[:, None], float("nan"))
        if cuda_events is not None:
            cuda_events[2].record()
        readback_started = time.perf_counter()
        xyz = np.ascontiguousarray(xyz_cuda.cpu().numpy(), dtype=np.float32)
        readback_wall_ms = float(
            (time.perf_counter() - readback_started) * 1000.0
        )
        maps_cuda_ms = unprojection_cuda_ms = 0.0
        if cuda_events is not None:
            maps_cuda_ms = float(
                cuda_events[0].elapsed_time(cuda_events[1])
            )
            unprojection_cuda_ms = float(
                cuda_events[1].elapsed_time(cuda_events[2])
            )

        compact_indices = np.arange(len(pixel_x), dtype=np.int32)
        index_image[pixel_y, pixel_x] = compact_indices
        compact_valid = np.all(np.isfinite(xyz), axis=1)
        valid_image[pixel_y[compact_valid], pixel_x[compact_valid]] = True
        geometry = _SegmentedGeometry(
            xyz=xyz,
            index_image=index_image,
            valid=valid_image,
        )
        return (
            geometry,
            _GeometryProfile(
                wall_ms=float((time.perf_counter() - started) * 1000.0),
                maps_cuda_ms=maps_cuda_ms,
                unprojection_cuda_ms=unprojection_cuda_ms,
                readback_wall_ms=readback_wall_ms,
                relevant_pixels=int(len(pixel_x)),
                valid_3d_points=int(np.count_nonzero(compact_valid)),
            ),
        )

    def build(
        self,
        masks: tuple[np.ndarray, ...],
        depth: np.ndarray,
        *,
        include_skeleton_debug: bool = False,
        include_graph_edges: bool = False,
    ) -> FrameObservation:
        started = time.perf_counter()
        if len(masks) < 3:
            empty = _empty_cable("PIDNet did not return cable and two endpoint channels")
            return FrameObservation((empty, empty), (time.perf_counter() - started) * 1000.0)
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3 and depth.shape[2] == 1:
            depth = depth[:, :, 0]
        if depth.ndim != 2:
            empty = _empty_cable(f"invalid ZED depth shape {depth.shape}")
            return FrameObservation((empty, empty), (time.perf_counter() - started) * 1000.0)
        shape = depth.shape
        if shape != (self.camera_model.height, self.camera_model.width):
            empty = _empty_cable(
                f"ZED depth shape {shape} does not match calibrated camera "
                f"{self.camera_model.height}x{self.camera_model.width}"
            )
            return FrameObservation((empty, empty), (time.perf_counter() - started) * 1000.0)
        stage_started = time.perf_counter()
        cable_mask = _mask_u8(masks[0], shape)
        endpoint_masks = (_mask_u8(masks[1], shape), _mask_u8(masks[2], shape))
        relevant_pixels = cv2.bitwise_or(cable_mask, endpoint_masks[0])
        cv2.bitwise_or(relevant_pixels, endpoint_masks[1], dst=relevant_pixels)
        masks_ms = float((time.perf_counter() - stage_started) * 1000.0)
        geometry, geometry_profile = self._prepare_geometry(
            depth,
            relevant_pixels,
        )

        stage_started = time.perf_counter()
        cable_count, cable_labels, cable_stats, _ = cv2.connectedComponentsWithStats(
            cable_mask, connectivity=8
        )
        connected_components_ms = float(
            (time.perf_counter() - stage_started) * 1000.0
        )
        stage_started = time.perf_counter()
        endpoint_slots: list[list[EndpointMeasurement | None]] = []
        endpoint_component_labels = np.zeros((2, 2), dtype=np.int32)
        for cable_index, endpoint_mask in enumerate(endpoint_masks):
            slots = self._endpoint_measurements(
                cable_index,
                endpoint_mask,
                geometry,
            )
            endpoint_slots.append(slots)
            for endpoint_index, endpoint in enumerate(slots):
                if endpoint is not None:
                    endpoint_component_labels[cable_index, endpoint_index] = (
                        _nearest_component_label(
                            cable_labels,
                            endpoint.pixel_xy,
                            self.config.endpoint_label_search_px,
                        )
                    )
        endpoints_ms = float((time.perf_counter() - stage_started) * 1000.0)

        graph_data: dict[
            int,
            tuple[_SkeletonGraph | None, dict[tuple[int, int], int], str],
        ] = {}
        component_preparation_ms = 0.0
        skeleton_ms = 0.0
        graph_ms = 0.0
        processed_component_count = 0
        skeleton_pixels = 0
        debug_skeleton_pixels: list[np.ndarray] = []
        debug_node_pixels: list[np.ndarray] = []
        debug_branch_pixels: list[np.ndarray] = []
        component_inputs: list[tuple[int, np.ndarray, int, int]] = []
        for component_label in range(1, cable_count):
            area = int(cable_stats[component_label, cv2.CC_STAT_AREA])
            if area < self.config.component_min_area_px:
                continue
            component_started = time.perf_counter()
            component_roi, (y0, x0) = _component_roi(
                cable_labels,
                cable_stats,
                component_label,
            )
            component_preparation_ms += float(
                (time.perf_counter() - component_started) * 1000.0
            )
            component_inputs.append((component_label, component_roi, y0, x0))

        processed_component_count = len(component_inputs)
        skeleton_started = time.perf_counter()
        if len(component_inputs) > 1:
            skeletons = list(
                self._cpu_executor.map(
                    _morphological_skeleton,
                    (item[1] for item in component_inputs),
                )
            )
        else:
            skeletons = [
                _morphological_skeleton(item[1]) for item in component_inputs
            ]
        skeleton_ms = float((time.perf_counter() - skeleton_started) * 1000.0)

        for (component_label, _component_roi_mask, y0, x0), skeleton in zip(
            component_inputs,
            skeletons,
        ):
            skeleton_pixels += int(np.count_nonzero(skeleton))
            if include_skeleton_debug:
                skeleton_y, skeleton_x = np.nonzero(skeleton)
                debug_skeleton_pixels.append(
                    np.column_stack(
                        (skeleton_x + x0, skeleton_y + y0)
                    ).astype(np.float32)
                )
            graph_started = time.perf_counter()
            owners: list[tuple[int, int]] = []
            anchors: list[tuple[int, int]] = []
            for cable_index in range(2):
                for endpoint_index, endpoint in enumerate(endpoint_slots[cable_index]):
                    if (
                        endpoint is None
                        or endpoint_component_labels[cable_index, endpoint_index]
                        != component_label
                    ):
                        continue
                    anchor = _nearest_pixel(
                        skeleton,
                        endpoint.pixel_xy - np.asarray((x0, y0), dtype=np.float32),
                    )
                    if anchor is not None:
                        owners.append((cable_index, endpoint_index))
                        anchors.append(anchor)
            graph, graph_reason = _build_skeleton_graph(
                skeleton,
                tuple(anchors),
                (y0, x0),
                cable_labels,
                component_label,
                geometry,
                self.config.route_depth_radius_px,
                self.config.graph_node_dilation_px,
            )
            graph_ms += float((time.perf_counter() - graph_started) * 1000.0)
            if graph is not None and include_skeleton_debug:
                debug_node_pixels.append(graph.node_pixels_xy)
                debug_branch_pixels.append(graph.branch_pixels_xy)
            node_by_owner = (
                {
                    owner: graph.endpoint_nodes[index]
                    for index, owner in enumerate(owners)
                }
                if graph is not None
                else {}
            )
            graph_data[component_label] = (
                graph,
                node_by_owner,
                graph_reason,
            )

        stage_started = time.perf_counter()
        graph_edges = []
        if include_graph_edges:
            for component_label, (
                graph,
                node_by_owner,
                _reason,
            ) in graph_data.items():
                if graph is None:
                    continue
                for edge in graph.edges:
                    observation = _graph_edge_observation(
                        component_label,
                        edge,
                        self.config.graph_edge_samples,
                        node_by_owner,
                    )
                    if observation is not None:
                        graph_edges.append(observation)
        graph_edges_ms = float(
            (time.perf_counter() - stage_started) * 1000.0
        )

        observations = []
        assembly_profiles = []
        for cable_index in range(2):
            cable_observation, assembly_profile = self._assemble_cable(
                cable_index,
                endpoint_slots[cable_index],
                endpoint_component_labels[cable_index],
                graph_data,
            )
            observations.append(cable_observation)
            assembly_profiles.append(assembly_profile)

        finished = time.perf_counter()
        processing_ms = float((finished - started) * 1000.0)
        profile = None
        if self.config.profile_stages:
            route_search_ms = float(
                sum(item.route_search_ms for item in assembly_profiles)
            )
            route_assembly_ms = float(
                sum(item.route_assembly_ms for item in assembly_profiles)
            )
            finalization_ms = float(
                sum(item.finalization_ms for item in assembly_profiles)
            )
            accounted_ms = float(
                masks_ms
                + geometry_profile.wall_ms
                + connected_components_ms
                + endpoints_ms
                + component_preparation_ms
                + skeleton_ms
                + graph_ms
                + graph_edges_ms
                + route_search_ms
                + route_assembly_ms
                + finalization_ms
            )
            valid_graphs = [
                item[0] for item in graph_data.values() if item[0] is not None
            ]
            profile = ObservationProfile(
                total_ms=processing_ms,
                masks_ms=masks_ms,
                geometry_ms=geometry_profile.wall_ms,
                maps_cuda_ms=geometry_profile.maps_cuda_ms,
                unprojection_cuda_ms=geometry_profile.unprojection_cuda_ms,
                readback_wall_ms=geometry_profile.readback_wall_ms,
                connected_components_ms=connected_components_ms,
                endpoints_ms=endpoints_ms,
                component_preparation_ms=component_preparation_ms,
                skeleton_ms=skeleton_ms,
                graph_ms=graph_ms,
                graph_edges_ms=graph_edges_ms,
                route_search_ms=route_search_ms,
                route_assembly_ms=route_assembly_ms,
                finalization_ms=finalization_ms,
                unaccounted_ms=max(0.0, processing_ms - accounted_ms),
                cable_pixels=int(
                    np.sum(cable_stats[1:, cv2.CC_STAT_AREA], dtype=np.int64)
                ),
                relevant_pixels=geometry_profile.relevant_pixels,
                valid_3d_points=geometry_profile.valid_3d_points,
                component_count=max(0, int(cable_count) - 1),
                processed_component_count=processed_component_count,
                endpoint_count=int(
                    sum(
                        endpoint is not None
                        for slots in endpoint_slots
                        for endpoint in slots
                    )
                ),
                skeleton_pixels=skeleton_pixels,
                graph_nodes=int(sum(graph.node_count for graph in valid_graphs)),
                graph_edges=int(sum(len(graph.edges) for graph in valid_graphs)),
                branch_pixels=int(sum(graph.branch_pixels for graph in valid_graphs)),
                route_candidates=int(sum(len(item.routes) for item in observations)),
                graph_edge_count=int(len(graph_edges)),
            )
        skeleton_debug = None
        if include_skeleton_debug:
            skeleton_debug = SkeletonDebug(
                pixels_xy=np.ascontiguousarray(
                    np.concatenate(debug_skeleton_pixels, axis=0)
                    if debug_skeleton_pixels
                    else np.empty((0, 2), dtype=np.float32),
                    dtype=np.float32,
                ),
                node_pixels_xy=np.ascontiguousarray(
                    np.concatenate(debug_node_pixels, axis=0)
                    if debug_node_pixels
                    else np.empty((0, 2), dtype=np.float32),
                    dtype=np.float32,
                ),
                branch_pixels_xy=np.ascontiguousarray(
                    np.concatenate(debug_branch_pixels, axis=0)
                    if debug_branch_pixels
                    else np.empty((0, 2), dtype=np.float32),
                    dtype=np.float32,
                ),
            )
        return FrameObservation(
            cables=(observations[0], observations[1]),
            processing_ms=processing_ms,
            graph_edges=tuple(graph_edges),
            profile=profile,
            skeleton_debug=skeleton_debug,
        )

    def _endpoint_measurements(
        self,
        cable_index: int,
        endpoint_mask: np.ndarray,
        geometry: _SegmentedGeometry,
    ) -> list[EndpointMeasurement | None]:
        slots: list[EndpointMeasurement | None] = [None, None]
        endpoint_u8 = np.ascontiguousarray(endpoint_mask, dtype=np.uint8)
        x_offset, y_offset, roi_width, roi_height = cv2.boundingRect(endpoint_u8)
        if roi_width == 0 or roi_height == 0:
            return slots

        # Removing empty border rows and columns preserves the row-major
        # component order while avoiding a full 1080-line component scan.
        endpoint_roi = endpoint_u8[
            y_offset : y_offset + roi_height,
            x_offset : x_offset + roi_width,
        ]
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            endpoint_roi, connectivity=8
        )
        candidates = []
        for component in range(1, count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if area < self.config.endpoint_min_area_px:
                continue
            local_x = int(stats[component, cv2.CC_STAT_LEFT])
            local_y = int(stats[component, cv2.CC_STAT_TOP])
            x = local_x + x_offset
            y = local_y + y_offset
            width = int(stats[component, cv2.CC_STAT_WIDTH])
            height = int(stats[component, cv2.CC_STAT_HEIGHT])
            label_region = np.s_[
                local_y : local_y + height,
                local_x : local_x + width,
            ]
            geometry_region = np.s_[y : y + height, x : x + width]
            selected = (
                (labels[label_region] == component)
                & geometry.valid[geometry_region]
            )
            if not np.any(selected):
                continue
            local_y, local_x = np.nonzero(selected)
            points = geometry.points_at(local_y + y, local_x + x)
            center = np.median(points, axis=0).astype(np.float32)
            candidates.append(
                EndpointMeasurement(
                    pixel_xy=np.asarray(
                        (
                            centroids[component, 0] + x_offset,
                            centroids[component, 1] + y_offset,
                        ),
                        dtype=np.float32,
                    ),
                    xyz=center,
                    area_px=area,
                )
            )
        if not candidates:
            return slots
        previous = self.previous_endpoints[cable_index]
        if np.all(np.isfinite(previous)):
            best_assignment = None
            best_key = None
            choices = (-1, *range(len(candidates)))
            for first in choices:
                for second in choices:
                    if first >= 0 and first == second:
                        continue
                    assignment = (first, second)
                    assigned = 0
                    cost = 0.0
                    valid = True
                    for endpoint_index, candidate_index in enumerate(assignment):
                        if candidate_index < 0:
                            continue
                        distance = float(
                            np.linalg.norm(
                                candidates[candidate_index].xyz - previous[endpoint_index]
                            )
                        )
                        if distance > self.config.endpoint_association_gate_m:
                            valid = False
                            break
                        assigned += 1
                        cost += distance
                    if not valid:
                        continue
                    key = (-assigned, cost)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_assignment = assignment
            if best_assignment is not None:
                for endpoint_index, candidate_index in enumerate(best_assignment):
                    if candidate_index >= 0:
                        slots[endpoint_index] = candidates[candidate_index]
            if sum(endpoint is not None for endpoint in slots) < 2 and len(candidates) >= 2:
                # With both class-specific endpoints visible, their pair is
                # observable even after motion beyond the partial-association
                # gate. Use the two strongest components and choose only their
                # ordering from the previous state.
                pair = sorted(candidates, key=lambda item: -item.area_px)[:2]
                direct = float(
                    np.linalg.norm(pair[0].xyz - previous[0])
                    + np.linalg.norm(pair[1].xyz - previous[1])
                )
                swapped = float(
                    np.linalg.norm(pair[1].xyz - previous[0])
                    + np.linalg.norm(pair[0].xyz - previous[1])
                )
                if swapped < direct:
                    pair.reverse()
                slots = [pair[0], pair[1]]
        else:
            selected = sorted(candidates, key=lambda item: -item.area_px)[:2]
            selected.sort(key=lambda item: (float(item.pixel_xy[0]), float(item.pixel_xy[1])))
            for endpoint_index, endpoint in enumerate(selected):
                slots[endpoint_index] = endpoint
        return slots

    def _assemble_cable(
        self,
        cable_index: int,
        endpoints: list[EndpointMeasurement | None],
        component_labels: np.ndarray,
        graph_data: dict[
            int,
            tuple[_SkeletonGraph | None, dict[tuple[int, int], int], str],
        ],
    ) -> tuple[CableObservation, _CableAssemblyProfile]:
        visible = np.asarray([endpoint is not None for endpoint in endpoints], dtype=bool)
        endpoints_xyz = np.full((2, 3), np.nan, dtype=np.float32)
        endpoint_pixels = np.full((2, 2), np.nan, dtype=np.float32)
        for endpoint_index, endpoint in enumerate(endpoints):
            if endpoint is None:
                continue
            endpoints_xyz[endpoint_index] = endpoint.xyz
            endpoint_pixels[endpoint_index] = endpoint.pixel_xy

        routes = []
        route_search_ms = 0.0
        route_assembly_ms = 0.0
        if visible.all() and int(component_labels[0]) == int(component_labels[1]):
            component_label = int(component_labels[0])
            data = graph_data.get(component_label)
            if component_label > 0 and data is not None and data[0] is not None:
                graph, node_by_owner, _reason = data
                assert graph is not None
                endpoint_nodes = (
                    node_by_owner.get((cable_index, 0), -1),
                    node_by_owner.get((cable_index, 1), -1),
                )
                if endpoint_nodes[0] >= 0 and endpoint_nodes[1] >= 0:
                    stage_started = time.perf_counter()
                    trails = _enumerate_graph_trails(
                        graph,
                        endpoint_nodes[0],
                        endpoint_nodes[1],
                        self.cable_lengths_m[cable_index],
                        self.config.route_candidate_limit,
                        self.config.route_search_state_limit,
                        self.config.route_edge_limit,
                        self.config.route_length_margin_m,
                    )
                    route_search_ms += float(
                        (time.perf_counter() - stage_started) * 1000.0
                    )
                    endpoint_pair = (endpoints[0], endpoints[1])
                    assert endpoint_pair[0] is not None and endpoint_pair[1] is not None
                    stage_started = time.perf_counter()
                    for trail in trails:
                        route = _assemble_route(
                            graph,
                            trail,
                            endpoint_pair,
                            self.config.route_samples,
                        )
                        if route is not None:
                            routes.append(route)
                    route_assembly_ms += float(
                        (time.perf_counter() - stage_started) * 1000.0
                    )
        stage_started = time.perf_counter()
        routes.sort(
            key=lambda route: abs(
                route.length_m - self.cable_lengths_m[cable_index]
            )
        )

        if np.all(np.isfinite(self.previous_endpoints[cable_index])):
            for endpoint_index, endpoint in enumerate(endpoints):
                if endpoint is not None:
                    self.previous_endpoints[cable_index, endpoint_index] = endpoint.xyz
        elif visible.all():
            self.previous_endpoints[cable_index] = endpoints_xyz

        if routes:
            reason = "complete endpoint-to-endpoint routes"
        elif not visible.any():
            reason = "no cable-specific endpoint evidence"
        elif not visible.all():
            reason = f"partial observation: {int(visible.sum())}/2 endpoints"
        elif int(component_labels[0]) != int(component_labels[1]):
            reason = (
                "no complete route: endpoints on different body components "
                f"{int(component_labels[0])}/{int(component_labels[1])}"
            )
        else:
            component_label = int(component_labels[0])
            data = graph_data.get(component_label)
            if component_label <= 0:
                reason = "no complete route: endpoints are not attached to cable body"
            elif data is None:
                reason = "no complete route: body component was not graphable"
            elif data[0] is None:
                reason = f"no complete route: {data[2]}"
            else:
                graph = data[0]
                node_by_owner = data[1]
                assert graph is not None
                endpoint_nodes = (
                    node_by_owner.get((cable_index, 0), -1),
                    node_by_owner.get((cable_index, 1), -1),
                )
                if endpoint_nodes[0] < 0 or endpoint_nodes[1] < 0:
                    reason = "no complete route: endpoint anchor missing from skeleton"
                else:
                    reason = "no complete route: no endpoint-to-endpoint graph trail"
        observation = CableObservation(
            valid=bool(routes or visible.any()),
            reason=reason,
            endpoints_xyz=np.ascontiguousarray(endpoints_xyz, dtype=np.float32),
            endpoint_pixels_xy=np.ascontiguousarray(endpoint_pixels, dtype=np.float32),
            endpoint_visible=np.ascontiguousarray(visible, dtype=bool),
            endpoint_component_labels=np.ascontiguousarray(
                component_labels, dtype=np.int32
            ),
            routes=tuple(routes),
        )
        finalization_ms = float((time.perf_counter() - stage_started) * 1000.0)
        return (
            observation,
            _CableAssemblyProfile(
                route_search_ms=route_search_ms,
                route_assembly_ms=route_assembly_ms,
                finalization_ms=finalization_ms,
            ),
        )
