"""Endpoint-conditioned graph observations for indistinguishable cable pixels."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import itertools
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class EndpointMeasurement:
    pixel_xy: np.ndarray
    xyz: np.ndarray
    area_px: int
    spread_m: float


@dataclass(frozen=True)
class RouteHypothesis:
    """One edge-simple trail through the complete observed skeleton graph."""

    edge_ids: tuple[int, ...]
    route_xyz: np.ndarray
    route_pixels_xy: np.ndarray
    length_m: float
    endpoint_alignment_error: float
    turn_rms_degrees: float


@dataclass(frozen=True)
class CableObservation:
    valid: bool
    reason: str
    endpoints_xyz: np.ndarray
    endpoint_pixels_xy: np.ndarray
    endpoint_tangents: np.ndarray
    routes: tuple[RouteHypothesis, ...]
    component_points: int
    graph_nodes: int
    graph_edges: int
    branch_pixels: int
    route_search_truncated: bool


@dataclass(frozen=True)
class FrameObservation:
    cables: tuple[CableObservation, CableObservation]
    processing_ms: float


@dataclass(frozen=True)
class ObservationConfig:
    endpoint_min_area_px: int = 24
    endpoint_label_search_px: int = 18
    endpoint_tangent_radius_px: int = 28
    endpoint_tangent_min_points: int = 24
    route_depth_radius_px: int = 2
    route_samples: int = 128
    graph_node_dilation_px: int = 3
    route_candidate_limit: int = 16
    route_search_state_limit: int = 4096
    route_edge_limit: int = 24
    route_length_margin_m: float = 0.18

    @classmethod
    def from_mapping(cls, values: dict | None) -> "ObservationConfig":
        values = values or {}
        return cls(
            endpoint_min_area_px=max(1, int(values.get("endpoint_min_area_px", 24))),
            endpoint_label_search_px=max(1, int(values.get("endpoint_label_search_px", 18))),
            endpoint_tangent_radius_px=max(
                3, int(values.get("endpoint_tangent_radius_px", 28))
            ),
            endpoint_tangent_min_points=max(
                3, int(values.get("endpoint_tangent_min_points", 24))
            ),
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
        )


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


def _empty_cable(reason: str) -> CableObservation:
    return CableObservation(
        valid=False,
        reason=str(reason),
        endpoints_xyz=np.empty((0, 3), dtype=np.float32),
        endpoint_pixels_xy=np.empty((0, 2), dtype=np.float32),
        endpoint_tangents=np.empty((0, 3), dtype=np.float32),
        routes=(),
        component_points=0,
        graph_nodes=0,
        graph_edges=0,
        branch_pixels=0,
        route_search_truncated=False,
    )


def _mask_bool(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(mask)
    if array.shape != shape:
        array = cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(array > 0)


def _finite_xyz(xyz: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(xyz), axis=-1)


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length <= 1e-12:
        return np.zeros(3, dtype=np.float32)
    return np.asarray(vector / length, dtype=np.float32)


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
_NEIGHBOR_CODE_KERNEL = np.asarray(
    ((128, 1, 2), (64, 0, 4), (32, 16, 8)), dtype=np.uint16
)
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
    """Zhang-Suen thinning using vectorized OpenCV neighborhood codes."""

    image = np.asarray(mask, dtype=np.uint8).copy()
    while True:
        removed = 0
        for table in _THINNING_LUTS:
            codes = cv2.filter2D(
                image,
                cv2.CV_16U,
                _NEIGHBOR_CODE_KERNEL,
                borderType=cv2.BORDER_CONSTANT,
            )
            delete = (image > 0) & (table[codes] > 0)
            count = int(np.count_nonzero(delete))
            if count:
                image[delete] = 0
                removed += count
        if removed == 0:
            break
    return image > 0


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


def _lift_route(
    route_yx: np.ndarray,
    cloud_xyz: np.ndarray,
    finite_cloud: np.ndarray,
    component: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Use local medians of actual segmented XYZ around skeleton samples."""

    height, width = component.shape
    offsets = np.arange(-radius, radius + 1, dtype=np.int32)
    offset_y, offset_x = np.meshgrid(offsets, offsets, indexing="ij")
    sample_y = route_yx[:, 0, None] + offset_y.reshape(1, -1)
    sample_x = route_yx[:, 1, None] + offset_x.reshape(1, -1)
    in_bounds = (
        (sample_y >= 0) & (sample_y < height) & (sample_x >= 0) & (sample_x < width)
    )
    safe_y = np.clip(sample_y, 0, height - 1)
    safe_x = np.clip(sample_x, 0, width - 1)
    local_xyz = cloud_xyz[safe_y, safe_x].astype(np.float32, copy=True)
    selected = in_bounds & component[safe_y, safe_x] & finite_cloud[safe_y, safe_x]
    local_xyz[~selected] = np.nan
    with np.errstate(all="ignore"):
        points = np.nanmedian(local_xyz, axis=1)
    valid = np.all(np.isfinite(points), axis=1)
    points = points[valid]
    pixels = route_yx[valid][:, ::-1]
    if len(points) < 2:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        )
    return (
        np.ascontiguousarray(points, dtype=np.float32),
        np.ascontiguousarray(pixels, dtype=np.float32),
    )


def _principal_tangent(points: np.ndarray, endpoint: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return np.zeros(3, dtype=np.float32)
    centered = points - np.mean(points, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(points) - 1)
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, int(np.argmax(values))]
    inward = np.mean(points, axis=0) - np.asarray(endpoint, dtype=np.float64)
    if float(np.dot(direction, inward)) < 0.0:
        direction = -direction
    return _normalize(direction)


def _measured_endpoint_tangent(
    endpoint: EndpointMeasurement,
    component: np.ndarray,
    cloud_xyz: np.ndarray,
    finite_cloud: np.ndarray,
    radius_px: int,
    minimum_points: int,
) -> np.ndarray:
    """Estimate inward direction from local observed cable points, independent of a route."""

    x = int(round(float(endpoint.pixel_xy[0])))
    y = int(round(float(endpoint.pixel_xy[1])))
    height, width = component.shape
    x0, x1 = max(0, x - radius_px), min(width, x + radius_px + 1)
    y0, y1 = max(0, y - radius_px), min(height, y + radius_px + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    circle = (xx - x) ** 2 + (yy - y) ** 2 <= radius_px * radius_px
    selected = component[y0:y1, x0:x1] & finite_cloud[y0:y1, x0:x1] & circle
    points = cloud_xyz[y0:y1, x0:x1][selected]
    if len(points) < minimum_points:
        return np.zeros(3, dtype=np.float32)
    return _principal_tangent(points, endpoint.xyz)


def _route_endpoint_tangents(route_xyz: np.ndarray, count: int = 12) -> np.ndarray:
    count = min(max(3, int(count)), len(route_xyz))
    return np.ascontiguousarray(
        (
            _principal_tangent(route_xyz[:count], route_xyz[0]),
            _principal_tangent(route_xyz[-count:], route_xyz[-1]),
        ),
        dtype=np.float32,
    )


def _turn_rms_degrees(route_xyz: np.ndarray) -> float:
    links = np.diff(np.asarray(route_xyz, dtype=np.float64), axis=0)
    lengths = np.linalg.norm(links, axis=1)
    valid = lengths > 1e-9
    if np.count_nonzero(valid) < 2:
        return 0.0
    unit = links[valid] / lengths[valid, None]
    cosine = np.sum(unit[:-1] * unit[1:], axis=1).clip(-1.0, 1.0)
    angles = np.arccos(cosine)
    return float(np.degrees(np.sqrt(np.mean(angles * angles))))


def _pixel_adjacency(skeleton: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    """Build a corner-pruned eight-neighbor pixel graph."""

    coordinates = np.column_stack(np.nonzero(skeleton)).astype(np.int32)
    index_by_key = {
        int(y) * skeleton.shape[1] + int(x): index
        for index, (y, x) in enumerate(coordinates)
    }
    adjacency: list[list[int]] = [[] for _ in range(len(coordinates))]
    height, width = skeleton.shape
    for index, (y_value, x_value) in enumerate(coordinates):
        y, x = int(y_value), int(x_value)
        for dy, dx in _OFFSETS:
            neighbor_y, neighbor_x = y + dy, x + dx
            if not (0 <= neighbor_y < height and 0 <= neighbor_x < width):
                continue
            neighbor = index_by_key.get(neighbor_y * width + neighbor_x)
            if neighbor is None:
                continue
            if dy != 0 and dx != 0:
                if neighbor_y * width + x in index_by_key:
                    continue
                if y * width + neighbor_x in index_by_key:
                    continue
            adjacency[index].append(neighbor)
    return coordinates, adjacency


def _order_chain(
    component_indices: set[int],
    adjacency: list[list[int]],
    start_index: int,
    end_index: int,
) -> list[int]:
    """Order pixels inside one already-compressed, non-branching graph edge."""

    queue = deque((start_index,))
    previous = {start_index: -1}
    while queue:
        current = queue.popleft()
        if current == end_index:
            break
        for neighbor in adjacency[current]:
            if neighbor not in component_indices or neighbor in previous:
                continue
            previous[neighbor] = current
            queue.append(neighbor)
    if end_index not in previous:
        return []
    path = []
    current = end_index
    while current >= 0:
        path.append(current)
        current = previous[current]
    path.reverse()
    return path


def _build_skeleton_graph(
    skeleton: np.ndarray,
    anchors_yx: tuple[tuple[int, int], ...],
    crop_origin_yx: tuple[int, int],
    component: np.ndarray,
    cloud_xyz: np.ndarray,
    finite_cloud: np.ndarray,
    depth_radius: int,
    node_dilation: int,
) -> tuple[_SkeletonGraph | None, str]:
    coordinates, adjacency = _pixel_adjacency(skeleton)
    if len(coordinates) < 2:
        return None, "skeleton contains fewer than two pixels"
    coordinate_to_index = {
        (int(y), int(x)): index for index, (y, x) in enumerate(coordinates)
    }
    anchor_indices = []
    for anchor in anchors_yx:
        index = coordinate_to_index.get((int(anchor[0]), int(anchor[1])))
        if index is None:
            return None, "endpoint anchor is absent from skeleton"
        anchor_indices.append(index)

    node_seed = np.zeros_like(skeleton, dtype=np.uint8)
    for index, neighbors in enumerate(adjacency):
        if len(neighbors) != 2:
            y, x = coordinates[index]
            node_seed[y, x] = 1
    for index in anchor_indices:
        y, x = coordinates[index]
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

    node_representatives: dict[int, int] = {}
    for node_label in range(1, node_count):
        node_pixels = np.column_stack(np.nonzero(node_labels == node_label))
        center = np.mean(node_pixels, axis=0)
        distance = np.sum((node_pixels - center[None, :]) ** 2, axis=1)
        representative_yx = tuple(node_pixels[int(np.argmin(distance))])
        node_representatives[node_label] = coordinate_to_index[representative_yx]

    endpoint_nodes = tuple(int(node_labels[coordinates[index][0], coordinates[index][1]]) for index in anchor_indices)
    if any(node <= 0 for node in endpoint_nodes):
        return None, "endpoint anchor was not assigned to a graph node"

    chain_mask = skeleton & ~node_region
    chain_count, chain_labels = cv2.connectedComponents(
        chain_mask.astype(np.uint8), connectivity=8
    )
    crop_y, crop_x = crop_origin_yx
    edges: list[_GraphEdge] = []
    for chain_label in range(1, chain_count):
        member_yx = np.column_stack(np.nonzero(chain_labels == chain_label))
        if len(member_yx) == 0:
            continue
        member_indices = {
            coordinate_to_index[(int(y), int(x))] for y, x in member_yx
        }
        contacts: dict[int, set[int]] = {}
        for member_index in member_indices:
            for neighbor in adjacency[member_index]:
                y, x = coordinates[neighbor]
                node_label = int(node_labels[y, x])
                if node_label > 0:
                    contacts.setdefault(node_label, set()).add(member_index)
        contact_nodes = sorted(contacts)
        if len(contact_nodes) > 2 or len(contact_nodes) == 0:
            continue
        if len(contact_nodes) == 2:
            node_a, node_b = contact_nodes
            start_candidates = contacts[node_a]
            end_candidates = contacts[node_b]
        else:
            node_a = node_b = contact_nodes[0]
            boundary = list(contacts[node_a])
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
            start_candidates,
            key=lambda index: float(
                np.sum((coordinates[index] - representative_a) ** 2)
            ),
        )
        end_index = min(
            end_candidates,
            key=lambda index: float(
                np.sum((coordinates[index] - representative_b) ** 2)
            ),
        )
        ordered_indices = _order_chain(
            member_indices,
            adjacency,
            start_index,
            end_index,
        )
        if not ordered_indices:
            continue
        route_local_yx = np.vstack(
            (
                representative_a,
                coordinates[ordered_indices],
                representative_b,
            )
        ).astype(np.int32)
        route_yx = route_local_yx + np.asarray((crop_y, crop_x), dtype=np.int32)
        edge_xyz, edge_pixels = _lift_route(
            route_yx,
            cloud_xyz,
            finite_cloud,
            component,
            depth_radius,
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
        branch_pixels=int(sum(len(neighbors) > 2 for neighbors in adjacency)),
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
) -> tuple[list[tuple[tuple[int, bool], ...]], bool]:
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
    return [trail for _length, trail in completed[:candidate_limit]], bool(queue)


def _assemble_route(
    graph: _SkeletonGraph,
    trail: tuple[tuple[int, bool], ...],
    endpoints: tuple[EndpointMeasurement, EndpointMeasurement],
    measured_tangents: np.ndarray,
    sample_count: int,
) -> RouteHypothesis | None:
    xyz_parts = []
    pixel_parts = []
    for edge_id, forward in trail:
        edge = graph.edges[edge_id]
        xyz = edge.xyz if forward else edge.xyz[::-1]
        pixels = edge.pixels_xy if forward else edge.pixels_xy[::-1]
        if xyz_parts:
            xyz = xyz[1:]
            pixels = pixels[1:]
        xyz_parts.append(xyz)
        pixel_parts.append(pixels)
    if not xyz_parts:
        return None
    raw_xyz = np.concatenate(xyz_parts, axis=0).astype(np.float32, copy=False)
    raw_pixels = np.concatenate(pixel_parts, axis=0).astype(np.float32, copy=False)
    if len(raw_xyz) < 2:
        return None
    raw_xyz[0] = endpoints[0].xyz
    raw_xyz[-1] = endpoints[1].xyz
    route_xyz = _resample_polyline(raw_xyz, sample_count)
    route_pixels = _resample_polyline(raw_pixels, sample_count)
    if len(route_xyz) != sample_count:
        return None
    route_xyz[0] = endpoints[0].xyz
    route_xyz[-1] = endpoints[1].xyz
    length_m = float(np.sum(np.linalg.norm(np.diff(route_xyz, axis=0), axis=1)))
    route_tangents = _route_endpoint_tangents(route_xyz)
    alignment_terms = []
    for endpoint_index in range(2):
        measured = measured_tangents[endpoint_index]
        if float(np.linalg.norm(measured)) <= 1e-6:
            continue
        alignment_terms.append(
            1.0
            - float(
                np.clip(
                    np.dot(measured, route_tangents[endpoint_index]),
                    -1.0,
                    1.0,
                )
            )
        )
    alignment_error = float(np.mean(alignment_terms)) if alignment_terms else 0.0
    return RouteHypothesis(
        edge_ids=tuple(edge_id for edge_id, _forward in trail),
        route_xyz=np.ascontiguousarray(route_xyz, dtype=np.float32),
        route_pixels_xy=np.ascontiguousarray(route_pixels, dtype=np.float32),
        length_m=length_m,
        endpoint_alignment_error=alignment_error,
        turn_rms_degrees=_turn_rms_degrees(route_xyz),
    )


class ObservationBuilder:
    """Preserve all useful graph branches for two endpoint-identified cables."""

    def __init__(
        self,
        config: ObservationConfig,
        cable_lengths_m: tuple[float, float] = (0.515, 0.515),
    ):
        self.config = config
        self.cable_lengths_m = tuple(float(value) for value in cable_lengths_m)
        if len(self.cable_lengths_m) != 2 or any(
            not np.isfinite(value) or value <= 0.0 for value in self.cable_lengths_m
        ):
            raise ValueError("ObservationBuilder requires two positive cable lengths")
        self.previous_endpoints: list[np.ndarray | None] = [None, None]

    def build(self, masks: tuple[np.ndarray, ...], cloud: np.ndarray) -> FrameObservation:
        started = time.perf_counter()
        if len(masks) < 3:
            empty = _empty_cable("PIDNet did not return cable and two endpoint channels")
            return FrameObservation((empty, empty), (time.perf_counter() - started) * 1000.0)
        cloud = np.asarray(cloud)
        if cloud.ndim != 3 or cloud.shape[2] < 3:
            empty = _empty_cable(f"invalid ZED cloud shape {cloud.shape}")
            return FrameObservation((empty, empty), (time.perf_counter() - started) * 1000.0)
        shape = cloud.shape[:2]
        cloud_xyz = np.asarray(cloud[:, :, :3], dtype=np.float32)
        cable_mask = _mask_bool(masks[0], shape)
        endpoint_masks = (_mask_bool(masks[1], shape), _mask_bool(masks[2], shape))
        relevant_pixels = cable_mask | endpoint_masks[0] | endpoint_masks[1]
        finite_cloud = np.zeros(shape, dtype=bool)
        finite_cloud[relevant_pixels] = _finite_xyz(cloud_xyz[relevant_pixels])
        _, cable_labels, _, _ = cv2.connectedComponentsWithStats(
            cable_mask.astype(np.uint8), connectivity=8
        )
        prepared: list[tuple[list[EndpointMeasurement], int, str]] = []
        for cable_index, endpoint_mask in enumerate(endpoint_masks):
            endpoints, reason = self._endpoint_measurements(
                cable_index,
                endpoint_mask,
                cloud_xyz,
                finite_cloud,
            )
            component_label = 0
            if not reason:
                labels = [
                    _nearest_component_label(
                        cable_labels,
                        endpoint.pixel_xy,
                        self.config.endpoint_label_search_px,
                    )
                    for endpoint in endpoints
                ]
                if labels[0] == 0 or labels[1] == 0:
                    reason = f"cable {cable_index + 1}: endpoint is not on cable mask"
                elif labels[0] != labels[1]:
                    reason = (
                        f"cable {cable_index + 1}: endpoint pair lies in "
                        "different components"
                    )
                else:
                    component_label = labels[0]
            prepared.append((endpoints, component_label, reason))

        graph_data: dict[
            int,
            tuple[np.ndarray, _SkeletonGraph | None, dict[tuple[int, int], int], str],
        ] = {}
        component_labels = sorted(
            {label for _endpoints, label, reason in prepared if not reason and label > 0}
        )
        for component_label in component_labels:
            component = cable_labels == component_label
            component_y, component_x = np.nonzero(component)
            if len(component_x) == 0:
                graph_data[component_label] = (
                    component,
                    None,
                    {},
                    "empty cable component",
                )
                continue
            x0, x1 = max(0, int(component_x.min()) - 2), min(
                component.shape[1], int(component_x.max()) + 3
            )
            y0, y1 = max(0, int(component_y.min()) - 2), min(
                component.shape[0], int(component_y.max()) + 3
            )
            skeleton = _morphological_skeleton(component[y0:y1, x0:x1])
            owners: list[tuple[int, int]] = []
            anchors: list[tuple[int, int]] = []
            for cable_index, (endpoints, label, reason) in enumerate(prepared):
                if reason or label != component_label:
                    continue
                for endpoint_index, endpoint in enumerate(endpoints):
                    anchor = _nearest_pixel(
                        skeleton,
                        endpoint.pixel_xy - np.asarray((x0, y0), dtype=np.float32),
                    )
                    if anchor is not None:
                        owners.append((cable_index, endpoint_index))
                        anchors.append(anchor)
            if len(anchors) < 2:
                graph_data[component_label] = (
                    component,
                    None,
                    {},
                    "skeleton has insufficient endpoint anchors",
                )
                continue
            graph, graph_reason = _build_skeleton_graph(
                skeleton,
                tuple(anchors),
                (y0, x0),
                component,
                cloud_xyz,
                finite_cloud,
                self.config.route_depth_radius_px,
                self.config.graph_node_dilation_px,
            )
            node_by_owner = (
                {
                    owner: graph.endpoint_nodes[index]
                    for index, owner in enumerate(owners)
                }
                if graph is not None
                else {}
            )
            graph_data[component_label] = (
                component,
                graph,
                node_by_owner,
                graph_reason,
            )

        observations = []
        for cable_index, (endpoints, component_label, reason) in enumerate(prepared):
            if reason:
                observations.append(_empty_cable(reason))
                continue
            component, graph, node_by_owner, graph_reason = graph_data[component_label]
            if graph is None:
                observations.append(
                    _empty_cable(f"cable {cable_index + 1}: {graph_reason}")
                )
                continue
            endpoint_nodes = (
                node_by_owner.get((cable_index, 0), -1),
                node_by_owner.get((cable_index, 1), -1),
            )
            if endpoint_nodes[0] < 0 or endpoint_nodes[1] < 0:
                observations.append(
                    _empty_cable(
                        f"cable {cable_index + 1}: graph did not retain both endpoints"
                    )
                )
                continue
            observations.append(
                self._finish_cable(
                    cable_index,
                    endpoints,
                    component,
                    graph,
                    endpoint_nodes,
                    cloud_xyz,
                    finite_cloud,
                )
            )
        return FrameObservation(
            cables=(observations[0], observations[1]),
            processing_ms=float((time.perf_counter() - started) * 1000.0),
        )

    def _endpoint_measurements(
        self,
        cable_index: int,
        endpoint_mask: np.ndarray,
        cloud_xyz: np.ndarray,
        finite_cloud: np.ndarray,
    ) -> tuple[list[EndpointMeasurement], str]:
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            endpoint_mask.astype(np.uint8), connectivity=8
        )
        valid_ids = [
            component
            for component in range(1, count)
            if int(stats[component, cv2.CC_STAT_AREA]) >= self.config.endpoint_min_area_px
        ]
        if len(valid_ids) != 2:
            return [], (
                f"endpoint class {cable_index + 1}: expected 2 components, "
                f"found {len(valid_ids)}"
            )
        measurements = []
        for component in valid_ids:
            x = int(stats[component, cv2.CC_STAT_LEFT])
            y = int(stats[component, cv2.CC_STAT_TOP])
            width = int(stats[component, cv2.CC_STAT_WIDTH])
            height = int(stats[component, cv2.CC_STAT_HEIGHT])
            region = np.s_[y : y + height, x : x + width]
            selected = (labels[region] == component) & finite_cloud[region]
            if not np.any(selected):
                return [], f"endpoint class {cable_index + 1}: component has no finite depth"
            points = cloud_xyz[region][selected]
            center = np.median(points, axis=0).astype(np.float32)
            spread = float(np.median(np.linalg.norm(points - center, axis=1)))
            measurements.append(
                EndpointMeasurement(
                    pixel_xy=np.asarray(centroids[component], dtype=np.float32),
                    xyz=center,
                    area_px=int(stats[component, cv2.CC_STAT_AREA]),
                    spread_m=spread,
                )
            )
        previous = self.previous_endpoints[cable_index]
        if previous is None:
            measurements.sort(
                key=lambda item: (float(item.pixel_xy[0]), float(item.pixel_xy[1]))
            )
        else:
            direct = float(
                np.linalg.norm(measurements[0].xyz - previous[0])
                + np.linalg.norm(measurements[1].xyz - previous[1])
            )
            swapped = float(
                np.linalg.norm(measurements[1].xyz - previous[0])
                + np.linalg.norm(measurements[0].xyz - previous[1])
            )
            if swapped < direct:
                measurements.reverse()
        return measurements, ""

    def _finish_cable(
        self,
        cable_index: int,
        endpoints: list[EndpointMeasurement],
        component: np.ndarray,
        graph: _SkeletonGraph,
        endpoint_nodes: tuple[int, int],
        cloud_xyz: np.ndarray,
        finite_cloud: np.ndarray,
    ) -> CableObservation:
        measured_tangents = np.stack(
            [
                _measured_endpoint_tangent(
                    endpoint,
                    component,
                    cloud_xyz,
                    finite_cloud,
                    self.config.endpoint_tangent_radius_px,
                    self.config.endpoint_tangent_min_points,
                )
                for endpoint in endpoints
            ]
        ).astype(np.float32)
        trails, truncated = _enumerate_graph_trails(
            graph,
            endpoint_nodes[0],
            endpoint_nodes[1],
            self.cable_lengths_m[cable_index],
            self.config.route_candidate_limit,
            self.config.route_search_state_limit,
            self.config.route_edge_limit,
            self.config.route_length_margin_m,
        )
        routes = []
        endpoint_pair = (endpoints[0], endpoints[1])
        for trail in trails:
            route = _assemble_route(
                graph,
                trail,
                endpoint_pair,
                measured_tangents,
                self.config.route_samples,
            )
            if route is not None:
                routes.append(route)
        routes.sort(
            key=lambda route: (
                abs(route.length_m - self.cable_lengths_m[cable_index]),
                route.endpoint_alignment_error,
            )
        )
        if not routes:
            return _empty_cable(
                f"cable {cable_index + 1}: graph has no length-bounded endpoint trail"
            )
        self.previous_endpoints[cable_index] = np.stack(
            [endpoint.xyz for endpoint in endpoints]
        )
        return CableObservation(
            valid=True,
            reason="ok",
            endpoints_xyz=np.ascontiguousarray(
                [endpoint.xyz for endpoint in endpoints], dtype=np.float32
            ),
            endpoint_pixels_xy=np.ascontiguousarray(
                [endpoint.pixel_xy for endpoint in endpoints], dtype=np.float32
            ),
            endpoint_tangents=np.ascontiguousarray(measured_tangents, dtype=np.float32),
            routes=tuple(routes),
            component_points=int(np.count_nonzero(component)),
            graph_nodes=graph.node_count,
            graph_edges=len(graph.edges),
            branch_pixels=graph.branch_pixels,
            route_search_truncated=truncated,
        )
