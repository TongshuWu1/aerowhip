"""Synchronized graph and metric observations for the improved cable PF.

The builder deliberately stops at evidence.  It extracts cable-body topology,
endpoint observations, visible graph edges, complete route hypotheses, and
degree-four crossing alternatives from one immutable PIDNet/RGB-D frame.  It
does not infer hidden cable geometry, run a particle filter, or construct a
surface mesh.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import heapq
import itertools
import math
import time

import cv2
import numpy as np

from .config import ObservationSettings
from .frames import CameraCalibration, FrameKey
from .perception import PerceptionFrame


_PIXEL_NEIGHBOR_OFFSETS = (
    (-1, 0),
    (0, 1),
    (1, 0),
    (0, -1),
    (-1, -1),
    (-1, 1),
    (1, 1),
    (1, -1),
)


def _owned_readonly(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    owned = np.array(value, dtype=dtype, order="C", copy=True)
    owned.setflags(write=False)
    return owned


def _require_shape(name: str, value: np.ndarray, shape: tuple[int, ...]) -> None:
    if not isinstance(value, np.ndarray) or value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")


def _require_sample_arrays(
    pixels: np.ndarray,
    xyz: np.ndarray,
    valid: np.ndarray,
    covariance: np.ndarray,
) -> None:
    if pixels.dtype != np.float32 or pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("sample pixels must be float32 Nx2")
    count = pixels.shape[0]
    if xyz.dtype != np.float32 or xyz.shape != (count, 3):
        raise ValueError("sample xyz must be float32 Nx3")
    if valid.dtype != np.bool_ or valid.shape != (count,):
        raise ValueError("sample validity must be bool N")
    if covariance.dtype != np.float32 or covariance.shape != (count, 3, 3):
        raise ValueError("sample covariance must be float32 Nx3x3")
    if not np.isfinite(pixels).all():
        raise ValueError("sample pixels must be finite")
    if valid.any():
        if not np.isfinite(xyz[valid]).all() or not np.isfinite(covariance[valid]).all():
            raise ValueError("valid 3D samples and covariances must be finite")
        if not np.allclose(
            covariance[valid],
            np.swapaxes(covariance[valid], -1, -2),
            rtol=1.0e-5,
            atol=1.0e-8,
        ):
            raise ValueError("valid sample covariance must be symmetric")
    if np.isfinite(xyz[~valid]).any() or np.isfinite(covariance[~valid]).any():
        raise ValueError("invalid samples must contain NaN xyz and covariance")


@dataclass(frozen=True, slots=True)
class EndpointObservation:
    """One visible endpoint component; ``endpoint_id=-1`` means unresolved end."""

    cable_id: int
    endpoint_id: int
    pixel_xy_f32: np.ndarray
    xyz_camera_m_f32: np.ndarray
    covariance_m2_f32: np.ndarray
    depth_valid: bool
    area_px: int
    body_component_label: int
    association_confidence: float

    def __post_init__(self) -> None:
        if self.cable_id not in (0, 1):
            raise ValueError("cable_id must be 0 or 1")
        if self.endpoint_id not in (-1, 0, 1):
            raise ValueError("endpoint_id must be -1, 0, or 1")
        _require_shape("pixel_xy_f32", self.pixel_xy_f32, (2,))
        _require_shape("xyz_camera_m_f32", self.xyz_camera_m_f32, (3,))
        _require_shape("covariance_m2_f32", self.covariance_m2_f32, (3, 3))
        if self.pixel_xy_f32.dtype != np.float32 or not np.isfinite(
            self.pixel_xy_f32
        ).all():
            raise ValueError("endpoint pixel must be finite float32")
        if self.depth_valid:
            if (
                self.xyz_camera_m_f32.dtype != np.float32
                or self.covariance_m2_f32.dtype != np.float32
                or not np.isfinite(self.xyz_camera_m_f32).all()
                or not np.isfinite(self.covariance_m2_f32).all()
            ):
                raise ValueError("valid endpoint depth requires finite float32 geometry")
        elif np.isfinite(self.xyz_camera_m_f32).any() or np.isfinite(
            self.covariance_m2_f32
        ).any():
            raise ValueError("invalid endpoint depth must contain NaN geometry")
        if self.area_px <= 0 or self.body_component_label < 0:
            raise ValueError("endpoint area/component metadata is invalid")
        if not 0.0 <= self.association_confidence <= 1.0:
            raise ValueError("association_confidence must be in [0, 1]")
        object.__setattr__(
            self, "pixel_xy_f32", _owned_readonly(self.pixel_xy_f32, np.float32)
        )
        object.__setattr__(
            self,
            "xyz_camera_m_f32",
            _owned_readonly(self.xyz_camera_m_f32, np.float32),
        )
        object.__setattr__(
            self,
            "covariance_m2_f32",
            _owned_readonly(self.covariance_m2_f32, np.float32),
        )


@dataclass(frozen=True, slots=True)
class GraphEndpointAnchor:
    edge_end: int
    cable_id: int
    endpoint_id: int

    def __post_init__(self) -> None:
        if self.edge_end not in (0, 1):
            raise ValueError("edge_end must be 0 or 1")
        if self.cable_id not in (0, 1) or self.endpoint_id not in (0, 1):
            raise ValueError("endpoint anchor identity is invalid")


@dataclass(frozen=True, slots=True)
class GraphEdgeObservation:
    """One ordered visible skeleton edge with explicit missing depth samples."""

    component_label: int
    edge_id: int
    node_a: int
    node_b: int
    pixels_xy_f32: np.ndarray
    xyz_camera_m_f32: np.ndarray
    depth_valid_bool: np.ndarray
    covariance_m2_f32: np.ndarray
    observed_length_m: float
    estimated_length_m: float | None
    endpoint_anchors: tuple[GraphEndpointAnchor, ...] = ()

    def __post_init__(self) -> None:
        if self.component_label <= 0 or min(self.edge_id, self.node_a, self.node_b) < 0:
            raise ValueError("graph edge identifiers must be nonnegative")
        _require_sample_arrays(
            self.pixels_xy_f32,
            self.xyz_camera_m_f32,
            self.depth_valid_bool,
            self.covariance_m2_f32,
        )
        if self.pixels_xy_f32.shape[0] < 2:
            raise ValueError("graph edge requires at least two samples")
        if not math.isfinite(self.observed_length_m) or self.observed_length_m < 0.0:
            raise ValueError("observed graph-edge length is invalid")
        if self.estimated_length_m is not None and (
            not math.isfinite(self.estimated_length_m)
            or self.estimated_length_m <= 0.0
        ):
            raise ValueError("estimated graph-edge length is invalid")
        object.__setattr__(
            self, "pixels_xy_f32", _owned_readonly(self.pixels_xy_f32, np.float32)
        )
        object.__setattr__(
            self,
            "xyz_camera_m_f32",
            _owned_readonly(self.xyz_camera_m_f32, np.float32),
        )
        object.__setattr__(
            self,
            "depth_valid_bool",
            _owned_readonly(self.depth_valid_bool, np.bool_),
        )
        object.__setattr__(
            self,
            "covariance_m2_f32",
            _owned_readonly(self.covariance_m2_f32, np.float32),
        )


@dataclass(frozen=True, slots=True)
class CrossingPairingObservation:
    """One perfect matching of the four independently lifted crossing arms."""

    rank: int
    edge_pairs_i32: np.ndarray
    tangent_cost: float

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError("crossing pairing rank must be nonnegative")
        if self.edge_pairs_i32.dtype != np.int32 or self.edge_pairs_i32.shape != (2, 2):
            raise ValueError("crossing edge pairs must be int32 2x2")
        if int(self.edge_pairs_i32.min()) < 0:
            raise ValueError("crossing edge identifiers must be nonnegative")
        if not math.isfinite(self.tangent_cost) or self.tangent_cost < 0.0:
            raise ValueError("crossing tangent cost must be finite and nonnegative")
        object.__setattr__(
            self,
            "edge_pairs_i32",
            _owned_readonly(self.edge_pairs_i32, np.int32),
        )


@dataclass(frozen=True, slots=True)
class CrossingObservation:
    """A degree-four image junction; central mixed depth is never asserted as 3D."""

    component_label: int
    node_id: int
    pixel_xy_f32: np.ndarray
    incident_edge_ids_i32: np.ndarray
    arm_xyz_camera_m_f32: np.ndarray
    arm_depth_valid_bool: np.ndarray
    arm_covariance_m2_f32: np.ndarray
    pairings: tuple[CrossingPairingObservation, ...]

    def __post_init__(self) -> None:
        if self.component_label <= 0 or self.node_id < 0:
            raise ValueError("crossing identifiers are invalid")
        if self.pixel_xy_f32.dtype != np.float32 or self.pixel_xy_f32.shape != (2,):
            raise ValueError("crossing pixel must be float32 length 2")
        if self.incident_edge_ids_i32.dtype != np.int32 or self.incident_edge_ids_i32.shape != (4,):
            raise ValueError("crossing must contain four incident edge ids")
        if self.arm_xyz_camera_m_f32.dtype != np.float32 or self.arm_xyz_camera_m_f32.shape != (4, 3):
            raise ValueError("crossing arm geometry must be float32 4x3")
        if self.arm_depth_valid_bool.dtype != np.bool_ or self.arm_depth_valid_bool.shape != (4,):
            raise ValueError("crossing arm validity must be bool length 4")
        if (
            self.arm_covariance_m2_f32.dtype != np.float32
            or self.arm_covariance_m2_f32.shape != (4, 3, 3)
        ):
            raise ValueError("crossing arm covariance must be float32 4x3x3")
        if self.arm_depth_valid_bool.any() and not np.isfinite(
            self.arm_xyz_camera_m_f32[self.arm_depth_valid_bool]
        ).all():
            raise ValueError("valid crossing arms must be finite")
        if self.arm_depth_valid_bool.any() and not np.isfinite(
            self.arm_covariance_m2_f32[self.arm_depth_valid_bool]
        ).all():
            raise ValueError("valid crossing covariance must be finite")
        if np.isfinite(self.arm_xyz_camera_m_f32[~self.arm_depth_valid_bool]).any():
            raise ValueError("invalid crossing arms must contain NaN")
        if np.isfinite(
            self.arm_covariance_m2_f32[~self.arm_depth_valid_bool]
        ).any():
            raise ValueError("invalid crossing covariance must contain NaN")
        if tuple(pairing.rank for pairing in self.pairings) != (0, 1, 2):
            raise ValueError("crossing must retain all three ranked pairings")
        incident_edges = set(int(value) for value in self.incident_edge_ids_i32)
        if len(incident_edges) != 4:
            raise ValueError("crossing incident edge ids must be unique")
        for pairing in self.pairings:
            if set(int(value) for value in pairing.edge_pairs_i32.ravel()) != incident_edges:
                raise ValueError("each crossing pairing must match all four incident edges")
        object.__setattr__(
            self, "pixel_xy_f32", _owned_readonly(self.pixel_xy_f32, np.float32)
        )
        object.__setattr__(
            self,
            "incident_edge_ids_i32",
            _owned_readonly(self.incident_edge_ids_i32, np.int32),
        )
        object.__setattr__(
            self,
            "arm_xyz_camera_m_f32",
            _owned_readonly(self.arm_xyz_camera_m_f32, np.float32),
        )
        object.__setattr__(
            self,
            "arm_depth_valid_bool",
            _owned_readonly(self.arm_depth_valid_bool, np.bool_),
        )
        object.__setattr__(
            self,
            "arm_covariance_m2_f32",
            _owned_readonly(self.arm_covariance_m2_f32, np.float32),
        )


@dataclass(frozen=True, slots=True)
class RouteHypothesis:
    """One complete edge-simple endpoint route without hidden-depth invention."""

    cable_id: int
    component_label: int
    edge_ids_i32: np.ndarray
    edge_forward_bool: np.ndarray
    pixels_xy_f32: np.ndarray
    xyz_camera_m_f32: np.ndarray
    depth_valid_bool: np.ndarray
    covariance_m2_f32: np.ndarray
    observed_length_m: float
    estimated_length_m: float
    target_length_m: float
    tangent_cost: float
    score: float

    def __post_init__(self) -> None:
        if self.cable_id not in (0, 1) or self.component_label <= 0:
            raise ValueError("route identity is invalid")
        if self.edge_ids_i32.dtype != np.int32 or self.edge_ids_i32.ndim != 1:
            raise ValueError("route edge ids must be an int32 vector")
        if self.edge_ids_i32.size and int(self.edge_ids_i32.min()) < 0:
            raise ValueError("route edge ids must be nonnegative")
        if self.edge_forward_bool.dtype != np.bool_ or self.edge_forward_bool.shape != self.edge_ids_i32.shape:
            raise ValueError("route direction vector must match edge ids")
        if self.edge_ids_i32.size == 0:
            raise ValueError("route requires at least one edge")
        _require_sample_arrays(
            self.pixels_xy_f32,
            self.xyz_camera_m_f32,
            self.depth_valid_bool,
            self.covariance_m2_f32,
        )
        for name, value, positive in (
            ("observed_length_m", self.observed_length_m, False),
            ("estimated_length_m", self.estimated_length_m, True),
            ("target_length_m", self.target_length_m, True),
            ("tangent_cost", self.tangent_cost, False),
            ("score", self.score, False),
        ):
            if not math.isfinite(value) or value < 0.0 or (positive and value <= 0.0):
                raise ValueError(f"route {name} is invalid")
        object.__setattr__(self, "edge_ids_i32", _owned_readonly(self.edge_ids_i32, np.int32))
        object.__setattr__(self, "edge_forward_bool", _owned_readonly(self.edge_forward_bool, np.bool_))
        object.__setattr__(self, "pixels_xy_f32", _owned_readonly(self.pixels_xy_f32, np.float32))
        object.__setattr__(self, "xyz_camera_m_f32", _owned_readonly(self.xyz_camera_m_f32, np.float32))
        object.__setattr__(self, "depth_valid_bool", _owned_readonly(self.depth_valid_bool, np.bool_))
        object.__setattr__(self, "covariance_m2_f32", _owned_readonly(self.covariance_m2_f32, np.float32))


@dataclass(frozen=True, slots=True)
class CableObservationTiming:
    total_ms: float
    endpoints_ms: float
    components_ms: float
    skeleton_ms: float
    graph_ms: float
    lifting_ms: float
    routes_ms: float
    skeleton_pixels: int
    graph_nodes: int
    graph_edges: int
    crossing_count: int
    route_count: int
    depth_valid_samples: int

    def __post_init__(self) -> None:
        for name in (
            "total_ms",
            "endpoints_ms",
            "components_ms",
            "skeleton_ms",
            "graph_ms",
            "lifting_ms",
            "routes_ms",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in (
            "skeleton_pixels",
            "graph_nodes",
            "graph_edges",
            "crossing_count",
            "route_count",
            "depth_valid_samples",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True, slots=True)
class CableObservationFrame:
    """All cable evidence associated one-to-one with a synchronized frame."""

    key: FrameKey
    calibration: CameraCalibration
    endpoints: tuple[EndpointObservation, ...]
    graph_edges: tuple[GraphEdgeObservation, ...]
    crossings: tuple[CrossingObservation, ...]
    routes_by_cable: tuple[tuple[RouteHypothesis, ...], tuple[RouteHypothesis, ...]]
    cable_reasons: tuple[str, str]
    body_component_count: int
    timing: CableObservationTiming

    def __post_init__(self) -> None:
        if not isinstance(self.key, FrameKey) or not isinstance(self.calibration, CameraCalibration):
            raise TypeError("observation requires a frame key and calibration")
        if self.calibration.coordinate_system != "RIGHT_HANDED_Y_UP" or self.calibration.depth_unit != "METER":
            raise ValueError("observation geometry requires metric RIGHT_HANDED_Y_UP calibration")
        if len(self.routes_by_cable) != 2 or len(self.cable_reasons) != 2:
            raise ValueError("observation requires exactly two cable route groups")
        if any(not isinstance(value, EndpointObservation) for value in self.endpoints):
            raise TypeError("endpoints must contain EndpointObservation values")
        if any(not isinstance(value, GraphEdgeObservation) for value in self.graph_edges):
            raise TypeError("graph_edges must contain GraphEdgeObservation values")
        if any(not isinstance(value, CrossingObservation) for value in self.crossings):
            raise TypeError("crossings must contain CrossingObservation values")
        if any(route.cable_id != cable_id for cable_id, routes in enumerate(self.routes_by_cable) for route in routes):
            raise ValueError("route group contains the wrong cable identity")
        if tuple(edge.edge_id for edge in self.graph_edges) != tuple(range(len(self.graph_edges))):
            raise ValueError("graph edge ids must be dense and ordered")
        if self.body_component_count < 0:
            raise ValueError("body_component_count must be nonnegative")


@dataclass(frozen=True, slots=True)
class _EndpointCandidate:
    pixel_xy: np.ndarray
    xyz: np.ndarray
    covariance: np.ndarray
    depth_valid: bool
    area_px: int
    body_component_label: int


@dataclass(frozen=True, slots=True)
class _RawEdge:
    edge_id: int
    node_a: int
    node_b: int
    pixels_xy: np.ndarray
    metric_length: float | None = None


@dataclass(frozen=True, slots=True)
class _SkeletonGraph:
    edges: tuple[_RawEdge, ...]
    endpoint_nodes: tuple[int, ...]
    node_count: int
    node_pixels_xy: np.ndarray


@dataclass(frozen=True, slots=True)
class _ComponentGraph:
    component_label: int
    graph: _SkeletonGraph
    node_by_owner: dict[tuple[int, int], int]
    global_edge_ids: tuple[int, ...]
    observed_edges: tuple[GraphEdgeObservation, ...]
    pixel_scale_m: float | None


def _resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    source = np.asarray(points, dtype=np.float64)
    if source.ndim != 2 or len(source) < 2:
        dimensions = source.shape[1] if source.ndim == 2 else 0
        return np.empty((0, dimensions), dtype=np.float32)
    lengths = np.linalg.norm(np.diff(source, axis=0), axis=1)
    keep = np.concatenate(([True], lengths > 1.0e-9))
    source = source[keep]
    if len(source) < 2:
        return np.repeat(source[:1], count, axis=0).astype(np.float32)
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(source, axis=0), axis=1))))
    total = float(cumulative[-1])
    if not math.isfinite(total) or total <= 1.0e-9:
        return np.repeat(source[:1], count, axis=0).astype(np.float32)
    targets = np.linspace(0.0, total, int(count), dtype=np.float64)
    output = np.empty((count, source.shape[1]), dtype=np.float64)
    for axis in range(source.shape[1]):
        output[:, axis] = np.interp(targets, cumulative, source[:, axis])
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


def _morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    source = np.asarray(mask, dtype=np.uint8)
    if source.ndim != 2:
        raise ValueError("skeleton mask must be two-dimensional")
    if not np.any(source):
        return np.zeros_like(source, dtype=bool)
    image = np.pad(source > 0, 1, mode="constant").astype(np.uint8, copy=False)
    flat = image.reshape(-1)
    stride = image.shape[1]
    active = np.flatnonzero(flat)
    neighbor_offsets = np.asarray(
        (-stride, -stride + 1, 1, stride + 1, stride, stride - 1, -1, -stride - 1),
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


def _component_roi(
    labels: np.ndarray,
    stats: np.ndarray,
    component_label: int,
    padding: int = 2,
) -> tuple[np.ndarray, tuple[int, int]]:
    left = int(stats[component_label, cv2.CC_STAT_LEFT])
    top = int(stats[component_label, cv2.CC_STAT_TOP])
    width = int(stats[component_label, cv2.CC_STAT_WIDTH])
    height = int(stats[component_label, cv2.CC_STAT_HEIGHT])
    x0 = max(0, left - padding)
    x1 = min(labels.shape[1], left + width + padding)
    y0 = max(0, top - padding)
    y1 = min(labels.shape[0], top + height + padding)
    return labels[y0:y1, x0:x1] == component_label, (y0, x0)


def _nearest_pixel(mask: np.ndarray, pixel_xy: np.ndarray) -> tuple[int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    distance = (
        (xs.astype(np.float64) - float(pixel_xy[0])) ** 2
        + (ys.astype(np.float64) - float(pixel_xy[1])) ** 2
    )
    index = int(np.argmin(distance))
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
    distance = (
        ((xs + x0).astype(np.float64) - float(pixel_xy[0])) ** 2
        + ((ys + y0).astype(np.float64) - float(pixel_xy[1])) ** 2
    )
    index = int(np.argmin(distance))
    return int(window[ys[index], xs[index]])


def _pixel_adjacency(
    skeleton: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.column_stack(np.nonzero(skeleton)).astype(np.int32)
    height, width = skeleton.shape
    index_image = np.full((height, width), -1, dtype=np.int32)
    if len(coordinates) == 0:
        return coordinates, np.empty((0, 8), dtype=np.int32), index_image
    index_image[coordinates[:, 0], coordinates[:, 1]] = np.arange(len(coordinates), dtype=np.int32)
    adjacency = np.full((len(coordinates), 8), -1, dtype=np.int32)
    y = coordinates[:, 0]
    x = coordinates[:, 1]
    for slot, (dy, dx) in enumerate(_PIXEL_NEIGHBOR_OFFSETS):
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
    if start_index == end_index:
        return [start_index]
    membership = np.zeros(len(adjacency), dtype=bool)
    membership[component_indices] = True
    if not membership[start_index] or not membership[end_index]:
        return []
    neighbors = adjacency[component_indices]
    inside = (neighbors >= 0) & membership[np.maximum(neighbors, 0)]
    if np.any(np.sum(inside, axis=1) > 2):
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
                if candidate >= 0 and candidate != previous and membership[candidate]:
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
    node_dilation: int,
) -> _SkeletonGraph | None:
    coordinates, adjacency, index_image = _pixel_adjacency(skeleton)
    if len(coordinates) < 2:
        return None
    anchor_indices = []
    for anchor in anchors_yx:
        y, x = int(anchor[0]), int(anchor[1])
        if not (0 <= y < skeleton.shape[0] and 0 <= x < skeleton.shape[1]):
            continue
        index = int(index_image[y, x])
        if index >= 0:
            anchor_indices.append(index)

    degree = np.count_nonzero(adjacency >= 0, axis=1)
    node_seed = np.zeros_like(skeleton, dtype=np.uint8)
    node_indices = np.flatnonzero(degree != 2)
    node_seed[coordinates[node_indices, 0], coordinates[node_indices, 1]] = 1
    for index in anchor_indices:
        y, x = coordinates[index]
        node_seed[y, x] = 1
    if not np.any(node_seed):
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
    node_count, node_labels = cv2.connectedComponents(node_region.astype(np.uint8), connectivity=8)
    if node_count <= 1:
        return None
    node_label_by_index = node_labels[coordinates[:, 0], coordinates[:, 1]]
    representatives: dict[int, int] = {}
    for label in range(1, node_count):
        members = np.flatnonzero(node_label_by_index == label)
        pixels = coordinates[members]
        center = np.mean(pixels, axis=0)
        representatives[label] = int(members[np.argmin(np.sum((pixels - center) ** 2, axis=1))])

    endpoint_nodes = tuple(
        int(node_label_by_index[index]) - 1
        for index in anchor_indices
        if int(node_label_by_index[index]) > 0
    )
    chain_mask = skeleton & ~node_region
    chain_count, chain_labels = cv2.connectedComponents(chain_mask.astype(np.uint8), connectivity=8)
    chain_label_by_index = chain_labels[coordinates[:, 0], coordinates[:, 1]]
    crop_y, crop_x = crop_origin_yx
    global_yx = coordinates + np.asarray((crop_y, crop_x), dtype=np.int32)
    edges: list[_RawEdge] = []
    for chain_label in range(1, chain_count):
        members = np.flatnonzero(chain_label_by_index == chain_label)
        if len(members) == 0:
            continue
        neighbor_indices = adjacency[members].reshape(-1)
        source_indices = np.repeat(members, adjacency.shape[1])
        present = neighbor_indices >= 0
        neighbor_indices = neighbor_indices[present]
        source_indices = source_indices[present]
        neighbor_node_labels = node_label_by_index[neighbor_indices]
        touches = neighbor_node_labels > 0
        neighbor_node_labels = neighbor_node_labels[touches]
        source_indices = source_indices[touches]
        contacts = {
            int(label): np.unique(source_indices[neighbor_node_labels == label])
            for label in np.unique(neighbor_node_labels)
        }
        nodes = sorted(contacts)
        if len(nodes) == 0 or len(nodes) > 2:
            continue
        if len(nodes) == 2:
            node_a, node_b = nodes
            start_candidates = contacts[node_a]
            end_candidates = contacts[node_b]
        else:
            node_a = node_b = nodes[0]
            boundary = contacts[node_a]
            if len(boundary) < 2:
                continue
            boundary_yx = coordinates[boundary].astype(np.float64)
            distance = np.sum((boundary_yx[:, None] - boundary_yx[None]) ** 2, axis=-1)
            first, second = np.unravel_index(np.argmax(distance), distance.shape)
            start_candidates = np.asarray((boundary[first],), dtype=np.int32)
            end_candidates = np.asarray((boundary[second],), dtype=np.int32)
        representative_a = coordinates[representatives[node_a]]
        representative_b = coordinates[representatives[node_b]]
        start = min(
            (int(index) for index in start_candidates),
            key=lambda index: float(np.sum((coordinates[index] - representative_a) ** 2)),
        )
        end = min(
            (int(index) for index in end_candidates),
            key=lambda index: float(np.sum((coordinates[index] - representative_b) ** 2)),
        )
        ordered = _order_nonbranching_chain(members, adjacency, start, end)
        if not ordered:
            continue
        route_indices = np.concatenate(
            (
                np.asarray((representatives[node_a],), dtype=np.int32),
                np.asarray(ordered, dtype=np.int32),
                np.asarray((representatives[node_b],), dtype=np.int32),
            )
        )
        pixels_xy = np.ascontiguousarray(global_yx[route_indices, ::-1], dtype=np.float32)
        pixel_length = float(np.sum(np.linalg.norm(np.diff(pixels_xy, axis=0), axis=1)))
        if pixel_length <= 0.0:
            continue
        edges.append(
            _RawEdge(
                edge_id=len(edges),
                node_a=node_a - 1,
                node_b=node_b - 1,
                pixels_xy=pixels_xy,
            )
        )
    if not edges:
        return None
    node_pixels = np.ascontiguousarray(
        global_yx[
            np.asarray([representatives[label] for label in range(1, node_count)], dtype=np.int32)
        ][:, ::-1],
        dtype=np.float32,
    )
    return _SkeletonGraph(
        edges=tuple(edges),
        endpoint_nodes=endpoint_nodes,
        node_count=node_count - 1,
        node_pixels_xy=node_pixels,
    )


def _lift_pixels(
    pixels_xy: np.ndarray,
    depth_m: np.ndarray,
    support_labels: np.ndarray,
    support_label: int,
    calibration: CameraCalibration,
    settings: ObservationSettings,
    *,
    exclude_pixels_xy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pixels = np.asarray(pixels_xy, dtype=np.float32)
    count = len(pixels)
    xyz = np.full((count, 3), np.nan, dtype=np.float32)
    covariance = np.full((count, 3, 3), np.nan, dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    if count == 0:
        return xyz, valid, covariance
    radius = settings.depth_radius_px
    offsets = np.arange(-radius, radius + 1, dtype=np.int32)
    offset_y, offset_x = np.meshgrid(offsets, offsets, indexing="ij")
    center_x = np.rint(pixels[:, 0]).astype(np.int32)
    center_y = np.rint(pixels[:, 1]).astype(np.int32)
    sample_y = center_y[:, None] + offset_y.reshape(1, -1)
    sample_x = center_x[:, None] + offset_x.reshape(1, -1)
    height, width = depth_m.shape
    in_bounds = (
        (sample_y >= 0) & (sample_y < height) & (sample_x >= 0) & (sample_x < width)
    )
    safe_y = np.clip(sample_y, 0, height - 1)
    safe_x = np.clip(sample_x, 0, width - 1)
    selected_depth = depth_m[safe_y, safe_x]
    selected = (
        in_bounds
        & (support_labels[safe_y, safe_x] == int(support_label))
        & np.isfinite(selected_depth)
        & (selected_depth > 0.0)
    )
    valid_counts = np.sum(selected, axis=1)
    ordered = np.sort(np.where(selected, selected_depth, np.float32(np.inf)), axis=1)
    rows = np.arange(count)
    lower = ordered[rows, np.maximum(0, (valid_counts - 1) // 2)]
    upper = ordered[rows, np.maximum(0, valid_counts // 2)]
    median_depth = (lower + upper) * np.float32(0.5)
    finite_median = valid_counts >= settings.depth_min_samples
    deviations = np.where(
        selected,
        np.abs(selected_depth - median_depth[:, None]),
        np.float32(np.inf),
    )
    ordered_deviation = np.sort(deviations, axis=1)
    mad_lower = ordered_deviation[rows, np.maximum(0, (valid_counts - 1) // 2)]
    mad_upper = ordered_deviation[rows, np.maximum(0, valid_counts // 2)]
    mad = (mad_lower + mad_upper) * np.float32(0.5)
    valid[:] = finite_median
    if exclude_pixels_xy is not None and len(exclude_pixels_xy):
        delta = pixels[:, None, :] - np.asarray(exclude_pixels_xy, dtype=np.float32)[None, :, :]
        near_crossing = np.any(
            np.sum(delta * delta, axis=2)
            <= settings.crossing_depth_exclusion_px ** 2,
            axis=1,
        )
        valid &= ~near_crossing
    if not valid.any():
        return xyz, valid, covariance
    indices = np.flatnonzero(valid)
    u = pixels[indices, 0].astype(np.float64)
    v = pixels[indices, 1].astype(np.float64)
    distance = median_depth[indices].astype(np.float64)
    x = (u - calibration.cx_px) * distance / calibration.fx_px
    y = (calibration.cy_px - v) * distance / calibration.fy_px
    z = -distance
    xyz[indices] = np.column_stack((x, y, z)).astype(np.float32)
    sigma_depth = np.maximum(
        1.4826 * mad[indices].astype(np.float64),
        settings.depth_sigma_floor_m,
    )
    j_depth = np.column_stack(
        (
            (u - calibration.cx_px) / calibration.fx_px,
            (calibration.cy_px - v) / calibration.fy_px,
            -np.ones_like(u),
        )
    )
    cov = sigma_depth[:, None, None] ** 2 * (
        j_depth[:, :, None] * j_depth[:, None, :]
    )
    sigma_u = distance * settings.pixel_sigma_px / calibration.fx_px
    sigma_v = distance * settings.pixel_sigma_px / calibration.fy_px
    cov[:, 0, 0] += sigma_u * sigma_u
    cov[:, 1, 1] += sigma_v * sigma_v
    covariance[indices] = cov.astype(np.float32)
    return xyz, valid, covariance


def _polyline_lengths(
    pixels_xy: np.ndarray,
    xyz: np.ndarray,
    valid: np.ndarray,
    pixel_scale_m: float | None,
) -> tuple[float, float | None]:
    if len(pixels_xy) < 2:
        return 0.0, None
    consecutive = valid[:-1] & valid[1:]
    xyz_lengths = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    pixel_lengths = np.linalg.norm(np.diff(pixels_xy, axis=0), axis=1)
    observed = float(np.sum(xyz_lengths[consecutive], dtype=np.float64))
    if consecutive.all():
        return max(0.0, observed), max(1.0e-6, observed)
    if pixel_scale_m is None:
        return max(0.0, observed), None
    estimated_segments = np.where(consecutive, xyz_lengths, pixel_lengths * pixel_scale_m)
    estimated = float(np.sum(estimated_segments, dtype=np.float64))
    return max(0.0, observed), max(1.0e-6, estimated)


def _enumerate_graph_trails(
    graph: _SkeletonGraph,
    start_node: int,
    end_node: int,
    target_length_m: float,
    settings: ObservationSettings,
) -> list[tuple[tuple[int, bool], ...]]:
    adjacency: list[list[tuple[int, int, bool]]] = [[] for _ in range(graph.node_count)]
    for edge in graph.edges:
        adjacency[edge.node_a].append((edge.edge_id, edge.node_b, True))
        if edge.node_b != edge.node_a:
            adjacency[edge.node_b].append((edge.edge_id, edge.node_a, False))
    counter = itertools.count()
    queue: list[tuple[float, int, int, float, int, tuple[tuple[int, bool], ...]]] = []
    heapq.heappush(queue, (target_length_m, next(counter), start_node, 0.0, 0, ()))
    completed: list[tuple[float, tuple[tuple[int, bool], ...]]] = []
    expanded = 0
    maximum_length = target_length_m + settings.route_length_margin_m
    while queue and expanded < settings.route_search_state_limit:
        _priority, _order, node, length_m, used_mask, trail = heapq.heappop(queue)
        expanded += 1
        if node == end_node and trail:
            completed.append((length_m, trail))
            continue
        if len(trail) >= settings.route_edge_limit:
            continue
        for edge_id, next_node, forward in adjacency[node]:
            edge_bit = 1 << edge_id
            if used_mask & edge_bit:
                continue
            edge_length = graph.edges[edge_id].metric_length
            if edge_length is None:
                continue
            new_length = length_m + edge_length
            if new_length > maximum_length:
                continue
            new_trail = trail + ((edge_id, forward),)
            heapq.heappush(
                queue,
                (
                    abs(target_length_m - new_length),
                    next(counter),
                    next_node,
                    new_length,
                    used_mask | edge_bit,
                    new_trail,
                ),
            )
    completed.sort(key=lambda item: (abs(item[0] - target_length_m), len(item[1])))
    return [trail for _length, trail in completed[: settings.route_candidate_limit]]


def _edge_tangent_away(edge: _RawEdge, node: int) -> np.ndarray:
    points = edge.pixels_xy
    if len(points) < 2:
        return np.zeros(2, dtype=np.float64)
    if edge.node_a == node:
        vector = points[min(3, len(points) - 1)] - points[0]
    elif edge.node_b == node:
        vector = points[max(0, len(points) - 4)] - points[-1]
    else:
        return np.zeros(2, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return vector.astype(np.float64) / norm if norm > 1.0e-9 else np.zeros(2, dtype=np.float64)


def _trail_turn_cost(graph: _SkeletonGraph, trail: tuple[tuple[int, bool], ...]) -> float:
    if len(trail) < 2:
        return 0.0
    costs = []
    for (first_id, first_forward), (second_id, second_forward) in zip(trail[:-1], trail[1:]):
        first = graph.edges[first_id]
        second = graph.edges[second_id]
        node = first.node_b if first_forward else first.node_a
        if node != (second.node_a if second_forward else second.node_b):
            continue
        incoming_away = _edge_tangent_away(first, node)
        outgoing_away = _edge_tangent_away(second, node)
        if np.linalg.norm(incoming_away) == 0.0 or np.linalg.norm(outgoing_away) == 0.0:
            continue
        costs.append(float((1.0 + np.clip(np.dot(incoming_away, outgoing_away), -1.0, 1.0)) * 0.5))
    return float(np.mean(costs)) if costs else 0.0


def _perfect_matchings_four() -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    return (
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    )


class CableObservationBuilder:
    """Stateful endpoint association plus stateless per-frame graph construction."""

    def __init__(self, settings: ObservationSettings) -> None:
        self.settings = settings
        self._previous_endpoint_pixels = np.full((2, 2, 2), np.nan, dtype=np.float32)
        self._previous_endpoint_xyz = np.full((2, 2, 3), np.nan, dtype=np.float32)

    def _endpoint_candidates(
        self,
        endpoint_mask: np.ndarray,
        depth_m: np.ndarray,
        calibration: CameraCalibration,
        body_labels: np.ndarray,
    ) -> list[_EndpointCandidate]:
        x0, y0, width, height = cv2.boundingRect(endpoint_mask)
        if width == 0 or height == 0:
            return []
        roi = endpoint_mask[y0 : y0 + height, x0 : x0 + width]
        count, labels_roi, stats, centroids = cv2.connectedComponentsWithStats(roi, connectivity=8)
        candidates = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.settings.endpoint_min_area_px:
                continue
            pixel = np.asarray((centroids[label, 0] + x0, centroids[label, 1] + y0), dtype=np.float32)
            xyz, valid, covariance = _lift_pixels(
                pixel[None, :],
                depth_m,
                endpoint_mask,
                255,
                calibration,
                self.settings,
            )
            candidates.append(
                _EndpointCandidate(
                    pixel_xy=pixel,
                    xyz=xyz[0],
                    covariance=covariance[0],
                    depth_valid=bool(valid[0]),
                    area_px=area,
                    body_component_label=_nearest_component_label(
                        body_labels,
                        pixel,
                        self.settings.endpoint_label_search_px,
                    ),
                )
            )
        return candidates

    def _candidate_cost(
        self,
        cable_id: int,
        endpoint_id: int,
        candidate: _EndpointCandidate,
        calibration: CameraCalibration,
    ) -> float:
        previous_xyz = self._previous_endpoint_xyz[cable_id, endpoint_id]
        if candidate.depth_valid and np.isfinite(previous_xyz).all():
            return float(
                np.linalg.norm(candidate.xyz - previous_xyz)
                / self.settings.endpoint_association_gate_m
            )
        previous_pixel = self._previous_endpoint_pixels[cable_id, endpoint_id]
        if np.isfinite(previous_pixel).all():
            pixel_gate = max(60.0, 0.1 * min(calibration.width_px, calibration.height_px))
            return float(np.linalg.norm(candidate.pixel_xy - previous_pixel) / pixel_gate)
        return 0.0

    def _associate_endpoints(
        self,
        cable_id: int,
        candidates: list[_EndpointCandidate],
        calibration: CameraCalibration,
    ) -> tuple[EndpointObservation, ...]:
        if not candidates:
            return ()
        selected = sorted(candidates, key=lambda item: -item.area_px)[:2]
        previous_available = np.isfinite(self._previous_endpoint_pixels[cable_id]).all(axis=1)
        assignments: list[tuple[int, _EndpointCandidate, float]] = []
        if len(selected) == 2:
            if previous_available.all():
                direct = self._candidate_cost(cable_id, 0, selected[0], calibration) + self._candidate_cost(cable_id, 1, selected[1], calibration)
                swapped = self._candidate_cost(cable_id, 0, selected[1], calibration) + self._candidate_cost(cable_id, 1, selected[0], calibration)
                if swapped < direct:
                    selected.reverse()
                total = min(direct, swapped)
                confidence = float(math.exp(-0.5 * total))
            else:
                selected.sort(key=lambda item: (float(item.pixel_xy[0]), float(item.pixel_xy[1])))
                confidence = 1.0
            assignments = [(0, selected[0], confidence), (1, selected[1], confidence)]
        else:
            candidate = selected[0]
            if previous_available.any():
                costs = [
                    self._candidate_cost(cable_id, endpoint_id, candidate, calibration)
                    if previous_available[endpoint_id]
                    else float("inf")
                    for endpoint_id in (0, 1)
                ]
                endpoint_id = int(np.argmin(costs))
                if costs[endpoint_id] <= 1.0:
                    assignments = [(endpoint_id, candidate, float(math.exp(-0.5 * costs[endpoint_id])))]
                else:
                    assignments = [(-1, candidate, 0.0)]
            else:
                assignments = [(-1, candidate, 0.0)]

        observations = []
        for endpoint_id, candidate, confidence in assignments:
            observation = EndpointObservation(
                cable_id=cable_id,
                endpoint_id=endpoint_id,
                pixel_xy_f32=candidate.pixel_xy,
                xyz_camera_m_f32=candidate.xyz,
                covariance_m2_f32=candidate.covariance,
                depth_valid=candidate.depth_valid,
                area_px=candidate.area_px,
                body_component_label=candidate.body_component_label,
                association_confidence=confidence,
            )
            observations.append(observation)
            if endpoint_id >= 0:
                self._previous_endpoint_pixels[cable_id, endpoint_id] = candidate.pixel_xy
                if candidate.depth_valid:
                    self._previous_endpoint_xyz[cable_id, endpoint_id] = candidate.xyz
        return tuple(observations)

    def build(
        self,
        perception: PerceptionFrame,
        calibration: CameraCalibration,
    ) -> CableObservationFrame:
        if not isinstance(perception, PerceptionFrame):
            raise TypeError("CableObservationBuilder requires a PerceptionFrame")
        if not isinstance(calibration, CameraCalibration):
            raise TypeError("CableObservationBuilder requires CameraCalibration")
        frame = perception.rgbd
        expected = (calibration.height_px, calibration.width_px)
        if frame.depth_m_f32.shape != expected or frame.bgr_u8.shape[:2] != expected:
            raise ValueError("RGB-D dimensions do not match observation calibration")
        started = time.perf_counter()
        body_mask = np.ascontiguousarray(perception.body_mask_u8, dtype=np.uint8)
        endpoint_masks = (
            np.ascontiguousarray(perception.endpoint1_mask_u8, dtype=np.uint8),
            np.ascontiguousarray(perception.endpoint2_mask_u8, dtype=np.uint8),
        )
        component_started = time.perf_counter()
        body_count, body_labels, body_stats, _ = cv2.connectedComponentsWithStats(body_mask, connectivity=8)
        components_ms = (time.perf_counter() - component_started) * 1000.0

        endpoint_started = time.perf_counter()
        endpoints_list: list[EndpointObservation] = []
        for cable_id, endpoint_mask in enumerate(endpoint_masks):
            candidates = self._endpoint_candidates(
                endpoint_mask,
                frame.depth_m_f32,
                calibration,
                body_labels,
            )
            endpoints_list.extend(self._associate_endpoints(cable_id, candidates, calibration))
        endpoints = tuple(endpoints_list)
        endpoints_ms = (time.perf_counter() - endpoint_started) * 1000.0

        endpoint_by_identity = {
            (endpoint.cable_id, endpoint.endpoint_id): endpoint
            for endpoint in endpoints
            if endpoint.endpoint_id >= 0
        }
        skeleton_ms = graph_ms = lifting_ms = routes_ms = 0.0
        skeleton_pixels = 0
        graph_node_count = 0
        observed_edges: list[GraphEdgeObservation] = []
        crossings: list[CrossingObservation] = []
        components: dict[int, _ComponentGraph] = {}
        next_node_id = 0

        for component_label in range(1, body_count):
            if int(body_stats[component_label, cv2.CC_STAT_AREA]) < self.settings.component_min_area_px:
                continue
            roi, (y0, x0) = _component_roi(body_labels, body_stats, component_label)
            skeleton_started = time.perf_counter()
            skeleton = _morphological_skeleton(roi)
            skeleton_ms += (time.perf_counter() - skeleton_started) * 1000.0
            skeleton_pixels += int(np.count_nonzero(skeleton))
            owners: list[tuple[int, int]] = []
            anchors: list[tuple[int, int]] = []
            for owner, endpoint in endpoint_by_identity.items():
                if endpoint.body_component_label != component_label:
                    continue
                anchor = _nearest_pixel(
                    skeleton,
                    endpoint.pixel_xy_f32 - np.asarray((x0, y0), dtype=np.float32),
                )
                if anchor is not None:
                    owners.append(owner)
                    anchors.append(anchor)
            graph_started = time.perf_counter()
            graph = _build_skeleton_graph(
                skeleton,
                tuple(anchors),
                (y0, x0),
                self.settings.graph_node_dilation_px,
            )
            graph_ms += (time.perf_counter() - graph_started) * 1000.0
            if graph is None:
                continue
            node_by_owner = {
                owner: graph.endpoint_nodes[index]
                for index, owner in enumerate(owners)
                if index < len(graph.endpoint_nodes)
            }
            incidence: list[list[int]] = [[] for _ in range(graph.node_count)]
            for edge in graph.edges:
                incidence[edge.node_a].append(edge.edge_id)
                incidence[edge.node_b].append(edge.edge_id)
            crossing_nodes = [node for node, edge_ids in enumerate(incidence) if len(edge_ids) == 4]
            crossing_pixels = np.ascontiguousarray(
                graph.node_pixels_xy[crossing_nodes]
                if crossing_nodes
                else np.empty((0, 2), dtype=np.float32),
                dtype=np.float32,
            )
            component_depth_roi = frame.depth_m_f32[
                y0 : y0 + roi.shape[0],
                x0 : x0 + roi.shape[1],
            ]
            valid_depth = component_depth_roi[
                roi
                & np.isfinite(component_depth_roi)
                & (component_depth_roi > 0.0)
            ]
            pixel_scale = (
                float(np.median(valid_depth))
                / (0.5 * (calibration.fx_px + calibration.fy_px))
                if valid_depth.size
                else None
            )

            component_lifting_started = time.perf_counter()
            global_edge_ids = []
            updated_raw_edges = []
            component_observed_edges: dict[int, GraphEdgeObservation] = {}
            for raw_edge in graph.edges:
                pixels = _resample_polyline(raw_edge.pixels_xy, self.settings.edge_samples)
                xyz, depth_valid, covariance = _lift_pixels(
                    pixels,
                    frame.depth_m_f32,
                    body_labels,
                    component_label,
                    calibration,
                    self.settings,
                    exclude_pixels_xy=crossing_pixels,
                )
                observed_length, estimated_length = _polyline_lengths(
                    pixels, xyz, depth_valid, pixel_scale
                )
                global_edge_id = len(observed_edges)
                global_edge_ids.append(global_edge_id)
                anchors_for_edge = []
                for owner, node in node_by_owner.items():
                    if node == raw_edge.node_a:
                        edge_end = 0
                    elif node == raw_edge.node_b:
                        edge_end = 1
                    else:
                        continue
                    anchors_for_edge.append(
                        GraphEndpointAnchor(
                            edge_end=edge_end,
                            cable_id=owner[0],
                            endpoint_id=owner[1],
                        )
                    )
                edge_observation = GraphEdgeObservation(
                    component_label=component_label,
                    edge_id=global_edge_id,
                    node_a=next_node_id + raw_edge.node_a,
                    node_b=next_node_id + raw_edge.node_b,
                    pixels_xy_f32=pixels,
                    xyz_camera_m_f32=xyz,
                    depth_valid_bool=depth_valid,
                    covariance_m2_f32=covariance,
                    observed_length_m=observed_length,
                    estimated_length_m=estimated_length,
                    endpoint_anchors=tuple(anchors_for_edge),
                )
                observed_edges.append(edge_observation)
                component_observed_edges[raw_edge.edge_id] = edge_observation
                updated_raw_edges.append(replace(raw_edge, metric_length=estimated_length))
            lifting_ms += (time.perf_counter() - component_lifting_started) * 1000.0
            graph = replace(graph, edges=tuple(updated_raw_edges))

            for node in crossing_nodes:
                incident_local = sorted(incidence[node])
                incident_global = np.asarray(
                    [global_edge_ids[edge_id] for edge_id in incident_local], dtype=np.int32
                )
                tangents = [_edge_tangent_away(graph.edges[edge_id], node) for edge_id in incident_local]
                ranked_pairings = []
                for matching in _perfect_matchings_four():
                    cost = float(
                        np.mean(
                            [
                                (1.0 + np.clip(np.dot(tangents[a], tangents[b]), -1.0, 1.0)) * 0.5
                                for a, b in matching
                            ]
                        )
                    )
                    edge_pairs = np.asarray(
                        [
                            (incident_global[a], incident_global[b])
                            for a, b in matching
                        ],
                        dtype=np.int32,
                    )
                    ranked_pairings.append((cost, edge_pairs))
                ranked_pairings.sort(key=lambda item: item[0])
                pairings = tuple(
                    CrossingPairingObservation(rank=rank, edge_pairs_i32=edge_pairs, tangent_cost=cost)
                    for rank, (cost, edge_pairs) in enumerate(ranked_pairings)
                )
                arm_xyz = np.full((4, 3), np.nan, dtype=np.float32)
                arm_valid = np.zeros(4, dtype=bool)
                arm_covariance = np.full((4, 3, 3), np.nan, dtype=np.float32)
                for arm_index, edge_id in enumerate(incident_local):
                    edge = component_observed_edges[edge_id]
                    valid_indices = np.flatnonzero(edge.depth_valid_bool)
                    if len(valid_indices) == 0:
                        continue
                    raw_edge = graph.edges[edge_id]
                    sample_index = int(valid_indices[0] if raw_edge.node_a == node else valid_indices[-1])
                    arm_xyz[arm_index] = edge.xyz_camera_m_f32[sample_index]
                    arm_valid[arm_index] = True
                    arm_covariance[arm_index] = edge.covariance_m2_f32[sample_index]
                crossings.append(
                    CrossingObservation(
                        component_label=component_label,
                        node_id=next_node_id + node,
                        pixel_xy_f32=graph.node_pixels_xy[node],
                        incident_edge_ids_i32=incident_global,
                        arm_xyz_camera_m_f32=arm_xyz,
                        arm_depth_valid_bool=arm_valid,
                        arm_covariance_m2_f32=arm_covariance,
                        pairings=pairings,
                    )
                )
            components[component_label] = _ComponentGraph(
                component_label=component_label,
                graph=graph,
                node_by_owner=node_by_owner,
                global_edge_ids=tuple(global_edge_ids),
                observed_edges=tuple(
                    component_observed_edges[edge_id]
                    for edge_id in range(len(graph.edges))
                ),
                pixel_scale_m=pixel_scale,
            )
            graph_node_count += graph.node_count
            next_node_id += graph.node_count

        routes_by_cable: list[tuple[RouteHypothesis, ...]] = []
        reasons = []
        route_started = time.perf_counter()
        for cable_id in (0, 1):
            endpoints_for_cable = {
                endpoint.endpoint_id: endpoint
                for endpoint in endpoints
                if endpoint.cable_id == cable_id and endpoint.endpoint_id >= 0
            }
            if set(endpoints_for_cable) != {0, 1}:
                routes_by_cable.append(())
                reasons.append(f"partial endpoint evidence: {len(endpoints_for_cable)}/2 identified")
                continue
            endpoint_a = endpoints_for_cable[0]
            endpoint_b = endpoints_for_cable[1]
            if endpoint_a.body_component_label <= 0 or endpoint_a.body_component_label != endpoint_b.body_component_label:
                routes_by_cable.append(())
                reasons.append("endpoints are not on one observed body component")
                continue
            component = components.get(endpoint_a.body_component_label)
            if component is None:
                routes_by_cable.append(())
                reasons.append("endpoint body component has no compressed graph")
                continue
            start_node = component.node_by_owner.get((cable_id, 0), -1)
            end_node = component.node_by_owner.get((cable_id, 1), -1)
            if start_node < 0 or end_node < 0:
                routes_by_cable.append(())
                reasons.append("endpoint anchor is absent from compressed graph")
                continue
            trails = _enumerate_graph_trails(
                component.graph,
                start_node,
                end_node,
                self.settings.cable_lengths_m[cable_id],
                self.settings,
            )
            routes = []
            for trail in trails:
                pixel_parts = []
                xyz_parts = []
                valid_parts = []
                covariance_parts = []
                for edge_id, forward in trail:
                    edge = component.observed_edges[edge_id]
                    direction = slice(None) if forward else slice(None, None, -1)
                    points = edge.pixels_xy_f32[direction]
                    xyz = edge.xyz_camera_m_f32[direction]
                    valid = edge.depth_valid_bool[direction]
                    covariance = edge.covariance_m2_f32[direction]
                    if pixel_parts:
                        points = points[1:]
                        xyz = xyz[1:]
                        valid = valid[1:]
                        covariance = covariance[1:]
                    pixel_parts.append(points)
                    xyz_parts.append(xyz)
                    valid_parts.append(valid)
                    covariance_parts.append(covariance)
                if not pixel_parts:
                    continue
                pixels = np.ascontiguousarray(np.concatenate(pixel_parts), dtype=np.float32)
                xyz = np.ascontiguousarray(np.concatenate(xyz_parts), dtype=np.float32)
                depth_valid = np.ascontiguousarray(
                    np.concatenate(valid_parts), dtype=np.bool_
                )
                covariance = np.ascontiguousarray(
                    np.concatenate(covariance_parts), dtype=np.float32
                )
                pixels[0] = endpoint_a.pixel_xy_f32
                pixels[-1] = endpoint_b.pixel_xy_f32
                if endpoint_a.depth_valid:
                    xyz[0] = endpoint_a.xyz_camera_m_f32
                    covariance[0] = endpoint_a.covariance_m2_f32
                    depth_valid[0] = True
                else:
                    xyz[0] = np.nan
                    covariance[0] = np.nan
                    depth_valid[0] = False
                if endpoint_b.depth_valid:
                    xyz[-1] = endpoint_b.xyz_camera_m_f32
                    covariance[-1] = endpoint_b.covariance_m2_f32
                    depth_valid[-1] = True
                else:
                    xyz[-1] = np.nan
                    covariance[-1] = np.nan
                    depth_valid[-1] = False
                observed_length, estimated_length = _polyline_lengths(
                    pixels, xyz, depth_valid, component.pixel_scale_m
                )
                if estimated_length is None:
                    continue
                target = self.settings.cable_lengths_m[cable_id]
                tangent_cost = _trail_turn_cost(component.graph, trail)
                score = abs(estimated_length - target) / target + 0.25 * tangent_cost
                routes.append(
                    RouteHypothesis(
                        cable_id=cable_id,
                        component_label=component.component_label,
                        edge_ids_i32=np.asarray(
                            [component.global_edge_ids[edge_id] for edge_id, _ in trail],
                            dtype=np.int32,
                        ),
                        edge_forward_bool=np.asarray([forward for _, forward in trail], dtype=bool),
                        pixels_xy_f32=pixels,
                        xyz_camera_m_f32=xyz,
                        depth_valid_bool=depth_valid,
                        covariance_m2_f32=covariance,
                        observed_length_m=observed_length,
                        estimated_length_m=estimated_length,
                        target_length_m=target,
                        tangent_cost=tangent_cost,
                        score=score,
                    )
                )
            routes.sort(key=lambda route: route.score)
            routes_by_cable.append(tuple(routes))
            reasons.append(
                f"{len(routes)} complete route hypotheses"
                if routes
                else "no length-compatible endpoint route"
            )
        routes_ms = (time.perf_counter() - route_started) * 1000.0
        total_ms = (time.perf_counter() - started) * 1000.0
        timing = CableObservationTiming(
            total_ms=total_ms,
            endpoints_ms=endpoints_ms,
            components_ms=components_ms,
            skeleton_ms=skeleton_ms,
            graph_ms=graph_ms,
            lifting_ms=lifting_ms,
            routes_ms=routes_ms,
            skeleton_pixels=skeleton_pixels,
            graph_nodes=graph_node_count,
            graph_edges=len(observed_edges),
            crossing_count=len(crossings),
            route_count=sum(len(routes) for routes in routes_by_cable),
            depth_valid_samples=sum(int(np.count_nonzero(edge.depth_valid_bool)) for edge in observed_edges),
        )
        return CableObservationFrame(
            key=frame.key,
            calibration=calibration,
            endpoints=endpoints,
            graph_edges=tuple(observed_edges),
            crossings=tuple(crossings),
            routes_by_cable=(routes_by_cable[0], routes_by_cable[1]),
            cable_reasons=(reasons[0], reasons[1]),
            body_component_count=sum(
                int(body_stats[label, cv2.CC_STAT_AREA])
                >= self.settings.component_min_area_px
                for label in range(1, body_count)
            ),
            timing=timing,
        )


__all__ = [
    "CableObservationBuilder",
    "CableObservationFrame",
    "CableObservationTiming",
    "CrossingObservation",
    "CrossingPairingObservation",
    "EndpointObservation",
    "GraphEdgeObservation",
    "GraphEndpointAnchor",
    "RouteHypothesis",
]
