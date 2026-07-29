"""Passive geometry-only tracking of a known yellow cube from registered RGB-D."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations, product
import math
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraModel:
    """Rectified left-camera model in the ZED right-handed Y-up frame."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def validate(self) -> None:
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("Camera focal lengths must be positive.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera dimensions must be positive.")


@dataclass(frozen=True)
class CubeTrackerConfig:
    """Only the physical, colour, and robust-fit parameters needed by the tester."""

    cube_side_m: float = 0.150
    hue_min: int = 18
    hue_max: int = 40
    saturation_min: int = 100
    value_min: int = 80
    minimum_component_area_px: int = 2500
    depth_min_m: float = 0.20
    depth_max_m: float = 3.00
    pixel_stride: int = 2
    maximum_points: int = 12000
    plane_inlier_threshold_m: float = 0.006
    ransac_iterations: int = 96
    minimum_face_points: int = 300
    minimum_face_fraction: float = 0.04
    orthogonality_tolerance_deg: float = 15.0
    two_face_minimum_span_fraction: float = 0.90
    two_face_maximum_span_fraction: float = 1.10
    random_seed: int = 7

    @classmethod
    def from_mapping(cls, values: dict | None) -> "CubeTrackerConfig":
        source = values or {}
        defaults = cls()
        config = cls(
            cube_side_m=float(source.get("cube_side_m", defaults.cube_side_m)),
            hue_min=int(source.get("hue_min", defaults.hue_min)),
            hue_max=int(source.get("hue_max", defaults.hue_max)),
            saturation_min=int(
                source.get("saturation_min", defaults.saturation_min)
            ),
            value_min=int(source.get("value_min", defaults.value_min)),
            minimum_component_area_px=int(
                source.get(
                    "minimum_component_area_px",
                    defaults.minimum_component_area_px,
                )
            ),
            depth_min_m=float(source.get("depth_min_m", defaults.depth_min_m)),
            depth_max_m=float(source.get("depth_max_m", defaults.depth_max_m)),
            pixel_stride=int(source.get("pixel_stride", defaults.pixel_stride)),
            maximum_points=int(
                source.get("maximum_points", defaults.maximum_points)
            ),
            plane_inlier_threshold_m=float(
                source.get(
                    "plane_inlier_threshold_m",
                    defaults.plane_inlier_threshold_m,
                )
            ),
            ransac_iterations=int(
                source.get("ransac_iterations", defaults.ransac_iterations)
            ),
            minimum_face_points=int(
                source.get("minimum_face_points", defaults.minimum_face_points)
            ),
            minimum_face_fraction=float(
                source.get(
                    "minimum_face_fraction",
                    defaults.minimum_face_fraction,
                )
            ),
            orthogonality_tolerance_deg=float(
                source.get(
                    "orthogonality_tolerance_deg",
                    defaults.orthogonality_tolerance_deg,
                )
            ),
            two_face_minimum_span_fraction=float(
                source.get(
                    "two_face_minimum_span_fraction",
                    defaults.two_face_minimum_span_fraction,
                )
            ),
            two_face_maximum_span_fraction=float(
                source.get(
                    "two_face_maximum_span_fraction",
                    defaults.two_face_maximum_span_fraction,
                )
            ),
            random_seed=int(source.get("random_seed", defaults.random_seed)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.cube_side_m <= 0.0:
            raise ValueError("cube_side_m must be positive.")
        if not 0 <= self.hue_min <= self.hue_max <= 179:
            raise ValueError("The OpenCV hue interval must satisfy 0 <= min <= max <= 179.")
        if not 0 <= self.saturation_min <= 255:
            raise ValueError("saturation_min must be in [0, 255].")
        if not 0 <= self.value_min <= 255:
            raise ValueError("value_min must be in [0, 255].")
        if self.minimum_component_area_px < 1:
            raise ValueError("minimum_component_area_px must be positive.")
        if not 0.0 < self.depth_min_m < self.depth_max_m:
            raise ValueError("Depth limits must satisfy 0 < depth_min_m < depth_max_m.")
        if self.pixel_stride < 1:
            raise ValueError("pixel_stride must be at least one.")
        if self.maximum_points < 2 * self.minimum_face_points:
            raise ValueError(
                "maximum_points must accommodate at least two minimum-size faces."
            )
        if self.plane_inlier_threshold_m <= 0.0:
            raise ValueError("plane_inlier_threshold_m must be positive.")
        if self.ransac_iterations < 8:
            raise ValueError("ransac_iterations must be at least eight.")
        if self.minimum_face_points < 3:
            raise ValueError("minimum_face_points must be at least three.")
        if not 0.0 <= self.minimum_face_fraction < 1.0 / 3.0:
            raise ValueError("minimum_face_fraction must be in [0, 1/3).")
        if not 0.0 < self.orthogonality_tolerance_deg < 45.0:
            raise ValueError("orthogonality_tolerance_deg must be in (0, 45).")
        if not 0.0 < self.two_face_minimum_span_fraction <= 1.0:
            raise ValueError("two_face_minimum_span_fraction must be in (0, 1].")
        if self.two_face_maximum_span_fraction < 1.0:
            raise ValueError("two_face_maximum_span_fraction must be at least one.")
        if (
            self.two_face_minimum_span_fraction
            >= self.two_face_maximum_span_fraction
        ):
            raise ValueError(
                "The two-face minimum span fraction must be below the maximum."
            )


@dataclass(frozen=True)
class PlaneEstimate:
    normal: np.ndarray
    offset_m: float
    centroid: np.ndarray
    point_indices: np.ndarray
    rms_m: float


@dataclass(frozen=True)
class CubeTrackingResult:
    valid: bool
    reason: str
    cube_side_m: float
    mask: np.ndarray
    center_m: np.ndarray | None
    rotation: np.ndarray | None
    quaternion_xyzw: np.ndarray | None
    candidate_plane_count: int
    face_count: int
    yellow_pixels: int
    sampled_mask_pixels: int
    valid_depth_points: int
    depth_coverage: float
    fit_point_count: int
    surface_rms_m: float
    observed_span_m: float
    processing_ms: float
    face_pixels: tuple[np.ndarray, ...]


def _cube_symmetries() -> tuple[np.ndarray, ...]:
    rotations: list[np.ndarray] = []
    identity = np.eye(3, dtype=np.float64)
    for order in permutations(range(3)):
        permuted = identity[:, order]
        for signs in product((-1.0, 1.0), repeat=3):
            candidate = permuted @ np.diag(signs)
            if np.linalg.det(candidate) > 0.5:
                rotations.append(candidate)
    if len(rotations) != 24:
        raise RuntimeError(f"Expected 24 proper cube symmetries, got {len(rotations)}.")
    return tuple(rotations)


CUBE_SYMMETRIES = _cube_symmetries()
CUBE_CORNERS = np.asarray(
    list(product((-1.0, 1.0), repeat=3)),
    dtype=np.float64,
)
CUBE_EDGES = tuple(
    (first, second)
    for first in range(8)
    for second in range(first + 1, 8)
    if np.count_nonzero(CUBE_CORNERS[first] != CUBE_CORNERS[second]) == 1
)
FACE_COLOURS = ((255, 80, 80), (80, 255, 80), (80, 160, 255))


def segment_yellow(bgr: np.ndarray, config: CubeTrackerConfig) -> np.ndarray:
    """Return only the largest sufficiently large yellow connected component."""

    image = np.asarray(bgr, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected a BGR image, got shape {image.shape}.")
    hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        (config.hue_min, config.saturation_min, config.value_min),
        (config.hue_max, 255, 255),
    )
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return np.zeros(mask.shape, dtype=np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[largest, cv2.CC_STAT_AREA])
    if area < config.minimum_component_area_px:
        return np.zeros(mask.shape, dtype=np.uint8)
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def unproject_masked_depth(
    mask: np.ndarray,
    depth: np.ndarray,
    camera: CameraModel,
    config: CubeTrackerConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Unproject sampled mask pixels into the ZED right-handed Y-up frame."""

    depth_image = np.asarray(depth, dtype=np.float32)
    if depth_image.shape != mask.shape:
        raise ValueError(f"Depth/mask shapes disagree: {depth_image.shape} vs {mask.shape}.")
    if depth_image.shape != (camera.height, camera.width):
        raise ValueError(
            "Camera model dimensions do not match the retrieved frame: "
            f"{camera.width}x{camera.height} vs "
            f"{depth_image.shape[1]}x{depth_image.shape[0]}."
        )

    stride = config.pixel_stride
    sampled_mask = mask[::stride, ::stride] != 0
    sample_y, sample_x = np.nonzero(sampled_mask)
    sampled_count = int(len(sample_x))
    if sampled_count == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.int32),
            0,
            0,
        )

    pixel_x = (sample_x * stride).astype(np.int32)
    pixel_y = (sample_y * stride).astype(np.int32)
    sampled_depth = depth_image[pixel_y, pixel_x]
    valid = (
        np.isfinite(sampled_depth)
        & (sampled_depth >= config.depth_min_m)
        & (sampled_depth <= config.depth_max_m)
    )
    pixel_x = pixel_x[valid]
    pixel_y = pixel_y[valid]
    sampled_depth = sampled_depth[valid].astype(np.float64)
    valid_count = int(len(sampled_depth))

    if valid_count > config.maximum_points:
        selected = rng.choice(valid_count, size=config.maximum_points, replace=False)
        pixel_x = pixel_x[selected]
        pixel_y = pixel_y[selected]
        sampled_depth = sampled_depth[selected]

    points = np.empty((len(sampled_depth), 3), dtype=np.float64)
    points[:, 0] = (pixel_x.astype(np.float64) - camera.cx) * sampled_depth / camera.fx
    points[:, 1] = (camera.cy - pixel_y.astype(np.float64)) * sampled_depth / camera.fy
    points[:, 2] = -sampled_depth
    pixels = np.column_stack((pixel_x, pixel_y)).astype(np.int32, copy=False)
    return points, pixels, sampled_count, valid_count


def _refine_plane(
    points: np.ndarray,
    initial_inliers: np.ndarray,
    threshold_m: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, float] | None:
    inliers = np.asarray(initial_inliers, dtype=bool)
    if np.count_nonzero(inliers) < 3:
        return None

    normal = np.zeros(3, dtype=np.float64)
    centroid = np.zeros(3, dtype=np.float64)
    offset = 0.0
    for _ in range(2):
        selected = points[inliers]
        if len(selected) < 3:
            return None
        centroid = np.mean(selected, axis=0)
        centered = selected - centroid
        covariance = centered.T @ centered / float(len(selected))
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if not np.all(np.isfinite(eigenvalues)):
            return None
        normal = eigenvectors[:, 0]
        normal /= np.linalg.norm(normal)
        if float(np.dot(normal, -centroid)) < 0.0:
            normal = -normal
        offset = float(np.dot(normal, centroid))
        inliers = np.abs(points @ normal - offset) <= threshold_m

    selected = points[inliers]
    if len(selected) < 3:
        return None
    centroid = np.mean(selected, axis=0)
    offset = float(np.dot(normal, centroid))
    residual = points[inliers] @ normal - offset
    rms_m = float(np.sqrt(np.mean(np.square(residual))))
    return normal, offset, centroid, inliers, rms_m


def _ransac_plane(
    points: np.ndarray,
    iterations: int,
    threshold_m: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, float] | None:
    if len(points) < 3:
        return None

    triplets = rng.integers(0, len(points), size=(iterations, 3))
    first = points[triplets[:, 0]]
    second = points[triplets[:, 1]]
    third = points[triplets[:, 2]]
    normals = np.cross(second - first, third - first)
    lengths = np.linalg.norm(normals, axis=1)
    usable = lengths > 1.0e-10
    if not np.any(usable):
        return None
    normals = normals[usable] / lengths[usable, None]
    first = first[usable]
    offsets = np.einsum("ij,ij->i", normals, first)

    distances = np.abs(points @ normals.T - offsets[None, :])
    counts = np.count_nonzero(distances <= threshold_m, axis=0)
    best = int(np.argmax(counts))
    initial_inliers = distances[:, best] <= threshold_m
    return _refine_plane(points, initial_inliers, threshold_m)


def extract_plane_candidates(
    points: np.ndarray,
    config: CubeTrackerConfig,
    rng: np.random.Generator,
) -> tuple[PlaneEstimate, ...]:
    """Sequentially extract the dominant planar subsets of the yellow cloud."""

    if len(points) < 3:
        return ()
    initial_count = len(points)
    minimum_count = max(
        config.minimum_face_points,
        int(math.ceil(config.minimum_face_fraction * initial_count)),
    )
    remaining_indices = np.arange(initial_count, dtype=np.int32)
    candidates: list[PlaneEstimate] = []

    # Three cube faces are expected; two additional candidates let a small
    # non-cube yellow plane be rejected by the orthogonality test.
    for _ in range(5):
        if len(remaining_indices) < minimum_count:
            break
        remaining_points = points[remaining_indices]
        fitted = _ransac_plane(
            remaining_points,
            config.ransac_iterations,
            config.plane_inlier_threshold_m,
            rng,
        )
        if fitted is None:
            break
        normal, offset, centroid, local_inliers, rms_m = fitted
        inlier_count = int(np.count_nonzero(local_inliers))
        if inlier_count < minimum_count:
            break
        point_indices = remaining_indices[local_inliers]
        candidates.append(
            PlaneEstimate(
                normal=normal,
                offset_m=offset,
                centroid=centroid,
                point_indices=point_indices,
                rms_m=rms_m,
            )
        )
        remaining_indices = remaining_indices[~local_inliers]
    return tuple(candidates)


def select_orthogonal_faces(
    candidates: tuple[PlaneEstimate, ...],
    tolerance_deg: float,
) -> tuple[PlaneEstimate, PlaneEstimate, PlaneEstimate] | None:
    maximum_dot = math.sin(math.radians(tolerance_deg))
    best: tuple[PlaneEstimate, PlaneEstimate, PlaneEstimate] | None = None
    best_score = -1
    for triple in combinations(candidates, 3):
        pairwise = (
            abs(float(np.dot(triple[0].normal, triple[1].normal))),
            abs(float(np.dot(triple[0].normal, triple[2].normal))),
            abs(float(np.dot(triple[1].normal, triple[2].normal))),
        )
        if max(pairwise) > maximum_dot:
            continue
        score = sum(len(plane.point_indices) for plane in triple)
        if score > best_score:
            best = triple
            best_score = score
    return best


def select_orthogonal_pair(
    candidates: tuple[PlaneEstimate, ...],
    tolerance_deg: float,
) -> tuple[PlaneEstimate, PlaneEstimate] | None:
    maximum_dot = math.sin(math.radians(tolerance_deg))
    best: tuple[PlaneEstimate, PlaneEstimate] | None = None
    best_score = -1
    for pair in combinations(candidates, 2):
        if abs(float(np.dot(pair[0].normal, pair[1].normal))) > maximum_dot:
            continue
        score = len(pair[0].point_indices) + len(pair[1].point_indices)
        if score > best_score:
            best = pair
            best_score = score
    return best


def _orthogonal_rotation_from_normals(normals: np.ndarray) -> np.ndarray:
    approximate = np.column_stack(tuple(normals))
    if np.linalg.det(approximate) < 0.0:
        approximate[:, 2] *= -1.0
    left, _, right_t = np.linalg.svd(approximate)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    return rotation


def choose_equivalent_rotation(
    rotation: np.ndarray,
    previous_rotation: np.ndarray | None,
) -> np.ndarray:
    """Choose the cube-symmetric representation nearest the previous frame."""

    if previous_rotation is None:
        return rotation
    candidates = tuple(rotation @ symmetry for symmetry in CUBE_SYMMETRIES)
    scores = tuple(float(np.trace(previous_rotation.T @ candidate)) for candidate in candidates)
    return candidates[int(np.argmax(scores))]


def cube_pose_from_faces(
    points: np.ndarray,
    faces: tuple[PlaneEstimate, ...],
    side_m: float,
    previous_rotation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Recover cube pose from either three faces or two faces plus their span."""

    if len(faces) not in (2, 3):
        raise ValueError(f"Cube pose requires two or three faces, got {len(faces)}.")
    normals = np.stack(tuple(face.normal for face in faces), axis=0)
    offsets = np.asarray(tuple(face.offset_m for face in faces), dtype=np.float64)
    half_side = 0.5 * side_m

    if len(faces) == 3:
        constraints = normals
        targets = offsets - half_side
        orientation_normals = normals
        observed_span_m = float("nan")
    else:
        edge_direction = np.cross(normals[0], normals[1])
        edge_length = float(np.linalg.norm(edge_direction))
        if edge_length <= 1.0e-8:
            raise ValueError("The two selected face normals are degenerate.")
        edge_direction /= edge_length
        fit_indices = np.unique(
            np.concatenate(tuple(face.point_indices for face in faces))
        )
        edge_coordinates = points[fit_indices] @ edge_direction
        lower, upper = np.quantile(edge_coordinates, (0.01, 0.99))
        observed_span_m = float(upper - lower)
        edge_midpoint = 0.5 * float(lower + upper)
        constraints = np.vstack((normals, edge_direction))
        targets = np.concatenate((offsets - half_side, (edge_midpoint,)))
        orientation_normals = constraints

    if np.linalg.cond(constraints) > 5.0:
        raise ValueError("The selected face geometry is numerically degenerate.")
    center = np.linalg.solve(constraints, targets)
    rotation = _orthogonal_rotation_from_normals(orientation_normals)
    rotation = choose_equivalent_rotation(rotation, previous_rotation)
    return center, rotation, observed_span_m


def two_face_span_is_valid(
    observed_span_m: float,
    config: CubeTrackerConfig,
) -> bool:
    minimum = config.two_face_minimum_span_fraction * config.cube_side_m
    maximum = config.two_face_maximum_span_fraction * config.cube_side_m
    return bool(minimum <= observed_span_m <= maximum)


def point_to_cube_surface_distance(
    points: np.ndarray,
    center_m: np.ndarray,
    rotation: np.ndarray,
    side_m: float,
) -> np.ndarray:
    """Unsigned distance to the surface of a closed oriented cube."""

    local = (points - center_m) @ rotation
    relative = np.abs(local) - 0.5 * side_m
    outside = np.linalg.norm(np.maximum(relative, 0.0), axis=1)
    inside = np.minimum(np.max(relative, axis=1), 0.0)
    return np.abs(outside + inside)


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Return an x, y, z, w unit quaternion."""

    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ),
            dtype=np.float64,
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                (
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                )
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                (
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                )
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                (
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                )
            )
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


class KnownCubeTracker:
    """Independent per-frame measurement with temporal symmetry selection only."""

    def __init__(self, camera: CameraModel, config: CubeTrackerConfig):
        camera.validate()
        config.validate()
        self.camera = camera
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self.previous_rotation: np.ndarray | None = None

    def _invalid(
        self,
        started: float,
        reason: str,
        mask: np.ndarray,
        *,
        candidate_plane_count: int = 0,
        yellow_pixels: int = 0,
        sampled_mask_pixels: int = 0,
        valid_depth_points: int = 0,
        depth_coverage: float = 0.0,
    ) -> CubeTrackingResult:
        return CubeTrackingResult(
            valid=False,
            reason=reason,
            cube_side_m=self.config.cube_side_m,
            mask=mask,
            center_m=None,
            rotation=None,
            quaternion_xyzw=None,
            candidate_plane_count=candidate_plane_count,
            face_count=0,
            yellow_pixels=yellow_pixels,
            sampled_mask_pixels=sampled_mask_pixels,
            valid_depth_points=valid_depth_points,
            depth_coverage=depth_coverage,
            fit_point_count=0,
            surface_rms_m=float("nan"),
            observed_span_m=float("nan"),
            processing_ms=(time.perf_counter() - started) * 1000.0,
            face_pixels=(),
        )

    def track(self, bgr: np.ndarray, depth: np.ndarray) -> CubeTrackingResult:
        started = time.perf_counter()
        mask = segment_yellow(bgr, self.config)
        yellow_pixels = int(np.count_nonzero(mask))
        if yellow_pixels == 0:
            return self._invalid(
                started,
                "no sufficiently large yellow component",
                mask,
            )

        points, pixels, sampled_count, valid_count = unproject_masked_depth(
            mask,
            depth,
            self.camera,
            self.config,
            self.rng,
        )
        coverage = float(valid_count / sampled_count) if sampled_count else 0.0
        minimum_cloud_points = 2 * self.config.minimum_face_points
        if len(points) < minimum_cloud_points:
            return self._invalid(
                started,
                f"only {len(points)} usable depth points; need {minimum_cloud_points}",
                mask,
                yellow_pixels=yellow_pixels,
                sampled_mask_pixels=sampled_count,
                valid_depth_points=valid_count,
                depth_coverage=coverage,
            )

        candidates = extract_plane_candidates(points, self.config, self.rng)
        three_faces = select_orthogonal_faces(
            candidates,
            self.config.orthogonality_tolerance_deg,
        )
        faces: tuple[PlaneEstimate, ...] | None = three_faces
        if faces is None:
            faces = select_orthogonal_pair(
                candidates,
                self.config.orthogonality_tolerance_deg,
            )
        if faces is None:
            return self._invalid(
                started,
                f"no orthogonal face pair among {len(candidates)} planes",
                mask,
                candidate_plane_count=len(candidates),
                yellow_pixels=yellow_pixels,
                sampled_mask_pixels=sampled_count,
                valid_depth_points=valid_count,
                depth_coverage=coverage,
            )

        try:
            center, rotation, observed_span = cube_pose_from_faces(
                points,
                faces,
                self.config.cube_side_m,
                self.previous_rotation,
            )
        except ValueError as exc:
            return self._invalid(
                started,
                str(exc),
                mask,
                candidate_plane_count=len(candidates),
                yellow_pixels=yellow_pixels,
                sampled_mask_pixels=sampled_count,
                valid_depth_points=valid_count,
                depth_coverage=coverage,
            )

        if len(faces) == 2:
            minimum_span = (
                self.config.two_face_minimum_span_fraction
                * self.config.cube_side_m
            )
            maximum_span = (
                self.config.two_face_maximum_span_fraction
                * self.config.cube_side_m
            )
            if not two_face_span_is_valid(observed_span, self.config):
                return self._invalid(
                    started,
                    f"two-face span {observed_span * 1000.0:.1f} mm is outside "
                    f"[{minimum_span * 1000.0:.1f}, "
                    f"{maximum_span * 1000.0:.1f}] mm",
                    mask,
                    candidate_plane_count=len(candidates),
                    yellow_pixels=yellow_pixels,
                    sampled_mask_pixels=sampled_count,
                    valid_depth_points=valid_count,
                    depth_coverage=coverage,
                )

        fit_indices = np.unique(
            np.concatenate(tuple(face.point_indices for face in faces))
        )
        fit_points = points[fit_indices]
        surface_distance = point_to_cube_surface_distance(
            fit_points,
            center,
            rotation,
            self.config.cube_side_m,
        )
        surface_rms = float(np.sqrt(np.mean(np.square(surface_distance))))
        maximum_rms = 1.5 * self.config.plane_inlier_threshold_m
        if not np.isfinite(surface_rms) or surface_rms > maximum_rms:
            return self._invalid(
                started,
                f"cube surface RMS {surface_rms * 1000.0:.1f} mm exceeds "
                f"{maximum_rms * 1000.0:.1f} mm",
                mask,
                candidate_plane_count=len(candidates),
                yellow_pixels=yellow_pixels,
                sampled_mask_pixels=sampled_count,
                valid_depth_points=valid_count,
                depth_coverage=coverage,
            )

        self.previous_rotation = rotation.copy()
        face_pixels = tuple(pixels[face.point_indices] for face in faces)
        if len(faces) == 3:
            reason = "three orthogonal faces fitted"
        else:
            reason = (
                f"two orthogonal faces and {observed_span * 1000.0:.1f} mm "
                "shared-edge span fitted"
            )
        return CubeTrackingResult(
            valid=True,
            reason=reason,
            cube_side_m=self.config.cube_side_m,
            mask=mask,
            center_m=center,
            rotation=rotation,
            quaternion_xyzw=rotation_matrix_to_quaternion(rotation),
            candidate_plane_count=len(candidates),
            face_count=len(faces),
            yellow_pixels=yellow_pixels,
            sampled_mask_pixels=sampled_count,
            valid_depth_points=valid_count,
            depth_coverage=coverage,
            fit_point_count=len(fit_points),
            surface_rms_m=surface_rms,
            observed_span_m=observed_span,
            processing_ms=(time.perf_counter() - started) * 1000.0,
            face_pixels=face_pixels,
        )


def project_points(points: np.ndarray, camera: CameraModel) -> np.ndarray:
    """Project ZED Y-up camera points into the rectified left image."""

    values = np.asarray(points, dtype=np.float64)
    forward_depth = -values[:, 2]
    projected = np.full((len(values), 2), np.nan, dtype=np.float64)
    valid = forward_depth > 1.0e-6
    projected[valid, 0] = (
        camera.fx * values[valid, 0] / forward_depth[valid] + camera.cx
    )
    projected[valid, 1] = (
        camera.cy - camera.fy * values[valid, 1] / forward_depth[valid]
    )
    return projected


def draw_cube_wireframe_overlay(
    bgr: np.ndarray,
    result: CubeTrackingResult,
    camera: CameraModel,
    cube_side_m: float,
) -> np.ndarray:
    """Draw the observed yellow boundary and raw cube wireframe on one RGB view."""

    output = np.asarray(bgr, dtype=np.uint8)[:, :, :3].copy()
    mask = np.asarray(result.mask, dtype=np.uint8)
    if mask.shape != output.shape[:2]:
        mask = cv2.resize(
            mask,
            (output.shape[1], output.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    if np.any(mask):
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(output, contours, -1, (0, 220, 255), 2, cv2.LINE_AA)

    if not result.valid or result.center_m is None or result.rotation is None:
        return output
    half_side = 0.5 * cube_side_m
    camera_corners = result.center_m + (half_side * CUBE_CORNERS) @ result.rotation.T
    projected = project_points(camera_corners, camera)
    for first, second in CUBE_EDGES:
        if not np.all(np.isfinite(projected[[first, second]])):
            continue
        first_xy = tuple(np.rint(projected[first]).astype(int))
        second_xy = tuple(np.rint(projected[second]).astype(int))
        cv2.line(output, first_xy, second_xy, (220, 60, 255), 3, cv2.LINE_AA)
    return output


def draw_tracking_result(
    bgr: np.ndarray,
    result: CubeTrackingResult,
    camera: CameraModel,
    cube_side_m: float,
) -> np.ndarray:
    """Render the raw measurement without any held or predicted pose."""

    output = np.asarray(bgr, dtype=np.uint8)[:, :, :3].copy()
    selected = result.mask != 0
    if np.any(selected):
        tint = np.empty_like(output)
        tint[:] = (0, 220, 255)
        output[selected] = cv2.addWeighted(
            output[selected],
            0.72,
            tint[selected],
            0.28,
            0.0,
        )
        contours, _ = cv2.findContours(
            result.mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(output, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)

    for pixels, colour in zip(result.face_pixels, FACE_COLOURS):
        step = max(1, len(pixels) // 500)
        for x, y in pixels[::step]:
            cv2.circle(output, (int(x), int(y)), 1, colour, -1, cv2.LINE_AA)

    if result.valid and result.center_m is not None and result.rotation is not None:
        half_side = 0.5 * cube_side_m
        camera_corners = result.center_m + (half_side * CUBE_CORNERS) @ result.rotation.T
        projected = project_points(camera_corners, camera)
        for first, second in CUBE_EDGES:
            if np.all(np.isfinite(projected[[first, second]])):
                first_xy = tuple(np.rint(projected[first]).astype(int))
                second_xy = tuple(np.rint(projected[second]).astype(int))
                cv2.line(output, first_xy, second_xy, (40, 255, 40), 2, cv2.LINE_AA)

        axis_length = 0.45 * cube_side_m
        axes = np.vstack(
            (
                result.center_m,
                result.center_m + axis_length * result.rotation[:, 0],
                result.center_m + axis_length * result.rotation[:, 1],
                result.center_m + axis_length * result.rotation[:, 2],
            )
        )
        projected_axes = project_points(axes, camera)
        if np.all(np.isfinite(projected_axes)):
            origin = tuple(np.rint(projected_axes[0]).astype(int))
            for axis_index, colour in enumerate(
                ((40, 40, 255), (40, 255, 40), (255, 80, 40)),
                start=1,
            ):
                endpoint = tuple(np.rint(projected_axes[axis_index]).astype(int))
                cv2.line(output, origin, endpoint, colour, 3, cv2.LINE_AA)

    panel_width = min(690, output.shape[1] - 20)
    panel_height = 145
    overlay = output.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (0, 0, 0), -1)
    output = cv2.addWeighted(overlay, 0.72, output, 0.28, 0.0)
    status_colour = (80, 240, 80) if result.valid else (70, 70, 255)
    status = (
        f"VALID {result.face_count}-FACE CUBE POSE"
        if result.valid
        else "INVALID MEASUREMENT"
    )
    lines = [
        status,
        result.reason,
        (
            f"faces={result.face_count}  candidates={result.candidate_plane_count}  "
            f"depth={100.0 * result.depth_coverage:.1f}%  "
            f"points={result.valid_depth_points}"
        ),
    ]
    if result.valid and result.center_m is not None:
        span_text = (
            f"  span={result.observed_span_m * 1000.0:.1f} mm"
            if math.isfinite(result.observed_span_m)
            else ""
        )
        lines.append(
            f"surface RMS={result.surface_rms_m * 1000.0:.2f} mm  "
            f"{span_text}"
            f"  center ZED xyz=({result.center_m[0]:+.3f}, "
            f"{result.center_m[1]:+.3f}, {result.center_m[2]:+.3f}) m  "
            f"time={result.processing_ms:.1f} ms"
        )
    else:
        lines.append(f"time={result.processing_ms:.1f} ms")

    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line[:110],
            (24, 38 + 32 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            status_colour if index == 0 else (245, 245, 245),
            2 if index == 0 else 1,
            cv2.LINE_AA,
        )
    cv2.putText(
        output,
        "Q/Esc: quit    S: save snapshot",
        (24, output.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output
