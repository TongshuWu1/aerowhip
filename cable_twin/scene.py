"""Immutable CPU contracts for the Phase-1 geometric scene twin.

Geometry is expressed in the rectified left-camera frame described by the
associated :class:`CameraCalibration`.  These contracts contain estimates,
not rendering state: an absent cable/cube is represented by an empty cable
tuple or ``cube=None`` rather than by fabricated geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Self

import numpy as np

from .frames import CameraCalibration, FrameKey, RgbdFrame


def _require_array(
    name: str,
    value: object,
    *,
    dtype: np.dtype,
    shape_tail: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a CPU numpy.ndarray")
    if value.dtype != dtype:
        raise ValueError(f"{name} must have dtype {np.dtype(dtype).name}")
    expected_ndim = 1 + len(shape_tail)
    if value.ndim != expected_ndim or value.shape[1:] != shape_tail:
        suffix = "x".join(str(size) for size in shape_tail)
        raise ValueError(f"{name} must have shape Nx{suffix}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _owned_readonly(value: np.ndarray) -> np.ndarray:
    owned = np.array(value, dtype=value.dtype, order="C", copy=True)
    owned.setflags(write=False)
    return owned


def _require_finite(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def _require_image(
    name: str,
    value: object,
    *,
    dtype: np.dtype,
    channels: int | None = None,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a CPU numpy.ndarray")
    expected_ndim = 2 if channels is None else 3
    if value.dtype != dtype or value.ndim != expected_ndim:
        kind = "HxW" if channels is None else f"HxWx{channels}"
        raise ValueError(f"{name} must be a {np.dtype(dtype).name} {kind} array")
    if channels is not None and value.shape[2] != channels:
        raise ValueError(f"{name} must have exactly {channels} channels")
    return _owned_readonly(value)


def _validate_mesh(
    vertices_name: str,
    vertices: np.ndarray,
    triangles_name: str,
    triangles: np.ndarray,
) -> None:
    """Validate an explicit empty mesh or an indexed triangle mesh."""

    _require_finite(vertices_name, vertices)
    if vertices.shape[0] == 0 or triangles.shape[0] == 0:
        if vertices.shape[0] != 0 or triangles.shape[0] != 0:
            raise ValueError(
                f"{vertices_name} and {triangles_name} must both be empty or "
                "both define a mesh"
            )
        return
    if vertices.shape[0] < 3:
        raise ValueError(f"{vertices_name} must contain at least three vertices")
    if int(triangles.min()) < 0 or int(triangles.max()) >= vertices.shape[0]:
        raise ValueError(f"{triangles_name} contains an out-of-range vertex index")
    if np.any(
        (triangles[:, 0] == triangles[:, 1])
        | (triangles[:, 1] == triangles[:, 2])
        | (triangles[:, 0] == triangles[:, 2])
    ):
        raise ValueError(f"{triangles_name} contains a degenerate index triplet")


def _validate_covariance(name: str, value: np.ndarray) -> None:
    _require_finite(name, value)
    if not np.allclose(value, np.swapaxes(value, -1, -2), rtol=1e-5, atol=1e-8):
        raise ValueError(f"{name} must be symmetric")
    if np.any(np.linalg.eigvalsh(value) < -1e-9):
        raise ValueError(f"{name} must be positive semidefinite")


def _validate_scene_calibration(calibration: CameraCalibration) -> None:
    if calibration.coordinate_system != "RIGHT_HANDED_Y_UP":
        raise ValueError("scene geometry requires RIGHT_HANDED_Y_UP coordinates")
    if calibration.depth_unit != "METER":
        raise ValueError("scene geometry requires metric depth")


@dataclass(frozen=True, slots=True)
class CableGeometry:
    """One ordered cable centerline and its fixed-topology physical surface.

    ``node_observed_bool`` is optional by design.  When present, ``True`` means
    that the node is directly supported by current RGB-D evidence; ``False``
    means it is inferred or temporally predicted.  A missing array means that
    observability has not been estimated.  Node covariance uses metric 3-D
    position coordinates in the camera frame.

    Mesh arrays may both be explicitly empty while centerline reconstruction
    is being developed.  Once populated, triangle indices are intended to stay
    fixed across frames while only vertex positions change.
    """

    cable_id: int
    radius_m: float
    centerline_camera_m_f32: np.ndarray
    mesh_vertices_camera_m_f32: np.ndarray
    mesh_triangles_i32: np.ndarray
    node_observed_bool: np.ndarray | None = None
    node_covariance_m2_f32: np.ndarray | None = None

    def __post_init__(self) -> None:
        if isinstance(self.cable_id, bool) or not isinstance(self.cable_id, int):
            raise TypeError("cable_id must be an integer")
        if self.cable_id < 0:
            raise ValueError("cable_id must be nonnegative")
        if not math.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError("radius_m must be finite and positive")

        centerline = _require_array(
            "centerline_camera_m_f32",
            self.centerline_camera_m_f32,
            dtype=np.dtype(np.float32),
            shape_tail=(3,),
        )
        if centerline.shape[0] < 2:
            raise ValueError("a cable centerline must contain at least two nodes")
        _require_finite("centerline_camera_m_f32", centerline)

        vertices = _require_array(
            "mesh_vertices_camera_m_f32",
            self.mesh_vertices_camera_m_f32,
            dtype=np.dtype(np.float32),
            shape_tail=(3,),
        )
        triangles = _require_array(
            "mesh_triangles_i32",
            self.mesh_triangles_i32,
            dtype=np.dtype(np.int32),
            shape_tail=(3,),
        )
        _validate_mesh(
            "mesh_vertices_camera_m_f32",
            vertices,
            "mesh_triangles_i32",
            triangles,
        )

        observed: np.ndarray | None = None
        if self.node_observed_bool is not None:
            observed = _require_array(
                "node_observed_bool",
                self.node_observed_bool,
                dtype=np.dtype(np.bool_),
                shape_tail=(),
            )
            if observed.shape != (centerline.shape[0],):
                raise ValueError("node_observed_bool must contain one value per node")

        covariance: np.ndarray | None = None
        if self.node_covariance_m2_f32 is not None:
            covariance = _require_array(
                "node_covariance_m2_f32",
                self.node_covariance_m2_f32,
                dtype=np.dtype(np.float32),
                shape_tail=(3, 3),
            )
            if covariance.shape[0] != centerline.shape[0]:
                raise ValueError(
                    "node_covariance_m2_f32 must contain one 3x3 matrix per node"
                )
            _validate_covariance("node_covariance_m2_f32", covariance)

        object.__setattr__(
            self,
            "centerline_camera_m_f32",
            _owned_readonly(centerline),
        )
        object.__setattr__(
            self,
            "mesh_vertices_camera_m_f32",
            _owned_readonly(vertices),
        )
        object.__setattr__(self, "mesh_triangles_i32", _owned_readonly(triangles))
        if observed is not None:
            object.__setattr__(self, "node_observed_bool", _owned_readonly(observed))
        if covariance is not None:
            object.__setattr__(
                self,
                "node_covariance_m2_f32",
                _owned_readonly(covariance),
            )


@dataclass(frozen=True, slots=True)
class CubeGeometry:
    """Known cube surface at an estimated camera-frame pose.

    ``pose_camera_from_cube_f64`` maps cube-local homogeneous coordinates into
    the rectified left-camera frame.  The optional 6x6 covariance follows the
    local ordering ``[tx, ty, tz, rx, ry, rz]`` (metres, radians).  It is absent
    until the pose estimator can produce a statistically meaningful value.

    Cube-local mesh topology can be created once and shared across frames; only
    the rigid pose then changes.  Both mesh arrays may be explicitly empty for
    a pose-only development stage.
    """

    pose_camera_from_cube_f64: np.ndarray
    mesh_vertices_cube_m_f32: np.ndarray
    mesh_triangles_i32: np.ndarray
    pose_covariance_f64: np.ndarray | None = None

    def __post_init__(self) -> None:
        pose = _require_array(
            "pose_camera_from_cube_f64",
            self.pose_camera_from_cube_f64,
            dtype=np.dtype(np.float64),
            shape_tail=(4,),
        )
        if pose.shape != (4, 4):
            raise ValueError("pose_camera_from_cube_f64 must have shape 4x4")
        _require_finite("pose_camera_from_cube_f64", pose)
        if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9, rtol=0.0):
            raise ValueError("pose_camera_from_cube_f64 must be homogeneous")
        rotation = pose[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0):
            raise ValueError("cube pose rotation must be orthonormal")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
            raise ValueError("cube pose rotation must have determinant +1")

        vertices = _require_array(
            "mesh_vertices_cube_m_f32",
            self.mesh_vertices_cube_m_f32,
            dtype=np.dtype(np.float32),
            shape_tail=(3,),
        )
        triangles = _require_array(
            "mesh_triangles_i32",
            self.mesh_triangles_i32,
            dtype=np.dtype(np.int32),
            shape_tail=(3,),
        )
        _validate_mesh(
            "mesh_vertices_cube_m_f32",
            vertices,
            "mesh_triangles_i32",
            triangles,
        )

        covariance: np.ndarray | None = None
        if self.pose_covariance_f64 is not None:
            covariance = _require_array(
                "pose_covariance_f64",
                self.pose_covariance_f64,
                dtype=np.dtype(np.float64),
                shape_tail=(6,),
            )
            if covariance.shape != (6, 6):
                raise ValueError("pose_covariance_f64 must have shape 6x6")
            _validate_covariance("pose_covariance_f64", covariance)

        object.__setattr__(
            self,
            "pose_camera_from_cube_f64",
            _owned_readonly(pose),
        )
        object.__setattr__(
            self,
            "mesh_vertices_cube_m_f32",
            _owned_readonly(vertices),
        )
        object.__setattr__(self, "mesh_triangles_i32", _owned_readonly(triangles))
        if covariance is not None:
            object.__setattr__(
                self,
                "pose_covariance_f64",
                _owned_readonly(covariance),
            )


@dataclass(frozen=True, slots=True)
class SceneGeometryFrame:
    """All available geometric estimates associated with exactly one RGB-D key."""

    key: FrameKey
    calibration: CameraCalibration
    cables: tuple[CableGeometry, ...] = ()
    cube: CubeGeometry | None = None
    reconstruction_ms: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, FrameKey):
            raise TypeError("key must be a FrameKey")
        if not isinstance(self.calibration, CameraCalibration):
            raise TypeError("calibration must be a CameraCalibration")
        _validate_scene_calibration(self.calibration)
        if not isinstance(self.cables, tuple):
            raise TypeError("cables must be a tuple")
        if any(not isinstance(cable, CableGeometry) for cable in self.cables):
            raise TypeError("cables must contain only CableGeometry values")
        cable_ids = tuple(cable.cable_id for cable in self.cables)
        if len(cable_ids) != len(set(cable_ids)):
            raise ValueError("cable identifiers must be unique within a frame")
        if self.cube is not None and not isinstance(self.cube, CubeGeometry):
            raise TypeError("cube must be CubeGeometry or None")
        if self.reconstruction_ms is not None and (
            not math.isfinite(self.reconstruction_ms) or self.reconstruction_ms < 0.0
        ):
            raise ValueError("reconstruction_ms must be finite and nonnegative")

    @classmethod
    def for_rgbd(
        cls,
        rgbd: RgbdFrame,
        calibration: CameraCalibration,
        *,
        cables: tuple[CableGeometry, ...] = (),
        cube: CubeGeometry | None = None,
        reconstruction_ms: float | None = None,
    ) -> Self:
        """Create a scene frame after checking calibration/image registration."""

        _validate_rgbd_calibration(rgbd, calibration)
        return cls(
            key=rgbd.key,
            calibration=calibration,
            cables=cables,
            cube=cube,
            reconstruction_ms=reconstruction_ms,
        )


@dataclass(frozen=True, slots=True)
class SceneViewerSnapshot:
    """Reduced registered RGB-D evidence and optional estimated geometry.

    The sampled arrays are the exact ``[::sample_stride_px,
    ::sample_stride_px]`` subset of their source frame.  Consequently the
    original pixel for sample ``(row, column)`` is ``(row*stride,
    column*stride)`` and can be deprojected with the original calibration.
    When present, mask order is cable body, endpoint set 1, endpoint set 2.
    """

    key: FrameKey
    calibration: CameraCalibration
    sampled_bgr_u8: np.ndarray
    sampled_depth_m_f32: np.ndarray
    sample_stride_px: int
    masks_u8: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    geometry: SceneGeometryFrame | None = None
    status_lines: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.key, FrameKey):
            raise TypeError("key must be a FrameKey")
        if not isinstance(self.calibration, CameraCalibration):
            raise TypeError("calibration must be a CameraCalibration")
        _validate_scene_calibration(self.calibration)
        if isinstance(self.sample_stride_px, bool) or not isinstance(
            self.sample_stride_px, int
        ):
            raise TypeError("sample_stride_px must be an integer")
        if self.sample_stride_px <= 0:
            raise ValueError("sample_stride_px must be positive")

        bgr = _require_image(
            "sampled_bgr_u8",
            self.sampled_bgr_u8,
            dtype=np.dtype(np.uint8),
            channels=3,
        )
        depth = _require_image(
            "sampled_depth_m_f32",
            self.sampled_depth_m_f32,
            dtype=np.dtype(np.float32),
        )
        expected_shape = (
            (self.calibration.height_px + self.sample_stride_px - 1)
            // self.sample_stride_px,
            (self.calibration.width_px + self.sample_stride_px - 1)
            // self.sample_stride_px,
        )
        if bgr.shape[:2] != expected_shape or depth.shape != expected_shape:
            raise ValueError(
                "sampled RGB and depth dimensions do not match calibration/stride"
            )

        owned_masks: list[np.ndarray] | None = None
        if self.masks_u8 is not None:
            if not isinstance(self.masks_u8, tuple) or len(self.masks_u8) != 3:
                raise ValueError("masks_u8 must be None or a tuple of three masks")
            owned_masks = []
            for index, mask in enumerate(self.masks_u8):
                value = _require_image(
                    f"masks_u8[{index}]",
                    mask,
                    dtype=np.dtype(np.uint8),
                )
                if value.shape != expected_shape:
                    raise ValueError(
                        f"masks_u8[{index}] must match the sampled image dimensions"
                    )
                owned_masks.append(value)

        if self.geometry is not None:
            if not isinstance(self.geometry, SceneGeometryFrame):
                raise TypeError("geometry must be SceneGeometryFrame or None")
            if self.geometry.key != self.key:
                raise ValueError("geometry and viewer evidence must have the same key")
            if self.geometry.calibration != self.calibration:
                raise ValueError(
                    "geometry and viewer evidence must use the same calibration"
                )

        if not isinstance(self.status_lines, tuple) or any(
            not isinstance(line, str) for line in self.status_lines
        ):
            raise TypeError("status_lines must be a tuple of strings")

        object.__setattr__(self, "sampled_bgr_u8", bgr)
        object.__setattr__(self, "sampled_depth_m_f32", depth)
        if owned_masks is not None:
            object.__setattr__(self, "masks_u8", tuple(owned_masks))

    @classmethod
    def from_rgbd(
        cls,
        rgbd: RgbdFrame,
        calibration: CameraCalibration,
        *,
        sample_stride_px: int = 4,
        masks_u8: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        geometry: SceneGeometryFrame | None = None,
        status_lines: tuple[str, ...] = (),
    ) -> Self:
        """Copy the registered regular sample needed by an asynchronous viewer."""

        _validate_rgbd_calibration(rgbd, calibration)
        if isinstance(sample_stride_px, bool) or not isinstance(sample_stride_px, int):
            raise TypeError("sample_stride_px must be an integer")
        if sample_stride_px <= 0:
            raise ValueError("sample_stride_px must be positive")

        sampled_masks: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        if masks_u8 is not None:
            if not isinstance(masks_u8, tuple) or len(masks_u8) != 3:
                raise ValueError("masks_u8 must be None or a tuple of three masks")
            checked_masks: list[np.ndarray] = []
            for index, mask in enumerate(masks_u8):
                if not isinstance(mask, np.ndarray):
                    raise TypeError(f"masks_u8[{index}] must be a CPU numpy.ndarray")
                if mask.dtype != np.uint8 or mask.shape != rgbd.depth_m_f32.shape:
                    raise ValueError(
                        f"masks_u8[{index}] must be uint8 and aligned with RGB-D"
                    )
                checked_masks.append(
                    mask[::sample_stride_px, ::sample_stride_px]
                )
            sampled_masks = tuple(checked_masks)  # type: ignore[assignment]

        return cls(
            key=rgbd.key,
            calibration=calibration,
            sampled_bgr_u8=rgbd.bgr_u8[::sample_stride_px, ::sample_stride_px],
            sampled_depth_m_f32=rgbd.depth_m_f32[
                ::sample_stride_px,
                ::sample_stride_px,
            ],
            sample_stride_px=sample_stride_px,
            masks_u8=sampled_masks,
            geometry=geometry,
            status_lines=status_lines,
        )


def _validate_rgbd_calibration(
    rgbd: RgbdFrame,
    calibration: CameraCalibration,
) -> None:
    if not isinstance(rgbd, RgbdFrame):
        raise TypeError("rgbd must be an RgbdFrame")
    if not isinstance(calibration, CameraCalibration):
        raise TypeError("calibration must be a CameraCalibration")
    _validate_scene_calibration(calibration)
    expected = (calibration.height_px, calibration.width_px)
    if rgbd.depth_m_f32.shape != expected:
        raise ValueError("RGB-D dimensions do not match camera calibration")


__all__ = [
    "CableGeometry",
    "CubeGeometry",
    "SceneGeometryFrame",
    "SceneViewerSnapshot",
]
