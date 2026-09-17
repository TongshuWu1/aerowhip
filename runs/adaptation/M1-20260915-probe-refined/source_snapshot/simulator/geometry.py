"""Geometry in SI units, independent of drone/cable dynamics and fitting.

T is the OptiTrack rigid-body frame at the top tracking origin; W is its
recording's world frame. T is not assumed to equal the firmware body frame.
XYZW quaternions rotate T vectors into W. The cable's first flexible span is
not part of the rigid offset. No center-of-mass location is inferred here.
"""
import numpy as np


def normalized_rotations_xyzw(quaternion):
    """Return active T-to-W rotations and a validity mask; retain invalids as NaN."""
    q = np.asarray(quaternion, dtype=np.float64)
    if q.ndim < 1 or q.shape[-1] != 4:
        raise ValueError('Expected XYZW quaternions with last dimension 4')
    norm = np.linalg.norm(q, axis=-1)
    valid = np.isfinite(q).all(axis=-1) & np.isfinite(norm) & (norm > 1e-12)
    unit = np.full_like(q, np.nan)
    np.divide(q, norm[..., None], out=unit, where=valid[..., None])
    x, y, z, w = np.moveaxis(unit, -1, 0)
    rotation = np.stack((
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=-1)
    return rotation.reshape(q.shape[:-1]+(3, 3)), valid


def _offset(value):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError('Expected one finite tracking-origin-to-attachment XYZ offset in meters')
    return value


def attachment_positions(tracked_position, tracking_to_world_xyzw, offset_tracking_m):
    """Measured/predicted positions; caller supplies the corresponding orientation."""
    p = np.asarray(tracked_position, dtype=np.float64)
    rotation, valid = normalized_rotations_xyzw(tracking_to_world_xyzw)
    if p.shape != rotation.shape[:-2]+(3,):
        raise ValueError('Tracked positions and orientations must have matching sample shapes')
    position = p + np.einsum('...ij,j->...i', rotation, _offset(offset_tracking_m))
    valid = valid & np.isfinite(p).all(axis=-1)
    return np.where(valid[..., None], position, np.nan), valid


def attachment_kinematics(position, velocity, acceleration, tracking_to_world_xyzw,
                          omega_tracking_rad_s, alpha_tracking_rad_s2, offset_tracking_m):
    """Exact rigid-offset P/V/A with angular rates expressed in tracking frame T.

This is a coordinate operation, not an attitude predictor. Angular data must
be provided explicitly; missing rotation terms are never silently set to zero.
"""
    p, valid = attachment_positions(position, tracking_to_world_xyzw, offset_tracking_m)
    rotation, _ = normalized_rotations_xyzw(tracking_to_world_xyzw)
    vectors = [np.asarray(x, dtype=np.float64) for x in
               (velocity, acceleration, omega_tracking_rad_s, alpha_tracking_rad_s2)]
    if any(x.shape != p.shape for x in vectors):
        raise ValueError('P/V/A and angular derivatives must have matching XYZ shapes')
    v, a, omega, alpha = vectors
    valid &= np.logical_and.reduce([np.isfinite(x).all(axis=-1) for x in vectors])
    r = _offset(offset_tracking_m)
    rotate = lambda x: np.einsum('...ij,...j->...i', rotation, x)
    va = v + rotate(np.cross(omega, r))
    aa = a + rotate(np.cross(alpha, r) + np.cross(omega, np.cross(omega, r)))
    return tuple(np.where(valid[..., None], x, np.nan) for x in (p, va, aa)), valid


def initial_world_offset(offset_tracking_m, tracking_to_world_xyzw):
    """Rotate one initial offset; this does not predict orientation during flight."""
    q = np.asarray(tracking_to_world_xyzw, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError('Expected one initial tracking-to-world XYZW quaternion')
    rotation, valid = normalized_rotations_xyzw(q)
    if not valid:
        raise ValueError('Initial tracking orientation must be finite and nonzero')
    return rotation @ _offset(offset_tracking_m)
