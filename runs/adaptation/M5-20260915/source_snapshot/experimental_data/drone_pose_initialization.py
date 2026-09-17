"""Causal pose initialization at the measured tracking origin, without dynamics fitting.

The result is timestamped at the last measurement before cutoff. It is never
extrapolated to the maneuver onset, and does not infer controller integral state
or firmware body axes. Callers supply the archived rigid attachment offset.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from simulator.geometry import attachment_positions, normalized_rotations_xyzw


def initialize_drone_pose(time_s, position_m, quaternion_xyzw, *, cutoff_s,
                          offset_tracking_m, derivative_samples=11,
                          maximum_age_s=.03, maximum_gap_s=.02):
    """Use the latest contiguous past samples for P/R and endpoint V/omega.

    P and R are the last measured values. Velocity is an endpoint quadratic
    estimate on actual timestamps. Angular velocity uses a local rotation log
    in the final tracked frame; quaternion sign changes do not change it.
    All samples at/after cutoff are excluded before pose validation or fitting.
    """
    t=np.asarray(time_s,dtype=float)
    p=np.asarray(position_m,dtype=float)
    q=np.asarray(quaternion_xyzw,dtype=float)
    if t.ndim!=1 or p.shape!=(len(t),3) or q.shape!=(len(t),4):
        raise ValueError('Expected T timestamps, Tx3 tracked positions and Tx4 XYZW quaternions')
    if (not np.isfinite(cutoff_s) or derivative_samples<3 or isinstance(derivative_samples,bool)
            or int(derivative_samples)!=derivative_samples):
        raise ValueError('Need a finite cutoff and at least three derivative samples')
    if not np.isfinite([maximum_age_s,maximum_gap_s]).all() or min(maximum_age_s,maximum_gap_s)<=0:
        raise ValueError('Age and gap limits must be positive and finite')
    before=np.flatnonzero(t<cutoff_s)
    n=int(derivative_samples)
    if len(before)<n:
        raise ValueError('Insufficient pre-cutoff pose history')
    ids=before[-n:]
    tt=t[ids]
    if (np.any(np.diff(ids)!=1) or not np.isfinite(tt).all()
            or np.any(np.diff(tt)<=0) or np.any(np.diff(tt)>maximum_gap_s)):
        raise ValueError('Pose history has a timestamp gap or non-increasing timestamps')
    if cutoff_s-tt[-1]>maximum_age_s:
        raise ValueError('Last pre-cutoff pose is stale')
    rotations,valid=normalized_rotations_xyzw(q[ids])
    if not valid.all() or not np.isfinite(p[ids]).all():
        raise ValueError('Recent pose history contains invalid positions or orientations')
    duration=tt[-1]-tt[0]
    x=(tt-tt[-1])/duration
    design=np.column_stack([np.ones(n),x,x*x])
    weights=np.linalg.pinv(design)[1]/duration
    velocity=weights@p[ids]
    relative=np.einsum('ij,tjk->tik',rotations[-1].T,rotations)
    rotation_log=Rotation.from_matrix(relative).as_rotvec()
    if np.max(np.linalg.norm(rotation_log,axis=1))>=np.pi/2:
        raise ValueError('Orientation history exceeds local derivative range; shorten window')
    omega_tracking=weights@rotation_log
    normalized_q=q[ids[-1]]/np.linalg.norm(q[ids[-1]])
    attachment,attachment_valid=attachment_positions(p[ids[-1]],normalized_q,offset_tracking_m)
    if not attachment_valid:
        raise ValueError('Invalid attachment geometry')
    attachment_velocity=velocity+rotations[-1]@np.cross(omega_tracking,np.asarray(offset_tracking_m))
    return dict(time_s=float(tt[-1]),cutoff_s=float(cutoff_s),sample_indices=ids,
        position_origin_m=p[ids[-1]].copy(),velocity_origin_m_s=velocity,
        orientation_xyzw=normalized_q,rotation_tracking_to_world=rotations[-1],
        omega_tracking_rad_s=omega_tracking,position_attachment_m=attachment,
        velocity_attachment_m_s=attachment_velocity,history_duration_s=float(duration),
        controller_memory_observed=False,reference_point='OptiTrack cf_7 origin',
        angular_frame='OptiTrack tracked rigid-body frame T; not verified firmware body axes')
