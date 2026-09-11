from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from simulator.geometry import (normalized_rotations_xyzw, attachment_positions,
    attachment_kinematics, initial_world_offset)
from experimental_data.quality import evaluate_quality
from experimental_data.processing import _processing_fingerprint

ROOT = Path(__file__).resolve().parents[2]


def test_rotation_direction_normalization_and_invalid_pose():
    # +90 degrees around world/tracking Y takes downward Z into negative X.
    q = np.array([0., np.sqrt(.5), 0., np.sqrt(.5)])
    expected = [-.055, 0., 0.]
    for scaled in (q, -q, q*3):
        np.testing.assert_allclose(initial_world_offset([0,0,-.055], scaled), expected, atol=1e-15)
    rotations, valid = normalized_rotations_xyzw([q, [0,0,0,0], [np.nan,0,0,1]])
    assert valid.tolist() == [True, False, False]
    assert np.isnan(rotations[1:]).all()
    with pytest.raises(ValueError, match='nonzero'):
        initial_world_offset([0,0,-.055], [0,0,0,0])


def test_rotating_offset_pva_matches_independent_finite_differences():
    t = np.linspace(.1, 1., 15)
    r = np.array([.007,-.014,-.055])
    def trajectory(t):
        angle = .6*t*t
        q = np.column_stack((0*t,np.sin(angle/2),0*t,np.cos(angle/2)))
        p = np.column_stack((t*t, .2*t, 1.5+0*t))
        return attachment_positions(p,q,r)[0]
    angle = .6*t*t
    q = np.column_stack((0*t,np.sin(angle/2),0*t,np.cos(angle/2)))
    p = np.column_stack((t*t,.2*t,1.5+0*t))
    v = np.column_stack((2*t,0*t+.2,0*t))
    a = np.column_stack((0*t+2,0*t,0*t))
    omega = np.column_stack((0*t,1.2*t,0*t))
    alpha = np.column_stack((0*t,0*t+1.2,0*t))
    (pa,va,aa),valid = attachment_kinematics(p,v,a,q,omega,alpha,r)
    h=1e-4
    np.testing.assert_allclose(pa,trajectory(t),atol=1e-14)
    np.testing.assert_allclose(va,(trajectory(t+h)-trajectory(t-h))/(2*h),atol=1e-8)
    np.testing.assert_allclose(aa,(trajectory(t+h)-2*trajectory(t)+trajectory(t-h))/h**2,atol=1e-7)
    assert valid.all()


def test_quality_accepts_bending_rejects_excess_and_zero_quaternion():
    settings=json.loads((ROOT/'experimental_data/default_processing.json').read_text())['quality']
    # Stationary, bent/short-chord first interval is physically possible.
    arrays=dict(time_s=np.arange(5)*.01,uav_valid=np.ones(5,bool),
        cable_marker_valid=np.ones((5,1),bool),uav_position_m=np.tile([0,0,1.5],(5,1)),
        cable_marker_positions_m=np.tile([[[0,0,1.43]]],(5,1,1)),
        uav_orientation_xyzw=np.tile([0,0,0,2.],(5,1)),
        command_valid=np.ones(5,bool),command_age_s=np.zeros(5))
    def check():return evaluate_quality(arrays,quality_config=settings,
        attachment_offset_body_m=(0,0,-.055),interval_lengths_m=(.063,))[0]
    assert check()['auto_frame_valid'].all()
    arrays['uav_orientation_xyzw'][2]=0
    flags=check()
    assert flags['quality_pose_invalid'][2] and not flags['auto_frame_valid'][2]
    arrays['uav_orientation_xyzw'][2]=[0,0,0,1]
    arrays['cable_marker_positions_m'][:]=[0,0,1.3]
    assert check()['quality_geometry_invalid'].all()


def test_quality_cache_depends_on_geometry_but_not_material_fit():
    model=json.loads((ROOT/'config/model.json').read_text())
    original=_processing_fingerprint({}, {}, model)
    changed=deepcopy(model)
    changed['recorded_data']['optitrack_to_attachment_offset_body_m'][2]-=.01
    assert _processing_fingerprint({}, {}, changed)!=original
    changed=deepcopy(model);changed['cable']['marker_interval_lengths_m'][0]+=.01
    assert _processing_fingerprint({}, {}, changed)!=original
    changed=deepcopy(model);changed['cable']['EI_n_m2']*=2
    assert _processing_fingerprint({}, {}, changed)==original
