import csv
import json
import numpy as np
import pytest

from deployment.gentle_recovery import plan_recovery, replace_recorded_recovery
from deployment.fullstate import write_rehearsal_fullstate
from deployment.fullstate_playback import load_reference


def test_analytic_pva_and_boundaries_are_continuous():
    initial = ([.49,.006,2.196],[-1.876,.009,.243],[-5.69,.003,-5.58])
    sample, meta = plan_recovery(*initial,[0,0,1.5])
    for got,wanted in zip(sample(np.array([0.])),initial):
        np.testing.assert_allclose(got[0],wanted,atol=1e-12)
    boundaries = [meta['transition_s'],*meta['axis_plateau_end_times_s'],
                  *meta['axis_stop_times_s'],meta['return_end_s']]
    for boundary in boundaries:
        left,right = sample(np.array([boundary-1e-7])),sample(np.array([boundary+1e-7]))
        for a,b in zip(left,right):
            np.testing.assert_allclose(a,b,atol=2e-6)
    t=np.linspace(.01,meta['total_duration_s']-.01,137)
    dt=1e-5
    p,v,a=sample(t)
    left,right=sample(t-dt),sample(t+dt)
    np.testing.assert_allclose((right[0]-left[0])/(2*dt),v,atol=2e-7)
    np.testing.assert_allclose((right[1]-left[1])/(2*dt),a,atol=2e-7)
    p,v,a=sample(np.linspace(meta['brake_end_s'],meta['return_end_s'],1001))
    assert np.linalg.norm(a,axis=1).max()<=.3+1e-9
    assert np.linalg.norm(v,axis=1).max()<=.4+1e-9
    p,v,a=sample(np.array([meta['total_duration_s']]))
    np.testing.assert_array_equal(p,[[0,0,1.5]])
    assert np.count_nonzero(v)==np.count_nonzero(a)==0


def test_large_return_extends_time_instead_of_raising_acceleration():
    sample,meta=plan_recovery([10,0,1.5],[0,0,0],[0,0,0],[0,0,1.5])
    assert meta['return_s']>10
    _,v,a=sample(np.linspace(meta['brake_end_s'],meta['return_end_s'],1001))
    assert np.linalg.norm(v,axis=1).max()<=.4+1e-9
    assert np.linalg.norm(a,axis=1).max()<=.3+1e-9


def test_export_preserves_whip_strings_and_original_recording(tmp_path):
    t=np.arange(401)*.01
    q=np.zeros((len(t),2,3));q[:,:,2]=[1.5,1.4]
    q[:,0,0]=t*.1
    velocity=np.zeros_like(q);velocity[:,0,0]=.1
    arrays=dict(time_s=t+5,cable_node_position_world_m=q,
        cable_node_velocity_world_m_s=velocity,controller_phase=np.where(t<=.8,1,np.where(t<2,2,0)))
    summary=dict(events=[dict(event='strike',time_s=5.,planned_duration_s=.8)],hover_position_m=[0,0,1.5])
    (tmp_path/'plan.npz').write_bytes(b'plan')
    meta=write_rehearsal_fullstate(arrays,summary,tmp_path,dict(cutoff_s=.8),gentle_recovery=True)
    assert meta['schema']=='gentle_recovery_fullstate_v4'
    old=(tmp_path/'recorded_pid_reference.csv').read_bytes().splitlines()
    new=(tmp_path/'fullstate_30hz.csv').read_bytes().splitlines()
    assert old[:26]==new[:26]  # Header + every original whip row, byte for byte.
    rows,loaded=load_reference(tmp_path)
    assert rows[-1]['pz_m']==1.5 and rows[-1]['phase']=='hover_hold'
    assert loaded['flight_ready'] is False


def test_full_model_export_maps_attachment_to_cf7_without_changing_time_va(tmp_path):
    t=np.arange(401)*.01
    q=np.zeros((len(t),2,3));q[:,:,2]=[1.5,1.4]
    velocity=np.zeros_like(q)
    arrays=dict(time_s=t+5,cable_node_position_world_m=q,
        cable_node_velocity_world_m_s=velocity,controller_phase=np.where(t<=.8,1,np.where(t<2,2,0)))
    summary=dict(events=[dict(event='strike',time_s=5.,planned_duration_s=.8)],hover_position_m=[0,0,1.5])
    (tmp_path/'plan.npz').write_bytes(b'plan')
    offset=np.array([.006,-.012,-.055])
    meta=write_rehearsal_fullstate(arrays,summary,tmp_path,dict(cutoff_s=.8),gentle_recovery=True,
                                 initial_world_attachment_offset=offset)
    with (tmp_path/'attachment_reference.csv').open(newline='') as stream:before=list(csv.DictReader(stream))
    with (tmp_path/'fullstate_30hz.csv').open(newline='') as stream:after=list(csv.DictReader(stream))
    assert len(before)==len(after)
    for old,new in zip(before,after):
        for key in old:
            if key in ('px_m','py_m','pz_m'):continue
            assert old[key]==new[key]
        np.testing.assert_allclose([float(new[k]) for k in ('px_m','py_m','pz_m')],
                                  np.array([float(old[k]) for k in ('px_m','py_m','pz_m')])-offset,atol=1e-12)
    rows,loaded=load_reference(tmp_path)
    assert loaded['reference_point']=='OptiTrack_cf7_origin'
    assert loaded['reference_mapping']['mode']=='fixed_initial_translation'
    assert loaded['reference_mapping']['dynamic_rigid_body_conversion'] is False
    assert rows[-1]['pz_m']==pytest.approx(1.555)
    np.testing.assert_allclose(meta['hover_position_m'],[-.006,.012,1.555])
    with np.load(tmp_path/'fullstate_source.npz') as recorded:
        np.testing.assert_array_equal(recorded['positions_m'],q)
    with pytest.raises(ValueError,match='original recorded'):
        replace_recorded_recovery(tmp_path)


def test_export_rotates_tracking_offset_and_records_pose_assumption(tmp_path):
    t=np.arange(201)*.01
    q=np.zeros((len(t),2,3));q[:,:,2]=[1.5,1.4]
    arrays=dict(time_s=t,cable_node_position_world_m=q,cable_node_velocity_world_m_s=np.zeros_like(q),
        controller_phase=np.where(t<=.8,1,np.where(t<1.5,2,0)))
    summary=dict(events=[dict(event='strike',time_s=0.,planned_duration_s=.8)],hover_position_m=[0,0,1.5])
    (tmp_path/'plan.npz').write_bytes(b'plan')
    meta=write_rehearsal_fullstate(arrays,summary,tmp_path,dict(cutoff_s=.8),gentle_recovery=True,
        attachment_offset_tracking_m=[0,0,-.055],initial_tracking_orientation_xyzw=[0,np.sqrt(.5),0,np.sqrt(.5)],
        initial_pose_source='synthetic 90 degree pitch; conversion test only')
    np.testing.assert_allclose(meta['initial_vehicle_position_m'],[.055,0,1.5],atol=1e-12)
    np.testing.assert_allclose(meta['reference_mapping']['initial_world_attachment_offset_m'],[-.055,0,0],atol=1e-12)
    assert meta['reference_mapping']['initial_pose_source'].startswith('synthetic')
    assert meta['reference_mapping']['dynamic_rigid_body_conversion'] is False


def test_export_requires_orientation_for_tracking_offset_before_writing(tmp_path):
    with pytest.raises(ValueError,match='initial orientation'):
        write_rehearsal_fullstate({}, {}, tmp_path, {}, attachment_offset_tracking_m=[0,0,-.055])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('origin',[0.,5.,30.63])
def test_cutoff_uses_whip_acceleration_not_pid_right_limit(tmp_path,origin):
    t=np.arange(201)*.01
    q=np.zeros((len(t),2,3));q[:,:,2]=[1.5,1.4]
    v=np.zeros_like(q)
    q[:,0,0]=.1*t*t+4.9*np.maximum(t-.8,0)**2
    v[:,0,0]=.2*t+9.8*np.maximum(t-.8,0)
    arrays=dict(time_s=t+origin,cable_node_position_world_m=q,
        cable_node_velocity_world_m_s=v,controller_phase=np.where(t<=.8,1,np.where(t<1.5,2,0)))
    summary=dict(events=[dict(event='strike',time_s=origin,planned_duration_s=.8)],hover_position_m=[0,0,1.5])
    (tmp_path/'plan.npz').write_bytes(b'plan')
    write_rehearsal_fullstate(arrays,summary,tmp_path,dict(cutoff_s=.8),gentle_recovery=True)
    rows,_=load_reference(tmp_path)
    assert rows[24]['time_s']==.8
    assert rows[24]['ax_m_s2']==pytest.approx(.2,abs=1e-8)
    assert rows[24]['vx_m_s']==pytest.approx(.16,abs=1e-10)


@pytest.mark.parametrize('settings',[dict(transition_s=0),dict(return_s=float('nan')),dict(unknown=1)])
def test_bad_settings_rejected(settings):
    with pytest.raises(ValueError):
        plan_recovery([0,0,1.5],[0,0,0],[0,0,0],[0,0,1.5],settings)
