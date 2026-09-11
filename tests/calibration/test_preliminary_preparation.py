import numpy as np
from pathlib import Path
from experimental_data.preliminary_prepare import recorded_packets,PreliminaryTrial,measured_clock_alignment
from experimental_data.adaptation_rounds import COMMAND_COLUMNS
from experimental_data.current_adaptation import read
from experimental_data.current_adaptation import causal_history_indices
from experimental_data.current_adaptation import endpoint_velocity
import pytest


def test_weighted_endpoint_velocity_recovers_accelerating_motion_and_ignores_future():
    time=np.arange(0,3.,.01);positions=np.stack([time**2,3*time,2*time**2-time],-1)
    ids=causal_history_indices(time,2.,1.);end=time[ids[-1]]
    for tau in (None,.02,.08,.16):
        v=endpoint_velocity(time[ids],positions[ids],tau)
        np.testing.assert_allclose(v,[2*end,3,4*end-1],atol=1e-10)
    original=endpoint_velocity(time[ids],positions[ids],.04);positions[time>=2.]=1e9
    np.testing.assert_array_equal(endpoint_velocity(time[ids],positions[ids],.04),original)


def test_one_second_cable_history_is_causal_and_consistent_at_each_window():
    time=np.arange(0,4.01,.01)
    for cutoff in (1.5,2.5):
        ids=causal_history_indices(time,cutoff,1.)
        assert len(ids)==101 and np.all(time[ids]<cutoff)
        assert abs(time[ids[-1]]-time[ids[0]]-1.)<1e-10
    with pytest.raises(ValueError,match='Missing causal'):causal_history_indices(time,.4,1.)
    missing=np.delete(time,120)
    with pytest.raises(ValueError,match='Missing causal'):causal_history_indices(missing,1.5,1.)

def test_cached_packet_jitter_is_grouped_without_changing_pva():
    receipt=np.repeat([0.,.03,.06],3)+np.tile([-2e-5,0,2e-5],3)
    time=np.arange(9)*.01;c={'time_s':time,'cmd_age':time-receipt,'cmd_valid':np.ones(9)}
    c['cmd_age'][0]=0.
    for k in COMMAND_COLUMNS:c[k]=np.repeat([1.,2.,3.],3)
    t,p,u,end=recorded_packets(c)
    assert len(t)==3 and np.max(abs(t-[0,.03,.06]))<=1e-5
    assert np.array_equal(p[:,0],[1,2,3]) and (u>t).all()

def test_clock_match_uses_observations_only():
    t=np.arange(0,10,.01);xyz=np.c_[t*.1,np.sin(t),np.cos(t*.4)]
    m={'time':t,'drone':xyz};ids=np.arange(0,len(t),10)
    c={'time_s':t[ids]+1.7,'x':xyz[ids,0],'y':xyz[ids,1],'z':xyz[ids,2]}
    c.update({k:np.full(len(ids),99999.) for k in COMMAND_COLUMNS})
    fit=measured_clock_alignment(m,c)
    assert abs(fit['offset_s']-1.7)<1e-6 and fit['rmse_m']<1e-6

def test_native_initial_state_does_not_use_future_measurements():
    job=Path(__file__).resolve().parents[2]/'runs/adaptation/20260909-preliminary1-M0-v2'
    if not job.exists():
        import pytest;pytest.skip('Local preliminary batch not installed')
    trial=PreliminaryTrial(job,read(job/'windows.json')[3],read(job/'source_candidate/model.json'),device='cpu')
    assert trial.cable_history_s is None  # Old jobs are not silently reinterpreted.
    gains=np.array([10,10,5,5,1,1]);first=trial.state(gains,'cpu')
    trial.data['position'][trial.data['time']>=0]=999
    trial.data['rotation'][trial.data['time']>=0]=0
    trial.data['sites'][trial.data['time']>=0]=999
    second=trial.state(gains,'cpu');first.validate();second.validate()
    assert first.time_s<0
    for key in ('position','velocity','rotation','omega_tracking','compensation','rotation_command_from_tracking'):
        assert np.array_equal(getattr(first,key).numpy(),getattr(second,key).numpy())
