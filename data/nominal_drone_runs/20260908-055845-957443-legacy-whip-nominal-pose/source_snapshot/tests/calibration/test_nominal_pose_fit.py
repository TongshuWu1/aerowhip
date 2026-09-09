from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from experimental_data.nominal_pose_fit import (fitting_masks,integration_steps,linear_prediction,
    translation_objective,fit_translation,fit_attitude,parameters,PRIOR,GAIN_NAMES)
from simulator.drone_pose_response import PoseResponseState,CommandSchedule,predict_pose


def setup():
    p=torch.tensor([[0.,0.,1.5]],dtype=torch.float64);zero=torch.zeros_like(p);eye=torch.eye(3,dtype=p.dtype)[None]
    state=PoseResponseState(p,zero,eye,zero,zero,eye,'explicit_calibration',0.)
    times=np.array([-.2,0.,1/30,2/30,.15,.27,.5])
    values=torch.zeros(1,len(times),11,dtype=p.dtype);values[:,:,2]=1.5
    values[0,1:,:3]+=torch.tensor([[.03,.01,.02],[.1,-.02,.04],[.2,.03,.06],[.15,-.03,.1],[0,.02,.05],[0,0,0.]])
    values[0,1:,6:9]=torch.tensor([[1,.1,.4],[2,-.1,.3],[-1,.2,-.2],[.5,-.3,.1],[-1,.1,-.3],[0,0,0.]])
    stream=CommandSchedule(times,values,coverage_end_s=.8)
    output=np.arange(61)*.01
    return state,stream,output


def test_linear_fit_evaluator_and_gain_jacobian_match_pose_engine():
    state,stream,t=setup();gains=np.array([8.,12.,4.,5.,.8,1.2])
    base=np.array([.02,-.01,.03]);basis=np.arange(18).reshape(3,6)*.0001;basis[:,4:]=0
    plan=integration_steps(stream,t,.017)
    p,v,j=linear_prediction(gains,state.position[0],state.velocity[0],base,basis,plan)
    actual=predict_pose(replace(state,compensation=torch.tensor((base+basis@gains)[None])),stream,t,
        parameters(gains,.08,.017),offset_tracking_m=[.007,-.014,-.055])
    np.testing.assert_allclose(p,actual['position_origin_m'][0],atol=1e-12,rtol=0)
    np.testing.assert_allclose(v,actual['velocity_origin_m_s'][0],atol=1e-12,rtol=0)
    for i in range(6):
        up=gains.copy();down=gains.copy();up[i]+=1e-5;down[i]-=1e-5
        difference=(linear_prediction(up,state.position[0],state.velocity[0],base,basis,plan)[0]
                    -linear_prediction(down,state.position[0],state.velocity[0],base,basis,plan)[0])/(2e-5)
        np.testing.assert_allclose(j[:,:,i],difference,atol=1e-9,rtol=1e-5)


def truth_fixture():
    t=np.arange(121)*.01
    truth=dict(position_origin_m=np.zeros((len(t),3)),execution_phase=np.where(t<.2,'pre_maneuver_fullstate_hold',
        np.where(t<.9,'csv_maneuver','post_maneuver_fullstate_hold')),reference_valid=np.ones(len(t),bool),
        phase_boundary_uncertain=np.zeros(len(t),bool),position_valid=np.ones(len(t),bool),
        orientation_valid=np.ones(len(t),bool),attachment_valid=np.ones(len(t),bool))
    return t,truth


def test_saved_interval_and_component_exclusions_are_applied_without_splicing():
    t,truth=truth_fixture();truth['phase_boundary_uncertain'][20]=True
    masks=fitting_masks(t,truth,review=dict(precontact_start_s=.1,precontact_end_s=.75),exclusions=[
        dict(start_s=.4,end_s=.45,component='drone',reason='measurement glitch'),
        dict(start_s=.5,end_s=.6,component='cable',reason='cable-only dropout')])
    assert len(masks['fit_position'])==len(t)
    assert not masks['fit_position'][t>.75].any()
    assert not masks['fit_position'][(t>=.4)&(t<=.45)].any()
    assert masks['fit_position'][(t>=.5)&(t<=.6)].all()
    assert not masks['fit_position'][20]
    with pytest.raises(ValueError,match='reason'):
        fitting_masks(t,truth,exclusions=[dict(start_s=.4,end_s=.5,component='drone')])


def test_hold_targets_and_masked_measurements_do_not_affect_fit_loss():
    state,stream,t=setup();plan=integration_steps(stream,t,.02)
    def linear(gains,delay):return linear_prediction(gains,state.position[0],state.velocity[0],np.zeros(3),np.zeros((3,6)),plan)
    pos=linear(PRIOR,.02)[0];mask=(t>.1)&(t<.5)
    trial=SimpleNamespace(linear=linear,masks={'fit_position':mask},truth={'position_origin_m':pos.copy()})
    expected=translation_objective([trial],PRIOR,.02)
    trial.truth['position_origin_m'][~mask]=np.nan
    got=translation_objective([trial],PRIOR,.02)
    for a,b in zip(expected,got):np.testing.assert_array_equal(a,b)


def test_constant_acceleration_sensitivity_has_analytic_solution():
    t=np.linspace(0,.3,31);dt=np.diff(t);commands=np.zeros((len(dt),11));commands[:,6:9]=[2,-1,.5]
    p,v,j=linear_prediction(np.array([0,0,0,0,1,1]),np.zeros(3),np.zeros(3),np.zeros(3),np.zeros((3,6)),(dt,commands,np.arange(len(t))))
    np.testing.assert_allclose(p,.5*t[:,None]**2*np.array([2,-1,.5]),atol=1e-14)
    np.testing.assert_allclose(j[:,0,4],t*t,atol=1e-14)
    np.testing.assert_allclose(j[:,2,5],.25*t*t,atol=1e-14)


def test_fitter_recovers_simple_known_feedforward_without_using_hold_truth():
    state,stream,t=setup();plan=integration_steps(stream,t,.02);target=PRIOR.copy();target[4:]=[.75,1.25]
    def linear(gains,delay):return linear_prediction(gains,state.position[0],state.velocity[0],np.zeros(3),np.zeros((3,6)),plan)
    trial=SimpleNamespace(linear=linear,masks={'fit_position':t<.5},truth={'position_origin_m':linear(target,.02)[0]})
    # Noise-free synthetic recovery uses no shrinkage; real fits explicitly
    # retain regularization and need not recover a unique physical parameter.
    fit=fit_translation([trial],delays=[.02],starts=[PRIOR],prior_weight=0.,progress=lambda *a,**k:None)
    got=np.array(fit['best']['gains'])
    assert fit['best']['objective']<translation_objective([trial],PRIOR,.02)[0]@translation_objective([trial],PRIOR,.02)[0]
    assert np.max(abs(got[4:]-target[4:]))<.04


def test_attitude_optimization_never_simulates_or_scores_post_hold():
    t,truth=truth_fixture();rotations=np.broadcast_to(np.eye(3),(len(t),3,3)).copy();rotations[t>=.9]=np.nan
    def pose(*args,maneuver_only=False,**kwargs):
        assert maneuver_only
        return {'rotation_tracking_to_world':torch.eye(3,dtype=torch.float64).expand(1,90,3,3)}
    trial=SimpleNamespace(pose=pose,masks={'fit_orientation':(t>=.2)&(t<.9)},truth={'rotation_tracking_to_world':rotations})
    result=fit_attitude([trial],PRIOR,.02,device='cpu',progress=lambda *a,**k:None)
    assert result['mean_squared_angle_rad2']==0


def test_attitude_domain_failure_is_recorded_without_hiding_other_errors():
    t,truth=truth_fixture();rotations=np.broadcast_to(np.eye(3),(len(t),3,3)).copy()
    def pose(gains,tau,delay,**kwargs):
        if tau>.29:
            raise ValueError('Attitude error is too close to 180 degrees for this local response model')
        return {'rotation_tracking_to_world':torch.eye(3,dtype=torch.float64).expand(1,90,3,3)}
    trial=SimpleNamespace(name='synthetic',pose=pose,masks={'fit_orientation':(t>=.2)&(t<.9)},
        truth={'rotation_tracking_to_world':rotations})
    result=fit_attitude([trial],PRIOR,.02,device='cpu',progress=lambda *a,**k:None)
    assert result['valid'] and result['tau_s']<.29
    invalid=[row for row in result['trace'] if not row['valid']]
    assert invalid and all(row['mean_squared_angle_rad2'] is None for row in invalid)
    assert any(row['tau_s']==pytest.approx(.30) for row in invalid)
    import json
    json.dumps(result,allow_nan=False)
    def missing(*args,**kwargs):raise ValueError('Missing command')
    trial.pose=missing
    with pytest.raises(ValueError,match='Missing command'):
        fit_attitude([trial],PRIOR,.02,device='cpu',progress=lambda *a,**k:None)
