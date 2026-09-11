from types import SimpleNamespace
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from experimental_data.attitude_identification import AttitudeTrial,fit_attitude_mapping,local_log
from experimental_data.nominal_pose_fit import parameters
from simulator.drone_pose_response import initialize_from_hover,CommandSchedule,predict_pose,rotation_log


def synthetic():
    history=np.linspace(-1,0,101);p0=np.array([0,0,1.5]);v0=np.array([.02,-.01,.005]);a0=np.array([.015,-.01,.008])
    pos=p0+history[:,None]*v0+.5*history[:,None]**2*a0
    r=Rotation.from_euler('xyz',[.02,-.01,.03]).as_matrix()
    rot=np.broadcast_to(r,(1,101,3,3)).copy()
    hold=np.zeros((1,101,11));hold[:,:,:3]=p0
    hover=(pos[None],rot,hold);gains=np.array([5,9,3,2,.4,1.8])
    def state(g,device,tau=.08,delay=.02,scales=(1,1)):
        return initialize_from_hover(history,*[torch.as_tensor(x,dtype=torch.float64,device=device) for x in hover],
            parameters(g,tau,delay,scales),alignment_mode='prehover_effective_alignment')[0]
    initial=state(gains,'cpu');zero=np.zeros(6);b0=state(zero,'cpu').compensation[0].numpy()
    basis=np.zeros((3,6))
    for i in range(4):
        x=zero.copy();x[i]=1;basis[:,i]=state(x,'cpu').compensation[0].numpy()-b0
    cmd=np.zeros((1,7,11));cmd[:,:,:3]=p0
    cmd[0,1:,:3]+=[[.02,.01,.01],[.05,-.02,.04],[.09,.04,.06],[.07,.02,.04],[.02,-.01,.03],[0,0,0]]
    cmd[0,1:,6:9]=[[2,.5,2],[-2,.3,-2],[3,-.5,1],[-1,-.4,-2],[1,.1,1],[0,0,0]]
    stream=CommandSchedule([-.2,0,.08,.16,.25,.35,.5],torch.as_tensor(cmd),coverage_end_s=.7)
    t=np.linspace(0,.6,61)
    trial=SimpleNamespace(name='synthetic',time=t,context={'csv_end_s':.5},schedule=stream,
        p0=initial.position[0].numpy(),v0=initial.velocity[0].numpy(),b0=b0,basis=basis,hover=hover,state=state,
        masks={'fit_orientation':(t>.05)&(t<=.5)},truth={'rotation_tracking_to_world':np.zeros((61,3,3))})
    return trial,gains,state


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required'))])
def test_cached_attitude_matches_full_engine_with_scale_dependent_hover_alignment(device):
    trial,gains,state=synthetic();cache=AttitudeTrial(trial,gains,.017)
    stream=CommandSchedule(trial.schedule.time,trial.schedule.values.to(device),coverage_end_s=.7)
    for scales,tau in [((.7,.4),.035),((1.2,.6),.09)]:
        expected=predict_pose(state(gains,device,tau,.017,scales),stream,cache.time,parameters(gains,tau,.017,scales),
            offset_tracking_m=[.007,-.014,-.055])['rotation_tracking_to_world'][0].cpu().numpy()
        np.testing.assert_allclose(cache.predict([*scales,tau]),expected,atol=1e-11,rtol=0)
    trial.truth['rotation_tracking_to_world'][trial.time>.5]=np.nan
    assert np.isfinite(AttitudeTrial(trial,gains,.017).predict([.7,.4,.035])).all()


def test_attitude_mapping_recovers_synthetic_parameters_without_hold_truth():
    trial,gains,state=synthetic();cache=AttitudeTrial(trial,gains,.017);truth=[.8,.45,.06]
    trial.truth['rotation_tracking_to_world'][:len(cache.time)]=cache.predict(truth)
    trial.truth['rotation_tracking_to_world'][trial.time>.5]=np.nan
    result=fit_attitude_mapping([trial],gains,.017,progress=lambda *a,**k:None)
    assert result['valid'] and result['mean_squared_angle_rad2']<1e-9
    np.testing.assert_allclose([*result['scales'],result['tau_s']],truth,atol=2e-3,rtol=0)


def test_numpy_local_log_matches_torch_including_domain_rejection():
    for angle in [0,1e-8,.5,2.9]:
        r=Rotation.from_rotvec(np.array([.3,.2,.1])/np.linalg.norm([.3,.2,.1])*angle).as_matrix()
        np.testing.assert_allclose(local_log(r),rotation_log(torch.as_tensor(r)),atol=1e-12)
    with pytest.raises(ValueError,match='180'):local_log(Rotation.from_euler('x',np.pi).as_matrix())
