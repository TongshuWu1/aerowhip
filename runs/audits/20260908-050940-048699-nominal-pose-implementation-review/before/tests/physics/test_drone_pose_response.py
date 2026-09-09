from dataclasses import replace
import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
import torch

from simulator.drone_pose_response import (PoseResponseParameters,PoseResponseState,CommandSchedule,
    initialize_from_hover,predict_pose,rotation_exp,rotation_log,attachment_pva)


def parameters(**kwargs):
    values=dict(kp_xy=4.,kp_z=4.,kd_xy=3.,kd_z=3.,feedforward_xy=1.,feedforward_z=1.,
                attitude_time_constant_s=.08,delay_s=.02)
    return PoseResponseParameters(**{**values,**kwargs})


def initial(device='cpu',batch=1):
    vector=torch.zeros(batch,3,dtype=torch.float64,device=device)
    p=vector.clone();p[:,2]=1.5
    eye=torch.eye(3,dtype=p.dtype,device=p.device).expand(batch,3,3).clone()
    return PoseResponseState(p,vector.clone(),eye,vector.clone(),vector.clone(),eye.clone(),'explicit_calibration',0.)


def schedule(state,times=(-.1,0.,.2),accelerations=None,positions=None,**kwargs):
    commands=state.position[:,None].new_zeros((len(state.position),len(times),11))
    commands[:,:,:3]=state.position[:,None]
    if accelerations is not None:commands[:,:,6:9]=commands.new_tensor(accelerations)
    if positions is not None:commands[:,:,:3]=commands.new_tensor(positions)
    return CommandSchedule(times,commands,coverage_end_s=.5,**kwargs)


def test_hover_compensation_prevents_invented_launch_transient():
    t=np.arange(101)*.01-1
    p=torch.tensor([.04,-.03,1.56],dtype=torch.float64).expand(2,101,3).clone()
    r=torch.tensor(Rotation.from_euler('xyz',[.03,-.02,.1]).as_matrix()).expand(2,101,3,3).clone()
    cmd=torch.zeros(2,101,11,dtype=p.dtype);cmd[:,:,:3]=torch.tensor([0.,0.,1.5])
    state,metadata=initialize_from_hover(t,p,r,cmd,parameters(),alignment_mode='prehover_effective_alignment')
    torch.testing.assert_close(state.compensation,4*(p[:,-1]-cmd[:,0,:3]),atol=1e-11,rtol=1e-11)
    stream=CommandSchedule([-.2,.2],cmd[:,[0,0]],coverage_end_s=.6)
    out=predict_pose(state,stream,np.linspace(0,.4,41),parameters(),offset_tracking_m=[.007,-.014,-.055])
    torch.testing.assert_close(out['position_origin_m'],p[:,:41],atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(out['rotation_tracking_to_world'],r[:,:41],atol=1e-10,rtol=1e-10)
    assert not metadata['controller_integral_observed'] and not metadata['hardware_mounting_verified_by_this_code']


def test_exact_30hz_pulse_timing_with_non_grid_delay():
    state=initial();on=1/30;off=2/30;delay=.017;end=.15
    stream=schedule(state,times=(-.1,0,on,off,.2),accelerations=[[0,0,0],[0,0,0],[2,0,0],[0,0,0],[0,0,0]])
    params=parameters(kp_xy=0.,kp_z=0.,kd_xy=0.,kd_z=0.,delay_s=delay)
    for step in (.005,.003):
        out=predict_pose(state,stream,np.array([0.,.04,.08,end]),params,offset_tracking_m=[0,0,-.055],maximum_step_s=step)
        expected_v=2*(off-on);expected_p=expected_v*(end-(on+delay+(off-on)/2))
        assert out['velocity_origin_m_s'][0,-1,0].item()==pytest.approx(expected_v,abs=1e-12)
        assert out['position_origin_m'][0,-1,0].item()==pytest.approx(expected_p,abs=1e-12)
        assert out['velocity_origin_m_s'][0,1,0].item()==0
        assert out['position_origin_m'][0,-1,2].item()==1.5 # no extra gravity in effective translation


def test_second_order_translation_converges_to_independent_linear_solution():
    state=initial();state=replace(state,position=state.position+state.position.new_tensor([.2,0,0]))
    stream=schedule(state,positions=[[0,0,1.5]]*3)
    exact=expm(np.array([[0,1],[-4,-3]])*.4)@np.array([.2,0])
    errors=[]
    for step in (.01,.005,.0025):
        out=predict_pose(state,stream,[0,.4],parameters(),offset_tracking_m=[0,0,-.055],maximum_step_s=step)
        predicted=np.array([out['position_origin_m'][0,-1,0].item(),out['velocity_origin_m_s'][0,-1,0].item()])
        errors.append(np.linalg.norm(predicted-exact))
    assert errors[0]/errors[1]>3.5 and errors[1]/errors[2]>3.5


def test_attitude_response_matches_critical_damping_solution_and_converges():
    state=initial();theta=.4;tau=.08;end=.2
    stream=schedule(state,accelerations=[[0,0,0],[9.80665*np.tan(theta),0,0],[0,0,0]])
    params=parameters(kp_xy=0.,kp_z=0.,kd_xy=0.,kd_z=0.,delay_s=0.,attitude_time_constant_s=tau)
    exact=theta*(1-(1+end/tau)*np.exp(-end/tau))
    errors=[]
    for step in (.01,.005,.0025):
        out=predict_pose(state,stream,[0,end],params,offset_tracking_m=[0,0,-.055],maximum_step_s=step)
        angle=rotation_log(out['rotation_tracking_to_world'][0,-1])[1].item()
        errors.append(abs(angle-exact))
        rotations=out['rotation_tracking_to_world']
        torch.testing.assert_close(rotations.transpose(-1,-2)@rotations,torch.eye(3,dtype=torch.float64).expand_as(rotations),atol=1e-12,rtol=1e-12)
    assert errors[0]/errors[1]>3.4 and errors[1]/errors[2]>3.4


def test_rotation_equivariance_to_tracking_frame_relabeling():
    a=initial();change=torch.tensor(Rotation.from_euler('xyz',[.4,-.3,.2]).as_matrix()).unsqueeze(0)
    b=replace(a,rotation=a.rotation@change,rotation_command_from_tracking=a.rotation_command_from_tracking@change)
    r=torch.tensor([[.007,-.014,-.055]],dtype=torch.float64)
    rp=torch.einsum('bij,bj->bi',change.transpose(-1,-2),r)
    stream=schedule(a,accelerations=[[0,0,0],[3,.4,1],[0,0,0]])
    times=np.linspace(0,.3,31)
    x=predict_pose(a,stream,times,parameters(),offset_tracking_m=r)
    y=predict_pose(b,stream,times,parameters(),offset_tracking_m=rp)
    for key in ['position_attachment_m','velocity_attachment_m_s','acceleration_attachment_m_s2']:
        torch.testing.assert_close(x[key],y[key],atol=1e-9,rtol=1e-9)
    torch.testing.assert_close(y['rotation_tracking_to_world'],x['rotation_tracking_to_world']@change[:,None],atol=1e-10,rtol=1e-10)


def test_attachment_acceleration_includes_tangential_and_centripetal_terms():
    t=.4;theta=.6*t*t;r=np.array([.007,-.014,-.055])
    rotation=Rotation.from_euler('xyz',[.2,-.1,.3])*Rotation.from_rotvec([0,theta,0])
    state=initial();state=replace(state,position=torch.tensor([[t*t,.2*t,1.5]]),
        velocity=torch.tensor([[2*t,.2,0.]]),rotation=torch.tensor(rotation.as_matrix())[None],
        omega_tracking=torch.tensor([[0,1.2*t,0.]]))
    # Use one consistent precision for independently supplied analytical states.
    state=replace(state,position=state.position.double(),velocity=state.velocity.double(),omega_tracking=state.omega_tracking.double())
    a=torch.tensor([[2.,0,0]],dtype=torch.float64);alpha=torch.tensor([[0,1.2,0]],dtype=torch.float64)
    pa,va,aa=attachment_pva(state,a,alpha,r)
    def f(x):
        R=Rotation.from_euler('xyz',[.2,-.1,.3])*Rotation.from_rotvec([0,.6*x*x,0])
        return np.array([x*x,.2*x,1.5])+R.apply(r)
    h=1e-4
    np.testing.assert_allclose(pa[0],f(t),atol=1e-7)
    np.testing.assert_allclose(va[0],(f(t+h)-f(t-h))/(2*h),atol=1e-7)
    np.testing.assert_allclose(aa[0],(f(t+h)-2*f(t)+f(t-h))/h**2,atol=1e-7)


def test_missing_command_gap_is_rejected_before_rollout():
    state=initial();stream=schedule(state,valid=[True,False,True])
    original=state.position.clone()
    with pytest.raises(ValueError,match='fresh observed command'):
        predict_pose(state,stream,[0,.3],parameters(),offset_tracking_m=[0,0,-.055])
    torch.testing.assert_close(state.position,original)
    with pytest.raises(ValueError,match='fresh observed command'):stream.sample(-.2)
    with pytest.raises(ValueError,match='fresh observed command'):stream.sample(.5)


def test_cache_expiry_inside_sampling_interval_is_not_bridged():
    state=initial();stream=schedule(state,valid_until_s=[0.,.015,.5])
    with pytest.raises(ValueError,match='fresh observed command'):
        predict_pose(state,stream,[0,.3],parameters(),offset_tracking_m=[0,0,-.055])


def test_future_command_cannot_change_earlier_prediction():
    state=initial();one=schedule(state);two=schedule(state,accelerations=[[0,0,0],[0,0,0],[8,2,3]])
    a=predict_pose(state,one,[0,.1],parameters(),offset_tracking_m=[0,0,-.055])
    b=predict_pose(state,two,[0,.1],parameters(),offset_tracking_m=[0,0,-.055])
    for key in a:torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)


def test_parameter_gradients_are_finite_and_match_difference():
    state=initial();stream=schedule(state,accelerations=[[0,0,0],[2,.2,.5],[0,0,0]])
    gain=torch.tensor(1.,dtype=torch.float64,requires_grad=True)
    tau=torch.tensor(.08,dtype=torch.float64,requires_grad=True)
    def objective(gain,tau):
        params=parameters(feedforward_xy=gain,attitude_time_constant_s=tau)
        out=predict_pose(state,stream,[0,.15],params,offset_tracking_m=[.007,-.014,-.055])
        return out['position_attachment_m'][0,-1,0]
    loss=objective(gain,tau);grad=torch.autograd.grad(loss,(gain,tau))
    for i,value in enumerate((gain,tau)):
        delta=1e-5;args=[gain.detach(),tau.detach()]
        args[i]=value.detach()+delta;up=objective(*args)
        args[i]=value.detach()-delta;down=objective(*args)
        assert torch.isfinite(grad[i])
        assert grad[i].item()==pytest.approx(((up-down)/(2*delta)).item(),abs=1e-7,rel=1e-4)
    z=torch.zeros(1,3,dtype=torch.float64,requires_grad=True)
    out=rotation_log(rotation_exp(z));gradient=torch.autograd.grad(out.sum(),z)[0]
    torch.testing.assert_close(gradient,torch.ones_like(z),atol=1e-10,rtol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hardware required')
def test_batched_cuda_matches_cpu():
    outputs=[]
    for device in ['cpu','cuda']:
        state=initial(device,batch=16);stream=schedule(state,accelerations=[[0,0,0],[2,.2,.5],[0,0,0]])
        out=predict_pose(state,stream,np.linspace(0,.3,31),parameters(),offset_tracking_m=[.007,-.014,-.055])
        outputs.append({key:value.cpu() for key,value in out.items()})
    for key in outputs[0]:torch.testing.assert_close(outputs[0][key],outputs[1][key],atol=1e-9,rtol=1e-9)


def test_no_implicit_alignment_and_no_unmodeled_body_rate_input():
    state=initial();params=parameters()
    with pytest.raises(ValueError,match='Alignment provenance'):
        replace(state,alignment_provenance='assumed_identity').validate()
    commands=torch.zeros(1,3,11,dtype=torch.float64);commands[...,10]=.2
    with pytest.raises(ValueError,match='body-rate'):
        CommandSchedule([0,.1,.2],commands,coverage_end_s=.3)
    with pytest.raises(ValueError,match='Reduce integration step'):
        predict_pose(state,schedule(state),[0,.3],params,offset_tracking_m=[0,0,-.055],maximum_step_s=.1)
    with pytest.raises(ValueError,match='180'):
        rotation_log(torch.tensor(Rotation.from_euler('x',np.pi).as_matrix()))
    with pytest.raises(ValueError,match='timestamp'):
        predict_pose(state,schedule(state),[.1,.3],params,offset_tracking_m=[0,0,-.055])
