from dataclasses import replace
import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
import torch

from simulator.drone_pose_response import (PoseResponseParameters,PoseResponseState,CommandSchedule,
    initialize_from_hover,predict_pose,rotation_exp,rotation_log,attachment_pva,attitude_drive_acceleration,command_attitude,derivatives)


def parameters(**kwargs):
    values=dict(kp_xy=4.,kp_z=4.,kd_xy=3.,kd_z=3.,feedforward_xy=1.,feedforward_z=1.,
                attitude_time_constant_s=.08,delay_s=.02,attitude_drive_model='independent_scale_v3')
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


def test_command_query_rejects_nan_time():
    with pytest.raises(ValueError,match='finite'):
        schedule(initial()).sample(float('nan'))


def test_large_translational_gains_require_resolved_timestep():
    state=initial()
    state=replace(state,velocity=torch.tensor([[0.,0.,.001]],dtype=torch.float64))
    params=parameters(kp_xy=0.,kp_z=0.,kd_z=1000.)
    with pytest.raises(ValueError,match='translational'):
        predict_pose(state,schedule(state),[0,.01],params,offset_tracking_m=[0,0,-.055])
    out=predict_pose(state,schedule(state),[0,.01],params,offset_tracking_m=[0,0,-.055],maximum_step_s=.0001)
    assert out['position_origin_m'][0,-1,2].item()==pytest.approx(1.5+.001/1000*(1-np.exp(-10)),abs=1e-10)


def test_multi_axis_pose_matches_independent_adaptive_ode():
    from scipy.integrate import solve_ivp
    state=initial()
    state=replace(state,rotation=torch.tensor(Rotation.from_euler('xyz',[.2,-.15,.25]).as_matrix())[None],
        velocity=torch.tensor([[.12,-.05,.04]],dtype=torch.float64),
        omega_tracking=torch.tensor([[.3,-.2,.1]],dtype=torch.float64),
        compensation=torch.tensor([[.02,-.01,.03]],dtype=torch.float64))
    stream=schedule(state,times=(-.1,.45),accelerations=[[2,.8,.4]]*2,positions=[[.1,-.06,1.55]]*2)
    values=stream.values[0,0].numpy();comp=state.compensation[0].numpy()
    def ode(t,y):
        p,v,R,w=y[:3],y[3:6],y[6:15].reshape(3,3),y[15:]
        gain=np.array([.35,.35,1.9])
        a=4*(values[:3]-p)+3*(values[3:6]-v)+gain*values[6:9]+comp
        z=a/gain+np.array([0,0,9.80665]);z/=np.linalg.norm(z)
        heading=np.array([np.cos(values[9]),np.sin(values[9]),0])
        yy=np.cross(z,heading);yy/=np.linalg.norm(yy)
        desired=np.column_stack([np.cross(yy,z),yy,z])
        error=Rotation.from_matrix(R.T@desired).as_rotvec()
        wx=np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])
        return np.r_[v,a,(R@wx).ravel(),error/.08**2-2*w/.08]
    y0=np.r_[state.position[0],state.velocity[0],state.rotation[0].numpy().ravel(),state.omega_tracking[0]]
    reference=solve_ivp(ode,[0,.4],y0,method='DOP853',rtol=1e-11,atol=1e-13)
    assert reference.success
    expected=reference.y[:,-1];errors=[]
    for step in [.01,.005,.0025]:
        out=predict_pose(state,stream,[0,.4],parameters(delay_s=0.,feedforward_xy=.35,feedforward_z=1.9,
            attitude_acceleration_scale_xy=1/.35,attitude_acceleration_scale_z=1/1.9),offset_tracking_m=[.007,-.014,-.055],maximum_step_s=step)
        got=np.r_[out['position_origin_m'][0,-1],out['velocity_origin_m_s'][0,-1],
                  out['rotation_tracking_to_world'][0,-1].numpy().ravel(),out['omega_tracking_rad_s'][0,-1]]
        errors.append(np.linalg.norm(got-expected))
    assert errors[0]/errors[1]>3.5 and errors[1]/errors[2]>3.5
    assert errors[-1]<2e-4


def test_normalized_attitude_drive_preserves_translation_and_has_unit_feedforward():
    params=parameters(feedforward_xy=.3,feedforward_z=1.9,delay_s=0.,attitude_drive_model='normalized_response_v2')
    state=initial();state=replace(state,compensation=torch.tensor([[.1,-.2,.3]],dtype=torch.float64))
    cmd=torch.zeros(1,11,dtype=torch.float64);cmd[:,:3]=state.position+torch.tensor([[.02,.01,-.03]])
    cmd[:,3:6]=torch.tensor([[.1,-.2,.05]]);cmd[:,6:9]=torch.tensor([[2.,.3,-4.6]])
    a,_=derivatives(state,cmd,params)
    gain=torch.tensor([.3,.3,1.9],dtype=torch.float64)
    expected=4/gain*(cmd[:,:3]-state.position)+3/gain*(cmd[:,3:6]-state.velocity)+cmd[:,6:9]+state.compensation/gain
    torch.testing.assert_close(attitude_drive_acceleration(a,params),expected)
    old=replace(params,attitude_drive_model='kinematic_v1')
    a_old,_=derivatives(state,cmd,old)
    torch.testing.assert_close(a,a_old,atol=0,rtol=0)
    torch.testing.assert_close(attitude_drive_acceleration(a,old),a,atol=0,rtol=0)
    with pytest.raises(ValueError,match='strictly positive'):
        replace(params,feedforward_z=0.).validate()


def test_response_gain_no_longer_turns_upright_control_direction_upside_down():
    params=parameters(feedforward_xy=.3,feedforward_z=1.9,attitude_drive_model='normalized_response_v2')
    u=torch.tensor([[-8.,.01,-6.]],dtype=torch.float64)
    a=u*torch.tensor([.3,.3,1.9],dtype=torch.float64)
    corrected=command_attitude(attitude_drive_acceleration(a,params),a.new_zeros(1),9.80665)
    legacy=command_attitude(a,a.new_zeros(1),9.80665)
    assert corrected[0,2,2]>0 and legacy[0,2,2]<0
    # A truly inverted inferred command remains inverted: there is no clipping
    # or removal of the local SO(3) domain guard.
    inverted=command_attitude(attitude_drive_acceleration(2*a,params),a.new_zeros(1),9.80665)
    assert inverted[0,2,2]<0


def test_independent_attitude_scale_gradients_and_translation_invariance():
    state=initial();stream=schedule(state,accelerations=[[0,0,0],[2,.4,-2],[0,0,0]])
    sx=torch.tensor(.8,dtype=torch.float64,requires_grad=True);sz=torch.tensor(.45,dtype=torch.float64,requires_grad=True)
    def evaluate(x,z):
        return predict_pose(state,stream,[0,.15],parameters(attitude_acceleration_scale_xy=x,attitude_acceleration_scale_z=z),
            offset_tracking_m=[.007,-.014,-.055])
    out=evaluate(sx,sz);loss=out['position_attachment_m'][0,-1,0];grad=torch.autograd.grad(loss,(sx,sz))
    for i,value in enumerate((sx,sz)):
        args=[sx.detach(),sz.detach()];args[i]=value.detach()+1e-5;up=evaluate(*args)['position_attachment_m'][0,-1,0]
        args[i]=value.detach()-1e-5;down=evaluate(*args)['position_attachment_m'][0,-1,0]
        torch.testing.assert_close(grad[i],(up-down)/2e-5,atol=1e-9,rtol=1e-5)
    other=evaluate(.4,.7)
    for key in ['position_origin_m','velocity_origin_m_s','acceleration_origin_m_s2']:
        torch.testing.assert_close(out[key],other[key],atol=0,rtol=0)


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hardware required'))])
def test_gain_gradients_include_recomputed_hover_compensation(device):
    t=np.arange(101)*.01-1
    p=torch.tensor([.04,0,1.5],dtype=torch.float64,device=device).expand(1,101,3).clone()
    p[0,:,0]+=.015*torch.as_tensor(t,device=device)
    r=torch.eye(3,dtype=torch.float64,device=device).expand(1,101,3,3)
    cmd=torch.zeros(1,101,11,dtype=torch.float64,device=device);cmd[...,2]=1.5
    def objective(gain):
        params=parameters(kp_xy=gain)
        state,_=initialize_from_hover(t,p,r,cmd,params,alignment_mode='prehover_effective_alignment')
        stream=schedule(state,positions=[[0,0,1.5],[.1,0,1.5],[0,0,1.5]])
        return predict_pose(state,stream,[0,.15],params,offset_tracking_m=[.007,-.014,-.055])['position_attachment_m'][0,-1,0]
    gain=torch.tensor(4.,dtype=torch.float64,device=device,requires_grad=True)
    grad=torch.autograd.grad(objective(gain),gain)[0]
    finite=(objective(gain.detach()+1e-4)-objective(gain.detach()-1e-4))/(2e-4)
    torch.testing.assert_close(grad,finite,atol=1e-9,rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hardware required')
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_distinct_batch_members_match_analytic_translation_and_cpu_cuda(dtype):
    outputs=[];t=np.array([0.,.07,.2,.35]);batch=7
    for device in ['cpu','cuda']:
        a=initial(device,batch)
        cast=lambda x:x.to(dtype=dtype)
        v=cast(a.velocity);v[:,0]=torch.arange(batch,device=device,dtype=dtype)*.02
        acceleration=torch.arange(batch*3,device=device,dtype=dtype).reshape(batch,3)*.04
        a=replace(a,position=cast(a.position),velocity=v,rotation=cast(a.rotation),
            omega_tracking=cast(a.omega_tracking),compensation=cast(a.compensation),
            rotation_command_from_tracking=cast(a.rotation_command_from_tracking))
        command=torch.zeros(batch,1,11,dtype=dtype,device=device)
        command[:,:,:3]=a.position[:,None];command[:,:,6:9]=acceleration[:,None]
        stream=CommandSchedule([-.1],command,coverage_end_s=.5)
        out=predict_pose(a,stream,t,parameters(kp_xy=0.,kp_z=0.,kd_xy=0.,kd_z=0.),offset_tracking_m=[.007,-.014,-.055])
        tt=torch.as_tensor(t,dtype=dtype,device=device)[None,:,None]
        expected=a.position[:,None]+v[:,None]*tt+.5*acceleration[:,None]*tt.square()
        tolerance=3e-6 if dtype==torch.float32 else 1e-10
        torch.testing.assert_close(out['position_origin_m'],expected,atol=tolerance,rtol=tolerance)
        outputs.append({key:value.cpu() for key,value in out.items()})
    for key in outputs[0]:
        torch.testing.assert_close(outputs[0][key],outputs[1][key],atol=tolerance*10,rtol=tolerance*10)
