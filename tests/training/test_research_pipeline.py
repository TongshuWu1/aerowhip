from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from simulator.workflow import read_json
from simulator.drone_pose_response import PoseResponseState,PoseResponseParameters,midpoint_step,CommandSchedule,predict_pose
from simulator.drone_pose_residual import DronePoseResidual
from simulator.research_pose import tensor_midpoint,PoseStepper,ResearchPoseModel,settled_initial
from simulator.research_reference import reference_packets,reference_feasibility,packet_cutoffs
from simulator.research_physics import ResearchPhysics

ROOT=Path(__file__).resolve().parents[2]


def test_native_clock_contract_rejects_silent_retiming_or_ignored_uncertainties():
    from copy import deepcopy
    from simulator.research_config import workspace_configs
    from run_ppo import validate_contract
    model,task,config=workspace_configs(ROOT);validate_contract(model,task,config)
    changed=deepcopy(task);changed['control_dt_s']=1/50
    with pytest.raises(ValueError,match='30 Hz'):validate_contract(model,changed,config)
    changed=deepcopy(task);changed['episode_duration_s']=.75
    with pytest.raises(ValueError,match='whole number'):validate_contract(model,changed,config)
    changed=deepcopy(config);changed['deployment']['damping_fraction']=.1
    with pytest.raises(ValueError,match='not modeled'):validate_contract(model,task,changed)


def test_tensor_pose_matches_checked_equations_and_invalid_rows():
    p=torch.zeros(3,3,dtype=torch.float64);r=torch.eye(3,dtype=p.dtype)[None].expand(3,-1,-1).clone()
    state=PoseResponseState(p,p.clone(),r,p.clone(),p.clone(),r.clone(),'prehover_effective_alignment',0.)
    params=PoseResponseParameters(6,21,6,2,.3,1.9,.022,.06,attitude_drive_model='independent_scale_v3',attitude_acceleration_scale_z=.46)
    cmd=p.new_zeros(3,11);cmd[:,6:9]=p.new_tensor([[2,1,1],[-1,2,-3],[4,0,-2]])
    nn=DronePoseResidual().double()
    with torch.no_grad():nn.net[-1].bias.fill_(.1)
    expected=midpoint_step(state,cmd,1/300,params,nn);actual,valid=tensor_midpoint(state,cmd,1/300,params,nn)
    assert valid.all()
    for name in ['position','velocity','rotation','omega_tracking']:torch.testing.assert_close(getattr(actual,name),getattr(expected,name),atol=1e-12,rtol=0)
    invalid=cmd.clone();invalid[1,6:8]=0.;invalid[1,8]=-params.gravity_m_s2/(params.feedforward_z*params.attitude_acceleration_scale_z)
    from simulator.research_pose import tensor_derivatives
    _,_,mask=tensor_derivatives(state,invalid,params,nn)
    assert mask.tolist()==[True,False,True]


def test_packet_reference_has_consistent_pva_without_hermite_ripple():
    dt=1/150;t=torch.arange(16,dtype=torch.float64)*dt;a=torch.tensor([2.,0.,-1.])
    v=t[None,:,None]*a;true_p=.5*t[None,:,None]**2*a
    # Model position has the known semiimplicit integration bias.
    measured_p=true_p+.5*(dt/8)*v
    packets,source=reference_packets(measured_p,v,dt,torch.tensor([15]),[0,0,-.055])
    torch.testing.assert_close(packets[0,:,:3]+packets.new_tensor([0,0,-.055]),true_p[0,::5],atol=1e-10,rtol=0)
    torch.testing.assert_close(packets[0,:,6:9],a.expand(4,3).double(),atol=1e-10,rtol=0)
    limits=dict(minimum_specific_vertical_m_s2=2.,maximum_tilt_deg=60.,maximum_specific_force_m_s2=20.,maximum_speed_m_s=5.)
    assert reference_feasibility(packets,torch.tensor([15]),dt,limits)[0].all()
    packets[:,1,8]=-10
    assert not reference_feasibility(packets,torch.tensor([15]),dt,limits)[0].any()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_pose_graph_and_complete_schedule_match_checked_predictor():
    model=read_json(ROOT/'config/research_30hz/model.json');execution=model['fullstate_execution']
    tracker=ResearchPoseModel(execution['checkpoint'],execution['sha256'],'cuda')
    root=torch.tensor([[0,0,1.5],[.03,-.02,1.52]],device='cuda',dtype=torch.float64)
    initial=settled_initial(root,torch.zeros_like(root),[0,0,-.055])
    packets=root.new_zeros(2,5,11);packets[:,:,:3]=initial.position[:,None]
    packets[:,:,6]=torch.tensor([0.,2.,-1.,1.,0.],device='cuda');packets[:,:,8]=.5
    hover=packets[:,0].clone();hover[:,3:9]=0.
    times=np.arange(21)/150;pt=np.arange(5)/30
    actual=tracker.predict(initial,packets,pt,times,[0,0,-.055],hover_command=hover)
    schedule=CommandSchedule(np.r_[-1.,pt],torch.cat((hover[:,None],packets),1),coverage_end_s=1.)
    expected=predict_pose(initial,schedule,times,tracker.parameters,offset_tracking_m=[0,0,-.055],residual=tracker.residual)
    assert actual['valid'].all()
    for key in ['position_origin_m','velocity_origin_m_s','rotation_tracking_to_world','position_attachment_m']:
        torch.testing.assert_close(actual[key],expected[key],atol=2e-10,rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_direct_cable_graph_matches_same_batched_eager_solver():
    from simulator.cable import CableConfiguration,DderModel,DderState
    payload=read_json(ROOT/'config/research_30hz/model.json');c=CableConfiguration.from_mapping(payload['cable'])
    physics=DderModel(c.dder_parameters(EI=payload['cable']['EI_n_m2'],Cb=payload['cable']['Cb_n_m2_s']))
    q=torch.zeros(2,c.node_count,3,device='cuda',dtype=torch.float64);q[:,:,2]=1.5-q.new_tensor([0.,*np.cumsum(c.rest_lengths_m)])
    state=DderState(q,torch.zeros_like(q));constants=physics.runtime_constants(q)
    fast=ResearchPhysics(physics,state,1/150,constants=constants,graph=True)
    slow=ResearchPhysics(physics,state,1/150,constants=constants,graph=False)
    a=b=state
    for step in range(5):
        root=q[:,0].clone();root[:,0]=.01*(step+1)
        a=fast(a,root);b=slow(b,root)
    torch.testing.assert_close(a.positions_m,b.positions_m,atol=1e-9,rtol=0)
    torch.testing.assert_close(a.velocities_m_s,b.velocities_m_s,atol=1e-8,rtol=0)


def test_hanging_estimate_and_packet_cutoff_do_not_use_truth_cable():
    from learning.point_force_env import PointForceWhipEnvironment
    from learning.deployment_rollout import sample_batch
    model,task,ppo=[read_json(ROOT/'config/research_30hz'/(n+'.json')) for n in ['model','task','ppo']]
    env=PointForceWhipEnvironment(model,task,ppo,batch_size=5,device=torch.device('cpu'))
    batch=sample_batch(env,dict(ppo['deployment'],nominal_fraction=0.),torch.Generator().manual_seed(9))
    assert torch.allclose(batch.estimate.positions_m[:,:,0],batch.estimate.positions_m[:,:1,0].expand(-1,12))
    assert (batch.truth.positions_m[:,:,0]-batch.truth.positions_m[:,:1,0]).abs().max()>0
    forces=torch.arange(15*2*3,dtype=torch.float64).reshape(15,2,3)
    output,cutoffs=packet_cutoffs(forces,torch.tensor([7,12]),5,15)
    assert cutoffs.tolist()==[10,15]
    torch.testing.assert_close(output[7:10,0],forces[6,0].expand(3,-1))


def test_failure_after_hit_removes_success_and_penalty_is_once():
    from learning.point_force_env import PointForceWhipEnvironment
    from learning.research_rollout import invalidate_attempts
    model,task,ppo=[read_json(ROOT/'config/research_30hz'/(n+'.json')) for n in ['model','task','ppo']]
    env=PointForceWhipEnvironment(model,task,ppo,batch_size=2,device=torch.device('cpu'));env.reset()
    env.episode_success[:]=True;env.episode_reward[:]=250.;env.episode_component_sums['success'][:]=200.;env.episode_hit_time_s[:]=.5
    invalid=torch.tensor([True,False]);invalidate_attempts(env,invalid)
    expected = 50. - ppo['reward']['numerical_failure_penalty']
    assert env.episode_reward.tolist()==[expected,250.]
    assert env.episode_success.tolist()==[False,True]
    invalidate_attempts(env,invalid);assert env.episode_reward.tolist()==[expected,250.]


@pytest.mark.parametrize('weight', [0., .2, 10.])
def test_open_loop_time_cost_covers_full_plan_after_any_early_contact(weight):
    from learning.point_force_env import PointForceWhipEnvironment
    from learning.research_rollout import charge_planned_duration
    from simulator.cable import DderState
    model,task,ppo=[read_json(ROOT/'config/research_30hz'/(n+'.json')) for n in ['model','task','ppo']]
    ppo['reward']['time_to_success_weight_per_s']=weight
    env=PointForceWhipEnvironment(model,task,ppo,batch_size=3,device=torch.device('cpu'))
    env.physics_steps_per_control=1
    q=env.state.positions_m.clone();q[:,-1]=env.target-q.new_tensor([.1,0,0])
    previous=DderState(q,torch.zeros_like(q));env.reset(previous)
    next_q=q.clone();next_q[:2,-1]=env.target[:2]
    velocity=torch.zeros_like(q);velocity[0,-1,0]=5.;velocity[1,-1,0]=1.
    transition=env.model._result(previous,DderState(next_q,velocity),env.hover_force_world_n,env.physics_dt_s)
    env.model.step_runtime=lambda *_:transition
    env.step(torch.zeros(3,3))
    assert env.episode_success.tolist()==[True,False,False]
    assert env.episode_invalid_tip_entry.tolist()==[False,True,False]
    assert env.active.tolist()==[False,False,True]
    other={k:v.clone() for k,v in env.episode_component_sums.items() if k!='time'}
    duration=env.episode_reward.new_tensor([2.,2.,5.])
    charge_planned_duration(env,duration)
    torch.testing.assert_close(env.episode_component_sums['time'],-weight*duration)
    torch.testing.assert_close(env.episode_reward,sum(other.values())-weight*duration)
    # Calling the finalizer twice cannot subtract twice; no hit/failure bonus changes.
    saved=env.episode_reward.clone();charge_planned_duration(env,duration)
    torch.testing.assert_close(env.episode_reward,saved,atol=0,rtol=0)
    for k,v in other.items():torch.testing.assert_close(env.episode_component_sums[k],v,atol=0,rtol=0)
