from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from simulator.workflow import read_json
from simulator.cable import DderState
from learning.point_force_env import PointForceWhipEnvironment,POINT_FORCE_OBSERVATION_DIM
from learning.deployment_rollout import DeploymentBatch,plan_batch,trim_terminal_rollout
from learning.simple_ppo import PPORollout

ROOT=Path(__file__).resolve().parents[2]


def scenario(size=4):
    model,task,config=[read_json(ROOT/'config/research_30hz'/f'{n}.json') for n in ('model','task','ppo')]
    config['deployment'].update(termination='execution_success_or_timeout',strike_followthrough_s=0.)
    config['reward']['time_to_success_weight_per_s']=10.
    config['reward']['timeout_penalty']=7.
    config['reward']['numerical_failure_penalty']=500.  # Fixture independent of editable workspace rewards.
    task['episode_duration_s']=2/30
    env=PointForceWhipEnvironment(model,task,config,batch_size=size,device=torch.device('cpu'))
    q=env.state.positions_m.clone();q[:,-1]=env.target-q.new_tensor([.1,0,0])
    state=DderState(q,torch.zeros_like(q));env.reset(state)
    one=q.new_ones(size)
    batch=DeploymentBatch(state,state,one,one,one[:,None].expand(-1,3),one[:,None]*0,
        torch.ones(size,dtype=torch.bool),env.target.clone())
    return env,batch


@pytest.mark.parametrize('early_failure',[False,True])
def test_each_execution_stops_on_hit_and_ignores_its_unused_tail(monkeypatch,early_failure):
    import learning.research_terminal as terminal
    env,batch=scenario();size=env.batch_size;steps=10
    q=batch.truth.positions_m
    roots=q[:,0,None].expand(-1,steps+1,-1).clone()
    monkeypatch.setattr(terminal,'frozen_reference',lambda *a:(roots,torch.zeros_like(roots)))
    original=terminal.reference_packets
    def packets(*a,**kw):
        out,reference=original(*a,**kw);out[0,0 if early_failure else 1,8]=-20.
        return out,reference
    monkeypatch.setattr(terminal,'reference_packets',packets)
    def predict(initial,packets,pt,times,offset,**kw):
        assert torch.isfinite(packets).all()
        valid=torch.ones(size,len(times),dtype=torch.bool);valid[0,8:]=False
        return dict(position_attachment_m=roots,position_origin_m=roots-roots.new_tensor(offset),
            rotation_tracking_to_world=torch.eye(3,dtype=q.dtype)[None,None].expand(size,len(times),3,3),valid=valid)
    env._research_tracker=SimpleNamespace(predict=predict)
    calls=[]
    def cable(previous,root):
        step=len(calls)+1;calls.append(step)
        positions=previous.positions_m.clone();vel=torch.zeros_like(positions)
        for row,hit in ((0,2),(1,5),(3,2)):
            if step==hit:
                positions[row,-1]=env.target[row];vel[row,-1,0]=1. if row==3 else 5.
        if step==3:positions[3,-1]=env.target[3]-q.new_tensor([.1,0,0])
        if step==5:positions[3,-1]=env.target[3];vel[3,-1,0]=5.
        if step>2:positions[0,-1]=torch.nan  # Speculative tail must not invalidate row 0.
        return DderState(positions,vel)
    env._research_cable=cable
    traces=[]
    forces=env.hover_force_world_n.expand(steps,size,3).clone()
    # A future virtual-planner failure likewise cannot erase an execution hit.
    env._planning_failure_steps=torch.tensor([9,11,11,11])
    score=terminal.execute_terminal_batch(env,batch,forces,torch.full((size,),steps),env.ppo_config['deployment'],
        trace=lambda i,f,s,*a:traces.append(s.positions_m.clone()))
    assert score.episode_success.tolist()==[not early_failure,True,False,False]
    assert score.terminal_steps.tolist()==[1 if early_failure else 2,5,10,10]
    assert score.command_cutoffs.tolist()==[5,5,10,10]
    assert score.episode_timed_out.tolist()==[False,False,True,True]
    assert score.episode_component_sums['timeout'].tolist()==[0.,0.,-7.,-7.]
    assert score.failed.tolist()==[early_failure,False,False,False]
    assert score.deployment['reference_infeasible'].tolist()==[early_failure,False,False,False]
    assert not score.deployment['pose_domain_failure'].any()
    assert not score.deployment['planning_failure'].any()
    assert score.episode_invalid_tip_entry[3]
    torch.testing.assert_close(score.episode_component_sums['time'],-10.*score.terminal_steps.double()/150)
    torch.testing.assert_close(sum(score.episode_component_sums.values()),score.episode_reward)
    if not early_failure:
        torch.testing.assert_close(score.episode_reward[0]-score.episode_reward[1],q.new_tensor(10.*3/150))
    else:
        assert score.episode_component_sums['numerical_failure'][0]==-500.
        assert score.episode_component_sums['success'][0]==0.
    for state in traces[2:]:torch.testing.assert_close(state[0],traces[1][0],rtol=0,atol=0)


def test_virtual_hit_does_not_cut_off_execution_candidate():
    env,batch=scenario(1)
    q=batch.truth.positions_m.clone();q[:,-1]=env.target
    v=torch.zeros_like(q);v[:,-1,0]=5.
    def step(previous,force,dt):return env.model._result(previous,DderState(q,v),force,dt)
    env.model.step_runtime=step
    agent=SimpleNamespace(deterministic_action=lambda obs:torch.zeros(len(obs),3))
    forces,cutoffs=plan_batch(env,agent,batch)
    assert env.episode_success.all() and env.episode_hit_time_s[0]==1/150
    assert len(forces)==10 and cutoffs.tolist()==[10]
    assert len(env._virtual_reference_positions)==11


def test_credit_excludes_all_actions_after_execution_terminal():
    rollout=PPORollout.allocate(150,3,POINT_FORCE_OBSERVATION_DIM,3,device=torch.device('cpu'))
    rollout.masks.fill_(1);rollout.rewards.fill_(123);rollout.dones.fill_(0)
    trim_terminal_rollout(rollout,torch.tensor([151,300,750]),5)
    assert rollout.masks[:,:,0].sum(0).tolist()==[31,60,150]
    assert rollout.dones[:,:,0].sum(0).tolist()==[1,1,1]
    assert rollout.dones[30,0,0]==1 and rollout.dones[59,1,0]==1 and rollout.dones[149,2,0]==1
    assert rollout.rewards.sum()==0


def test_export_uses_short_prefix_and_only_completes_terminal_packet():
    from deployment.research_rehearsal import rehearsal_prefix,complete_packets
    env,batch=scenario(1);q=batch.truth.positions_m
    roots=q[:,0,None].expand(1,11,3).clone();roots[:,:,0]=torch.arange(11,dtype=q.dtype)/100
    packets=q.new_zeros(1,3,11);packets[0,:,:3]=roots[0,::5]-q.new_tensor([0,0,-.055])
    frames=[q[0].numpy().copy() for _ in range(3)]
    def cable(state,root):
        p=state.positions_m.clone();p[:,0]=root
        return DderState(p,state.velocities_m_s)
    env._research_cable=cable
    score=SimpleNamespace(execution_state=batch.truth,command_cutoffs=torch.tensor([5]),reference_packets=packets,
        predicted_pose=dict(position_attachment_m=roots,position_origin_m=roots-q.new_tensor([0,0,-.055]),
            valid=torch.ones(1,11,dtype=torch.bool)))
    whip,finished,state,pose=rehearsal_prefix(score,frames,env)
    assert len(finished)==6 and len(whip)==2 and pose['valid'].shape[1]==6
    torch.testing.assert_close(state.positions_m[:,0],roots[:,5])
    np.testing.assert_array_equal(whip,packets[0,:2].numpy())
    times,commands,phases,recovery=complete_packets(whip,whip[0,:3])
    np.testing.assert_array_equal(commands[:2],whip)
    assert times[1]==1/30 and phases[2]!=1
