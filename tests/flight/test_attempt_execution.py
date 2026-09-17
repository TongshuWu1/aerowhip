from copy import deepcopy
from types import SimpleNamespace
import torch
from tests.force_config import load_configs
from learning.point_force_env import PointForceWhipEnvironment
from learning.deployment_rollout import sample_batch,plan_batch,execute_batch
from simulator.cable import DderState


def test_execute_missed_plan_once_without_predicted_success():
    torch.set_num_threads(1)
    m,t,c=deepcopy(load_configs());t['episode_duration_s']=.2
    c['deployment'].update(require_predicted_success=False,nominal_fraction=1.,recovery_duration_s=.5)
    env=PointForceWhipEnvironment(m,t,c,batch_size=2,device=torch.device('cpu'))
    batch=sample_batch(env,c['deployment'],torch.Generator().manual_seed(1))
    agent=SimpleNamespace(deterministic_action=lambda obs:torch.zeros(len(obs),3))
    forces,cutoffs=plan_batch(env,agent,batch)
    assert cutoffs.tolist()==[20,20]
    assert not env.episode_success.any()
    score=execute_batch(env,batch,forces,cutoffs,c['deployment'])
    assert score.deployment['planned'].all()
    assert not score.deployment['predicted_valid_hit'].any()
    assert score.deployment['recovered'].all()
    torch.testing.assert_close(sum(score.episode_component_sums.values()),score.episode_reward)


def test_mixed_invalid_first_contact_and_timeout_have_frozen_cutoffs():
    torch.set_num_threads(1)
    m,t,c=deepcopy(load_configs());t['episode_duration_s']=.2
    c['deployment'].update(require_predicted_success=False,nominal_fraction=1.)
    base=PointForceWhipEnvironment(m,t,c,batch_size=2,device=torch.device('cpu'))
    t['target_position_m']=(base.state.positions_m[0,-1]+torch.tensor([.1,0.,0.])).tolist()
    env=PointForceWhipEnvironment(m,t,c,batch_size=2,device=torch.device('cpu'))
    batch=sample_batch(env,c['deployment'],torch.Generator().manual_seed(1))
    def transition(state,force,dt):
        q=state.positions_m.clone();q[0,-1,0]+=.2
        v=torch.zeros_like(q);v[0,-1,0]=1.
        return env.model._result(state,DderState(q,v),force,dt)
    env.model.step_runtime=transition
    agent=SimpleNamespace(deterministic_action=lambda obs:torch.zeros(len(obs),3))
    _,cutoffs=plan_batch(env,agent,batch)
    tail = round(c['deployment']['strike_followthrough_s'] / env.physics_dt_s)
    assert cutoffs.tolist()==[1+tail,20]
    assert not env.episode_success.any()
    c['deployment']['require_predicted_success']=True
    _,legacy_cutoffs=plan_batch(env,agent,batch)
    assert legacy_cutoffs.tolist()==[0,0]


def test_early_hit_records_impact_speed_before_episode_horizon():
    torch.set_num_threads(1)
    m,t,c=deepcopy(load_configs());t['episode_duration_s']=.2
    env=PointForceWhipEnvironment(m,t,c,batch_size=1,device=torch.device('cpu'))
    env.target=env.state.positions_m[:,-1]+torch.tensor([.1,0.,0.])
    def transition(state,force,dt):
        q=state.positions_m.clone();q[:,-1,0]+=.2
        v=torch.zeros_like(q);v[:,-1,0]=5.
        return env.model._result(state,DderState(q,v),force,dt)
    env.model.step_runtime=transition
    env.step(torch.zeros(1,3),stop_when_all_done=True)
    assert env.episode_success.item()
    assert env.episode_hit_time_s.item()==.01
    assert env.episode_impact_speed.item()==5.
    env.step(torch.zeros(1,3))
    assert env.episode_impact_speed.item()==5.
