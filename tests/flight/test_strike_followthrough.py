import pytest
import torch
from simulator.strike_sequence import freeze_followthrough


def test_frozen_tail_uses_each_rows_last_force_and_preserves_horizon_and_refusal():
    forces=torch.arange(6*3*3).reshape(6,3,3).float()
    cutoffs=torch.tensor([2,6,0])
    result,ends=freeze_followthrough(forces,cutoffs,dt_s=.01,maximum_steps=6,duration_s=.02)
    assert ends.tolist()==[4,6,0]
    torch.testing.assert_close(result[:2,0],forces[:2,0])
    torch.testing.assert_close(result[2:4,0],forces[1,0].expand(2,-1))
    torch.testing.assert_close(result[:,1],forces[:,1])


def test_zero_followthrough_is_exact_legacy_sequence():
    forces=torch.randn(6,2,3);cutoffs=torch.tensor([2,6])
    result,ends=freeze_followthrough(forces,cutoffs,dt_s=.01,maximum_steps=6)
    assert result is forces and ends is cutoffs


@pytest.mark.parametrize('duration',[-.01,.015,float('nan')])
def test_invalid_followthrough_is_rejected(duration):
    with pytest.raises(ValueError):
        freeze_followthrough(torch.zeros(5,1,3),torch.tensor([3]),dt_s=.01,maximum_steps=5,duration_s=duration)


def test_live_and_training_compile_the_same_single_strike_tail():
    from copy import deepcopy
    from types import SimpleNamespace
    from run_ppo import load_configs
    from simulator.cable import DderState
    from simulator.strike_plan import compile_strike_plan
    from learning.point_force_env import PointForceWhipEnvironment
    from learning.deployment_rollout import sample_batch,plan_batch
    torch.set_num_threads(1)
    m,t,p=deepcopy(load_configs());t['episode_duration_s']=.2
    p['deployment'].update(strike_followthrough_s=.02,require_predicted_success=False,nominal_fraction=1.)
    base=PointForceWhipEnvironment(m,t,p,batch_size=1,device=torch.device('cpu'))
    t['target_position_m']=(base.state.positions_m[0,-1]+torch.tensor([.1,0.,0.])).tolist()
    env=PointForceWhipEnvironment(m,t,p,batch_size=1,device=torch.device('cpu'))
    batch=sample_batch(env,p['deployment'],torch.Generator().manual_seed(1))
    calls=[]
    def policy(obs):
        calls.append(1)
        return torch.tensor([[.1,-.2,.3]])
    def physics(q,v,force):
        q=q.clone();q[:,-1,0]+=.2
        v=torch.zeros_like(q);v[:,-1,0]=5.
        return q,v
    def transition(state,force,dt):
        q,v=physics(state.positions_m,state.velocities_m_s,force)
        return env.model._result(state,DderState(q,v),force,dt)
    env.model.step_runtime=transition
    forces,cutoffs=plan_batch(env,SimpleNamespace(deterministic_action=policy),batch)
    assert cutoffs.tolist()==[3] and len(calls)==1
    plan=compile_strike_plan(m,t,p,batch.estimate,policy,physics=physics)
    assert len(calls)==2 and plan.duration_s==.03
    torch.testing.assert_close(plan.forces_world_n,forces[:,0])
    torch.testing.assert_close(forces[:,0],forces[0,0].expand(3,-1))
