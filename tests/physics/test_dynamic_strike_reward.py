from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from simulator.cable import DderState
from learning.point_force_env import (PointForceWhipEnvironment, near_target_world_relative_quality,
    required_reach_allowance, excursion_cost)
from tools.dynamic_strike_setup import configure_dynamic_strike

ROOT = Path(__file__).resolve().parents[2]


def configs():
    values = [json.loads((ROOT/'config/research_30hz'/f'{n}.json').read_text()) for n in ('model','task','ppo')]
    return configure_dynamic_strike(*values)


def test_quality_rejects_carrying_and_retreat_only_and_is_bounded():
    world = torch.tensor([5., 5., 0., -1., 100., 5.], dtype=torch.float64)
    relative = torch.tensor([0., 6., 6., 7., 100., 6.], dtype=torch.float64)
    distance = world.new_tensor([0., 0., 0., 0., 0., 1.5])
    q = near_target_world_relative_quality(distance, world, relative,
        proximity_scale_m=.3, world_cap_m_s=4., relative_cap_m_s=6.)
    assert q[:5].tolist() == [0., 1., 0., 0., 1.]
    assert 0 < q[5] < .00001
    speeds = torch.linspace(0, 6, 100)
    increasing = near_target_world_relative_quality(torch.zeros_like(speeds), speeds+4, speeds,
        proximity_scale_m=.3, world_cap_m_s=4., relative_cap_m_s=6.)
    assert (torch.diff(increasing) >= 0).all()


def test_allowance_uses_each_initial_state_and_target_and_penalizes_only_excess():
    root = torch.tensor([[0.,0.,1.5], [1.,0.,1.5], [0.,0.,1.5]], dtype=torch.float64)
    target = root.new_tensor([[1.5,0.,1.4], [1.5,0.,1.4], [1.6,0.,1.4]])
    allowance = required_reach_allowance(root, target, .9525, .05, .25)
    torch.testing.assert_close(allowance, root.new_tensor([np.sqrt(2.26)-.7525, .25, np.sqrt(2.57)-.7525]))
    assert excursion_cost(allowance, allowance, .5).sum() == 0
    assert (excursion_cost(allowance+.5, allowance, .5) > 0).all()


def test_default_allowance_preserves_legacy_cost_exactly():
    d = torch.tensor([0., .1, .5, 1., 4.], dtype=torch.float64)
    torch.testing.assert_close(excursion_cost(d, torch.zeros_like(d), .35),
        torch.log1p((d/.35).square()), rtol=0, atol=0)


def test_scored_hit_prefers_relative_motion_without_changing_hit_gate(tmp_path):
    model, task, config = configs()
    env = PointForceWhipEnvironment(model, task, config, batch_size=3, device=torch.device('cpu'))
    # Scoring fixture: identical swept target entry, different root velocities.
    # Prescribed positions isolate reward accounting; no claim of physical rollout.
    q = env.state.positions_m.clone(); q[:,-1] = env.target - q.new_tensor([.1,0,0])
    previous = DderState(q, torch.zeros_like(q)); env.reset(previous)
    candidate_q = q.clone(); candidate_q[:,-1] = env.target
    velocity = torch.zeros_like(q); velocity[:,-1,0] = 5.; velocity[:,0,0] = q.new_tensor([5.,-1.,5.])
    velocity[2,-1,0] = 3.  # Invalid world speed, despite relative shaping options.
    candidate = DderState(candidate_q, velocity)
    transition = env.model._result(previous, candidate, env.hover_force_world_n.expand(3,-1), env.physics_dt_s)
    env.model.step_runtime = lambda *_: transition
    result = env.step(torch.zeros(3,3), stop_when_all_done=True)
    assert env.episode_success.tolist() == [True,True,False]
    assert result.components.success[:,0].tolist() == [250.,250.,0.]
    assert result.components.strike_quality[1] > result.components.strike_quality[0]
    assert env.episode_hit_relative_tip_directed_speed[:2].tolist() == [0.,6.]
    from learning.attempt_records import save_attempts
    save_attempts(tmp_path, env, 3, 1.)
    with np.load(tmp_path/'attempts/0000000003.npz') as data:
        assert data['hit_relative_tip_directed_speed_m_s'][:2].tolist() == [0.,6.]
        assert np.isnan(data['hit_relative_tip_directed_speed_m_s'][2])


def test_quality_budget_cannot_be_farmed_by_repeated_approaches():
    model, task, config = configs(); task['success']['tip_target_distance_m'] = .001
    env = PointForceWhipEnvironment(model, task, config, batch_size=1, device=torch.device('cpu'))
    env.physics_steps_per_control = 1
    totals = []
    for distance in (.2, .4, .2, .1):
        q = env.state.positions_m.clone(); v = torch.zeros_like(q)
        q[:,-1] = env.target - q.new_tensor([distance,0,0]); v[:,-1,0] = 5.; v[:,0,0] = -1.
        transition = env.model._result(env.state, DderState(q,v), env.hover_force_world_n, env.physics_dt_s)
        env.model.step_runtime = lambda *_: transition
        totals.append(float(env.step(torch.zeros(1,3)).components.strike_quality[0,0]))
    assert totals[0] > 0 and totals[1] == 0 and totals[2] == 0 and totals[3] > 0
    assert sum(totals) <= 120.


@pytest.mark.parametrize('key,value', [('relative_directed_speed_reward_cap_m_s', float('nan')),
    ('displacement_allowance_margin_m', -.1), ('displacement_allowance_mode','unknown')])
def test_bad_reward_settings_rejected(key,value):
    model,task,config = configs(); config['reward'][key] = value
    with pytest.raises(ValueError):
        PointForceWhipEnvironment(model,task,config,batch_size=1,device=torch.device('cpu'))
