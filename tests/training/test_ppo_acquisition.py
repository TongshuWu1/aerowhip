from copy import deepcopy
from pathlib import Path
import torch
from learning.point_force_env import PointForceWhipEnvironment
from learning.deployment_rollout import sample_batch
from simulator.cable import DderState
from run_ppo import load_configs


def test_training_stops_at_exact_budget_with_partial_batch(tmp_path, monkeypatch):
    import run_ppo
    from run_ppo import train
    from simulator.workflow import atomic_json, read_json
    torch.set_num_threads(1)
    write_active = run_ppo.write_active_run
    monkeypatch.setattr(run_ppo, 'write_active_run', lambda _root, path: write_active(tmp_path, path))
    updates=[]
    original=run_ppo._atomic_json
    def capture(path,payload):
        if path.name=='status.json':
            updates.append(dict(payload))
        original(path,payload)
    monkeypatch.setattr(run_ppo,'_atomic_json',capture)
    m, t, c = deepcopy(load_configs())
    t['episode_duration_s'] = 0.2
    # The saved one-second action prior cannot describe this shortened fixture.
    c['bootstrap'] = None
    c['deployment']['enabled'] = False
    c['validation']['enabled'] = False
    c['ppo']['update_epochs'] = 1
    for name, data in [('model', m), ('task', t), ('ppo', c)]:
        atomic_json(tmp_path / 'config' / f'{name}.json', data)
    output = train(device_name='cpu', requested_episodes=4, batch_size=3,
                   artifact=tmp_path / 'run', resume_checkpoint=None,
                   config_directory=tmp_path / 'config')
    status = read_json(output / 'status.json')
    assert status['status'] == 'COMPLETED'
    assert status['episodes'] == 4
    assert Path((tmp_path / 'runs/ppo/ACTIVE_RUN.txt').read_text().strip()) == output
    assert any(row.get('phase_step',0)>0 and row['episodes']==0 for row in updates)
    assert any(row.get('stage')=='Updating PPO' for row in updates)
    import csv
    with (output / 'training_log.csv').open() as stream:
        assert [int(row['episodes']) for row in csv.DictReader(stream)] == [3, 4]


def test_physics_precision_is_independent_of_policy_precision():
    m,t,c=load_configs();c=deepcopy(c);c['physics_dtype']='float64'
    e=PointForceWhipEnvironment(m,t,c,batch_size=1,device=torch.device('cpu'))
    assert e.state.positions_m.dtype==torch.float64
    assert e.observe().dtype==torch.float32


def test_maximum_travel_cost_applies_to_misses_and_is_not_erased_by_return():
    import math
    m,t,c=deepcopy(load_configs())
    t['target_position_m']=[10.,0.,1.4]
    c['reward']['maximum_displacement_weight']=15.
    e=PointForceWhipEnvironment(m,t,c,batch_size=1,device=torch.device('cpu'))
    e.physics_steps_per_control=1
    initial=e.state.positions_m.clone()
    distances=iter([.35,.7,.35])
    def move(state,force,dt):
        q=initial.clone();q[:,:,0]+=next(distances)
        return e.model._result(state,DderState(q,torch.zeros_like(q)),force,dt)
    e.model.step_runtime=move
    costs=[e.step(torch.zeros(1,3)).components.maximum_displacement.item() for _ in range(3)]
    assert abs(costs[0]+15*math.log(2))<1e-10
    assert abs(sum(costs)+15*math.log(5))<1e-10
    assert costs[-1]==0 and not e.episode_success.item()


def test_world_speed_shaping_does_not_reward_a_tip_moving_away():
    m,t,c=deepcopy(load_configs())
    reference=PointForceWhipEnvironment(m,t,c,batch_size=1,device=torch.device('cpu'))
    t['target_position_m']=(reference.state.positions_m[0,-1]+torch.tensor([.1,0.,0.])).tolist()
    rewards={}
    for frame in ('world','attachment_relative'):
        c['reward']['directed_speed_shaping_reference']=frame
        e=PointForceWhipEnvironment(m,t,c,batch_size=1,device=torch.device('cpu'))
        e.physics_steps_per_control=1
        def transition(state,force,dt):
            v=torch.zeros_like(state.velocities_m_s)
            v[:,0,0]=-5.;v[:,-1,0]=-1.
            return e.model._result(state,DderState(state.positions_m,v),force,dt)
        e.model.step_runtime=transition
        rewards[frame]=e.step(torch.zeros(1,3)).components.strike_quality.item()
    assert rewards['world']==0 and rewards['attachment_relative']>0


def test_invalid_first_contact_cannot_become_success_inside_target():
    torch.set_num_threads(1)
    m,t,c=load_configs();t=deepcopy(t);t['success']['first_contact_only']=True
    e=PointForceWhipEnvironment(m,t,c,batch_size=1,device=torch.device('cpu'))
    q=e.state.positions_m.clone();q[:,-1]=e.target+q.new_tensor([-.06,0,0])
    e.reset(DderState(q,torch.zeros_like(q)));e.physics_steps_per_control=1
    speeds=iter([1.,5.])
    def transition(state,force,dt):
        q=state.positions_m.clone();q[:,-1,0]+=.02
        v=torch.zeros_like(q);v[:,-1,0]=next(speeds)
        return e.model._result(state,DderState(q,v),force,dt)
    e.model.step_runtime=transition
    first=e.step(torch.zeros(1,3))
    assert first.done.item()==1 and e.episode_invalid_tip_entry.item()
    second=e.step(torch.zeros(1,3))
    assert not e.episode_success.item() and not second.include_transition.item()


def test_bent_initial_states_preserve_lengths_and_velocity_constraints():
    torch.set_num_threads(1)
    m,t,c=load_configs();e=PointForceWhipEnvironment(m,t,c,batch_size=32,device=torch.device('cpu'))
    settings={**c['deployment'],'nominal_fraction':0.,'initial_cable_bend_deg':2.,'state_cable_bend_error_deg':.2}
    batch=sample_batch(e,settings,torch.Generator().manual_seed(9))
    for state in (batch.truth,batch.estimate):
        edges=state.positions_m[:,1:]-state.positions_m[:,:-1]
        lengths=edges.norm(dim=-1)
        torch.testing.assert_close(lengths,lengths.new_tensor(e.model.cable_configuration.rest_lengths_m).expand_as(lengths))
        dv=state.velocities_m_s[:,1:]-state.velocities_m_s[:,:-1]
        assert (edges*dv).sum(-1).abs().max()<1e-12
        directions=edges/lengths[...,None]
        assert (directions[:,1:]-directions[:,:-1]).abs().max()>.001


def test_failure_diagnostics_distinguish_bounds_from_nonfinite():
    torch.set_num_threads(1)
    m,t,c=load_configs();e=PointForceWhipEnvironment(m,t,c,batch_size=3,device=torch.device('cpu'))
    e.physics_steps_per_control=1
    def transition(state,force,dt):
        q=state.positions_m.clone();v=state.velocities_m_s.clone()
        q[0,0,0]=float('nan');q[1,0,0]=21.;v[2,0,0]=101.
        return e.model._result(state,DderState(q,v),force,dt)
    e.model.step_runtime=transition;e.step(torch.zeros(3,3))
    assert e.episode_nonfinite.tolist()==[True,False,False]
    assert e.episode_position_limit.tolist()==[False,True,False]
    assert e.episode_speed_limit.tolist()==[False,False,True]


def test_live_first_contact_scoring_cannot_recover_from_invalid_contact():
    from simulator.live_flight import LiveFlight
    m,t,c=load_configs();t=deepcopy(t);t['success']['first_contact_only']=True
    e=PointForceWhipEnvironment(m,t,c,batch_size=1,device=torch.device('cpu'))
    # Exercise the live scorer without changing the independent force-sequence controller.
    from types import SimpleNamespace
    scorer=SimpleNamespace(environment=e,tip_contact=False,non_tip_first=False,success=False)
    q=e.state.positions_m.clone();q[:,-1]=e.target+q.new_tensor([-.06,0,0])
    previous=DderState(q,torch.zeros_like(q))
    for speed in (1.,5.):
        q=previous.positions_m.clone();q[:,-1,0]+=.02;v=torch.zeros_like(q);v[:,-1,0]=speed
        scorer.state=DderState(q,v);LiveFlight._check_contact(scorer,previous);previous=scorer.state
    assert scorer.tip_contact and not scorer.success
