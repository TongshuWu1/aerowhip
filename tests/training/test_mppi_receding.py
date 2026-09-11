from pathlib import Path
import numpy as np
import torch
import pytest
from learning.pva_env import PVAEnvironment,defaults
from simulator.workflow import read_json
from planning.pva_job import validate_settings
from planning.mppi_receding import shift_proposal

ROOT=Path(__file__).resolve().parents[2]


def test_horizon_and_episode_are_separate_and_shift_appends_neutral_jerk():
    cfg=defaults('mppi');validate_settings(cfg)
    assert cfg['mppi']['horizon_s']==2. and cfg['task']['duration_s']==5.
    x=torch.arange(36).reshape(12,3)
    shifted=shift_proposal(x)
    torch.testing.assert_close(shifted[:-1],x[1:]);assert not shifted[-1].any()
    cfg['mppi']['horizon_s']=.41
    with pytest.raises(ValueError,match='lookahead'):validate_settings(cfg)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_nonzero_time_branch_preserves_delayed_packets_and_does_not_modify_source():
    job=ROOT/'runs/mppi_pva/20260909-121945-649228'
    if not (job/'model.json').exists():pytest.skip('Frozen diagnostic model unavailable')
    cfg=defaults();cfg['task']['duration_s']=1.2;model=read_json(job/'model.json')
    actual=PVAEnvironment(model,cfg,root=job)
    branch=PVAEnvironment(model,cfg,root=job,batch_size=3)
    rng=np.random.default_rng(78);actions=actual.tensor(rng.normal(0,.02,(1,36,3)))
    for k in range(7):actual.step(actions[:,k])
    pose=actual.pose.position.clone();q=actual.state.positions_m.clone();packet=actual.command.clone()
    branch.branch_from(actual)
    # A bad independent candidate must not contaminate the other rows or source.
    branch.command[1,2]=.5
    result=branch.rollout(actions=actions[:,7:19].expand(3,-1,-1),max_steps=12)
    assert result['failed'][1] and not result['failed'][0]
    assert actual.index==7 and branch.index==19 and branch.active[0]
    torch.testing.assert_close(actual.pose.position,pose,atol=0,rtol=0)
    torch.testing.assert_close(actual.state.positions_m,q,atol=0,rtol=0)
    torch.testing.assert_close(actual.command,packet,atol=0,rtol=0)
    for k in range(7,19):actual.step(actions[:,k])
    torch.testing.assert_close(branch.pose.position[:1],actual.pose.position,atol=1e-9,rtol=0)
    torch.testing.assert_close(branch.state.positions_m[:1],actual.state.positions_m,atol=1e-9,rtol=0)
    torch.testing.assert_close(branch.state.velocities_m_s[:1],actual.state.velocities_m_s,atol=1e-8,rtol=0)
    torch.testing.assert_close(branch.total[:1],actual.total,atol=1e-9,rtol=0)
    assert len(branch.packets)==20 and len(actual.packets)==20


def test_predictions_do_not_stop_committed_maneuver_and_manual_stop_keeps_partial(tmp_path,monkeypatch):
    import planning.mppi_receding as planner
    class Environment:
        device=torch.device('cpu');steps=15
        def __init__(self,*args,batch_size=1,**kwargs):
            self.batch_size=batch_size;self.index=0;self.actions=[]
            self.active=torch.ones(batch_size,dtype=torch.bool);self.success=torch.zeros_like(self.active);self.failed=torch.zeros_like(self.active)
            self.contact=torch.zeros_like(self.active)
            self.total=torch.zeros(batch_size,dtype=torch.float64);self.minimum_distance=torch.ones_like(self.total)
            self.termination_time=torch.full_like(self.total,.5)
        def tensor(self,x):return torch.as_tensor(x,dtype=torch.float64)
        def branch_from(self,actual):self.index=actual.index;self.total=actual.total.expand(self.batch_size).clone()
        def rollout(self,actions,max_steps):
            # An optimistic candidate hit must never become an actual hit.
            return dict(reward=self.total+1,success=torch.ones(self.batch_size,dtype=torch.bool),failed=self.failed,
                minimum_tip_distance_m=torch.full_like(self.total,.03))
        def step(self,action,trace=False):
            self.actions.append(action.clone());self.index+=1;self.total+=1
            if self.index==14:self.success[:]=True;self.active[:]=False;self.termination_time[:]=14/30;self.minimum_distance[:]=.04
    monkeypatch.setattr(planner,'PVAEnvironment',Environment)
    monkeypatch.setattr(planner,'terminal_value',lambda env,s:torch.zeros_like(env.total))
    cfg=defaults('mppi');cfg['task']['duration_s']=.5
    cfg['visualization']={'live_mppi':False}
    cfg['mppi'].update(initialization='zero',horizon_s=.4,samples=2,minimum_iterations=1,patience=1)
    cfg['mppi'].update(initial_minimum_iterations=5,initial_patience=2)
    result=planner.optimize(tmp_path,{},cfg)
    assert result['command_steps']==14 and result['stop_reason']=='modeled_hit'
    assert result['maneuver_duration_s']>.4 and len(read_json(tmp_path/'windows.json'))==14
    rows=read_json(tmp_path/'windows.json')
    assert rows[0]['iterations']==5 and rows[1]['iterations']==2
    with np.load(tmp_path/'plan.npz') as data:
        assert data['plan_complete'] and data['normalized_jerk'].shape==(14,3)
    # A real cooperative STOP preserves only actions actually committed so far.
    partial=tmp_path/'partial';partial.mkdir()
    class StoppingEnvironment(Environment):
        def step(self,action,trace=False):
            super().step(action,trace=trace)
            if self.index==5:(partial/'STOP').touch()
    monkeypatch.setattr(planner,'PVAEnvironment',StoppingEnvironment)
    with pytest.raises(InterruptedError):planner.optimize(partial,{},cfg)
    with np.load(partial/'plan.npz') as data:
        assert not data['plan_complete'] and data['normalized_jerk'].shape==(5,3)
    resumed=tmp_path/'resumed';resumed.mkdir()
    np.savez(resumed/'continuation.npz',prefix=np.zeros((5,3)),proposal=np.zeros((12,3)))
    monkeypatch.setattr(planner,'PVAEnvironment',Environment)
    result=planner.optimize(resumed,{},cfg)
    assert result['command_steps']==14
    assert read_json(resumed/'history.json')[0]['command_step']==5
    assert all(row['reused_prefix'] for row in read_json(resumed/'windows.json')[:5])
    with np.load(resumed/'plan.npz') as data:np.testing.assert_array_equal(data['normalized_jerk'][:5],0.)

    # A longer seed must slide through a shorter lookahead without extending
    # the actual rollout or silently discarding its later release commands.
    seeded=tmp_path/'seeded';seeded.mkdir();seed=np.linspace(-.2,.2,45).reshape(15,3)
    np.savez(seeded/'initial_proposal.npz',normalized_jerk=seed)
    counts=[]
    class SeedEnvironment(Environment):
        def rollout(self,actions,max_steps):
            counts.append(max_steps)
            assert actions.shape[1]==12
            return super().rollout(actions,max_steps)
    monkeypatch.setattr(planner,'PVAEnvironment',SeedEnvironment)
    monkeypatch.setattr(planner,'sample_noise',lambda actual,s,samples,horizon,rng:torch.zeros(samples,horizon,3,dtype=torch.float64))
    planner.optimize(seeded,{},cfg)
    assert max(counts)==12 and read_json(seeded/'initialization.json')['seed_command_steps']==15
    with np.load(seeded/'plan.npz') as data:np.testing.assert_allclose(data['normalized_jerk'],seed[:14],atol=1e-15)


def test_strike_exit_guidance_penalizes_climbing_accelerating_handover_without_changing_hit():
    from types import SimpleNamespace
    from planning.mppi_receding import terminal_value
    cfg=defaults('mppi');s=cfg['mppi'].copy()
    for k in s:
        if k.startswith('terminal_'):s[k]=0.
    s.update(strike_exit_speed=.5,strike_exit_climb=5.,strike_exit_acceleration=1.)
    env=SimpleNamespace(state=SimpleNamespace(positions_m=torch.zeros(3,2,3),velocities_m_s=torch.zeros(3,2,3)),
        target=torch.zeros(3,3),direction=torch.tensor([1.,0.,0.]),settings=cfg,
        engine=SimpleNamespace(cable=SimpleNamespace(rest_lengths_m=[1.])),pose=SimpleNamespace(position=torch.zeros(3,3)),
        command=torch.zeros(3,11),success=torch.tensor([True,True,False]),failed=torch.zeros(3,dtype=torch.bool))
    env.command[:,3:6]=torch.tensor([3.,0.,1.]);env.command[0,6:9]=torch.tensor([3.,0.,1.])
    env.command[1,6:9]=torch.tensor([-3.,0.,-1.])
    hit=env.success.clone();values=terminal_value(env,s)
    torch.testing.assert_close(values,torch.tensor([-20.,-10.,0.]))
    assert torch.equal(env.success,hit)
