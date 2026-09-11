from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch
from planning.ppo_whip import settings_from_mppi,release_learning_settings,joint_strike_settings,first_contact_settings
from planning.pva_job import validate_settings,make_agent,save_checkpoint,load_policy
from learning.pva_env import PVAEnvironment,defaults
from simulator.workflow import read_json

ROOT=Path(__file__).resolve().parents[2]
PARENT=ROOT/'runs/mppi_pva/20260909-173305-488091'


def test_ppo_uses_same_task_but_independent_exploration_rewards():
    mppi=read_json(ROOT/'config/pva/mppi.json');original=deepcopy(mppi)
    cfg=settings_from_mppi(mppi,defaults('ppo'));validate_settings(cfg)
    assert mppi==original and cfg['method']=='ppo'
    assert cfg['task']==mppi['task'] and cfg['limits']==mppi['limits']
    assert cfg['reward']['vertical_excursion']<mppi['reward']['vertical_excursion']
    assert cfg['reward']['reach_progress']>0 and cfg['launch']['target_radius_m']==.05
    assert cfg['task']['duration_s']==5. and cfg['training']['gae_lambda']==.99
    cfg.pop('observation_contract')
    with pytest.raises(ValueError,match='observable whip phase'):validate_settings(cfg)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('release',[False,True,'joint','first-contact'])
def test_ppo_whip_phase_capture_reset_and_policy_roundtrip(tmp_path,release):
    if not (PARENT/'plan.npz').exists():pytest.skip('Saved MPPI fixture unavailable')
    cfg=settings_from_mppi(read_json(PARENT/'settings.json'),defaults('ppo'))
    if release:cfg=release_learning_settings(cfg)
    if release in ('joint','first-contact'):cfg=joint_strike_settings(cfg)
    if release=='first-contact':cfg=first_contact_settings(cfg)
    validate_settings(cfg)
    model=read_json(PARENT/'model.json')
    ref=PVAEnvironment(model,cfg,root=PARENT)
    fast=PVAEnvironment(model,cfg,root=PARENT,batch_size=2)
    with np.load(PARENT/'plan.npz') as z:actions=ref.tensor(z['normalized_jerk'])[None]
    result=ref.rollout(actions=actions,max_steps=actions.shape[1],trace=True)
    actual=fast.rollout(actions=actions.expand(2,-1,-1),max_steps=actions.shape[1])
    assert result['success'][0] and actual['success'].all()
    assert ref.tip_contact.all() and fast.tip_contact.all()
    torch.testing.assert_close(result['reward'],actual['reward'][:1],atol=1e-8,rtol=0)
    assert 0<ref.reach_credit[0]<=1 and ref.wave_stage[0]==3
    torch.testing.assert_close(ref.observation()[:,-4:],fast.observation()[:1,-4:])
    # Identical geometry with different phase history must be distinguishable.
    observed=ref.observation();ref.reach_credit.zero_()
    assert not torch.equal(observed,ref.observation())
    fast.reset();assert not fast.reach_credit.any() and not fast.wave_stage.any() and not fast.brake_credit.any() and not fast.tip_contact.any()
    agent=make_agent(ref,cfg);save_checkpoint(tmp_path/'policy.pt',agent,ref,0)
    loaded=load_policy(tmp_path/'policy.pt',ref,cfg)
    torch.testing.assert_close(agent.deterministic_action(observed),loaded.deterministic_action(observed),atol=0,rtol=0)
    if release=='joint':
        continued=load_policy(tmp_path/'policy.pt',ref,cfg,optimizer=True,policy_only=True)
        torch.testing.assert_close(agent.deterministic_action(observed),continued.deterministic_action(observed),atol=0,rtol=0)
        assert not torch.equal(agent.value(observed),continued.value(observed))
        assert not continued.policy_optimizer.state and not continued.value_optimizer.state
    old=deepcopy(cfg);old.pop('observation_contract')
    old_env=PVAEnvironment(model,old,root=PARENT)
    with pytest.raises(ValueError,match='observation contract'):load_policy(tmp_path/'policy.pt',old_env,old)
    if release=='first-contact':
        # A tip touch at rest cannot pass the strike gates. It ends this PPO
        # attempt, and later inactive transitions cannot earn phase rewards.
        fast.reset(target=fast.state.positions_m[0,-1].clone())
        fast.step(torch.zeros(2,3,device=fast.device))
        assert fast.tip_contact.all() and fast.contact.all()
        assert not fast.active.any() and not fast.success.any() and not fast.failed.any()
        total=fast.total.clone()
        _,reward,_,mask=fast.step(torch.zeros(2,3,device=fast.device))
        assert not reward.any() and not mask.any()
        torch.testing.assert_close(total,fast.total,atol=0,rtol=0)


def test_manual_stop_reports_latest_completed_history(tmp_path,monkeypatch):
    import planning.pva_job as jobs
    from experimental_data.io import atomic_json
    cfg=defaults('ppo')
    atomic_json(tmp_path/'settings.json',cfg);atomic_json(tmp_path/'model.json',{})
    atomic_json(tmp_path/'status.json',dict(status='prepared',attempts=0))
    def fake_ppo(job,model,settings):
        atomic_json(job/'status.json',dict(status='running',attempts=2048))
        atomic_json(job/'history.json',[dict(attempts=4096,success=.25)])
        raise InterruptedError('Stopped after saving update')
    monkeypatch.setattr(jobs,'ppo',fake_ppo)
    with pytest.raises(InterruptedError):jobs.run(tmp_path)
    status=read_json(tmp_path/'status.json')
    assert status['status']=='stopped' and status['attempts']==4096
    assert status['success']==.25 and not (tmp_path/'worker.lock').exists()


def test_braking_credit_starts_before_reversal_and_cannot_be_farmed():
    from types import SimpleNamespace
    from learning.pva_env import update_pullback
    cfg=release_learning_settings(settings_from_mppi(read_json(ROOT/'config/pva/mppi.json'),defaults('ppo')))
    env=SimpleNamespace(settings=cfg,direction=torch.tensor([1.,0.,0.]),origin0=torch.zeros(1,3),
        pose=SimpleNamespace(position=torch.tensor([[.3,0.,0.]]),velocity=torch.tensor([[1.,0.,0.]])),
        pull_peak=torch.zeros(1),pull_ready=torch.zeros(1,dtype=torch.bool),pull_credit=torch.zeros(1),
        reverse_credit=torch.zeros(1),reverse_ready=torch.zeros(1,dtype=torch.bool),brake_credit=torch.zeros(1))
    running=torch.ones(1,dtype=torch.bool)
    update_pullback(env,running)
    env.pose.velocity[:,0]=.25
    reward,allowed=update_pullback(env,running)
    assert reward.item()==pytest.approx(30.) and env.brake_credit.item()==pytest.approx(.5)
    assert not allowed.any() and not env.reverse_ready.any()
    env.pose.velocity[:,0]=1.;update_pullback(env,running)
    env.pose.velocity[:,0]=.25
    assert update_pullback(env,running)[0].item()==0
    cfg.pop('observation_contract')
    with pytest.raises(ValueError,match='v3 brake credit'):validate_settings(cfg)


def test_omitting_quarantined_tail_preserves_gae():
    from learning.simple_ppo import generalized_advantage_estimate
    reward=torch.tensor([1.,2.,3.,0.,0.])[:,None,None]
    done=torch.tensor([0.,0.,1.,1.,1.])[:,None,None]
    mask=torch.tensor([1.,1.,1.,0.,0.])[:,None,None]
    value=torch.tensor([.2,.3,.4,.4,.4])[:,None,None]
    full=generalized_advantage_estimate(reward,done,mask,value,gamma=1.,gae_lambda=.99)
    cut=generalized_advantage_estimate(reward[:3],done[:3],mask[:3],value[:3],gamma=1.,gae_lambda=.99)
    for a,b in zip(full,cut):torch.testing.assert_close(a[:3],b,atol=0,rtol=0)


def test_joint_guidance_rewards_closer_aligned_simultaneous_release():
    from learning.ppo_strike import joint_quality
    cfg=joint_strike_settings(release_learning_settings(settings_from_mppi(read_json(ROOT/'config/pva/mppi.json'),defaults('ppo'))))
    validate_settings(cfg);task=cfg['task'];direction=torch.tensor([1.,0.,0.])
    def quality(distance=.15,tip=(4.,0.,0.),drone=(-.5,0.,0.),retreat=.1,pull=True,wave=1.):
        return joint_quality(torch.tensor([distance]),torch.tensor([tip]),torch.tensor([drone]),
            torch.tensor([retreat]),torch.tensor([pull]),torch.tensor([wave]),direction,task,.35).item()
    assert 0<quality()<quality(.05)<quality(.0)<=1
    assert quality(drone=(.5,0.,0.))<quality(drone=(-.25,0.,0.))<quality()
    assert 0<quality(wave=.5)<quality()
    assert quality(tip=(4.,0.,5.))<quality()
    assert quality(tip=(-4.,0.,0.))==0 and quality(pull=False)==0
    assert cfg['task']==read_json(ROOT/'config/pva/mppi.json')['task']
