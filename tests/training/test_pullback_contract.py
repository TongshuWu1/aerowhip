from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import pytest
from learning.pva_env import PVAEnvironment,update_pullback
from simulator.workflow import read_json

ROOT=Path(__file__).resolve().parents[2]
CRITERIA=dict(require_pullback=True,minimum_pull_distance_m=.25,minimum_pull_speed_m_s=1.,
    minimum_backward_distance_m=.1,minimum_backward_speed_m_s=.5)


def test_seed_bank_is_bounded_direct_jerk_and_contains_forward_backward_commands():
    from planning.mppi_receding import pullback_seed_bank
    from simulator.pva_commands import jerk_packets
    env=SimpleNamespace(direction=torch.tensor([1.,0.,0.]),limit=torch.tensor([60.,60.,60.]),
        tensor=lambda x:torch.as_tensor(x,dtype=torch.float64))
    bank=pullback_seed_bank(env,60)
    assert bank.shape==(321,60,3) and bank.abs().max()<1 and not bank[0].any()
    packets=jerk_packets(torch.zeros(321,11,dtype=torch.float64),bank*env.limit)
    assert bool(((packets[:,:,3].max(1).values>1)&(packets[:,-1,3]<-.5)).any())


def test_pull_then_reverse_order_and_direction_with_no_repeat_reward():
    env=SimpleNamespace(settings=dict(task=CRITERIA,reward=dict(pull_phase=20.,reverse_phase=40.)),
        origin0=torch.zeros(3,3),direction=torch.tensor([1.,0.,0.]),
        pose=SimpleNamespace(position=torch.zeros(3,3),velocity=torch.zeros(3,3)),
        pull_peak=torch.zeros(3),pull_credit=torch.zeros(3),reverse_credit=torch.zeros(3),
        pull_ready=torch.zeros(3,dtype=torch.bool),reverse_ready=torch.zeros(3,dtype=torch.bool))
    active=torch.ones(3,dtype=torch.bool)
    env.pose.position[:,0]=-.15;env.pose.velocity[:,0]=-1.
    bonus,allowed=update_pullback(env,active)
    assert not allowed.any() and not bonus.any()
    env.pose.position[1:,0]=.3;env.pose.velocity[1:,0]=1.5
    bonus,allowed=update_pullback(env,active)
    torch.testing.assert_close(bonus,torch.tensor([0.,20.,20.]));assert not allowed.any()
    env.pose.position[:,0]=.18;env.pose.velocity[:,0]=torch.tensor([-.8,-.8,.8])
    bonus,allowed=update_pullback(env,active)
    assert allowed.tolist()==[False,True,False]
    torch.testing.assert_close(bonus,torch.tensor([0.,40.,0.]))
    bonus,_=update_pullback(env,active);assert not bonus.any()
    env.pose.velocity[1,0]=.8
    _,allowed=update_pullback(env,active);assert not allowed.any()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_previous_forward_only_hit_is_rejected_and_phase_state_matches_gpu_replay():
    job=ROOT/'runs/mppi_pva/20260909-131121-677417'
    if not (job/'plan.npz').exists():pytest.skip('Saved forward-only diagnostic unavailable')
    cfg=read_json(job/'settings.json');cfg['task'].update(CRITERIA);cfg['reward'].update(pull_phase=20.,reverse_phase=40.)
    model=read_json(job/'model.json')
    fast=PVAEnvironment(model,cfg,root=job,batch_size=2)
    reference=PVAEnvironment(model,cfg,root=job)
    with np.load(job/'plan.npz') as z:actions=fast.tensor(z['normalized_jerk'])[None]
    for k in range(20):reference.step(actions[:,k],trace=True)
    fast.branch_from(reference)
    a=fast.rollout(actions=actions[:,20:].expand(2,-1,-1),max_steps=len(actions[0])-20)
    b=reference.rollout(actions=actions[:,20:],max_steps=len(actions[0])-20,trace=True)
    assert reference.contact[0] and reference.pull_ready[0]
    assert not reference.success[0] and not reference.reverse_ready[0]
    assert not a['success'].any() and not a['failed'].any()
    torch.testing.assert_close(a['reward'][:1],b['reward'],atol=1e-8,rtol=0)
    for name in ('pull_ready','reverse_ready','pull_peak','pull_credit','reverse_credit'):
        torch.testing.assert_close(getattr(fast,name)[:1],getattr(reference,name),atol=1e-9,rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_reversal_must_qualify_at_interpolated_contact_not_at_the_end_of_its_tick():
    job=ROOT/'runs/mppi_pva/20260909-133459-410882'
    if not (job/'plan.npz').exists():pytest.skip('Saved interval-end diagnostic unavailable')
    cfg=read_json(job/'settings.json');env=PVAEnvironment(read_json(job/'model.json'),cfg,root=job)
    with np.load(job/'plan.npz') as z:actions=env.tensor(z['normalized_jerk'])[None]
    result=env.rollout(actions=actions,max_steps=actions.shape[1],trace=True)
    assert env.pull_ready[0] and env.reverse_ready[0] and env.contact[0]
    assert not result['success'][0] and not result['failed'][0]
