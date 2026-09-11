import torch
import pytest
import numpy as np
from pathlib import Path
from learning.horizontal_strike import features,cost_rate,contact_bonus


def test_horizontal_quality_and_preparation_gating_are_soft():
    q=torch.zeros(2,5,3);v=torch.zeros_like(q)
    q[0,:,0]=torch.arange(5);q[1,:,2]=torch.arange(5)
    v[0,-1,0]=5;v[1,-1,2]=5
    weights=dict(proximity_scale_m=.35,vertical_excursion=20.,vertical_tip_velocity=40.,
        vertical_tip_alignment=40.,horizontal_contact=160.)
    bonus=contact_bonus(q,v,weights)
    torch.testing.assert_close(bonus,torch.tensor([160.,0.]))
    origin=torch.zeros(2,3);target=q[:,-1].clone()
    cost=cost_rate(q,v,origin,origin,target,torch.zeros(2,dtype=torch.bool),weights)
    assert not cost.any()
    cost=cost_rate(q,v,origin,origin,target,torch.ones(2,dtype=torch.bool),weights)
    torch.testing.assert_close(cost,torch.tensor([0.,80.]))
    elevated=origin.clone();elevated[:,2]=2
    cost=cost_rate(q,v,elevated,origin,target,torch.zeros(2,dtype=torch.bool),weights)
    torch.testing.assert_close(cost,torch.tensor([80.,80.]))
    assert torch.isfinite(cost).all()  # Costs grow; no new height rejection.


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_soft_reward_gpu_capture_matches_reference_without_changing_hit():
    from learning.pva_env import PVAEnvironment
    from simulator.workflow import read_json
    root=Path(__file__).resolve().parents[2];job=root/'runs/mppi_pva/20260909-160208-467697'
    if not (job/'plan.npz').exists():pytest.skip('Saved wave fixture unavailable')
    cfg=read_json(job/'settings.json');cfg['reward'].update(vertical_excursion=20.,vertical_tip_velocity=40.,vertical_tip_alignment=40.,horizontal_contact=160.,
        drone_approach=180.,lateral_excursion=180.,near_target_reach=120.,outward_contact=1000.)
    model=read_json(job/'model.json');ref=PVAEnvironment(model,cfg,root=job);fast=PVAEnvironment(model,cfg,root=job,batch_size=2)
    with np.load(job/'plan.npz') as z:actions=ref.tensor(z['normalized_jerk'])[None]
    a=ref.rollout(actions=actions,max_steps=actions.shape[1],trace=True)
    b=fast.rollout(actions=actions.expand(2,-1,-1),max_steps=actions.shape[1])
    assert a['success'][0] and b['success'].all() and not b['failed'].any()
    torch.testing.assert_close(a['reward'],b['reward'][:1],atol=1e-8,rtol=0)


def test_outward_reach_rewards_extension_not_carrier_translation():
    from learning.horizontal_strike import outward_reach,reach_cost_rate,reach_contact_bonus
    q=torch.zeros(2,5,3);q[0,:,0]=torch.linspace(0,1,5);q[1,:,2]=-torch.linspace(0,1,5)
    direction=torch.tensor([1.,0.,0.]);weights=dict(proximity_scale_m=.35,
        drone_approach=180.,lateral_excursion=180.,near_target_reach=120.,outward_contact=1000.)
    torch.testing.assert_close(outward_reach(q,direction,1.),torch.tensor([1.,0.]))
    torch.testing.assert_close(reach_contact_bonus(q+5,direction,1.,weights),torch.tensor([1000.,0.]))
    origin=torch.zeros(2,3);target=q[:,-1].clone()
    assert not reach_cost_rate(q,origin,origin,target,torch.zeros(2,dtype=torch.bool),direction,1.,weights).any()
    torch.testing.assert_close(reach_cost_rate(q,origin,origin,target,torch.ones(2,dtype=torch.bool),direction,1.,weights),torch.tensor([0.,120.]))
    moved=origin.clone();moved[:,0]=.5;moved[:,1]=.5
    torch.testing.assert_close(reach_cost_rate(q,moved,origin,target,torch.zeros(2,dtype=torch.bool),direction,1.,weights),torch.tensor([90.,90.]))
