import pytest
import torch
from learning.pva_success import impact_bonus,record_hit_velocity
from learning.pva_env import defaults
from planning.pva_job import validate_settings


def test_more_directed_impact_speed_earns_more_even_above_old_four_m_s_gate():
    velocity=torch.tensor([[2.,0,0],[4.,0,0],[8.,0,0],[4.,0,8.],[-8.,0,0]])
    yes=torch.ones(5,dtype=torch.bool);direction=torch.tensor([[1.,0,0]]).expand(5,-1)
    bonus=impact_bonus({'impact':200.,'impact_scale_m_s':4.},yes,~yes,velocity,direction)
    torch.testing.assert_close(bonus,torch.tensor([40.,100.,160.,100.,0.]))
    assert bonus[0]<bonus[1]<bonus[2]<200


def test_miss_failure_and_invalid_impact_cannot_earn_bonus():
    velocity=torch.tensor([[9.,0,0],[9.,0,0],[float('nan'),0,0]])
    result=impact_bonus({'impact':200},torch.tensor([False,True,True]),torch.tensor([False,True,False]),velocity,torch.tensor([[1.,0,0]]))
    assert not result.any()
    assert not impact_bonus({},torch.ones(3,dtype=torch.bool),torch.zeros(3,dtype=torch.bool),velocity,torch.tensor([[1.,0,0]])).any()


def test_success_velocity_interpolates_at_tip_hit_and_ignores_other_events():
    old=torch.tensor([[2.,0,0],[2.,0,0]]);new=torch.tensor([[10.,0,0],[10.,0,0]])
    saved=torch.zeros_like(old)
    value=record_hit_velocity(old,new,torch.tensor([.25,.8]),torch.tensor([True,False]),saved)
    torch.testing.assert_close(value,torch.tensor([[4.,0,0],[0.,0,0]]))
    torch.testing.assert_close(record_hit_velocity(new,new*10,torch.ones(2),torch.zeros(2,dtype=torch.bool),value),value)


def test_defaults_replace_timing_preference_and_preserve_ppo():
    cfg=defaults('mppi');assert cfg['reward']['impact']==200 and cfg['reward'].get('early_hit',0)==0
    assert defaults('ppo')['reward'].get('impact',0)==0
    cfg['reward']['impact_scale_m_s']=0
    with pytest.raises(ValueError,match='impact speed scale'):validate_settings(cfg)
