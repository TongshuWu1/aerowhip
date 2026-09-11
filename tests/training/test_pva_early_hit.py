import math
import pytest
import torch
from learning.pva_success import hit_time_bonus
from learning.pva_env import defaults
from planning.pva_job import validate_settings


def test_earlier_valid_hits_get_more_bonus_without_rewarding_early_failure():
    time=torch.tensor([.8,1.2,.1,.1,float('nan'),float('inf'),-1.],dtype=torch.float64)
    success=torch.tensor([True,True,False,True,False,True,True])
    failed=torch.tensor([False,False,False,True,False,False,False])
    bonus=hit_time_bonus({'early_hit':100.,'early_hit_scale_s':1.},success,failed,time)
    assert bonus[0]==pytest.approx(100*math.exp(-.8))
    assert bonus[1]==pytest.approx(100*math.exp(-1.2))
    assert bonus[0]>bonus[1]>0 and not bonus[2:].any()


def test_missing_weight_preserves_old_scores_and_ppo_defaults():
    time=torch.tensor([.2,1.],dtype=torch.float64);hit=torch.ones(2,dtype=torch.bool)
    assert not hit_time_bonus({},hit,~hit,time).any()
    assert defaults('mppi')['reward'].get('early_hit',0)==0
    assert defaults('ppo')['reward'].get('early_hit',0)==0


def test_bonus_is_once_per_hit_and_reversible_not_per_remaining_timestep():
    time=torch.tensor([.7],dtype=torch.float64);yes=torch.tensor([True]);no=~yes
    credits=[hit_time_bonus({'early_hit':100},s,f,time) for s,f in [(no,no),(yes,no),(yes,no),(no,yes)]]
    increments=[b-a for a,b in zip(credits[:-1],credits[1:])]
    assert increments[0]>0 and increments[1]==0 and increments[2]==-increments[0]
    assert sum(increments)==0


def test_hit_time_is_absolute_and_has_no_lookahead_or_episode_duration_dependency():
    hit=torch.tensor([True]);time=torch.tensor([1.1]);bad=~hit
    cfg=defaults('mppi')
    baseline=hit_time_bonus(cfg['reward'],hit,bad,time)
    cfg['task']['duration_s']=5.;cfg['mppi']['horizon_s']=.4
    torch.testing.assert_close(baseline,hit_time_bonus(cfg['reward'],hit,bad,time))
    cfg['reward']['early_hit_scale_s']=0
    with pytest.raises(ValueError,match='early-hit time scale'):validate_settings(cfg)
