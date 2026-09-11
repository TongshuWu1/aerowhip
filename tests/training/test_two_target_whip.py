from copy import deepcopy
import pytest
import torch
from learning.pva_env import defaults
from learning.two_target_whip import crossings,validate,completion_terms
from learning.pva_success import TWO_TARGET,label


def state():
    return (torch.zeros((1,2),dtype=torch.bool),torch.full((1,2),torch.inf,dtype=torch.float64),
        torch.zeros((1,2,3),dtype=torch.float64),torch.full((1,2),10.,dtype=torch.float64))


def tick(a,b,s,*,running=True,left=0.):
    p=lambda x:torch.tensor([[x,0.,0.]],dtype=torch.float64)
    return crossings(p(a),p(b),p(3.),p(6.),torch.tensor([[1.,0,0],[2.,0,0]],dtype=torch.float64),
        .1,*s,torch.tensor([running]),torch.tensor([left]),1.)


def test_two_distinct_entries_in_one_tick_preserve_order_and_velocity():
    hits,times,velocities,minimum,finished,fraction=tick(0,3,state())
    assert hits.tolist()==[[True,True]] and finished.item()
    torch.testing.assert_close(times,torch.tensor([[.3,1.9/3]],dtype=torch.float64))
    torch.testing.assert_close(velocities[0,:,0],torch.tensor([3.9,4.9],dtype=torch.float64))
    assert minimum.max()<1e-12


def test_reverse_visit_cannot_count_second_target_or_its_distance():
    out=tick(3,0,state())
    assert out[0].tolist()==[[True,False]] and not out[-2].item()
    assert out[3][0,1]>.89
    # Passing T2 again, now after T1, is an ordered hit. T1 is never recounted.
    again=tick(0,3,out[:4],left=1.)
    assert again[0].all() and again[-2].item()
    assert again[1][0,0]==out[1][0,0] and again[1][0,1]>1


def test_first_contact_does_not_finish_and_repeated_hits_do_not_recount():
    first=tick(0,1.5,state())
    assert first[0].tolist()==[[True,False]] and not first[-2].item()
    second=tick(1.5,2.5,first[:4],left=1.)
    assert second[-2].item()
    repeated=tick(0,3,second[:4],left=2.)
    assert not repeated[-2].item()
    torch.testing.assert_close(repeated[1],second[1])
    torch.testing.assert_close(repeated[2],second[2])


def test_invalid_rollout_cannot_earn_any_contact_or_distance_credit():
    initial=state();out=tick(0,3,initial,running=False)
    assert not out[0].any() and not out[-2].any()
    for a,b in zip(initial,out[:4]):torch.testing.assert_close(a,b)


def test_sequence_is_explicit_frozen_and_nonoverlapping():
    cfg=defaults('mppi');cfg['mppi']['mode']='open_loop';cfg['launch']['target_radius_m']=0
    cfg['mppi']['parameterization']='control_points';cfg['trajectory_objective']={'schema':'preferred_fold_v1'}
    cfg['task'].update(success_criterion=TWO_TARGET,target_sequence_m=[cfg['launch']['target_m'],[2.,.3,1.]])
    validate(cfg);assert label(cfg['task'])=='Tip reaches T1 then T2'
    for key,value,match in [('target_sequence_m',[cfg['launch']['target_m']]*2,'non-overlapping'),
                          ('target_sequence_m',[[0,0,1],[2,0,1]],'must agree')]:
        broken=deepcopy(cfg);broken['task'][key]=value
        with pytest.raises(ValueError,match=match):validate(broken)
    cfg['method']='ppo'
    with pytest.raises(ValueError,match='offline MPPI'):validate(cfg)


def test_completing_sweep_drives_progress_and_partial_hit_has_no_speed_bonus():
    from types import SimpleNamespace
    env=SimpleNamespace(settings={'task':{'two_target_reward':'completion_v2'}},
        target_minimum_distances=torch.tensor([[.02,.3],[.02,.1],[.02,.02]]),
        target_hits=torch.tensor([[True,False],[True,False],[True,True]]),failed=torch.zeros(3,dtype=torch.bool))
    style=dict(fold=torch.tensor([600.,100.,100.]),impact=torch.tensor([700.,300.,400.]),
        first_target_credit=torch.ones(3)*450,second_target=torch.zeros(3))
    result=completion_terms(env,dict(sequence_progress=1000.,sequence_style_fraction=.1,proximity_scale_m=.35),style)
    assert result['impact'].tolist()==[0.,0.,400.]
    score=sum(result.values());assert score[0]<score[1]<score[2]
    env.settings['task'].clear();assert completion_terms(env,{},style) is style
