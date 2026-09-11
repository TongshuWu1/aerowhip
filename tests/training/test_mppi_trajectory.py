import torch
import pytest
from planning.mppi_trajectory import interpolation_matrix,adaptive_weights,joint_quality
from planning.pva_job import validate_settings
from learning.pva_env import defaults


def test_contact_selection_retains_feasible_hit_over_higher_soft_score_miss():
    from planning.mppi_trajectory import candidate_index,candidate_better
    score=torch.tensor([900.,700.,800.,1000.],dtype=torch.float64)
    hit=torch.tensor([False,True,True,True]);failed=torch.tensor([False,False,False,True])
    assert candidate_index(score,hit,failed,True)==2
    assert candidate_index(score,hit,failed,False)==0
    assert candidate_better(800.,True,900.,False,True)
    assert not candidate_better(900.,False,800.,True,True)
    assert candidate_better(801.,True,800.,True,True)
    assert not candidate_better(800.,True,900.,False,False)
    hit[:]=False
    assert candidate_index(score,hit,failed,True)==0
    with pytest.raises(ValueError,match='No feasible'):
        candidate_index(score,hit,torch.ones_like(failed),True)


def test_contact_selection_is_versioned_and_requires_actual_tip_contact_task():
    cfg=defaults('mppi');cfg['mppi'].update(mode='open_loop',parameterization='control_points',
        support_points=10,proposal_count=4,target_ess_fraction=.2,selection_criterion='tip_contact_then_score_v1')
    cfg['task']['success_criterion']='legacy_strike_v1'
    with pytest.raises(ValueError,match='Contact-priority'):validate_settings(cfg)
    cfg['task']['success_criterion']='tip_contact_v1';validate_settings(cfg)
    cfg['mppi']['selection_criterion']='unknown'
    with pytest.raises(ValueError,match='Unknown MPPI candidate'):validate_settings(cfg)


def test_interpolation_preserves_constant_endpoints_and_bounds():
    basis=interpolation_matrix(45,10)
    torch.testing.assert_close(basis.sum(1),torch.ones(45,dtype=torch.float64))
    torch.testing.assert_close(basis[0],torch.eye(10,dtype=torch.float64)[0])
    torch.testing.assert_close(basis[-1],torch.eye(10,dtype=torch.float64)[-1])
    controls=torch.full((10,3),.3,dtype=torch.float64)
    torch.testing.assert_close(basis@controls,torch.full((45,3),.3,dtype=torch.float64))
    assert (torch.tanh(basis@(controls*1e4)).abs()<=1).all()


def test_temperature_targets_feasible_ess_and_ignores_invalid_samples():
    score=torch.linspace(-500,50,128,dtype=torch.float64)[None].repeat(2,1)
    score[0,:64]=-torch.inf;score[1,:]=-torch.inf
    w,temp,ess=adaptive_weights(score,.2)
    torch.testing.assert_close(w[0].sum(),torch.tensor(1.,dtype=torch.float64))
    assert not w[0,:64].any() and not w[1].any()
    assert abs(float(ess[0])-12.8)<1e-5 and ess[1]==0
    assert torch.isfinite(temp).all()


def test_weights_invariant_to_constant_score_shift():
    score=torch.tensor([[0.,1.,3.,7.]],dtype=torch.float64)
    torch.testing.assert_close(adaptive_weights(score,.7)[0],adaptive_weights(score+1e4,.7)[0])


def test_continuous_quality_rewards_joint_release_without_binary_wave_gate():
    task=defaults('mppi')['task'];t=lambda x:torch.tensor(x,dtype=torch.float64)
    def value(distance=.1,tip=4.,drone=-.5,backward=.1,reach=1.):
        return joint_quality(t(distance),t(tip),t(drone),t(backward),t(reach),task,.35)
    assert value(distance=.1)>value(distance=.7)
    assert value(tip=4)>value(tip=0)>0
    assert value(drone=-.5)>value(drone=.5)>0
    assert value(backward=.1)>value(backward=0)>0
    assert value(reach=1)>value(reach=0)>0


def test_control_point_mode_cannot_silently_change_receding_or_prior_contract():
    cfg=defaults('mppi');cfg['mppi'].update(parameterization='control_points',support_points=10,
        proposal_count=4,target_ess_fraction=.2)
    with pytest.raises(ValueError,match='offline'):validate_settings(cfg)
    cfg['mppi']['mode']='open_loop';validate_settings(cfg)
    cfg['mppi']['control_prior']=.2
    with pytest.raises(ValueError,match='zero control prior'):validate_settings(cfg)
