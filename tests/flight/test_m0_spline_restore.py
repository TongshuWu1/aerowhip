import json
from pathlib import Path
import numpy as np
import torch
from planning.position_spline import PositionSpline,fit_bounded_positions,RetainedM0SplineProposals
from planning.pva_job import validate_settings

ROOT=Path(__file__).resolve().parents[2]


def test_bounded_conversion_preserves_a_representable_motion_and_limits():
    spline=PositionSpline(1.5);origin=torch.tensor([0.,0.,1.255],dtype=torch.float64)
    free=origin+torch.linspace(0,.015,9,dtype=torch.float64)[:,None]*torch.tensor([1.,.1,.3])
    packets,_=spline.decode(free,origin)
    converted=fit_bounded_positions(spline,packets[:,:3],origin,torch.tensor([60.]*3))
    replay,_=spline.decode(converted,origin)
    np.testing.assert_allclose(replay[:,:3],packets[:,:3],atol=1e-5)
    np.testing.assert_allclose(replay[0,3:9],0,atol=1e-12)
    assert spline.jerk_valid(converted,origin,torch.tensor([60.]*3))


def test_retained_sampler_zero_controls_decode_exact_selected_seed():
    spline=PositionSpline(1.5);seed=torch.randn(9,3,dtype=torch.float64)
    cfg=dict(samples=16,position_noise_scales_m=[.005,.015,.04])
    sampler=RetainedM0SplineProposals(seed,cfg,spline.scratch_noise_basis())
    means=torch.zeros(4,9,3,dtype=torch.float64)
    np.testing.assert_array_equal(sampler.decode(means),seed.expand(4,-1,-1))
    samples=sampler.sample(means,torch.Generator().manual_seed(657))
    assert samples.shape==(4,4,9,3) and torch.isfinite(samples).all()
    assert (sampler.decode(samples)-seed).abs().max()>0


def test_m0_profile_retains_objective_and_uses_soft_angle():
    cfg=json.loads((ROOT/'config/pva/systematic_strike.json').read_text())
    validate_settings(cfg)
    assert cfg['launch']['origin_m']==[0.,0.,1.4]
    assert cfg['launch']['target_m']==[1.25,0.,1.25]
    assert cfg['task']['success_criterion']=='tip_contact_v1'
    assert cfg['task']['angle_mode']=='soft_reward'
    assert cfg['trajectory_objective']['schema']=='preferred_fold_v1'
    assert 'minimum_tip_speed_gain_m_s' not in cfg['trajectory_objective']
    assert cfg['mppi']['proposal_parameterization']=='position_bspline'
    assert cfg['recovery']['schema']=='brake_return_hold_v1'


def test_straighter_cast_gets_more_reward_without_angle_discontinuity():
    from planning.whip_objective import encounter_quality
    from learning.pva_success import contact_success
    angle=torch.deg2rad(torch.tensor([0.,15.,30.,44.99,45.01,60.],dtype=torch.float64))
    velocity=torch.stack((4*torch.ones_like(angle),4*torch.tan(angle),torch.zeros_like(angle)),-1)
    cfg=json.loads((ROOT/'config/pva/systematic_strike.json').read_text());n=len(angle)
    _,cast,_=encounter_quality(torch.zeros(n),velocity,torch.tensor([[-1.,0.,0.]]).repeat(n,1),
        torch.ones(n),torch.ones(n),torch.tensor([1.,0.,0.]),torch.ones(n,dtype=torch.bool),
        torch.ones(n,dtype=torch.bool),cfg['task'],.35)
    assert torch.all(cast[:-1]>cast[1:]) and cast[-1]>0
    assert abs(float(cast[3]-cast[4]))<.001
    assert contact_success(cfg['task'],torch.ones(n,dtype=torch.bool),torch.zeros(n),torch.zeros(n,dtype=torch.bool)).all()
