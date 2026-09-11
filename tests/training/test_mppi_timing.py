import torch
import pytest
from planning.mppi_timing import TimedProposals,phase_durations,retime_jerk
from planning.mppi_trajectory import interpolation_matrix
from planning.pva_job import validate_settings
from learning.pva_env import defaults


def settings():
    return dict(proposal_count=4,samples=16,timing_source_knots=[0,.3,.6,1],
        control_point_noise_scales=[.03,.09,.2],timing_noise_scales=[.12,.3,.6],strength_noise_scales=[.1,.2,.4])


def test_zero_parameters_preserve_each_exact_baseline_and_family():
    base=torch.randn(2,45,3,dtype=torch.float64,generator=torch.Generator().manual_seed(1)).tanh()
    sampler=TimedProposals(base,interpolation_matrix(45,10),settings())
    actual=sampler.decode(torch.zeros(4,13,3,dtype=torch.float64))
    torch.testing.assert_close(actual,base[[0,1,0,1]],atol=2e-14,rtol=0)


def test_retiming_integrates_short_pulses_instead_of_missing_them():
    # One 1/12 s source pulse in the first phase compressed by 2x. Its area
    # becomes half a command interval; a knot-only sample could miss it.
    base=torch.zeros(1,12,3,dtype=torch.float64);base[:,1]=.8
    duration=torch.tensor([[[1/6]*3,[1/3]*3,[1/2]*3]],dtype=torch.float64)
    out=retime_jerk(base,duration,torch.tensor([0,1/3,2/3,1],dtype=torch.float64))
    torch.testing.assert_close(out.sum(1),torch.full((1,3),.4,dtype=torch.float64),atol=1e-14,rtol=0)
    assert out.abs().max()<=.8 and torch.count_nonzero(out)>0


def test_axis_timing_and_strength_are_independently_editable_and_bounded():
    base=torch.zeros(2,45,3,dtype=torch.float64);base[:,:13]=.5;base[:,13:27]=-.4
    sampler=TimedProposals(base,interpolation_matrix(45,10),settings())
    params=torch.zeros(4,13,3,dtype=torch.float64);params[:,-3,0]=.6
    shifted=sampler.decode(params)
    assert (shifted[:,:,0]-base[0,:,0]).abs().max()>.1
    torch.testing.assert_close(shifted[:,:,1:],base[[0,1,0,1],:,1:],atol=1e-14,rtol=0)
    params.zero_();params[:,-1,2]=.3
    stronger=sampler.decode(params)
    assert stronger[0,0,2]>.5
    torch.testing.assert_close(stronger[:,:,:2],base[[0,1,0,1],:,:2],atol=1e-14,rtol=0)
    params.fill_(100);out=sampler.decode(params)
    assert torch.isfinite(out).all() and out.abs().max()<=1
    duration=phase_durations(params,sampler.source_knots)
    assert (duration>0).all()
    torch.testing.assert_close(duration.sum(-2),torch.ones(4,3,dtype=torch.float64))


def test_batched_decoding_equals_decoding_each_draw_and_rng_reproducible():
    base=torch.linspace(-.8,.9,270,dtype=torch.float64).reshape(2,45,3)
    sampler=TimedProposals(base,interpolation_matrix(45,10),settings())
    means=torch.zeros(4,13,3,dtype=torch.float64)
    a=sampler.sample(means,torch.Generator().manual_seed(65));b=sampler.sample(means,torch.Generator().manual_seed(65))
    torch.testing.assert_close(a,b)
    batch=sampler.decode(a)
    for i in range(4):torch.testing.assert_close(batch[:,i],sampler.decode(a[:,i]),atol=1e-14,rtol=0)


def test_timing_configuration_rejects_wrong_order_and_missing_baseline_families():
    cfg=defaults('mppi');cfg['mppi'].update(settings(),mode='open_loop',parameterization='control_points',
        proposal_parameterization='timed_baselines_v1',support_points=10,target_ess_fraction=.2)
    cfg['trajectory_objective']={'schema':'preferred_fold_v1'};validate_settings(cfg)
    cfg['mppi']['timing_source_knots']=[0,.7,.5,1]
    with pytest.raises(ValueError,match='ordered'):validate_settings(cfg)
    cfg['mppi']['timing_source_knots']=[0,.3,.6,1];cfg['mppi']['proposal_count']=1
    with pytest.raises(ValueError,match='both baseline'):validate_settings(cfg)
