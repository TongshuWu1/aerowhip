import numpy as np
from planning.local_reference_correction import residual_vector,local_step,active_spline_basis
from planning.position_spline import PositionSpline


def test_local_solve_reduces_known_error_and_respects_jerk_limit():
    residual=np.array([2.,-3.]);jac=np.eye(2)
    delta=local_step(residual,jac,4,np.eye(2),np.zeros(2),np.array([.5,1.]))
    assert np.linalg.norm(residual+delta)<np.linalg.norm(residual)
    np.testing.assert_allclose(delta,[-.5,1.],atol=1e-6)


def test_conditioned_basis_preserves_inactive_control_and_settled_start():
    spline=PositionSpline(1.5);basis,active=active_spline_basis(spline,35)
    assert basis.shape==(27,24) and active.tolist()==list(range(8))
    assert np.all(basis[-3:]==0)
    import torch
    free=torch.tensor((basis@np.ones(24)).reshape(9,3),dtype=torch.float64)
    commands,_=spline.decode(free,[0,0,0])
    np.testing.assert_allclose(commands[0],0,atol=1e-12)


def test_residual_preserves_fixed_timing_and_mean_squared_3d_units():
    tip=np.array([[3.,4.,0.],[0.,0.,0.]])
    ref=dict(tip=np.zeros((2,3)),quadrotor=np.zeros((2,3)),command=np.zeros((2,3)))
    r=residual_vector(tip,np.zeros((2,3)),np.zeros((2,11)),ref,dict(tip=1,quadrotor=.1,command=.01))
    assert np.isclose(r@r,12.5)


def test_local_solve_respects_actual_command_speed_not_only_jerk():
    limits=dict(minimum_origin_z_m=.96,maximum_origin_z_m=2.8,
        maximum_speed_m_s=1.,maximum_specific_force_m_s2=18.,
        minimum_specific_vertical_m_s2=2.,maximum_tilt_deg=60.)
    packets=np.zeros((2,11));packets[:,2]=1.4
    mapping=np.zeros((2,11,1));mapping[:,3,0]=1.
    delta=local_step(np.array([-4.]),np.ones((1,1)),5.,np.ones((1,1)),
        np.zeros(1),np.array([60.]),packet_map=mapping,current_packets=packets,limits=limits)
    assert .99<delta[0]<=1.+1e-7


def test_local_residual_matches_original_torch_cost_for_full_3d_motion():
    import torch
    from planning.reference_correction import tracking_cost
    rng=np.random.default_rng(7)
    tip=rng.normal(size=(11,3));quad=rng.normal(size=(11,3));commands=rng.normal(size=(4,11))
    ref=dict(tip=rng.normal(size=tip.shape),quadrotor=rng.normal(size=quad.shape),command=rng.normal(size=(4,3)))
    weights=dict(tip=1.,quadrotor=.1,command=.01)
    residual=residual_vector(tip,quad,commands,ref,weights)
    cost,_=tracking_cost(torch.tensor(tip)[None],torch.tensor(quad)[None],torch.tensor(commands)[None],
        {k:torch.tensor(v) for k,v in ref.items()},weights)
    assert np.isclose(residual@residual,cost.item(),rtol=1e-12)
