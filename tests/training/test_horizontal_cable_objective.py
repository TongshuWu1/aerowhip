import math
from types import SimpleNamespace

import pytest
import torch

from planning.strike_objective import (OBJECTIVE,FOLD,horizontal_cable_error,
    event_terms,initialize,observe,StrikeCapture)


def test_horizontal_error_distinguishes_hanging_cable_and_preserves_geometry():
    s=torch.linspace(0,1,11,dtype=torch.float64)
    flat=torch.stack((s,s*0,s*0),-1)[None]
    hanging=torch.stack((s*0,s*0,-s),-1)[None]
    assert float(horizontal_cable_error(flat))==0
    assert float(horizontal_cable_error(hanging))==pytest.approx(1/math.sqrt(3))
    tilted=flat.clone();tilted[...,2]=s*.4
    moved=tilted@torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],dtype=s.dtype).T+torch.tensor([3.,-2.,7.])
    midpoint=(tilted[:,1:]+tilted[:,:-1])/2
    fine=torch.cat((torch.stack((tilted[:,:-1],midpoint),2).flatten(1,2),tilted[:,-1:]),1)
    torch.testing.assert_close(horizontal_cable_error(tilted),horizontal_cable_error(moved))
    torch.testing.assert_close(horizontal_cable_error(tilted),horizontal_cable_error(fine))
    args=(s.new_tensor([.01]),s.new_tensor([[0.,4.,0.]]),s.new_tensor([0.,1.,0.]))
    legacy=event_terms(*args,OBJECTIVE)
    assert 'horizontal_cable' not in legacy
    settings=dict(OBJECTIVE,horizontal_cable_weight=4.,horizontal_cable_scale_m=.1)
    with pytest.raises(ValueError,match='co-timed'):event_terms(*args,settings)
    assert sum(event_terms(*args,settings,horizontal_error_m=horizontal_cable_error(flat)).values())>sum(event_terms(*args,settings,horizontal_error_m=horizontal_cable_error(hanging)).values())


def test_event_selection_and_capture_use_the_same_cable_geometry():
    s=torch.linspace(0,1,11,dtype=torch.float64)
    flat=torch.stack((s,s*0,s*0),-1)[None]
    hanging=torch.stack((s*0+1,s*0,1-s),-1)[None]
    settings=dict(OBJECTIVE,horizontal_cable_weight=4.,horizontal_cable_scale_m=.1)
    env=SimpleNamespace(active=torch.ones(1,dtype=torch.bool),total=s.new_zeros(1),
        cutoff=torch.zeros(1,dtype=torch.long),failed=torch.zeros(1,dtype=torch.bool),
        target=flat[:,-1].clone(),dt=.01,wave_material=s[1:-1],
        settings=dict(fold_constraint=FOLD,trajectory_objective=settings,fold_requirement='diagnostic_only'),
        direction=s.new_tensor([0.,1.,0.]),encounter_distance=s.new_full((1,),10.),
        encounter_tip_velocity=s.new_zeros(1,3),encounter_time=s.new_zeros(1),encounter_q=flat.clone())
    velocity=torch.zeros_like(flat);velocity[:,-1,1]=4.
    initialize(env)
    # Both endpoints share tip position and velocity; only cable geometry changes.
    observe(env,SimpleNamespace(positions_m=flat,velocities_m_s=velocity),
        SimpleNamespace(positions_m=hanging,velocities_m_s=velocity),env.active,s.new_tensor([1.]))
    assert bool(env.strike_valid[0])
    assert float(env.strike_time[0])==1.
    assert float(env.strike_horizontal_error_m[0])==0.
    score,terms=StrikeCapture().score(env,dict(failed=env.failed),s.new_zeros(1,2,3),settings)
    torch.testing.assert_close(score,env.strike_value)
    assert float(terms['horizontal_cable'][0])==0.
