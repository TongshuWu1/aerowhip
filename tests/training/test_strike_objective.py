import math
from types import SimpleNamespace

import pytest
import torch

from planning.strike_objective import (FOLD, OBJECTIVE, advance_fold,
    fold_features, initialize, observe, strike_terms)


def hairpin(center, amplitude=math.pi, nodes=201):
    material = torch.linspace(0, 1, nodes, dtype=torch.float64)
    midpoint = (material[1:]+material[:-1])/2
    theta = amplitude*(1+torch.tanh((midpoint-center)/.045))/2
    edges = torch.stack((theta.cos(), theta.sin(), torch.zeros_like(theta)), -1)/(nodes-1)
    q = torch.cat((torch.zeros(1, 3, dtype=edges.dtype), edges.cumsum(0)))
    return q[None], material


def classify(sequence, material):
    tracking = complete = torch.zeros(1, dtype=torch.bool)
    location = torch.zeros(1, dtype=torch.float64)
    count = torch.zeros(1, dtype=torch.long)
    for q in sequence:
        tracking, location, count, complete = advance_fold(
            tracking, location, count, complete, fold_features(q, material, FOLD),
            torch.ones(1, dtype=torch.bool), FOLD)
    return bool(complete)


def test_travelling_fold_passes_but_static_shallow_and_reverse_bends_do_not():
    centers = torch.linspace(.25, .85, 25).tolist()
    moving = [hairpin(c)[0] for c in centers]
    _, material = hairpin(.25)
    assert classify(moving, material)
    assert not classify([moving[0]]*25, material)
    assert not classify([hairpin(c, math.pi/4)[0] for c in centers], material)
    assert not classify([hairpin(c)[0] for c in reversed(centers)], material)


def test_rigid_swing_and_rotating_static_fold_do_not_count_as_propagation():
    folded, material = hairpin(.25)
    straight = torch.stack((material, material*0, material*0), -1)[None]
    rotate = []
    for angle in torch.linspace(0, 2, 30).tolist():
        c, s = math.cos(angle), math.sin(angle)
        rotate.append(torch.tensor([[c,-s,0],[s,c,0],[0,0,1]],dtype=torch.float64))
    assert not classify([straight@r.T for r in rotate], material)
    assert not classify([folded@r.T for r in rotate], material)


def test_disconnected_peak_jump_is_rejected():
    q, material = hairpin(.25)
    assert not classify([q]*5+[hairpin(.85)[0]]*10, material)


def test_features_preserve_translation_rotation_and_edge_subdivision():
    q, material = hairpin(.45, nodes=41)
    baseline = fold_features(q, material, FOLD)
    r = torch.tensor([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]],dtype=torch.float64)
    transformed = fold_features(q@r.T+torch.tensor([2.,-3.,5.]), material, FOLD)
    midpoint = (q[:,1:]+q[:,:-1])/2
    fine = torch.stack((q[:,:-1],midpoint),2).flatten(1,2)
    fine = torch.cat((fine,q[:,-1:]),1)
    refined = fold_features(fine,torch.linspace(0,1,81,dtype=torch.float64),FOLD)
    for a,b,c in zip(baseline,transformed,refined):
        torch.testing.assert_close(a,b,atol=1e-7,rtol=0)
        torch.testing.assert_close(a,c,atol=1e-7,rtol=0)


def test_intensity_increases_above_four_but_cannot_pay_for_a_large_miss():
    def value(distance, speed):
        terms = strike_terms(torch.tensor([distance]), torch.tensor([[speed,0.,0.]]),
            torch.tensor([1.,0.,0.]), torch.zeros(1,45,3), OBJECTIVE)
        return float(sum(terms.values()))
    assert value(0,8)>value(0,4)>value(0,0)
    assert value(0,-8)==value(0,0)
    assert value(.1,8)>value(.2,8)
    assert value(1,100)<value(0,0)


def test_jerk_penalty_is_independent_of_command_resampling():
    values=[]
    for steps in [15,45,90]:
        values.append(strike_terms(torch.tensor([.05]),torch.tensor([[6.,0.,0.]]),
            torch.tensor([1.,0.,0.]),torch.full((1,steps,3),.3),OBJECTIVE)['smoothness'])
    for value in values[1:]:torch.testing.assert_close(value,values[0])


def test_encounter_uses_same_time_velocity_and_cannot_use_a_future_fold():
    q, material=hairpin(.25)
    env=SimpleNamespace(active=torch.ones(1,dtype=torch.bool),total=torch.zeros(1,dtype=torch.float64),
        cutoff=torch.zeros(1,dtype=torch.long),failed=torch.zeros(1,dtype=torch.bool),
        target=torch.tensor([[0.,0.,0.]],dtype=torch.float64),dt=.01,
        wave_material=material[1:-1],settings={'fold_constraint':FOLD,'trajectory_objective':OBJECTIVE},
        direction=torch.tensor([1.,0.,0.],dtype=torch.float64),
        encounter_distance=torch.tensor([10.],dtype=torch.float64),
        encounter_tip_velocity=torch.zeros(1,3,dtype=torch.float64),
        encounter_time=torch.zeros(1,dtype=torch.float64),encounter_q=q.clone())
    initialize(env)
    before=q.clone();after=q.clone()
    before[:,-1]=torch.tensor([-1.,0.,0.]);after[:,-1]=torch.tensor([1.,0.,0.])
    vb=torch.zeros_like(q);va=torch.zeros_like(q);vb[:,-1,0]=2;va[:,-1,0]=6
    observe(env,SimpleNamespace(positions_m=before,velocities_m_s=vb),
        SimpleNamespace(positions_m=after,velocities_m_s=va),env.active,torch.tensor([1.]))
    assert float(env.encounter_distance)==0
    assert float(env.encounter_time)==pytest.approx(1.005)
    assert float(env.encounter_tip_velocity[0,0])==4
    assert not bool(env.encounter_fold_valid)
    env.fold_completed.fill_(True)
    # A later, equally close pass with greater speed must not rewrite the event.
    observe(env,SimpleNamespace(positions_m=before,velocities_m_s=vb*10),
        SimpleNamespace(positions_m=after,velocities_m_s=va*10),env.active,torch.tensor([2.]))
    assert float(env.encounter_tip_velocity[0,0])==4
    assert not bool(env.encounter_fold_valid)

    assert bool(env.strike_valid)
    assert float(env.strike_velocity[0,0])==40
    best=float(env.strike_value)
    # A subsequent slow pass preserves the high-speed task event.
    observe(env,SimpleNamespace(positions_m=before,velocities_m_s=vb/2),
        SimpleNamespace(positions_m=after,velocities_m_s=va/2),env.active,torch.tensor([3.]))
    assert float(env.strike_value)==best
    assert float(env.strike_time)==pytest.approx(2.005)
    initialize(env)
    env.fold_completed.fill_(True)
    observe(env,SimpleNamespace(positions_m=before,velocities_m_s=-vb),
        SimpleNamespace(positions_m=after,velocities_m_s=-va),env.active,torch.tensor([4.]))
    assert not bool(env.strike_valid)


def test_new_profile_rejects_legacy_rewards_and_invalid_fold_tracking():
    import json
    from pathlib import Path
    from copy import deepcopy
    from planning.strike_objective import validate_settings
    cfg=json.loads((Path(__file__).resolve().parents[2]/'config/pva/systematic_strike.json').read_text())
    validate_settings(cfg)
    old=deepcopy(cfg);old['reward']={'success':1}
    with pytest.raises(ValueError,match='Legacy reward'):validate_settings(old)
    bad=deepcopy(cfg);bad['fold_constraint']['maximum_forward_step']=.8
    with pytest.raises(ValueError,match='disconnected'):validate_settings(bad)


def test_reference_recovery_rejects_an_unrecoverable_exit_without_changing_limits():
    import json
    from pathlib import Path
    from copy import deepcopy
    from planning.strike_mppi import check_recovery_reference
    cfg=json.loads((Path(__file__).resolve().parents[2]/'config/pva/systematic_strike.json').read_text())
    original=deepcopy(cfg)
    hover=torch.zeros(2,11,dtype=torch.float64);hover[:,:3]=torch.tensor(cfg['launch']['origin_m'])
    check_recovery_reference(hover,cfg)
    # Development counterexample: admissible last packet, but no admissible return.
    exit=torch.tensor([.40806381956546084,1.1295114499578909,2.2966101477171423,
        -2.810508131894701,1.8366780183361358,1.7163444260077036,
        -4.081260097930588,-.25943051936808176,6.060049971646352,0.,0.],dtype=torch.float64)
    with pytest.raises(ValueError,match='No curved recovery'):
        check_recovery_reference(torch.stack((exit,exit)),cfg)
    assert cfg==original
