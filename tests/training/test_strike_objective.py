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
    # New task: forward strike credit is available without detector completion.
    env.settings['fold_requirement']='diagnostic_only'
    initialize(env)
    observe(env,SimpleNamespace(positions_m=before,velocities_m_s=vb),
        SimpleNamespace(positions_m=after,velocities_m_s=va),env.active,torch.tensor([5.]))
    assert bool(env.strike_valid) and not bool(env.strike_fold_valid)
    # A later detected fold cannot relabel an earlier stronger event.
    env.fold_completed.fill_(True)
    observe(env,SimpleNamespace(positions_m=before,velocities_m_s=vb/2),
        SimpleNamespace(positions_m=after,velocities_m_s=va/2),env.active,torch.tensor([6.]))
    assert not bool(env.strike_fold_valid)
    # Root velocity must be interpolated and latched at that same event too.
    from planning.strike_objective import rewarded_speed
    env.settings['trajectory_objective']=dict(OBJECTIVE,speed_metric='tip_gain_over_root')
    initialize(env);vb[:,0,0]=1.;va[:,0,0]=3.
    observe(env,SimpleNamespace(positions_m=before,velocities_m_s=vb),
        SimpleNamespace(positions_m=after,velocities_m_s=va),env.active,torch.tensor([7.]))
    assert float(env.strike_velocity[0,0])==4.
    assert float(env.strike_root_velocity[0,0])==2.
    assert float(rewarded_speed(env.strike_velocity,env.direction,env.settings['trajectory_objective'],env.strike_root_velocity)[0])==2.
    env.settings['trajectory_objective']['minimum_tip_speed_gain_m_s']=2.
    initialize(env);vb[:,0,0]=2.;va[:,0,0]=6.  # Root and tip carried together.
    observe(env,SimpleNamespace(positions_m=before,velocities_m_s=vb),
        SimpleNamespace(positions_m=after,velocities_m_s=va),env.active,torch.tensor([8.]))
    assert not bool(env.strike_valid)
    assert float(env.encounter_distance)==0.  # A perfect geometric pass is insufficient.


def test_new_profile_rejects_legacy_rewards_and_invalid_fold_tracking():
    import json
    from pathlib import Path
    from copy import deepcopy
    from planning.strike_objective import validate_settings
    cfg=json.loads((Path(__file__).resolve().parents[2]/'tests/fixtures/targeted_strike_settings.json').read_text())
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
    cfg=json.loads((Path(__file__).resolve().parents[2]/'tests/fixtures/targeted_strike_settings.json').read_text())
    cfg.pop('recovery',None)  # Exercise the preserved historical curved-recovery counterexample.
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


def test_soft_lateral_cost_leaves_forward_vertical_motion_and_all_axes_available():
    from planning.strike_objective import lateral_cost
    packets=torch.zeros(2,46,11,dtype=torch.float64)
    packets[:,:,0]=torch.linspace(0,2,46)
    packets[:,:,2]=torch.linspace(1,2,46)
    packets[1,:,1]=.25
    origin=torch.zeros(2,3,dtype=torch.float64)
    direction=torch.tensor([1.,0.,0.],dtype=torch.float64)
    settings=dict(lateral_weight=.25,lateral_scale_m=.25)
    cost=lateral_cost(packets,origin,direction,settings)
    torch.testing.assert_close(cost,torch.tensor([0.,-.25],dtype=torch.float64))
    assert torch.isfinite(cost).all()  # Drift is penalized, not rejected.
    torch.testing.assert_close(lateral_cost(packets,origin,direction,{}),torch.zeros_like(cost))
    rotated=packets.clone();rotated[:,:,:2]=packets[:,:,:2]@torch.tensor([[0.,1.],[-1.,0.]],dtype=torch.float64)
    torch.testing.assert_close(lateral_cost(rotated,origin,torch.tensor([0.,1.,0.]),settings),cost)


def test_broader_speed_guidance_preserves_accuracy_and_historical_scores():
    from planning.strike_objective import event_terms
    old = dict(OBJECTIVE)
    new = dict(old, speed_proximity_scale_m=.30)
    direction = torch.tensor([1., 0., 0.], dtype=torch.float64)
    def terms(d, v, cfg):
        return event_terms(torch.tensor([d], dtype=torch.float64),
            torch.tensor([[v, 0., 0.]], dtype=torch.float64), direction, cfg)
    legacy = terms(.25, 4., old)
    expected = -(.25/.1)**2 + 4*math.exp(-(.25/.1)**2)/2
    assert float(sum(legacy.values())) == pytest.approx(expected)
    wider = terms(.25, 4., new)
    assert float(wider['target']) == float(legacy['target'])
    assert float(wider['strike']) > 100*float(legacy['strike'])
    assert float(sum(terms(.05, 4., new).values())) > float(sum(wider.values()))
    assert float(sum(terms(.5, 100., new).values())) < float(sum(terms(0., 0., new).values()))
    assert float(terms(.05, -4., new)['strike']) == 0
    assert float(terms(.05, 8., new)['strike']) > float(terms(.05, 4., new)['strike'])


def test_nonpositive_speed_proximity_is_rejected():
    import json
    from pathlib import Path
    from planning.strike_objective import validate_settings
    cfg=json.loads((Path(__file__).resolve().parents[2]/'tests/fixtures/targeted_strike_settings.json').read_text())
    for value in (0., -.1):
        cfg['trajectory_objective']['speed_proximity_scale_m']=value
        with pytest.raises(ValueError, match='speed proximity'):validate_settings(cfg)


@pytest.mark.parametrize('angle,expected',[(0,True),(29.99,True),(30,True),(30.01,False),(90,False),(180,False)])
def test_strike_cone_boundary_and_rotation(angle,expected):
    from planning.strike_objective import strike_direction_allowed,strike_angle_deg
    theta=math.radians(angle)
    v=torch.tensor([[4*math.cos(theta),0,4*math.sin(theta)]],dtype=torch.float64)
    d=torch.tensor([1.,0.,0.],dtype=torch.float64)
    cfg={'maximum_strike_angle_deg':30.}
    assert bool(strike_direction_allowed(v,d,cfg)[0]) is expected
    assert float(strike_angle_deg(v,d)[0])==pytest.approx(angle,abs=1e-6)
    r=torch.tensor([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]],dtype=torch.float64)
    assert bool(strike_direction_allowed(v@r.T,d@r.T,cfg)[0]) is expected
    assert not bool(strike_direction_allowed(torch.zeros_like(v),d,cfg)[0])
    # Historical profiles still allow a forward, steeply angled pass.
    assert bool(strike_direction_allowed(torch.tensor([[1.,0.,10.]]),d,{})[0])


def test_strike_cone_is_applied_at_the_scored_event():
    q,material=hairpin(.25)
    cfg={'fold_constraint':FOLD,'trajectory_objective':dict(OBJECTIVE,maximum_strike_angle_deg=30.),'fold_requirement':'diagnostic_only'}
    env=SimpleNamespace(active=torch.ones(1,dtype=torch.bool),total=torch.zeros(1,dtype=torch.float64),
        cutoff=torch.zeros(1,dtype=torch.long),failed=torch.zeros(1,dtype=torch.bool),
        target=torch.zeros(1,3,dtype=torch.float64),dt=.01,wave_material=material[1:-1],settings=cfg,
        direction=torch.tensor([1.,0.,0.],dtype=torch.float64),encounter_distance=torch.tensor([10.],dtype=torch.float64),
        encounter_tip_velocity=torch.zeros(1,3,dtype=torch.float64),encounter_time=torch.zeros(1,dtype=torch.float64),encounter_q=q.clone())
    initialize(env)
    before=q.clone();after=q.clone();before[:,-1]=torch.tensor([-.01,0.,0.]);after[:,-1]=torch.tensor([.01,0.,0.])
    velocity=torch.zeros_like(q);velocity[:,-1]=torch.tensor([5.,0.,8.])
    observe(env,SimpleNamespace(positions_m=before,velocities_m_s=velocity),SimpleNamespace(positions_m=after,velocities_m_s=velocity),env.active,torch.tensor([0.]))
    assert not bool(env.strike_valid) and float(env.encounter_distance)==0
    # A later aligned event is eligible, without borrowing the earlier speed.
    velocity[:,-1]=torch.tensor([2.,0.,0.])
    observe(env,SimpleNamespace(positions_m=before,velocities_m_s=velocity),SimpleNamespace(positions_m=after,velocities_m_s=velocity),env.active,torch.tensor([1.]))
    assert bool(env.strike_valid)
    assert float(env.strike_time)==pytest.approx(1.005)
    torch.testing.assert_close(env.strike_velocity[0],torch.tensor([2.,0.,0.],dtype=torch.float64))


def test_stronger_weight_keeps_distance_penalty_and_rejects_invalid_angles():
    import json
    from pathlib import Path
    from planning.strike_objective import validate_settings,event_terms
    cfg=json.loads((Path(__file__).resolve().parents[2]/'tests/fixtures/targeted_strike_settings.json').read_text())
    new=cfg['trajectory_objective'];old=dict(new,intensity_weight=4.)
    v=torch.tensor([[5.,0.,0.]]);d=torch.tensor([1.,0.,0.]);distance=torch.tensor([.05])
    a=event_terms(distance,v,d,old,root_velocity=torch.zeros_like(v));b=event_terms(distance,v,d,new,root_velocity=torch.zeros_like(v))
    assert new['intensity_weight']==8. and new['maximum_strike_angle_deg']==30.
    torch.testing.assert_close(a['target'],b['target']);torch.testing.assert_close(2*a['strike'],b['strike'])
    for angle in (0.,-1.,91.,float('nan')):
        cfg['trajectory_objective']['maximum_strike_angle_deg']=angle
        with pytest.raises(ValueError):validate_settings(cfg)


def test_alignment_bonus_prefers_small_angles_at_same_forward_speed_and_distance():
    from planning.strike_objective import event_terms
    cfg=dict(OBJECTIVE,intensity_weight=8.,maximum_strike_angle_deg=30.,prefer_aligned_strike=True)
    direction=torch.tensor([1.,0.,0.],dtype=torch.float64)
    # Keep forward speed fixed, so extra transverse speed cannot earn credit.
    angles=torch.tensor([0.,10.,20.,29.,30.],dtype=torch.float64)
    v=torch.stack((torch.full_like(angles,4.),torch.zeros_like(angles),4*torch.tan(torch.deg2rad(angles))),-1)
    distance=torch.full_like(angles,.05)
    terms=event_terms(distance,v,direction,cfg)
    assert bool((torch.diff(terms['strike'])<0).all())
    assert float(terms['strike'][-1])==pytest.approx(0,abs=1e-12)
    assert bool((terms['target']==terms['target'][0]).all())
    historical=event_terms(distance,v,direction,dict(cfg,prefer_aligned_strike=False))
    torch.testing.assert_close(historical['strike'],historical['strike'][0].expand_as(angles))
    torch.testing.assert_close(terms['strike'][0],historical['strike'][0])
    faster=event_terms(distance,v*2,direction,cfg)
    assert bool((faster['strike'][:-1]>terms['strike'][:-1]).all())
    zero=event_terms(distance,torch.zeros_like(v),direction,cfg)
    assert bool(torch.isfinite(zero['strike']).all()) and not bool(zero['strike'].any())


def test_speed_gain_removes_carrying_and_cannot_exploit_backward_root_motion():
    from planning.strike_objective import rewarded_speed,event_terms
    cfg=dict(OBJECTIVE,speed_metric='tip_gain_over_root')
    d=torch.tensor([1.,0.,0.]);tip=torch.tensor([[4.,0.,0.],[5.,0.,0.],[1.,0.,0.],[0.,0.,0.],[-1.,0.,0.]])
    root=torch.tensor([[4.,0.,0.],[2.,0.,0.],[-5.,0.,0.],[-5.,0.,0.],[-5.,0.,0.]])
    gain=rewarded_speed(tip,d,cfg,root)
    torch.testing.assert_close(gain,torch.tensor([0.,3.,1.,0.,0.]))
    terms=event_terms(torch.zeros(5),tip,d,cfg,root_velocity=root)
    assert float(terms['strike'][0])==0 and float(terms['strike'][3])==0
    assert float(terms['strike'][1])>float(terms['strike'][2])
    # Original metric is unchanged and ignores supplied root velocity.
    torch.testing.assert_close(rewarded_speed(tip,d,OBJECTIVE,root),tip[:,0].clamp_min(0))
    with pytest.raises(ValueError,match='Root velocity'):event_terms(torch.zeros(5),tip,d,cfg)


@pytest.mark.parametrize('tip,root,expected',[(3.174,3.161,False),(4.,2.0001,False),(4.,2.,True),(5.,2.,True),(1.,-5.,False),(2.,-5.,True)])
def test_minimum_gain_rejects_carrying_and_checks_exact_boundary(tip,root,expected):
    from planning.strike_objective import strike_speed_allowed
    cfg=dict(OBJECTIVE,speed_metric='tip_gain_over_root',minimum_tip_speed_gain_m_s=2.)
    d=torch.tensor([1.,0.,0.],dtype=torch.float64)
    assert bool(strike_speed_allowed(torch.tensor([[tip,0.,0.]],dtype=torch.float64),d,cfg,torch.tensor([[root,0.,0.]],dtype=torch.float64))[0]) is expected


def test_gain_guidance_does_not_promote_or_modify_unqualified_task_scores():
    from planning.strike_mppi import gain_guided_scores
    task=torch.tensor([-torch.inf,-torch.inf,1.,-torch.inf]);original=task.clone()
    gain=torch.tensor([.5,1.5,2.,3.]);failed=torch.tensor([False,True,False,False])
    guide,missing=gain_guided_scores(task,gain,failed,2)
    torch.testing.assert_close(guide,torch.tensor([.5,-torch.inf,1.,-torch.inf]))
    torch.testing.assert_close(task,original)
    assert missing.tolist()==[True,False]
    assert not torch.isfinite(task[:2]).any()  # Still no acceptable plan in group zero.
