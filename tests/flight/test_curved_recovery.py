import numpy as np
import pytest
from deployment.curved_recovery import plan_curved_recovery, coefficients, evaluate, metrics

P=np.array([.45902956386464117,.635943908779064,2.4060561744134112])
V=np.array([-1.9039101126955464,.7852816385254994,1.6896370922779511])
A=np.array([-3.990902280082975,-2.347060127213063,-1.0920592443595112])
H=np.array([-.0067,.0129,1.555])


def test_native_whip_packets_and_clock_are_exactly_preserved():
    from deployment.research_rehearsal import complete_packets
    whip=np.zeros((31,11));whip[:,:3]=P;whip[-1,3:6]=V;whip[-1,6:9]=A
    t,packets,phase,meta=complete_packets(whip,H)
    np.testing.assert_array_equal(packets[:31],whip)
    np.testing.assert_allclose(np.diff(t),1/30,atol=1e-12)
    assert meta['schema']=='curved_moving_recovery_v1'
    assert (phase[:31]==1).all() and phase[-1]==4


def test_observed_exit_turns_without_a_stop_and_then_approaches_slowly():
    sample,m=plan_curved_recovery(P,V,A,H)
    t=np.linspace(0,m['return_end_s'],10001);p,v,a=sample(t)
    assert p[:,2].max()<3.
    assert np.linalg.norm(v[t<=m['turn_end_s']],axis=1).min()>.34
    top=np.argmax(p[:,2]);assert np.linalg.norm(v[top,:2])>.1
    assert (v[top+1:-1,2]<=1e-10).all()
    approach=t>=m['turn_end_s']
    assert np.linalg.norm(v[approach],axis=1).max()<=.35+1e-9
    assert a[:,2].min()>=-3.5-1e-9
    assert not m['stationary_braking_waypoint']


@pytest.mark.parametrize('p,v,a',[(P,V,A),(P,np.zeros(3),np.zeros(3)),
    (H,np.zeros(3),np.zeros(3)),(P,[.2,-.1,-1.],[1.,0.,.5]),(P,[0.,0.,1.],[0.,0.,0.])])
def test_join_continuity_analytic_derivatives_and_stationary_final_hold(p,v,a):
    sample,m=plan_curved_recovery(p,v,a,H)
    for actual,expected in zip(sample(np.array([0.])),(p,v,a)):
        np.testing.assert_allclose(actual[0],expected,atol=1e-12)
    for boundary in [m['turn_end_s'],m['return_end_s']]:
        for left,right in zip(sample(np.array([boundary-1e-9])),sample(np.array([boundary+1e-9]))):
            np.testing.assert_allclose(left,right,atol=1e-7)
    t=np.linspace(.01,m['total_duration_s']-.01,217);delta=1e-5
    _,vv,aa=sample(t);left=sample(t-delta);right=sample(t+delta)
    np.testing.assert_allclose((right[0]-left[0])/(2*delta),vv,atol=1e-7)
    np.testing.assert_allclose((right[1]-left[1])/(2*delta),aa,atol=1e-7)
    for actual,expected in zip(sample(np.array([m['total_duration_s']])),(H,np.zeros(3),np.zeros(3))):
        np.testing.assert_allclose(actual[0],expected,atol=1e-12)


def test_polynomial_limits_cover_interior_extrema_not_just_command_samples():
    rng=np.random.default_rng(1824)
    for _ in range(8):
        T=rng.uniform(1,7);c=coefficients(*rng.normal(size=(5,3)),T)
        p,v,a=evaluate(c,np.linspace(0,T,50001),T);m=metrics(c,T)
        assert m['peak_height_m']>=p[:,2].max()-1e-9
        assert m['peak_speed_m_s']>=np.linalg.norm(v,axis=1).max()-1e-9
        assert m['minimum_az_m_s2']<=a[:,2].min()+1e-9
        assert abs(m['peak_speed_m_s']-np.linalg.norm(v,axis=1).max())<1e-6


@pytest.mark.parametrize('settings',[dict(minimum_turn_s=0),dict(hold_s=float('nan')),
    dict(unknown=1),dict(minimum_turn_s=9),dict(maximum_tilt_deg=90),
    dict(minimum_turn_s=1,maximum_turn_s=1)])
def test_invalid_or_infeasible_plan_is_rejected(settings):
    with pytest.raises(ValueError):plan_curved_recovery(P,V,A,H,settings)


def test_invalid_sampling_and_boundary_rejected():
    with pytest.raises(ValueError):plan_curved_recovery([float('nan'),0,0],V,A,H)
    sample,m=plan_curved_recovery(P,V,A,H)
    for time in [np.array([-1]),np.array([m['total_duration_s']+1]),np.array([float('nan')]),np.array(1.)]:
        with pytest.raises(ValueError):sample(time)
