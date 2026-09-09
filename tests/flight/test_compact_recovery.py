import numpy as np
import pytest
from deployment.compact_recovery import plan_compact_recovery
from deployment.gentle_recovery import plan_recovery


# Boundary from the user's 20260908-181353-400309 rehearsal, in tracked-origin coordinates.
P=np.array([.45902956386464117,.635943908779064,2.4060561744134112])
V=np.array([-1.9039101126955464,.7852816385254994,1.6896370922779511])
A=np.array([-3.990902280082975,-2.347060127213063,-1.0920592443595112])
H=np.array([-.0067,.0129,1.555])


def test_observed_upward_exit_has_lower_peak_and_earlier_descent():
    old,om=plan_recovery(P,V,A,H);new,nm=plan_compact_recovery(P,V,A,H)
    op=old(np.linspace(0,om['total_duration_s'],10001))[0]
    t=np.linspace(0,nm['total_duration_s'],10001);p,v,a=new(t)
    assert p[:,2].max()<op[:,2].max()-.4
    assert nm['return_start_times_s'][2]<nm['return_start_times_s'][0]
    early=(t>nm['return_start_times_s'][2]+.05)&(t<nm['return_start_times_s'][0])
    assert (v[early,2]<0).all()
    assert a[:,2].min()>=-3.5-1e-10


@pytest.mark.parametrize('velocity,acceleration',[(V,A),([0,0,0],[0,0,0]),([.2,-.1,-1.],[1.,0.,.5])])
def test_continuous_pva_derivatives_and_combined_return_limits(velocity,acceleration):
    sample,meta=plan_compact_recovery(P,velocity,acceleration,H)
    for actual,expected in zip(sample(np.array([0.])),(P,velocity,acceleration)):
        np.testing.assert_allclose(actual[0],expected,atol=1e-12)
    boundaries=np.unique([meta['transition_s'],meta['vertical_transition_s'],*meta['axis_plateau_end_times_s'],
        *meta['axis_stop_times_s'],*meta['return_start_times_s'],
        *(np.array(meta['return_start_times_s'])+meta['axis_return_durations_s']),meta['return_end_s']])
    for b in boundaries:
        # Another axis can have nonzero jerk at this axis's boundary; approach
        # the boundary closely enough to distinguish that slope from a jump.
        for left,right in zip(sample(np.array([b-1e-9])),sample(np.array([b+1e-9]))):
            np.testing.assert_allclose(left,right,atol=1e-7)
    t=np.linspace(.01,meta['total_duration_s']-.01,211);h=1e-5
    p,v,a=sample(t);left=sample(t-h);right=sample(t+h)
    np.testing.assert_allclose((right[0]-left[0])/(2*h),v,atol=2e-7)
    np.testing.assert_allclose((right[1]-left[1])/(2*h),a,atol=2e-7)
    _,v,a=sample(np.linspace(meta['brake_end_s'],meta['return_end_s'],10001))
    assert np.linalg.norm(v,axis=1).max()<=.4+1e-9
    assert np.linalg.norm(a,axis=1).max()<=.3+1e-9
    for actual,expected in zip(sample(np.array([meta['total_duration_s']])),(H,np.zeros(3),np.zeros(3))):
        np.testing.assert_allclose(actual[0],expected,atol=1e-12)


@pytest.mark.parametrize('settings',[dict(vertical_transition_s=0),dict(vertical_return_s=float('nan')),dict(unknown=1)])
def test_invalid_settings_rejected(settings):
    with pytest.raises(ValueError):plan_compact_recovery(P,V,A,H,settings)
