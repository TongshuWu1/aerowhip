import numpy as np
import pytest
from experimental_data.flight_performance import local_velocity,encounter


def test_quadratic_velocity_exact_on_irregular_samples_and_gaps_not_bridged():
    t=np.arange(21)*.01;t[1::2]+=.0001
    q=np.c_[2*t+3*t*t,-t,np.zeros_like(t)]
    v=local_velocity(t,q)
    np.testing.assert_allclose(v[2:-2,0],2+6*t[2:-2],atol=1e-12)
    q[10]=np.nan;v=local_velocity(t,q)
    assert np.isnan(v[8:13]).all() and np.isfinite(v[7]).all()
    gap=t.copy();gap[10:]+=.1;v=local_velocity(gap,np.c_[gap,gap,gap])
    assert np.isnan(v[8:12]).all()


def test_swept_entry_and_directed_speed_have_correct_fraction():
    t=np.arange(11)*.01;q=np.c_[4*t,np.zeros((11,2))];v=np.tile([4.,0.,0.],(11,1))
    result=encounter(t,q,[.21,0,0],.01,v)
    assert result['first_entry']['time_s']==pytest.approx(.05)
    assert result['first_entry']['outward_speed_m_s']==pytest.approx(4.)
    assert result['nearest']['distance_m']==pytest.approx(0.)


def test_missing_target_crossing_is_unknown_not_an_invented_hit():
    t=np.arange(7)*.01;q=np.c_[t,np.zeros((7,2))];q[3]=np.nan
    result=encounter(t,q,[.03,0,0],.002)
    assert result['first_entry'] is None
    assert result['contiguous_segment_coverage']<1


def test_stationary_sample_inside_target_does_not_divide_by_zero():
    result=encounter(np.arange(3)*.01,np.zeros((3,3)),[0,0,0],.05)
    assert result['first_entry']['time_s']==0
    assert result['nearest']['distance_m']==0
