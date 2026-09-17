import numpy as np
import pytest

from tools.evaluate_vertical_generations import at_time, closest_approach, event_metrics


def test_interpolation_preserves_clock_and_does_not_bridge_missing_samples():
    t=np.array([0.,1.,2.]);q=np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]])
    np.testing.assert_allclose(at_time(t,q,np.ones(3,bool),.4),[.4,0.,0.])
    assert at_time(t,q,np.array([True,False,True]),.4) is None
    assert at_time(t,q,np.ones(3,bool),2.1) is None


def test_closest_approach_is_subsample_and_endpoint_censoring_is_explicit():
    t=np.array([0.,1.,2.]);q=np.array([[0.,1.,0.],[2.,1.,0.],[4.,1.,0.]])
    result=closest_approach(t,q,np.ones(3,bool),[1.,0.,0.])
    assert result['time_s']==pytest.approx(.5)
    assert result['distance_m']==pytest.approx(1.)
    assert not result['at_observed_window_boundary']
    result=closest_approach(t,q,np.ones(3,bool),[5.,0.,0.])
    assert result['at_observed_window_boundary']
    assert result['time_s']==2.


def test_missing_interval_is_not_treated_as_observed_motion():
    t=np.array([0.,1.,2.]);q=np.array([[-1.,0.,0.],[0.,0.,0.],[1.,0.,0.]])
    result=closest_approach(t,q,np.array([True,False,True]),[0.,0.,0.])
    assert result['distance_m']==1.
    assert result['incomplete_observation']
    assert closest_approach(t,q,np.zeros(3,bool),[0.,0.,0.]) is None


def test_event_alignment_does_not_erase_fixed_time_error():
    t=np.arange(5.)
    truth=np.zeros((5,2,3));truth[:,-1,0]=t-2
    prediction=truth.copy();prediction[:,-1,0]-=.5
    a=dict(time_s=t,measured_sites=truth,coupled_cable=prediction,mask=np.ones((5,1),bool))
    result=event_metrics(a,[1],2.,np.zeros(3),5.)
    assert result['planned_time_prediction_error_m']==pytest.approx(.5)
    assert result['closest_approach_time_error_s']==pytest.approx(.5)
    assert result['closest_approach_position_error_m']==pytest.approx(0.)
    assert not result['timing_censored']
