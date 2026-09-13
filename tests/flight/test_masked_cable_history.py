"""Missing history is omitted from regression, never filled or extrapolated."""
from types import SimpleNamespace
import numpy as np
import pytest
from experimental_data.current_adaptation import (observed_cable_history,
    observed_endpoint_velocity, endpoint_velocity)


def data():
    t=np.linspace(-1.,0.,101)
    sites=np.zeros((101,3,3))
    sites[:,:,0]=t[:,None]**2+2*t[:,None]+np.arange(3)[None]
    sites[:,:,1]=3*t[:,None]
    return t,dict(sites=sites,pose_valid=np.ones(101,bool),marker_valid=np.ones((101,2),bool))


def test_missing_node_samples_preserve_known_derivative_and_other_observations():
    t,d=data();d['marker_valid'][25:29,1]=False
    d['sites'][25:29,2]=1e6 # invalid finite marker cannot leak into the regression
    nodes,valid=observed_cable_history(d,np.arange(101),SimpleNamespace(interval_subdivisions=[1,1]),.8)
    v=observed_endpoint_velocity(t,nodes,valid,.02)
    np.testing.assert_allclose(v,np.tile([2.,3.,0.],(3,1)),atol=1e-10)
    assert valid[:,0].all() and valid[:,1].all()
    assert np.isnan(nodes[25:29,2]).all()
    assert np.isfinite(d['sites']).all() # source data is untouched


def test_complete_history_is_numerically_identical_to_existing_initializer():
    t,d=data();nodes,valid=observed_cable_history(d,np.arange(101),SimpleNamespace(interval_subdivisions=[1,1]))
    np.testing.assert_array_equal(observed_endpoint_velocity(t,nodes,valid,.02),endpoint_velocity(t,nodes,.02))


@pytest.mark.parametrize('kind,message',[('endpoint','endpoint'),('coverage','Insufficient'),('strict','Missing')])
def test_no_endpoint_extrapolation_or_silent_rule_relaxation(kind,message):
    t,d=data()
    if kind=='endpoint':d['marker_valid'][-1,1]=False
    elif kind=='coverage':d['marker_valid'][:25,1]=False
    else:d['marker_valid'][25:29,1]=False
    with pytest.raises(ValueError,match=message):
        observed_cable_history(d,np.arange(101),SimpleNamespace(interval_subdivisions=[1,1]),None if kind=='strict' else .8)
