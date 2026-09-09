import numpy as np
import pytest
from planning.spline import sample,decode,fit_seed
from planning.cem import optimize
from planning.cem_run import validate_settings,DEFAULTS


def test_spline_analytic_derivatives_and_hover_boundary():
    rng=np.random.default_rng(1);c=rng.normal(size=(12,3));c[:3]=[-2,.01,1.255]
    t=np.linspace(.01,1.69,105);h=1e-5
    p=sample(c,1.7,t);before=sample(c,1.7,t-h);after=sample(c,1.7,t+h)
    np.testing.assert_allclose((after[:,:3]-before[:,:3])/(2*h),p[:,3:6],atol=2e-6)
    np.testing.assert_allclose((after[:,3:6]-before[:,3:6])/(2*h),p[:,6:9],atol=2e-5)
    first=sample(c,1.7,[0])[0]
    np.testing.assert_allclose(first[:3],c[0]);np.testing.assert_allclose(first[3:],0,atol=1e-12)
    changed,duration=decode(np.r_[c[3:].ravel(),1.023],[-3,0,1.2],12,(.8,2))
    assert duration==1+1/30
    np.testing.assert_array_equal(changed[3:],c[3:]) # No rigid translation of the seed.


def test_seed_fit_recovers_a_representable_curve():
    c=np.random.default_rng(2).normal(size=(12,3));c[:3]=[0,0,1.5]
    packets=sample(c,1.2,np.arange(37)/30)
    np.testing.assert_allclose(fit_seed(packets,1.2,12),c,atol=1e-10)


def test_continuous_height_check_finds_peak_between_endpoints():
    from planning.spline import spline_feasible
    # Degree-elevated z(u)=1+8u(1-u): endpoints are 1 m, middle 3 m.
    c=np.zeros((6,3));c[:,2]=[1,2.6,3.4,3.4,2.6,1]
    limits=dict(maximum_speed_m_s=100,maximum_specific_force_m_s2=100,
                minimum_specific_vertical_m_s2=1,maximum_tilt_deg=60)
    assert spline_feasible(c,5,limits,(0,3.01))
    assert not spline_feasible(c,5,limits,(0,2.8))


def test_cem_converges_and_retains_incumbent_reproducibly():
    def objective(x):return -np.sum((x-.3)**2,axis=1),{}
    args=dict(population=64,iterations=15,seed=2)
    best,history,bank=optimize([2.,-1.],[1.,1.],objective,**args)
    np.testing.assert_allclose(best,[.3,.3],atol=.02)
    assert np.all(np.diff([h['best_score'] for h in history])>=0)
    np.testing.assert_array_equal(best,optimize([2.,-1.],[1.,1.],objective,**args)[0])


def test_invalid_candidates_never_become_elites_and_stop_does_not_evaluate():
    with pytest.raises(ValueError,match='No feasible'):
        optimize([0.],[1.],lambda x:(np.full(len(x),np.nan),{}),population=4)
    result=optimize([0.],[1.],lambda _:pytest.fail('evaluated after stop'),cancelled=lambda:True)
    assert result[1:]==([],[])


@pytest.mark.parametrize('change',[{'maximum_duration_s':6},{'maximum_height_m':-.1},{'population':0},{'position_std_m':0},
    {'launch_setup':{'initial_tracking_origin_m':[0,0], 'target_position_m':[1,0,1]}},
    {'launch_setup':{'initial_tracking_origin_m':[0,0,1], 'target_position_m':[float('nan'),0,1]}}])
def test_invalid_settings_rejected(change):
    with pytest.raises(ValueError):validate_settings(dict(DEFAULTS,**change))


def test_cem_zip_has_exact_csv_and_relative_assets(tmp_path):
    import json,zipfile
    from planning.cem_run import export_package
    source=tmp_path/'run';source.mkdir()
    model={'motion_residual':{'checkpoint':'absolute/cable'},'fullstate_execution':{'checkpoint':'absolute/drone'}}
    (source/'model.json').write_text(json.dumps(model))
    (source/'rehearsal.json').write_text(json.dumps({'schema':'cem_fullstate_30hz_v1'}))
    (source/'fullstate_30hz.csv').write_bytes(b'exact,CSV\r\n')
    with zipfile.ZipFile(export_package(source,tmp_path/'cem.zip')) as archive:
        assert archive.read('fullstate_30hz.csv')==b'exact,CSV\r\n'
        assert json.loads(archive.read('model.json'))['fullstate_execution']['checkpoint']=='assets/drone_model.json'
        assert 'No force policy inference' in archive.read('README.txt').decode()
    assert json.loads((source/'model.json').read_text())==model
