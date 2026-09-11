import numpy as np
import pytest
from planning.mppi import optimize, importance_weights


def test_new_origin_reanchors_entire_initial_guess_without_mutating_seed():
    from planning.spline import initialize_at_origin,sample
    c=np.random.default_rng(19).normal(0,.03,(12,3))+[-2,0,1.255]
    c[:3]=[-2,0,1.255]
    packets=sample(c,1,np.arange(31)/30);before=packets.copy()
    origin=np.array([0,0,1.225])
    initialized=initialize_at_origin(packets,1,12,origin)
    guess=sample(initialized,1,np.arange(31)/30)
    np.testing.assert_allclose(guess[:,:3],packets[:,:3]+origin-c[0],atol=1e-12)
    np.testing.assert_allclose(guess[:,3:],packets[:,3:],atol=1e-10)
    np.testing.assert_array_equal(packets,before)


def test_importance_update_matches_analytic_gaussian_posterior():
    # Prior N(0,1), likelihood exp(-(x-1)^2/2) -> posterior N(.5,.5).
    # Deliberately shifted proposal; softmax of reward alone would be biased.
    samples=np.random.default_rng(7).normal(1.4,1,(200000,1))
    weights=importance_weights(samples,-.5*(samples[:,0]-1)**2,np.array([1.4]),np.array([0.]),np.ones(1),1.)
    assert abs(weights@samples[:,0]-.5)<.006
    assert abs(weights@((samples[:,0]-.5)**2)-.5)<.006


def test_optimizer_reproducible_retains_best_and_excludes_deterministic_rows():
    seen=[]
    def score(x):
        seen.append(x.copy())
        return -100*np.sum((x-.3)**2,axis=1),{}
    args=dict(population=256,iterations=8,temperature=1.,seed=8)
    result=optimize([1.,-1.],[.7,.7],score,**args)
    np.testing.assert_allclose(result[0],[.3,.3],atol=.04)
    assert np.all(np.diff([r['best_score'] for r in result[1]])>=0)
    assert all(len(x)==258 for x in seen)
    first=seen[0]
    weights=importance_weights(first[:256],score(first[:256])[0],np.array([1.,-1.]),np.array([1.,-1.]),np.array([.7,.7]),1.)
    np.testing.assert_allclose(seen[1][-2],weights@first[:256],atol=1e-12)
    np.testing.assert_array_equal(result[0],optimize([1.,-1.],[.7,.7],score,**args)[0])


def test_invalid_candidates_stop_and_stable_weights():
    weights=importance_weights(np.zeros((4,1)),[10000,np.nan,-np.inf,9999],np.zeros(1),np.zeros(1),np.ones(1),.001)
    np.testing.assert_array_equal(weights,[1,0,0,0])
    with pytest.raises(ValueError,match='No feasible'):
        optimize([0.],[1.],lambda x:(np.full(len(x),-np.inf),{}),population=4)
    _,history,bank=optimize([0.],[1.],lambda _:pytest.fail('evaluated'),cancelled=lambda:True)
    assert not history and not bank


def test_only_incumbent_feasible_does_not_fake_sampling_update():
    def score(x):
        values=np.full(len(x),-np.inf);values[-1]=1
        return values,{}
    best,history,_=optimize([2.],[.5],score,population=4,iterations=2)
    np.testing.assert_array_equal(best,[2.])
    assert all(h['effective_sample_size']==0 for h in history)


def test_mppi_job_freezes_m1_not_seed_model(tmp_path):
    from pathlib import Path
    from planning.cem_run import prepare_job, MPPI_DEFAULTS
    from simulator.workflow import read_json
    from experimental_data.io import sha256_file
    root=Path(__file__).resolve().parents[2]
    seed=root/'runs/rehearsals/20260908-203914-039721'
    source=root/MPPI_DEFAULTS['model_path']
    if not source.is_file() or not seed.is_dir():pytest.skip('Local fitted M1 and saved flight required')
    before={p:sha256_file(p) for p in (source,seed/'model.json',seed/'rehearsal.npz',root/'config/cem.json',root/'config/research_30hz/model.json')}
    output=tmp_path/'mppi'
    prepare_job(root,seed,output,dict(MPPI_DEFAULTS,population=4,iterations=1,
        launch_setup=dict(initial_tracking_origin_m=[-2,0,1.255],target_position_m=[-1,0,1.1])))
    actual=read_json(output/'model.json');expected=read_json(source)
    assert actual['cable']==expected['cable']
    assert actual['motion_residual']['sha256']==expected['motion_residual']['sha256']
    assert actual['fullstate_execution']['sha256']==expected['fullstate_execution']['sha256']
    assert read_json(output/'model_provenance.json')['source_sha256']==before[source]
    assert read_json(output/'mppi.json')['optimizer']=='mppi'
    assert all(sha256_file(p)==h for p,h in before.items())
    with pytest.raises(ValueError,match='empty'):
        prepare_job(root,seed,output,MPPI_DEFAULTS)
