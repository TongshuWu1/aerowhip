"""Regression checks for the fresh collection and numerical review findings."""
import json
import numpy as np
import pytest
import torch
from experimental_data.whip_full_cable import gradient_check
from experimental_data.whip_full_fit import validation_changes
from planning.local_reference_correction import strike_metrics,strike_allowed,update_trust,local_step


def test_stationary_gradient_is_valid_and_parameters_rng_restored(tmp_path):
    net=torch.nn.Linear(2,2,dtype=torch.float64)
    with torch.no_grad():
        for p in net.parameters():p.zero_()
    rng=torch.random.get_rng_state().clone()
    before={n:v.clone() for n,v in net.state_dict().items()}
    result=gradient_check(net,lambda:sum(p.square().sum() for p in net.parameters()),tmp_path/'check.json')
    assert result['passed'] and len(result['directions'])==4
    assert all(len(d['checks'])>=2 for d in result['directions'])
    assert torch.equal(rng,torch.random.get_rng_state())
    for n,v in net.state_dict().items():assert torch.equal(v,before[n])


def test_wrong_gradient_in_nonfinal_tensor_is_detected(tmp_path):
    net=torch.nn.Linear(2,1,dtype=torch.float64)
    with torch.no_grad():net.weight.fill_(1);net.bias.fill_(1)
    before=net.weight.detach().clone()
    handle=net.weight.register_hook(lambda g:g*0)
    with pytest.raises(ValueError,match='gradient mismatch'):
        gradient_check(net,lambda:sum(p.square().sum() for p in net.parameters()),tmp_path/'bad.json')
    handle.remove()
    assert torch.equal(before,net.weight)
    result=json.loads((tmp_path/'bad.json').read_text())
    assert not result['passed'] and result['directions'][0]['direction']=='weight'


def test_gradient_probe_exception_restores_all_parameters(tmp_path):
    net=torch.nn.Linear(2,1,dtype=torch.float64)
    before={n:v.clone() for n,v in net.state_dict().items()}
    def objective():
        if not torch.is_grad_enabled():raise RuntimeError('broken numerical rollout')
        return sum(p.square().sum() for p in net.parameters())
    with pytest.raises(RuntimeError,match='broken numerical'):
        gradient_check(net,objective,tmp_path/'check.json')
    for n,v in net.state_dict().items():assert torch.equal(v,before[n])


def test_ten_training_takes_do_not_masquerade_as_validation():
    protocol=dict(takes={f'M0_{i:03d}':dict(role='adaptation') for i in range(1,11)})
    result=validation_changes({},protocol,'M0','M1')
    assert not result['available'] and result['equal_take_mean']=={} and result['takes']=={}


def test_no_holdout_registration_keeps_candidate_unpromoted(tmp_path,monkeypatch):
    from experimental_data import whip_full_fit as fit
    from experimental_data.io import atomic_json
    job=tmp_path/'job';job.mkdir();comparison=tmp_path/'comparison'
    protocol=dict(parent_generation=0,full_update=dict(parent_id='M0',candidate_id='M1'),
        takes={'M0_001':dict(role='adaptation')})
    atomic_json(job/'protocol.json',protocol)
    atomic_json(job/'fit/result.json',dict(status='completed',candidate_hashes={},training_takes=['M0_001']))
    atomic_json(job/'source_hashes.json',{'M0_001.csv':'new-raw-hash'})
    atomic_json(comparison/'report.json',{})
    catalog=dict(models=[dict(id='M0',signature='parent',generation_index=0,training_sources=[])])
    monkeypatch.setattr(fit,'load_catalog',lambda root:catalog)
    monkeypatch.setattr(fit,'save_catalog',lambda root,value:None)
    monkeypatch.setattr(fit,'model_identity',lambda path:('parent' if path.parent.name=='source_candidate' else 'child',{}))
    result=fit.register(job,comparison)
    assert result['promoted'] is False and not result['validation']['available']
    assert 'next-batch evaluation pending' in result['status']
    assert catalog['models'][-1]['training_sources']==['new-raw-hash']


@pytest.mark.parametrize('overlap',[False,True])
def test_before_update_evaluation_precedes_training_and_blocks_reused_raw_data(tmp_path,monkeypatch,overlap):
    from experimental_data import whip_full_fit as fit
    from experimental_data.io import atomic_json
    job=tmp_path/'job';job.mkdir();events=[]
    protocol=dict(full_update=dict(parent_id='M0',evaluate_before_training=True,seed=1),takes={'take':dict(role='adaptation')})
    atomic_json(job/'protocol.json',protocol);atomic_json(job/'source_hashes.json',{'take.csv':'raw'})
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(fit.data,'load',lambda *a:({},protocol,None))
    monkeypatch.setattr(fit,'load_catalog',lambda root:dict(models=[dict(id='M0',signature='parent',training_sources=['raw'] if overlap else [])]))
    monkeypatch.setattr(fit,'model_identity',lambda path:('parent',{}))
    monkeypatch.setattr(fit,'evaluate_pair',lambda *a,**kw:events.append(('evaluate',kw['before_update'])))
    monkeypatch.setattr(fit,'progress',lambda *a,**kw:None)
    def training(*args):events.append(('training',True));raise RuntimeError('test stops before fitting')
    monkeypatch.setattr(fit.data,'drone_trials',training)
    with pytest.raises(ValueError if overlap else RuntimeError,match='overlaps' if overlap else 'test stops'):
        fit.fit(job)
    assert events==([] if overlap else [('evaluate',True),('training',True)])


def test_strike_guard_detects_worse_strike_despite_better_average():
    times=np.arange(100,dtype=float);ref=np.zeros((100,3))
    baseline=np.zeros_like(ref);baseline[:,0]=.02
    candidate=np.zeros_like(ref);candidate[50,0]=.1
    assert np.mean(candidate**2)<np.mean(baseline**2)
    b=strike_metrics(baseline,times,ref,50,np.zeros(3));c=strike_metrics(candidate,times,ref,50,np.zeros(3))
    assert not strike_allowed(c,b,'target') and not strike_allowed(c,b,'reference')
    assert strike_allowed(c,b,'none')
    with pytest.raises(ValueError):strike_metrics(candidate,times,ref,100)
    with pytest.raises(ValueError):strike_allowed({}, {}, 'target')


def test_trust_update_uses_model_agreement_and_step_size():
    r,d=update_trust(.5,1e-4,.1,.5,1,1e-4)
    assert r<.5 and d>1e-4
    r,d=update_trust(.5,1e-4,.9,.01,1,1e-4)
    assert r==.5 and d<1e-4  # Tiny accepted step does not expand radius.
    assert update_trust(.5,1e-4,.9,.5,1,1e-4)[0]>.5


def test_constrained_optimality_at_active_bound_differs_from_raw_gradient():
    # x >= 0, gradient points out of the feasible domain: stationary at x=0.
    step=local_step(np.array([1.]),np.eye(1),1,np.eye(1),np.array([-1.]),np.array([1.]),damping=0.)
    assert np.linalg.norm(step,np.inf)<1e-7


def test_setup_creates_ten_roles_and_binds_exact_csv(tmp_path):
    from experimental_data.whip_adaptation import setup,protocol
    rehearsal=tmp_path/'rehearsal';rehearsal.mkdir()
    (rehearsal/'model.json').write_text(json.dumps(dict(provenance=dict(generation_index=0))))
    (rehearsal/'rehearsal.npz').write_bytes(b'frozen forecast')
    (rehearsal/'fullstate_30hz.csv').write_text('frozen command')
    batch=setup(tmp_path,tmp_path/'batch',rehearsal=rehearsal,command_csv=rehearsal/'fullstate_30hz.csv',
        all_training=True,take_count=10,take_prefix='M0')
    roles=protocol(batch)['planned_roles']
    assert len(roles)==10 and set(roles.values())=={'adaptation'} and 'M0_010' in roles
    (batch/'simulation_csv/fullstate_30hz.csv').write_text('changed')
    with pytest.raises(ValueError,match='Flown command'):protocol(batch)


def test_tiny_backtracking_steps_do_not_stop_after_one_iteration(tmp_path,monkeypatch):
    from planning.position_spline import PositionSpline
    from planning.local_reference_correction import optimize
    from planning import reference_correction
    class Rollout:
        def __init__(self,*args,**kwargs):pass
        def __call__(self,packets,times,grid):
            pos=packets[:,:,:3]
            return dict(cable_positions_m=torch.stack((pos,pos),dim=2),position_origin_m=pos,
                complete_valid=torch.ones(len(packets),dtype=torch.bool))
    monkeypatch.setattr(reference_correction,'CoupledRollout',Rollout)
    spline=PositionSpline(1.5);origin=np.array([0,0,1.4]);cutoff=35
    baseline=torch.tensor(np.tile(origin,(9,1)),dtype=torch.float64)
    packets,_=spline.decode(baseline,origin);packets=packets[:cutoff]
    ref=dict(tip=packets[:,:3]+torch.tensor([.03,0,0]),quadrotor=packets[:,:3],command=packets[:,:3])
    settings=dict(initial_trust_radius=.5,maximum_trust_radius=1.,finite_difference_step=.01,
        damping=1e-4,maximum_iterations=4,maximum_derivative_refinements=1,maximum_jacobian_relative_difference=.05,
        backtracking_scales=[1/32],relative_improvement_stop=.99,planned_strike_time_s=.8,target_position_m=[.03,0,1.4])
    limits=dict(minimum_origin_z_m=.96,maximum_origin_z_m=2.8,maximum_speed_m_s=5.,
        maximum_specific_force_m_s2=18.,minimum_specific_vertical_m_s2=2.,maximum_tilt_deg=60.)
    grid=np.arange(cutoff)/30
    _,cost,_,info=optimize(spline,baseline,origin,cutoff,None,None,grid,grid,ref,
        dict(tip=1.,quadrotor=.1,command=.01),limits,[60,60,60],lambda x:None,tmp_path,lambda *a,**kw:None,settings)
    assert cost<.03**2 and info['iterations']==4 and info['stop_reason']=='Iteration budget reached'
    history=json.loads((tmp_path/'history.json').read_text())
    assert all(r['accepted'] for r in history)
    assert all(r['attempts'][0]['strike'] is not None for r in history)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_initial_state_sensitivity_with_frozen_model_and_synthetic_history(tmp_path,monkeypatch):
    from pathlib import Path
    from simulator.cable import DderState
    from tools import check_initial_state_sensitivity as diagnostic
    root=Path(__file__).resolve().parents[1]
    reference=root/'runs/reference_tracking/M0-paper-fixed-reference'
    model=root/'runs/rehearsals_pva/20260913-012740-484590-M0-slower-brake-1s/model.json'
    if not model.exists() or not reference.exists():pytest.skip('Local frozen M0 fixture is unavailable')
    with np.load(reference/'reference.npz') as z:nominal=z['cable_position_m'][0].copy()
    class SyntheticHistory:
        def __init__(self,*a,**k):pass
        def cable_state(self,*a,**k):
            q=torch.tensor(nominal[None],device='cuda',dtype=torch.float64);v=torch.zeros_like(q)
            v[0,-1,0]=.03
            return DderState(q,v),-.001,0.
    monkeypatch.setattr(diagnostic,'WhipTrial',SyntheticHistory)
    # Geometry lookup is independently exercised elsewhere; this test isolates
    # scenario generation and the actual CUDA rollout, without inventing raw data.
    monkeypatch.setattr(diagnostic,'immutable_identity',lambda p:'same-fixture')
    prepared=tmp_path/'prepared';prepared.mkdir()
    for name in ('prepared_hashes','source_hashes'):(prepared/(name+'.json')).write_text('{}')
    (prepared/'protocol.json').write_text(json.dumps(dict(takes={'synthetic':dict(role='adaptation')})))
    report=diagnostic.run(model,reference,prepared,tmp_path/'sensitivity')
    assert len(report['scenarios'])==4 and all(r['valid'] for r in report['scenarios'])
    nominal_report=report['scenarios'][0]
    assert nominal_report['tip_reference_rmse_m']<1e-5
    assert report['scenarios'][-1]['initial_tip_speed_m_s']>0
    with np.load(tmp_path/'sensitivity/scenarios.npz') as z:
        np.testing.assert_allclose(z['initial_velocities_m_s'][:,0],0,atol=1e-10)
        assert np.isfinite(z['tip_positions_m']).all()
