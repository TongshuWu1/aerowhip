import numpy as np
import pytest
import torch
from experimental_data.current_adaptation import (phase_weights, nodes_from_sites, Trial, JOB, read)
from experimental_data.current_adaptation_fit import folds, load_engine, pose_prediction, WeightedTranslationBatch
from simulator.cable import CableConfiguration
from experimental_data.current_adaptation_fit import cable_windows,join_windows,cable_forward,record_weights
from experimental_data.adaptation_ensemble import CableEnsemble,repeat_data
from copy import deepcopy
from experimental_data.adaptation_attitude import ResidualAttitudeTrial
from scipy.spatial.transform import Rotation
from experimental_data.current_adaptation_validation import summarize_prediction

HAS_ADP0=(JOB/'source_candidate/model.json').exists()


def test_whole_flight_splits_and_phase_balance():
    names=['a','b','c','d','e']
    for _,training,heldout in folds(names)[:-1]:
        assert set(training).isdisjoint(heldout)
        assert set(training+heldout)==set(names)
        assert len(heldout)==1
    t=np.arange(0,3.001,.01);w=phase_weights(t,np.ones(len(t),bool))
    assert w[t<1].sum()==pytest.approx(.8)
    assert w[t>=1].sum()==pytest.approx(.2)


@pytest.mark.skipif(not HAS_ADP0,reason='Local adp0 integration data unavailable')
def test_report_scores_native_observations_not_resampled_count():
    model=read(JOB/'source_candidate/model.json');trial=Trial(JOB,'whip_adp_0_001',model,device='cpu')
    times=trial.grid(1.);p,r,sites=trial.measured(times)
    nodes=nodes_from_sites(sites,CableConfiguration.from_mapping(model['cable']))
    metrics,_=summarize_prediction(trial,times,dict(position_origin_m=torch.tensor(p[None]),rotation_tracking_to_world=torch.tensor(r[None])),torch.tensor(nodes[None]))
    assert metrics['origin']['whip']['samples']==100
    assert metrics['all_cable_markers']['whip']['samples']==1000
    assert metrics['origin']['whip']['rmse_m']<.001


def test_geometry_subdivisions_preserve_marker_locations():
    c=CableConfiguration((.1,)*10,.008,(.001,)*10,.0035,(2,)+(1,)*9,(0.,0.,-9.80665),8,4)
    sites=np.random.default_rng(2).normal(size=(4,11,3));nodes=nodes_from_sites(sites,c)
    np.testing.assert_array_equal(nodes[:,c.marker_node_indices],sites)


def test_cable_heldout_receives_zero_selection_weight():
    records=[dict(name=n,cutoff=t) for n in ['train','test'] for t in [0.,.2,1.2,2.]]
    w=record_weights(records,['train'])
    assert w[4:].sum()==0
    assert w[:2].sum()==pytest.approx(.8)
    assert w[2:4].sum()==pytest.approx(.2)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_ADP0,reason='CUDA/local adp0 fitting runtime required')
def test_measured_boundary_gradient_inference_parity():
    torch.set_num_threads(1);model=read(JOB/'source_candidate/model.json');engine=load_engine(JOB,trainable=True)
    trial=Trial(JOB,'whip_adp_0_002',model)
    records,rejected=cable_windows([trial],engine.physics,[0.],.02)
    assert not rejected
    data=join_windows(records);params=torch.tensor([1e-7,1e-4],device='cuda',dtype=torch.float64,requires_grad=True)
    q,_=cable_forward(engine,data,params,gradients=True)
    fast,_=cable_forward(engine,data,params.detach())
    torch.testing.assert_close(q,fast,atol=1e-12,rtol=0)
    loss=(q[:,-1,-1]-data['truth'][:,-1,-1]).square().sum();loss.backward()
    assert torch.isfinite(params.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in engine.physics.motion_residual.correction_head.parameters())


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_ADP0,reason='CUDA/local adp0 fitting runtime required')
def test_independent_fold_batch_matches_single_and_excludes_other_gradient():
    torch.set_num_threads(1);model=read(JOB/'source_candidate/model.json');engine=load_engine(JOB,trainable=True)
    t=Trial(JOB,'whip_adp_0_002',model);r,_=cable_windows([t],engine.physics,[0.],.02);d=join_windows(r)
    params=torch.tensor([1e-7,1e-4],device='cuda',dtype=torch.float64)
    q,_=cable_forward(engine,d,params,gradients=True)
    one=engine.physics.motion_residual;two=deepcopy(one)
    objective=q[:,-1,-1].square().sum();objective.backward()
    gradient=one.correction_head.bias.grad.clone();one.zero_grad()
    engine.physics.motion_residual=CableEnsemble([one,two])
    batched,_=cable_forward(engine,repeat_data(d,2),params,gradients=True)
    torch.testing.assert_close(q,batched[:1],atol=1e-12,rtol=0)
    batched[:1,-1,-1].square().sum().backward()
    torch.testing.assert_close(gradient,one.correction_head.bias.grad,atol=1e-10,rtol=1e-9)
    assert two.correction_head.bias.grad.abs().max()==0


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_ADP0,reason='CUDA/local adp0 fitting runtime required')
def test_current_initialization_is_causal_and_training_matches_engine():
    torch.set_num_threads(1)
    model=read(JOB/'source_candidate/model.json');engine=load_engine(JOB)
    t=Trial(JOB,'whip_adp_0_002',model,end=.2)
    state=t.initial_pose(engine.drone.parameters)
    assert state.time_s<0
    t.data['position'][t.data['time']>=0]=1e6
    t.data['sites'][t.data['time']>=0]=1e6
    again=t.initial_pose(engine.drone.parameters)
    torch.testing.assert_close(state.position,again.position,rtol=0,atol=0)
    cable,when,_=t.cable_state(engine.physics)
    assert when==state.time_s
    torch.testing.assert_close(cable.positions_m[:,0],state.position+state.rotation@torch.tensor(t.offset,dtype=torch.float64,device='cuda'),atol=1e-12,rtol=0)
    p=pose_prediction(t,engine.drone,t.time)['position_origin_m'][0]
    train=WeightedTranslationBatch([t],engine.drone.parameters).predict(engine.drone.residual)[0][0]
    torch.testing.assert_close(p,train,atol=2e-5,rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_ADP0,reason='CPU/GPU local-data training parity')
def test_drone_cpu_training_matches_cuda_gradients():
    torch.set_num_threads(1);model=read(JOB/'source_candidate/model.json');engine=load_engine(JOB,trainable=True)
    t=Trial(JOB,'whip_adp_0_002',model,end=.2);gpu=engine.drone.residual;cpu=deepcopy(gpu).cpu()
    gl=WeightedTranslationBatch([t],engine.drone.parameters,device='cuda').loss(gpu)
    cl=WeightedTranslationBatch([t],engine.drone.parameters,device='cpu').loss(cpu)
    gl.backward();cl.backward()
    torch.testing.assert_close(gl.cpu(),cl,atol=1e-12,rtol=1e-10)
    for a,b in zip(gpu.parameters(),cpu.parameters()):torch.testing.assert_close(a.grad.cpu(),b.grad,atol=1e-12,rtol=1e-9)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_ADP0,reason='Attitude CUDA/local-data runtime parity')
def test_attitude_cache_includes_residual_translation():
    torch.set_num_threads(1);model=read(JOB/'source_candidate/model.json');engine=load_engine(JOB)
    t=Trial(JOB,'whip_adp_0_002',model,end=.3);p=engine.drone.parameters
    cache=ResidualAttitudeTrial(t,p,deepcopy(engine.drone.residual).cpu())
    r=cache.predict([p.attitude_acceleration_scale_xy,p.attitude_acceleration_scale_z,p.attitude_time_constant_s])
    expected=pose_prediction(t,engine.drone,t.time)['rotation_tracking_to_world'][0].cpu().numpy()
    err=Rotation.from_matrix(r.transpose(0,2,1)@expected).as_rotvec()
    assert np.max(np.linalg.norm(err,axis=1))<1e-10
