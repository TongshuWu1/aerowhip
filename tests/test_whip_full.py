"""Data weighting, scientific scope and optimizer retention for full adaptation."""
import numpy as np
import pytest
import torch
from experimental_data.whip_full_data import group_weights,default_contract
from experimental_data.whip_full_cable import residual_vector
from experimental_data.whip_full_optim import trust_fit


def test_equal_take_weight_not_window_count():
    rows=[dict(category='whip',take=n) for n in ('001','002','004')]
    rows += [dict(category='preliminary',take=n) for n in ['a']*20+['b']*2]
    w=group_weights(rows)
    assert np.isclose(w.sum(),1)
    assert np.allclose(w[:3],2/9)
    assert np.isclose(w[3:23].sum(),1/6)
    assert np.isclose(w[23:].sum(),1/6)


def test_fixed_missingness_and_padding_have_zero_loss_gradient():
    q=torch.zeros(1,1,4,3,3,dtype=torch.float64,requires_grad=True)
    mask=torch.tensor([[[True,True],[True,False],[False,False],[False,False]]])
    truth=torch.ones(1,4,2,3,dtype=torch.float64)
    data=dict(truth=truth,mask=mask,weights=torch.ones(1,dtype=torch.float64))
    residual_vector(q,data,[1,2],.02).square().sum().backward()
    assert q.grad[0,0,0,1:].abs().sum()>0
    assert not q.grad[0,0,1,2].any()
    assert not q.grad[0,0,2:].any()
    assert not q.grad[:,:,:,0].any()


def test_full_scope_has_separate_residuals_and_no_promotion():
    c=default_contract()
    assert c['cable_tip_weight']==0.
    assert len(c['drone_gain_bounds'][0])==6
    assert len(c['cable_bounds'][0])==3
    assert 'drone_residual' in c['stages'] and 'cable_residual' in c['stages']
    assert c['promotion'] is False
    assert c['residual_stopping']['ceiling'] is None


def test_all_marker_loss_counts_tip_once_and_masks_missing_samples():
    q=torch.zeros(1,1,2,4,3,dtype=torch.float64,requires_grad=True)
    mask=torch.tensor([[[True,True,False],[True,False,False]]])
    truth=torch.zeros(1,2,3,3,dtype=torch.float64)
    truth[0,0,0,0]=1;truth[0,0,1,0]=2
    truth[~mask]=100  # Missing values must not affect either loss or gradient.
    data=dict(truth=truth,mask=mask,weights=torch.ones(1,dtype=torch.float64))
    loss=residual_vector(q,data,[1,2,3],1.,tip_weight=0.).square().sum()
    expected=2*(np.sqrt(2)-1+np.sqrt(5)-1)/3
    assert float(loss.detach())==pytest.approx(expected)
    loss.backward()
    assert torch.isfinite(q.grad).all()
    assert not q.grad[0,0,:,3].any()  # No valid tip sample is required by this loss.
    assert not q.grad[0,0,1,2].any()


@pytest.mark.parametrize('weight',[0.,.25,.5,None])
def test_both_cable_fit_stages_use_contract_weight(monkeypatch,weight):
    from types import SimpleNamespace
    from experimental_data import whip_full_cable as cable
    q=torch.zeros(1,1,3,3,dtype=torch.float64)
    q[0,0,2,0]=1  # Only the tip has unit error; all-marker loss assigns half.
    data=dict(q=q[:,0],truth=torch.zeros(1,1,2,3,dtype=torch.float64),
        mask=torch.ones(1,1,2,dtype=torch.bool),weights=torch.ones(1))
    c=default_contract();c.update(cable_scale_m=1.,nominal_prior=0.,residual_magnitude=0.,residual_change=0.)
    if weight is None:c.pop('cable_tip_weight')
    else:c['cable_tip_weight']=weight
    effective=.5 if weight is None else weight
    expected=2*(np.sqrt(2)-1)*((1-effective)/2+effective)
    forward=cable.FullCableForward.__new__(cable.FullCableForward)
    forward.contract=c;forward.cached=None;forward.data=data;forward.ids=[1,2];forward.log_prior=np.zeros(3)
    forward.predict=lambda values:q[None].expand(len(values),-1,-1,-1,-1)
    r,_=forward.evaluate(np.zeros(3))
    assert np.square(r).sum()==pytest.approx(expected)
    class ZeroNet(torch.nn.Module):
        acceleration_limit=.5
        def forward(self,q,v):return torch.zeros_like(q)
    engine=SimpleNamespace(physics=SimpleNamespace(motion_residual=ZeroNet()),
        cable=SimpleNamespace(marker_node_indices=[0,1,2]))
    monkeypatch.setattr(cable,'CudaCableFit',lambda *a,**k:lambda *a,**k:(q,torch.zeros_like(q)))
    objective,_=cable.residual_objective(engine,data,[1,1,1],c)
    assert float(objective())==pytest.approx(expected)


def test_m2_replay_weights_and_new_validation_aggregation():
    from experimental_data.whip_full_fit import validation_changes
    rows=[dict(category=f,take=n) for f in ('whip','prior_whip','preliminary') for n in ('a','b','c')]
    w=group_weights(rows)
    np.testing.assert_allclose([w[:3].sum(),w[3:6].sum(),w[6:].sum()],[.5,.25,.25])
    protocol={'takes':{'003':{'role':'validation'},'005':{'role':'validation'},'004':{'role':'adaptation'}}}
    report={'models':{name:{'takes':{t:{'metrics':{m:{'rmse_m':v} for m in ('drone','command_driven_tip')}}
        for t,v in zip(('003','005','004'),values)}} for name,values in [('M1-full',[1,1,100]),('M2',[.8,1.4,0])]}}
    changes=validation_changes(report,protocol,'M1-full','M2')
    assert set(changes['takes'])=={'003','005'}
    np.testing.assert_allclose(changes['equal_take_mean']['drone'],[1,1.1])


def test_cable_residual_warm_start_is_an_independent_exact_copy():
    from types import SimpleNamespace
    from simulator.cable.residual import MotionResidual
    from experimental_data.whip_full_cable import make_residual
    c=default_contract();parent=MotionResidual(4,**c['cable_residual']).double()
    with torch.no_grad():list(parent.parameters())[-1].fill_(.25)
    before={k:v.clone() for k,v in parent.state_dict().items()}
    engine=SimpleNamespace(physics=SimpleNamespace(motion_residual=parent),cable=SimpleNamespace(node_count=4),
        drone=SimpleNamespace(residual=torch.nn.Linear(1,1).double()))
    child=make_residual(engine,c)
    assert child is not parent
    for k,v in child.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    with torch.no_grad():list(child.parameters())[-1].zero_()
    for k,v in parent.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)


def test_cable_change_regularizer_uses_parent_output(monkeypatch):
    from types import SimpleNamespace
    from experimental_data import whip_full_cable as cable
    class Net(torch.nn.Module):
        acceleration_limit=.5
        def __init__(self):
            super().__init__();self.a=torch.nn.Parameter(torch.tensor(.2,dtype=torch.float64))
        def forward(self,q,v):return torch.ones_like(q)*self.a
    q=torch.zeros(1,2,3,3,dtype=torch.float64);net=Net()
    d=dict(q=q[:,0],truth=q[:,:,1:],mask=torch.ones(1,2,2,dtype=torch.bool),weights=torch.ones(1))
    monkeypatch.setattr(cable,'CudaCableFit',lambda *a,**k:lambda *a,**k:(q,torch.zeros_like(q)))
    engine=SimpleNamespace(physics=SimpleNamespace(motion_residual=net),cable=SimpleNamespace(marker_node_indices=[0,1,2]))
    c=default_contract();c.update(residual_magnitude=0,residual_change=1)
    objective,_=cable.residual_objective(engine,d,[1,1,1],c)
    assert float(objective().detach())==0
    with torch.no_grad():net.a.add_(.1)
    torch.testing.assert_close(objective(),torch.tensor(.04,dtype=torch.float64))


def test_trust_fit_retains_best_and_explains_stop(tmp_path):
    def evaluate(x):return x-np.array([.2,-.4]),np.eye(2)
    x,r=trust_fit(evaluate,np.zeros(2),(np.full(2,-2.),np.full(2,2.)),tmp_path/'stage',tmp_path,'test',
        dict(minimum=2,patience=2,relative=.001,ceiling=30))
    assert np.allclose(x,[.2,-.4],atol=1e-5)
    assert r['best_loss']<r['baseline_loss']
    assert r['stop_reason']=='solver_termination'
    assert (tmp_path/'stage/search_state.json').exists()


def test_model_contract_preserves_geometry_and_drone_architecture(tmp_path):
    from copy import deepcopy
    from experimental_data.io import atomic_json
    from experimental_data.whip_full_fit import immutable_identity
    root=__import__('pathlib').Path(__file__).resolve().parents[1]
    from simulator.workflow import read_json
    source=root/'runs/rehearsals_pva/20260910-022818-648386-M0-development-whip/model.json'
    if not source.exists():__import__('pytest').skip('Local immutable flight fixture is not included in source-only exports')
    m=read_json(source);original=immutable_identity(source)
    m['cable']['external_drag_s_inv']=.7
    atomic_json(tmp_path/'model.json',m)
    assert immutable_identity(tmp_path/'model.json')==original
    m['recorded_data']['optitrack_to_attachment_offset_body_m'][0]+=.01
    atomic_json(tmp_path/'model.json',m)
    assert immutable_identity(tmp_path/'model.json')!=original


def test_training_windows_exclude_whole_validation_takes():
    from pathlib import Path
    from simulator.workflow import read_json
    root=Path(__file__).resolve().parents[1]
    path=root/'runs/adaptation/20260909-preliminary1-M0-v2/windows.json'
    if not path.exists():__import__('pytest').skip('Local preliminary recording fixture is not included in source-only exports')
    windows=read_json(path)
    training=[w for w in windows if w['role']=='training']
    assert len(training)==69 and all(w['take']!='figure8_002' for w in training)


def test_neural_plateau_can_continue_past_400_without_ceiling(tmp_path,monkeypatch):
    from experimental_data import whip_full_continuation as continuation
    monkeypatch.setattr(continuation,'progress',lambda *args,**kwargs:None)
    net=torch.nn.Linear(1,1,bias=False,dtype=torch.float64)
    with torch.no_grad():net.weight.zero_()
    result=continuation.train_to_plateau(net,lambda:1+net.weight.square().sum(),
        dict(residual_stopping=dict(minimum=405,patience=2,relative=.001,check_every=5,ceiling=None)),
        tmp_path/'nn',tmp_path,'test')
    assert result['updates']==405 and result['stop_reason']=='practical_plateau'
    assert result['ceiling'] is None and result['selected_update']==0
