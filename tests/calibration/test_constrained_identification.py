import numpy as np
import torch
from dataclasses import replace

from experimental_data.constrained_geometry import fit_offset
from experimental_data.cable_fit import PreparedTake
from experimental_data.differentiable_fit import rollout
from simulator.cable import CableConfiguration
from tests.calibration.test_differentiable_fit import fixture
from tests.physics.test_point_mass import _model
from simulator import ForceControlledPointCable
from experimental_data.constrained_identification import evaluate_together
from experimental_data.differentiable_fit import evaluate,subset


def test_geometry_is_training_only_and_keeps_ruler_values():
    offset = np.array([.006, -.012, -.055])
    tangent = np.broadcast_to([0., 0., -1.], (100, 3))
    c1 = offset + .063*tangent
    train = dict(role='training', c1=c1, quiet_c1=c1, quiet_tangent=tangent)
    data = {'fit':train, 'val':dict(train,role='validation',quiet_c1=c1+100)}
    result = fit_offset(data)
    np.testing.assert_allclose(result['offset_body_m'], offset, atol=.001)
    assert result['offset_body_m'][2] == -.055 and result['first_span_m'] == .063
    data['val']['quiet_c1'] *= -100
    assert fit_offset(data) == result


def test_drag_rollout_gradient_matches_finite_difference_and_config():
    cable, model, q = fixture()
    cable = replace(cable, external_drag_s_inv=.3)
    assert cable.dder_parameters(EI=1e-6, Cb=1e-4).external_drag_s_inv == .3
    roots=q[:,:1].expand(-1,5,-1).clone()
    measured=q[:,None,cable.marker_node_indices[1:]].expand(-1,5,-1,-1).clone()
    take=PreparedTake('s','training',q, q.new_full(q.shape,.1),roots,measured,(0,1),0.,.01)
    p=q.new_tensor([2e-6,1e-4,.3],requires_grad=True)
    weight=torch.linspace(.1,1,measured.numel(),dtype=q.dtype).reshape_as(measured)
    loss=(rollout(take,model,cable,p,gradients=True)*weight).sum()
    gradient=torch.autograd.grad(loss,p)[0]
    assert torch.isfinite(gradient).all()
    with torch.no_grad():
        hi=p.detach().clone();lo=hi.clone();hi[2]+=1e-5;lo[2]-=1e-5
        fd=((rollout(take,model,cable,hi)-rollout(take,model,cable,lo))*weight).sum()/2e-5
    torch.testing.assert_close(gradient[2],fd,rtol=.002,atol=1e-7)


def test_cable_drag_does_not_add_unidentified_drone_drag():
    payload,_=_model()
    payload['cable']['external_drag_s_inv']=.3
    payload['cable']['gravity_m_s2']=[0.,0.,0.]
    model=ForceControlledPointCable.from_mapping(payload)
    state=model.hanging_state(torch.tensor([[0.,0.,1.]],dtype=torch.float64),
                              torch.tensor([[.1,0.,0.]],dtype=torch.float64))
    masses=torch.tensor(model.dder.parameters.vertex_masses_kg,dtype=torch.float64)
    dt=.0001;command=torch.zeros((1,3),dtype=torch.float64)
    direct=model.step(state,command,dt).state
    runtime=model.step_runtime(state,command,dt).state
    torch.testing.assert_close(runtime.positions_m,direct.positions_m,atol=1e-10,rtol=1e-8)
    change=((state.velocities_m_s-direct.velocities_m_s)*masses[None,:,None]).sum(1)[0,0]
    expected=masses[1:].sum()*.1*(1-np.exp(-.3*dt))
    torch.testing.assert_close(change,expected,rtol=.001,atol=1e-11)


def test_batched_candidate_metrics_match_separate_take_evaluation():
    cable,model,q=fixture()
    roots=q[:,:1].expand(-1,5,-1).clone()
    measured=q[:,None,cable.marker_node_indices[1:]].expand(-1,5,-1,-1).clone()
    take=PreparedTake('fit','training',q,q*0,roots,measured,(0,1),0.,.01)
    a=subset(take,[0]);b=replace(subset(take,[1]),take_id='val',role='validation')
    p=q.new_tensor([2e-6,1e-4,.3])
    combined,_,_=evaluate_together([a,b],model,cable,p)
    for t in (a,b):
        expected=evaluate([t],model,cable,p)
        for key in ['objective','equal_take_marker_rmse_m','equal_take_tip_rmse_m']:
            np.testing.assert_allclose(combined[t.role][key],expected[key],rtol=1e-8,atol=1e-10)
