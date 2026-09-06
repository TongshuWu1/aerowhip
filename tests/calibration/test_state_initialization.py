from dataclasses import replace
from pathlib import Path
import json

import pytest
import numpy as np
import torch

from simulator.cable import CableConfiguration, DderModel, DderState
from simulator.cable.initialization import hanging_positions
from experimental_data.state_initialization import (HistoryFitSettings, endpoint_velocity,
    causal_state, history_rollout, physics_assisted_state)


def setup():
    torch.set_num_threads(1)
    payload=json.loads((Path(__file__).parents[2]/'config/model.json').read_text())
    cable=CableConfiguration.from_mapping(payload['cable'])
    model=DderModel(cable.dder_parameters(EI=2e-6,Cb=1e-4))
    q=hanging_positions(torch.zeros((1,3),dtype=torch.float64),cable)
    return cable,model,q


def test_endpoint_derivative_uses_only_available_past_and_recovers_quadratic():
    t=torch.arange(31,dtype=torch.float64)*.01
    data=(2+3*t+4*t.square())[None,:,None,None].expand(1,-1,12,3).clone()
    handover=20
    result=endpoint_velocity(data[:,:handover+1],.01)
    torch.testing.assert_close(result,torch.full_like(result,3+8*t[handover]))
    data[:,handover+1:]=1e9
    torch.testing.assert_close(endpoint_velocity(data[:,:handover+1],.01),result)
    with pytest.raises(ValueError):endpoint_velocity(data[:,:2],.01)


def test_causal_state_preserves_translating_straight_cable_and_root_velocity():
    _,model,q=setup()
    t=torch.arange(11,dtype=q.dtype)*.01
    velocity=q.new_tensor([.2,-.1,.05])
    history=q[:,None]+t[None,:,None,None]*velocity
    state=causal_state(history,.01,model)
    torch.testing.assert_close(state.positions_m,history[:,-1],atol=1e-9,rtol=1e-9)
    torch.testing.assert_close(state.velocities_m_s,velocity.expand_as(q),atol=1e-9,rtol=1e-9)


def test_history_rollout_initial_velocity_gradient_matches_finite_difference():
    _,model,q=setup()
    boundary=q[:,:1].expand(-1,5,-1).clone()
    direction=torch.zeros_like(q);direction[:,1:,0]=.1
    def objective(scale,gradients):
        predicted,_=history_rollout(model,DderState(q,direction*scale),boundary,.01,gradients=gradients)
        return predicted[:,-1,-1,0].sum()
    scale=q.new_tensor(.2,requires_grad=True)
    gradient=torch.autograd.grad(objective(scale,True),scale)[0]
    with torch.no_grad():
        finite_difference=(objective(scale+1e-5,False)-objective(scale-1e-5,False))/(2e-5)
    torch.testing.assert_close(gradient,finite_difference,rtol=1e-3,atol=1e-7)


def test_history_state_fit_keeps_finite_best_candidate_without_mutating_physics():
    cable,model,q=setup()
    boundary=q[:,:1].expand(-1,15,-1).clone()
    boundary[:,:,0]=torch.linspace(0,.01,15,dtype=q.dtype)
    with torch.no_grad():history,_=history_rollout(model,DderState(q,torch.zeros_like(q)),boundary,.01)
    parameters=model.parameters
    settings=HistoryFitSettings(derivative_samples=5,history_steps=5,updates=2)
    result,info=physics_assisted_state(history,.01,model,cable.marker_node_indices[1:],settings)
    assert torch.isfinite(result.positions_m).all() and torch.isfinite(result.velocities_m_s).all()
    assert info['selected_history_rmse_m'][0]<=info['initial_history_rmse_m'][0]+1e-12
    assert model.parameters==parameters and model.motion_residual is None
    torch.testing.assert_close(result.positions_m[:,0],history[:,-1,0])


def test_parameter_population_matches_individual_candidates():
    from experimental_data.cable_fit import PreparedTake
    from experimental_data.initialization_parameter_audit import population
    cable,model,q=setup()
    q=q.repeat(2,1,1);q[1,:,0]=.01
    roots=q[:,:1].expand(-1,4,-1).clone()
    measured=q[:,None,cable.marker_node_indices[1:]].expand(-1,4,-1,-1).clone()
    take=PreparedTake('test','training',q,torch.zeros_like(q),roots,measured,(0,1),0.,.01)
    candidates=np.array([[2e-6,1e-4],[1e-5,3e-5]])
    together=population(take,model,cable,candidates,torch.device('cpu'))
    for i in range(2):
        single=population(take,model,cable,candidates[i:i+1],torch.device('cpu'))
        for key in together:np.testing.assert_allclose(together[key][i],single[key][0],rtol=1e-9,atol=1e-12)


def test_fit_preparation_causal_mode_ignores_centered_velocities_and_future_for_initial_state(tmp_path):
    from experimental_data.cable_fit import _prepare_take
    cable,model,q=setup()
    t=np.arange(41)*.01
    positions=np.repeat(q.numpy(),len(t),axis=0)
    positions[:,:,0]+=t[:,None]*.2
    arrays=dict(time_s=t,state_valid=np.ones(len(t),dtype=bool),
        cable_node_position_world_m=positions,cable_node_velocity_world_m_s=np.zeros_like(positions),
        root_position_world_m=positions[:,0].copy(),root_velocity_world_m_s=np.zeros((len(t),3)))
    path=tmp_path/'take.npz';np.savez(path,**arrays)
    def prepare():
        return _prepare_take('test','training',path,model=model,cable=cable,device=torch.device('cpu'),
            dtype=torch.float64,horizon_s=.1,stride_s=.1,projection_passes=8,initialization='causal_polynomial')
    first=prepare()
    assert first.starts[0]==10
    torch.testing.assert_close(first.initial_velocities_m_s[0,:,0],q.new_full((12,),.2),atol=1e-9,rtol=1e-9)
    arrays['cable_node_velocity_world_m_s'][:]=999
    arrays['root_velocity_world_m_s'][:]=999
    arrays['cable_node_position_world_m'][11:,:,0]+=.3
    arrays['root_position_world_m'][11:,0]+=.3
    np.savez(path,**arrays)
    second=prepare()
    torch.testing.assert_close(first.initial_positions_m[0],second.initial_positions_m[0])
    torch.testing.assert_close(first.initial_velocities_m_s[0],second.initial_velocities_m_s[0])
