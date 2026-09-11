from copy import deepcopy
from pathlib import Path

import pytest
import torch

from experimental_data.differentiable_fit import save_weights
from experimental_data.io import sha256_file
from simulator.cable import DderState, FREE_ENDPOINTS
from simulator.cable.residual import MotionResidual, FrozenMotionResidual, cable_drag_description
from simulator.point_mass import ForceControlledPointCable
from simulator.workflow import read_json

ROOT = Path(__file__).resolve().parents[2]


def models():
    torch.set_num_threads(1)
    original = read_json(ROOT/'config/model.json')
    # Exercise a nonzero physical damping coefficient independently of the live
    # unfitted template (zero drag makes the softplus gradient test degenerate).
    original['cable']['external_drag_s_inv']=.3
    baseline = ForceControlledPointCable.from_mapping(original)
    payload = deepcopy(original)
    payload['cable']['external_drag_s_inv'] = 0
    candidate = ForceControlledPointCable.from_mapping(payload)
    residual = MotionResidual(candidate.cable_configuration.node_count, hidden=8,
        learn_drag=True, initial_drag_s_inv=original['cable']['external_drag_s_inv']).double()
    candidate.dder.motion_residual = residual
    state = baseline.hanging_state(torch.tensor([0., 0., 1.5], dtype=torch.float64))
    state = DderState(state.positions_m, torch.full_like(state.velocities_m_s, .2))
    return payload, baseline, candidate, residual, state


@pytest.mark.parametrize('method', ['step', 'step_runtime'])
def test_moving_drag_into_residual_preserves_transition_and_root_exclusion(method):
    _, baseline, candidate, residual, state = models()
    rates = residual.drag_rates(state.positions_m)
    assert rates[0] == 0 and (rates[1:] > 0).all()
    force = baseline.hover_force_world_n(dtype=torch.float64, device='cpu')[None]
    force[:, 0] += .15
    left, right = state, state
    for _ in range(4):
        left = getattr(baseline, method)(left, force, .01).state
        right = getattr(candidate, method)(right, force, .01).state
    torch.testing.assert_close(left.positions_m, right.positions_m, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(left.velocities_m_s, right.velocities_m_s, atol=1e-12, rtol=1e-12)


def test_drag_gradient_through_transition_matches_finite_difference():
    _, _, candidate, residual, state = models()
    def loss():
        result = candidate.dder.step_runtime(state, state.positions_m[:, :0],
            state.positions_m.new_tensor([.01]), candidate.dder.runtime_constants(state.positions_m),
            pinned_endpoints=FREE_ENDPOINTS, dense_constraint_solve=True,
            iterative_damping=False, create_graph=True, analytic_bending=True)
        return result.velocities_m_s.square().sum()
    analytic = torch.autograd.grad(loss(), residual.raw_drag)[0]
    initial = residual.raw_drag.detach().clone()
    epsilon = 1e-4
    with torch.no_grad():
        residual.raw_drag.copy_(initial+epsilon)
    plus = loss().detach()
    with torch.no_grad():
        residual.raw_drag.copy_(initial-epsilon)
    minus = loss().detach()
    with torch.no_grad():
        residual.raw_drag.copy_(initial)
    finite = (plus-minus)/(2*epsilon)
    assert abs(analytic) > 1e-8
    torch.testing.assert_close(analytic, finite, rtol=1e-4, atol=1e-9)


def test_checkpoint_roundtrip_and_reject_double_counting(tmp_path):
    payload, _, candidate, residual, state = models()
    with torch.no_grad():
        residual.raw_drag.add_(.5)  # Saved learned value differs from initialization.
    path = tmp_path/'residual.pt'
    save_weights(path, residual)
    frozen = FrozenMotionResidual(path, sha256_file(path))
    torch.testing.assert_close(frozen.drag_rates(state.positions_m), residual.drag_rates(state.positions_m))
    payload['motion_residual'] = dict(enabled=True, checkpoint=str(path), sha256=sha256_file(path),
                                      specification=residual.specification())
    loaded = ForceControlledPointCable.from_mapping(payload)
    force = candidate.hover_force_world_n(dtype=torch.float64, device='cpu')[None]
    expected = candidate.step_runtime(state, force, .01).state
    actual = loaded.step_runtime(state, force, .01).state
    torch.testing.assert_close(actual.velocities_m_s, expected.velocities_m_s)
    assert 'learned in residual' in cable_drag_description(payload)
    payload['cable']['external_drag_s_inv'] = .3
    with pytest.raises(ValueError, match='double-counting'):
        ForceControlledPointCable.from_mapping(payload)
    assert 'fixed cable damping' in cable_drag_description(read_json(ROOT/'config/model.json'))


def test_nn_only_model_rejects_separate_drag(tmp_path):
    payload, _, candidate, _, _ = models()
    network = MotionResidual(candidate.cable_configuration.node_count).double()
    path = tmp_path/'nn_only.pt'
    save_weights(path, network)
    payload['motion_residual'] = dict(enabled=True, checkpoint=str(path), sha256=sha256_file(path),
                                      drag_mode='nn_only', specification=network.specification())
    ForceControlledPointCable.from_mapping(payload)
    assert 'no separate drag coefficient' in cable_drag_description(payload)
    payload['cable']['external_drag_s_inv'] = .3
    with pytest.raises(ValueError, match='double-counting'):
        ForceControlledPointCable.from_mapping(payload)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires CUDA')
@pytest.mark.parametrize('rehearsal', [False, True])
def test_learned_drag_survives_gpu_graph_replay(tmp_path, rehearsal):
    from simulator.cuda_graph_physics import CudaGraphPhysics
    from simulator.cable.cuda_rehearsal_solvers import RehearsalSolvers
    payload, _, _, residual, state = models()
    with torch.no_grad():
        residual.raw_drag.add_(.4)
        residual.net[-1].bias.fill_(.03)
    path = tmp_path/'residual.pt'
    save_weights(path, residual)
    payload['motion_residual'] = dict(enabled=True, checkpoint=str(path), sha256=sha256_file(path))
    model = ForceControlledPointCable.from_mapping(payload)
    state = DderState(state.positions_m.cuda(), state.velocities_m_s.cuda())
    solvers = RehearsalSolvers() if rehearsal else None
    graph = CudaGraphPhysics(model, state, .01, linear_solvers=solvers)
    force = model.hover_force_world_n(dtype=torch.float64, device='cuda')[None]
    force[:, 0] += .1
    eager, captured = state, state
    for _ in range(5):
        eager = model.dder.step_runtime(eager, eager.positions_m[:, :0], graph.dt,
            model.dder.runtime_constants(eager.positions_m),
            external_force_world_n=model.controller.node_forces(force, validate=False),
            iterative_damping=True, damping_backend='pcg32_experimental',
            linear_solvers=solvers, analytic_bending=rehearsal,
            pinned_endpoints=FREE_ENDPOINTS, create_graph=False)
        captured = graph(captured, force)
    torch.testing.assert_close(captured.positions_m, eager.positions_m, atol=1e-9, rtol=1e-8)
    torch.testing.assert_close(captured.velocities_m_s, eager.velocities_m_s, atol=1e-8, rtol=1e-7)
