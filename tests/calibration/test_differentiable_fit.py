from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from simulator.cable import CableConfiguration, DderModel, DderState, START_PINNED_FREE_END
from simulator.cable.dder import analytic_isotropic_bending_force
from simulator.cable.residual import MotionResidual, FrozenMotionResidual
from experimental_data.cable_fit import PreparedTake
from experimental_data.differentiable_fit import rollout, save_weights
from experimental_data.io import sha256_file


def fixture():
    torch.set_num_threads(1)
    payload = json.loads((Path(__file__).parents[2] / 'config/model.json').read_text())
    cable = CableConfiguration.from_mapping(payload['cable'])
    model = DderModel(cable.dder_parameters(EI=2e-6, Cb=1e-4))
    q = torch.zeros(2, cable.node_count, 3, dtype=torch.float64)
    arc = torch.tensor([0., *np.cumsum(cable.rest_lengths_m)])
    q[:, :, 2] = -arc
    q[:, :, 0] = .08 * torch.sin(arc*3)
    q[:, :, 1] = .04 * torch.sin(arc*5)
    q = model.project_lengths(q, q[:, :1], pinned_endpoints=START_PINNED_FREE_END)
    return cable, model, q


def test_analytic_force_and_derivative_match_energy_gradient():
    cable, model, q = fixture()
    q.requires_grad_()
    ei = q.new_full((len(q),), 2e-6, requires_grad=True)
    constants = replace(model.runtime_constants(q), bending_stiffness_n_m2=ei)
    reference = model._runtime_internal_force(q, constants, create_graph=True)
    actual = analytic_isotropic_bending_force(q, ei, constants.dual_lengths_m)
    torch.testing.assert_close(actual, reference, rtol=1e-8, atol=1e-11)
    direction = torch.randn_like(q)
    expected = torch.autograd.grad((reference * direction).sum(), (q, ei), retain_graph=True)
    observed = torch.autograd.grad((actual * direction).sum(), (q, ei))
    for left, right in zip(observed, expected):
        torch.testing.assert_close(left, right, rtol=1e-7, atol=1e-9)


def test_dense_transition_matches_runtime_and_gradient_matches_finite_difference():
    cable, model, q = fixture()
    p = q.new_tensor([2e-6, 1e-4], requires_grad=True)
    def run(parameters, dense, gradients=False):
        state = DderState(q, torch.zeros_like(q))
        constants = replace(model.runtime_constants(q),
            bending_stiffness_n_m2=parameters[0].expand(len(q)),
            bending_damping_n_m2_s=parameters[1].expand(len(q)))
        for i in range(3):
            boundary = q[:, :1] + q.new_tensor([.0002*(i+1), 0, 0])
            state = model.step_runtime(state, boundary, q.new_full((len(q),), .01), constants,
                pinned_endpoints=START_PINNED_FREE_END, dense_constraint_solve=dense,
                analytic_bending=dense, create_graph=gradients)
        return state.positions_m
    actual = run(p, True, True)
    with torch.no_grad():
        torch.testing.assert_close(actual, run(p, False), rtol=1e-8, atol=1e-10)
    weight = torch.linspace(.1, 1, q.numel(), dtype=q.dtype).reshape_as(q)
    gradient = torch.autograd.grad((actual * weight).sum(), p)[0]
    for i in range(2):
        step = float(p[i].detach()) * 1e-5
        minus, plus = p.detach().clone(), p.detach().clone()
        minus[i] -= step
        plus[i] += step
        with torch.no_grad():
            fd = ((run(plus, True) - run(minus, True)) * weight).sum() / (2*step)
        torch.testing.assert_close(gradient[i], fd, rtol=.002, atol=1e-5)


def test_residual_is_bounded_translation_invariant_and_roundtrips(tmp_path):
    cable, model, q = fixture()
    network = MotionResidual(cable.node_count).double()
    v = torch.randn_like(q)
    assert torch.count_nonzero(network(q, v)) == 0
    with torch.no_grad():
        network.net[-1].weight.normal_()
    acceleration = network(q, v)
    assert acceleration.abs().max() <= 2
    assert torch.count_nonzero(acceleration[:, 0]) == 0
    torch.testing.assert_close(acceleration, network(q + 20, v))
    path = tmp_path / 'residual.pt'
    save_weights(path, network)
    frozen = FrozenMotionResidual(path, sha256_file(path))
    torch.testing.assert_close(frozen(q, v), acceleration)
    with pytest.raises(ValueError, match='hash mismatch'):
        FrozenMotionResidual(path, 'incorrect')


def test_recursive_rollout_does_not_read_future_measured_targets():
    cable, model, q = fixture()
    roots = q[:, :1].expand(-1, 4, -1).clone()
    roots[:, :, 0] += torch.arange(4) * .0002
    measured = q[:, None, cable.marker_node_indices[1:]].expand(-1, 4, -1, -1).clone()
    take = PreparedTake('synthetic', 'training', q, q*0, roots, measured, (0, 1), 0., .01)
    parameters = q.new_tensor([2e-6, 1e-4], requires_grad=True)
    network = MotionResidual(cable.node_count).double()
    model.motion_residual = network
    first = rollout(take, model, cable, parameters, gradients=True, block_steps=2)
    second = rollout(replace(take, measured_marker_positions_m=measured+100), model, cable,
                     parameters, gradients=True, block_steps=2)
    torch.testing.assert_close(first, second)
    first[:, -1, -1, 0].sum().backward()
    assert bool(torch.isfinite(parameters.grad).all())
    assert network.net[-1].weight.grad.norm() > 0


def test_versioned_neural_baseline_loads_and_changes_runtime(tmp_path):
    from simulator.workflow import apply_baseline, read_json
    from experimental_data.io import atomic_json
    from simulator.point_mass import ForceControlledPointCable
    cable, _, _ = fixture()
    model = json.loads((Path(__file__).parents[2] / 'config/model.json').read_text())
    job = tmp_path / 'job'
    job.mkdir()
    network = MotionResidual(cable.node_count).double()
    with torch.no_grad():
        network.net[-1].bias.fill_(.1)
    checkpoint_path = job / 'residual_candidate.pt'
    save_weights(checkpoint_path, network)
    atomic_json(job / 'model.json', model)
    fit = dict(fitted_parameters={'EI_n_m2': 2e-6, 'Cb_n_m2_s': 1e-4}, residual=dict(
        checkpoint=checkpoint_path.name, sha256=sha256_file(checkpoint_path),
        specification=network.specification(), validation_improved_at_all_horizons=True))
    atomic_json(job / 'fit_result.json', fit)
    apply_baseline(tmp_path, model, fit_directory=job, include_residual=True)
    applied = read_json(tmp_path / 'config/model.json')
    hybrid = ForceControlledPointCable.from_mapping(applied, root=tmp_path)
    plain = ForceControlledPointCable.from_mapping({k:v for k,v in applied.items() if k != 'motion_residual'})
    state = hybrid.hanging_state(torch.tensor([0., 0., 1.5], dtype=torch.float64))
    command = hybrid.hover_force_world_n(dtype=torch.float64, device='cpu')[None]
    hybrid_q = hybrid.step_runtime(state, command, .01).state.positions_m
    plain_q = plain.step_runtime(state, command, .01).state.positions_m
    assert (hybrid_q - plain_q).abs().max() > 1e-7
    apply_baseline(tmp_path, applied)
    assert 'motion_residual' not in read_json(tmp_path / 'config/model.json')
    fit['residual']['validation_improved_at_all_horizons'] = False
    atomic_json(job / 'fit_result.json', fit)
    with pytest.raises(ValueError, match='not improved validation'):
        apply_baseline(tmp_path, model, fit_directory=job, include_residual=True)


@pytest.mark.parametrize('learn_drag', [False, True])
def test_live_compilation_retains_neural_correction(tmp_path, learn_drag):
    from simulator.point_mass import ForceControlledPointCable
    from simulator.live_flight import prepare_live_physics
    cable, _, _ = fixture()
    payload = json.loads((Path(__file__).parents[2] / 'config/model.json').read_text())
    network = MotionResidual(cable.node_count, learn_drag=learn_drag,
                            initial_drag_s_inv=payload['cable']['external_drag_s_inv']).double()
    if learn_drag:
        payload['cable']['external_drag_s_inv'] = 0
    with torch.no_grad():
        network.net[-1].bias.fill_(.1)
    path = tmp_path / 'residual.pt'
    save_weights(path, network)
    payload['motion_residual'] = dict(enabled=True, checkpoint=str(path), sha256=sha256_file(path))
    model = ForceControlledPointCable.from_mapping(payload)
    state = model.hanging_state(torch.tensor([0., 0., 1.5], dtype=torch.float64))
    compiled = prepare_live_physics(json.dumps(payload, sort_keys=True))
    force = model.hover_force_world_n(dtype=torch.float64, device='cpu')[None]
    force[:, 0] += .2
    for _ in range(5):
        expected = model.step_runtime(state, force, .01).state
        q, v = compiled(state.positions_m, state.velocities_m_s, force)
        torch.testing.assert_close(q, expected.positions_m, rtol=1e-8, atol=1e-9)
        torch.testing.assert_close(v, expected.velocities_m_s, rtol=1e-7, atol=1e-8)
        state = expected
