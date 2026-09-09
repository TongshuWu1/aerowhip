from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from learning.deployment_rollout import sample_batch, plan_batch, execute_batch, evaluate_deployment
from learning.point_force_env import PointForceWhipEnvironment
from simulator.cable import DderState
from simulator.cable.dder import DderModel
from run_ppo import load_configs, validate_contract


@pytest.mark.parametrize('device_name', ['cpu', 'cuda'])
def test_sampled_targets_and_physical_starts_stay_in_independent_balls(device_name):
    if device_name == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    torch.set_num_threads(1)
    model, task, config = load_configs()
    settings = {**config['deployment'], 'initial_position_radius_m': .05,
                'target_position_radius_m': .05, 'nominal_fraction': 0.}
    device = torch.device(device_name)
    env = PointForceWhipEnvironment(model, task, config, batch_size=512, device=device)
    before = torch.random.get_rng_state().clone()
    first = sample_batch(env, settings, torch.Generator(device=device).manual_seed(92))
    second = sample_batch(env, settings, torch.Generator(device=device).manual_seed(92))
    assert torch.equal(before, torch.random.get_rng_state())
    torch.testing.assert_close(first.target_position_m, second.target_position_m)
    torch.testing.assert_close(first.truth.positions_m, second.truth.positions_m)
    target_delta = first.target_position_m - env.target
    root_delta = first.truth.positions_m[:, 0] - env._batch_vector(task['initial_root_position_m'])
    for delta in (target_delta, root_delta):
        assert delta.norm(dim=-1).max() <= .05 + 1e-12
        assert (delta.std(dim=0) > .01).all()
    assert not torch.allclose(target_delta, root_delta)
    for state in (first.estimate, first.truth):
        lengths = state.positions_m.diff(dim=1).norm(dim=-1)
        torch.testing.assert_close(lengths, lengths.new_tensor(env.model.cable_configuration.rest_lengths_m).expand_as(lengths))
    nominal = sample_batch(env, {**settings, 'nominal_fraction': 1.}, torch.Generator(device=device).manual_seed(92))
    torch.testing.assert_close(nominal.target_position_m, env.target)
    torch.testing.assert_close(nominal.truth.positions_m[:, 0], env._batch_vector(task['initial_root_position_m']))


def test_actor_sees_each_target_and_independent_execution_scores_that_target(monkeypatch, tmp_path):
    torch.set_num_threads(1)
    model, task, config = load_configs()
    task['episode_duration_s'] = .05
    config['cuda_graph_physics'] = False
    env = PointForceWhipEnvironment(model, task, config, batch_size=2, device=torch.device('cpu'))
    batch = sample_batch(env, {**config['deployment'], 'nominal_fraction': 1.}, torch.Generator().manual_seed(2))
    batch.target_position_m = batch.estimate.positions_m[:, -1].clone()
    batch.target_position_m += torch.tensor([[.1, 0., 0.], [.1, .2, 0.]])
    observations = []
    def action(observation):
        observations.append(observation.clone())
        return torch.zeros(2, 3)
    env.model.step_runtime = lambda state, force, dt: env.model._result(state, state, force, dt)
    plan_batch(env, SimpleNamespace(deterministic_action=action), batch)
    torch.testing.assert_close(observations[0][:, 72:75],
        ((batch.target_position_m - batch.estimate.positions_m[:, 0]) / env.position_scale_m).float())
    calls = len(observations)
    def dynamics(_self, state, *args, **kwargs):
        q = state.positions_m.clone(); q[:, -1, 0] += .1
        v = torch.zeros_like(q); v[:, -1, 0] = 5.
        return DderState(q, v)
    monkeypatch.setattr(DderModel, 'step_runtime', dynamics)
    forces = env.hover_force_world_n.expand(1, 2, 3).clone()
    original = forces.clone()
    score = execute_batch(env, batch, forces, torch.ones(2, dtype=torch.long),
                          {**config['deployment'], 'recovery_duration_s': 0.})
    assert score.episode_success.tolist() == [True, False]
    assert len(observations) == calls
    torch.testing.assert_close(forces, original)
    from learning.attempt_records import save_attempts
    save_attempts(tmp_path, score, 2, 1.)
    with np.load(tmp_path/'attempts/0000000002.npz') as saved:
        np.testing.assert_allclose(saved['target_y_m'], batch.target_position_m[:, 1])
        np.testing.assert_allclose(saved['initial_attachment_z_m'], batch.truth.positions_m[:, 0, 2])


def test_validation_saves_actual_targets_and_first_trial_replay(monkeypatch):
    torch.set_num_threads(1)
    model, task, config = load_configs()
    task['episode_duration_s'] = .05
    config['deployment'].update(target_position_radius_m=.05, initial_position_radius_m=.05,
        nominal_fraction=0., recovery_duration_s=0.)
    config['cuda_graph_physics'] = False
    monkeypatch.setattr(DderModel, 'step_runtime', lambda _self, state, *args, **kwargs: state)
    agent = SimpleNamespace(deterministic_action=lambda obs: torch.zeros(len(obs), 3))
    first = evaluate_deployment(model, task, config, agent, episodes=3, batch_size=2, device=torch.device('cpu'))
    second = evaluate_deployment(model, task, config, agent, episodes=3, batch_size=2, device=torch.device('cpu'))
    targets = first.scenarios['target_position_m']
    assert targets.shape == (3, 3)
    np.testing.assert_array_equal(targets, second.scenarios['target_position_m'])
    np.testing.assert_allclose(first.recording[1]['target_position_m'], targets[0])
    np.testing.assert_allclose([row['target_x_m'] for row in first.trials], targets[:, 0])


@pytest.mark.parametrize('radius', [-.01, float('nan'), float('inf')])
def test_invalid_randomization_radius_is_rejected(radius):
    model, task, config = deepcopy(load_configs())
    config['deployment']['target_position_radius_m'] = radius
    with pytest.raises(ValueError, match='target_position_radius_m'):
        validate_contract(model, task, config)


def test_new_ppo_run_updates_with_varied_targets_and_saves_the_scenarios(tmp_path, monkeypatch):
    import run_ppo
    from simulator.workflow import atomic_json, read_json
    torch.set_num_threads(1)
    monkeypatch.setattr(run_ppo, 'write_active_run', lambda *args: None)
    model, task, config = load_configs()
    task['episode_duration_s'] = .1
    config['bootstrap'] = None
    config['cuda_graph_physics'] = False
    config['deployment'].update(initial_position_radius_m=.05, target_position_radius_m=.05,
        nominal_fraction=0., recovery_duration_s=.5, evaluate_final_holdout=False)
    config['ppo'].update(hidden_dim=16, minibatch_transitions=4, update_epochs=1)
    config['validation'].update(enabled=True, episodes=2, every_episodes=2)
    for name, data in [('model', model), ('task', task), ('ppo', config)]:
        atomic_json(tmp_path/'config'/f'{name}.json', data)
    output = run_ppo.train(device_name='cpu', requested_episodes=4, batch_size=2,
        artifact=tmp_path/'run', resume_checkpoint=None, config_directory=tmp_path/'config')
    assert read_json(output/'status.json')['status'] == 'COMPLETED'
    checkpoint = torch.load(output/'checkpoints/latest.pt', weights_only=False)
    assert checkpoint['episodes'] == 4
    with np.load(output/'attempts/0000000004.npz') as attempts:
        targets = np.stack([attempts[f'target_{axis}_m'] for axis in 'xyz'], axis=1)
        assert (np.linalg.norm(targets-np.array(task['target_position_m']), axis=1) <= .05).all()
        assert not np.allclose(targets, task['target_position_m'])
    record = read_json(output/'validation/latest.json')
    assert record['training_episodes'] == 4
