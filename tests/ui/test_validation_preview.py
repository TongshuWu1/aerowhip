import json

import numpy as np
import pytest
import torch

from simulator.cable import DderState
from simulator.cable.dder import DderModel
from simulator.validation_preview import (
    checkpoint_signature, generate, latest_checkpoint, read_recording, record_trials,
)
from run_ppo import load_configs


@pytest.fixture
def short_task():
    torch.set_num_threads(1)
    model, task, shared = load_configs()
    task['episode_duration_s'] = .2
    shared['deployment']['nominal_fraction'] = 1.
    shared['deployment']['recovery_duration_s'] = .03
    # This fixture checks the zero-follow-through timeline explicitly.
    shared['deployment']['strike_followthrough_s'] = 0.
    shared['training']['device'] = 'cpu'
    return model, task, shared


def test_records_independent_execution_through_pid_after_hit(short_task, monkeypatch):
    from learning.point_force_env import PointForceWhipEnvironment
    model, task, shared = short_task
    env = PointForceWhipEnvironment(model, task, shared, batch_size=5, device=torch.device('cpu'))
    target = env.state.positions_m[0, -1].clone()
    target[0] += .2
    task['target_position_m'] = target.tolist()

    def dynamics(_self, state, *args, **kwargs):
        q = state.positions_m.clone()
        q[:, -1, 0] += .1
        v = torch.zeros_like(q)
        v[:, -1, 0] = 10.
        return DderState(q, v)

    monkeypatch.setattr(DderModel, 'step_runtime', dynamics)
    calls = []
    class Agent:
        def deterministic_action(self, observation):
            calls.append(observation.clone())
            return torch.zeros((5, 3))

    before = torch.random.get_rng_state().clone()
    arrays, metadata = record_trials(model, task, shared, Agent(), device=torch.device('cpu'))
    assert len(calls) == 1  # No policy calls while the independent plant executes.
    assert torch.equal(before, torch.random.get_rng_state())
    assert metadata['planned'] == metadata['success'] == [True] * 5
    assert metadata['duration_s'] == pytest.approx([.02] * 5)
    assert arrays['positions_m'].shape == (6, 5, 12, 3)
    np.testing.assert_allclose(arrays['positions_m'][-1, :, -1, 0], .5)
    assert arrays['hit'][2:].all()  # Contact scoring freezes, recorded motion does not.
    assert not arrays['striking'][-1].any()


def test_refused_plans_show_actual_held_state_and_repeat_fixed_trials(short_task):
    model, task, shared = short_task
    shared['deployment']['nominal_fraction'] = 0.
    shared['deployment']['require_predicted_success'] = True
    task['target_position_m'] = [30., 0., 1.4]
    class Agent:
        def deterministic_action(self, observation):
            return torch.zeros((5, 3))
    first, metadata = record_trials(model, task, shared, Agent(), device=torch.device('cpu'))
    second, _ = record_trials(model, task, shared, Agent(), device=torch.device('cpu'))
    assert metadata['planned'] == metadata['success'] == [False] * 5
    assert len(first['positions_m']) == 1
    np.testing.assert_array_equal(first['positions_m'], second['positions_m'])
    assert not np.allclose(first['positions_m'][0, 0], first['positions_m'][0, 1])


@pytest.mark.parametrize('algorithm', ['ppo', 'sac'])
def test_checkpoint_preview_uses_saved_configs_latest_weights_and_identity(tmp_path, short_task, algorithm):
    from simulator.rollout import _build_policy_agent
    from run_sac import build_agent, checkpoint, configs
    model, task, shared = short_task
    task['episode_duration_s'] = .1
    run = tmp_path / algorithm
    (run / 'checkpoints').mkdir(parents=True)
    for name, config in zip(('model', 'task', 'ppo'), (model, task, shared)):
        (run / f'{name}.json').write_text(json.dumps(config), encoding='utf-8')
    if algorithm == 'ppo':
        agent = _build_policy_agent(shared, torch.device('cpu'))
        payload = {**agent.checkpoint(), 'episodes': 42}
    else:
        config = configs()[-1]
        config['sac']['hidden_dim'] = 32
        payload = checkpoint(build_agent(config, torch.device('cpu'), task), 42)
    path = run / 'checkpoints/latest.pt'
    torch.save(payload, path)
    torch.save({**payload, 'episodes': 12}, run / 'checkpoints/best_validation.pt')
    assert latest_checkpoint(run) == path
    destination = run / 'validation_preview/latest.npz'
    metadata = generate(path, destination, device_name='cpu')
    arrays, loaded = read_recording(destination)
    assert loaded == metadata
    assert loaded['episodes'] == 42 and loaded['algorithm'] == algorithm.upper()
    assert loaded['checkpoint_signature'] == checkpoint_signature(path)
    assert loaded['recovery_duration_s'] == .03
    assert arrays['positions_m'].shape[1:] == (5, 12, 3)
    assert not list(destination.parent.glob('*.tmp'))
