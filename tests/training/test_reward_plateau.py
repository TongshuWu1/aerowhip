from copy import deepcopy
import json

import pytest
import torch

from learning.reward_plateau import RewardPlateau


def monitor():
    return RewardPlateau(dict(enabled=True, window_evaluations=4, patience_episodes=80,
                              minimum_additional_episodes=100, minimum_reward_improvement=2.,
                              minimum_relative_improvement=.01), start_episodes=1000)


def feed(monitor, rewards):
    for index, reward in enumerate(rewards):
        result = monitor.observe(dict(training_episodes=1000+index*10,
                                      evaluation_id=str(index), mean_episode_reward=reward))
    return result


def test_sustained_plateau_stops_after_patience_and_warmup():
    m = monitor()
    assert not feed(m, [200.] * 11)['should_stop']
    result = m.observe(dict(training_episodes=1110, evaluation_id='11', mean_episode_reward=200.))
    assert result['should_stop'] and result['status'] == 'PLATEAU'


@pytest.mark.parametrize('rewards', [list(range(180, 240)), [200.]*10+[100.]*30])
def test_improvement_or_collapse_is_not_called_convergence(rewards):
    assert not feed(monitor(), rewards)['should_stop']


def test_single_peak_does_not_stop_a_growing_rolling_reward():
    assert not feed(monitor(), [200., 280., 200., 200.]+list(range(201, 241)))['should_stop']


def test_reused_nonfinite_and_out_of_order_metrics_do_not_exhaust_patience():
    m = monitor()
    feed(m, [200.]*4)
    before = dict(m.decision)
    for change in [dict(evaluation_id='3'), dict(evaluation_reused=True),
                   dict(mean_episode_reward=float('nan')), dict(training_episodes=1020)]:
        record = dict(training_episodes=10000, evaluation_id='new', mean_episode_reward=200.)
        record.update(change)
        assert m.observe(record) == before


def test_training_stops_cleanly_and_retains_reward_checkpoint(tmp_path, monkeypatch):
    import run_ppo
    from experimental_data.io import atomic_json
    torch.set_num_threads(1)
    model, task, config = deepcopy(run_ppo.load_configs())
    task['episode_duration_s'] = .1
    config['bootstrap'] = None
    config['deployment']['enabled'] = False
    config['ppo'].update(hidden_dim=16, update_epochs=1)
    config['validation'].update(enabled=True, episodes=2, every_episodes=2)
    config['update_guard']['enabled'] = False
    config['early_stopping'] = dict(enabled=True, window_evaluations=4, patience_episodes=8,
                                    minimum_additional_episodes=10)
    for name, data in zip(('model', 'task', 'ppo'), (model, task, config)):
        atomic_json(tmp_path/'config'/f'{name}.json', data)
    monkeypatch.setattr(run_ppo, 'write_active_run', lambda *args: None)
    monkeypatch.setattr(run_ppo, 'evaluate', lambda *args, **kwargs: dict(
        success_rate=.5, mean_episode_reward=200., mean_point_displacement_cost_integral_s=1.))
    run = run_ppo.train(device_name='cpu', requested_episodes=100, batch_size=2,
                        artifact=tmp_path/'run', resume_checkpoint=None,
                        config_directory=tmp_path/'config')
    status = json.loads((run/'status.json').read_text())
    assert status['status'] == 'STOPPED' and status['episodes'] == 14
    assert status['stop_reason'] == 'Validation reward plateau'
    assert status['reward_convergence']['should_stop']
    terminal = torch.load(run/'checkpoints/terminal.pt', map_location='cpu', weights_only=False)
    latest = torch.load(run/'checkpoints/latest.pt', map_location='cpu', weights_only=False)
    assert terminal['episodes'] == latest['episodes'] == 14
    best = torch.load(run/'checkpoints/best_reward.pt', map_location='cpu', weights_only=False)
    assert best['episodes'] == 0
    assert json.loads((run/'best_reward.json').read_text())['mean_episode_reward'] == 200.


def test_continuation_preserves_parent_and_requests_optimizer_state(tmp_path):
    import run_ppo
    from experimental_data.io import atomic_json
    from simulator.workflow import prepare_training
    model, task, config = deepcopy(run_ppo.load_configs())
    model.pop('fullstate_execution', None)
    parent = tmp_path/'runs/ppo/parent'
    for name, data in zip(('model', 'task', 'ppo'), (model, task, config)):
        atomic_json(parent/f'{name}.json', data)
    atomic_json(tmp_path/'config/ppo.json', config)
    old = (parent/'ppo.json').read_bytes()
    settings = dict(enabled=True, window_evaluations=20)
    folder, command = prepare_training(tmp_path, 'ppo', seed=config['seed'], episodes=1000,
        batch=16, device='cpu', resume=parent/'checkpoints/latest.pt',
        keep_optimizer_state=True, early_stopping=settings)
    saved = json.loads((folder/'launch_config/ppo.json').read_text())
    assert not saved['deployment']['reset_optimizer_on_resume']
    assert saved['early_stopping'] == settings
    assert (parent/'ppo.json').read_bytes() == old
    assert '--resume-checkpoint' in command
    atomic_json(parent/'early_stopping_override.json', dict(settings=dict(
        patience_episodes=20000, minimum_additional_episodes=20000)))
    amended, _ = prepare_training(tmp_path, 'ppo', seed=config['seed'], episodes=2000,
        batch=16, device='cpu', resume=parent/'checkpoints/latest.pt', keep_optimizer_state=True)
    inherited = json.loads((amended/'launch_config/ppo.json').read_text())['early_stopping']
    assert inherited['patience_episodes'] == 20000
    assert inherited['minimum_additional_episodes'] == 20000
    assert (parent/'ppo.json').read_bytes() == old
