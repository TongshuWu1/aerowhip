import json
from pathlib import Path

from experimental_data.io import atomic_json
from tools.watch_reward_plateau import assess_run, request_stop


def make_run(root):
    settings = dict(enabled=True, window_evaluations=20, patience_episodes=102400,
                    minimum_additional_episodes=102400)
    atomic_json(root/'ppo.json', dict(early_stopping=settings))
    atomic_json(root/'run.json', dict(parent_training_episodes=167936))
    atomic_json(root/'status.json', dict(status='RUNNING'))
    atomic_json(root/'early_stopping_override.json', dict(settings=dict(
        patience_episodes=20000, minimum_additional_episodes=20000)))
    for i in range(40):
        atomic_json(root/'validation'/f'{167936+i*1024:010d}.json', dict(
            training_episodes=167936+i*1024, evaluation_id=str(i), mean_episode_reward=200.))
    return root


def test_live_rule_reuses_history_and_rounds_to_next_complete_validation(tmp_path):
    run = make_run(tmp_path)
    frozen = (run/'ppo.json').read_bytes()
    last = run/'validation'/f'{167936+39*1024:010d}.json'
    saved = last.read_bytes()
    last.unlink()
    assert not assess_run(run)['should_stop']  # Only 19 batches since the first smoothed mean.
    last.write_bytes(saved)
    decision = assess_run(run)
    assert decision['should_stop']
    assert decision['attempts_without_improvement'] == 20480
    assert decision['settings']['patience_episodes'] == 20000
    assert request_stop(run, decision)
    assert json.loads((run/'STOP_REQUESTED').read_text())['source'] == 'reward_plateau_override'
    assert (run/'ppo.json').read_bytes() == frozen


def test_existing_user_stop_is_preserved(tmp_path):
    run = make_run(tmp_path)
    (run/'STOP_REQUESTED').write_text('User stop')
    assert not request_stop(run, assess_run(run))
    assert (run/'STOP_REQUESTED').read_text() == 'User stop'


def test_failed_worker_is_not_relabelled_as_a_plateau(tmp_path):
    run = make_run(tmp_path)
    atomic_json(run/'status.json', dict(status='FAILED'))
    assert not request_stop(run, assess_run(run))
    assert not (run/'STOP_REQUESTED').exists()
