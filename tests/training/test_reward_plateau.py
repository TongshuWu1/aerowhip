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
