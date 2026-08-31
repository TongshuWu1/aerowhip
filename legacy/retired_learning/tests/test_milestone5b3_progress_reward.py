from __future__ import annotations

import math

import pytest
import torch

from planning.rl_reward import RLWhipRewardConfig, compose_rl_terminal_reward


def _compose(config: RLWhipRewardConfig, d0: float, d_min: float):
    return compose_rl_terminal_reward(
        strike_reward=torch.tensor([0.25]),
        success=torch.tensor([False]),
        first_entry_marker=torch.tensor([0]),
        first_hit_time_s=torch.tensor([float("nan")]),
        maximum_uav_displacement_m=torch.tensor([0.10]),
        maximum_uav_speed_m_s=torch.tensor([0.50]),
        maximum_command_acceleration_m_s2=torch.tensor([5.0]),
        effort_cost=torch.tensor([0.20]),
        smoothness_cost=torch.tensor([0.10]),
        displacement_limit_m=0.50,
        speed_limit_m_s=3.0,
        acceleration_limit_m_s2=20.0,
        config=config,
        initial_tip_distance_m=torch.tensor([d0]),
        minimum_tip_distance_m=torch.tensor([d_min]),
    )


def test_reward_v2_is_exactly_v1_plus_four_times_progress():
    v1 = _compose(RLWhipRewardConfig(), 1.25, 0.50)
    v2 = _compose(
        RLWhipRewardConfig(profile="rl_whip_reward_v2", progress_weight=4.0),
        1.25,
        0.50,
    )
    expected_progress = (1.25 - 0.50) / 1.25
    assert float(v2.normalized_progress) == pytest.approx(expected_progress)
    assert float(v2.progress) == pytest.approx(4.0 * expected_progress)
    assert float(v2.total) == pytest.approx(
        float(v1.total) + 4.0 * expected_progress
    )


def test_progress_is_zero_for_hover_and_one_at_target_center():
    config = RLWhipRewardConfig(profile="rl_whip_reward_v2", progress_weight=4.0)
    hover = _compose(config, 1.3, 1.3)
    center = _compose(config, 1.3, 0.0)
    assert float(hover.normalized_progress) == pytest.approx(0.0)
    assert float(hover.progress) == pytest.approx(0.0)
    assert float(center.normalized_progress) == pytest.approx(1.0)
    assert float(center.progress) == pytest.approx(4.0)
    assert torch.isfinite(hover.total).all()
    assert torch.isfinite(center.total).all()


def test_unsuccessful_time_term_remains_exactly_zero():
    result = _compose(
        RLWhipRewardConfig(profile="rl_whip_reward_v2", progress_weight=4.0),
        1.0,
        0.5,
    )
    assert float(result.time) == pytest.approx(0.0)
    assert math.isfinite(float(result.total))


def test_reward_versions_enforce_their_frozen_progress_weights():
    with pytest.raises(ValueError, match="progress_weight"):
        RLWhipRewardConfig(profile="rl_whip_reward_v1", progress_weight=4.0)
    with pytest.raises(ValueError, match="progress_weight"):
        RLWhipRewardConfig(profile="rl_whip_reward_v2", progress_weight=0.0)

