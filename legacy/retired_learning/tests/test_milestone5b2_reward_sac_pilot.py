from __future__ import annotations

import math

import pytest
import torch

from learning.sac import TerminalSacAgent
from planning.rl_reward import (
    RLWhipRewardConfig,
    compose_rl_terminal_reward,
    rl_event_reward,
)


def test_rl_event_reward_is_bounded_smooth_and_proximity_gated():
    config = RLWhipRewardConfig()
    distance = torch.tensor([0.0, 0.2, 0.8], dtype=torch.float64)
    directed = torch.full_like(distance, 4.0)
    speed = torch.full_like(distance, 4.0)
    cosine = torch.full_like(distance, math.cos(math.radians(20.0)))
    reward = rl_event_reward(distance, directed, speed, cosine, config)
    assert torch.isfinite(reward).all()
    assert reward[0] > reward[1] > reward[2] >= 0.0
    assert float(reward.max()) <= 5.0


def test_zero_speed_has_no_direction_reward_and_no_nan():
    config = RLWhipRewardConfig()
    reward = rl_event_reward(
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        config,
    )
    assert float(reward) == pytest.approx(2.0)


def test_rl_terminal_reward_components_are_finite_and_safety_is_clipped():
    components = compose_rl_terminal_reward(
        strike_reward=torch.tensor([4.5, 1.0]),
        success=torch.tensor([True, False]),
        first_entry_marker=torch.tensor([10, 5]),
        first_hit_time_s=torch.tensor([1.1, float("nan")]),
        maximum_uav_displacement_m=torch.tensor([0.48, 100.0]),
        maximum_uav_speed_m_s=torch.tensor([2.5, 100.0]),
        maximum_command_acceleration_m_s2=torch.tensor([19.0, 100.0]),
        effort_cost=torch.tensor([0.2, 0.2]),
        smoothness_cost=torch.tensor([0.3, 0.3]),
        displacement_limit_m=0.5,
        speed_limit_m_s=3.0,
        acceleration_limit_m_s2=20.0,
        config=RLWhipRewardConfig(),
    )
    assert torch.isfinite(components.total).all()
    assert float(components.success[0]) == pytest.approx(5.0)
    assert float(components.non_tip[1]) == pytest.approx(-1.0)
    assert float(components.safety[1]) == pytest.approx(-24.0)


def test_terminal_sac_supports_huber_critic_regression():
    torch.manual_seed(5)
    agent = TerminalSacAgent.create(device="cpu", critic_loss="smooth_l1")
    result = agent.update(
        torch.randn(32, 83),
        torch.tanh(torch.randn(32, 49)),
        torch.linspace(-5.0, 5.0, 32),
    )
    assert agent.critic_loss == "smooth_l1"
    assert all(math.isfinite(value) for value in result.values())
