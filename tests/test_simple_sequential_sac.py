from __future__ import annotations

import math

import torch

from learning.sequential_sac_env import (
    diagnostic_scientific_whip_success,
    SimpleRewardWeights,
    simple_dense_reward,
    simple_endpoint_success,
    strike_quality,
    task_whip_success,
)
from learning.simple_sac import (
    SIMPLE_SAC_ACTION_DIM,
    SimpleSACAgent,
    SquashedGaussianActor,
    TransitionReplayBuffer,
)


def test_simple_endpoint_success_uses_position_speed_and_direction() -> None:
    target = torch.tensor([[1.0, 0.0, 0.0]])
    direction = torch.tensor([[1.0, 0.0, 0.0]])
    assert bool(
        simple_endpoint_success(
            torch.tensor([[1.04, 0.0, 0.0]]),
            torch.tensor([[4.1, 0.0, 0.0]]),
            target,
            direction,
        )[0]
    )
    assert not bool(
        simple_endpoint_success(
            torch.tensor([[1.04, 0.0, 0.0]]),
            torch.tensor([[-4.1, 0.0, 0.0]]),
            target,
            direction,
        )[0]
    )


def test_simple_reward_rewards_progress_and_penalizes_displacement() -> None:
    weights = SimpleRewardWeights()
    good = simple_dense_reward(
        torch.tensor([1.0]),
        torch.tensor([0.8]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([0.1]),
        torch.tensor([False]),
        torch.tensor([False]),
        weights,
    )
    displaced = simple_dense_reward(
        torch.tensor([1.0]),
        torch.tensor([0.8]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([0.5]),
        torch.tensor([False]),
        torch.tensor([False]),
        weights,
    )
    assert float(good) > float(displaced)


def test_complete_scientific_success_requires_tip_first_and_all_safety_gates() -> None:
    common = {
        "endpoint_event_found": torch.tensor([True, True, True]),
        "first_entry_marker": torch.tensor([10, 9, 10]),
        "maximum_uav_displacement_m": torch.tensor([0.49, 0.40, 0.51]),
        "maximum_uav_speed_m_s": torch.tensor([2.9, 2.0, 2.0]),
        "maximum_command_acceleration_m_s2": torch.tensor([19.9, 10.0, 10.0]),
        "finite": torch.tensor([True, True, True]),
    }
    assert diagnostic_scientific_whip_success(**common).tolist() == [True, False, False]


def test_task_whip_success_is_single_tip_first_attempt_without_time_gate() -> None:
    success = task_whip_success(
        endpoint_event_found=torch.tensor([True, True, True, True]),
        first_entry_marker=torch.tensor([10, 10, 9, 10]),
        finite=torch.tensor([True, True, True, False]),
    )
    assert success.tolist() == [True, True, False, False]


def test_strike_quality_strongly_rejects_sideways_target_entry() -> None:
    quality = strike_quality(
        torch.zeros(3),
        torch.full((3,), 8.0),
        torch.tensor(
            [
                1.0,
                math.cos(math.radians(30.0)),
                math.cos(math.radians(73.0)),
            ]
        ),
        proximity_scale_m=0.15,
        target_directed_speed_m_s=4.0,
        maximum_direction_error_deg=30.0,
    )
    assert float(quality[0]) > float(quality[1]) > float(quality[2])
    assert float(quality[2] / quality[0]) < 0.05


def test_actor_and_replay_implement_standard_sequential_sac_contract() -> None:
    observation_dim = 84
    actor = SquashedGaussianActor(observation_dim, SIMPLE_SAC_ACTION_DIM)
    action, log_probability = actor.sample(torch.randn(8, observation_dim))
    assert action.shape == (8, SIMPLE_SAC_ACTION_DIM)
    assert log_probability.shape == (8, 1)
    assert bool(torch.isfinite(action).all() and torch.isfinite(log_probability).all())
    assert float(action.detach().abs().max()) <= 1.0

    replay = TransitionReplayBuffer(64, observation_dim, SIMPLE_SAC_ACTION_DIM)
    observation = torch.randn(16, observation_dim)
    replay.add(
        observation,
        torch.rand(16, SIMPLE_SAC_ACTION_DIM) * 2.0 - 1.0,
        torch.randn(16, 1),
        torch.randn(16, observation_dim),
        torch.zeros(16, 1),
    )
    generator = torch.Generator().manual_seed(1)
    batch = replay.sample(8, device=torch.device("cpu"), generator=generator)
    agent = SimpleSACAgent(observation_dim, SIMPLE_SAC_ACTION_DIM, device=torch.device("cpu"))
    metrics = agent.update(batch)
    assert all(
        torch.isfinite(torch.tensor(value))
        for value in (
            metrics.critic1_loss,
            metrics.critic2_loss,
            metrics.actor_loss,
            metrics.alpha,
        )
    )
