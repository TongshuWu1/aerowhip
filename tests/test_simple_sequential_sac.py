from __future__ import annotations

import math

import torch

from learning.sequential_sac_env import (
    ATTACHMENT_RELATIVE_SPEED_SHAPING,
    TARGET_ALIGNED_SAGITTAL_ACTION_MODE,
    decode_target_aligned_sagittal_action,
    direction_gate_quality,
    diagnostic_scientific_whip_success,
    SimpleRewardWeights,
    simple_dense_reward,
    simple_endpoint_success,
    sequential_action_dimension,
    shaping_tip_velocity,
    soft_near_target_strike_components,
    smooth_success_compactness,
    smooth_displacement_cost,
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


def test_target_aligned_sagittal_action_has_no_lateral_acceleration() -> None:
    action = torch.tensor([[0.6, -0.8, 0.5], [2.0, 0.0, -1.0]])
    direction = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    acceleration, body_rate = decode_target_aligned_sagittal_action(
        action,
        direction,
        maximum_acceleration_m_s2=10.0,
        maximum_body_rate_rad_s=4.0,
    )
    assert sequential_action_dimension(TARGET_ALIGNED_SAGITTAL_ACTION_MODE) == 3
    assert torch.allclose(acceleration[0], torch.tensor([6.0, 0.0, -8.0]))
    assert torch.allclose(acceleration[1], torch.tensor([0.0, 10.0, 0.0]))
    assert torch.allclose(body_rate[:, [0, 2]], torch.zeros(2, 2))
    assert torch.allclose(body_rate[:, 1], torch.tensor([2.0, -4.0]))


def test_attachment_relative_speed_shaping_rejects_rigid_translation() -> None:
    tip_velocity = torch.tensor([[5.0, 2.0, 0.0]])
    attachment_velocity = torch.tensor([[5.0, 2.0, 0.0]])
    relative = shaping_tip_velocity(
        tip_velocity,
        attachment_velocity,
        mode=ATTACHMENT_RELATIVE_SPEED_SHAPING,
    )
    assert torch.equal(relative, torch.zeros_like(relative))
    shared_translation = torch.tensor([[3.0, -4.0, 1.0]])
    translated = shaping_tip_velocity(
        tip_velocity + shared_translation,
        attachment_velocity + shared_translation,
        mode=ATTACHMENT_RELATIVE_SPEED_SHAPING,
    )
    assert torch.equal(translated, relative)


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


def test_soft_strike_components_center_direction_credit_on_scientific_gate() -> None:
    speed_quality, direction_quality = soft_near_target_strike_components(
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([0.0, 7.0, 4.0]),
        torch.tensor([0.0, 0.0, 4.0]),
        torch.tensor([0.0, 0.0, 1.0]),
        proximity_scale_m=0.15,
        target_directed_speed_m_s=4.0,
    )
    assert torch.allclose(speed_quality[:2], torch.zeros(2), atol=1.0e-6)
    assert float(speed_quality[2]) > 0.9
    assert float(direction_quality[0]) == 0.0
    assert float(direction_quality[1]) < 0.01
    assert float(direction_quality[2]) > 0.9


def test_direction_gate_quality_rejects_observed_sideways_strike() -> None:
    quality = direction_gate_quality(
        torch.tensor(
            [
                1.0,
                math.cos(math.radians(30.0)),
                math.cos(math.radians(78.0)),
            ]
        )
    )
    assert float(quality[0]) > 0.99
    assert 0.65 < float(quality[1]) < 0.75
    assert float(quality[2]) < 0.025


def test_speed_shaping_is_flat_above_configured_reward_cap() -> None:
    directed_speed = torch.tensor([4.0, 7.0, 10.0])
    strict = strike_quality(
        torch.zeros(3),
        directed_speed,
        torch.ones(3),
        proximity_scale_m=0.15,
        target_directed_speed_m_s=4.0,
        speed_reward_cap_m_s=4.0,
    )
    soft_speed, _ = soft_near_target_strike_components(
        torch.zeros(3),
        directed_speed,
        directed_speed,
        torch.ones(3),
        proximity_scale_m=0.15,
        target_directed_speed_m_s=4.0,
        speed_reward_cap_m_s=4.0,
    )
    assert torch.allclose(strict, strict[:1].expand_as(strict), atol=1.0e-6)
    assert torch.allclose(
        soft_speed, soft_speed[:1].expand_as(soft_speed), atol=1.0e-6
    )


def test_success_compactness_is_smooth_and_never_a_hard_gate() -> None:
    preference = smooth_success_compactness(
        torch.tensor([0.0, 0.5, 1.0]), scale_m=0.5
    )
    assert float(preference[0]) == 1.0
    assert float(preference[0]) > float(preference[1]) > float(preference[2]) > 0.0


def test_displacement_cost_is_zero_at_origin_and_increases_smoothly() -> None:
    cost = smooth_displacement_cost(
        torch.tensor([0.0, 0.25, 0.5]), scale_m=0.35
    )
    assert float(cost[0]) == 0.0
    assert float(cost[0]) < float(cost[1]) < float(cost[2])


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
