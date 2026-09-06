from __future__ import annotations

import torch

from learning import FORCE_ACTION_DIM, SimplePPOAgent
from learning.simple_ppo import PPORollout
from run_ppo import (
    guarded_update_is_acceptable,
    guarded_update_is_unsafe,
    load_configs,
    training_success_condition,
    validate_contract,
)


def test_force_policy_has_exactly_three_outputs() -> None:
    agent = SimplePPOAgent(17, device=torch.device("cpu"), hidden_dim=32)
    observation = torch.zeros((5, 17))
    action = agent.deterministic_action(observation)
    assert FORCE_ACTION_DIM == 3
    assert action.shape == (5, 3)
    assert torch.isfinite(action).all()


def test_fresh_policy_can_start_from_a_time_indexed_force_prior() -> None:
    prior = {
        "enabled": True,
        "source": "test",
        "total_control_steps": 10,
        "phase_control_steps": [2, 3],
        "actions": [[-0.25, 0.0, 0.1], [0.5, 0.0, -0.1]],
    }
    agent = SimplePPOAgent(
        7,
        device=torch.device("cpu"),
        hidden_dim=32,
        action_prior=prior,
        initial_log_std=-4.0,
    )
    observation = torch.zeros((5, 7))
    observation[:, -1] = torch.tensor([1.0, 0.9, 0.8, 0.6, 0.5])
    expected = torch.tensor(
        [
            [-0.25, 0.0, 0.1],
            [-0.25, 0.0, 0.1],
            [0.5, 0.0, -0.1],
            [0.5, 0.0, -0.1],
            [0.0, 0.0, 0.0],
        ]
    )
    assert torch.allclose(agent.deterministic_action(observation), expected)
    assert agent.checkpoint()["action_prior"] == agent.policy.action_prior
    assert "policy_optimizer" in agent.checkpoint()
    assert "value_optimizer" in agent.checkpoint()
    assert "optimizer" not in agent.checkpoint()


def test_disabled_curriculum_uses_the_final_contract_from_the_start() -> None:
    model, task, config = load_configs()
    validate_contract(model, task, config)
    first = training_success_condition(
        config, task, episodes=0, episodes_target=100
    )
    assert first["name"] == "final"
    assert first["target_radius_m"] == task["success"]["tip_target_distance_m"]
    assert (
        first["maximum_tip_velocity_direction_error_deg"]
        == task["success"]["maximum_tip_velocity_to_desired_direction_error_deg"]
    )


def test_guard_preserves_hit_and_accepts_lower_point_cost() -> None:
    accepted = {
        "success_rate": 1.0,
        "mean_episode_reward": 130.7,
        "mean_point_displacement_cost_integral_s": 5.14,
    }
    candidate = {
        "success_rate": 1.0,
        "mean_episode_reward": 130.6,
        "mean_point_displacement_cost_integral_s": 5.13,
    }
    guard = {
        "minimum_final_success_rate": 1.0,
        "maximum_point_cost_regression_s": 0.0,
        "maximum_reward_regression": 1.0,
    }
    assert guarded_update_is_acceptable(candidate, accepted, guard)
    assert not guarded_update_is_unsafe(candidate, accepted, guard)
    candidate["success_rate"] = 0.0
    assert not guarded_update_is_acceptable(candidate, accepted, guard)
    assert guarded_update_is_unsafe(candidate, accepted, guard)


def test_symmetric_policy_keeps_lateral_force_deterministic() -> None:
    agent = SimplePPOAgent(
        7,
        device=torch.device("cpu"),
        hidden_dim=32,
        stochastic_action_indices=[0, 2],
    )
    observation = torch.zeros((64, 7))
    action, _, _ = agent.act(observation)
    assert torch.count_nonzero(action[:, 1]) == 0
    assert torch.count_nonzero(action[:, 0]) > 0
    assert torch.count_nonzero(action[:, 2]) > 0


def test_deterministic_policy_uses_the_training_distribution_action_mask() -> None:
    agent = SimplePPOAgent(
        7,
        device=torch.device("cpu"),
        hidden_dim=32,
        stochastic_action_indices=[0, 2],
    )
    # A nonzero masked output must never leak into validation or replay.
    with torch.no_grad():
        agent.policy.mean_network[-1].bias[1] = 0.7
    observation = torch.randn((8, 7))
    expected = torch.tanh(agent.policy.distribution(observation).loc)
    actual = agent.deterministic_action(observation)
    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(actual[:, 1]) == 0


def test_deterministic_mask_preserves_a_configured_force_prior() -> None:
    agent = SimplePPOAgent(
        7,
        device=torch.device("cpu"),
        hidden_dim=32,
        stochastic_action_indices=[0, 2],
        action_prior={
            "enabled": True,
            "total_control_steps": 10,
            "phase_control_steps": [2],
            "actions": [[0.2, 0.3, -0.1]],
        },
    )
    with torch.no_grad():
        agent.policy.mean_network[-1].bias[1] = 0.7
    observation = torch.zeros((2, 7))
    observation[:, -1] = torch.tensor([1.0, 0.5])
    actual = agent.deterministic_action(observation)
    torch.testing.assert_close(actual, torch.tanh(agent.policy.distribution(observation).loc))
    torch.testing.assert_close(actual[:, 1], torch.tensor([0.3, 0.0]))


def test_small_ppo_update_is_finite() -> None:
    agent = SimplePPOAgent(7, device=torch.device("cpu"), hidden_dim=32)
    rollout = PPORollout.allocate(4, 4, 7, FORCE_ACTION_DIM, device=torch.device("cpu"))
    observations = torch.randn_like(rollout.observations)
    for step in range(4):
        action, log_probability, value = agent.act(observations[step])
        rollout.observations[step].copy_(observations[step])
        rollout.actions[step].copy_(action)
        rollout.log_probabilities[step].copy_(log_probability)
        rollout.values[step].copy_(value)
    rollout.rewards.normal_()
    rollout.dones.zero_()
    rollout.masks.fill_(1.0)
    metrics = agent.update(
        rollout,
        minibatch_size=8,
        epochs=2,
        generator=torch.Generator().manual_seed(4),
    )
    assert metrics.valid_transitions == 16
    assert metrics.gradient_norm >= 0.0
