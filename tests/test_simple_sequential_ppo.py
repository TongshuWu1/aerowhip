from __future__ import annotations

import torch

from learning.simple_ppo import (
    SIMPLE_PPO_ACTION_DIM,
    BoundedGaussianPolicy,
    PPORollout,
    SimplePPOAgent,
    generalized_advantage_estimate,
)
from learning.ppo_validation import rolling_success_rate


def test_bounded_policy_has_finite_actions_and_consistent_log_probabilities() -> None:
    torch.manual_seed(3)
    policy = BoundedGaussianPolicy(84, SIMPLE_PPO_ACTION_DIM, hidden_dim=32)
    observation = torch.randn(16, 84)
    action, sampled_log_probability, entropy = policy.sample(observation)
    evaluated_log_probability, evaluated_entropy = policy.evaluate(observation, action)
    assert action.shape == (16, 6)
    assert float(action.detach().abs().max()) < 1.0
    assert torch.allclose(sampled_log_probability, evaluated_log_probability, atol=2e-5)
    assert torch.allclose(entropy, evaluated_entropy)
    assert bool(torch.isfinite(action).all() and torch.isfinite(entropy).all())


def test_gae_excludes_post_failure_transitions() -> None:
    rewards = torch.tensor([[[1.0]], [[2.0]], [[50.0]], [[50.0]]])
    dones = torch.tensor([[[0.0]], [[1.0]], [[0.0]], [[1.0]]])
    masks = torch.tensor([[[1.0]], [[1.0]], [[0.0]], [[0.0]]])
    values = torch.zeros_like(rewards)
    advantages, returns = generalized_advantage_estimate(
        rewards, dones, masks, values, gamma=1.0, gae_lambda=1.0
    )
    assert advantages[:, 0, 0].tolist() == [3.0, 2.0, 0.0, 0.0]
    assert torch.equal(advantages, returns)


def test_ppo_update_is_finite_and_changes_policy_parameters() -> None:
    torch.manual_seed(7)
    device = torch.device("cpu")
    agent = SimplePPOAgent(84, device=device, hidden_dim=32)
    rollout = PPORollout.allocate(8, 16, 84, 6, device=device)
    rollout.observations.normal_()
    with torch.no_grad():
        for step in range(8):
            action, log_probability, value = agent.act(rollout.observations[step])
            rollout.actions[step].copy_(action)
            rollout.log_probabilities[step].copy_(log_probability)
            rollout.values[step].copy_(value)
    rollout.rewards.normal_()
    rollout.dones.zero_()
    rollout.dones[-1].fill_(1.0)
    rollout.masks.fill_(1.0)
    before = [parameter.detach().clone() for parameter in agent.policy.parameters()]
    metrics = agent.update(
        rollout,
        minibatch_size=32,
        epochs=2,
        generator=torch.Generator(device="cpu").manual_seed(11),
    )
    after = list(agent.policy.parameters())
    assert metrics.valid_transitions == 128
    assert metrics.epochs_completed >= 1
    assert all(
        torch.isfinite(torch.tensor(value))
        for value in (
            metrics.policy_loss,
            metrics.value_loss,
            metrics.entropy,
            metrics.approximate_kl,
            metrics.clip_fraction,
            metrics.gradient_norm,
        )
    )
    assert any(not torch.equal(old, new) for old, new in zip(before, after, strict=True))


def test_training_success_rolling_rate_uses_episode_counts() -> None:
    episodes = torch.tensor([2_048, 4_096, 6_144, 8_192]).numpy()
    successes = torch.tensor([0, 1, 1, 3]).numpy()
    rolling = rolling_success_rate(episodes, successes, window_episodes=4_096)
    assert rolling.tolist() == [0.0, 1.0 / 4_096.0, 1.0 / 4_096.0, 2.0 / 4_096.0]
