"""Conventional on-policy PPO for the simple sequential whip environment."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.distributions import Normal


SIMPLE_PPO_ACTION_DIM = 6


def _mlp(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


class BoundedGaussianPolicy(nn.Module):
    """State-dependent Gaussian mean with a tanh-bounded physical action."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.mean_network = _mlp(observation_dim, action_dim, hidden_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        final = self.mean_network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.orthogonal_(final.weight, gain=0.01)
        nn.init.zeros_(final.bias)

    def distribution(self, observation: torch.Tensor) -> Normal:
        mean = self.mean_network(observation)
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        return Normal(mean, std)

    @staticmethod
    def _log_probability(distribution: Normal, latent: torch.Tensor) -> torch.Tensor:
        action = torch.tanh(latent)
        correction = torch.log(torch.clamp(1.0 - action.square(), min=1.0e-6))
        return (distribution.log_prob(latent) - correction).sum(dim=-1, keepdim=True)

    def sample(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        latent = distribution.rsample()
        action = torch.tanh(latent)
        log_probability = self._log_probability(distribution, latent)
        entropy = distribution.entropy().sum(dim=-1, keepdim=True)
        return action, log_probability, entropy

    def evaluate(
        self, observation: torch.Tensor, bounded_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action = bounded_action.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
        latent = torch.atanh(action)
        distribution = self.distribution(observation)
        log_probability = self._log_probability(distribution, latent)
        entropy = distribution.entropy().sum(dim=-1, keepdim=True)
        return log_probability, entropy

    def deterministic(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mean_network(observation))


class StateValue(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = _mlp(observation_dim, 1, hidden_dim)
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.orthogonal_(final.weight, gain=1.0)
        nn.init.zeros_(final.bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


@dataclass(frozen=True, slots=True)
class PPOUpdateMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    gradient_norm: float
    epochs_completed: int
    valid_transitions: int


@dataclass(slots=True)
class PPORollout:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    masks: torch.Tensor
    log_probabilities: torch.Tensor
    values: torch.Tensor

    @classmethod
    def allocate(
        cls,
        steps: int,
        batch_size: int,
        observation_dim: int,
        action_dim: int,
        *,
        device: torch.device,
    ) -> "PPORollout":
        return cls(
            observations=torch.empty(
                (steps, batch_size, observation_dim), dtype=torch.float32, device=device
            ),
            actions=torch.empty(
                (steps, batch_size, action_dim), dtype=torch.float32, device=device
            ),
            rewards=torch.empty((steps, batch_size, 1), dtype=torch.float32, device=device),
            dones=torch.empty((steps, batch_size, 1), dtype=torch.float32, device=device),
            masks=torch.empty((steps, batch_size, 1), dtype=torch.float32, device=device),
            log_probabilities=torch.empty(
                (steps, batch_size, 1), dtype=torch.float32, device=device
            ),
            values=torch.empty((steps, batch_size, 1), dtype=torch.float32, device=device),
        )


def generalized_advantage_estimate(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    masks: torch.Tensor,
    values: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE while excluding post-failure quarantine transitions."""

    if not (rewards.shape == dones.shape == masks.shape == values.shape):
        raise ValueError("PPO rollout scalar tensors must have identical shapes.")
    advantages = torch.zeros_like(rewards)
    accumulator = torch.zeros_like(rewards[0])
    zero = torch.zeros_like(values[0])
    for step in range(rewards.shape[0] - 1, -1, -1):
        next_value = zero if step == rewards.shape[0] - 1 else values[step + 1]
        nonterminal = 1.0 - dones[step]
        delta = (
            rewards[step] + gamma * nonterminal * next_value - values[step]
        ) * masks[step]
        accumulator = (
            delta + gamma * gae_lambda * nonterminal * accumulator
        ) * masks[step]
        advantages[step] = accumulator
    returns = advantages + values
    return advantages, returns


class SimplePPOAgent:
    """Clipped PPO with GAE, one value baseline, and no replay."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int = SIMPLE_PPO_ACTION_DIM,
        *,
        device: torch.device,
        hidden_dim: int = 256,
        learning_rate: float = 3.0e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coefficient: float = 0.5,
        entropy_coefficient: float = 0.01,
        maximum_gradient_norm: float = 0.5,
        target_kl: float = 0.02,
    ) -> None:
        self.device = device
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_ratio = float(clip_ratio)
        self.value_coefficient = float(value_coefficient)
        self.entropy_coefficient = float(entropy_coefficient)
        self.maximum_gradient_norm = float(maximum_gradient_norm)
        self.target_kl = float(target_kl)
        self.policy = BoundedGaussianPolicy(
            observation_dim, action_dim, hidden_dim
        ).to(device)
        self.value = StateValue(observation_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=learning_rate,
            eps=1.0e-5,
        )
        self.gradient_updates = 0

    @torch.no_grad()
    def act(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action, log_probability, _entropy = self.policy.sample(observation)
        return action, log_probability, self.value(observation)

    @torch.no_grad()
    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        return self.policy.deterministic(observation)

    def update(
        self,
        rollout: PPORollout,
        *,
        minibatch_size: int,
        epochs: int,
        generator: torch.Generator,
    ) -> PPOUpdateMetrics:
        advantages, returns = generalized_advantage_estimate(
            rollout.rewards,
            rollout.dones,
            rollout.masks,
            rollout.values,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        valid = rollout.masks.reshape(-1) > 0.5
        valid_count = int(valid.sum().detach().cpu())
        if valid_count < 2:
            raise RuntimeError("PPO rollout contains too few valid transitions.")
        observations = rollout.observations.reshape(
            -1, rollout.observations.shape[-1]
        )[valid]
        actions = rollout.actions.reshape(-1, rollout.actions.shape[-1])[valid]
        old_log_probability = rollout.log_probabilities.reshape(-1, 1)[valid]
        old_values = rollout.values.reshape(-1, 1)[valid]
        advantages = advantages.reshape(-1, 1)[valid]
        returns = returns.reshape(-1, 1)[valid]
        advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1.0e-6)

        totals = {
            "policy": 0.0,
            "value": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
            "clip": 0.0,
            "grad": 0.0,
        }
        update_count = 0
        epochs_completed = 0
        for epoch in range(int(epochs)):
            permutation = torch.randperm(valid_count, generator=generator, device="cpu")
            stop_for_kl = False
            for start in range(0, valid_count, int(minibatch_size)):
                indices = permutation[start : start + int(minibatch_size)].to(self.device)
                new_log_probability, entropy = self.policy.evaluate(
                    observations[indices], actions[indices]
                )
                value = self.value(observations[indices])
                log_ratio = new_log_probability - old_log_probability[indices]
                ratio = torch.exp(log_ratio)
                unclipped = ratio * advantages[indices]
                clipped = torch.clamp(
                    ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                ) * advantages[indices]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_clipped = old_values[indices] + torch.clamp(
                    value - old_values[indices], -self.clip_ratio, self.clip_ratio
                )
                value_loss = 0.5 * torch.maximum(
                    (value - returns[indices]).square(),
                    (value_clipped - returns[indices]).square(),
                ).mean()
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + self.value_coefficient * value_loss
                    - self.entropy_coefficient * entropy_mean
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    self.maximum_gradient_norm,
                )
                self.optimizer.step()
                self.gradient_updates += 1

                with torch.no_grad():
                    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        torch.abs(ratio - 1.0) > self.clip_ratio
                    ).float().mean()
                values = (
                    float(policy_loss.detach()),
                    float(value_loss.detach()),
                    float(entropy_mean.detach()),
                    float(approximate_kl.detach()),
                    float(clip_fraction.detach()),
                    float(torch.as_tensor(gradient_norm).detach()),
                )
                if not all(math.isfinite(item) for item in values):
                    raise FloatingPointError("PPO update produced a non-finite metric.")
                for name, item in zip(totals, values, strict=True):
                    totals[name] += item
                update_count += 1
                if float(approximate_kl) > 1.5 * self.target_kl:
                    stop_for_kl = True
                    break
            epochs_completed = epoch + 1
            if stop_for_kl:
                break
        scale = 1.0 / max(update_count, 1)
        return PPOUpdateMetrics(
            policy_loss=totals["policy"] * scale,
            value_loss=totals["value"] * scale,
            entropy=totals["entropy"] * scale,
            approximate_kl=totals["kl"] * scale,
            clip_fraction=totals["clip"] * scale,
            gradient_norm=totals["grad"] * scale,
            epochs_completed=epochs_completed,
            valid_transitions=valid_count,
        )

    def checkpoint(self) -> dict[str, Any]:
        return {
            "schema": "simple_sequential_ppo_checkpoint_v1",
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "gradient_updates": self.gradient_updates,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_ratio": self.clip_ratio,
        }
