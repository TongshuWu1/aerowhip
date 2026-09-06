"""Small, conventional on-policy PPO implementation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.distributions import Normal


FORCE_ACTION_DIM = 3


def _mlp(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


class BoundedGaussianPolicy(nn.Module):
    """Gaussian residual around an optional time-indexed action prior."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        *,
        action_prior: dict[str, Any] | None = None,
        stochastic_action_indices: list[int] | tuple[int, ...] | None = None,
        initial_log_std: float | list[float] | tuple[float, ...] = -0.5,
        minimum_log_std: float = -10.0,
    ) -> None:
        super().__init__()
        self.mean_network = _mlp(observation_dim, action_dim, hidden_dim)
        indices = (
            tuple(range(action_dim))
            if stochastic_action_indices is None
            else tuple(int(index) for index in stochastic_action_indices)
        )
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("stochastic_action_indices must be unique and non-empty.")
        if any(index < 0 or index >= action_dim for index in indices):
            raise ValueError("A stochastic action index is out of range.")
        stochastic_mask = torch.zeros(action_dim, dtype=torch.bool)
        stochastic_mask[list(indices)] = True
        self.register_buffer("_stochastic_mask", stochastic_mask, persistent=False)
        self.minimum_log_std = float(minimum_log_std)
        if not math.isfinite(self.minimum_log_std) or self.minimum_log_std >= 1.0:
            raise ValueError("minimum_log_std must be finite and below 1.")
        initial_std = torch.as_tensor(initial_log_std, dtype=torch.float32)
        if initial_std.ndim == 0:
            initial_std = initial_std.expand(action_dim).clone()
        if initial_std.shape != (action_dim,):
            raise ValueError("initial_log_std must be a scalar or one value per action.")
        if not torch.isfinite(initial_std).all():
            raise ValueError("initial_log_std must be finite.")
        self.log_std = nn.Parameter(initial_std)
        self.action_prior: dict[str, Any] | None = None
        self.set_action_prior(action_prior)
        final = self.mean_network[-1]
        assert isinstance(final, nn.Linear)
        if self.action_prior is None:
            nn.init.orthogonal_(final.weight, gain=0.01)
        else:
            nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def set_action_prior(self, value: dict[str, Any] | None) -> None:
        if value is None or not bool(value.get("enabled", True)):
            self.action_prior = None
            return
        phase_steps = [int(item) for item in value["phase_control_steps"]]
        actions = [[float(number) for number in row] for row in value["actions"]]
        total_steps = int(value["total_control_steps"])
        if len(phase_steps) != len(actions) or not phase_steps:
            raise ValueError("Action-prior phases and actions must have equal length.")
        if any(step < 1 for step in phase_steps) or sum(phase_steps) > total_steps:
            raise ValueError("Action-prior phase durations are invalid.")
        if any(len(row) != self.log_std.numel() for row in actions):
            raise ValueError("Action-prior vectors have the wrong dimension.")
        if any(abs(number) >= 1.0 for row in actions for number in row):
            raise ValueError("Action-prior values must lie strictly inside (-1, 1).")
        self.action_prior = {
            "enabled": True,
            "source": str(value.get("source", "configured_action_prior")),
            "total_control_steps": total_steps,
            "phase_control_steps": phase_steps,
            "actions": actions,
        }

    def _prior_latent(self, observation: torch.Tensor) -> torch.Tensor:
        prior = self.action_prior
        if prior is None:
            return torch.zeros(
                (*observation.shape[:-1], self.log_std.numel()),
                dtype=observation.dtype,
                device=observation.device,
            )
        elapsed = torch.round(
            (1.0 - observation[..., -1]).clamp(0.0, 1.0)
            * int(prior["total_control_steps"])
        )
        action = torch.zeros(
            (*observation.shape[:-1], self.log_std.numel()),
            dtype=observation.dtype,
            device=observation.device,
        )
        start = 0
        for duration, values in zip(
            prior["phase_control_steps"], prior["actions"], strict=True
        ):
            end = start + int(duration)
            phase_action = torch.as_tensor(
                values, dtype=observation.dtype, device=observation.device
            )
            action = torch.where(
                ((elapsed >= start) & (elapsed < end))[..., None],
                phase_action,
                action,
            )
            start = end
        return torch.atanh(action.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6))

    def distribution(self, observation: torch.Tensor) -> Normal:
        residual = self.mean_network(observation)
        mean = self._prior_latent(observation) + residual * self._stochastic_mask
        std = self.log_std.clamp(self.minimum_log_std, 1.0).exp().expand_as(mean)
        return Normal(mean, std)

    def _log_probability(
        self, distribution: Normal, latent: torch.Tensor
    ) -> torch.Tensor:
        action = torch.tanh(latent)
        correction = torch.log(torch.clamp(1.0 - action.square(), min=1.0e-6))
        terms = (distribution.log_prob(latent) - correction) * self._stochastic_mask
        return terms.sum(dim=-1, keepdim=True)

    def sample(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        noise = torch.randn_like(distribution.loc) * self._stochastic_mask
        latent = distribution.loc + distribution.scale * noise
        action = torch.tanh(latent)
        log_probability = self._log_probability(distribution, latent)
        entropy = (distribution.entropy() * self._stochastic_mask).sum(
            dim=-1, keepdim=True
        )
        return action, log_probability, entropy

    def evaluate(
        self, observation: torch.Tensor, bounded_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action = bounded_action.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
        latent = torch.atanh(action)
        distribution = self.distribution(observation)
        log_probability = self._log_probability(distribution, latent)
        entropy = (distribution.entropy() * self._stochastic_mask).sum(
            dim=-1, keepdim=True
        )
        return log_probability, entropy

    def deterministic(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(
            self._prior_latent(observation)
            + self.mean_network(observation) * self._stochastic_mask
        )


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
        action_dim: int = FORCE_ACTION_DIM,
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
        action_prior: dict[str, Any] | None = None,
        stochastic_action_indices: list[int] | tuple[int, ...] | None = None,
        initial_log_std: float | list[float] | tuple[float, ...] = -0.5,
        minimum_log_std: float = -10.0,
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
            observation_dim,
            action_dim,
            hidden_dim,
            action_prior=action_prior,
            stochastic_action_indices=stochastic_action_indices,
            initial_log_std=initial_log_std,
            minimum_log_std=minimum_log_std,
        ).to(device)
        self.value = StateValue(observation_dim, hidden_dim).to(device)
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=learning_rate,
            eps=1.0e-5,
        )
        self.value_optimizer = torch.optim.Adam(
            self.value.parameters(),
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
            permutation = torch.randperm(
                valid_count, generator=generator, device=self.device
            )
            stop_for_kl = False
            for start in range(0, valid_count, int(minibatch_size)):
                from .training_control import check_training_stop
                check_training_stop()
                indices = permutation[start : start + int(minibatch_size)]
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
                policy_objective = (
                    policy_loss - self.entropy_coefficient * entropy_mean
                )
                self.policy_optimizer.zero_grad(set_to_none=True)
                policy_objective.backward()
                policy_gradient_norm = nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.maximum_gradient_norm,
                )
                self.policy_optimizer.step()

                self.value_optimizer.zero_grad(set_to_none=True)
                (self.value_coefficient * value_loss).backward()
                value_gradient_norm = nn.utils.clip_grad_norm_(
                    self.value.parameters(),
                    self.maximum_gradient_norm,
                )
                self.value_optimizer.step()
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
                    max(
                        float(torch.as_tensor(policy_gradient_norm).detach()),
                        float(torch.as_tensor(value_gradient_norm).detach()),
                    ),
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
            "schema": "force_ppo_checkpoint_v1",
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "gradient_updates": self.gradient_updates,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_ratio": self.clip_ratio,
            "action_prior": self.policy.action_prior,
        }
