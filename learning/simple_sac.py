"""Conventional sequential Soft Actor-Critic for the simple whip baseline.

This module intentionally contains no trajectory codec, one-shot terminal
critic, CEM data, demonstrations, or curriculum.  The actor emits one bounded
six-dimensional control at each MDP step and the critics use the standard
bootstrapped SAC target.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


SIMPLE_SAC_ACTION_DIM = 6


def _mlp(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class SquashedGaussianActor(nn.Module):
    """Tanh-squashed diagonal Gaussian actor used by ordinary SAC."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.trunk = _mlp(observation_dim, 2 * action_dim, hidden_dim)

    def distribution_parameters(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_log_std = self.trunk(observation).chunk(2, dim=-1)
        # The usual SAC range is broad enough for exploration while avoiding
        # singular variances during a multi-million-episode unattended run.
        log_std = -5.0 + 0.5 * (2.0 + 5.0) * (torch.tanh(raw_log_std) + 1.0)
        return mean, log_std

    def sample(
        self, observation: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution_parameters(observation)
        if deterministic:
            pre_tanh = mean
        else:
            pre_tanh = mean + log_std.exp() * torch.randn_like(mean)
        action = torch.tanh(pre_tanh)
        # Gaussian log density plus the exact tanh Jacobian correction.
        gaussian_log_prob = -0.5 * (
            ((pre_tanh - mean) / log_std.exp()).square()
            + 2.0 * log_std
            + math.log(2.0 * math.pi)
        )
        correction = 2.0 * (
            math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh)
        )
        log_probability = (gaussian_log_prob - correction).sum(dim=-1, keepdim=True)
        return action, log_probability


class SoftQNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = _mlp(observation_dim + action_dim, 1, hidden_dim)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((observation, action), dim=-1))


class TransitionReplayBuffer:
    """Fixed-size CPU replay for genuine sequential MDP transitions."""

    def __init__(self, capacity: int, observation_dim: int, action_dim: int) -> None:
        if capacity < 1:
            raise ValueError("Replay capacity must be positive.")
        self.capacity = int(capacity)
        self.observations = torch.empty((capacity, observation_dim), dtype=torch.float32)
        self.actions = torch.empty((capacity, action_dim), dtype=torch.float32)
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32)
        self.next_observations = torch.empty((capacity, observation_dim), dtype=torch.float32)
        self.dones = torch.empty((capacity, 1), dtype=torch.float32)
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_observation: torch.Tensor,
        done: torch.Tensor,
        *,
        include: torch.Tensor | None = None,
    ) -> None:
        tensors = (observation, action, reward, next_observation, done)
        batch = int(observation.shape[0])
        if any(int(value.shape[0]) != batch for value in tensors):
            raise ValueError("Every replay field must have the same leading batch.")
        if include is not None:
            mask = include.to(device=observation.device, dtype=torch.bool)
            tensors = tuple(value[mask] for value in tensors)
            batch = int(tensors[0].shape[0])
        if batch == 0:
            return
        if batch > self.capacity:
            tensors = tuple(value[-self.capacity :] for value in tensors)
            batch = self.capacity
        cpu = tuple(value.detach().to(device="cpu", dtype=torch.float32) for value in tensors)
        first = min(batch, self.capacity - self.position)
        second = batch - first
        destinations = (
            self.observations,
            self.actions,
            self.rewards,
            self.next_observations,
            self.dones,
        )
        for destination, source in zip(destinations, cpu, strict=True):
            destination[self.position : self.position + first].copy_(source[:first])
            if second:
                destination[:second].copy_(source[first:])
        self.position = (self.position + batch) % self.capacity
        self.size = min(self.capacity, self.size + batch)

    def sample(
        self, batch_size: int, *, device: torch.device, generator: torch.Generator
    ) -> tuple[torch.Tensor, ...]:
        if self.size < batch_size:
            raise ValueError("Replay does not yet contain one minibatch.")
        indices = torch.randint(self.size, (batch_size,), generator=generator)
        return tuple(
            value[indices].to(device=device)
            for value in (
                self.observations,
                self.actions,
                self.rewards,
                self.next_observations,
                self.dones,
            )
        )

    def metadata(self) -> dict[str, int]:
        return {"capacity": self.capacity, "size": self.size, "position": self.position}


@dataclass(frozen=True, slots=True)
class SacUpdateMetrics:
    critic1_loss: float
    critic2_loss: float
    actor_loss: float
    alpha_loss: float
    alpha: float
    mean_q: float
    mean_log_probability: float


class SimpleSACAgent:
    """Minimal, standard twin-Q SAC agent with automatic temperature."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        device: torch.device,
        hidden_dim: int = 256,
        actor_lr: float = 3.0e-4,
        critic_lr: float = 3.0e-4,
        alpha_lr: float = 3.0e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        initial_alpha: float = 0.2,
        target_entropy: float | None = None,
    ) -> None:
        self.device = device
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.target_entropy = float(-action_dim if target_entropy is None else target_entropy)
        self.actor = SquashedGaussianActor(observation_dim, action_dim, hidden_dim).to(device)
        self.critic1 = SoftQNetwork(observation_dim, action_dim, hidden_dim).to(device)
        self.critic2 = SoftQNetwork(observation_dim, action_dim, hidden_dim).to(device)
        self.target_critic1 = SoftQNetwork(observation_dim, action_dim, hidden_dim).to(device)
        self.target_critic2 = SoftQNetwork(observation_dim, action_dim, hidden_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)
        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(initial_alpha), dtype=torch.float32, device=device)
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, observation: torch.Tensor, *, deterministic: bool = False) -> torch.Tensor:
        action, _ = self.actor.sample(observation.to(self.device), deterministic=deterministic)
        return action

    @staticmethod
    def _clip(parameters: Any, maximum_norm: float) -> None:
        nn.utils.clip_grad_norm_(parameters, maximum_norm)

    def update(
        self,
        batch: tuple[torch.Tensor, ...],
        *,
        gradient_clip: float = 10.0,
    ) -> SacUpdateMetrics:
        observation, action, reward, next_observation, done = batch
        with torch.no_grad():
            next_action, next_log_probability = self.actor.sample(next_observation)
            target_q = torch.minimum(
                self.target_critic1(next_observation, next_action),
                self.target_critic2(next_observation, next_action),
            ) - self.alpha.detach() * next_log_probability
            target = reward + self.gamma * (1.0 - done) * target_q

        predicted1 = self.critic1(observation, action)
        predicted2 = self.critic2(observation, action)
        critic1_loss = F.mse_loss(predicted1, target)
        critic2_loss = F.mse_loss(predicted2, target)
        self.critic1_optimizer.zero_grad(set_to_none=True)
        critic1_loss.backward()
        self._clip(self.critic1.parameters(), gradient_clip)
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad(set_to_none=True)
        critic2_loss.backward()
        self._clip(self.critic2.parameters(), gradient_clip)
        self.critic2_optimizer.step()

        sampled_action, log_probability = self.actor.sample(observation)
        for critic in (self.critic1, self.critic2):
            critic.requires_grad_(False)
        sampled_q = torch.minimum(
            self.critic1(observation, sampled_action),
            self.critic2(observation, sampled_action),
        )
        actor_loss = (self.alpha.detach() * log_probability - sampled_q).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self._clip(self.actor.parameters(), gradient_clip)
        self.actor_optimizer.step()
        for critic in (self.critic1, self.critic2):
            critic.requires_grad_(True)

        alpha_loss = -(
            self.log_alpha * (log_probability + self.target_entropy).detach()
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_critic1.parameters(), self.critic1.parameters(), strict=True
            ):
                target_parameter.lerp_(parameter, self.tau)
            for target_parameter, parameter in zip(
                self.target_critic2.parameters(), self.critic2.parameters(), strict=True
            ):
                target_parameter.lerp_(parameter, self.tau)

        return SacUpdateMetrics(
            critic1_loss=float(critic1_loss.detach()),
            critic2_loss=float(critic2_loss.detach()),
            actor_loss=float(actor_loss.detach()),
            alpha_loss=float(alpha_loss.detach()),
            alpha=float(self.alpha.detach()),
            mean_q=float(sampled_q.detach().mean()),
            mean_log_probability=float(log_probability.detach().mean()),
        )

    def checkpoint(self) -> dict[str, Any]:
        return {
            "schema": "simple_sequential_sac_checkpoint_v1",
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic1_optimizer": self.critic1_optimizer.state_dict(),
            "critic2_optimizer": self.critic2_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "gamma": self.gamma,
            "tau": self.tau,
            "target_entropy": self.target_entropy,
        }
