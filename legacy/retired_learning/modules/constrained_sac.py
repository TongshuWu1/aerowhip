"""Feasibility-constrained local spectral SAC for Milestone 5B.6."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .policy_action import POLICY_ACTION_DIM
from .policy_context import POLICY_CONTEXT_DIM
from .sac import OneShotCritic, radial_squash, terminal_critic_target
from .spectral_sac import (
    ACCELERATION_AXES,
    FIXED_DURATION_NORMALIZED,
    TEMPORAL_KNOTS,
    orthonormal_idct_matrix,
    spectral_to_temporal,
)


ACTIVE_FREQUENCIES = 4
LOCAL_ACTION_DIM = ACTIVE_FREQUENCIES * ACCELERATION_AXES


def local_frequency_base(
    *, dtype: torch.dtype = torch.float32, device: torch.device | str = "cpu"
) -> torch.Tensor:
    frequency = torch.arange(ACTIVE_FREQUENCIES, dtype=dtype, device=device)
    return 1.0 / torch.sqrt(1.0 + (frequency / 4.0).square())


def _radial_jacobian(raw_temporal: torch.Tensor) -> torch.Tensor:
    radius = torch.linalg.vector_norm(raw_temporal, dim=-1)
    tiny = torch.finfo(raw_temporal.dtype).tiny
    small = radius < 1.0e-6
    r2 = radius.square()
    tangential_series = 1.0 - r2 / 3.0 + 2.0 * r2.square() / 15.0
    tangential = torch.where(
        small,
        tangential_series,
        torch.tanh(radius) / radius.clamp_min(tiny),
    )
    radial = 1.0 / torch.cosh(radius).square()
    unit = raw_temporal / radius.clamp_min(tiny)[..., None]
    unit = torch.where(small[..., None], torch.zeros_like(unit), unit)
    identity = torch.eye(3, dtype=raw_temporal.dtype, device=raw_temporal.device)
    return (
        tangential[..., None, None] * identity
        + (radial - tangential)[..., None, None]
        * unit[..., :, None]
        * unit[..., None, :]
    )


def local_radial_log_volume(raw_temporal: torch.Tensor) -> torch.Tensor:
    """Log 12-D volume Jacobian of the local spectral action manifold."""

    if raw_temporal.ndim != 3 or raw_temporal.shape[1:] != (16, 3):
        raise ValueError("Raw temporal knots must have shape Bx16x3.")
    batch = raw_temporal.shape[0]
    basis = orthonormal_idct_matrix(
        dtype=raw_temporal.dtype, device=raw_temporal.device
    )[:, :ACTIVE_FREQUENCIES]
    derivative = torch.zeros(
        (TEMPORAL_KNOTS, 3, LOCAL_ACTION_DIM),
        dtype=raw_temporal.dtype,
        device=raw_temporal.device,
    )
    for axis in range(3):
        derivative[:, axis, axis * ACTIVE_FREQUENCIES : (axis + 1) * ACTIVE_FREQUENCIES] = basis
    radial_jacobian = _radial_jacobian(raw_temporal)
    temporal_jacobian = torch.einsum(
        "btij,tjk->btik", radial_jacobian, derivative
    )
    gram = torch.einsum("btik,btil->bkl", temporal_jacobian, temporal_jacobian)
    jitter = 1.0e-8 * torch.eye(
        LOCAL_ACTION_DIM, dtype=gram.dtype, device=gram.device
    )
    sign, logdet = torch.linalg.slogdet(gram + jitter)
    if not bool((sign > 0).all()):
        raise RuntimeError("Local spectral action Jacobian lost full column rank.")
    return 0.5 * logdet.reshape(batch, 1)


@dataclass(frozen=True, slots=True)
class LocalSpectralPolicySample:
    normalized_action: torch.Tensor
    log_prob: torch.Tensor
    deterministic_mean_action: torch.Tensor
    active_delta: torch.Tensor
    active_mean: torch.Tensor
    active_std: torch.Tensor
    spectral_coefficients: torch.Tensor


class LocalSpectralActor(nn.Module):
    """A 12-D stochastic offset around the exact 5B.4 Level-1 center."""

    def __init__(
        self,
        center_coefficients: torch.Tensor,
        *,
        initial_scale: float = 0.05,
        minimum_scale: float = 0.005,
        maximum_scale: float = 0.10,
    ) -> None:
        super().__init__()
        center = torch.as_tensor(center_coefficients, dtype=torch.float32)
        if center.shape != (3, 16) or not bool(torch.isfinite(center).all()):
            raise ValueError("Local spectral center must be finite with shape 3x16.")
        if not (0.0 < minimum_scale < initial_scale < maximum_scale):
            raise ValueError("Local spectral scale bounds/initialization are invalid.")
        self.register_buffer("center_coefficients", center.clone())
        self.minimum_scale = float(minimum_scale)
        self.maximum_scale = float(maximum_scale)
        self.trunk = nn.Sequential(
            nn.Linear(POLICY_CONTEXT_DIM, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
        )
        self.mean_head = nn.Linear(256, LOCAL_ACTION_DIM)
        self.std_head = nn.Linear(256, LOCAL_ACTION_DIM)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.std_head.weight)
        ratio = (initial_scale - minimum_scale) / (maximum_scale - minimum_scale)
        raw_initial = math.log(ratio / (1.0 - ratio))
        nn.init.constant_(self.std_head.bias, raw_initial)

    def statistics(
        self, normalized_context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = torch.as_tensor(normalized_context)
        if context.ndim != 2 or context.shape[-1] != POLICY_CONTEXT_DIM:
            raise ValueError(f"Actor context must have shape Bx{POLICY_CONTEXT_DIM}.")
        hidden = self.trunk(context)
        mean = self.mean_head(hidden)
        scale = self.minimum_scale + (
            self.maximum_scale - self.minimum_scale
        ) * torch.sigmoid(self.std_head(hidden))
        base = local_frequency_base(dtype=scale.dtype, device=scale.device).repeat(3)
        return mean, scale * base[None]

    def _decode(
        self, active_delta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coefficients = self.center_coefficients[None].expand(active_delta.shape[0], -1, -1).clone()
        coefficients[:, :, :ACTIVE_FREQUENCIES] += active_delta.reshape(
            -1, 3, ACTIVE_FREQUENCIES
        )
        temporal = spectral_to_temporal(coefficients)
        acceleration, _ = radial_squash(temporal)
        duration = torch.full(
            (acceleration.shape[0], 1),
            FIXED_DURATION_NORMALIZED,
            dtype=acceleration.dtype,
            device=acceleration.device,
        )
        action = torch.cat((acceleration.reshape(-1, 48), duration), dim=-1)
        return action, local_radial_log_volume(temporal), coefficients

    def forward(
        self,
        normalized_context: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> LocalSpectralPolicySample:
        mean, std = self.statistics(normalized_context)
        delta = mean if deterministic else mean + std * torch.randn_like(mean)
        action, log_volume, coefficients = self._decode(delta)
        deterministic_action, _, _ = self._decode(mean)
        gaussian_log_prob = -0.5 * (
            ((delta - mean) / std).square()
            + 2.0 * torch.log(std)
            + math.log(2.0 * math.pi)
        ).sum(dim=-1, keepdim=True)
        return LocalSpectralPolicySample(
            action,
            gaussian_log_prob - log_volume,
            deterministic_action,
            delta,
            mean,
            std,
            coefficients,
        )


@dataclass(frozen=True, slots=True)
class ConstrainedReplayBatch:
    context: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    safety_cost: torch.Tensor


class ConstrainedReplayBuffer:
    """Compact terminal replay including task, safety, and audit fields."""

    def __init__(self, capacity: int = 1_000_000) -> None:
        self.capacity = int(capacity)
        self.context = torch.empty((capacity, POLICY_CONTEXT_DIM), dtype=torch.float32)
        self.action = torch.empty((capacity, POLICY_ACTION_DIM), dtype=torch.float32)
        self.reward = torch.empty((capacity, 1), dtype=torch.float32)
        self.safety_cost = torch.empty((capacity, 1), dtype=torch.float32)
        self.failure_time_s = torch.empty((capacity, 1), dtype=torch.float32)
        self.failure_gate_mask = torch.empty((capacity, 1), dtype=torch.int16)
        self.progress = torch.empty((capacity, 1), dtype=torch.float32)
        self.tip_distance_m = torch.empty((capacity, 1), dtype=torch.float32)
        self.directed_speed_m_s = torch.empty((capacity, 1), dtype=torch.float32)
        self.direction_error_deg = torch.empty((capacity, 1), dtype=torch.float32)
        self.tip_first = torch.empty((capacity, 1), dtype=torch.bool)
        self.success = torch.empty((capacity, 1), dtype=torch.bool)
        self._size = 0
        self._position = 0
        self.total_inserted = 0

    def __len__(self) -> int:
        return self._size

    def add(self, **values: torch.Tensor) -> None:
        rows = int(values["context"].shape[0])
        if rows < 1 or rows > self.capacity:
            raise ValueError("Invalid constrained replay insertion size.")
        indices = (torch.arange(rows) + self._position) % self.capacity
        for name in (
            "context", "action", "reward", "safety_cost", "failure_time_s",
            "failure_gate_mask", "progress", "tip_distance_m",
            "directed_speed_m_s", "direction_error_deg", "tip_first", "success",
        ):
            destination = getattr(self, name)
            source = torch.as_tensor(values[name]).detach().to("cpu")
            source = source.to(destination.dtype).reshape(rows, *destination.shape[1:])
            destination[indices] = source
        self._position = int((self._position + rows) % self.capacity)
        self._size = min(self.capacity, self._size + rows)
        self.total_inserted += rows

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> ConstrainedReplayBatch:
        if self._size < batch_size:
            raise ValueError("Replay does not contain one requested minibatch.")
        indices = torch.randint(self._size, (batch_size,), generator=generator)
        selected = torch.device(device)
        return ConstrainedReplayBatch(
            self.context[indices].to(selected),
            self.action[indices].to(selected),
            self.reward[indices].to(selected),
            self.safety_cost[indices].to(selected),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": "constrained_terminal_replay_v1",
            "capacity": self.capacity,
            "size": self._size,
            "position": self._position,
            "total_inserted": self.total_inserted,
            "trajectory_storage": False,
            "exact_key_actions_persisted_separately": True,
        }


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


@dataclass(slots=True)
class ConstrainedSacAgent:
    actor: LocalSpectralActor
    reward_critic1: OneShotCritic
    reward_critic2: OneShotCritic
    safety_critic1: OneShotCritic
    safety_critic2: OneShotCritic
    log_alpha: torch.Tensor
    raw_lambda: torch.Tensor
    actor_optimizer: torch.optim.Optimizer
    reward_critic1_optimizer: torch.optim.Optimizer
    reward_critic2_optimizer: torch.optim.Optimizer
    safety_critic1_optimizer: torch.optim.Optimizer
    safety_critic2_optimizer: torch.optim.Optimizer
    alpha_optimizer: torch.optim.Optimizer
    lambda_optimizer: torch.optim.Optimizer
    target_entropy: float
    cost_target: float = 0.20
    gradient_clip_norm: float = 10.0

    @classmethod
    def create(
        cls,
        center_coefficients: torch.Tensor,
        *,
        device: torch.device | str,
        target_entropy: float,
        actor_lr: float = 3.0e-4,
        reward_critic_lr: float = 3.0e-4,
        safety_critic_lr: float = 3.0e-4,
        alpha_lr: float = 3.0e-4,
        lambda_lr: float = 3.0e-4,
    ) -> "ConstrainedSacAgent":
        selected = torch.device(device)
        actor = LocalSpectralActor(center_coefficients).to(selected)
        reward1, reward2 = OneShotCritic().to(selected), OneShotCritic().to(selected)
        safety1, safety2 = OneShotCritic().to(selected), OneShotCritic().to(selected)
        log_alpha = torch.zeros((), device=selected, requires_grad=True)
        raw_lambda = torch.tensor(
            _inverse_softplus(1.0), device=selected, requires_grad=True
        )
        return cls(
            actor,
            reward1,
            reward2,
            safety1,
            safety2,
            log_alpha,
            raw_lambda,
            torch.optim.Adam(actor.parameters(), lr=actor_lr),
            torch.optim.Adam(reward1.parameters(), lr=reward_critic_lr),
            torch.optim.Adam(reward2.parameters(), lr=reward_critic_lr),
            torch.optim.Adam(safety1.parameters(), lr=safety_critic_lr),
            torch.optim.Adam(safety2.parameters(), lr=safety_critic_lr),
            torch.optim.Adam([log_alpha], lr=alpha_lr),
            torch.optim.Adam([raw_lambda], lr=lambda_lr),
            float(target_entropy),
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def lambda_safe(self) -> torch.Tensor:
        return F.softplus(self.raw_lambda).clamp(max=100.0)

    def update(self, batch: ConstrainedReplayBatch) -> dict[str, float]:
        reward_target = terminal_critic_target(batch.reward)
        safety_target = batch.safety_cost.reshape(-1, 1)
        q1 = self.reward_critic1(batch.context, batch.action)
        q2 = self.reward_critic2(batch.context, batch.action)
        q1_loss = F.smooth_l1_loss(q1, reward_target, beta=1.0)
        q2_loss = F.smooth_l1_loss(q2, reward_target, beta=1.0)
        self.reward_critic1_optimizer.zero_grad(set_to_none=True)
        q1_loss.backward()
        q1_grad = float(nn.utils.clip_grad_norm_(self.reward_critic1.parameters(), self.gradient_clip_norm))
        self.reward_critic1_optimizer.step()
        self.reward_critic2_optimizer.zero_grad(set_to_none=True)
        q2_loss.backward()
        q2_grad = float(nn.utils.clip_grad_norm_(self.reward_critic2.parameters(), self.gradient_clip_norm))
        self.reward_critic2_optimizer.step()

        c1 = self.safety_critic1(batch.context, batch.action)
        c2 = self.safety_critic2(batch.context, batch.action)
        c1_loss = F.binary_cross_entropy_with_logits(c1, safety_target)
        c2_loss = F.binary_cross_entropy_with_logits(c2, safety_target)
        self.safety_critic1_optimizer.zero_grad(set_to_none=True)
        c1_loss.backward()
        c1_grad = float(nn.utils.clip_grad_norm_(self.safety_critic1.parameters(), self.gradient_clip_norm))
        self.safety_critic1_optimizer.step()
        self.safety_critic2_optimizer.zero_grad(set_to_none=True)
        c2_loss.backward()
        c2_grad = float(nn.utils.clip_grad_norm_(self.safety_critic2.parameters(), self.gradient_clip_norm))
        self.safety_critic2_optimizer.step()

        sample = self.actor(batch.context)
        critics = (
            self.reward_critic1,
            self.reward_critic2,
            self.safety_critic1,
            self.safety_critic2,
        )
        for critic in critics:
            for parameter in critic.parameters():
                parameter.requires_grad_(False)
        reward_q = torch.minimum(
            self.reward_critic1(batch.context, sample.normalized_action),
            self.reward_critic2(batch.context, sample.normalized_action),
        )
        p_fail = torch.maximum(
            torch.sigmoid(self.safety_critic1(batch.context, sample.normalized_action)),
            torch.sigmoid(self.safety_critic2(batch.context, sample.normalized_action)),
        )
        entropy_term = (self.alpha.detach() * sample.log_prob).mean()
        reward_term = -reward_q.mean()
        safety_term = self.lambda_safe.detach() * p_fail.mean()
        actor_loss = entropy_term + reward_term + safety_term
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad = float(nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip_norm))
        self.actor_optimizer.step()
        for critic in critics:
            for parameter in critic.parameters():
                parameter.requires_grad_(True)

        alpha_loss = -(
            self.log_alpha * (sample.log_prob.detach() + self.target_entropy)
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        lambda_loss = -self.lambda_safe * (
            p_fail.detach().mean() - self.cost_target
        )
        self.lambda_optimizer.zero_grad(set_to_none=True)
        lambda_loss.backward()
        self.lambda_optimizer.step()
        with torch.no_grad():
            self.raw_lambda.clamp_(max=_inverse_softplus(100.0))

        return {
            "reward_critic1_loss": float(q1_loss.detach()),
            "reward_critic2_loss": float(q2_loss.detach()),
            "safety_critic1_bce": float(c1_loss.detach()),
            "safety_critic2_bce": float(c2_loss.detach()),
            "reward_critic1_gradient_norm": q1_grad,
            "reward_critic2_gradient_norm": q2_grad,
            "safety_critic1_gradient_norm": c1_grad,
            "safety_critic2_gradient_norm": c2_grad,
            "actor_gradient_norm": actor_grad,
            "actor_loss": float(actor_loss.detach()),
            "reward_actor_term": float(reward_term.detach()),
            "safety_actor_term": float(safety_term.detach()),
            "entropy_actor_term": float(entropy_term.detach()),
            "alpha": float(self.alpha.detach()),
            "alpha_loss": float(alpha_loss.detach()),
            "lambda_safe": float(self.lambda_safe.detach()),
            "lambda_loss": float(lambda_loss.detach()),
            "mean_predicted_p_fail": float(p_fail.detach().mean()),
            "policy_entropy": float((-sample.log_prob.detach()).mean()),
        }

    def checkpoint(self) -> dict[str, Any]:
        return {
            "schema": "constrained_local_sac_checkpoint_v1",
            "actor": self.actor.state_dict(),
            "reward_critic1": self.reward_critic1.state_dict(),
            "reward_critic2": self.reward_critic2.state_dict(),
            "safety_critic1": self.safety_critic1.state_dict(),
            "safety_critic2": self.safety_critic2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "raw_lambda": self.raw_lambda.detach().cpu(),
            "target_entropy": self.target_entropy,
            "cost_target": self.cost_target,
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "reward_critic1_optimizer": self.reward_critic1_optimizer.state_dict(),
            "reward_critic2_optimizer": self.reward_critic2_optimizer.state_dict(),
            "safety_critic1_optimizer": self.safety_critic1_optimizer.state_dict(),
            "safety_critic2_optimizer": self.safety_critic2_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "lambda_optimizer": self.lambda_optimizer.state_dict(),
        }

    def load_checkpoint(self, payload: dict[str, Any]) -> None:
        if payload.get("schema") != "constrained_local_sac_checkpoint_v1":
            raise ValueError("Unsupported constrained local SAC checkpoint.")
        self.actor.load_state_dict(payload["actor"])
        self.reward_critic1.load_state_dict(payload["reward_critic1"])
        self.reward_critic2.load_state_dict(payload["reward_critic2"])
        self.safety_critic1.load_state_dict(payload["safety_critic1"])
        self.safety_critic2.load_state_dict(payload["safety_critic2"])
        with torch.no_grad():
            self.log_alpha.copy_(payload["log_alpha"].to(self.log_alpha.device))
            self.raw_lambda.copy_(payload["raw_lambda"].to(self.raw_lambda.device))
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.reward_critic1_optimizer.load_state_dict(payload["reward_critic1_optimizer"])
        self.reward_critic2_optimizer.load_state_dict(payload["reward_critic2_optimizer"])
        self.safety_critic1_optimizer.load_state_dict(payload["safety_critic1_optimizer"])
        self.safety_critic2_optimizer.load_state_dict(payload["safety_critic2_optimizer"])
        self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        self.lambda_optimizer.load_state_dict(payload["lambda_optimizer"])


def save_constrained_checkpoint(
    path: str | Path,
    agent: ConstrainedSacAgent,
    *,
    episodes: int,
    updates: int,
    replay_metadata: dict[str, Any],
) -> None:
    torch.save(
        {
            **agent.checkpoint(),
            "episodes": int(episodes),
            "updates": int(updates),
            "replay_metadata": replay_metadata,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        Path(path),
    )

