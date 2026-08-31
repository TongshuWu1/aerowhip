"""Single-decision terminal SAC for one-shot open-loop manipulation.

The policy distribution lives in the normalized 49-D maneuver space.  Each
three-axis acceleration knot uses a smooth radial map into the open unit L2
ball; duration uses the usual scalar tanh map.  The environment remains a
black box: critic targets are terminal rewards and never bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .policy_action import POLICY_ACTION_DIM
from .policy_context import POLICY_CONTEXT_DIM


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
ACCELERATION_GROUPS = 16
ACCELERATION_GROUP_DIM = 3


def terminal_critic_target(terminal_rewards: torch.Tensor) -> torch.Tensor:
    """Return the stationary reward-only target for a terminal decision."""

    return torch.as_tensor(terminal_rewards).reshape(-1, 1).detach()


def _log_tanh_over_radius(radius: torch.Tensor) -> torch.Tensor:
    """Stable log(tanh(r)/r), including its analytic r=0 limit."""

    small = radius < 1.0e-3
    r2 = radius.square()
    series = -r2 / 3.0 + 7.0 * r2.square() / 90.0
    safe_radius = radius.clamp_min(torch.finfo(radius.dtype).tiny)
    exp_term = torch.exp(-2.0 * safe_radius)
    regular = torch.log(-torch.expm1(-2.0 * safe_radius)) - torch.log1p(
        exp_term
    ) - torch.log(safe_radius)
    return torch.where(small, series, regular)


def _log_sech_squared(value: torch.Tensor) -> torch.Tensor:
    """Stable log(sech(value)^2)."""

    absolute = torch.abs(value)
    return 2.0 * (math.log(2.0) - absolute - F.softplus(-2.0 * absolute))


def radial_squash(raw_groups: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Map (...,3) Gaussian vectors into the open unit L2 ball.

    Returns the normalized vectors and log absolute Jacobian determinant for
    every three-dimensional group.
    """

    values = torch.as_tensor(raw_groups)
    if values.shape[-1] != ACCELERATION_GROUP_DIM:
        raise ValueError("Radial squash requires vectors with final dimension 3.")
    radius = torch.linalg.vector_norm(values, dim=-1)
    small = radius < 1.0e-6
    r2 = radius.square()
    # tanh(r)/r = 1 - r^2/3 + 2r^4/15 + O(r^6).
    series_scale = 1.0 - r2 / 3.0 + 2.0 * r2.square() / 15.0
    regular_scale = torch.tanh(radius) / radius.clamp_min(
        torch.finfo(values.dtype).tiny
    )
    scale = torch.where(small, series_scale, regular_scale)
    squashed = values * scale[..., None]
    log_det = 2.0 * _log_tanh_over_radius(radius) + _log_sech_squared(radius)
    return squashed, log_det


def squash_raw_action(raw_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the 16 radial transforms and scalar duration tanh transform."""

    raw = torch.as_tensor(raw_action)
    if raw.ndim != 2 or raw.shape[-1] != POLICY_ACTION_DIM:
        raise ValueError(f"Raw SAC action must have shape Bx{POLICY_ACTION_DIM}.")
    groups = raw[:, :48].reshape(-1, ACCELERATION_GROUPS, ACCELERATION_GROUP_DIM)
    acceleration, group_log_det = radial_squash(groups)
    duration_raw = raw[:, 48]
    duration = torch.tanh(duration_raw)
    duration_log_det = _log_sech_squared(duration_raw)
    action = torch.cat((acceleration.reshape(-1, 48), duration[:, None]), dim=-1)
    total_log_det = group_log_det.sum(dim=-1) + duration_log_det
    return action, total_log_det[:, None]


@dataclass(frozen=True, slots=True)
class PolicySample:
    normalized_action: torch.Tensor
    log_prob: torch.Tensor
    deterministic_mean_action: torch.Tensor
    raw_mean: torch.Tensor
    log_std: torch.Tensor


def _mlp(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.SiLU(),
        nn.Linear(256, 256),
        nn.SiLU(),
        nn.Linear(256, 256),
        nn.SiLU(),
        nn.Linear(256, output_dim),
    )


class OneShotActor(nn.Module):
    """83-D context to a radial-squashed 49-D stochastic maneuver."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Sequential(*list(_mlp(POLICY_CONTEXT_DIM, 256).children())[:-1])
        self.mean_head = nn.Linear(256, POLICY_ACTION_DIM)
        self.log_std_head = nn.Linear(256, POLICY_ACTION_DIM)
        nn.init.uniform_(self.mean_head.weight, -1.0e-3, 1.0e-3)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.uniform_(self.log_std_head.weight, -1.0e-3, 1.0e-3)
        nn.init.constant_(self.log_std_head.bias, -0.5)

    def statistics(self, normalized_context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = torch.as_tensor(normalized_context)
        if context.ndim != 2 or context.shape[-1] != POLICY_CONTEXT_DIM:
            raise ValueError(f"Actor context must have shape Bx{POLICY_CONTEXT_DIM}.")
        hidden = self.trunk(context)
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def forward(
        self,
        normalized_context: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> PolicySample:
        mean, log_std = self.statistics(normalized_context)
        std = torch.exp(log_std)
        raw = mean if deterministic else mean + std * torch.randn_like(mean)
        action, log_det = squash_raw_action(raw)
        deterministic_action, _ = squash_raw_action(mean)
        base_log_prob = -0.5 * (
            ((raw - mean) / std).square()
            + 2.0 * log_std
            + math.log(2.0 * math.pi)
        ).sum(dim=-1, keepdim=True)
        log_prob = base_log_prob - log_det
        return PolicySample(action, log_prob, deterministic_action, mean, log_std)


class OneShotCritic(nn.Module):
    """Terminal state-action value; there is no future-state bootstrap."""

    def __init__(self) -> None:
        super().__init__()
        self.network = _mlp(POLICY_CONTEXT_DIM + POLICY_ACTION_DIM, 1)

    def forward(self, normalized_context: torch.Tensor, normalized_action: torch.Tensor) -> torch.Tensor:
        context = torch.as_tensor(normalized_context)
        action = torch.as_tensor(normalized_action, dtype=context.dtype, device=context.device)
        if context.ndim != 2 or context.shape[-1] != POLICY_CONTEXT_DIM:
            raise ValueError("Critic context has the wrong shape.")
        if action.shape != (context.shape[0], POLICY_ACTION_DIM):
            raise ValueError("Critic action has the wrong shape.")
        return self.network(torch.cat((context, action), dim=-1))


@dataclass(slots=True)
class TerminalSacAgent:
    actor: nn.Module
    critic1: OneShotCritic
    critic2: OneShotCritic
    log_alpha: torch.Tensor
    actor_optimizer: torch.optim.Optimizer
    critic1_optimizer: torch.optim.Optimizer
    critic2_optimizer: torch.optim.Optimizer
    alpha_optimizer: torch.optim.Optimizer
    target_entropy: float
    gradient_clip_norm: float
    critic_loss: str

    @classmethod
    def create(
        cls,
        *,
        device: torch.device | str,
        actor_lr: float = 3.0e-4,
        critic_lr: float = 3.0e-4,
        alpha_lr: float = 3.0e-4,
        target_entropy: float = -49.0,
        gradient_clip_norm: float = 10.0,
        critic_loss: str = "mse",
    ) -> "TerminalSacAgent":
        if critic_loss not in {"mse", "smooth_l1"}:
            raise ValueError("Critic loss must be 'mse' or 'smooth_l1'.")
        selected = torch.device(device)
        actor = OneShotActor().to(selected)
        critic1 = OneShotCritic().to(selected)
        critic2 = OneShotCritic().to(selected)
        log_alpha = torch.zeros((), dtype=torch.float32, device=selected, requires_grad=True)
        return cls(
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            log_alpha=log_alpha,
            actor_optimizer=torch.optim.Adam(actor.parameters(), lr=actor_lr),
            critic1_optimizer=torch.optim.Adam(critic1.parameters(), lr=critic_lr),
            critic2_optimizer=torch.optim.Adam(critic2.parameters(), lr=critic_lr),
            alpha_optimizer=torch.optim.Adam([log_alpha], lr=alpha_lr),
            target_entropy=float(target_entropy),
            gradient_clip_norm=float(gradient_clip_norm),
            critic_loss=critic_loss,
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def update(
        self,
        contexts: torch.Tensor,
        actions: torch.Tensor,
        terminal_rewards: torch.Tensor,
        *,
        actor_entropy_bonus: float = 0.0,
    ) -> dict[str, float]:
        """One terminal SAC cycle; targets are rewards with no bootstrap term."""

        reward = torch.as_tensor(terminal_rewards, device=contexts.device).reshape(-1, 1)
        if reward.shape[0] != contexts.shape[0]:
            raise ValueError("Terminal reward batch size mismatch.")
        if not math.isfinite(actor_entropy_bonus) or actor_entropy_bonus < 0.0:
            raise ValueError("Actor entropy bonus must be finite and nonnegative.")
        target = terminal_critic_target(reward)

        q1 = self.critic1(contexts, actions)
        q2 = self.critic2(contexts, actions)
        loss_function = F.mse_loss if self.critic_loss == "mse" else F.smooth_l1_loss
        q1_loss = loss_function(q1, target)
        q2_loss = loss_function(q2, target)
        self.critic1_optimizer.zero_grad(set_to_none=True)
        q1_loss.backward()
        q1_grad = float(nn.utils.clip_grad_norm_(self.critic1.parameters(), self.gradient_clip_norm))
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad(set_to_none=True)
        q2_loss.backward()
        q2_grad = float(nn.utils.clip_grad_norm_(self.critic2.parameters(), self.gradient_clip_norm))
        self.critic2_optimizer.step()

        sample = self.actor(contexts)
        for parameter in self.critic1.parameters():
            parameter.requires_grad_(False)
        for parameter in self.critic2.parameters():
            parameter.requires_grad_(False)
        policy_q = torch.minimum(
            self.critic1(contexts, sample.normalized_action),
            self.critic2(contexts, sample.normalized_action),
        )
        effective_alpha = self.alpha.detach() + float(actor_entropy_bonus)
        actor_loss = (effective_alpha * sample.log_prob - policy_q).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad = float(nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip_norm))
        self.actor_optimizer.step()
        for parameter in self.critic1.parameters():
            parameter.requires_grad_(True)
        for parameter in self.critic2.parameters():
            parameter.requires_grad_(True)

        alpha_loss = -(
            self.log_alpha * (sample.log_prob.detach() + self.target_entropy)
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return {
            "q1_loss": float(q1_loss.detach()),
            "q2_loss": float(q2_loss.detach()),
            "q1_mean": float(q1.detach().mean()),
            "q2_mean": float(q2.detach().mean()),
            "actor_loss": float(actor_loss.detach()),
            "alpha_loss": float(alpha_loss.detach()),
            "alpha": float(self.alpha.detach()),
            "alpha_explore": float(actor_entropy_bonus),
            "alpha_effective": float(self.alpha.detach()) + float(actor_entropy_bonus),
            "entropy": float((-sample.log_prob.detach()).mean()),
            "actor_gradient_norm": actor_grad,
            "critic1_gradient_norm": q1_grad,
            "critic2_gradient_norm": q2_grad,
            "action_mean": float(sample.normalized_action.detach().mean()),
            "action_std": float(sample.normalized_action.detach().std()),
            "duration_normalized_mean": float(sample.normalized_action[:, -1].detach().mean()),
            "acceleration_normalized_norm_mean": float(
                torch.linalg.vector_norm(
                    sample.normalized_action[:, :48].detach().reshape(-1, 16, 3), dim=-1
                ).mean()
            ),
        }

    def checkpoint(self) -> dict[str, Any]:
        return {
            "schema": "oneshot_terminal_sac_checkpoint_v1",
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic1_optimizer": self.critic1_optimizer.state_dict(),
            "critic2_optimizer": self.critic2_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "target_entropy": self.target_entropy,
            "gradient_clip_norm": self.gradient_clip_norm,
            "critic_loss": self.critic_loss,
        }

    def load_checkpoint(self, payload: dict[str, Any]) -> None:
        if payload.get("schema") != "oneshot_terminal_sac_checkpoint_v1":
            raise ValueError("Unsupported terminal SAC checkpoint.")
        self.actor.load_state_dict(payload["actor"])
        self.critic1.load_state_dict(payload["critic1"])
        self.critic2.load_state_dict(payload["critic2"])
        with torch.no_grad():
            self.log_alpha.copy_(torch.as_tensor(payload["log_alpha"], device=self.log_alpha.device))
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic1_optimizer.load_state_dict(payload["critic1_optimizer"])
        self.critic2_optimizer.load_state_dict(payload["critic2_optimizer"])
        self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        checkpoint_loss = str(payload.get("critic_loss", "mse"))
        if checkpoint_loss != self.critic_loss:
            raise ValueError(
                f"Checkpoint critic loss {checkpoint_loss!r} does not match "
                f"agent loss {self.critic_loss!r}."
            )
