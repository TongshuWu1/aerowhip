"""Full-rank temporal-spectral policy distribution for one-shot SAC.

The Gaussian lives in 48 orthonormal DCT coordinates (16 per acceleration
axis).  An orthonormal inverse DCT maps coefficients to raw temporal knots,
after which the established three-axis radial squash enforces the open L2
ball.  Duration is deterministic and remains the 49th external action value.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .policy_action import POLICY_ACTION_DIM
from .policy_context import POLICY_CONTEXT_DIM
from .sac import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    OneShotCritic,
    TerminalSacAgent,
    radial_squash,
)


TEMPORAL_KNOTS = 16
ACCELERATION_AXES = 3
SPECTRAL_ACTION_DIM = TEMPORAL_KNOTS * ACCELERATION_AXES
FIXED_DURATION_NORMALIZED = 1.0


def orthonormal_idct_matrix(
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return B where temporal values are ``B @ DCT_coefficients``."""

    time_index = torch.arange(TEMPORAL_KNOTS, dtype=dtype, device=device)[:, None]
    frequency = torch.arange(TEMPORAL_KNOTS, dtype=dtype, device=device)[None, :]
    basis = torch.cos(
        math.pi
        / TEMPORAL_KNOTS
        * (time_index + 0.5)
        * frequency
    )
    scales = torch.full(
        (TEMPORAL_KNOTS,),
        math.sqrt(2.0 / TEMPORAL_KNOTS),
        dtype=dtype,
        device=device,
    )
    scales[0] = math.sqrt(1.0 / TEMPORAL_KNOTS)
    return basis * scales[None, :]


def initial_frequency_std(
    *,
    std_low: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    frequency = torch.arange(TEMPORAL_KNOTS, dtype=dtype, device=device)
    return float(std_low) / torch.sqrt(1.0 + (frequency / 4.0).square())


def spectral_to_temporal(coefficients: torch.Tensor) -> torch.Tensor:
    """Map Bx3x16 axis-major DCT coefficients to Bx16x3 knots."""

    values = torch.as_tensor(coefficients)
    if values.ndim != 3 or values.shape[1:] != (ACCELERATION_AXES, TEMPORAL_KNOTS):
        raise ValueError("Spectral coefficients must have shape Bx3x16.")
    basis = orthonormal_idct_matrix(dtype=values.dtype, device=values.device)
    axis_temporal = torch.einsum("tf,baf->bat", basis, values)
    return axis_temporal.transpose(1, 2).contiguous()


def temporal_to_spectral(raw_temporal: torch.Tensor) -> torch.Tensor:
    """Map Bx16x3 temporal knots to Bx3x16 orthonormal DCT coefficients."""

    values = torch.as_tensor(raw_temporal)
    if values.ndim != 3 or values.shape[1:] != (TEMPORAL_KNOTS, ACCELERATION_AXES):
        raise ValueError("Temporal knots must have shape Bx16x3.")
    basis = orthonormal_idct_matrix(dtype=values.dtype, device=values.device)
    return torch.einsum("tf,bta->baf", basis, values)


def inverse_radial_squash(normalized_groups: torch.Tensor) -> torch.Tensor:
    """Invert ``y=tanh(||z||) z/||z||`` for points inside the unit ball."""

    values = torch.as_tensor(normalized_groups)
    if values.shape[-1] != ACCELERATION_AXES:
        raise ValueError("Inverse radial squash requires final dimension 3.")
    radius = torch.linalg.vector_norm(values, dim=-1)
    if bool((radius >= 1.0).any()):
        raise ValueError("Inverse radial squash requires open-unit-ball inputs.")
    small = radius < 1.0e-6
    r2 = radius.square()
    # atanh(r)/r = 1 + r^2/3 + r^4/5 + O(r^6).
    series = 1.0 + r2 / 3.0 + r2.square() / 5.0
    regular = torch.atanh(radius) / radius.clamp_min(
        torch.finfo(values.dtype).tiny
    )
    scale = torch.where(small, series, regular)
    return values * scale[..., None]


def spectral_coefficients_to_normalized_action(
    coefficients: torch.Tensor,
    *,
    duration_normalized: float = FIXED_DURATION_NORMALIZED,
) -> torch.Tensor:
    """Decode Bx3x16 spectral coefficients into the external 49-D action."""

    temporal = spectral_to_temporal(coefficients)
    acceleration, _ = radial_squash(temporal)
    duration = torch.full(
        (acceleration.shape[0], 1),
        float(duration_normalized),
        dtype=acceleration.dtype,
        device=acceleration.device,
    )
    return torch.cat((acceleration.reshape(-1, 48), duration), dim=-1)


def alpha_explore_schedule(collected_episodes: int) -> float:
    """Actor-only entropy curriculum specified for Milestone 5B.4."""

    episodes = max(0, int(collected_episodes))
    if episodes < 75_000:
        return 0.25
    if episodes < 300_000:
        return 0.25 * (1.0 - (episodes - 75_000) / 225_000.0)
    return 0.0


@dataclass(frozen=True, slots=True)
class SpectralPolicySample:
    normalized_action: torch.Tensor
    log_prob: torch.Tensor
    deterministic_mean_action: torch.Tensor
    raw_mean: torch.Tensor
    log_std: torch.Tensor
    spectral_coefficients: torch.Tensor


class SpectralOneShotActor(nn.Module):
    """83-D context to a 48-D full-rank spectral acceleration policy."""

    def __init__(self, *, std_low: float = 1.0) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(POLICY_CONTEXT_DIM, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
        )
        self.mean_head = nn.Linear(256, SPECTRAL_ACTION_DIM)
        self.log_std_head = nn.Linear(256, SPECTRAL_ACTION_DIM)
        nn.init.uniform_(self.mean_head.weight, -1.0e-3, 1.0e-3)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        spectrum = initial_frequency_std(std_low=std_low)
        nn.init.constant_(self.log_std_head.bias, 0.0)
        with torch.no_grad():
            self.log_std_head.bias.copy_(spectrum.log().repeat(ACCELERATION_AXES))

    def statistics(self, normalized_context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = torch.as_tensor(normalized_context)
        if context.ndim != 2 or context.shape[-1] != POLICY_CONTEXT_DIM:
            raise ValueError(f"Actor context must have shape Bx{POLICY_CONTEXT_DIM}.")
        hidden = self.trunk(context)
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    @staticmethod
    def _decode_spectral(coefficients: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        temporal = spectral_to_temporal(
            coefficients.reshape(-1, ACCELERATION_AXES, TEMPORAL_KNOTS)
        )
        acceleration, radial_log_det = radial_squash(temporal)
        duration = torch.full(
            (acceleration.shape[0], 1),
            FIXED_DURATION_NORMALIZED,
            dtype=acceleration.dtype,
            device=acceleration.device,
        )
        action = torch.cat((acceleration.reshape(-1, 48), duration), dim=-1)
        return action, radial_log_det.sum(dim=-1, keepdim=True)

    def forward(
        self,
        normalized_context: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> SpectralPolicySample:
        mean, log_std = self.statistics(normalized_context)
        std = torch.exp(log_std)
        coefficients = mean if deterministic else mean + std * torch.randn_like(mean)
        action, radial_log_det = self._decode_spectral(coefficients)
        deterministic_action, _ = self._decode_spectral(mean)
        gaussian_log_prob = -0.5 * (
            ((coefficients - mean) / std).square()
            + 2.0 * log_std
            + math.log(2.0 * math.pi)
        ).sum(dim=-1, keepdim=True)
        # The orthonormal IDCT has |det|=1; duration is deterministic.
        log_prob = gaussian_log_prob - radial_log_det
        return SpectralPolicySample(
            action,
            log_prob,
            deterministic_action,
            mean,
            log_std,
            coefficients,
        )


def create_spectral_agent(
    *,
    device: torch.device | str,
    actor_lr: float = 3.0e-4,
    critic_lr: float = 3.0e-4,
    alpha_lr: float = 3.0e-4,
    target_entropy: float = -48.0,
    gradient_clip_norm: float = 10.0,
    std_low: float = 1.0,
) -> TerminalSacAgent:
    selected = torch.device(device)
    actor = SpectralOneShotActor(std_low=std_low).to(selected)
    critic1 = OneShotCritic().to(selected)
    critic2 = OneShotCritic().to(selected)
    log_alpha = torch.zeros((), dtype=torch.float32, device=selected, requires_grad=True)
    return TerminalSacAgent(
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
        critic_loss="smooth_l1",
    )


def temporal_roughness(acceleration_knots_m_s2: torch.Tensor) -> torch.Tensor:
    knots = torch.as_tensor(acceleration_knots_m_s2)
    if knots.ndim != 3 or knots.shape[1:] != (16, 3):
        raise ValueError("Acceleration knots must have shape Bx16x3.")
    return (knots[:, 1:] - knots[:, :-1]).square().sum(dim=-1).mean(dim=-1)


def dominant_horizontal_sign_changes(acceleration_knots_m_s2: torch.Tensor) -> torch.Tensor:
    knots = torch.as_tensor(acceleration_knots_m_s2)
    horizontal_energy = knots[:, :, :2].square().sum(dim=1)
    dominant = horizontal_energy.argmax(dim=1)
    selected = torch.gather(
        knots[:, :, :2], 2, dominant[:, None, None].expand(-1, 16, 1)
    ).squeeze(-1)
    sign = torch.sign(selected)
    return ((sign[:, 1:] * sign[:, :-1]) < 0).sum(dim=1)


def frequency_energy_fractions(coefficients: torch.Tensor) -> dict[str, torch.Tensor]:
    coeff = torch.as_tensor(coefficients).reshape(-1, 3, 16)
    energy = coeff.square().sum(dim=1)
    total = energy.sum(dim=1).clamp_min(torch.finfo(energy.dtype).tiny)
    return {
        "low": energy[:, 0:4].sum(dim=1) / total,
        "mid": energy[:, 4:8].sum(dim=1) / total,
        "high": energy[:, 8:16].sum(dim=1) / total,
    }
