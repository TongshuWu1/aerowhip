"""Bounded RL-native terminal reward for one-shot cable whipping.

This module deliberately does not define task success. Scientific success and
feasibility remain properties of the unchanged variable-duration evaluator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class RLWhipRewardConfig:
    profile: str = "rl_whip_reward_v1"
    progress_weight: float = 0.0
    sigma_position_m: float = 0.20
    directed_velocity_scale_m_s: float = 4.0
    position_weight: float = 2.0
    velocity_weight: float = 2.0
    direction_weight: float = 1.0
    success_bonus: float = 5.0
    safety_weight: float = 2.0
    normalized_safety_clip: float = 4.0
    non_tip_penalty: float = 1.0
    control_effort_weight: float = 0.05
    control_smoothness_weight: float = 0.05
    successful_time_weight_per_s: float = 0.10
    zero_speed_epsilon_m_s: float = 1.0e-6
    direction_activation_speed_m_s: float = 0.10

    def __post_init__(self) -> None:
        if self.profile not in {"rl_whip_reward_v1", "rl_whip_reward_v2"}:
            raise ValueError("Unsupported RL whip reward profile.")
        positive_values = tuple(
            value
            for name, value in asdict(self).items()
            if name not in {"profile", "progress_weight"}
        )
        if not all(value > 0.0 for value in positive_values):
            raise ValueError("RL whip reward coefficients must be positive.")
        expected_progress = 0.0 if self.profile == "rl_whip_reward_v1" else 4.0
        if self.progress_weight != expected_progress:
            raise ValueError(
                f"{self.profile} requires progress_weight={expected_progress}."
            )

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RLWhipRewardComponents:
    total: torch.Tensor
    progress: torch.Tensor
    strike: torch.Tensor
    success: torch.Tensor
    safety: torch.Tensor
    non_tip: torch.Tensor
    control: torch.Tensor
    time: torch.Tensor
    displacement_violation: torch.Tensor
    speed_violation: torch.Tensor
    acceleration_violation: torch.Tensor
    normalized_progress: torch.Tensor
    initial_tip_distance_m: torch.Tensor
    minimum_tip_distance_m: torch.Tensor


def rl_event_reward(
    distance_m: torch.Tensor,
    directed_velocity_m_s: torch.Tensor,
    tip_speed_m_s: torch.Tensor,
    direction_cosine: torch.Tensor,
    config: RLWhipRewardConfig,
) -> torch.Tensor:
    """Evaluate the smooth proximity-gated instantaneous strike score."""

    distance = torch.as_tensor(distance_m)
    directed = torch.as_tensor(
        directed_velocity_m_s, dtype=distance.dtype, device=distance.device
    )
    speed = torch.as_tensor(tip_speed_m_s, dtype=distance.dtype, device=distance.device)
    cosine = torch.as_tensor(direction_cosine, dtype=distance.dtype, device=distance.device)
    proximity = torch.exp(
        -distance.square() / (2.0 * config.sigma_position_m**2)
    )
    velocity = torch.tanh(directed / config.directed_velocity_scale_m_s)
    # Smoothly suppress direction reward at zero speed. At ordinary whip
    # speeds this converges to 0.5*(1+alignment), while avoiding a hard branch
    # and undefined direction exactly at rest.
    direction = 0.5 * (1.0 + cosine.clamp(-1.0, 1.0)) * torch.tanh(
        speed / config.direction_activation_speed_m_s
    )
    return proximity * (
        config.position_weight
        + config.velocity_weight * velocity
        + config.direction_weight * direction
    )


def compose_rl_terminal_reward(
    *,
    strike_reward: torch.Tensor,
    success: torch.Tensor,
    first_entry_marker: torch.Tensor,
    first_hit_time_s: torch.Tensor,
    maximum_uav_displacement_m: torch.Tensor,
    maximum_uav_speed_m_s: torch.Tensor,
    maximum_command_acceleration_m_s2: torch.Tensor,
    effort_cost: torch.Tensor,
    smoothness_cost: torch.Tensor,
    displacement_limit_m: float,
    speed_limit_m_s: float,
    acceleration_limit_m_s2: float,
    config: RLWhipRewardConfig,
    initial_tip_distance_m: torch.Tensor | None = None,
    minimum_tip_distance_m: torch.Tensor | None = None,
) -> RLWhipRewardComponents:
    """Compose the terminal return from bounded, auditable components."""

    strike = torch.as_tensor(strike_reward)
    dtype, device = strike.dtype, strike.device
    succeeded = torch.as_tensor(success, dtype=torch.bool, device=device)
    marker = torch.as_tensor(first_entry_marker, device=device)

    if config.progress_weight > 0.0:
        if initial_tip_distance_m is None or minimum_tip_distance_m is None:
            raise ValueError(
                "rl_whip_reward_v2 requires initial and minimum tip distances."
            )
        initial_distance = torch.as_tensor(
            initial_tip_distance_m, dtype=dtype, device=device
        )
        minimum_distance = torch.as_tensor(
            minimum_tip_distance_m, dtype=dtype, device=device
        )
        denominator = torch.clamp(
            initial_distance, min=torch.finfo(dtype).eps
        )
        normalized_progress = (
            (initial_distance - minimum_distance) / denominator
        ).clamp(0.0, 1.0)
    else:
        initial_distance = torch.zeros_like(strike)
        minimum_distance = torch.zeros_like(strike)
        normalized_progress = torch.zeros_like(strike)
    progress_component = config.progress_weight * normalized_progress

    def violation(value: torch.Tensor, limit: float) -> torch.Tensor:
        normalized = torch.relu((value - limit) / limit).square()
        return normalized.clamp(max=config.normalized_safety_clip)

    displacement_violation = violation(
        torch.as_tensor(maximum_uav_displacement_m, dtype=dtype, device=device),
        displacement_limit_m,
    )
    speed_violation = violation(
        torch.as_tensor(maximum_uav_speed_m_s, dtype=dtype, device=device),
        speed_limit_m_s,
    )
    acceleration_violation = violation(
        torch.as_tensor(maximum_command_acceleration_m_s2, dtype=dtype, device=device),
        acceleration_limit_m_s2,
    )
    success_component = succeeded.to(dtype) * config.success_bonus
    safety_component = -config.safety_weight * (
        displacement_violation + speed_violation + acceleration_violation
    )
    non_tip_component = -config.non_tip_penalty * (
        (marker > 0) & (marker < 10)
    ).to(dtype)
    control_component = -(
        config.control_effort_weight
        * torch.as_tensor(effort_cost, dtype=dtype, device=device)
        + config.control_smoothness_weight
        * torch.as_tensor(smoothness_cost, dtype=dtype, device=device)
    )
    safe_hit_time = torch.where(
        succeeded,
        torch.as_tensor(first_hit_time_s, dtype=dtype, device=device),
        torch.zeros_like(strike),
    )
    time_component = -config.successful_time_weight_per_s * safe_hit_time
    total = (
        progress_component
        + strike
        + success_component
        + safety_component
        + non_tip_component
        + control_component
        + time_component
    )
    return RLWhipRewardComponents(
        total=total,
        progress=progress_component,
        strike=strike,
        success=success_component,
        safety=safety_component,
        non_tip=non_tip_component,
        control=control_component,
        time=time_component,
        displacement_violation=displacement_violation,
        speed_violation=speed_violation,
        acceleration_violation=acceleration_violation,
        normalized_progress=normalized_progress,
        initial_tip_distance_m=initial_distance,
        minimum_tip_distance_m=minimum_distance,
    )
