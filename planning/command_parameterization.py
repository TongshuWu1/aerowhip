"""Acceleration-knot parameterization for consistent FullState commands."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from simulator.uav.state import FullStateCommandSequence


@dataclass(frozen=True, slots=True)
class FullStateCommandTrajectory:
    """Command samples at state times from zero through the full horizon."""

    times_s: torch.Tensor
    positions_m: torch.Tensor
    velocities_m_s: torch.Tensor
    accelerations_m_s2: torch.Tensor
    orientations_xyzw: torch.Tensor
    angular_velocities_body_rad_s: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.positions_m.shape[1])

    @property
    def interval_count(self) -> int:
        return int(self.times_s.numel() - 1)

    def simulator_sequence(self) -> FullStateCommandSequence:
        """Return commands at each interval start for the production stepper."""

        return FullStateCommandSequence(
            self.positions_m[:-1],
            self.velocities_m_s[:-1],
            self.accelerations_m_s2[:-1],
            self.orientations_xyzw[:-1],
            self.angular_velocities_body_rad_s[:-1],
        )


def project_acceleration_knots(
    knots_m_s2: torch.Tensor,
    maximum_norm_m_s2: float,
) -> torch.Tensor:
    """Project each three-axis knot onto the configured Euclidean ball."""

    knots = torch.as_tensor(knots_m_s2)
    if knots.ndim not in (2, 3) or knots.shape[-1] != 3:
        raise ValueError("Acceleration knots must have shape Kx3 or BxKx3.")
    if maximum_norm_m_s2 <= 0.0:
        raise ValueError("Acceleration bound must be positive.")
    norms = torch.linalg.vector_norm(knots, dim=-1, keepdim=True)
    scale = torch.clamp(maximum_norm_m_s2 / torch.clamp(norms, min=1.0e-12), max=1.0)
    return knots * scale


def _interpolate_knots(
    knots: torch.Tensor,
    times_s: torch.Tensor,
    horizon_s: float,
) -> torch.Tensor:
    batch, knot_count, _ = knots.shape
    normalized = times_s * (knot_count - 1) / horizon_s
    lower = torch.floor(normalized).to(torch.int64).clamp(0, knot_count - 2)
    fraction = (normalized - lower.to(normalized.dtype)).clamp(0.0, 1.0)
    # The horizon sample belongs exactly to the last knot.
    fraction = torch.where(
        times_s >= horizon_s - 4.0 * torch.finfo(times_s.dtype).eps,
        torch.ones_like(fraction),
        fraction,
    )
    lower_values = knots[:, lower, :]
    upper_values = knots[:, lower + 1, :]
    return (
        lower_values * (1.0 - fraction)[None, :, None]
        + upper_values * fraction[None, :, None]
    ).transpose(0, 1)


def acceleration_knots_to_fullstate(
    knots_m_s2: torch.Tensor,
    *,
    initial_position_m: torch.Tensor,
    initial_velocity_m_s: torch.Tensor,
    yaw_rad: float,
    horizon_s: float,
    dt_s: float,
) -> FullStateCommandTrajectory:
    """Linearly interpolate acceleration and integrate it without inconsistency.

    Acceleration is piecewise linear. Velocity uses its exact trapezoidal
    integral, and position uses the exact integral of that linear acceleration
    on every physics interval.
    """

    knots = torch.as_tensor(knots_m_s2)
    if knots.ndim == 2:
        knots = knots.unsqueeze(0)
    if knots.ndim != 3 or knots.shape[-1] != 3 or knots.shape[1] < 2:
        raise ValueError("Acceleration knots must have shape Kx3 or BxKx3.")
    if horizon_s <= 0.0 or dt_s <= 0.0:
        raise ValueError("Horizon and time step must be positive.")
    raw_steps = horizon_s / dt_s
    if abs(raw_steps - round(raw_steps)) > 1.0e-8:
        raise ValueError("Horizon must contain an integer number of physics steps.")
    step_count = int(round(raw_steps))
    batch = knots.shape[0]
    device, dtype = knots.device, knots.dtype

    def batched_vector(value: torch.Tensor, name: str) -> torch.Tensor:
        result = torch.as_tensor(value, dtype=dtype, device=device)
        if result.ndim == 1:
            result = result.unsqueeze(0)
        if result.shape == (1, 3) and batch != 1:
            result = result.expand(batch, 3)
        if result.shape != (batch, 3):
            raise ValueError(f"{name} must have shape 3, 1x3, or Bx3.")
        return result

    initial_position = batched_vector(initial_position_m, "initial_position_m")
    initial_velocity = batched_vector(initial_velocity_m_s, "initial_velocity_m_s")
    times = torch.arange(step_count + 1, dtype=dtype, device=device) * dt_s
    acceleration = _interpolate_knots(knots, times, horizon_s)
    delta_velocity = 0.5 * (acceleration[:-1] + acceleration[1:]) * dt_s
    velocities = torch.cat(
        (
            initial_velocity[None],
            initial_velocity[None] + torch.cumsum(delta_velocity, dim=0),
        ),
        dim=0,
    )
    delta_position = (
        velocities[:-1] * dt_s
        + (acceleration[:-1] / 3.0 + acceleration[1:] / 6.0) * (dt_s**2)
    )
    positions = torch.cat(
        (
            initial_position[None],
            initial_position[None] + torch.cumsum(delta_position, dim=0),
        ),
        dim=0,
    )
    orientations = torch.zeros(
        (step_count + 1, batch, 4), dtype=dtype, device=device
    )
    half_yaw = torch.as_tensor(0.5 * yaw_rad, dtype=dtype, device=device)
    orientations[..., 2] = torch.sin(half_yaw)
    orientations[..., 3] = torch.cos(half_yaw)
    angular_velocities = torch.zeros_like(positions)
    return FullStateCommandTrajectory(
        times,
        positions,
        velocities,
        acceleration,
        orientations,
        angular_velocities,
    )


def command_consistency_errors(
    command: FullStateCommandTrajectory,
) -> dict[str, float]:
    """Return exact-discrete integration residuals for structural tests."""

    dt = command.times_s[1:] - command.times_s[:-1]
    dv_expected = 0.5 * (
        command.accelerations_m_s2[:-1] + command.accelerations_m_s2[1:]
    ) * dt[:, None, None]
    dp_expected = (
        command.velocities_m_s[:-1] * dt[:, None, None]
        + (
            command.accelerations_m_s2[:-1] / 3.0
            + command.accelerations_m_s2[1:] / 6.0
        )
        * dt[:, None, None].square()
    )
    velocity_error = torch.max(
        torch.abs(command.velocities_m_s[1:] - command.velocities_m_s[:-1] - dv_expected)
    )
    position_error = torch.max(
        torch.abs(command.positions_m[1:] - command.positions_m[:-1] - dp_expected)
    )
    return {
        "maximum_velocity_consistency_error": float(velocity_error.detach().cpu()),
        "maximum_position_consistency_error": float(position_error.detach().cpu()),
    }
