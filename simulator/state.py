"""Coupled simulator state and differentiable rollout output."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .cable.dder import DderState
from .uav.state import UAVState


@dataclass(frozen=True, slots=True)
class SimulatorState:
    time_s: float
    uav: UAVState
    cable: DderState


@dataclass(frozen=True, slots=True)
class UAVTrajectory:
    """Time-major UAV-only view produced by the common coupled simulator."""

    times_s: torch.Tensor
    positions_m: torch.Tensor
    velocities_m_s: torch.Tensor
    orientations_xyzw: torch.Tensor
    angular_velocities_world_rad_s: torch.Tensor
    residual_accelerations_m_s2: torch.Tensor | None = None
    final_state: UAVState | None = None

    @property
    def step_count(self) -> int:
        return int(self.times_s.shape[0] - 1)


@dataclass(frozen=True, slots=True)
class SimulatorTrajectory:
    """Time-major state history, including the initial state at index zero."""

    times_s: torch.Tensor
    uav_positions_m: torch.Tensor
    uav_velocities_m_s: torch.Tensor
    uav_orientations_xyzw: torch.Tensor
    uav_angular_velocities_world_rad_s: torch.Tensor
    attachment_positions_m: torch.Tensor
    attachment_velocities_analytic_m_s: torch.Tensor
    prescribed_root_tangents_world: torch.Tensor | None
    cable_positions_m: torch.Tensor
    cable_velocities_m_s: torch.Tensor
    final_state: SimulatorState | None = None

    @property
    def step_count(self) -> int:
        return int(self.times_s.shape[0] - 1)

    @property
    def tip_positions_m(self) -> torch.Tensor:
        return self.cable_positions_m[..., -1, :]
