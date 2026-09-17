"""Physically transparent cable initialization helpers."""

from __future__ import annotations

import torch

from .config import CableConfiguration
from .dder import DderModel, DderState


def hanging_positions(
    root_positions_m: torch.Tensor,
    configuration: CableConfiguration,
) -> torch.Tensor:
    """Return a straight, vertically hanging cable for each batched root."""

    roots = torch.as_tensor(root_positions_m)
    if roots.ndim == 1:
        roots = roots.unsqueeze(0)
    if roots.ndim != 2 or roots.shape[1] != 3:
        raise ValueError("root_positions_m must have shape 3 or Bx3.")
    arc = torch.cat(
        (
            torch.zeros(1, dtype=roots.dtype, device=roots.device),
            torch.cumsum(
                torch.as_tensor(
                    configuration.rest_lengths_m,
                    dtype=roots.dtype,
                    device=roots.device,
                ),
                dim=0,
            ),
        )
    )
    positions = roots[:, None, :].expand(-1, configuration.node_count, -1).clone()
    positions[:, :, 2] = positions[:, :, 2] - arc[None]
    return positions


def hanging_cable_state(
    model: DderModel,
    root_positions_m: torch.Tensor,
    configuration: CableConfiguration,
    root_velocities_m_s: torch.Tensor | None = None,
) -> DderState:
    positions = hanging_positions(root_positions_m, configuration)
    velocities = torch.zeros_like(positions)
    if root_velocities_m_s is not None:
        root_velocity = torch.as_tensor(
            root_velocities_m_s,
            dtype=positions.dtype,
            device=positions.device,
        )
        if root_velocity.ndim == 1:
            root_velocity = root_velocity.unsqueeze(0)
        if root_velocity.shape != positions[:, 0].shape:
            raise ValueError("root_velocities_m_s must have shape 3 or Bx3.")
        velocities[:, 0] = root_velocity
    return model.initial_state(positions, velocities)


def clamped_hanging_cable_state(
    model: DderModel,
    boundary_positions_m: torch.Tensor,
    configuration: CableConfiguration,
    boundary_velocities_m_s: torch.Tensor | None = None,
) -> DderState:
    """Initialize a feasible cable with its first physical edge prescribed.

    Nodes 0 and 1 are the discrete position+tangent boundary.  The remaining
    edges start vertically downward under gravity while retaining their exact
    material lengths.  This is a feasible initial configuration, not an
    equilibrium solve for a generally tilted clamp.
    """

    boundary = torch.as_tensor(boundary_positions_m)
    if boundary.ndim != 3 or boundary.shape[1:] != (2, 3):
        raise ValueError("boundary_positions_m must have shape Bx2x3.")
    rest = torch.as_tensor(
        configuration.rest_lengths_m,
        dtype=boundary.dtype,
        device=boundary.device,
    )
    positions = boundary[:, 1:2].expand(-1, configuration.node_count, -1).clone()
    positions[:, 0] = boundary[:, 0]
    positions[:, 1] = boundary[:, 1]
    if configuration.node_count > 2:
        remaining_arc = torch.cumsum(rest[1:], dim=0)
        positions[:, 2:, 2] = positions[:, 2:, 2] - remaining_arc[None]
    velocities = torch.zeros_like(positions)
    if boundary_velocities_m_s is not None:
        boundary_velocity = torch.as_tensor(
            boundary_velocities_m_s,
            dtype=positions.dtype,
            device=positions.device,
        )
        if boundary_velocity.shape != boundary.shape:
            raise ValueError("boundary_velocities_m_s must have shape Bx2x3.")
        velocities[:, :2] = boundary_velocity
    return model.initial_state(positions, velocities)
