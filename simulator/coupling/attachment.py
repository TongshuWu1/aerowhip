"""One-way rigid UAV-pose to position-driven DDER attachment coupling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ..uav.state import UAVState
from ..uav.quaternion import quaternion_to_rotation_matrix_xyzw


@dataclass(frozen=True, slots=True)
class AttachmentState:
    position_m: torch.Tensor
    analytic_velocity_m_s: torch.Tensor


@dataclass(frozen=True, slots=True)
class RigidAttachmentState:
    root: AttachmentState
    tangent_world: torch.Tensor
    first_edge_position_m: torch.Tensor
    first_edge_analytic_velocity_m_s: torch.Tensor


def attachment_state(
    uav_state: UAVState,
    attachment_offset_body_m: torch.Tensor | Sequence[float],
) -> AttachmentState:
    """Compute attachment position and analytic velocity in world coordinates."""

    offset = torch.as_tensor(
        attachment_offset_body_m,
        dtype=uav_state.position_m.dtype,
        device=uav_state.position_m.device,
    )
    if offset.ndim == 1:
        if offset.shape != (3,):
            raise ValueError("attachment_offset_body_m must have three components.")
        offset = offset.unsqueeze(0).expand(uav_state.batch_size, -1)
    if offset.shape != uav_state.position_m.shape:
        raise ValueError("Attachment offset must have shape 3 or Bx3.")
    rotation = quaternion_to_rotation_matrix_xyzw(uav_state.orientation_xyzw)
    offset_world = torch.matmul(rotation, offset.unsqueeze(-1)).squeeze(-1)
    position = uav_state.position_m + offset_world
    velocity = uav_state.velocity_m_s + torch.linalg.cross(
        uav_state.angular_velocity_world_rad_s,
        offset_world,
        dim=-1,
    )
    return AttachmentState(position, velocity)


def rigid_attachment_state(
    uav_state: UAVState,
    attachment_offset_body_m: torch.Tensor | Sequence[float],
    attachment_tangent_body: torch.Tensor | Sequence[float],
    first_edge_length_m: torch.Tensor | float,
) -> RigidAttachmentState:
    """Return the centerline-clamped boundary implied by the UAV pose."""

    offset = torch.as_tensor(
        attachment_offset_body_m,
        dtype=uav_state.position_m.dtype,
        device=uav_state.position_m.device,
    )
    tangent = torch.as_tensor(
        attachment_tangent_body,
        dtype=uav_state.position_m.dtype,
        device=uav_state.position_m.device,
    )
    if offset.shape != (3,) or tangent.shape != (3,):
        raise ValueError("Rigid attachment offset and tangent must each have shape 3.")
    tangent_norm = torch.linalg.vector_norm(tangent)
    minimum_norm = torch.finfo(tangent.dtype).eps
    tangent_body = tangent / torch.clamp(tangent_norm, min=minimum_norm)
    rotation = quaternion_to_rotation_matrix_xyzw(uav_state.orientation_xyzw)
    offset_world = torch.matmul(rotation, offset.view(1, 3, 1)).squeeze(-1)
    tangent_world = torch.matmul(
        rotation, tangent_body.view(1, 3, 1)
    ).squeeze(-1)
    tangent_world = tangent_world / torch.clamp(
        torch.linalg.vector_norm(tangent_world, dim=-1, keepdim=True),
        min=minimum_norm,
    )
    root_position = uav_state.position_m + offset_world
    root_velocity = uav_state.velocity_m_s + torch.linalg.cross(
        uav_state.angular_velocity_world_rad_s, offset_world, dim=-1
    )
    edge_length = torch.as_tensor(
        first_edge_length_m,
        dtype=uav_state.position_m.dtype,
        device=uav_state.position_m.device,
    )
    first_edge_offset_world = offset_world + edge_length * tangent_world
    first_edge_position = uav_state.position_m + first_edge_offset_world
    first_edge_velocity = uav_state.velocity_m_s + torch.linalg.cross(
        uav_state.angular_velocity_world_rad_s,
        first_edge_offset_world,
        dim=-1,
    )
    return RigidAttachmentState(
        AttachmentState(root_position, root_velocity),
        tangent_world,
        first_edge_position,
        first_edge_velocity,
    )


def attachment_boundary_position(
    uav_state: UAVState,
    attachment_offset_body_m: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the prescribed DDER node-0 position with shape Bx1x3.

    This compatibility helper preserves the Milestone-1 position-only call.
    The current coupling remains strictly one-way from UAV pose to cable.
    """

    offset = (
        torch.zeros(3, dtype=uav_state.position_m.dtype, device=uav_state.position_m.device)
        if attachment_offset_body_m is None
        else attachment_offset_body_m
    )
    return attachment_state(uav_state, offset).position_m[:, None, :]
