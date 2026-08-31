"""Nominal-physics training and fixed-validation context construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from simulator.simulator import CoupledSimulator
from simulator.uav.quaternion import quaternion_to_rotation_matrix_xyzw, yaw_from_quaternion_xyzw

from .policy_context import PolicyContext, build_policy_context
from .state_bank import InitialStateBank, StateBankBatch


@dataclass(frozen=True, slots=True)
class ContextSpecification:
    state_indices: torch.Tensor
    target_position_local_m: torch.Tensor
    desired_direction_local: torch.Tensor
    split: str

    def __post_init__(self) -> None:
        indices = torch.as_tensor(self.state_indices, dtype=torch.int64, device="cpu").reshape(-1)
        target = torch.as_tensor(self.target_position_local_m, dtype=torch.float32, device="cpu")
        direction = torch.as_tensor(self.desired_direction_local, dtype=torch.float32, device="cpu")
        if target.shape != (indices.numel(), 3) or direction.shape != target.shape:
            raise ValueError("Context specification batch shape mismatch.")
        norms = torch.linalg.vector_norm(direction, dim=-1)
        if not bool(torch.isfinite(target).all() and torch.isfinite(direction).all()):
            raise ValueError("Context specification must be finite.")
        if not bool(torch.allclose(norms, torch.ones_like(norms), atol=1.0e-6, rtol=1.0e-6)):
            raise ValueError("Every context direction must be normalized.")
        object.__setattr__(self, "state_indices", indices)
        object.__setattr__(self, "target_position_local_m", target)
        object.__setattr__(self, "desired_direction_local", direction)

    @property
    def count(self) -> int:
        return int(self.state_indices.numel())


def target_direction_from_local_target(target_local_m: torch.Tensor) -> torch.Tensor:
    target = torch.as_tensor(target_local_m)
    horizontal = torch.stack((target[..., 0], target[..., 1], torch.zeros_like(target[..., 0])), dim=-1)
    norm = torch.linalg.vector_norm(horizontal, dim=-1, keepdim=True)
    if not bool((norm > torch.finfo(target.dtype).eps).all()):
        raise ValueError("Phase-1 target has a degenerate horizontal direction.")
    return horizontal / norm


def sample_context_specification(
    bank: InitialStateBank,
    *,
    count: int,
    generator: torch.Generator,
    split: Literal["training", "validation_iid", "edge"],
    canonical_fraction: float = 0.20,
) -> ContextSpecification:
    if count < 1:
        raise ValueError("At least one context must be sampled.")
    indices = torch.randint(len(bank), (count,), generator=generator)
    random = torch.rand((count, 3), generator=generator)
    target = torch.empty((count, 3), dtype=torch.float32)
    target[:, 0] = 0.85 + 0.20 * random[:, 0]
    target[:, 1] = -0.15 + 0.30 * random[:, 1]
    target[:, 2] = -0.12 + 0.14 * random[:, 2]
    if split == "training":
        canonical = torch.rand((count,), generator=generator) < canonical_fraction
        near = torch.randn((count, 3), generator=generator) * torch.tensor([0.025, 0.025, 0.015])
        canonical_target = torch.tensor([1.0, 0.0, -0.045], dtype=torch.float32) + near
        canonical_target[:, 0].clamp_(0.85, 1.05)
        canonical_target[:, 1].clamp_(-0.15, 0.15)
        canonical_target[:, 2].clamp_(-0.12, 0.02)
        target = torch.where(canonical[:, None], canonical_target, target)
    elif split == "edge":
        target[:, 0] = torch.where(random[:, 0] < 0.5, 0.85, 1.05)
        target[:, 1] = torch.where(random[:, 1] < 0.5, -0.15, 0.15)
        target[:, 2] = torch.where(random[:, 2] < 0.5, -0.12, 0.02)
        # Use the dynamically most excited validation-bank rows at the domain edge.
        excitation = torch.linalg.vector_norm(bank.uav_velocity_m_s, dim=-1) + torch.linalg.vector_norm(
            bank.cable_velocities_m_s[:, 2:12], dim=-1
        ).mean(dim=-1)
        order = torch.argsort(excitation, descending=True)
        indices = order[torch.arange(count) % len(order)]
    elif split != "validation_iid":
        raise ValueError(f"Unsupported context split: {split}")
    return ContextSpecification(indices, target, target_direction_from_local_target(target), split)


def _world_from_local_frame(state_batch: StateBankBatch) -> torch.Tensor:
    yaw = yaw_from_quaternion_xyzw(state_batch.state.uav.orientation_xyzw)
    quaternion = torch.zeros((yaw.shape[0], 4), dtype=yaw.dtype, device=yaw.device)
    quaternion[:, 2] = torch.sin(0.5 * yaw)
    quaternion[:, 3] = torch.cos(0.5 * yaw)
    return quaternion_to_rotation_matrix_xyzw(quaternion)


def build_context_from_specification(
    simulator: CoupledSimulator,
    bank: InitialStateBank,
    specification: ContextSpecification,
) -> PolicyContext:
    selected = bank.select(specification.state_indices, device=simulator.device)
    boundary = simulator.root_boundary.evaluate(
        selected.state.uav, simulator.cable_configuration.rest_lengths_m[0]
    )
    rotation = _world_from_local_frame(selected)
    target_local = specification.target_position_local_m.to(simulator.device)
    direction_local = specification.desired_direction_local.to(simulator.device)
    target_world = boundary.attachment_position_m + torch.einsum(
        "bij,bj->bi", rotation, target_local
    )
    direction_world = torch.einsum("bij,bj->bi", rotation, direction_local)
    return build_policy_context(
        simulator,
        selected.state,
        target_position_world_m=target_world,
        desired_direction_world=direction_world,
        command_initial_position_world_m=selected.command_position_world_m,
        command_initial_velocity_world_m_s=selected.command_velocity_world_m_s,
        command_yaw_world_rad=selected.command_yaw_world_rad,
    )


def pad_context_specification(specification: ContextSpecification, size: int) -> ContextSpecification:
    if size < specification.count:
        raise ValueError("Cannot pad a context specification to a smaller size.")
    if size == specification.count:
        return specification
    repeat = size - specification.count
    source = torch.arange(repeat) % specification.count
    return ContextSpecification(
        torch.cat((specification.state_indices, specification.state_indices[source])),
        torch.cat((specification.target_position_local_m, specification.target_position_local_m[source])),
        torch.cat((specification.desired_direction_local, specification.desired_direction_local[source])),
        specification.split,
    )
