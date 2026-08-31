"""Structured, invariant context for a one-shot open-loop manipulation policy."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Any

import torch

from planning.rollout import clone_state_batch
from simulator.parameters import SimulatorParameters
from simulator.simulator import CoupledSimulator
from simulator.state import SimulatorState
from simulator.uav.quaternion import (
    normalize_quaternion_xyzw,
    quaternion_conjugate_xyzw,
    quaternion_multiply_xyzw,
    quaternion_to_rotation_matrix_xyzw,
    yaw_from_quaternion_xyzw,
)


POLICY_CONTEXT_DIM = 83

_TENSOR_FIELDS: tuple[tuple[str, int, tuple[int, ...], str], ...] = (
    ("uav_velocity_local_m_s", 3, (3,), "linear"),
    ("uav_orientation_local_xyzw", 4, (4,), "canonical_xyzw"),
    ("uav_angular_velocity_local_rad_s", 3, (3,), "linear"),
    ("cable_positions_local_m", 30, (10, 3), "flattened_c1_to_c10"),
    ("cable_velocities_local_m_s", 30, (10, 3), "flattened_c1_to_c10"),
    ("target_position_local_m", 3, (3,), "linear"),
    ("target_direction_local", 3, (3,), "unit_vector"),
    ("physics_learning_features", 7, (7,), "Kp,Kv,ka,KR,Komega,logEI,logCb"),
)


def policy_context_tensor_metadata() -> dict[str, Any]:
    """Return the authoritative flat-tensor slices without scattering indices."""

    start = 0
    items: list[dict[str, Any]] = []
    for name, width, shape, encoding in _TENSOR_FIELDS:
        items.append(
            {
                "name": name,
                "start": start,
                "stop": start + width,
                "shape": list(shape),
                "encoding": encoding,
            }
        )
        start += width
    if start != POLICY_CONTEXT_DIM:
        raise RuntimeError("Policy-context tensor metadata has an inconsistent width.")
    return {
        "dimension": POLICY_CONTEXT_DIM,
        "batch_axis": 0,
        "fields": items,
        "quaternion_convention": "xyzw active body-to-local; sign canonicalized",
        "angular_velocity_convention": "yaw-aligned local frame, rad/s",
        "physics_feature_order": [
            "K_p",
            "K_v",
            "k_a",
            "K_R",
            "K_omega",
            "log_EI",
            "log_Cb",
        ],
    }


def _batch_scalar(
    value: float | torch.Tensor,
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.ndim == 0:
        tensor = tensor.expand(batch_size)
    if tensor.shape != (batch_size,) or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite and scalar or shape B.")
    return tensor


def _batch_vector(
    value: torch.Tensor | tuple[float, float, float],
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.shape == (1, 3) and batch_size > 1:
        tensor = tensor.expand(batch_size, -1)
    if tensor.shape != (batch_size, 3) or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite with shape 3, 1x3, or Bx3.")
    return tensor


def _world_to_local(rotation_world_from_local: torch.Tensor, vector_world: torch.Tensor) -> torch.Tensor:
    return torch.matmul(
        rotation_world_from_local.transpose(-1, -2), vector_world.unsqueeze(-1)
    ).squeeze(-1)


def _local_to_world(rotation_world_from_local: torch.Tensor, vector_local: torch.Tensor) -> torch.Tensor:
    return torch.matmul(rotation_world_from_local, vector_local.unsqueeze(-1)).squeeze(-1)


def _canonicalize_quaternion_sign(quaternion_xyzw: torch.Tensor) -> torch.Tensor:
    """Choose one deterministic representative for the q/-q equivalence class."""

    quaternion = normalize_quaternion_xyzw(quaternion_xyzw)
    dominant = torch.argmax(torch.abs(quaternion), dim=-1, keepdim=True)
    component = torch.gather(quaternion, dim=-1, index=dominant)
    sign = torch.where(component < 0.0, -torch.ones_like(component), torch.ones_like(component))
    return quaternion * sign


@dataclass(frozen=True, slots=True)
class PhysicsContext:
    """Raw seven-parameter effective model context carried per policy query."""

    K_p: torch.Tensor
    K_v: torch.Tensor
    k_a: torch.Tensor
    K_R: torch.Tensor
    K_omega: torch.Tensor
    EI: torch.Tensor
    Cb: torch.Tensor

    def __post_init__(self) -> None:
        reference = torch.as_tensor(self.K_p)
        if reference.ndim != 1 or reference.numel() < 1:
            raise ValueError("Physics-context fields must be one-dimensional batches.")
        for field in fields(self):
            value = torch.as_tensor(
                getattr(self, field.name), dtype=reference.dtype, device=reference.device
            )
            if value.shape != reference.shape or not bool(torch.isfinite(value).all()):
                raise ValueError("Every physics-context field must be finite with shape B.")
            if not bool((value > 0.0).all()):
                raise ValueError("Every physics-context parameter must be positive.")
            object.__setattr__(self, field.name, value)

    @property
    def batch_size(self) -> int:
        return int(self.K_p.shape[0])

    @classmethod
    def from_parameters(
        cls,
        parameters: SimulatorParameters,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "PhysicsContext":
        values = {
            "K_p": parameters.uav.K_p,
            "K_v": parameters.uav.K_v,
            "k_a": parameters.uav.k_a,
            "K_R": parameters.uav.K_R,
            "K_omega": parameters.uav.K_omega,
            "EI": parameters.cable.EI,
            "Cb": parameters.cable.Cb,
        }
        return cls(
            **{
                name: _batch_scalar(
                    value,
                    batch_size=batch_size,
                    dtype=dtype,
                    device=device,
                    name=name,
                )
                for name, value in values.items()
            }
        )

    def raw_tensor(self) -> torch.Tensor:
        return torch.stack(
            (self.K_p, self.K_v, self.k_a, self.K_R, self.K_omega, self.EI, self.Cb),
            dim=-1,
        )

    def learning_tensor(self) -> torch.Tensor:
        return torch.stack(
            (
                self.K_p,
                self.K_v,
                self.k_a,
                self.K_R,
                self.K_omega,
                torch.log(self.EI),
                torch.log(self.Cb),
            ),
            dim=-1,
        )


@dataclass(frozen=True, slots=True)
class PolicyFrame:
    """World provenance for the root-centered, yaw-aligned policy frame."""

    root_position_world_m: torch.Tensor
    yaw_world_rad: torch.Tensor
    rotation_world_from_local: torch.Tensor

    def __post_init__(self) -> None:
        root = torch.as_tensor(self.root_position_world_m)
        yaw = torch.as_tensor(self.yaw_world_rad, dtype=root.dtype, device=root.device)
        rotation = torch.as_tensor(
            self.rotation_world_from_local, dtype=root.dtype, device=root.device
        )
        if root.ndim != 2 or root.shape[-1] != 3:
            raise ValueError("Policy-frame roots must have shape Bx3.")
        if yaw.shape != (root.shape[0],) or rotation.shape != (root.shape[0], 3, 3):
            raise ValueError("Policy-frame yaw/rotation batch shape mismatch.")
        if not bool(torch.isfinite(root).all() and torch.isfinite(yaw).all() and torch.isfinite(rotation).all()):
            raise ValueError("Policy frame must be finite.")
        object.__setattr__(self, "root_position_world_m", root)
        object.__setattr__(self, "yaw_world_rad", yaw)
        object.__setattr__(self, "rotation_world_from_local", rotation)

    @property
    def batch_size(self) -> int:
        return int(self.root_position_world_m.shape[0])

    def vectors_to_world(self, vectors_local: torch.Tensor) -> torch.Tensor:
        vectors = torch.as_tensor(vectors_local, device=self.root_position_world_m.device)
        if vectors.shape[0] != self.batch_size or vectors.shape[-1] != 3:
            raise ValueError("Local vectors must have batch-leading shape Bx...x3.")
        rotation = self.rotation_world_from_local.to(dtype=vectors.dtype)
        return torch.einsum("bij,b...j->b...i", rotation, vectors)

    def vectors_to_local(self, vectors_world: torch.Tensor) -> torch.Tensor:
        vectors = torch.as_tensor(vectors_world, device=self.root_position_world_m.device)
        if vectors.shape[0] != self.batch_size or vectors.shape[-1] != 3:
            raise ValueError("World vectors must have batch-leading shape Bx...x3.")
        rotation = self.rotation_world_from_local.to(dtype=vectors.dtype)
        return torch.einsum(
            "bij,b...j->b...i", rotation.transpose(-1, -2), vectors
        )

    def points_to_world(self, points_local: torch.Tensor) -> torch.Tensor:
        points = self.vectors_to_world(points_local)
        root = self.root_position_world_m.to(dtype=points.dtype)
        while root.ndim < points.ndim:
            root = root.unsqueeze(1)
        return points + root


@dataclass(frozen=True, slots=True)
class PolicyContext:
    """One batched policy query plus non-learning world-frame rollout provenance."""

    uav_velocity_local_m_s: torch.Tensor
    uav_orientation_local_xyzw: torch.Tensor
    uav_angular_velocity_local_rad_s: torch.Tensor
    cable_positions_local_m: torch.Tensor
    cable_velocities_local_m_s: torch.Tensor
    target_position_local_m: torch.Tensor
    target_direction_local: torch.Tensor
    physics: PhysicsContext
    frame: PolicyFrame
    initial_state_world: SimulatorState
    command_initial_position_world_m: torch.Tensor
    command_initial_velocity_world_m_s: torch.Tensor
    command_yaw_world_rad: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.uav_velocity_local_m_s.shape[0])

    def to_tensor(self) -> torch.Tensor:
        """Flatten only learning features; world provenance never enters the network."""

        batch = self.batch_size
        tensor = torch.cat(
            (
                self.uav_velocity_local_m_s,
                self.uav_orientation_local_xyzw,
                self.uav_angular_velocity_local_rad_s,
                self.cable_positions_local_m.reshape(batch, -1),
                self.cable_velocities_local_m_s.reshape(batch, -1),
                self.target_position_local_m,
                self.target_direction_local,
                self.physics.learning_tensor(),
            ),
            dim=-1,
        )
        if tensor.shape != (batch, POLICY_CONTEXT_DIM) or not bool(torch.isfinite(tensor).all()):
            raise RuntimeError("Policy-context tensor failed its shape/finite contract.")
        return tensor

    @property
    def tensor_metadata(self) -> dict[str, Any]:
        return policy_context_tensor_metadata()

    def target_position_world_m(self) -> torch.Tensor:
        return self.frame.points_to_world(self.target_position_local_m)

    def target_direction_world(self) -> torch.Tensor:
        direction = self.frame.vectors_to_world(self.target_direction_local)
        return direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True)


def build_policy_context(
    simulator: CoupledSimulator,
    initial_state: SimulatorState,
    *,
    target_position_world_m: torch.Tensor | tuple[float, float, float],
    desired_direction_world: torch.Tensor | tuple[float, float, float],
    physics_context: PhysicsContext | None = None,
    command_initial_position_world_m: torch.Tensor | tuple[float, float, float] | None = None,
    command_initial_velocity_world_m_s: torch.Tensor | tuple[float, float, float] | None = None,
    command_yaw_world_rad: torch.Tensor | float | None = None,
) -> PolicyContext:
    """Build the root-centered, gravity-preserving yaw-aligned context once."""

    state = clone_state_batch(initial_state, initial_state.uav.batch_size)
    batch = state.uav.batch_size
    dtype, device = state.uav.position_m.dtype, state.uav.position_m.device
    if state.cable.positions_m.shape[1:] != (12, 3):
        raise ValueError("Policy context requires the production 12-node cable state.")
    boundary = simulator.root_boundary.evaluate(
        state.uav, simulator.cable_configuration.rest_lengths_m[0]
    )
    root = boundary.attachment_position_m
    yaw = yaw_from_quaternion_xyzw(state.uav.orientation_xyzw)
    # Remove numerically insignificant float32 hover drift so the canonical
    # yaw-zero CEM action can pass through normalized encode/decode without a
    # spurious frame rotation.  Larger physical yaw is never quantized.
    yaw_tolerance = 32.0 * torch.finfo(dtype).eps
    yaw = torch.where(torch.abs(yaw) <= yaw_tolerance, torch.zeros_like(yaw), yaw)
    yaw_quaternion = torch.zeros((batch, 4), dtype=dtype, device=device)
    yaw_quaternion[:, 2] = torch.sin(0.5 * yaw)
    yaw_quaternion[:, 3] = torch.cos(0.5 * yaw)
    rotation = quaternion_to_rotation_matrix_xyzw(yaw_quaternion)
    frame = PolicyFrame(root, yaw, rotation)

    target_world = _batch_vector(
        target_position_world_m,
        batch_size=batch,
        dtype=dtype,
        device=device,
        name="target_position_world_m",
    )
    direction_world = _batch_vector(
        desired_direction_world,
        batch_size=batch,
        dtype=dtype,
        device=device,
        name="desired_direction_world",
    )
    direction_norm = torch.linalg.vector_norm(direction_world, dim=-1, keepdim=True)
    if not bool((direction_norm > torch.finfo(dtype).eps).all()):
        raise ValueError("Desired strike direction must be nonzero.")
    direction_world = direction_world / direction_norm

    local_orientation = quaternion_multiply_xyzw(
        quaternion_conjugate_xyzw(yaw_quaternion), state.uav.orientation_xyzw
    )
    local_orientation = _canonicalize_quaternion_sign(local_orientation)
    cable_world = state.cable.positions_m[:, 2:12]
    cable_velocity_world = state.cable.velocities_m_s[:, 2:12]
    cable_relative_world = cable_world - root[:, None]
    cable_local = torch.einsum(
        "bij,bnj->bni", rotation.transpose(-1, -2), cable_relative_world
    )
    cable_velocity_local = torch.einsum(
        "bij,bnj->bni", rotation.transpose(-1, -2), cable_velocity_world
    )
    physics = physics_context or PhysicsContext.from_parameters(
        simulator.parameters,
        batch_size=batch,
        dtype=dtype,
        device=device,
    )
    if physics.batch_size != batch:
        raise ValueError("Physics-context batch size must match the initial state.")
    command_position = _batch_vector(
        state.uav.position_m
        if command_initial_position_world_m is None
        else command_initial_position_world_m,
        batch_size=batch,
        dtype=dtype,
        device=device,
        name="command_initial_position_world_m",
    )
    command_velocity = _batch_vector(
        state.uav.velocity_m_s
        if command_initial_velocity_world_m_s is None
        else command_initial_velocity_world_m_s,
        batch_size=batch,
        dtype=dtype,
        device=device,
        name="command_initial_velocity_world_m_s",
    )
    command_yaw = _batch_scalar(
        yaw if command_yaw_world_rad is None else command_yaw_world_rad,
        batch_size=batch,
        dtype=dtype,
        device=device,
        name="command_yaw_world_rad",
    )
    context = PolicyContext(
        uav_velocity_local_m_s=_world_to_local(rotation, state.uav.velocity_m_s),
        uav_orientation_local_xyzw=local_orientation,
        uav_angular_velocity_local_rad_s=_world_to_local(
            rotation, state.uav.angular_velocity_world_rad_s
        ),
        cable_positions_local_m=cable_local,
        cable_velocities_local_m_s=cable_velocity_local,
        target_position_local_m=_world_to_local(rotation, target_world - root),
        target_direction_local=_world_to_local(rotation, direction_world),
        physics=physics,
        frame=frame,
        initial_state_world=state,
        command_initial_position_world_m=command_position,
        command_initial_velocity_world_m_s=command_velocity,
        command_yaw_world_rad=command_yaw,
    )
    context.to_tensor()
    return context
