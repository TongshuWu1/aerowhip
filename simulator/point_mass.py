"""Force-controlled point mass coupled directly to the first DDER cable node.

There is no vehicle, attitude, motor, or tracking-controller state here. The
three-dimensional command is an external world-frame force applied at node 0.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping

import torch

from .cable import CableConfiguration, DderModel, DderState, FREE_ENDPOINTS


@dataclass(frozen=True, slots=True)
class PointForceController:
    """Apply a commanded world-frame force directly to cable node 0."""

    node_count: int

    def node_forces(
        self, commanded_force_world_n: torch.Tensor, *, validate: bool = True
    ) -> torch.Tensor:
        force = torch.as_tensor(commanded_force_world_n)
        if force.ndim == 1:
            force = force.unsqueeze(0)
        if force.ndim != 2 or force.shape[1] != 3:
            raise ValueError("commanded_force_world_n must have shape 3 or Bx3.")
        if validate and not bool(torch.isfinite(force).all().detach().cpu()):
            raise ValueError("commanded_force_world_n must contain finite values.")
        remaining = torch.zeros(
            (force.shape[0], self.node_count - 1, 3),
            dtype=force.dtype,
            device=force.device,
        )
        return torch.cat((force[:, None], remaining), dim=1)


@dataclass(frozen=True, slots=True)
class PointForceBreakdown:
    """Forces acting on the point mass during one discrete transition."""

    commanded_force_world_n: torch.Tensor
    gravity_force_on_point_world_n: torch.Tensor
    effective_cable_reaction_on_point_world_n: torch.Tensor
    net_force_on_point_world_n: torch.Tensor
    system_gravity_force_world_n: torch.Tensor


@dataclass(frozen=True, slots=True)
class PointCableStep:
    state: DderState
    forces: PointForceBreakdown


class ForceControlledPointCable:
    """One force-controlled point mass sharing node 0 with a DDER cable.

    Force accounting:

    * The commanded force acts only on node 0.
    * Gravity acts on the point mass and every cable vertex.
    * Elastic bending, Kelvin--Voigt damping, and inextensibility reactions are
      internal cable forces. They reach the point through the shared node.
    * Optional effective cable drag acts on free cable vertices. It does not
      identify drone drag, attitude dynamics, motor lag, or a UAV controller.
    """

    def __init__(
        self,
        cable: CableConfiguration,
        *,
        point_mass_kg: float,
        EI_n_m2: float,
        Cb_n_m2_s: float,
    ) -> None:
        if not math.isfinite(point_mass_kg) or point_mass_kg <= 0.0:
            raise ValueError("point_mass_kg must be finite and positive.")
        cable_parameters = cable.dder_parameters(EI=EI_n_m2, Cb=Cb_n_m2_s)
        combined_vertex_masses = list(cable.vertex_masses_kg)
        combined_vertex_masses[0] += float(point_mass_kg)
        combined_parameters = replace(
            cable_parameters,
            cable_mass_kg=cable.total_dynamic_mass_kg + float(point_mass_kg),
            vertex_masses_kg=tuple(combined_vertex_masses),
            external_drag_node_weights=(0.,) + (1.,) * (cable.node_count - 1),
        )
        self.cable_configuration = cable
        self.point_mass_kg = float(point_mass_kg)
        self.cable_mass_kg = cable.total_dynamic_mass_kg
        self.system_mass_kg = self.point_mass_kg + self.cable_mass_kg
        self.gravity_world_m_s2 = torch.tensor(
            cable.gravity_m_s2, dtype=torch.float64
        )
        self.dder = DderModel(combined_parameters)
        self.controller = PointForceController(cable.node_count)

    @classmethod
    def from_mapping(cls, model: Mapping[str, Any], *, root=None) -> "ForceControlledPointCable":
        cable = CableConfiguration.from_mapping(model["cable"])
        result = cls(
            cable,
            point_mass_kg=float(model["point_mass"]["mass_kg"]),
            EI_n_m2=float(model["cable"]["EI_n_m2"]),
            Cb_n_m2_s=float(model["cable"]["Cb_n_m2_s"]),
        )
        residual = model.get('motion_residual', {})
        if residual.get('enabled', False):
            from pathlib import Path
            from .cable.residual import FrozenMotionResidual
            path = Path(residual['checkpoint'])
            if not path.is_absolute():
                path = (Path(root) if root is not None else Path(__file__).resolve().parents[1]) / path
            result.dder.motion_residual = FrozenMotionResidual(path, residual['sha256'])
            if result.dder.motion_residual.payload['specification']['node_count'] != cable.node_count:
                raise ValueError('Residual node count does not match the physical cable.')
        return result

    def hanging_state(
        self,
        root_position_world_m: torch.Tensor,
        root_velocity_world_m_s: torch.Tensor | None = None,
    ) -> DderState:
        root = torch.as_tensor(root_position_world_m)
        if root.ndim == 1:
            root = root.unsqueeze(0)
        if root.ndim != 2 or root.shape[1] != 3:
            raise ValueError("root_position_world_m must have shape 3 or Bx3.")
        rest = torch.as_tensor(
            self.cable_configuration.rest_lengths_m,
            dtype=root.dtype,
            device=root.device,
        )
        arc = torch.cat((torch.zeros_like(rest[:1]), torch.cumsum(rest, dim=0)))
        positions = root[:, None].expand(-1, self.cable_configuration.node_count, -1).clone()
        positions[:, :, 2] -= arc[None]
        velocities = torch.zeros_like(positions)
        if root_velocity_world_m_s is not None:
            root_velocity = torch.as_tensor(
                root_velocity_world_m_s, dtype=root.dtype, device=root.device
            )
            if root_velocity.ndim == 1:
                root_velocity = root_velocity.unsqueeze(0)
            if root_velocity.shape != root.shape:
                raise ValueError("root_velocity_world_m_s must have shape 3 or Bx3.")
            velocities[:] = root_velocity[:, None]
        return self.dder.initial_state(positions, velocities)

    def hover_force_world_n(
        self, *, dtype: torch.dtype, device: torch.device | str
    ) -> torch.Tensor:
        gravity = self.gravity_world_m_s2.to(dtype=dtype, device=device)
        return -self.system_mass_kg * gravity

    def step(
        self,
        state: DderState,
        commanded_force_world_n: torch.Tensor,
        dt_s: torch.Tensor | float,
    ) -> PointCableStep:
        command = torch.as_tensor(
            commanded_force_world_n,
            dtype=state.positions_m.dtype,
            device=state.positions_m.device,
        )
        if command.ndim == 1:
            command = command.unsqueeze(0)
        if command.shape != (state.positions_m.shape[0], 3):
            raise ValueError("commanded_force_world_n must have shape Bx3.")
        external_force = self.controller.node_forces(command)
        empty_boundary = state.positions_m[:, :0]
        next_state = self.dder.step(
            state,
            empty_boundary,
            dt_s,
            external_force_world_n=external_force,
            pinned_endpoints=FREE_ENDPOINTS,
        )
        return self._result(state, next_state, command, dt_s)

    def step_runtime(
        self,
        state: DderState,
        commanded_force_world_n: torch.Tensor,
        dt_s: torch.Tensor | float,
    ) -> PointCableStep:
        """Host-synchronization-free transition for batched PPO rollouts."""

        command = torch.as_tensor(
            commanded_force_world_n,
            dtype=state.positions_m.dtype,
            device=state.positions_m.device,
        )
        if command.ndim == 1:
            command = command.unsqueeze(0)
        if command.shape != (state.positions_m.shape[0], 3):
            raise ValueError("commanded_force_world_n must have shape Bx3.")
        batch = state.positions_m.shape[0]
        dt = torch.as_tensor(
            dt_s, dtype=state.positions_m.dtype, device=state.positions_m.device
        )
        if dt.ndim == 0:
            dt = dt.expand(batch)
        if dt.shape != (batch,):
            raise ValueError("dt_s must be scalar or contain one value per batch item.")
        external_force = self.controller.node_forces(command, validate=False)
        next_state = self.dder.step_runtime(
            state,
            state.positions_m[:, :0],
            dt,
            self.dder.runtime_constants(state.positions_m),
            external_force_world_n=external_force,
            iterative_damping=state.positions_m.is_cuda,
            damping_backend=(
                "pcg32_experimental"
                if state.positions_m.is_cuda
                else "pcg60_reference"
            ),
            pinned_endpoints=FREE_ENDPOINTS,
            create_graph=False,
        )
        return self._result(state, next_state, command, dt)

    def _result(
        self,
        state: DderState,
        next_state: DderState,
        command: torch.Tensor,
        dt_s: torch.Tensor | float,
    ) -> PointCableStep:
        dt = torch.as_tensor(
            dt_s, dtype=state.positions_m.dtype, device=state.positions_m.device
        )
        if dt.ndim == 0:
            dt = dt.expand(state.positions_m.shape[0])
        if dt.shape != (state.positions_m.shape[0],):
            raise ValueError("dt_s must be scalar or contain one value per batch item.")
        point_acceleration = (
            next_state.velocities_m_s[:, 0] - state.velocities_m_s[:, 0]
        ) / dt[:, None]
        gravity = self.gravity_world_m_s2.to(
            dtype=state.positions_m.dtype, device=state.positions_m.device
        )
        gravity_on_point = self.point_mass_kg * gravity[None]
        net_on_point = self.point_mass_kg * point_acceleration
        cable_reaction = net_on_point - command - gravity_on_point
        return PointCableStep(
            next_state,
            PointForceBreakdown(
                commanded_force_world_n=command,
                gravity_force_on_point_world_n=gravity_on_point.expand_as(command),
                effective_cable_reaction_on_point_world_n=cable_reaction,
                net_force_on_point_world_n=net_on_point,
                system_gravity_force_world_n=(
                    self.system_mass_kg * gravity[None]
                ).expand_as(command),
            ),
        )
