"""Common differentiable UAV-root plus distributed-cable simulator API."""

from __future__ import annotations

from dataclasses import replace
import os

import torch

from .cable.config import CableConfiguration
from .cable.cuda_fixed_pcg import VALIDATED_OPTIMIZED_DAMPING_BACKEND
from .cable.dder import DderModel, DderRuntimeConstants
from .coupling.root_boundary import (
    ClampedRootBoundary,
    PivotRootBoundary,
    RootBoundary,
)
from .parameters import SimulatorParameters
from .state import SimulatorState, SimulatorTrajectory, UAVTrajectory
from .uav.model import FullStateUAVModel, PrescribedRootModel, UAVModel
from .uav.state import (
    FullStateCommand,
    FullStateCommandSequence,
    UAVCommand,
    UAVCommandSequence,
    UAVState,
)


Command = UAVCommand | FullStateCommand
CommandSequence = UAVCommandSequence | FullStateCommandSequence


class CoupledSimulator:
    """One differentiable UAV–boundary–DDER implementation.

    A prescribed-root UAV model selects the legacy position-only pivot.  The
    FullState response model selects a two-node centerline clamp driven by the
    simulated UAV pose.  Both paths use the same interior DDER equations.
    """

    def __init__(
        self,
        cable_configuration: CableConfiguration,
        parameters: SimulatorParameters,
        *,
        dt_s: float,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
        uav_model: UAVModel | None = None,
        attachment_offset_body_m: torch.Tensor | tuple[float, float, float] = (0.0, 0.0, 0.0),
        attachment_tangent_body: torch.Tensor | tuple[float, float, float] = (0.0, 0.0, -1.0),
        root_boundary: RootBoundary | None = None,
        functional_force_autograd: bool = False,
    ) -> None:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive.")
        self.cable_configuration = cable_configuration
        self.parameters = parameters
        self.dt_s = float(dt_s)
        self.device = torch.device(device)
        if dtype is None:
            dtype = torch.float32 if self.device.type == "cuda" else torch.float64
        self.dtype = dtype
        self.uav_model = uav_model or PrescribedRootModel()
        self.functional_force_autograd = bool(functional_force_autograd)
        offset = torch.as_tensor(
            attachment_offset_body_m, dtype=dtype, device=self.device
        )
        tangent = torch.as_tensor(
            attachment_tangent_body, dtype=dtype, device=self.device
        )
        if offset.shape != (3,) or not bool(torch.isfinite(offset).all()):
            raise ValueError("attachment_offset_body_m must have exactly three values.")
        if tangent.shape != (3,):
            raise ValueError("attachment_tangent_body must have exactly three values.")
        if not bool(torch.isfinite(tangent).all()) or float(
            torch.linalg.vector_norm(tangent)
        ) <= torch.finfo(dtype).eps:
            raise ValueError("attachment_tangent_body must be finite and nonzero.")
        offset_values = tuple(float(value) for value in offset.detach().cpu())
        tangent_values = tuple(float(value) for value in tangent.detach().cpu())
        self.root_boundary = root_boundary or (
            ClampedRootBoundary(offset_values, tangent_values)
            if isinstance(self.uav_model, FullStateUAVModel)
            else PivotRootBoundary(offset_values)
        )

        # The model object requires scalar reference values for fixed geometry
        # and stability metadata. Every actual step receives the original EI/Cb
        # tensor again, preserving the differentiable parameter path.
        reference_ei = float(torch.as_tensor(parameters.cable.EI).detach().cpu())
        reference_cb = float(torch.as_tensor(parameters.cable.Cb).detach().cpu())
        self.cable_model = DderModel(
            cable_configuration.dder_parameters(EI=reference_ei, Cb=reference_cb)
        )
        self._state: SimulatorState | None = None
        self._runtime_constants_cache: dict[
            tuple[torch.device, torch.dtype, int], DderRuntimeConstants
        ] = {}
        self._runtime_dt_cache: dict[
            tuple[torch.device, torch.dtype, int], torch.Tensor
        ] = {}
        self.last_dder_execution = "not_started"

    @property
    def state(self) -> SimulatorState:
        if self._state is None:
            raise RuntimeError("Simulator has not been reset.")
        return self._state

    def reset(
        self,
        uav_position_m: torch.Tensor,
        uav_velocity_m_s: torch.Tensor | None = None,
        uav_orientation_xyzw: torch.Tensor | None = None,
        uav_angular_velocity_world_rad_s: torch.Tensor | None = None,
    ) -> SimulatorState:
        position = torch.as_tensor(
            uav_position_m, dtype=self.dtype, device=self.device
        )
        if position.ndim == 1:
            position = position.unsqueeze(0)
        velocity = (
            torch.zeros_like(position)
            if uav_velocity_m_s is None
            else torch.as_tensor(
                uav_velocity_m_s, dtype=self.dtype, device=self.device
            )
        )
        if velocity.ndim == 1:
            velocity = velocity.unsqueeze(0)
        orientation = None if uav_orientation_xyzw is None else torch.as_tensor(
            uav_orientation_xyzw, dtype=self.dtype, device=self.device
        )
        angular_velocity = (
            None
            if uav_angular_velocity_world_rad_s is None
            else torch.as_tensor(
                uav_angular_velocity_world_rad_s,
                dtype=self.dtype,
                device=self.device,
            )
        )
        uav = UAVState(position, velocity, orientation, angular_velocity)
        if isinstance(self.uav_model, FullStateUAVModel):
            uav = self.uav_model.initialize_residual_state(uav)
        cable = self.root_boundary.initialize_cable(
            self.cable_model,
            uav,
            self.cable_configuration,
        )
        self._state = SimulatorState(0.0, uav, cable)
        return self._state

    def _propagate(
        self,
        state: SimulatorState,
        command: Command,
        parameters: SimulatorParameters,
        *,
        create_graph: bool,
    ) -> SimulatorState:
        next_uav = self.uav_model.step(
            state.uav,
            command,
            self.dt_s,
            parameters.uav,
        )
        next_boundary = self.root_boundary.evaluate(
            next_uav,
            self.cable_configuration.rest_lengths_m[0],
        )
        EI, Cb = parameters.cable.tensors(state.cable.positions_m)
        production_cuda = (
            self.device.type == "cuda"
            and state.cable.positions_m.dtype == torch.float32
            and self.root_boundary.pinned_endpoints in ((1, 0), (2, 0))
        )
        if production_cuda:
            batch = state.cable.positions_m.shape[0]

            def population_scalar(value: torch.Tensor, name: str) -> torch.Tensor:
                if value.ndim == 0 or value.numel() == 1:
                    return value.reshape(1).expand(batch)
                if value.shape == (batch,):
                    return value
                raise ValueError(
                    f"Batched {name} must contain one value per simulator row."
                )

            cache_key = (self.device, self.dtype, batch)
            base_constants = self._runtime_constants_cache.get(cache_key)
            if base_constants is None:
                base_constants = self.cable_model.runtime_constants(
                    state.cable.positions_m
                )
                self._runtime_constants_cache[cache_key] = base_constants
            constants = replace(
                base_constants,
                bending_stiffness_n_m2=population_scalar(EI, "EI"),
                bending_damping_n_m2_s=population_scalar(Cb, "Cb"),
            )
            dt = self._runtime_dt_cache.get(cache_key)
            if dt is None:
                dt = torch.full(
                    (batch,),
                    self.dt_s,
                    dtype=self.dtype,
                    device=self.device,
                )
                self._runtime_dt_cache[cache_key] = dt
            next_cable = self.cable_model.step_runtime(
                state.cable,
                next_boundary.prescribed_positions_m,
                dt,
                constants,
                # The fused fixed-count PCG is the low-latency inference
                # implementation.  Gradient execution uses differentiable
                # batched Cholesky on the exact same SPD damping equation;
                # differentiating 32 explicit Krylov recurrences creates an
                # unnecessarily large graph and changes no physical model.
                iterative_damping=not create_graph,
                damping_backend=(
                    "pcg60_reference"
                    if create_graph
                    else VALIDATED_OPTIMIZED_DAMPING_BACKEND
                ),
                pinned_endpoints=self.root_boundary.pinned_endpoints,
                create_graph=create_graph,
                functional_force_autograd=self.functional_force_autograd,
            )
            self.last_dder_execution = (
                "production_cuda_differentiable_cholesky"
                if create_graph
                else "production_cuda_fused"
            )
        else:
            next_cable = self.cable_model.step(
                state.cable,
                next_boundary.prescribed_positions_m,
                self.dt_s,
                create_graph=create_graph,
                bending_stiffness_n_m2=EI,
                bending_damping_n_m2_s=Cb,
                pinned_endpoints=self.root_boundary.pinned_endpoints,
            )
            self.last_dder_execution = "reference_pytorch"
        return SimulatorState(state.time_s + self.dt_s, next_uav, next_cable)

    def forward_backend_audit(self) -> dict[str, object]:
        """Describe and validate the accepted no-gradient production backend."""

        fused = (
            self.device.type == "cuda"
            and self.dtype == torch.float32
            and self.cable_configuration.node_count in (11, 12, 21, 31)
            and self.root_boundary.pinned_endpoints in ((1, 0), (2, 0))
            and self.cable_configuration.constraint_iterations == 4
            and os.environ.get("CABLE_TWIN_FUSED_FIXED_DAMPING", "1") != "0"
            and os.environ.get("CABLE_TWIN_FUSED_FIXED_PROJECTION", "1") != "0"
        )
        return {
            "device": str(self.device),
            "dtype": str(self.dtype),
            "node_count": self.cable_configuration.node_count,
            "substeps": self.cable_configuration.substeps,
            "position_projection_iterations": (
                self.cable_configuration.constraint_iterations
            ),
            "pinned_start_nodes": int(self.root_boundary.pinned_endpoints[0]),
            "damping_backend": (
                VALIDATED_OPTIMIZED_DAMPING_BACKEND
                if fused
                else "reference_pytorch_direct"
            ),
            "projection_backend": (
                "fixed_projection_supported_nodes"
                if fused
                else "reference_pytorch_projection"
            ),
            "production_fused_forward_active": fused,
        }

    def step(
        self,
        command: Command,
        parameters: SimulatorParameters | None = None,
        *,
        create_graph: bool = False,
    ) -> SimulatorState:
        """Propagate the internally held state by one physics frame."""

        self._state = self._propagate(
            self.state,
            command,
            self.parameters if parameters is None else parameters,
            create_graph=create_graph,
        )
        return self._state

    def rollout(
        self,
        initial_state: SimulatorState,
        commands: CommandSequence,
        parameters: SimulatorParameters | None = None,
        *,
        create_graph: bool = True,
    ) -> SimulatorTrajectory:
        """Propagate a command sequence without mutating the held simulator state."""

        if commands.batch_size != initial_state.uav.batch_size:
            raise ValueError("Command batch size must match the initial state.")
        active_parameters = self.parameters if parameters is None else parameters
        states = [initial_state]
        state = initial_state
        for index in range(commands.step_count):
            state = self._propagate(
                state,
                commands.command_at(index),
                active_parameters,
                create_graph=create_graph,
            )
            states.append(state)
        boundaries = [
            self.root_boundary.evaluate(
                item.uav,
                self.cable_configuration.rest_lengths_m[0],
            )
            for item in states
        ]
        return SimulatorTrajectory(
            times_s=torch.arange(
                commands.step_count + 1,
                dtype=self.dtype,
                device=self.device,
            )
            * self.dt_s,
            uav_positions_m=torch.stack([item.uav.position_m for item in states]),
            uav_velocities_m_s=torch.stack([item.uav.velocity_m_s for item in states]),
            uav_orientations_xyzw=torch.stack(
                [item.uav.orientation_xyzw for item in states]
            ),
            uav_angular_velocities_world_rad_s=torch.stack(
                [item.uav.angular_velocity_world_rad_s for item in states]
            ),
            attachment_positions_m=torch.stack(
                [item.attachment_position_m for item in boundaries]
            ),
            attachment_velocities_analytic_m_s=torch.stack(
                [item.attachment_velocity_analytic_m_s for item in boundaries]
            ),
            prescribed_root_tangents_world=(
                None
                if boundaries[0].root_tangent_world is None
                else torch.stack(
                    [item.root_tangent_world for item in boundaries]  # type: ignore[list-item]
                )
            ),
            cable_positions_m=torch.stack(
                [item.cable.positions_m for item in states]
            ),
            cable_velocities_m_s=torch.stack(
                [item.cable.velocities_m_s for item in states]
            ),
            final_state=states[-1],
        )

    def rollout_uav_only(
        self,
        initial_state: UAVState,
        commands: FullStateCommandSequence,
        parameters: SimulatorParameters | None = None,
    ) -> UAVTrajectory:
        """Run the production UAV model without unnecessary DDER propagation.

        This Stage-A path belongs to the common ``CoupledSimulator`` and calls
        the exact same ``uav_model.step`` used by the coupled rollout.  It does
        not create separate fitting dynamics and cannot return cable outputs.
        """

        if not isinstance(self.uav_model, FullStateUAVModel):
            raise TypeError("UAV-only rollout requires FullStateUAVModel.")
        if commands.batch_size != initial_state.batch_size:
            raise ValueError("Command batch size must match the initial UAV state.")
        active_parameters = self.parameters if parameters is None else parameters
        states = [initial_state]
        state = initial_state
        for index in range(commands.step_count):
            state = self.uav_model.step(
                state,
                commands.command_at(index),
                self.dt_s,
                active_parameters.uav,
            )
            states.append(state)
        return UAVTrajectory(
            times_s=torch.arange(
                commands.step_count + 1,
                dtype=initial_state.position_m.dtype,
                device=initial_state.position_m.device,
            ) * self.dt_s,
            positions_m=torch.stack([item.position_m for item in states]),
            velocities_m_s=torch.stack([item.velocity_m_s for item in states]),
            orientations_xyzw=torch.stack([item.orientation_xyzw for item in states]),
            angular_velocities_world_rad_s=torch.stack(
                [item.angular_velocity_world_rad_s for item in states]
            ),
            residual_accelerations_m_s2=(
                torch.stack(
                    [item.residual_acceleration_m_s2 for item in states[1:]]
                )
                if all(
                    item.residual_acceleration_m_s2 is not None
                    for item in states[1:]
                )
                else None
            ),
            final_state=states[-1],
        )
