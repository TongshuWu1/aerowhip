"""Prescribed-root and effective FullState UAV response models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from ..parameters import UAVResponseParameters
from .quaternion import (
    desired_orientation_error_body,
    desired_rotation_from_acceleration_and_yaw,
    integrate_world_angular_velocity,
    normalize_quaternion_xyzw,
    quaternion_to_rotation_matrix_xyzw,
    yaw_from_quaternion_xyzw,
)
from .residual import CausalTranslationalResidual, residual_feature
from .state import FullStateCommand, ResidualHistoryState, UAVCommand, UAVState


Command = UAVCommand | FullStateCommand

class UAVModel(ABC):
    """Interface reserved for the future identified UAV command-response model."""

    @abstractmethod
    def step(
        self,
        state: UAVState,
        command: Command,
        dt_s: float,
        parameters: UAVResponseParameters,
    ) -> UAVState:
        raise NotImplementedError


class PrescribedRootModel(UAVModel):
    """Debug-only model that makes commanded root motion exact.

    This is deliberately not the research UAV model. It only decouples DDER
    validation from the future command-response identification problem.
    """

    def step(
        self,
        state: UAVState,
        command: Command,
        dt_s: float,
        parameters: UAVResponseParameters,
    ) -> UAVState:
        del dt_s, parameters
        if not isinstance(command, UAVCommand):
            raise TypeError("PrescribedRootModel requires UAVCommand.")
        if command.commanded_position_m.shape != state.position_m.shape:
            raise ValueError("Prescribed command batch size must match the UAV state.")
        position = command.commanded_position_m.to(state.position_m)
        velocity = command.commanded_velocity_m_s.to(state.velocity_m_s)
        return UAVState(
            position,
            velocity,
            state.orientation_xyzw,
            state.angular_velocity_world_rad_s,
            state.residual_history,
            state.residual_acceleration_m_s2,
        )


class FullStateUAVModel(UAVModel):
    """Minimal stock-Crazyswarm2/Mellinger effective response.

    The recording supplies world-frame position, velocity and acceleration,
    yaw heading, and body-frame angular-rate feedforward.  The angular-rate
    field is zero throughout the current Stage A data.
    Roll/pitch are constructed from the
    translational desired acceleration plus gravity, rather than read from the
    yaw-only command quaternion.  State orientation is body-to-world and state
    angular velocity remains world-frame for rigid attachment kinematics; only
    the command is converted through the body-frame response equation.
    """

    def __init__(
        self,
        gravity_m_s2: float = 9.80665,
        *,
        residual_model: CausalTranslationalResidual | None = None,
    ) -> None:
        if gravity_m_s2 <= 0.0:
            raise ValueError("gravity_m_s2 must be positive.")
        self.gravity_m_s2 = float(gravity_m_s2)
        self.residual_model = residual_model
        self._fixed_evaluation_batch_size: int | None = None

    @property
    def model_version(self) -> str:
        return "aerial_cable_attitude_coupled_v1"

    @property
    def residual_enabled(self) -> bool:
        return self.residual_model is not None

    @property
    def fixed_evaluation_batch_size(self) -> int | None:
        """Return the optional fixed CUDA batch used for planning numerics."""

        return self._fixed_evaluation_batch_size

    def set_fixed_evaluation_batch_size(self, batch_size: int | None) -> None:
        """Canonicalize no-gradient planning arithmetic without changing equations.

        CUDA kernels can select different floating-point reduction paths for
        B=1 and B=2048.  The aggressive cable rollout amplifies the resulting
        sub-micro UAV boundary differences.  Planning can therefore request a
        fixed UAV evaluation shape; smaller batches are padded with copies of
        their final row and sliced back immediately.  The cable batch and all
        physical state equations remain unchanged.
        """

        if batch_size is not None and batch_size < 1:
            raise ValueError("Fixed UAV evaluation batch size must be positive.")
        self._fixed_evaluation_batch_size = batch_size

    def initialize_residual_state(self, state: UAVState) -> UAVState:
        """Attach deterministic zero history when no measured history is supplied.

        Identification replaces this reset-only fallback with real causal
        pre-window history.  The fallback keeps the public reset API usable for
        simulation/debug rollouts that have no physical prehistory.
        """

        if self.residual_model is None or state.residual_history is not None:
            return state
        return UAVState(
            state.position_m,
            state.velocity_m_s,
            state.orientation_xyzw,
            state.angular_velocity_world_rad_s,
            ResidualHistoryState.zeros(
                state.batch_size,
                dtype=state.position_m.dtype,
                device=state.position_m.device,
            ),
            torch.zeros_like(state.position_m),
        )

    @staticmethod
    def _pad_rows(value: torch.Tensor, target_batch: int) -> torch.Tensor:
        if value.shape[0] > target_batch:
            raise ValueError("UAV batch exceeds the fixed evaluation batch size.")
        if value.shape[0] == target_batch:
            return value
        padding = value[-1:].expand(
            (target_batch - value.shape[0],) + value.shape[1:]
        )
        return torch.cat((value, padding), dim=0)

    @staticmethod
    def _slice_state(state: UAVState, batch_size: int) -> UAVState:
        def sliced(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value[:batch_size].clone()

        return UAVState(
            sliced(state.position_m),
            sliced(state.velocity_m_s),
            sliced(state.orientation_xyzw),
            sliced(state.angular_velocity_world_rad_s),
            None
            if state.residual_history is None
            else ResidualHistoryState(
                state.residual_history.features[:batch_size].clone()
            ),
            sliced(state.residual_acceleration_m_s2),
        )

    def step(
        self,
        state: UAVState,
        command: Command,
        dt_s: float,
        parameters: UAVResponseParameters,
    ) -> UAVState:
        target_batch = self._fixed_evaluation_batch_size
        if (
            target_batch is None
            or state.batch_size == target_batch
            or not isinstance(command, FullStateCommand)
        ):
            return self._step_impl(state, command, dt_s, parameters)
        if state.batch_size > target_batch:
            raise ValueError(
                "Planning UAV batch exceeds the configured fixed evaluation size."
            )

        padded_state = UAVState(
            self._pad_rows(state.position_m, target_batch),
            self._pad_rows(state.velocity_m_s, target_batch),
            self._pad_rows(state.orientation_xyzw, target_batch),
            self._pad_rows(state.angular_velocity_world_rad_s, target_batch),
            None
            if state.residual_history is None
            else ResidualHistoryState(
                self._pad_rows(state.residual_history.features, target_batch)
            ),
            None
            if state.residual_acceleration_m_s2 is None
            else self._pad_rows(state.residual_acceleration_m_s2, target_batch),
        )
        padded_command = FullStateCommand(
            self._pad_rows(command.position_m, target_batch),
            self._pad_rows(command.velocity_m_s, target_batch),
            self._pad_rows(command.acceleration_m_s2, target_batch),
            self._pad_rows(command.orientation_xyzw, target_batch),
            self._pad_rows(command.angular_velocity_body_rad_s, target_batch),
        )
        padded_next = self._step_impl(
            padded_state, padded_command, dt_s, parameters
        )
        return self._slice_state(padded_next, state.batch_size)

    def _step_impl(
        self,
        state: UAVState,
        command: Command,
        dt_s: float,
        parameters: UAVResponseParameters,
    ) -> UAVState:
        if not isinstance(command, FullStateCommand):
            raise TypeError("FullStateUAVModel requires FullStateCommand.")
        if command.position_m.shape != state.position_m.shape:
            raise ValueError("FullState command batch size must match UAV state.")
        dt = torch.as_tensor(
            dt_s, dtype=state.position_m.dtype, device=state.position_m.device
        )
        K_p, K_v, k_a, K_R, K_omega = parameters.tensors(state.position_m)
        command_position = command.position_m.to(state.position_m)
        command_velocity = command.velocity_m_s.to(state.velocity_m_s)
        command_acceleration = command.acceleration_m_s2.to(state.velocity_m_s)
        command_heading = command.orientation_xyzw.to(state.orientation_xyzw)
        command_angular_velocity_body = command.angular_velocity_body_rad_s.to(
            state.angular_velocity_world_rad_s
        )
        controller_acceleration = (
            k_a * command_acceleration
            + K_p * (command_position - state.position_m)
            + K_v * (command_velocity - state.velocity_m_s)
        )

        # One causal semi-implicit attitude update is shared by both the UAV
        # translation and the rigid cable clamp.  The command quaternion is
        # yaw provenance only; roll/pitch come from controller acceleration
        # demand plus gravity.
        current_orientation = normalize_quaternion_xyzw(state.orientation_xyzw)
        current_rotation = quaternion_to_rotation_matrix_xyzw(current_orientation)
        desired_rotation = desired_rotation_from_acceleration_and_yaw(
            controller_acceleration,
            yaw_from_quaternion_xyzw(command_heading),
            self.gravity_m_s2,
        )
        orientation_error_body = desired_orientation_error_body(
            desired_rotation, current_rotation
        )
        current_angular_velocity_body = torch.matmul(
            current_rotation.transpose(-1, -2),
            state.angular_velocity_world_rad_s.unsqueeze(-1),
        ).squeeze(-1)
        angular_acceleration_body = (
            K_R * orientation_error_body
            + K_omega
            * (
                command_angular_velocity_body
                - current_angular_velocity_body
            )
        )
        next_angular_velocity_body = (
            current_angular_velocity_body + dt * angular_acceleration_body
        )
        integration_angular_velocity_world = torch.matmul(
            current_rotation, next_angular_velocity_body.unsqueeze(-1)
        ).squeeze(-1)
        next_orientation = integrate_world_angular_velocity(
            current_orientation,
            integration_angular_velocity_world,
            dt,
        )
        next_rotation = quaternion_to_rotation_matrix_xyzw(next_orientation)
        next_angular_velocity_world = torch.matmul(
            next_rotation, next_angular_velocity_body.unsqueeze(-1)
        ).squeeze(-1)

        residual_history = state.residual_history
        residual_acceleration: torch.Tensor | None = None
        if self.residual_model is not None:
            if residual_history is None:
                raise ValueError(
                    "Residual-enabled UAV state requires explicit causal history."
                )
            current_feature = residual_feature(
                command_position,
                command_velocity,
                command_acceleration,
                state.position_m,
                state.velocity_m_s,
            )
            residual_history = residual_history.append(current_feature)
            residual_acceleration = self.residual_model(residual_history.features)

        gravity_axis = torch.zeros_like(controller_acceleration)
        gravity_axis[..., 2] = self.gravity_m_s2
        desired_specific_force = controller_acceleration + gravity_axis
        thrust_acceleration = torch.linalg.vector_norm(
            desired_specific_force, dim=-1, keepdim=True
        )
        actual_body_z_world = next_rotation[..., :, 2]
        acceleration = thrust_acceleration * actual_body_z_world - gravity_axis
        if residual_acceleration is not None:
            # The causal residual alters realized translation only.  It never
            # changes controller demand, desired attitude, or angular response.
            acceleration = acceleration + residual_acceleration

        next_velocity = state.velocity_m_s + dt * acceleration
        next_position = state.position_m + dt * next_velocity
        return UAVState(
            next_position,
            next_velocity,
            next_orientation,
            next_angular_velocity_world,
            residual_history,
            residual_acceleration,
        )
