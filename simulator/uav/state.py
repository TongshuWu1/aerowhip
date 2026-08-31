"""Batched UAV state and command containers used by the common simulator."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ResidualHistoryState:
    """Fixed causal FIFO carried explicitly with every simulated UAV state."""

    features: torch.Tensor

    def __post_init__(self) -> None:
        features = torch.as_tensor(self.features)
        if features.ndim != 3 or features.shape[1:] != (10, 9):
            raise ValueError("Residual history features must have shape Bx10x9.")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("Residual history must be finite.")
        object.__setattr__(self, "features", features)

    @property
    def batch_size(self) -> int:
        return int(self.features.shape[0])

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> "ResidualHistoryState":
        return cls(torch.zeros((batch_size, 10, 9), dtype=dtype, device=device))

    def append(self, current_feature: torch.Tensor) -> "ResidualHistoryState":
        feature = torch.as_tensor(
            current_feature, dtype=self.features.dtype, device=self.features.device
        )
        if feature.shape != (self.batch_size, 9):
            raise ValueError("Current residual feature must have shape Bx9.")
        return ResidualHistoryState(
            torch.cat((self.features[:, 1:], feature[:, None, :]), dim=1)
        )


def _batch_vectors(value: torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[1] != 3:
        raise ValueError(f"{name} must have shape 3 or Bx3.")
    return tensor


def _batch_quaternions(
    value: torch.Tensor,
    name: str,
    reference: torch.Tensor,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[1] != 4:
        raise ValueError(f"{name} must have shape 4 or Bx4.")
    return tensor


@dataclass(frozen=True, slots=True)
class UAVState:
    position_m: torch.Tensor
    velocity_m_s: torch.Tensor
    orientation_xyzw: torch.Tensor | None = None
    angular_velocity_world_rad_s: torch.Tensor | None = None
    residual_history: ResidualHistoryState | None = None
    residual_acceleration_m_s2: torch.Tensor | None = None

    def __post_init__(self) -> None:
        position = _batch_vectors(self.position_m, "position_m")
        velocity = _batch_vectors(
            torch.as_tensor(
                self.velocity_m_s,
                dtype=position.dtype,
                device=position.device,
            ),
            "velocity_m_s",
        )
        if velocity.shape != position.shape:
            raise ValueError("UAV position and velocity must have identical shapes.")
        if self.orientation_xyzw is None:
            orientation = torch.zeros(
                (position.shape[0], 4), dtype=position.dtype, device=position.device
            )
            orientation[:, 3] = 1.0
        else:
            orientation = _batch_quaternions(
                self.orientation_xyzw, "orientation_xyzw", position
            )
        if orientation.shape[0] != position.shape[0]:
            raise ValueError("UAV orientation batch size must match position.")
        if self.angular_velocity_world_rad_s is None:
            angular_velocity = torch.zeros_like(position)
        else:
            angular_velocity = torch.as_tensor(
                self.angular_velocity_world_rad_s,
                dtype=position.dtype,
                device=position.device,
            )
            angular_velocity = _batch_vectors(
                angular_velocity, "angular_velocity_world_rad_s"
            )
        if angular_velocity.shape != position.shape:
            raise ValueError("UAV angular velocity batch size must match position.")
        residual_history = self.residual_history
        if residual_history is not None and residual_history.batch_size != position.shape[0]:
            raise ValueError("Residual history batch size must match UAV state.")
        residual_acceleration = self.residual_acceleration_m_s2
        if residual_acceleration is not None:
            residual_acceleration = _batch_vectors(
                torch.as_tensor(
                    residual_acceleration,
                    dtype=position.dtype,
                    device=position.device,
                ),
                "residual_acceleration_m_s2",
            )
            if residual_acceleration.shape != position.shape:
                raise ValueError("Residual acceleration batch size must match UAV state.")
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "velocity_m_s", velocity)
        object.__setattr__(self, "orientation_xyzw", orientation)
        object.__setattr__(self, "angular_velocity_world_rad_s", angular_velocity)
        object.__setattr__(self, "residual_history", residual_history)
        object.__setattr__(self, "residual_acceleration_m_s2", residual_acceleration)

    @property
    def batch_size(self) -> int:
        return int(self.position_m.shape[0])


@dataclass(frozen=True, slots=True)
class UAVCommand:
    """One prescribed-root command used only by the Milestone-1 debug model."""

    commanded_position_m: torch.Tensor
    commanded_velocity_m_s: torch.Tensor

    def __post_init__(self) -> None:
        position = _batch_vectors(self.commanded_position_m, "commanded_position_m")
        velocity = _batch_vectors(
            torch.as_tensor(
                self.commanded_velocity_m_s,
                dtype=position.dtype,
                device=position.device,
            ),
            "commanded_velocity_m_s",
        )
        if velocity.shape != position.shape:
            raise ValueError("Commanded position and velocity shapes must match.")
        object.__setattr__(self, "commanded_position_m", position)
        object.__setattr__(self, "commanded_velocity_m_s", velocity)


@dataclass(frozen=True, slots=True)
class UAVCommandSequence:
    """Time-major prescribed positions and velocities with shape TxBx3."""

    commanded_positions_m: torch.Tensor
    commanded_velocities_m_s: torch.Tensor

    def __post_init__(self) -> None:
        position = torch.as_tensor(self.commanded_positions_m)
        velocity = torch.as_tensor(self.commanded_velocities_m_s)
        if position.ndim == 2 and position.shape[1] == 3:
            position = position.unsqueeze(1)
        if velocity.ndim == 2 and velocity.shape[1] == 3:
            velocity = velocity.unsqueeze(1)
        if position.ndim != 3 or position.shape[2] != 3:
            raise ValueError("commanded_positions_m must have shape Tx3 or TxBx3.")
        if velocity.shape != position.shape:
            raise ValueError("Command position and velocity sequences must match.")
        if position.shape[0] < 1:
            raise ValueError("A rollout requires at least one command frame.")
        object.__setattr__(self, "commanded_positions_m", position)
        object.__setattr__(self, "commanded_velocities_m_s", velocity)

    @property
    def step_count(self) -> int:
        return int(self.commanded_positions_m.shape[0])

    @property
    def batch_size(self) -> int:
        return int(self.commanded_positions_m.shape[1])

    def command_at(self, index: int) -> UAVCommand:
        return UAVCommand(
            self.commanded_positions_m[index],
            self.commanded_velocities_m_s[index],
        )


@dataclass(frozen=True, slots=True)
class FullStateCommand:
    """One recorded FullState command in SI units.

    The command quaternion carries yaw only and angular rate is body-frame
    rad/s under the standard cmdFullState contract.  The current recordings
    have identically zero angular-rate commands, so this contract clarification
    does not require changing the simulator's world-frame angular-velocity
    state convention.
    """

    position_m: torch.Tensor
    velocity_m_s: torch.Tensor
    acceleration_m_s2: torch.Tensor
    orientation_xyzw: torch.Tensor
    angular_velocity_body_rad_s: torch.Tensor

    def __post_init__(self) -> None:
        position = _batch_vectors(self.position_m, "position_m")
        velocity = torch.as_tensor(
            self.velocity_m_s, dtype=position.dtype, device=position.device
        )
        acceleration = torch.as_tensor(
            self.acceleration_m_s2, dtype=position.dtype, device=position.device
        )
        angular_velocity = torch.as_tensor(
            self.angular_velocity_body_rad_s,
            dtype=position.dtype,
            device=position.device,
        )
        velocity = _batch_vectors(velocity, "velocity_m_s")
        acceleration = _batch_vectors(acceleration, "acceleration_m_s2")
        angular_velocity = _batch_vectors(
            angular_velocity, "angular_velocity_body_rad_s"
        )
        orientation = _batch_quaternions(
            self.orientation_xyzw, "orientation_xyzw", position
        )
        if not (
            velocity.shape == acceleration.shape == angular_velocity.shape == position.shape
            and orientation.shape[0] == position.shape[0]
        ):
            raise ValueError("All FullState command batch sizes must match.")
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "velocity_m_s", velocity)
        object.__setattr__(self, "acceleration_m_s2", acceleration)
        object.__setattr__(self, "orientation_xyzw", orientation)
        object.__setattr__(
            self, "angular_velocity_body_rad_s", angular_velocity
        )


@dataclass(frozen=True, slots=True)
class FullStateCommandSequence:
    """Time-major FullState commands with shapes TxBx3 and TxBx4."""

    positions_m: torch.Tensor
    velocities_m_s: torch.Tensor
    accelerations_m_s2: torch.Tensor
    orientations_xyzw: torch.Tensor
    angular_velocities_body_rad_s: torch.Tensor

    def __post_init__(self) -> None:
        position = torch.as_tensor(self.positions_m)
        if position.ndim == 2 and position.shape[1] == 3:
            position = position.unsqueeze(1)
        if position.ndim != 3 or position.shape[2] != 3 or position.shape[0] < 1:
            raise ValueError("positions_m must have shape Tx3 or TxBx3 with T >= 1.")

        def vector_sequence(value: torch.Tensor, name: str) -> torch.Tensor:
            tensor = torch.as_tensor(value, dtype=position.dtype, device=position.device)
            if tensor.ndim == 2 and tensor.shape[1] == 3:
                tensor = tensor.unsqueeze(1)
            if tensor.shape != position.shape:
                raise ValueError(f"{name} must have the same TxBx3 shape as positions_m.")
            return tensor

        velocity = vector_sequence(self.velocities_m_s, "velocities_m_s")
        acceleration = vector_sequence(self.accelerations_m_s2, "accelerations_m_s2")
        angular_velocity = vector_sequence(
            self.angular_velocities_body_rad_s,
            "angular_velocities_body_rad_s",
        )
        orientation = torch.as_tensor(
            self.orientations_xyzw, dtype=position.dtype, device=position.device
        )
        if orientation.ndim == 2 and orientation.shape[1] == 4:
            orientation = orientation.unsqueeze(1)
        if orientation.shape != (*position.shape[:2], 4):
            raise ValueError("orientations_xyzw must have shape Tx4 or TxBx4.")
        object.__setattr__(self, "positions_m", position)
        object.__setattr__(self, "velocities_m_s", velocity)
        object.__setattr__(self, "accelerations_m_s2", acceleration)
        object.__setattr__(self, "orientations_xyzw", orientation)
        object.__setattr__(
            self, "angular_velocities_body_rad_s", angular_velocity
        )

    @property
    def step_count(self) -> int:
        return int(self.positions_m.shape[0])

    @property
    def batch_size(self) -> int:
        return int(self.positions_m.shape[1])

    def command_at(self, index: int) -> FullStateCommand:
        return FullStateCommand(
            self.positions_m[index],
            self.velocities_m_s[index],
            self.accelerations_m_s2[index],
            self.orientations_xyzw[index],
            self.angular_velocities_body_rad_s[index],
        )
