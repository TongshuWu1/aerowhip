"""Deterministic virtual OptiTrack projection.

This module deliberately contains no dynamics and no noise model.  It maps the
latent simulator state onto exactly the pose and moving cable-marker positions
planned for the physical OptiTrack experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

from ..state import SimulatorState, SimulatorTrajectory


OPTITRACK_CABLE_MARKER_NODE_INDICES: Final[tuple[int, ...]] = (
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
)


@dataclass(frozen=True, slots=True)
class OptiTrackObservation:
    """Noise-free pose and moving-marker observation with arbitrary leading axes.

    A single batched state uses leading shape ``[B]``.  A batched trajectory
    uses leading shape ``[T, B]``.  No velocity or angular-velocity channel is
    part of this experimental observation contract.
    """

    uav_position_m: torch.Tensor
    uav_orientation_xyzw: torch.Tensor
    cable_marker_positions_m: torch.Tensor

    def __post_init__(self) -> None:
        position = torch.as_tensor(self.uav_position_m)
        orientation = torch.as_tensor(
            self.uav_orientation_xyzw,
            dtype=position.dtype,
            device=position.device,
        )
        markers = torch.as_tensor(
            self.cable_marker_positions_m,
            dtype=position.dtype,
            device=position.device,
        )
        if position.ndim < 2 or position.shape[-1] != 3:
            raise ValueError("uav_position_m must have shape [..., 3].")
        if orientation.shape != (*position.shape[:-1], 4):
            raise ValueError(
                "uav_orientation_xyzw must share leading axes with UAV position "
                "and have shape [..., 4]."
            )
        if markers.shape != (*position.shape[:-1], 10, 3):
            raise ValueError(
                "cable_marker_positions_m must share leading axes with UAV "
                "position and have shape [..., 10, 3]."
            )
        object.__setattr__(self, "uav_position_m", position)
        object.__setattr__(self, "uav_orientation_xyzw", orientation)
        object.__setattr__(self, "cable_marker_positions_m", markers)


def _select_moving_markers(cable_positions_m: torch.Tensor) -> torch.Tensor:
    if cable_positions_m.ndim < 3 or cable_positions_m.shape[-1] != 3:
        raise ValueError("Cable positions must have shape [..., N, 3].")
    required_nodes = OPTITRACK_CABLE_MARKER_NODE_INDICES[-1] + 1
    if cable_positions_m.shape[-2] < required_nodes:
        raise ValueError(
            f"OptiTrack mapping requires at least {required_nodes} DDER nodes."
        )
    indices = torch.tensor(
        OPTITRACK_CABLE_MARKER_NODE_INDICES,
        dtype=torch.long,
        device=cable_positions_m.device,
    )
    # index_select creates an observation-owned tensor rather than exposing a
    # mutable view of latent simulator storage.
    return torch.index_select(cable_positions_m, -2, indices)


def observe_simulator_state(state: SimulatorState) -> OptiTrackObservation:
    """Project one simulator state onto the deterministic OptiTrack contract."""

    return OptiTrackObservation(
        uav_position_m=state.uav.position_m.clone(),
        uav_orientation_xyzw=state.uav.orientation_xyzw.clone(),
        cable_marker_positions_m=_select_moving_markers(state.cable.positions_m),
    )


def observe_simulator_trajectory(
    trajectory: SimulatorTrajectory,
) -> OptiTrackObservation:
    """Project a time-major simulator trajectory without changing its leading axes."""

    return OptiTrackObservation(
        uav_position_m=trajectory.uav_positions_m.clone(),
        uav_orientation_xyzw=trajectory.uav_orientations_xyzw.clone(),
        cable_marker_positions_m=_select_moving_markers(
            trajectory.cable_positions_m
        ),
    )
