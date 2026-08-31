"""Compact immutable bridge from simulator state to the render thread."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch

from ..simulator import Command, CoupledSimulator
from ..state import SimulatorState
from ..uav.state import FullStateCommand


def _immutable_array(values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True).reshape(shape)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class VisualizationSnapshot:
    """Graphics-only state; no solver workspace or latent velocity arrays."""

    time_s: float
    cable_positions_m: np.ndarray
    uav_position_m: np.ndarray
    uav_orientation_xyzw: np.ndarray
    attachment_position_m: np.ndarray
    commanded_uav_position_m: np.ndarray | None
    commanded_uav_orientation_xyzw: np.ndarray | None
    tip_speed_m_s: float
    maximum_edge_error_m: float
    boundary_name: str
    dder_execution: str
    transfer_ms: float


def create_visualization_snapshot(
    simulator: CoupledSimulator,
    state: SimulatorState,
    command: Command | None,
) -> VisualizationSnapshot:
    """Materialize the latest compact graphics snapshot in one device transfer."""

    started = time.perf_counter()
    cable = state.cable.positions_m[0]
    boundary = simulator.root_boundary.evaluate(
        state.uav,
        simulator.cable_configuration.rest_lengths_m[0],
    )
    has_fullstate_command = isinstance(command, FullStateCommand)
    command_position = (
        state.uav.position_m[0]
        if not has_fullstate_command
        else command.position_m[0].to(dtype=cable.dtype, device=cable.device)
    )
    command_orientation = (
        state.uav.orientation_xyzw[0]
        if not has_fullstate_command
        else command.orientation_xyzw[0].to(dtype=cable.dtype, device=cable.device)
    )
    tip_speed = torch.linalg.vector_norm(
        state.cable.velocities_m_s[0, -1]
    ).reshape(1)
    edge_error = simulator.cable_model.maximum_segment_error_m(
        state.cable.positions_m
    ).reshape(1)
    packed = torch.cat(
        (
            cable.reshape(-1),
            state.uav.position_m[0].reshape(-1),
            state.uav.orientation_xyzw[0].reshape(-1),
            boundary.attachment_position_m[0].reshape(-1),
            command_position.reshape(-1),
            command_orientation.reshape(-1),
            tip_speed,
            edge_error,
        )
    ).detach().cpu().numpy()

    node_values = 3 * cable.shape[0]
    offset = 0
    cable_values = _immutable_array(packed[offset : offset + node_values], (-1, 3))
    offset += node_values
    uav_position = _immutable_array(packed[offset : offset + 3], (3,))
    offset += 3
    uav_orientation = _immutable_array(packed[offset : offset + 4], (4,))
    offset += 4
    attachment = _immutable_array(packed[offset : offset + 3], (3,))
    offset += 3
    commanded_position = _immutable_array(packed[offset : offset + 3], (3,))
    offset += 3
    commanded_orientation = _immutable_array(packed[offset : offset + 4], (4,))
    offset += 4
    transfer_ms = 1000.0 * (time.perf_counter() - started)
    return VisualizationSnapshot(
        time_s=float(state.time_s),
        cable_positions_m=cable_values,
        uav_position_m=uav_position,
        uav_orientation_xyzw=uav_orientation,
        attachment_position_m=attachment,
        commanded_uav_position_m=(
            commanded_position if has_fullstate_command else None
        ),
        commanded_uav_orientation_xyzw=(
            commanded_orientation if has_fullstate_command else None
        ),
        tip_speed_m_s=float(packed[offset]),
        maximum_edge_error_m=float(packed[offset + 1]),
        boundary_name=type(simulator.root_boundary).__name__,
        dder_execution=simulator.last_dder_execution,
        transfer_ms=transfer_ms,
    )
