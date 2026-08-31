"""Causal UAV and measured-site-conditioned clamped DDER initialization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from simulator.coupling.attachment import rigid_attachment_state
from simulator.parameters import SimulatorSettings
from simulator.simulator import CoupledSimulator
from simulator.state import SimulatorState
from simulator.uav.model import FullStateUAVModel
from simulator.uav.quaternion import quaternion_conjugate_xyzw, quaternion_multiply_xyzw
from simulator.uav.state import UAVState

from .config import FitConfiguration
from .dataset import ProcessedTake
from .windows import PredictionWindow


@dataclass(frozen=True, slots=True)
class InitializedWindow:
    state: SimulatorState
    marker_initialization_rmse_m: float


def causal_polynomial_derivative(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    time = np.asarray(time_s, dtype=np.float64)
    if len(time) < 3 or not np.all(np.diff(time) > 0.0):
        raise ValueError("Causal derivative requires at least three ordered samples.")
    relative = time - time[-1]
    degree = min(2, len(time) - 1)
    design = np.column_stack([relative**power for power in range(degree + 1)])
    coefficients, *_ = np.linalg.lstsq(design, values.reshape(len(time), -1), rcond=None)
    return coefficients[1].reshape(values.shape[1:])


def causal_world_angular_velocity(quaternion_xyzw: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    quaternion = torch.as_tensor(quaternion_xyzw, dtype=torch.float64)
    previous = quaternion[-2:-1]
    current = quaternion[-1:]
    delta = quaternion_multiply_xyzw(current, quaternion_conjugate_xyzw(previous))[0]
    if float(delta[3]) < 0.0:
        delta = -delta
    vector_norm = torch.linalg.vector_norm(delta[:3])
    angle = 2.0 * torch.atan2(vector_norm, torch.clamp(delta[3], min=1.0e-15))
    axis = delta[:3] / torch.clamp(vector_norm, min=1.0e-15)
    return (axis * angle / float(time_s[-1] - time_s[-2])).numpy()


def initialize_uav_state(
    take: ProcessedTake,
    window: PredictionWindow,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> UAVState:
    """Build the causal measured UAV state shared by Stage A and joint fitting.

    Only samples at or before the prediction-window start are used.  Keeping
    this in the existing initialization module prevents the UAV-only Stage A
    path from acquiring a second set of fitting dynamics or initial-state
    semantics.
    """

    arrays = take.arrays
    history = slice(window.history_start_index, window.start_index + 1)
    history_time = arrays["time_s"][history]
    return UAVState(
        torch.as_tensor(
            arrays["uav_position_m"][window.start_index],
            dtype=dtype,
            device=device,
        ),
        torch.as_tensor(
            causal_polynomial_derivative(
                arrays["uav_position_m"][history], history_time
            ),
            dtype=dtype,
            device=device,
        ),
        torch.as_tensor(
            arrays["uav_orientation_xyzw"][window.start_index],
            dtype=dtype,
            device=device,
        ),
        torch.as_tensor(
            causal_world_angular_velocity(
                arrays["uav_orientation_xyzw"][history], history_time
            ),
            dtype=dtype,
            device=device,
        ),
    )


def initialize_window(
    take: ProcessedTake,
    window: PredictionWindow,
    config: FitConfiguration,
    settings: SimulatorSettings,
    simulator: CoupledSimulator,
) -> InitializedWindow:
    arrays = take.arrays
    history = slice(window.history_start_index, window.start_index + 1)
    history_time = arrays["time_s"][history]
    device, dtype = simulator.device, simulator.dtype
    uav = initialize_uav_state(
        take,
        window,
        device=device,
        dtype=dtype,
    )
    boundary = simulator.root_boundary.evaluate(
        uav, settings.cable_configuration.rest_lengths_m[0]
    )
    markers = arrays["cable_marker_positions_m"][window.start_index]
    marker_velocity = causal_polynomial_derivative(
        arrays["cable_marker_positions_m"][history], history_time
    )
    positions = torch.empty(
        (1, settings.cable_configuration.node_count, 3), dtype=dtype, device=device
    )
    velocities = torch.empty_like(positions)
    positions[:, :2] = boundary.prescribed_positions_m
    velocities[:, :2] = boundary.analytic_velocities_m_s
    marker_nodes = settings.cable_configuration.marker_node_indices[1:]
    for marker_index, node in enumerate(marker_nodes):
        positions[:, node] = torch.as_tensor(markers[marker_index], dtype=dtype, device=device)
        velocities[:, node] = torch.as_tensor(marker_velocity[marker_index], dtype=dtype, device=device)
    material_coordinates = np.concatenate(
        ([0.0], np.cumsum(settings.cable_configuration.rest_lengths_m))
    )
    measured_sites = settings.cable_configuration.marker_node_indices
    pinned_start = int(simulator.root_boundary.pinned_endpoints[0])
    for left, right in zip(measured_sites[:-1], measured_sites[1:], strict=True):
        interval = material_coordinates[right] - material_coordinates[left]
        for node in range(left + 1, right):
            # A rigid clamp owns node 1. Any remaining latent nodes are smooth
            # material-coordinate interpolants between adjacent measured sites.
            if node < pinned_start:
                continue
            fraction = (material_coordinates[node] - material_coordinates[left]) / interval
            positions[:, node] = (
                (1.0 - fraction) * positions[:, left]
                + fraction * positions[:, right]
            )
            velocities[:, node] = (
                (1.0 - fraction) * velocities[:, left]
                + fraction * velocities[:, right]
            )
    # Adapt the validated measured-site interpolation by projecting the smooth
    # seed with the actual production DDER constraint operators and fixed two-
    # node clamp. Measured positions are evidence, not impossible hard rods.
    for _ in range(8):
        positions = simulator.cable_model.project_lengths(
            positions,
            boundary.prescribed_positions_m,
            pinned_endpoints=simulator.root_boundary.pinned_endpoints,
        )
    velocities = simulator.cable_model.project_velocities(
        positions,
        velocities,
        boundary.analytic_velocities_m_s,
        pinned_endpoints=simulator.root_boundary.pinned_endpoints,
    )
    predicted_markers = positions[0, torch.as_tensor(marker_nodes, device=device)]
    marker_tensor = torch.as_tensor(markers, dtype=dtype, device=device)
    rmse = float(
        torch.sqrt(torch.mean(torch.sum((predicted_markers - marker_tensor).square(), dim=1))).detach().cpu()
    )
    if rmse > config.maximum_initialization_rmse_m:
        raise ValueError(
            f"initialization_rmse={rmse:.6g} exceeds {config.maximum_initialization_rmse_m:.6g} m"
        )
    cable = simulator.cable_model.initial_state(positions, velocities)
    return InitializedWindow(SimulatorState(float(window.start_s), uav, cable), rmse)
