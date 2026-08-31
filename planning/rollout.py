"""Batched full-horizon rollouts through the immutable production simulator."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from simulator.cable.dder import DderState
from simulator.simulator import CoupledSimulator
from simulator.state import SimulatorState
from simulator.uav.state import (
    FullStateCommand,
    ResidualHistoryState,
    UAVState,
)

from .command_parameterization import (
    FullStateCommandTrajectory,
    acceleration_knots_to_fullstate,
)
from .metrics import PopulationRolloutResult, WhipMetricAccumulator
from .task import CanonicalWhipTask


@dataclass(frozen=True, slots=True)
class DeterministicReplay:
    command: FullStateCommandTrajectory
    metrics: PopulationRolloutResult
    times_s: torch.Tensor
    uav_positions_m: torch.Tensor
    uav_velocities_m_s: torch.Tensor
    uav_orientations_xyzw: torch.Tensor
    uav_angular_velocities_world_rad_s: torch.Tensor
    cable_positions_m: torch.Tensor
    cable_velocities_m_s: torch.Tensor
    residual_accelerations_m_s2: torch.Tensor


def _repeat(tensor: torch.Tensor | None, batch_size: int) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.shape[0] == batch_size:
        return tensor.clone()
    if tensor.shape[0] == 1:
        return tensor.repeat((batch_size,) + (1,) * (tensor.ndim - 1)).clone()
    raise ValueError("Source state batch must be one or match the requested batch size.")


def clone_state_batch(state: SimulatorState, batch_size: int) -> SimulatorState:
    """Deep-copy every simulator component into a repeated or matched batch."""

    if batch_size < 1 or state.uav.batch_size not in (1, batch_size):
        raise ValueError(
            "State cloning requires a positive batch and either one source row "
            "or one source row per requested output row."
        )
    residual_history = (
        None
        if state.uav.residual_history is None
        else ResidualHistoryState(_repeat(state.uav.residual_history.features, batch_size))
    )
    uav = UAVState(
        _repeat(state.uav.position_m, batch_size),
        _repeat(state.uav.velocity_m_s, batch_size),
        _repeat(state.uav.orientation_xyzw, batch_size),
        _repeat(state.uav.angular_velocity_world_rad_s, batch_size),
        residual_history,
        _repeat(state.uav.residual_acceleration_m_s2, batch_size),
    )
    cable = DderState(
        _repeat(state.cable.positions_m, batch_size),
        _repeat(state.cable.velocities_m_s, batch_size),
        _repeat(state.cable.endpoint_orientations, batch_size),
        _repeat(state.cable.endpoint_twist_rad, batch_size),
    )
    # Planning time is relative to the post-hover boundary, while every
    # physical tensor is an exact clone of that final pre-roll state.
    return SimulatorState(0.0, uav, cable)


def hover_preroll(
    simulator: CoupledSimulator,
    task: CanonicalWhipTask,
) -> SimulatorState:
    """Create a settled, causally populated production state once."""

    device, dtype = simulator.device, simulator.dtype
    position = torch.tensor(
        task.initial_uav_position_m, dtype=dtype, device=device
    ).view(1, 3)
    velocity = torch.tensor(
        task.initial_uav_velocity_m_s, dtype=dtype, device=device
    ).view(1, 3)
    half_yaw = 0.5 * task.initial_yaw_rad
    orientation = torch.tensor(
        [[0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)]],
        dtype=dtype,
        device=device,
    )
    simulator.reset(position, velocity, orientation, torch.zeros_like(position))
    command = FullStateCommand(
        position,
        torch.zeros_like(position),
        torch.zeros_like(position),
        orientation,
        torch.zeros_like(position),
    )
    steps = int(round(task.hover_preroll_s / simulator.dt_s))
    with torch.no_grad():
        for _ in range(steps):
            simulator.step(command, create_graph=False)
    if simulator.device.type == "cuda":
        torch.cuda.synchronize(simulator.device)
    state = clone_state_batch(simulator.state, 1)
    if state.uav.residual_history is None:
        raise RuntimeError("Production pre-roll did not create residual history.")
    if not bool(torch.isfinite(state.uav.residual_history.features).all().detach().cpu()):
        raise RuntimeError("Production pre-roll residual history is non-finite.")
    return state


def _commands_from_knots(
    knots_m_s2: torch.Tensor,
    task: CanonicalWhipTask,
    dt_s: float,
) -> FullStateCommandTrajectory:
    return acceleration_knots_to_fullstate(
        knots_m_s2,
        initial_position_m=torch.tensor(
            task.initial_uav_position_m,
            dtype=knots_m_s2.dtype,
            device=knots_m_s2.device,
        ),
        initial_velocity_m_s=torch.zeros(
            3, dtype=knots_m_s2.dtype, device=knots_m_s2.device
        ),
        yaw_rad=task.initial_yaw_rad,
        horizon_s=task.mppi.horizon_s,
        dt_s=dt_s,
    )


def run_population_rollout(
    simulator: CoupledSimulator,
    initial_state: SimulatorState,
    knots_m_s2: torch.Tensor,
    task: CanonicalWhipTask,
) -> PopulationRolloutResult:
    """Advance all MPPI candidates together while retaining only online metrics."""

    knots = torch.as_tensor(knots_m_s2, dtype=simulator.dtype, device=simulator.device)
    if knots.ndim == 2:
        knots = knots.unsqueeze(0)
    batch = int(knots.shape[0])
    command = _commands_from_knots(knots, task, simulator.dt_s)
    sequence = command.simulator_sequence()
    state = clone_state_batch(initial_state, batch)
    metrics = WhipMetricAccumulator(
        task, batch_size=batch, device=simulator.device, dtype=simulator.dtype
    )
    metrics.observe(
        0.0,
        uav_position_m=state.uav.position_m,
        uav_velocity_m_s=state.uav.velocity_m_s,
        cable_positions_m=state.cable.positions_m,
        cable_velocities_m_s=state.cable.velocities_m_s,
    )
    with torch.no_grad():
        for index in range(sequence.step_count):
            # This is the same non-mutating state transition used internally by
            # CoupledSimulator.rollout. It avoids materializing a 2048-row
            # population trajectory while preserving production physics.
            state = simulator._propagate(  # noqa: SLF001
                state,
                sequence.command_at(index),
                simulator.parameters,
                create_graph=False,
            )
            metrics.observe(
                (index + 1) * simulator.dt_s,
                uav_position_m=state.uav.position_m,
                uav_velocity_m_s=state.uav.velocity_m_s,
                cable_positions_m=state.cable.positions_m,
                cable_velocities_m_s=state.cable.velocities_m_s,
            )
    return metrics.finalize(
        final_uav_position_m=state.uav.position_m,
        final_uav_velocity_m_s=state.uav.velocity_m_s,
        final_cable_positions_m=state.cable.positions_m,
        command=command,
        knots_m_s2=knots,
    )


def replay_selected_trajectory(
    simulator: CoupledSimulator,
    initial_state: SimulatorState,
    knots_m_s2: torch.Tensor,
    task: CanonicalWhipTask,
) -> DeterministicReplay:
    """Replay one selected command and retain its complete authoritative history."""

    knots = torch.as_tensor(knots_m_s2, dtype=simulator.dtype, device=simulator.device)
    if knots.ndim == 2:
        knots = knots.unsqueeze(0)
    if knots.shape[0] != 1:
        raise ValueError("Deterministic selected replay requires exactly one row.")
    command = _commands_from_knots(knots, task, simulator.dt_s)
    sequence = command.simulator_sequence()
    state = clone_state_batch(initial_state, 1)
    states = [state]
    metrics = WhipMetricAccumulator(
        task, batch_size=1, device=simulator.device, dtype=simulator.dtype
    )
    metrics.observe(
        0.0,
        uav_position_m=state.uav.position_m,
        uav_velocity_m_s=state.uav.velocity_m_s,
        cable_positions_m=state.cable.positions_m,
        cable_velocities_m_s=state.cable.velocities_m_s,
    )
    with torch.no_grad():
        for index in range(sequence.step_count):
            state = simulator._propagate(  # noqa: SLF001
                state,
                sequence.command_at(index),
                simulator.parameters,
                create_graph=False,
            )
            states.append(state)
            metrics.observe(
                (index + 1) * simulator.dt_s,
                uav_position_m=state.uav.position_m,
                uav_velocity_m_s=state.uav.velocity_m_s,
                cable_positions_m=state.cable.positions_m,
                cable_velocities_m_s=state.cable.velocities_m_s,
            )
    result = metrics.finalize(
        final_uav_position_m=state.uav.position_m,
        final_uav_velocity_m_s=state.uav.velocity_m_s,
        final_cable_positions_m=state.cable.positions_m,
        command=command,
        knots_m_s2=knots,
    )
    residual = torch.stack(
        [
            torch.zeros_like(item.uav.position_m)
            if item.uav.residual_acceleration_m_s2 is None
            else item.uav.residual_acceleration_m_s2
            for item in states
        ]
    )
    return DeterministicReplay(
        command=command,
        metrics=result,
        times_s=command.times_s,
        uav_positions_m=torch.stack([item.uav.position_m for item in states]),
        uav_velocities_m_s=torch.stack([item.uav.velocity_m_s for item in states]),
        uav_orientations_xyzw=torch.stack(
            [item.uav.orientation_xyzw for item in states]
        ),
        uav_angular_velocities_world_rad_s=torch.stack(
            [item.uav.angular_velocity_world_rad_s for item in states]
        ),
        cable_positions_m=torch.stack([item.cable.positions_m for item in states]),
        cable_velocities_m_s=torch.stack([item.cable.velocities_m_s for item in states]),
        residual_accelerations_m_s2=residual,
    )
