"""Reusable physically valid nominal-simulator initial-state banks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from planning.cem_task import VariableDurationWhipTask
from planning.command_parameterization import project_acceleration_knots
from planning.rollout import clone_state_batch, hover_preroll
from planning.variable_duration import variable_duration_fullstate
from simulator.cable.dder import DderState
from simulator.simulator import CoupledSimulator
from simulator.state import SimulatorState
from simulator.uav.state import ResidualHistoryState, UAVState


@dataclass(frozen=True, slots=True)
class StateBankBatch:
    state: SimulatorState
    command_position_world_m: torch.Tensor
    command_velocity_world_m_s: torch.Tensor
    command_yaw_world_rad: torch.Tensor
    source_indices: torch.Tensor


@dataclass(frozen=True, slots=True)
class InitialStateBank:
    uav_position_m: torch.Tensor
    uav_velocity_m_s: torch.Tensor
    uav_orientation_xyzw: torch.Tensor
    uav_angular_velocity_world_rad_s: torch.Tensor
    residual_history: torch.Tensor
    residual_acceleration_m_s2: torch.Tensor
    cable_positions_m: torch.Tensor
    cable_velocities_m_s: torch.Tensor
    command_position_world_m: torch.Tensor
    command_velocity_world_m_s: torch.Tensor
    command_yaw_world_rad: torch.Tensor
    seed: int
    generation: dict[str, Any]

    def __post_init__(self) -> None:
        count = int(self.uav_position_m.shape[0])
        expected = {
            "uav_position_m": (count, 3),
            "uav_velocity_m_s": (count, 3),
            "uav_orientation_xyzw": (count, 4),
            "uav_angular_velocity_world_rad_s": (count, 3),
            "residual_history": (count, 10, 9),
            "residual_acceleration_m_s2": (count, 3),
            "cable_positions_m": (count, 12, 3),
            "cable_velocities_m_s": (count, 12, 3),
            "command_position_world_m": (count, 3),
            "command_velocity_world_m_s": (count, 3),
            "command_yaw_world_rad": (count,),
        }
        for name, shape in expected.items():
            value = torch.as_tensor(getattr(self, name), dtype=torch.float32, device="cpu")
            if value.shape != shape or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Initial-state bank field {name} must be finite with shape {shape}.")
            object.__setattr__(self, name, value.contiguous())
        if count < 1:
            raise ValueError("Initial-state bank cannot be empty.")

    def __len__(self) -> int:
        return int(self.uav_position_m.shape[0])

    def select(self, indices: torch.Tensor, *, device: torch.device | str) -> StateBankBatch:
        index = torch.as_tensor(indices, dtype=torch.int64, device="cpu").reshape(-1)
        if index.numel() < 1 or int(index.min()) < 0 or int(index.max()) >= len(self):
            raise IndexError("State-bank index is outside the stored range.")
        selected = torch.device(device)

        def take(value: torch.Tensor) -> torch.Tensor:
            return value[index].to(selected, non_blocking=True)

        state = SimulatorState(
            0.0,
            UAVState(
                take(self.uav_position_m),
                take(self.uav_velocity_m_s),
                take(self.uav_orientation_xyzw),
                take(self.uav_angular_velocity_world_rad_s),
                ResidualHistoryState(take(self.residual_history)),
                take(self.residual_acceleration_m_s2),
            ),
            DderState(take(self.cable_positions_m), take(self.cable_velocities_m_s)),
        )
        return StateBankBatch(
            state,
            take(self.command_position_world_m),
            take(self.command_velocity_world_m_s),
            take(self.command_yaw_world_rad),
            index,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "oneshot_nominal_initial_state_bank_v1",
            "count": len(self),
            "seed": self.seed,
            "dtype": "float32",
            "contains_residual_fifo": True,
            "contains_command_boundary": True,
            "generated_from_real_data": False,
            "nominal_physics_only": True,
            "generation": self.generation,
        }

    def save(self, path: str | Path, manifest_path: str | Path | None = None) -> None:
        destination = Path(path)
        np.savez_compressed(
            destination,
            **{
                name: getattr(self, name).numpy()
                for name in (
                    "uav_position_m",
                    "uav_velocity_m_s",
                    "uav_orientation_xyzw",
                    "uav_angular_velocity_world_rad_s",
                    "residual_history",
                    "residual_acceleration_m_s2",
                    "cable_positions_m",
                    "cable_velocities_m_s",
                    "command_position_world_m",
                    "command_velocity_world_m_s",
                    "command_yaw_world_rad",
                )
            },
            seed=np.asarray(self.seed, dtype=np.int64),
        )
        if manifest_path is not None:
            Path(manifest_path).write_text(
                json.dumps({**self.manifest(), "artifact": str(destination)}, indent=2) + "\n",
                encoding="utf-8",
            )

    @classmethod
    def load(cls, path: str | Path, manifest_path: str | Path) -> "InitialStateBank":
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if manifest.get("schema") != "oneshot_nominal_initial_state_bank_v1":
            raise ValueError("Unsupported initial-state bank manifest.")
        with np.load(path, allow_pickle=False) as archive:
            values = {
                name: torch.from_numpy(np.asarray(archive[name]).copy())
                for name in (
                    "uav_position_m",
                    "uav_velocity_m_s",
                    "uav_orientation_xyzw",
                    "uav_angular_velocity_world_rad_s",
                    "residual_history",
                    "residual_acceleration_m_s2",
                    "cable_positions_m",
                    "cable_velocities_m_s",
                    "command_position_world_m",
                    "command_velocity_world_m_s",
                    "command_yaw_world_rad",
                )
            }
        return cls(**values, seed=int(manifest["seed"]), generation=dict(manifest["generation"]))


def _extract_valid_rows(
    state: SimulatorState,
    command_position: torch.Tensor,
    command_velocity: torch.Tensor,
    command_yaw: torch.Tensor,
    *,
    initial_position: torch.Tensor,
    maximum_speed_m_s: float,
    maximum_displacement_m: float,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    if state.uav.residual_history is None or state.uav.residual_acceleration_m_s2 is None:
        raise RuntimeError("Production state-bank generation requires the frozen residual state.")
    tensors = (
        state.uav.position_m,
        state.uav.velocity_m_s,
        state.uav.orientation_xyzw,
        state.uav.angular_velocity_world_rad_s,
        state.uav.residual_history.features,
        state.uav.residual_acceleration_m_s2,
        state.cable.positions_m,
        state.cable.velocities_m_s,
        command_position,
        command_velocity,
        command_yaw,
    )
    finite = torch.ones(state.uav.batch_size, dtype=torch.bool, device=state.uav.position_m.device)
    for value in tensors:
        finite &= torch.isfinite(value).reshape(value.shape[0], -1).all(dim=1)
    speed = torch.linalg.vector_norm(state.uav.velocity_m_s, dim=-1)
    displacement = torch.linalg.vector_norm(state.uav.position_m - initial_position, dim=-1)
    valid = finite & (speed < maximum_speed_m_s) & (displacement < maximum_displacement_m)
    if not bool(valid.any()):
        raise RuntimeError("No physically valid state-bank rows survived the configured gate.")
    names = (
        "uav_position_m",
        "uav_velocity_m_s",
        "uav_orientation_xyzw",
        "uav_angular_velocity_world_rad_s",
        "residual_history",
        "residual_acceleration_m_s2",
        "cable_positions_m",
        "cable_velocities_m_s",
        "command_position_world_m",
        "command_velocity_world_m_s",
        "command_yaw_world_rad",
    )
    selected = {name: value[valid].detach().cpu() for name, value in zip(names, tensors, strict=True)}
    return selected, {
        "generated": int(valid.numel()),
        "accepted": int(valid.sum()),
        "maximum_observed_speed_m_s": float(speed.max()),
        "maximum_observed_displacement_m": float(displacement.max()),
    }


def generate_initial_state_bank(
    simulator: CoupledSimulator,
    task: VariableDurationWhipTask,
    *,
    count: int,
    seed: int,
    logical_batch_size: int = 2048,
    excitation_max_acceleration_m_s2: float = 3.0,
    generation_horizon_s: float = 0.80,
    excitation_duration_range_s: tuple[float, float] = (0.20, 0.55),
    maximum_state_speed_m_s: float = 1.5,
    maximum_state_displacement_m: float = 0.30,
) -> InitialStateBank:
    """Generate modestly excited, fully causal states from the production simulator."""

    if count < 1 or logical_batch_size < 1:
        raise ValueError("State-bank count and logical batch size must be positive.")
    if simulator.device.type != "cuda" or simulator.dtype != torch.float32:
        raise ValueError("Production state-bank generation requires CUDA float32.")
    base = hover_preroll(simulator, task)
    device, dtype = simulator.device, simulator.dtype
    generator = torch.Generator(device=device).manual_seed(seed)
    collected: dict[str, list[torch.Tensor]] = {}
    summaries: list[dict[str, float]] = []
    accepted = 0
    attempts = 0
    while accepted < count:
        attempts += 1
        if attempts > 8:
            raise RuntimeError("Initial-state bank generation exhausted its bounded attempts.")
        batch = logical_batch_size
        state = clone_state_batch(base, batch)
        knots = torch.randn((batch, 5, 3), generator=generator, device=device, dtype=dtype)
        knots[:, 0] = 0.0
        knots[:, -1] *= 0.25
        knots = project_acceleration_knots(knots, excitation_max_acceleration_m_s2)
        lower, upper = excitation_duration_range_s
        durations = lower + (upper - lower) * torch.rand(
            (batch,), generator=generator, device=device, dtype=dtype
        )
        initial_position = torch.tensor(task.initial_uav_position_m, device=device, dtype=dtype)
        initial_velocity = torch.zeros(3, device=device, dtype=dtype)
        command = variable_duration_fullstate(
            knots,
            durations,
            initial_position_m=initial_position,
            initial_velocity_m_s=initial_velocity,
            yaw_rad=task.initial_yaw_rad,
            maximum_time_s=generation_horizon_s,
            dt_s=simulator.dt_s,
        )
        sequence = command.simulator_sequence()
        with torch.no_grad():
            for step in range(sequence.step_count):
                state = simulator._propagate(  # noqa: SLF001
                    state, sequence.command_at(step), simulator.parameters, create_graph=False
                )
        yaw = torch.full((batch,), task.initial_yaw_rad, device=device, dtype=dtype)
        rows, summary = _extract_valid_rows(
            state,
            command.positions_m[-1],
            command.velocities_m_s[-1],
            yaw,
            initial_position=initial_position,
            maximum_speed_m_s=maximum_state_speed_m_s,
            maximum_displacement_m=maximum_state_displacement_m,
        )
        summaries.append(summary)
        for name, value in rows.items():
            collected.setdefault(name, []).append(value)
        accepted += int(next(iter(rows.values())).shape[0])
    combined = {name: torch.cat(values, dim=0)[:count] for name, values in collected.items()}
    return InitialStateBank(
        **combined,
        seed=seed,
        generation={
            "method": "nominal_production_modest_smooth_uav_excitation_v1",
            "logical_batch_size": logical_batch_size,
            "excitation_max_acceleration_m_s2": excitation_max_acceleration_m_s2,
            "generation_horizon_s": generation_horizon_s,
            "excitation_duration_range_s": list(excitation_duration_range_s),
            "maximum_state_speed_m_s": maximum_state_speed_m_s,
            "maximum_state_displacement_m": maximum_state_displacement_m,
            "attempt_summaries": summaries,
            "source_physical_data": "NONE",
        },
    )


def initial_state_bank_from_state(
    state: SimulatorState,
    *,
    command_position_world_m: torch.Tensor,
    command_velocity_world_m_s: torch.Tensor,
    command_yaw_world_rad: torch.Tensor | float,
    seed: int = 0,
) -> InitialStateBank:
    """Create a reusable bank from a causal production state batch."""

    if state.uav.residual_history is None or state.uav.residual_acceleration_m_s2 is None:
        raise ValueError("A policy state bank requires a residual FIFO and residual output.")
    count = state.uav.batch_size

    def cpu(value: torch.Tensor) -> torch.Tensor:
        return value.detach().to(device="cpu", dtype=torch.float32).contiguous()

    position = torch.as_tensor(command_position_world_m, device=state.uav.position_m.device, dtype=state.uav.position_m.dtype)
    velocity = torch.as_tensor(command_velocity_world_m_s, device=state.uav.position_m.device, dtype=state.uav.position_m.dtype)
    yaw = torch.as_tensor(command_yaw_world_rad, device=state.uav.position_m.device, dtype=state.uav.position_m.dtype)
    if position.ndim == 1:
        position = position.unsqueeze(0).expand(count, -1)
    if velocity.ndim == 1:
        velocity = velocity.unsqueeze(0).expand(count, -1)
    if yaw.ndim == 0:
        yaw = yaw.expand(count)
    return InitialStateBank(
        uav_position_m=cpu(state.uav.position_m),
        uav_velocity_m_s=cpu(state.uav.velocity_m_s),
        uav_orientation_xyzw=cpu(state.uav.orientation_xyzw),
        uav_angular_velocity_world_rad_s=cpu(state.uav.angular_velocity_world_rad_s),
        residual_history=cpu(state.uav.residual_history.features),
        residual_acceleration_m_s2=cpu(state.uav.residual_acceleration_m_s2),
        cable_positions_m=cpu(state.cable.positions_m),
        cable_velocities_m_s=cpu(state.cable.velocities_m_s),
        command_position_world_m=cpu(position),
        command_velocity_world_m_s=cpu(velocity),
        command_yaw_world_rad=cpu(yaw),
        seed=seed,
        generation={"method": "explicit_causal_production_state", "source_physical_data": "NONE"},
    )
