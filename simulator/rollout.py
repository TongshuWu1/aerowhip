"""Deterministic rollouts for the force-controlled point-cable model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .point_mass import ForceControlledPointCable


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def simulate_constant_force(
    model_config: dict[str, Any],
    task_config: dict[str, Any],
    *,
    force_world_n: tuple[float, float, float] | None = None,
    duration_s: float | None = None,
    device: torch.device = torch.device("cpu"),
    cancel_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return one trajectory and a compact force/geometry summary."""

    model = ForceControlledPointCable.from_mapping(model_config)
    dt_s = float(model_config["simulation"]["dt_s"])
    duration = (
        float(task_config["episode_duration_s"])
        if duration_s is None
        else float(duration_s)
    )
    if duration <= 0.0:
        raise ValueError("duration_s must be positive.")
    steps = int(round(duration / dt_s))
    if steps < 1:
        raise ValueError("duration_s is shorter than one simulation step.")
    dtype = torch.float32 if device.type == "cuda" else torch.float64
    root_position = torch.tensor(
        task_config["initial_root_position_m"], dtype=dtype, device=device
    )[None]
    root_velocity = torch.tensor(
        task_config["initial_root_velocity_m_s"], dtype=dtype, device=device
    )[None]
    state = model.hanging_state(root_position, root_velocity)
    command = (
        model.hover_force_world_n(dtype=dtype, device=device)
        if force_world_n is None
        else torch.tensor(force_world_n, dtype=dtype, device=device)
    )[None]

    positions = [state.positions_m[0].detach().cpu().numpy().copy()]
    velocities = [state.velocities_m_s[0].detach().cpu().numpy().copy()]
    cable_reactions: list[np.ndarray] = []
    for step_index in range(steps):
        if cancel_requested is not None and cancel_requested():
            raise InterruptedError("Simulation was cancelled.")
        result = model.step_runtime(state, command, dt_s)
        state = result.state
        positions.append(state.positions_m[0].detach().cpu().numpy().copy())
        velocities.append(state.velocities_m_s[0].detach().cpu().numpy().copy())
        cable_reactions.append(
            result.forces.effective_cable_reaction_on_point_world_n[0]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )
        if progress_callback is not None and (
            step_index == steps - 1 or (step_index + 1) % max(1, steps // 100) == 0
        ):
            progress_callback(step_index + 1, steps)
    position_array = np.stack(positions).astype(np.float64)
    velocity_array = np.stack(velocities).astype(np.float64)
    reaction_array = np.stack(cable_reactions).astype(np.float64)
    command_array = np.broadcast_to(
        command[0].detach().cpu().numpy(), (steps, 3)
    ).copy().astype(np.float64)
    arrays = {
        "time_s": np.arange(steps + 1, dtype=np.float64) * dt_s,
        "cable_node_position_world_m": position_array,
        "cable_node_velocity_world_m_s": velocity_array,
        "commanded_force_world_n": command_array,
        "effective_cable_reaction_on_point_world_n": reaction_array,
    }
    summary = {
        "schema": "point_force_simulation_summary_v1",
        "source": "constant_force",
        "device": str(device),
        "dt_s": dt_s,
        "steps": steps,
        "duration_s": steps * dt_s,
        "point_mass_kg": model.point_mass_kg,
        "cable_mass_kg": model.cable_mass_kg,
        "system_mass_kg": model.system_mass_kg,
        "commanded_force_world_n": command_array[-1].tolist(),
        "hover_force_world_n": model.hover_force_world_n(
            dtype=dtype, device=device
        ).detach().cpu().tolist(),
        "initial_root_position_world_m": position_array[0, 0].tolist(),
        "final_root_position_world_m": position_array[-1, 0].tolist(),
        "final_root_velocity_world_m_s": velocity_array[-1, 0].tolist(),
        "final_tip_position_world_m": position_array[-1, -1].tolist(),
        "final_tip_velocity_world_m_s": velocity_array[-1, -1].tolist(),
        "final_cable_reaction_on_point_world_n": reaction_array[-1].tolist(),
        "maximum_segment_error_m": float(
            model.dder.maximum_segment_error_m(state.positions_m).max().detach().cpu()
        ),
        "finite": bool(
            np.isfinite(position_array).all()
            and np.isfinite(velocity_array).all()
            and np.isfinite(reaction_array).all()
        ),
    }
    return arrays, summary
