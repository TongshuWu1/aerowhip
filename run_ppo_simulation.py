"""Run the frozen PPO whip controller once in the production simulator.

The controller is queried deterministically at 10 Hz until task success or the
configured horizon.  This runner never trains and never accesses hardware.  It
writes one replay directory that the desktop Simulator tab can load.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np
import torch

from run_simple_ppo import _build_agent, _load_config
from run_simple_sac import _build_environment


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = (
    ROOT
    / "results"
    / "ppo"
    / "policies"
    / "PPO_WHIP_FORWARD_REVERSE_RELEASE_D50_V1"
    / "config.json"
)
DEFAULT_CHECKPOINT = (
    ROOT
    / "results"
    / "ppo"
    / "policies"
    / "PPO_WHIP_FORWARD_REVERSE_RELEASE_D50_V1"
    / "checkpoints"
    / "terminal.pt"
)
DEFAULT_OUTPUT = ROOT / "data/ppo_simulation/current"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _scalar(tensor: torch.Tensor, default: float | int = 0.0) -> float:
    value = float(tensor.reshape(-1)[0].detach().cpu())
    return value if math.isfinite(value) else float(default)


def _row(tensor: torch.Tensor) -> np.ndarray:
    return tensor[:, 0].detach().cpu().numpy()


def _terminal_pad(value: torch.Tensor) -> np.ndarray:
    array = _row(value)
    return np.concatenate((array, array[-1:]), axis=0)


def _truncate_replay_at_physics_step(
    arrays: dict[str, np.ndarray], terminal_physics_step: int
) -> dict[str, np.ndarray]:
    """End every time-series array at the state produced by the hit step."""

    full_length = int(arrays["time_s"].shape[0])
    keep = min(full_length, max(1, int(terminal_physics_step) + 1))
    return {
        name: value[:keep]
        if value.ndim >= 1 and int(value.shape[0]) == full_length
        else value
        for name, value in arrays.items()
    }


def _replay_arrays(
    environment: Any,
    states: dict[str, torch.Tensor],
    commands: Any,
) -> dict[str, np.ndarray]:
    uav_position = _row(states["uav_position_m"])
    uav_velocity = _row(states["uav_velocity_m_s"])
    cable_position = _row(states["cable_positions_m"])
    cable_velocity = _row(states["cable_velocities_m_s"])
    target = np.asarray(environment.task.target_position_m, dtype=np.float32)
    direction = np.asarray(environment.task.desired_direction, dtype=np.float32)

    tip_delta = cable_position[:, -1] - target[None, :]
    tip_distance = np.linalg.norm(tip_delta, axis=-1)
    tip_velocity = cable_velocity[:, -1]
    tip_speed = np.linalg.norm(tip_velocity, axis=-1)
    directed_speed = tip_velocity @ direction
    direction_cosine = np.divide(
        directed_speed,
        np.maximum(tip_speed, np.finfo(np.float32).eps),
    )
    direction_error = np.degrees(np.arccos(np.clip(direction_cosine, -1.0, 1.0)))
    non_tip_distance = np.linalg.norm(
        cable_position[:, 2:-1] - target[None, None, :], axis=-1
    ).min(axis=-1)
    uav_speed = np.linalg.norm(uav_velocity, axis=-1)
    uav_displacement = np.linalg.norm(uav_position - uav_position[:1], axis=-1)
    command_orientation = _terminal_pad(commands.orientations_xyzw)
    yaw_command = 2.0 * np.arctan2(
        command_orientation[:, 2], command_orientation[:, 3]
    )

    step_count = uav_position.shape[0]
    return {
        "authorization": np.asarray("SIMULATION_ONLY"),
        "real_flight_authorized": np.asarray(False),
        "task_id": np.asarray("canonical_whip_v1"),
        "controller": np.asarray("PPO_10_HZ_CLOSED_LOOP"),
        "time_s": np.arange(step_count, dtype=np.float32)
        * np.float32(environment.simulator.dt_s),
        "uav_position_m": uav_position,
        "uav_velocity_m_s": uav_velocity,
        "uav_orientation_xyzw": _row(states["uav_orientation_xyzw"]),
        "uav_angular_velocity_world_rad_s": _row(
            states["uav_angular_velocity_world_rad_s"]
        ),
        "cable_position_m": cable_position,
        "cable_velocity_m_s": cable_velocity,
        "c1_c10_position_m": cable_position[:, 2:12],
        "c1_c10_velocity_m_s": cable_velocity[:, 2:12],
        "residual_acceleration_m_s2": _row(states["residual_acceleration_m_s2"]),
        "p_cmd_m": _terminal_pad(commands.positions_m),
        "v_cmd_m_s": _terminal_pad(commands.velocities_m_s),
        "a_cmd_m_s2": _terminal_pad(commands.accelerations_m_s2),
        "yaw_cmd_rad": yaw_command.astype(np.float32),
        "omega_cmd_body_rad_s": _terminal_pad(
            commands.angular_velocities_body_rad_s
        ),
        "tip_target_distance_m": tip_distance,
        "tip_speed_m_s": tip_speed,
        "directed_tip_speed_m_s": directed_speed,
        "impact_direction_error_deg": direction_error,
        "minimum_non_tip_target_distance_m": non_tip_distance,
        "uav_speed_m_s": uav_speed,
        "uav_displacement_m": uav_displacement,
    }


def run_simulation(config_path: Path, checkpoint_path: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config = _load_config(config_path)
    environment, device = _build_environment(
        config,
        batch_size=1,
        record_fullstate_commands=True,
        record_state_trajectory=True,
    )
    agent = _build_agent(config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent.policy.load_state_dict(checkpoint["policy"])
    agent.value.load_state_dict(checkpoint["value"])
    agent.policy.eval()
    agent.value.eval()

    observation = environment.reset()
    with torch.no_grad():
        for _ in range(environment.control_step_count):
            observation = environment.step(
                agent.deterministic_action(observation)
            ).next_observation

    states = environment.recorded_state_trajectory(clone=False)
    commands = environment.recorded_command_sequence(clone=False)
    arrays = _replay_arrays(environment, states, commands)
    success = bool(environment.episode_success[0].item())
    terminate_on_success = not bool(
        config.get("reported_success", {}).get(
            "episode_continues_after_success", True
        )
    )
    if success and terminate_on_success:
        arrays = _truncate_replay_at_physics_step(
            arrays,
            int(environment.episode_first_entry_physics_step[0].item()),
        )
    replay_temporary = output / "final_replay.tmp.npz"
    np.savez_compressed(replay_temporary, **arrays)
    os.replace(replay_temporary, output / "final_replay.npz")

    metrics = {
        "schema": "ppo_closed_loop_simulation_metrics_v1",
        "optimizer": "PPO",
        "controller": "PPO_10_HZ_CLOSED_LOOP",
        "task_classification": "PASS" if success else "FAIL",
        "success": success,
        "terminal_on_success": terminate_on_success,
        "replay_duration_s": float(arrays["time_s"][-1]),
        "replay_physics_steps": int(arrays["time_s"].shape[0] - 1),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_episodes": int(checkpoint.get("episodes", -1)),
        "first_entry_marker": int(environment.episode_first_entry_marker[0].item()),
        "first_entry_time_s": _scalar(environment.episode_first_entry_time_s),
        "first_entry_tip_distance_m": _scalar(
            environment.episode_first_entry_tip_distance
        ),
        "first_entry_tip_speed_m_s": _scalar(environment.episode_first_entry_tip_speed),
        "first_entry_directed_speed_m_s": _scalar(
            environment.episode_first_entry_directed_speed
        ),
        "first_entry_direction_error_deg": _scalar(
            environment.episode_first_entry_direction_error_deg,
            default=180.0,
        ),
        "minimum_tip_target_distance_m": _scalar(
            environment.episode_minimum_tip_distance
        ),
        "maximum_uav_displacement_m": _scalar(
            environment.episode_maximum_displacement
        ),
        "maximum_uav_speed_m_s": _scalar(environment.episode_maximum_uav_speed),
        "maximum_command_acceleration_m_s2": _scalar(
            environment.episode_maximum_command_acceleration
        ),
        "numerical_failure": bool(environment.failed[0].item()),
        "elapsed_wall_time_s": time.perf_counter() - started,
        "authorization": "SIMULATION_ONLY",
        "hardware_executed": False,
    }
    _write_json(output / "final_metrics.json", metrics)
    task_snapshot = json.loads((ROOT / config["task_config"]).read_text(encoding="utf-8"))
    _write_json(output / "task_config_snapshot.json", task_snapshot)
    _write_json(output / "mppi_iteration_history.json", [])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    config_path = arguments.config.resolve()
    checkpoint_path = arguments.checkpoint.resolve()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "status.json",
        {
            "status": "RUNNING",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "controller": "PPO_10_HZ_CLOSED_LOOP",
            "authorization": "SIMULATION_ONLY",
        },
    )
    try:
        metrics = run_simulation(config_path, checkpoint_path, output)
        _write_json(
            output / "status.json",
            {
                "status": "COMPLETE",
                "controller": "PPO_10_HZ_CLOSED_LOOP",
                "success": metrics["success"],
                "elapsed_wall_time_s": metrics["elapsed_wall_time_s"],
                "authorization": "SIMULATION_ONLY",
            },
        )
        print(json.dumps(metrics, indent=2), flush=True)
    except BaseException as error:
        _write_json(
            output / "status.json",
            {
                "status": "FAILED",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "authorization": "SIMULATION_ONLY",
            },
        )
        raise


if __name__ == "__main__":
    main()
