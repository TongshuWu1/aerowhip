"""Export the saved Milestone-6A canonical CEM action as a GUI replay.

This does not optimize or run CEM.  It authoritative-replays the already saved
normalized 49-D canonical action through the fixed-2048 production evaluator
and stores the complete 2.40-s ACTIVE -> SETTLE -> HOLD trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from planning.cem_task import load_variable_duration_task
from planning.production_cem import record_normalized_actions_fixed_batch
from run_milestone6a import _canonical_setup, _repeated_context, _settings
from simulator.parameters import SimulatorSettings
from simulator.production import (
    PROJECT_ROOT,
    active_model_paths,
    build_production_simulator,
    load_active_model_manifest,
)


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "planning" / "production_cem_benchmark_v1.json"
DEFAULT_ROWS = PROJECT_ROOT / "results" / "cem" / "data" / "canonical_seed_results.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "cem"
    / "replays"
    / "canonical_whip_variable_duration_tuned_reward_v1"
    / "current"
)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if bool(row.get("success"))]
    if not successes:
        raise RuntimeError("The saved canonical CEM rows contain no authoritative success.")
    return min(
        successes,
        key=lambda row: (
            float(row["authoritative_metrics"]["task_cost"]),
            float(row["authoritative_metrics"]["first_entry_tip_distance_m"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--canonical-rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config_path = args.config.resolve()
    rows_path = args.canonical_rows.resolve()
    output = args.output.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    selected = _selected_row(rows)
    selected_metrics = selected["authoritative_metrics"]
    maneuver_duration_s = float(selected_metrics["maneuver_duration_s"])
    task_path = PROJECT_ROOT / str(config["task_config"])
    task = load_variable_duration_task(task_path)
    settings = _settings(config)

    active = load_active_model_manifest()
    simulator_settings = SimulatorSettings.load(active_model_paths(active)["configuration"])
    simulator = build_production_simulator(
        simulator_settings, device="cuda", dtype=torch.float32
    )
    simulator.uav_model.set_fixed_evaluation_batch_size(2048)
    canonical_bank, canonical_context = _canonical_setup(simulator, task)
    repeated = _repeated_context(
        simulator,
        canonical_bank,
        0,
        canonical_context.target_position_local_m[0].detach().cpu(),
        canonical_context.target_direction_local[0].detach().cpu(),
        split="canonical",
    )
    action = torch.tensor(selected["normalized_action"], dtype=torch.float64)
    trajectory = record_normalized_actions_fixed_batch(
        simulator,
        repeated,
        action,
        task,
        settings,
        record_count=1,
    )
    metrics = trajectory.metrics.row(0)
    if not bool(metrics["success"]):
        raise RuntimeError("The saved CEM action failed its authoritative export replay.")

    time_s = trajectory.times_s.numpy().astype(np.float32)
    uav_position = trajectory.uav_positions_m[:, 0].numpy().astype(np.float32)
    uav_velocity = trajectory.uav_velocities_m_s[:, 0].numpy().astype(np.float32)
    cable_position = trajectory.cable_positions_m[:, 0].numpy().astype(np.float32)
    cable_velocity = trajectory.cable_velocities_m_s[:, 0].numpy().astype(np.float32)
    target = np.asarray(task.target_position_m, dtype=np.float32)
    direction = np.asarray(task.desired_direction, dtype=np.float32)
    direction /= np.linalg.norm(direction)
    tip_offset = cable_position[:, -1] - target
    tip_velocity = cable_velocity[:, -1]
    tip_speed = np.linalg.norm(tip_velocity, axis=-1)
    directed_speed = tip_velocity @ direction
    direction_cosine = np.divide(
        directed_speed,
        tip_speed,
        out=np.zeros_like(directed_speed),
        where=tip_speed > 1.0e-8,
    )
    direction_error = np.degrees(np.arccos(np.clip(direction_cosine, -1.0, 1.0)))
    non_tip_distance = np.linalg.norm(cable_position[:, 2:-1] - target, axis=-1).min(axis=-1)

    output.mkdir(parents=True, exist_ok=True)
    replay_path = output / "final_replay.npz"
    np.savez_compressed(
        replay_path,
        authorization=np.asarray("SIMULATION_ONLY"),
        real_flight_authorized=np.asarray(False),
        task_id=np.asarray(task.task_id),
        time_s=time_s,
        uav_position_m=uav_position,
        uav_velocity_m_s=uav_velocity,
        cable_position_m=cable_position,
        cable_velocity_m_s=cable_velocity,
        c1_c10_position_m=cable_position[:, 2:],
        c1_c10_velocity_m_s=cable_velocity[:, 2:],
        p_cmd_m=trajectory.command_positions_m[:, 0].numpy().astype(np.float32),
        v_cmd_m_s=trajectory.command_velocities_m_s[:, 0].numpy().astype(np.float32),
        a_cmd_m_s2=trajectory.command_accelerations_m_s2[:, 0].numpy().astype(np.float32),
        tip_target_distance_m=np.linalg.norm(tip_offset, axis=-1),
        tip_speed_m_s=tip_speed,
        directed_tip_speed_m_s=directed_speed,
        impact_direction_error_deg=direction_error,
        minimum_non_tip_target_distance_m=non_tip_distance,
        uav_speed_m_s=np.linalg.norm(uav_velocity, axis=-1),
        uav_displacement_m=np.linalg.norm(uav_position - uav_position[0], axis=-1),
    )

    task_snapshot = json.loads(task_path.read_text(encoding="utf-8"))
    task_snapshot["production_replay_contract"] = {
        "action": "normalized_49d",
        "active_maneuver_duration_s": maneuver_duration_s,
        "settle_duration_s": settings.settle_duration_s,
        "evaluation_duration_s": settings.evaluation_time_s,
        "command_segments": ["ACTIVE", "SETTLE", "HOLD"],
        "fixed_numerical_batch_size": 2048,
    }
    _write_json(output / "task_config_snapshot.json", task_snapshot)
    metrics.update(
        {
            "optimizer": "PRODUCTION_VARIABLE_DURATION_CEM",
            "controller": "OPEN_LOOP_CEM_REFERENCE",
            "task_classification": "PASS",
            "hit_time_s": metrics.get("first_entry_time_s"),
            "maneuver_duration_s": maneuver_duration_s,
            "optimized_duration_s": maneuver_duration_s,
            "hit_segment": selected_metrics.get("hit_segment"),
            "source_cem_seed": selected["seed"],
            "source_cem_context_id": selected["context_id"],
            "evaluation_horizon_s": settings.evaluation_time_s,
            "authorization": "SIMULATION_ONLY",
        }
    )
    _write_json(output / "final_metrics.json", metrics)
    _write_json(output / "cem_iteration_history.json", [])
    _write_json(
        output / "source_manifest.json",
        {
            "schema": "production_cem_gui_reference_v1",
            "new_cem_solves": 0,
            "canonical_seed_results": str(rows_path),
            "canonical_seed_results_sha256": _sha256(rows_path),
            "selected_seed": selected["seed"],
            "normalized_action": selected["normalized_action"],
            "model_freeze": config["model_freeze"],
            "replay_sha256": _sha256(replay_path),
        },
    )
    print(output)
    print(
        f"PASS; T={time_s[-1]:.2f}s; hit={float(metrics['first_entry_time_s']):.2f}s; "
        f"tip={1000.0 * float(metrics['first_entry_tip_distance_m']):.1f}mm"
    )


if __name__ == "__main__":
    main()
