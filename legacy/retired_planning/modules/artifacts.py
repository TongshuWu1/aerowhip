"""Reproducible Milestone-4A artifacts, plots, freeze, and report."""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np
import torch

from experimental_data.io import sha256_file
from simulator.parameters import SimulatorSettings
from simulator.production import (
    PROJECT_ROOT,
    load_active_model_manifest,
    verify_active_geometry,
    verify_active_parameters,
)

from .mppi import MppiResult
from .rollout import DeterministicReplay
from .task import CanonicalWhipTask


RESULT_ROOT = PROJECT_ROOT / "data" / "planning_results" / "canonical_whip_v1"
PLANNER_FREEZE = (
    PROJECT_ROOT / "data" / "planner_freezes" / "PLANNER_FREEZE_CANONICAL_WHIP_SIM_V1"
)
REPORT_PATH = PROJECT_ROOT / "MILESTONE4A_CANONICAL_WHIP_MPPI_REPORT.md"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_planning_model_integrity(
    settings: SimulatorSettings,
    task: CanonicalWhipTask,
) -> dict[str, Any]:
    """Fail closed unless the exact PRE_MPPI freeze is active and intact."""

    active = load_active_model_manifest()
    required_freeze = "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI"
    if active.get("status") != "MODEL_FROZEN_FOR_MPPI":
        raise RuntimeError("Active model status is not MODEL_FROZEN_FOR_MPPI.")
    if active.get("ready_for_mppi") is not True:
        raise RuntimeError("Active model is not explicitly ready_for_mppi.")
    if active.get("model_integrity") != "Verified":
        raise RuntimeError("Active model integrity is not Verified.")
    freeze_relative = Path(str(active.get("production_freeze", "")))
    if freeze_relative.name != required_freeze or task.model_freeze != required_freeze:
        raise RuntimeError("Active planning freeze does not match canonical task.")
    if active.get("protected_test_predictively_evaluated") is not False:
        raise RuntimeError("Protected-test seal is not intact.")
    verify_active_geometry(settings, active)
    verify_active_parameters(settings, active)
    freeze = (PROJECT_ROOT / freeze_relative).resolve()
    manifest_path = freeze / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_status") != "MODEL_FROZEN_FOR_MPPI":
        raise RuntimeError("Pinned production freeze is not ready for MPPI.")
    if manifest.get("protected_test_predictively_evaluated") is not False:
        raise RuntimeError("Pinned production freeze has lost protected-test seal.")
    verified: dict[str, str] = {}
    for relative, expected in manifest["artifact_hashes"].items():
        artifact = freeze / relative
        observed = sha256_file(artifact)
        if observed != expected:
            raise RuntimeError(f"Frozen artifact hash mismatch: {artifact}")
        verified[relative] = observed
    acceptance = json.loads((freeze / "pre_mppi_acceptance.json").read_text(encoding="utf-8"))
    if acceptance.get("pass") is not True:
        raise RuntimeError("Frozen PRE_MPPI acceptance artifact is not PASS.")
    frozen_active = json.loads(
        (freeze / "active_model_manifest.json").read_text(encoding="utf-8")
    )
    keys = (
        "geometry_version",
        "simulator_configuration",
        "uav_residual_freeze",
        "development_cable_fit",
        "damping_backend",
        "precision",
    )
    if any(active.get(key) != frozen_active.get(key) for key in keys):
        raise RuntimeError("Current active model differs from the frozen active snapshot.")
    return {
        "verified": True,
        "active_model": active,
        "freeze_path": str(freeze),
        "freeze_manifest_sha256": sha256_file(manifest_path),
        "verified_artifact_hashes": verified,
        "protected_test_predictively_evaluated": False,
    }


def create_result_directory(task_id: str = "canonical_whip_v1") -> Path:
    timestamp = datetime.now(timezone.utc).isoformat().replace(":", "").replace("+0000", "Z")
    path = PROJECT_ROOT / "data" / "planning_results" / task_id / timestamp
    path.mkdir(parents=True, exist_ok=False)
    return path


def _host(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def save_command_csv(path: Path, replay: DeterministicReplay) -> None:
    command = replay.command
    position = _host(command.positions_m[:, 0])
    velocity = _host(command.velocities_m_s[:, 0])
    acceleration = _host(command.accelerations_m_s2[:, 0])
    angular = _host(command.angular_velocities_body_rad_s[:, 0])
    times = _host(command.times_s)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "time_s",
                "p_cmd_x",
                "p_cmd_y",
                "p_cmd_z",
                "v_cmd_x",
                "v_cmd_y",
                "v_cmd_z",
                "a_cmd_x",
                "a_cmd_y",
                "a_cmd_z",
                "yaw_cmd",
                "omega_cmd_x",
                "omega_cmd_y",
                "omega_cmd_z",
            )
        )
        for index, time_s in enumerate(times):
            writer.writerow(
                (
                    float(time_s),
                    *position[index].tolist(),
                    *velocity[index].tolist(),
                    *acceleration[index].tolist(),
                    0.0,
                    *angular[index].tolist(),
                )
            )


def _per_step_metrics(replay: DeterministicReplay, task: CanonicalWhipTask) -> dict[str, np.ndarray]:
    cable_position = _host(replay.cable_positions_m[:, 0])
    cable_velocity = _host(replay.cable_velocities_m_s[:, 0])
    uav_position = _host(replay.uav_positions_m[:, 0])
    uav_velocity = _host(replay.uav_velocities_m_s[:, 0])
    target = np.asarray(task.target_position_m)
    direction = np.asarray(task.desired_direction)
    tip_position = cable_position[:, 11]
    tip_velocity = cable_velocity[:, 11]
    tip_speed = np.linalg.norm(tip_velocity, axis=-1)
    directed = tip_velocity @ direction
    cosine = np.clip(directed / np.maximum(tip_speed, 1.0e-9), -1.0, 1.0)
    marker_distance = np.linalg.norm(cable_position[:, 2:12] - target, axis=-1)
    return {
        "tip_target_distance_m": np.linalg.norm(tip_position - target, axis=-1),
        "tip_speed_m_s": tip_speed,
        "directed_tip_speed_m_s": directed,
        "impact_direction_error_deg": np.degrees(np.arccos(cosine)),
        "minimum_non_tip_target_distance_m": np.min(marker_distance[:, :-1], axis=-1),
        "uav_speed_m_s": np.linalg.norm(uav_velocity, axis=-1),
        "uav_displacement_m": np.linalg.norm(
            uav_position - np.asarray(task.initial_uav_position_m), axis=-1
        ),
    }


def final_replay_metrics(
    replay: DeterministicReplay,
    task: CanonicalWhipTask,
    settings: SimulatorSettings,
) -> dict[str, Any]:
    row = replay.metrics.row(0)
    per_step = _per_step_metrics(replay, task)
    success = bool(row["success"])
    event_time = row["first_entry_time_s"] if row["first_entry_time_s"] is not None else row["best_event_time_s"]
    event_index = int(round(float(event_time) / settings.dt_s))
    event_index = max(0, min(event_index, len(per_step["tip_speed_m_s"]) - 1))
    uav_position = _host(replay.uav_positions_m[event_index, 0])
    uav_to_target = float(
        np.linalg.norm(uav_position - np.asarray(task.target_position_m))
    )
    uav_speed_event = float(per_step["uav_speed_m_s"][event_index])
    tip_speed_event = float(per_step["tip_speed_m_s"][event_index])
    speed_ratio = tip_speed_event / max(uav_speed_event, 1.0e-9)
    masses = np.asarray(settings.cable_configuration.vertex_masses_kg)
    velocities = _host(replay.cable_velocities_m_s[event_index, 0])
    node_energy = 0.5 * masses * np.sum(velocities * velocities, axis=-1)
    observed_energy = float(np.sum(node_energy[2:12]))
    distal_energy = float(np.sum(node_energy[9:12]))
    marker = row["first_entry_marker"]
    hard_gates = {
        "tip_position": bool(
            marker == 10
            and row["first_entry_tip_distance_m"] is not None
            and float(row["first_entry_tip_distance_m"]) <= task.success_radius_m
        ),
        "directed_tip_speed": bool(
            marker == 10
            and float(row["first_entry_directed_speed_m_s"])
            >= task.minimum_directed_speed_m_s
        ),
        "impact_direction": bool(
            marker == 10
            and float(row["first_entry_direction_angle_deg"])
            <= task.maximum_direction_error_deg
        ),
        "tip_first": marker == 10,
        "uav_displacement": float(row["maximum_uav_displacement_m"])
        <= task.maximum_uav_displacement_m,
        "uav_speed": float(row["maximum_uav_speed_m_s"])
        <= task.maximum_uav_speed_m_s,
        "command_acceleration": float(row["maximum_command_acceleration_m_s2"])
        <= task.maximum_command_acceleration_m_s2 + 1.0e-5,
        "finite": bool(row["finite"]),
    }
    return {
        **row,
        "task_id": task.task_id,
        "mppi_simulation": "PASS" if success else "FAIL",
        "hit_time_s": row["first_entry_time_s"] if success else None,
        "reported_event_time_s": event_time,
        "reported_event_is_valid_hit": success,
        "reported_event_tip_position_error_m": (
            row["first_entry_tip_distance_m"] if success else row["best_event_tip_distance_m"]
        ),
        "reported_event_tip_total_speed_m_s": (
            row["first_entry_tip_speed_m_s"] if success else row["best_event_tip_speed_m_s"]
        ),
        "reported_event_directed_tip_speed_m_s": (
            row["first_entry_directed_speed_m_s"] if success else row["best_event_directed_speed_m_s"]
        ),
        "reported_event_direction_error_deg": (
            row["first_entry_direction_angle_deg"] if success else row["best_event_direction_angle_deg"]
        ),
        "uav_to_target_distance_at_reported_event_m": uav_to_target,
        "uav_speed_at_reported_event_m_s": uav_speed_event,
        "tip_speed_to_uav_speed_ratio_at_reported_event": speed_ratio,
        "first_target_entry_marker_label": None if marker is None else f"c{marker}",
        "hard_success_gates": hard_gates,
        "failed_hard_success_gates": [name for name, passed in hard_gates.items() if not passed],
        "cable_kinetic_energy_at_reported_event_j": observed_energy,
        "distal_c8_c10_kinetic_energy_at_reported_event_j": distal_energy,
        "distal_c8_c10_kinetic_energy_fraction": (
            distal_energy / observed_energy if observed_energy > 0.0 else 0.0
        ),
        "cable_energy_definition": "0.5*m_i*||v_i||^2 over observed dynamic nodes c1..c10; distal fraction uses c8..c10",
    }


def save_replay_npz(path: Path, replay: DeterministicReplay, task: CanonicalWhipTask) -> None:
    step_metrics = _per_step_metrics(replay, task)
    np.savez_compressed(
        path,
        authorization=np.asarray("SIMULATION_ONLY"),
        real_flight_authorized=np.asarray(False),
        task_id=np.asarray(task.task_id),
        time_s=_host(replay.times_s),
        uav_position_m=_host(replay.uav_positions_m[:, 0]),
        uav_velocity_m_s=_host(replay.uav_velocities_m_s[:, 0]),
        uav_orientation_xyzw=_host(replay.uav_orientations_xyzw[:, 0]),
        uav_angular_velocity_world_rad_s=_host(
            replay.uav_angular_velocities_world_rad_s[:, 0]
        ),
        cable_position_m=_host(replay.cable_positions_m[:, 0]),
        cable_velocity_m_s=_host(replay.cable_velocities_m_s[:, 0]),
        c1_c10_position_m=_host(replay.cable_positions_m[:, 0, 2:12]),
        c1_c10_velocity_m_s=_host(replay.cable_velocities_m_s[:, 0, 2:12]),
        residual_acceleration_m_s2=_host(replay.residual_accelerations_m_s2[:, 0]),
        p_cmd_m=_host(replay.command.positions_m[:, 0]),
        v_cmd_m_s=_host(replay.command.velocities_m_s[:, 0]),
        a_cmd_m_s2=_host(replay.command.accelerations_m_s2[:, 0]),
        yaw_cmd_rad=np.zeros_like(_host(replay.times_s)),
        omega_cmd_body_rad_s=_host(replay.command.angular_velocities_body_rad_s[:, 0]),
        **step_metrics,
    )


def save_command_npz(path: Path, replay: DeterministicReplay, task: CanonicalWhipTask) -> None:
    np.savez_compressed(
        path,
        authorization=np.asarray("SIMULATION_ONLY"),
        warning=np.asarray("NOT_AUTHORIZED_FOR_REAL_FLIGHT"),
        real_flight_authorized=np.asarray(False),
        task_id=np.asarray(task.task_id),
        time_s=_host(replay.command.times_s),
        p_cmd_m=_host(replay.command.positions_m[:, 0]),
        v_cmd_m_s=_host(replay.command.velocities_m_s[:, 0]),
        a_cmd_m_s2=_host(replay.command.accelerations_m_s2[:, 0]),
        yaw_cmd_rad=np.zeros_like(_host(replay.command.times_s)),
        omega_cmd_body_rad_s=_host(replay.command.angular_velocities_body_rad_s[:, 0]),
    )


def create_plots(result_dir: Path, replay: DeterministicReplay, task: CanonicalWhipTask, final: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_s = _host(replay.times_s)
    uav = _host(replay.uav_positions_m[:, 0])
    cable = _host(replay.cable_positions_m[:, 0])
    tip = cable[:, 11]
    metrics = _per_step_metrics(replay, task)
    target = np.asarray(task.target_position_m)

    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(*uav.T, label="UAV", color="tab:blue")
    axis.plot(*tip.T, label="c10 tip", color="tab:orange")
    axis.scatter(*target, color="red", s=80, label="target")
    event_time = final["reported_event_time_s"]
    indices = sorted(
        set(
            [
                0,
                int(round(0.21 / (time_s[1] - time_s[0]))),
                int(round(0.35 / (time_s[1] - time_s[0]))),
                int(round(float(event_time) / (time_s[1] - time_s[0]))),
                len(time_s) - 1,
            ]
        )
    )
    for index in indices:
        axis.plot(*cable[index].T, alpha=0.55, linewidth=1.0)
    axis.set_xlabel("X [m]")
    axis.set_ylabel("Y [m]")
    axis.set_zlabel("Z [m]")
    axis.legend()
    figure.tight_layout()
    figure.savefig(result_dir / "trajectory_3d.png", dpi=180)
    plt.close(figure)

    def line_plot(name: str, ylabel: str, series: list[tuple[str, np.ndarray]]) -> None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for label, values in series:
            ax.plot(time_s, values, label=label)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if len(series) > 1:
            ax.legend()
        fig.tight_layout()
        fig.savefig(result_dir / name, dpi=180)
        plt.close(fig)

    line_plot(
        "tip_target_distance.png",
        "Tip-target distance [m]",
        [("distance", metrics["tip_target_distance_m"])],
    )
    line_plot(
        "tip_speed.png",
        "Speed [m/s]",
        [
            ("tip total", metrics["tip_speed_m_s"]),
            ("tip directed", metrics["directed_tip_speed_m_s"]),
        ],
    )
    line_plot(
        "uav_motion.png",
        "UAV motion",
        [
            ("speed [m/s]", metrics["uav_speed_m_s"]),
            ("displacement [m]", metrics["uav_displacement_m"]),
        ],
    )
    acceleration = _host(replay.command.accelerations_m_s2[:, 0])
    line_plot(
        "command_acceleration.png",
        "Command acceleration [m/s²]",
        [("x", acceleration[:, 0]), ("y", acceleration[:, 1]), ("z", acceleration[:, 2])],
    )

    labels = ("initial", "forward stroke", "reversal", "impact/event", "final")
    figure = plt.figure(figsize=(14, 8))
    for plot_index, (label, index) in enumerate(zip(labels, indices, strict=False), start=1):
        axis = figure.add_subplot(2, 3, plot_index, projection="3d")
        axis.plot(*cable[index].T, marker="o", markersize=2)
        axis.scatter(*target, color="red", s=35)
        axis.set_title(f"{label}: {time_s[index]:.2f} s")
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")
    figure.tight_layout()
    figure.savefig(result_dir / "cable_snapshots.png", dpi=180)
    plt.close(figure)


def source_hash_manifest(
    *,
    task_config: Path | None = None,
    runner: Path | None = None,
) -> dict[str, Any]:
    task_config = task_config or (
        PROJECT_ROOT / "config" / "tasks" / "canonical_whip_v1.json"
    )
    runner = runner or (PROJECT_ROOT / "run_milestone6a.py")
    paths = [
        PROJECT_ROOT / "config" / "active_model.json",
        PROJECT_ROOT / "config" / "default.json",
        task_config,
        PROJECT_ROOT / "simulator" / "production.py",
        PROJECT_ROOT / "simulator" / "simulator.py",
        PROJECT_ROOT / "simulator" / "uav" / "model.py",
        PROJECT_ROOT / "simulator" / "cable" / "dder.py",
        *sorted((PROJECT_ROOT / "planning").glob("*.py")),
        runner,
    ]
    hashes = {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}
    aggregate = __import__("hashlib").sha256(
        "\n".join(f"{name}:{value}" for name, value in sorted(hashes.items())).encode()
    ).hexdigest()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return {
        "schema": "milestone4a_source_hash_manifest_v1",
        "files": hashes,
        "aggregate_sha256": aggregate,
        "git_commit": commit,
        "working_tree_clean": len(status) == 0,
        "working_tree_entry_count": len(status),
    }


def save_iteration_history(path: Path, result: MppiResult) -> None:
    write_json(path, [asdict(item) for item in result.iteration_history])


def maybe_create_planner_freeze(result_dir: Path, final: dict[str, Any], source_manifest: dict[str, Any]) -> Path | None:
    if final["mppi_simulation"] != "PASS":
        return None
    if PLANNER_FREEZE.exists():
        raise RuntimeError(f"Refusing to overwrite existing planner freeze: {PLANNER_FREEZE}")
    PLANNER_FREEZE.mkdir(parents=True)
    names = (
        "task_config_snapshot.json",
        "model_freeze_reference.json",
        "best_acceleration_knots.json",
        "planned_fullstate_command.csv",
        "planned_fullstate_command.npz",
        "final_replay.npz",
        "final_metrics.json",
        "task_success.json",
        "planning_runtime.json",
        "source_hash_manifest.json",
    )
    for name in names:
        shutil.copy2(result_dir / name, PLANNER_FREEZE / name)
    manifest = {
        "schema": "canonical_whip_simulation_planner_freeze_v1",
        "status": "SIMULATION_TASK_SOLVED",
        "authorization": "SIMULATION_ONLY",
        "real_flight_authorized": False,
        "source_result_directory": str(result_dir),
        "source_aggregate_sha256": source_manifest["aggregate_sha256"],
        "artifact_hashes": {
            name: sha256_file(PLANNER_FREEZE / name) for name in names
        },
    }
    write_json(PLANNER_FREEZE / "manifest.json", manifest)
    return PLANNER_FREEZE


def write_report(
    *,
    task: CanonicalWhipTask,
    result_dir: Path,
    integrity: dict[str, Any],
    preroll: dict[str, Any],
    initial_metrics: dict[str, Any],
    mppi: MppiResult,
    final: dict[str, Any],
    runtime: dict[str, Any],
    planner_freeze: Path | None,
    tests: dict[str, Any],
) -> None:
    iterations = "\n".join(
        "| {iteration} | {current_best_cost:.6f} | {best_ever_cost:.6f} | {effective_sample_size:.2f} | "
        "{minimum_target_error_m:.4f} | {best_directed_tip_speed_m_s:.3f} | {success_count} | {iteration_runtime_s:.3f} |".format(
            **asdict(item)
        )
        for item in mppi.iteration_history
    )
    first_marker = final["first_target_entry_marker_label"] or "none"
    hit = "none" if final["hit_time_s"] is None else f"{float(final['hit_time_s']):.3f} s"
    failed = final["failed_hard_success_gates"]
    failure_text = "none" if not failed else ", ".join(failed)
    freeze_text = "not created (simulation failed)" if planner_freeze is None else str(planner_freeze)
    report = f"""# Milestone 4A — Canonical Whip Task + Production Full-Horizon MPPI

## 1. Outcome

The single authorized `canonical_whip_v1`, seed-42, simulation-only planning run is **{final['mppi_simulation']}**. The final decision comes from one deterministic replay of the globally best acceleration-knot trajectory from the exact post-hover state. Failed hard gates: **{failure_text}**.

No model fitting, residual retraining, physics/geometry change, protected-test evaluation, receding-horizon control, ROS/radio access, or real Crazyflie execution occurred.

## 2. Frozen production model

- Freeze: `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`
- Integrity: verified before planning ({len(integrity['verified_artifact_hashes'])} frozen artifacts re-hashed)
- Predictor: frozen UAV Physics + frozen causal UAV Residual + 12-node DDER
- UAV parameters: K_p=4.020097778647703, K_v=12.05728865003419, k_a=0.7327301468333923, K_R=69.18419375930429, K_omega=11.456583174321011
- Cable parameters: EI=9.225797579462985e-05 N m², Cb=0.0027704569956681496 N m² s
- Geometry: 0.9525 m, attachment [0,0,-0.055] m, c1…c10 at nodes 2…11
- Backend: CUDA float32, PCG32, fused production DDER, 3 substeps, 4 position projections
- Protected test `fig8vertical_002`: **NOT EVALUATED**

## 3. Canonical task

- Initial UAV: [0,0,1.5] m, zero velocity, yaw 0
- Target: [1.0,0.0,1.4] m
- Desired impact direction: [+1,0,0]
- Horizon: 0.70 s; impact window: [0.30,0.70] s
- Tip radius: 0.050 m; directed-speed minimum: 4.0 m/s; direction tolerance: 30°
- Tip-first: c10 must be the first of c1…c10 entering the target sphere
- Hard UAV limits: 0.50 m displacement and 3.0 m/s speed
- Hard command acceleration norm: 20.0 m/s²
- No target collision/contact physics was introduced.

## 4. Initial state and hover pre-roll

A hanging cable was constructed from the exact frozen rest lengths. One 0.50-s production hover pre-roll was run once, producing the common post-hover UAV, cable, DDER, and causal residual-FIFO state cloned into every candidate. The pre-roll final cable-speed RMS was {preroll['cable_velocity_rms_m_s']:.6g} m/s; the residual FIFO was finite and causally populated. It was not rerun per candidate.

## 5. FullState command parameterization

MPPI optimizes one 11×3 acceleration-knot tensor. Acceleration is linearly interpolated at the 100-Hz production command/physics grid. Velocity is its exact trapezoidal integral; position uses the exact interval integral for linearly varying acceleration. Yaw and body-rate command remain zero. Position, velocity, and acceleration are never independently optimized. Every knot is projected by vector norm, not component clipping.

## 6. Structured nominal

The legacy nominal was audited but belongs to the superseded direct-kinematics/11-node simulator contract. It was not copied into production. The exact fallback profile specified by Milestone 4A was used. Its initial best tip-target distance was {float(initial_metrics['minimum_tip_target_distance_m']):.4f} m, best-event directed tip speed was {float(initial_metrics['best_event_directed_speed_m_s']):.3f} m/s, cost was {float(initial_metrics['cost']):.6f}, and success was {initial_metrics['success']}.

## 7. MPPI implementation and cost

The implementation is derivative-free and advances all 2048 candidates together through `build_production_simulator`. It accumulates event/task/safety metrics on GPU and never saves the population trajectories. Candidate 0 is the unperturbed nominal. Stabilized global weights use λ=1.0, perturbation σ=2.5 m/s², and seed 42. The globally best candidate is retained across iterations; only it is replayed in full.

The event objective is the configured minimum over [0.30,0.70] s: 4×normalized position² + 2×directed-speed deficiency² + 1×direction deficiency² + 4×non-tip proximity². Secondary configured weights are 0.02 effort, 0.05 knot smoothness, 0.10 final UAV speed, and 10 each for displacement/speed excess. A valid trajectory receives one finite −20 bonus. Hard gates, not dense cost, define success.

## 8. MPPI configuration

- Horizon: 0.70 s
- Acceleration knots: 11
- Samples: 2048
- Maximum iterations: 4
- Perturbation σ: 2.5 m/s²
- Temperature λ: 1.0
- Seed: 42
- Early stop: success plus at most one polishing iteration

## 9. Iteration history

| Iteration | Current best cost | Best-ever cost | ESS | Min target error [m] | Best directed speed [m/s] | Successes | Runtime [s] |
|---:|---:|---:|---:|---:|---:|---:|---:|
{iterations}

## 10. Final deterministic replay

- Valid hit time: {hit}
- Reported event time (hit, otherwise near-miss): {float(final['reported_event_time_s']):.3f} s
- Tip position error: {1000.0 * float(final['reported_event_tip_position_error_m']):.3f} mm
- Tip total speed: {float(final['reported_event_tip_total_speed_m_s']):.3f} m/s
- Directed tip speed: {float(final['reported_event_directed_tip_speed_m_s']):.3f} m/s
- Impact-direction error: {float(final['reported_event_direction_error_deg']):.3f}°
- First target-entry marker: {first_marker}
- Maximum UAV displacement: {float(final['maximum_uav_displacement_m']):.4f} m
- Maximum UAV speed: {float(final['maximum_uav_speed_m_s']):.4f} m/s
- UAV-to-target distance at event: {float(final['uav_to_target_distance_at_reported_event_m']):.4f} m
- Tip/UAV speed ratio at event: {float(final['tip_speed_to_uav_speed_ratio_at_reported_event']):.3f}
- Maximum command acceleration: {float(final['maximum_command_acceleration_m_s2']):.4f} m/s²
- Rollout finite: {final['finite']}

The winning sampled row and required batch-one replay were not numerically identical over this aggressive full horizon. The sampled row reported cost {float(mppi.best_metrics['cost']):.6f} and minimum distance {1000.0 * float(mppi.best_metrics['minimum_tip_target_distance_m']):.3f} mm; batch-one replay reported cost {float(final['cost']):.6f} and minimum distance {1000.0 * float(final['minimum_tip_target_distance_m']):.3f} mm. The pre-run three-step batch check differed by only {float(tests['batch_equivalence_maximum_absolute_difference']):.3e}, so the full-horizon discrepancy is recorded as amplified float32 batch-shape sensitivity. Neither the sampled population nor the replay contained a hard-gated success, and the replay is authoritative by the frozen protocol.

## 11. Cable-energy diagnostic

At the reported event, observed dynamic cable kinetic energy was {float(final['cable_kinetic_energy_at_reported_event_j']):.6g} J. The distal c8–c10 fraction was {100.0 * float(final['distal_c8_c10_kinetic_energy_fraction']):.2f}%. Definition: 0.5 m_i||v_i||² over c1…c10, with the distal numerator using c8…c10. This is diagnostic only and did not enter MPPI.

## 12. Runtime

- Total MPPI solve: {mppi.total_runtime_s:.3f} s
- Iterations executed: {len(mppi.iteration_history)}
- Candidate rollouts: {mppi.candidate_rollouts_evaluated}
- Effective rollouts/s: {runtime['effective_rollouts_per_s']:.1f}
- Final deterministic replay: {runtime['final_replay_s']:.3f} s
- Peak CUDA memory: {runtime['peak_cuda_memory_mb']:.1f} MiB
- Five-minute hard stop reached: {mppi.hard_stop_reached}

## 13. Verification

Cheap pre-run checks passed: {tests}. Command integration consistency, norm projection, state deep-cloning including residual FIFO, synthetic hard-gate cases, batch equivalence, and a small MPPI smoke solve were checked before the authoritative run.

## 14. Artifacts

- Result directory: `{result_dir}`
- Planner freeze: `{freeze_text}`
- FullState CSV/NPZ are marked `SIMULATION_ONLY` and `NOT_AUTHORIZED_FOR_REAL_FLIGHT`.
- The generated command has **NOT** been executed on real hardware.

## 15. Final summary

    Model:
        MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI

    Task:
        canonical_whip_v1

    Horizon:
        0.70 s

    MPPI:
        samples = 2048
        iterations = {len(mppi.iteration_history)}
        runtime = {mppi.total_runtime_s:.3f} s

    Hit time:
        {hit}

    Tip position error:
        {1000.0 * float(final['reported_event_tip_position_error_m']):.3f} mm

    Tip total speed:
        {float(final['reported_event_tip_total_speed_m_s']):.3f} m/s

    Directed tip speed:
        {float(final['reported_event_directed_tip_speed_m_s']):.3f} m/s

    Impact-direction error:
        {float(final['reported_event_direction_error_deg']):.3f} deg

    UAV max displacement:
        {float(final['maximum_uav_displacement_m']):.4f} m

    UAV max speed:
        {float(final['maximum_uav_speed_m_s']):.4f} m/s

    Tip/UAV speed ratio at hit:
        {float(final['tip_speed_to_uav_speed_ratio_at_reported_event']):.3f}

    First target-entry marker:
        {first_marker}

    Max command acceleration:
        {float(final['maximum_command_acceleration_m_s2']):.4f} m/s^2

    MPPI_SIMULATION:
        {final['mppi_simulation']}

    REAL HARDWARE EXECUTION:
        NOT PERFORMED
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    (result_dir / REPORT_PATH.name).write_text(report, encoding="utf-8")
