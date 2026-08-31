"""Run one Milestone-4B.1 simulation-only full-horizon MPPI task."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from planning.artifacts import (
    create_plots,
    create_result_directory,
    final_replay_metrics,
    save_command_csv,
    save_command_npz,
    save_iteration_history,
    save_replay_npz,
    source_hash_manifest,
    verify_planning_model_integrity,
    write_json,
)
from planning.mppi import optimize_full_horizon_mppi
from planning.results import load_planning_result
from planning.rollout import hover_preroll, replay_selected_trajectory
from planning.task import DEFAULT_CANONICAL_TASK, load_canonical_whip_task
from planning.video import render_replay_video
from simulator.parameters import SimulatorSettings
from simulator.production import (
    active_model_paths,
    build_production_simulator,
    load_active_model_manifest,
)


ROOT = Path(__file__).resolve().parent


def _maximum_vector_difference(left: object, right: object) -> float:
    return max(
        abs(float(left_value) - float(right_value))
        for left_value, right_value in zip(left, right)
    )


def _sampled_replay_comparison(
    sampled: dict[str, object], replay: dict[str, object]
) -> dict[str, object]:
    final_uav_difference = _maximum_vector_difference(
        sampled["final_uav_position_m"], replay["final_uav_position_m"]
    )
    final_tip_difference = _maximum_vector_difference(
        sampled["final_c10_position_m"], replay["final_c10_position_m"]
    )
    minimum_distance_difference = abs(
        float(sampled["minimum_tip_target_distance_m"])
        - float(replay["minimum_tip_target_distance_m"])
    )
    classification_equal = (
        bool(sampled["success"]) == bool(replay["success"])
        and sampled["first_entry_marker"] == replay["first_entry_marker"]
    )
    passed = bool(
        final_uav_difference <= 0.001
        and final_tip_difference <= 0.002
        and minimum_distance_difference <= 0.002
        and classification_equal
    )
    return {
        "sampled_winner": sampled,
        "batch_one_authoritative_replay": replay,
        "fixed_uav_evaluation_shape_used": True,
        "final_uav_position_difference_m": final_uav_difference,
        "final_c10_position_difference_m": final_tip_difference,
        "minimum_tip_target_distance_difference_m": minimum_distance_difference,
        "success_and_first_entry_classification_identical": classification_equal,
        "tolerances": {
            "final_uav_position_m": 0.001,
            "final_c10_position_m": 0.002,
            "minimum_tip_target_distance_m": 0.002,
        },
        "pass": passed,
    }


def run_task(task_path: Path) -> dict[str, object]:
    task = load_canonical_whip_task(task_path)
    active = load_active_model_manifest()
    settings = SimulatorSettings.load(active_model_paths(active)["configuration"])
    task.validate_for_dt(settings.dt_s)
    integrity = verify_planning_model_integrity(settings, task)
    if not torch.cuda.is_available():
        raise RuntimeError("Milestone 4B.1 requires the frozen CUDA backend.")
    simulator = build_production_simulator(
        settings, device="cuda", dtype=torch.float32
    )
    backend = simulator.forward_backend_audit()
    if not backend["production_fused_forward_active"]:
        raise RuntimeError("Fused production DDER backend is not active.")
    if backend["damping_backend"] != "pcg32_experimental":
        raise RuntimeError("Milestone 4B.1 requires frozen PCG32.")
    simulator.uav_model.set_fixed_evaluation_batch_size(
        task.mppi.fixed_uav_evaluation_batch_size
    )

    result_dir = create_result_directory(task.task_id)
    write_json(result_dir / "task_config_snapshot.json", task.snapshot())
    write_json(result_dir / "model_freeze_reference.json", integrity)
    initial_state = hover_preroll(simulator, task)
    write_json(
        result_dir / "hover_preroll.json",
        {
            "duration_s": task.hover_preroll_s,
            "steps": int(round(task.hover_preroll_s / settings.dt_s)),
            "residual_history_shape": list(
                initial_state.uav.residual_history.features.shape
            ),
            "fixed_uav_evaluation_batch_size": (
                simulator.uav_model.fixed_evaluation_batch_size
            ),
        },
    )
    nominal = task.nominal_knots(device=simulator.device, dtype=simulator.dtype)
    initial_replay = replay_selected_trajectory(
        simulator, initial_state, nominal, task
    )
    initial_metrics = final_replay_metrics(initial_replay, task, settings)
    write_json(result_dir / "initial_nominal_metrics.json", initial_metrics)
    save_command_csv(result_dir / "initial_nominal_command.csv", initial_replay)

    torch.cuda.reset_peak_memory_stats(simulator.device)

    def report_iteration(record) -> None:
        payload = asdict(record)
        payload.update(
            {
                "maximum_iterations": task.mppi.maximum_iterations,
                "samples": task.mppi.sample_count,
            }
        )
        print("MPPI_PROGRESS " + json.dumps(payload, sort_keys=True), flush=True)

    mppi = optimize_full_horizon_mppi(
        simulator,
        initial_state,
        nominal,
        task,
        iteration_callback=report_iteration,
    )
    save_iteration_history(result_dir / "mppi_iteration_history.json", mppi)
    write_json(
        result_dir / "best_acceleration_knots.json",
        {
            "selection": "best feasible task cost when available; otherwise lowest feasibility violation",
            "best_iteration": mppi.best_iteration,
            "best_row_index": mppi.best_row_index,
            "units": "m/s^2",
            "values": mppi.best_knots_m_s2.detach().cpu().tolist(),
        },
    )
    write_json(
        result_dir / "best_feasible_metrics.json",
        mppi.best_feasible_metrics
        or {"available": False, "reason": "No feasible candidate was sampled."},
    )
    write_json(
        result_dir / "best_overall_metrics.json",
        mppi.best_overall_metrics,
    )
    if mppi.best_feasible_knots_m_s2 is not None:
        write_json(
            result_dir / "best_feasible_acceleration_knots.json",
            {"units": "m/s^2", "values": mppi.best_feasible_knots_m_s2.detach().cpu().tolist()},
        )
    write_json(
        result_dir / "best_overall_acceleration_knots.json",
        {"units": "m/s^2", "values": mppi.best_overall_knots_m_s2.detach().cpu().tolist()},
    )

    torch.cuda.synchronize(simulator.device)
    replay_start = time.perf_counter()
    final_replay = replay_selected_trajectory(
        simulator, initial_state, mppi.best_knots_m_s2, task
    )
    torch.cuda.synchronize(simulator.device)
    replay_runtime_s = time.perf_counter() - replay_start
    final = final_replay_metrics(final_replay, task, settings)
    comparison = _sampled_replay_comparison(mppi.best_metrics, final)
    classification = (
        "NUMERICALLY_UNTRUSTWORTHY"
        if not comparison["pass"]
        else "PASS"
        if final["success"]
        else "FAIL"
    )
    final["numerical_consistency_pass"] = comparison["pass"]
    final["task_classification"] = classification
    write_json(result_dir / "final_metrics.json", final)
    write_json(result_dir / "sampled_winner_vs_replay.json", comparison)
    write_json(
        result_dir / "task_success.json",
        {
            "task_id": task.task_id,
            "classification": classification,
            "success": bool(final["success"]) and bool(comparison["pass"]),
            "hard_success_gates": final["hard_success_gates"],
            "failed_hard_success_gates": final["failed_hard_success_gates"],
            "numerical_consistency_pass": comparison["pass"],
            "authorization": "SIMULATION_ONLY",
            "real_hardware_execution": "NOT EXECUTED",
        },
    )
    save_command_csv(result_dir / "planned_fullstate_command.csv", final_replay)
    save_command_npz(
        result_dir / "planned_fullstate_command.npz", final_replay, task
    )
    save_replay_npz(result_dir / "final_replay.npz", final_replay, task)
    create_plots(result_dir, final_replay, task, final)
    peak_memory_mb = torch.cuda.max_memory_allocated(simulator.device) / (1024.0**2)
    runtime = {
        "total_mppi_solve_s": mppi.total_runtime_s,
        "iteration_runtime_s": [
            item.iteration_runtime_s for item in mppi.iteration_history
        ],
        "candidate_rollouts_evaluated": mppi.candidate_rollouts_evaluated,
        "effective_rollouts_per_s": (
            mppi.candidate_rollouts_evaluated / max(mppi.total_runtime_s, 1.0e-12)
        ),
        "final_replay_s": replay_runtime_s,
        "peak_cuda_memory_mb": peak_memory_mb,
        "device": torch.cuda.get_device_name(simulator.device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    write_json(result_dir / "planning_runtime.json", runtime)
    write_json(
        result_dir / "email_delivery.json",
        {"status": "NOT_AVAILABLE_IN_LOCAL_ENVIRONMENT"},
    )
    write_json(
        result_dir / "source_hash_manifest.json",
        source_hash_manifest(task_config=task_path, runner=Path(__file__)),
    )
    video_path, video_metadata = render_replay_video(
        load_planning_result(result_dir)
    )
    output = {
        "task_id": task.task_id,
        "classification": classification,
        "result_directory": str(result_dir),
        "video": str(video_path),
        "video_metadata": video_metadata,
        "final_metrics": final,
        "sampled_winner_vs_replay": comparison,
        "runtime": runtime,
    }
    print("MPPI_RESULT " + json.dumps(output, sort_keys=True), flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, default=DEFAULT_CANONICAL_TASK)
    parser.add_argument("--result-json", type=Path)
    arguments = parser.parse_args()
    output = run_task(arguments.task.resolve())
    if arguments.result_json is not None:
        write_json(arguments.result_json.resolve(), output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
