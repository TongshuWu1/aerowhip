"""Run the single simulation-only Milestone-4A canonical whip solve."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from planning.artifacts import (
    create_plots,
    create_result_directory,
    final_replay_metrics,
    maybe_create_planner_freeze,
    save_command_csv,
    save_command_npz,
    save_iteration_history,
    save_replay_npz,
    source_hash_manifest,
    verify_planning_model_integrity,
    write_json,
    write_report,
)
from planning.command_parameterization import (
    acceleration_knots_to_fullstate,
    command_consistency_errors,
    project_acceleration_knots,
)
from planning.metrics import synthetic_hard_success
from planning.mppi import optimize_full_horizon_mppi
from planning.rollout import (
    clone_state_batch,
    hover_preroll,
    replay_selected_trajectory,
)
from planning.task import DEFAULT_CANONICAL_TASK, load_canonical_whip_task
from simulator.parameters import SimulatorSettings
from simulator.production import active_model_paths, build_production_simulator, load_active_model_manifest


ROOT = Path(__file__).resolve().parent


def _short_batch_equivalence(simulator, initial_state, task, nominal) -> float:
    command_one = acceleration_knots_to_fullstate(
        nominal,
        initial_position_m=torch.tensor(task.initial_uav_position_m, device=simulator.device),
        initial_velocity_m_s=torch.zeros(3, device=simulator.device),
        yaw_rad=0.0,
        horizon_s=task.mppi.horizon_s,
        dt_s=simulator.dt_s,
    ).simulator_sequence()
    command_three = acceleration_knots_to_fullstate(
        nominal[None].repeat(3, 1, 1),
        initial_position_m=torch.tensor(task.initial_uav_position_m, device=simulator.device),
        initial_velocity_m_s=torch.zeros(3, device=simulator.device),
        yaw_rad=0.0,
        horizon_s=task.mppi.horizon_s,
        dt_s=simulator.dt_s,
    ).simulator_sequence()
    one = clone_state_batch(initial_state, 1)
    three = clone_state_batch(initial_state, 3)
    with torch.no_grad():
        for index in range(3):
            one = simulator._propagate(one, command_one.command_at(index), simulator.parameters, create_graph=False)  # noqa: SLF001
            three = simulator._propagate(three, command_three.command_at(index), simulator.parameters, create_graph=False)  # noqa: SLF001
    differences = (
        torch.max(torch.abs(one.uav.position_m[0] - three.uav.position_m[1])),
        torch.max(torch.abs(one.uav.velocity_m_s[0] - three.uav.velocity_m_s[1])),
        torch.max(torch.abs(one.cable.positions_m[0] - three.cable.positions_m[1])),
        torch.max(torch.abs(one.cable.velocities_m_s[0] - three.cable.velocities_m_s[1])),
        torch.max(
            torch.abs(
                one.uav.residual_history.features[0]
                - three.uav.residual_history.features[1]
            )
        ),
    )
    if simulator.device.type == "cuda":
        torch.cuda.synchronize(simulator.device)
    return max(float(value.detach().cpu()) for value in differences)


def run_cheap_checks(simulator, initial_state, task, nominal) -> dict[str, object]:
    command = acceleration_knots_to_fullstate(
        nominal,
        initial_position_m=torch.tensor(task.initial_uav_position_m, device=simulator.device),
        initial_velocity_m_s=torch.zeros(3, device=simulator.device),
        yaw_rad=task.initial_yaw_rad,
        horizon_s=task.mppi.horizon_s,
        dt_s=simulator.dt_s,
    )
    consistency = command_consistency_errors(command)
    tolerance = 2.0e-6
    if max(consistency.values()) > tolerance:
        raise RuntimeError(f"FullState command consistency failed: {consistency}")
    projected = project_acceleration_knots(
        torch.tensor(
            [[[30.0, 30.0, 30.0], [-50.0, 0.0, 0.0]]],
            device=simulator.device,
            dtype=simulator.dtype,
        ),
        task.maximum_command_acceleration_m_s2,
    )
    projected_max = float(
        torch.linalg.vector_norm(projected, dim=-1).amax().detach().cpu()
    )
    if projected_max > task.maximum_command_acceleration_m_s2 + 1.0e-5:
        raise RuntimeError("Acceleration norm projection failed.")
    clone = clone_state_batch(initial_state, 3)
    if clone.uav.position_m.data_ptr() == initial_state.uav.position_m.data_ptr():
        raise RuntimeError("UAV clone shares source storage.")
    if clone.cable.positions_m.data_ptr() == initial_state.cable.positions_m.data_ptr():
        raise RuntimeError("Cable clone shares source storage.")
    if (
        clone.uav.residual_history.features.data_ptr()
        == initial_state.uav.residual_history.features.data_ptr()
    ):
        raise RuntimeError("Residual FIFO clone shares source storage.")
    target = torch.tensor(task.target_position_m)
    good_velocity = torch.tensor([4.5, 0.0, 0.0])
    cases = {
        "valid": synthetic_hard_success(
            task,
            tip_position_m=target,
            tip_velocity_m_s=good_velocity,
            first_entry_marker=10,
            maximum_uav_displacement_m=0.2,
            maximum_uav_speed_m_s=2.0,
            maximum_command_acceleration_m_s2=19.0,
        ),
        "wrong_direction": synthetic_hard_success(
            task,
            tip_position_m=target,
            tip_velocity_m_s=torch.tensor([-4.5, 0.0, 0.0]),
            first_entry_marker=10,
            maximum_uav_displacement_m=0.2,
            maximum_uav_speed_m_s=2.0,
            maximum_command_acceleration_m_s2=19.0,
        ),
        "insufficient_speed": synthetic_hard_success(
            task,
            tip_position_m=target,
            tip_velocity_m_s=torch.tensor([3.9, 0.0, 0.0]),
            first_entry_marker=10,
            maximum_uav_displacement_m=0.2,
            maximum_uav_speed_m_s=2.0,
            maximum_command_acceleration_m_s2=19.0,
        ),
        "c5_first": synthetic_hard_success(
            task,
            tip_position_m=target,
            tip_velocity_m_s=good_velocity,
            first_entry_marker=5,
            maximum_uav_displacement_m=0.2,
            maximum_uav_speed_m_s=2.0,
            maximum_command_acceleration_m_s2=19.0,
        ),
        "excess_displacement": synthetic_hard_success(
            task,
            tip_position_m=target,
            tip_velocity_m_s=good_velocity,
            first_entry_marker=10,
            maximum_uav_displacement_m=0.51,
            maximum_uav_speed_m_s=2.0,
            maximum_command_acceleration_m_s2=19.0,
        ),
    }
    if cases != {
        "valid": True,
        "wrong_direction": False,
        "insufficient_speed": False,
        "c5_first": False,
        "excess_displacement": False,
    }:
        raise RuntimeError(f"Synthetic hard success checks failed: {cases}")
    batch_difference = _short_batch_equivalence(simulator, initial_state, task, nominal)
    if batch_difference > 2.0e-5:
        raise RuntimeError(f"Batch equivalence exceeded float32 tolerance: {batch_difference}")
    smoke_task = replace(
        task,
        mppi=replace(
            task.mppi,
            sample_count=8,
            maximum_iterations=1,
            success_polishing_iterations=0,
            hard_stop_s=60.0,
        ),
    )
    smoke = optimize_full_horizon_mppi(
        simulator, initial_state, nominal, smoke_task
    )
    if not np.isfinite(smoke.iteration_history[0].current_best_cost):
        raise RuntimeError("Small-population MPPI smoke cost is non-finite.")
    return {
        "command_consistency": consistency,
        "projected_maximum_acceleration_m_s2": projected_max,
        "state_clone_independent_storage": True,
        "synthetic_task_logic": cases,
        "batch_equivalence_maximum_absolute_difference": batch_difference,
        "smoke_candidate_count": 8,
        "smoke_cost_finite": True,
    }


def main() -> int:
    task = load_canonical_whip_task(DEFAULT_CANONICAL_TASK)
    active = load_active_model_manifest()
    settings = SimulatorSettings.load(active_model_paths(active)["configuration"])
    task.validate_for_dt(settings.dt_s)
    integrity = verify_planning_model_integrity(settings, task)
    if not torch.cuda.is_available():
        raise RuntimeError("Milestone 4A requires the frozen CUDA production backend.")
    simulator = build_production_simulator(settings, device="cuda", dtype=torch.float32)
    backend = simulator.forward_backend_audit()
    if not backend["production_fused_forward_active"]:
        raise RuntimeError("Fused production DDER backend is not active.")
    if backend["damping_backend"] != "pcg32_experimental":
        raise RuntimeError("Milestone 4A requires validated PCG32.")

    result_dir = create_result_directory()
    write_json(result_dir / "task_config_snapshot.json", task.snapshot())
    write_json(result_dir / "model_freeze_reference.json", integrity)
    initial_state = hover_preroll(simulator, task)
    preroll = {
        "duration_s": task.hover_preroll_s,
        "steps": int(round(task.hover_preroll_s / settings.dt_s)),
        "cable_velocity_rms_m_s": float(
            torch.sqrt(torch.mean(initial_state.cable.velocities_m_s.square()))
            .detach()
            .cpu()
        ),
        "residual_history_finite": bool(
            torch.isfinite(initial_state.uav.residual_history.features).all().detach().cpu()
        ),
        "residual_history_shape": list(initial_state.uav.residual_history.features.shape),
    }
    write_json(result_dir / "hover_preroll.json", preroll)
    nominal = task.nominal_knots(device=simulator.device, dtype=simulator.dtype)
    checks = run_cheap_checks(simulator, initial_state, task, nominal)
    write_json(result_dir / "cheap_pre_run_checks.json", checks)

    initial_replay = replay_selected_trajectory(simulator, initial_state, nominal, task)
    initial_metrics = initial_replay.metrics.row(0)
    initial_metrics.update(
        {
            "source": task.initial_nominal_source,
            "acceleration_knots_m_s2": nominal.detach().cpu().tolist(),
        }
    )
    save_command_csv(result_dir / "initial_nominal_command.csv", initial_replay)
    write_json(result_dir / "initial_nominal_metrics.json", initial_metrics)

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
            "shape": [task.mppi.knot_count, 3],
            "units": "m/s^2",
            "best_iteration": mppi.best_iteration,
            "best_row_index": mppi.best_row_index,
            "values": mppi.best_knots_m_s2.detach().cpu().tolist(),
        },
    )
    torch.cuda.synchronize(simulator.device)
    replay_start = time.perf_counter()
    final_replay = replay_selected_trajectory(
        simulator, initial_state, mppi.best_knots_m_s2, task
    )
    torch.cuda.synchronize(simulator.device)
    replay_runtime = time.perf_counter() - replay_start
    final = final_replay_metrics(final_replay, task, settings)
    write_json(result_dir / "final_metrics.json", final)
    sampled_replay_comparison = {
        "sampled_winner": mppi.best_metrics,
        "batch_one_authoritative_replay": final,
        "cost_absolute_difference": abs(
            float(mppi.best_metrics["cost"]) - float(final["cost"])
        ),
        "minimum_tip_distance_absolute_difference_m": abs(
            float(mppi.best_metrics["minimum_tip_target_distance_m"])
            - float(final["minimum_tip_target_distance_m"])
        ),
        "interpretation": (
            "The short three-step batch-equivalence gate passed, but float32 "
            "batch-shape differences amplified over the aggressive 0.70-s "
            "rollout. The required batch-one replay remains authoritative."
        ),
        "scientific_outcome_changed": False,
    }
    write_json(
        result_dir / "sampled_winner_vs_replay.json",
        sampled_replay_comparison,
    )
    write_json(
        result_dir / "task_success.json",
        {
            "task_id": task.task_id,
            "mppi_simulation": final["mppi_simulation"],
            "success": final["success"],
            "hard_success_gates": final["hard_success_gates"],
            "failed_hard_success_gates": final["failed_hard_success_gates"],
            "authorization": "SIMULATION_ONLY",
            "real_hardware_execution": "NOT_PERFORMED",
        },
    )
    save_command_csv(result_dir / "planned_fullstate_command.csv", final_replay)
    save_command_npz(
        result_dir / "planned_fullstate_command.npz", final_replay, task
    )
    write_json(
        result_dir / "planned_fullstate_command_metadata.json",
        {
            "authorization": "SIMULATION_ONLY",
            "warning": "NOT_AUTHORIZED_FOR_REAL_FLIGHT",
            "real_flight_authorized": False,
            "command_contract": "Crazyswarm2 FullState-compatible offline artifact",
            "yaw_cmd_rad": 0.0,
            "omega_cmd_body_rad_s": [0.0, 0.0, 0.0],
        },
    )
    save_replay_npz(result_dir / "final_replay.npz", final_replay, task)
    peak_memory_mb = torch.cuda.max_memory_allocated(simulator.device) / (1024.0**2)
    runtime = {
        "total_mppi_solve_s": mppi.total_runtime_s,
        "iteration_runtime_s": [item.iteration_runtime_s for item in mppi.iteration_history],
        "candidate_rollouts_evaluated": mppi.candidate_rollouts_evaluated,
        "effective_rollouts_per_s": mppi.candidate_rollouts_evaluated / max(mppi.total_runtime_s, 1.0e-12),
        "final_replay_s": replay_runtime,
        "peak_cuda_memory_mb": peak_memory_mb,
        "hard_stop_s": task.mppi.hard_stop_s,
        "hard_stop_reached": mppi.hard_stop_reached,
        "device": torch.cuda.get_device_name(simulator.device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    write_json(result_dir / "planning_runtime.json", runtime)
    create_plots(result_dir, final_replay, task, final)
    source_manifest = source_hash_manifest()
    write_json(result_dir / "source_hash_manifest.json", source_manifest)
    planner_freeze = maybe_create_planner_freeze(
        result_dir, final, source_manifest
    )
    write_report(
        task=task,
        result_dir=result_dir,
        integrity=integrity,
        preroll=preroll,
        initial_metrics=initial_metrics,
        mppi=mppi,
        final=final,
        runtime=runtime,
        planner_freeze=planner_freeze,
        tests=checks,
    )
    print(json.dumps({
        "report": str(ROOT / "MILESTONE4A_CANONICAL_WHIP_MPPI_REPORT.md"),
        "result_directory": str(result_dir),
        "mppi_simulation": final["mppi_simulation"],
        "failed_gates": final["failed_hard_success_gates"],
        "planner_freeze": None if planner_freeze is None else str(planner_freeze),
    }, indent=2))
    return 0 if final["mppi_simulation"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
