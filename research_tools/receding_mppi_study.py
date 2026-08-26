"""Evaluate full-state receding-horizon DDER-MPPI for drone whipping.

This harness does not redefine the DDER dynamics or MPPI objective.  It uses
the production simulator, optimizer, hard contact/safety evaluator, and a
saved successful forward-recoil plan as the first warm start.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import median
from typing import Iterable

import numpy as np
import torch

from drone_mpc.model import load_cable_model
from drone_mpc.problem import MpcProblem
from drone_mpc.mppi import MppiSettings
from drone_mpc.perfect_model import PerfectMpcSettings
from drone_mpc.receding_mppi import (
    RecedingMppiExecution,
    RecedingMppiSettings,
    perturb_cable_state,
    replay_open_loop,
    run_receding_horizon_mppi,
)
from drone_mpc.simulator import DroneCableState, WhipSimulator
from optitrack_offline.config import DEFAULT_MODEL_PATH


SCHEMA = "full_state_receding_dder_mppi_v1"
REFERENCE = Path(
    "data/drone_mpc/ablations/mppi_discovery_pilot/"
    "initialization_forward_recoil__T2__k11__vg0_12__pw0__"
    "pilot__i6__n128__b128__seed17.npz"
)
DEFAULT_OUTPUT = Path(
    "data/drone_mpc/ablations/full_state_receding_dder_mppi"
)
EFFICIENCY_SAMPLES = (32, 64, 128, 256)
EFFICIENCY_ITERATIONS = (1, 2, 4)
EFFICIENCY_SEEDS = (11, 17, 29, 43, 71)
COMPARISON_SEEDS = (11, 17, 29)


@dataclass(frozen=True, slots=True)
class PerturbationSpec:
    label: str
    time_s: float
    position_delta_m: tuple[float, float, float]
    velocity_delta_m_s: tuple[float, float, float]
    profile: str


PERTURBATIONS = (
    PerturbationSpec("unperturbed", math.inf, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "sway"),
    PerturbationSpec("forward_sway", 0.0, (0.030, 0.0, 0.0), (0.0, 0.0, 0.0), "sway"),
    PerturbationSpec("backward_sway", 0.0, (-0.030, 0.0, 0.0), (0.0, 0.0, 0.0), "sway"),
    PerturbationSpec("lateral_sway", 0.0, (0.0, 0.030, 0.0), (0.0, 0.0, 0.0), "sway"),
    PerturbationSpec("distributed_velocity", 0.0, (0.0, 0.0, 0.0), (0.0, 0.40, 0.0), "interior"),
    PerturbationSpec("before_reversal", 0.30, (0.0, 0.0, 0.0), (0.0, 0.40, 0.0), "interior"),
    PerturbationSpec("after_reversal", 0.50, (0.0, 0.0, 0.0), (-0.40, 0.0, 0.0), "interior"),
    PerturbationSpec("distal_lash", 0.60, (0.0, 0.0, 0.0), (0.0, -0.55, 0.0), "distal"),
)


def _read_metadata(path: Path) -> dict[str, object]:
    payload = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Reference metadata must be an object.")
    return payload


def _load_configuration(
    reference: Path,
) -> tuple[PerfectMpcSettings, MpcProblem, np.ndarray, np.ndarray, dict[str, object]]:
    payload = _read_metadata(reference)
    raw_settings = payload.get("settings")
    raw_problem = payload.get("problem")
    raw_parameterization = payload.get("control_parameterization")
    if not isinstance(raw_settings, dict) or not isinstance(raw_problem, dict):
        raise ValueError("Reference settings/problem are missing.")
    if not isinstance(raw_parameterization, dict):
        raise ValueError("Reference control parameterization is missing.")
    known = {field.name for field in fields(PerfectMpcSettings)}
    settings = PerfectMpcSettings(
        **{name: value for name, value in raw_settings.items() if name in known}
    )
    settings = replace(settings, horizon_s=2.0, mppi_knot_count=11)
    # The current established task uses the user-selected 3.5 m/s threshold;
    # all other target/contact/safety fields come directly from the reference.
    problem = replace(
        MpcProblem(**raw_problem), minimum_impact_speed_m_s=3.5
    )
    warm_knots = np.asarray(raw_parameterization["knots_m_s2"], dtype=np.float32)
    with np.load(reference) as archive:
        controls = np.asarray(archive["accelerations_m_s2"], dtype=np.float32)
    return settings, problem, warm_knots, controls, payload


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    names = sorted({name for row in rows for name in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _load_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Checkpoint must contain a list: {path}")
    return payload


def _failure_mode(execution: RecedingMppiExecution, problem: MpcProblem) -> str:
    terms = execution.cost_terms
    if execution.feasible:
        return "valid"
    if float(terms["non_tip_contact_violation"]) > 0.0:
        return "non_tip_first"
    if float(terms["position_error_m"]) > problem.maximum_tip_error_m:
        return "position_miss"
    if float(terms["directional_speed_m_s"]) < problem.minimum_impact_speed_m_s:
        return "wrong_speed"
    if float(terms["direction_error_deg"]) > problem.maximum_impact_angle_deg:
        return "wrong_direction"
    return "safety_failure"


def _execution_row(
    execution: RecedingMppiExecution,
    problem: MpcProblem,
    *,
    study: str,
    run_key: str,
    samples: int,
    iterations: int,
    seed: int,
    replan_interval_s: float,
    condition: str,
    feedback: str,
) -> dict[str, object]:
    terms = execution.cost_terms
    planning = [update.planning_wall_time_s for update in execution.updates]
    prefix_correction = [update.prefix_correction_l2_m_s2 for update in execution.updates]
    return {
        "schema": SCHEMA,
        "study": study,
        "run_key": run_key,
        "condition": condition,
        "feedback": feedback,
        "samples": samples,
        "iterations": iterations,
        "seed": seed,
        "replan_interval_s": replan_interval_s,
        "requested_replan_rate_hz": 1.0 / replan_interval_s,
        "updates": len(execution.updates),
        "valid_strike": execution.feasible,
        "terminal_reason": execution.terminal_reason,
        "failure_mode": _failure_mode(execution, problem),
        "tip_first": bool(terms["geometric_tip_contact"])
        and float(terms["non_tip_contact_violation"]) == 0.0,
        "target_error_m": float(terms["position_error_m"]),
        "directed_tip_speed_m_s": float(terms["directional_speed_m_s"]),
        "total_tip_speed_m_s": float(terms["tip_speed_m_s"]),
        "direction_error_deg": float(terms["direction_error_deg"]),
        "impact_time_s": execution.impact_time_s,
        "non_tip_first_failure": float(terms["non_tip_contact_violation"]) > 0.0,
        "non_tip_clearance_m": float(terms["minimum_non_tip_target_distance_m"])
        - problem.maximum_tip_error_m,
        "maximum_drone_speed_m_s": float(terms["maximum_drone_speed_m_s"]),
        "maximum_drone_displacement_m": float(terms["maximum_drone_excursion_m"]),
        "real_mppi_cost": execution.cost,
        "median_replanning_wall_s": median(planning) if planning else 0.0,
        "maximum_replanning_wall_s": max(planning) if planning else 0.0,
        "mean_replanning_wall_s": float(np.mean(planning)) if planning else 0.0,
        "deadline_met_pct": (
            100.0 * sum(value <= replan_interval_s for value in planning) / len(planning)
            if planning
            else 100.0
        ),
        "total_planning_wall_s": execution.total_planning_wall_time_s,
        "simulated_execution_time_s": float(execution.result.time_s[-1]),
        "total_rollouts": execution.total_rollouts,
        "rollouts_per_successful_strike": (
            execution.total_rollouts if execution.feasible else ""
        ),
        "total_prefix_correction_l2_m_s2": float(np.sum(prefix_correction)),
    }


def _save_execution(
    path: Path,
    execution: RecedingMppiExecution,
    row: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result = execution.result
    arrays: dict[str, np.ndarray] = {
        "time_s": result.time_s,
        "drone_positions_m": result.drone_positions_m,
        "drone_velocities_m_s": result.drone_velocities_m_s,
        "attachment_positions_m": result.attachment_positions_m,
        "cable_positions_m": result.cable_positions_m,
        "cable_velocities_m_s": result.cable_velocities_m_s,
        "accelerations_m_s2": result.accelerations_m_s2,
        "target_position_m": result.target_position_m,
        "impact_direction": result.impact_direction,
    }
    if execution.updates:
        arrays.update(
            {
                "prediction_start_times_s": np.asarray(
                    [update.start_time_s for update in execution.updates],
                    dtype=np.float32,
                ),
                "replanning_wall_times_s": np.asarray(
                    [update.planning_wall_time_s for update in execution.updates],
                    dtype=np.float32,
                ),
                "nominal_knots_m_s2": np.stack(
                    [update.nominal_knots_m_s2 for update in execution.updates]
                ),
                "optimized_knots_m_s2": np.stack(
                    [update.optimized_knots_m_s2 for update in execution.updates]
                ),
                "predicted_drone_positions_m": np.stack(
                    [update.prediction.drone_positions_m for update in execution.updates]
                ),
                "predicted_cable_positions_m": np.stack(
                    [update.prediction.cable_positions_m for update in execution.updates]
                ),
                "predicted_cable_velocities_m_s": np.stack(
                    [update.prediction.cable_velocities_m_s for update in execution.updates]
                ),
                "predicted_accelerations_m_s2": np.stack(
                    [update.prediction.accelerations_m_s2 for update in execution.updates]
                ),
            }
        )
    np.savez_compressed(path, **arrays)
    metadata = {
        **row,
        "model_sha256": result.model_sha256,
        "recorded_predictions": len(execution.updates),
        "cost_terms": execution.cost_terms,
    }
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def _warm_runtime(
    planner: WhipSimulator,
    plant: WhipSimulator,
    problem: MpcProblem,
    rollout_batch_size: int,
) -> None:
    state = planner.initial_state(problem.drone_workspace_center_m)
    zero = torch.zeros(
        (rollout_batch_size, planner.settings.control_count, 3),
        dtype=planner.dtype,
        device=planner.device,
    )
    with torch.no_grad():
        planner.rollout(state, zero, create_graph=False)
        planner.rollout(state, zero[:1], create_graph=False)
        plant.rollout(state, zero[:1], create_graph=False)
    if planner.device.type == "cuda":
        torch.cuda.synchronize(planner.device)


def _simulators(snapshot, settings: PerfectMpcSettings, device: str):
    simulation = settings.simulation_settings()
    return (
        WhipSimulator(snapshot, simulation, device=device),
        WhipSimulator(snapshot, simulation, device=device),
    )


def _mppi_settings(
    settings: PerfectMpcSettings,
    *,
    samples: int,
    iterations: int,
    seed: int,
) -> MppiSettings:
    return replace(
        settings.mppi_settings(),
        samples=samples,
        rollout_batch_size=min(samples, 128),
        iterations=iterations,
        seed=seed,
        gradient_guidance_fraction=0.0,
    )


def _transform_for(spec: PerturbationSpec):
    applied = False

    def transform(elapsed_s: float, state: DroneCableState) -> DroneCableState:
        nonlocal applied
        if applied or elapsed_s + 1.0e-9 < spec.time_s:
            return state
        applied = True
        return perturb_cable_state(
            state,
            position_delta_m=spec.position_delta_m,
            velocity_delta_m_s=spec.velocity_delta_m_s,
            profile=spec.profile,
        )

    return transform


def _run_closed_loop(
    snapshot,
    settings: PerfectMpcSettings,
    problem: MpcProblem,
    warm_knots: np.ndarray,
    *,
    samples: int,
    iterations: int,
    seed: int,
    replan_interval_s: float,
    timeout_s: float,
    feedback: str,
    device: str,
    transform=None,
    simulators=None,
) -> RecedingMppiExecution:
    planner, plant = simulators or _simulators(snapshot, settings, device)
    mppi = _mppi_settings(
        settings, samples=samples, iterations=iterations, seed=seed
    )
    state = plant.initial_state(problem.drone_workspace_center_m)
    return run_receding_horizon_mppi(
        planner,
        plant,
        state,
        problem,
        mppi,
        RecedingMppiSettings(
            replan_interval_s=replan_interval_s,
            timeout_s=timeout_s,
            feedback_mode=feedback,
        ),
        warm_knots,
        state_transform=transform,
    )


def _checkpoint_run(
    output: Path,
    study: str,
    row_path: Path,
    rows: list[dict[str, object]],
    row: dict[str, object],
    execution: RecedingMppiExecution,
) -> None:
    run_key = str(row["run_key"])
    _save_execution(output / study / f"{run_key}.npz", execution, row)
    rows.append(row)
    row_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(row_path.with_suffix(".csv"), rows)
    print(
        f"{study}/{run_key}: valid={row['valid_strike']} "
        f"error={1000.0 * float(row['target_error_m']):.1f}mm "
        f"speed={float(row['directed_tip_speed_m_s']):.2f}m/s "
        f"updates={row['updates']} plan={float(row['median_replanning_wall_s']):.3f}s"
    )


def run_pilot(context, output: Path, device: str) -> None:
    snapshot, settings, problem, warm_knots, _controls = context
    planner, plant = _simulators(snapshot, settings, device)
    mppi = _mppi_settings(settings, samples=32, iterations=1, seed=17)
    _warm_runtime(planner, plant, problem, mppi.rollout_batch_size)
    execution = _run_closed_loop(
        snapshot,
        settings,
        problem,
        warm_knots,
        samples=32,
        iterations=1,
        seed=17,
        replan_interval_s=0.10,
        timeout_s=1.20,
        feedback="full",
        device=device,
        simulators=(planner, plant),
    )
    row = _execution_row(
        execution,
        problem,
        study="pilot",
        run_key="pilot_s32_i1_seed17",
        samples=32,
        iterations=1,
        seed=17,
        replan_interval_s=0.10,
        condition="nominal",
        feedback="full",
    )
    _save_execution(output / "pilot" / "pilot_s32_i1_seed17.npz", execution, row)
    print(json.dumps(row, indent=2, sort_keys=True))


def run_efficiency(context, output: Path, device: str) -> None:
    snapshot, settings, problem, warm_knots, _controls = context
    row_path = output / "efficiency_runs.json"
    rows = _load_rows(row_path)
    complete = {str(row["run_key"]) for row in rows}
    for samples in EFFICIENCY_SAMPLES:
        for iterations in EFFICIENCY_ITERATIONS:
            planner, plant = _simulators(snapshot, settings, device)
            mppi = _mppi_settings(
                settings, samples=samples, iterations=iterations, seed=17
            )
            _warm_runtime(planner, plant, problem, mppi.rollout_batch_size)
            for seed in EFFICIENCY_SEEDS:
                run_key = f"s{samples}_i{iterations}_seed{seed}"
                if run_key in complete:
                    continue
                execution = _run_closed_loop(
                    snapshot,
                    settings,
                    problem,
                    warm_knots,
                    samples=samples,
                    iterations=iterations,
                    seed=seed,
                    replan_interval_s=0.50,
                    timeout_s=1.20,
                    feedback="full",
                    device=device,
                    simulators=(planner, plant),
                )
                row = _execution_row(
                    execution,
                    problem,
                    study="efficiency",
                    run_key=run_key,
                    samples=samples,
                    iterations=iterations,
                    seed=seed,
                    replan_interval_s=0.50,
                    condition="nominal",
                    feedback="full",
                )
                _checkpoint_run(output, "efficiency", row_path, rows, row, execution)
            del planner, plant
            torch.cuda.empty_cache()
    _write_summaries(output, "efficiency", rows, ("samples", "iterations"))


def run_rates(
    context, output: Path, device: str, samples: int, iterations: int
) -> None:
    snapshot, settings, problem, warm_knots, _controls = context
    row_path = output / "rate_runs.json"
    rows = _load_rows(row_path)
    complete = {str(row["run_key"]) for row in rows}
    intervals = (0.50, 0.20, 0.10, 0.04)
    planner, plant = _simulators(snapshot, settings, device)
    mppi = _mppi_settings(
        settings, samples=samples, iterations=iterations, seed=17
    )
    _warm_runtime(planner, plant, problem, mppi.rollout_batch_size)
    for interval in intervals:
        for seed in COMPARISON_SEEDS:
            run_key = f"hz{1.0 / interval:g}_seed{seed}"
            if run_key in complete:
                continue
            execution = _run_closed_loop(
                snapshot,
                settings,
                problem,
                warm_knots,
                samples=samples,
                iterations=iterations,
                seed=seed,
                replan_interval_s=interval,
                timeout_s=1.20,
                feedback="full",
                device=device,
                simulators=(planner, plant),
            )
            row = _execution_row(
                execution,
                problem,
                study="rates",
                run_key=run_key,
                samples=samples,
                iterations=iterations,
                seed=seed,
                replan_interval_s=interval,
                condition="nominal",
                feedback="full",
            )
            _checkpoint_run(output, "rates", row_path, rows, row, execution)
    _write_summaries(output, "rates", rows, ("requested_replan_rate_hz",))


def run_disturbances(
    context, output: Path, device: str, samples: int, iterations: int
) -> None:
    snapshot, settings, problem, warm_knots, open_controls = context
    row_path = output / "disturbance_runs.json"
    rows = _load_rows(row_path)
    complete = {str(row["run_key"]) for row in rows}
    planner, plant = _simulators(snapshot, settings, device)
    mppi = _mppi_settings(
        settings, samples=samples, iterations=iterations, seed=17
    )
    _warm_runtime(planner, plant, problem, mppi.rollout_batch_size)
    for spec in PERTURBATIONS:
        open_key = f"{spec.label}_open_loop"
        if open_key not in complete:
            open_plant = WhipSimulator(
                snapshot, settings.simulation_settings(), device=device
            )
            execution = replay_open_loop(
                open_plant,
                open_plant.initial_state(problem.drone_workspace_center_m),
                problem,
                mppi,
                open_controls,
                observation_interval_s=0.10,
                timeout_s=1.20,
                state_transform=_transform_for(spec),
            )
            row = _execution_row(
                execution,
                problem,
                study="disturbances",
                run_key=open_key,
                samples=0,
                iterations=0,
                seed=0,
                replan_interval_s=0.10,
                condition=spec.label,
                feedback="open_loop",
            )
            _checkpoint_run(output, "disturbances", row_path, rows, row, execution)
            complete.add(open_key)
        for feedback in ("full", "endpoint"):
            for seed in COMPARISON_SEEDS:
                run_key = f"{spec.label}_{feedback}_seed{seed}"
                if run_key in complete:
                    continue
                execution = _run_closed_loop(
                    snapshot,
                    settings,
                    problem,
                    warm_knots,
                    samples=samples,
                    iterations=iterations,
                    seed=seed,
                    replan_interval_s=0.10,
                    timeout_s=1.20,
                    feedback=feedback,
                    device=device,
                    transform=_transform_for(spec),
                    simulators=(planner, plant),
                )
                row = _execution_row(
                    execution,
                    problem,
                    study="disturbances",
                    run_key=run_key,
                    samples=samples,
                    iterations=iterations,
                    seed=seed,
                    replan_interval_s=0.10,
                    condition=spec.label,
                    feedback=feedback,
                )
                _checkpoint_run(
                    output, "disturbances", row_path, rows, row, execution
                )
    _write_summaries(output, "disturbances", rows, ("condition", "feedback"))


def run_same_tip(
    context, output: Path, device: str, samples: int, iterations: int
) -> None:
    snapshot, settings, problem, warm_knots, _controls = context
    row_path = output / "same_tip_runs.json"
    rows = _load_rows(row_path)
    complete = {str(row["run_key"]) for row in rows}
    planner, plant = _simulators(snapshot, settings, device)
    mppi = _mppi_settings(
        settings, samples=samples, iterations=iterations, seed=17
    )
    _warm_runtime(planner, plant, problem, mppi.rollout_batch_size)
    states = (
        PerturbationSpec("interior_A", 0.0, (0.0, 0.020, 0.0), (0.0, 0.30, 0.0), "interior"),
        PerturbationSpec("interior_B", 0.0, (0.0, -0.020, 0.0), (0.0, -0.30, 0.0), "interior"),
    )
    for spec in states:
        for feedback in ("full", "endpoint"):
            for seed in COMPARISON_SEEDS:
                run_key = f"{spec.label}_{feedback}_seed{seed}"
                if run_key in complete:
                    continue
                execution = _run_closed_loop(
                    snapshot,
                    settings,
                    problem,
                    warm_knots,
                    samples=samples,
                    iterations=iterations,
                    seed=seed,
                    replan_interval_s=0.10,
                    timeout_s=1.20,
                    feedback=feedback,
                    device=device,
                    transform=_transform_for(spec),
                    simulators=(planner, plant),
                )
                row = _execution_row(
                    execution,
                    problem,
                    study="same_tip",
                    run_key=run_key,
                    samples=samples,
                    iterations=iterations,
                    seed=seed,
                    replan_interval_s=0.10,
                    condition=spec.label,
                    feedback=feedback,
                )
                if execution.updates:
                    row["first_optimized_knots_json"] = json.dumps(
                        execution.updates[0].optimized_knots_m_s2.tolist()
                    )
                _checkpoint_run(output, "same_tip", row_path, rows, row, execution)
    _write_summaries(output, "same_tip", rows, ("condition", "feedback"))


def _goal_set(problem: MpcProblem) -> tuple[tuple[str, MpcProblem], ...]:
    angle = math.radians(15.0)
    return (
        ("baseline", problem),
        ("shorter_0p90", replace(problem, target_position_m=(0.90, 0.0, 1.40))),
        ("farther_1p05", replace(problem, target_position_m=(1.05, 0.0, 1.40))),
        ("higher_5cm", replace(problem, target_position_m=(1.00, 0.0, 1.45))),
        ("lower_5cm", replace(problem, target_position_m=(1.00, 0.0, 1.35))),
        ("lateral_8cm", replace(problem, target_position_m=(1.00, 0.08, 1.40))),
        (
            "direction_yaw15",
            replace(problem, impact_direction=(math.cos(angle), math.sin(angle), 0.0)),
        ),
        ("speed_4p0", replace(problem, minimum_impact_speed_m_s=4.0)),
    )


def run_goals(
    context, output: Path, device: str, samples: int, iterations: int
) -> None:
    snapshot, settings, problem, warm_knots, _controls = context
    row_path = output / "goal_runs.json"
    rows = _load_rows(row_path)
    complete = {str(row["run_key"]) for row in rows}
    planner, plant = _simulators(snapshot, settings, device)
    base_mppi = _mppi_settings(
        settings, samples=samples, iterations=iterations, seed=17
    )
    _warm_runtime(planner, plant, problem, base_mppi.rollout_batch_size)
    for label, goal in _goal_set(problem):
        for seed in COMPARISON_SEEDS:
            run_key = f"{label}_seed{seed}"
            if run_key in complete:
                continue
            execution = _run_closed_loop(
                snapshot,
                settings,
                goal,
                warm_knots,
                samples=samples,
                iterations=iterations,
                seed=seed,
                replan_interval_s=0.10,
                timeout_s=1.20,
                feedback="full",
                device=device,
                simulators=(planner, plant),
            )
            row = _execution_row(
                execution,
                goal,
                study="goals",
                run_key=run_key,
                samples=samples,
                iterations=iterations,
                seed=seed,
                replan_interval_s=0.10,
                condition=label,
                feedback="full",
            )
            row["target_position_m"] = json.dumps(goal.target_position_m)
            row["impact_direction"] = json.dumps(goal.impact_direction)
            row["requested_speed_m_s"] = goal.minimum_impact_speed_m_s
            _checkpoint_run(output, "goals", row_path, rows, row, execution)
    _write_summaries(output, "goals", rows, ("condition",))


def _write_summaries(
    output: Path,
    name: str,
    rows: list[dict[str, object]],
    keys: tuple[str, ...],
) -> None:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    summaries: list[dict[str, object]] = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        successes = [row for row in group if bool(row["valid_strike"])]
        summary = {name: value for name, value in zip(keys, key)}
        summary.update(
            {
                "runs": len(group),
                "successes": len(successes),
                "success_rate_pct": 100.0 * len(successes) / len(group),
                "non_tip_first_count": sum(
                    bool(row["non_tip_first_failure"]) for row in group
                ),
                "mean_target_error_mm": 1000.0
                * float(np.mean([float(row["target_error_m"]) for row in group])),
                "mean_directed_speed_m_s": float(
                    np.mean([float(row["directed_tip_speed_m_s"]) for row in group])
                ),
                "mean_direction_error_deg": float(
                    np.mean([float(row["direction_error_deg"]) for row in group])
                ),
                "mean_non_tip_clearance_mm": 1000.0
                * float(
                    np.mean([float(row["non_tip_clearance_m"]) for row in group])
                ),
                "median_replanning_wall_s": float(
                    median(float(row["median_replanning_wall_s"]) for row in group)
                ),
                "mean_total_planning_wall_s": float(
                    np.mean([float(row["total_planning_wall_s"]) for row in group])
                ),
                "mean_total_rollouts": float(
                    np.mean([float(row["total_rollouts"]) for row in group])
                ),
                "mean_deadline_met_pct": float(
                    np.mean([float(row["deadline_met_pct"]) for row in group])
                ),
            }
        )
        summaries.append(summary)
    (output / f"{name}_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(output / f"{name}_summary.csv", summaries)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("pilot", "efficiency", "rates", "disturbances", "same-tip", "goals", "all"),
        default="pilot",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=2)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    output = arguments.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference = arguments.reference.expanduser().resolve()
    settings, problem, warm_knots, controls, payload = _load_configuration(reference)
    snapshot = load_cable_model(arguments.model)
    if payload.get("model_sha256") != snapshot.sha256:
        raise ValueError("Reference trajectory and selected cable model do not match.")
    context = (snapshot, settings, problem, warm_knots, controls)
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_sha256": snapshot.sha256,
        "reference": str(reference),
        "physics_unchanged": True,
        "impact_cost_unchanged": True,
        "contact_logic_unchanged": True,
        "safety_logic_unchanged": True,
        "target_definition": asdict(problem),
        "horizon_s": settings.horizon_s,
        "control_interval_s": settings.control_interval_s,
        "physics_dt_s": settings.physics_dt_s,
        "knot_count": settings.mppi_knot_count,
        "endpoint_observer": (
            "DDER-predicted interior with smooth root-to-tip correction from observed "
            "tip position/velocity; no true interior nodes"
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    modes: Iterable[str] = (
        ("efficiency", "rates", "disturbances", "same-tip", "goals")
        if arguments.mode == "all"
        else (arguments.mode,)
    )
    for mode in modes:
        if mode == "pilot":
            run_pilot(context, output, arguments.device)
        elif mode == "efficiency":
            run_efficiency(context, output, arguments.device)
        elif mode == "rates":
            run_rates(
                context, output, arguments.device, arguments.samples, arguments.iterations
            )
        elif mode == "disturbances":
            run_disturbances(
                context, output, arguments.device, arguments.samples, arguments.iterations
            )
        elif mode == "same-tip":
            run_same_tip(
                context, output, arguments.device, arguments.samples, arguments.iterations
            )
        elif mode == "goals":
            run_goals(
                context, output, arguments.device, arguments.samples, arguments.iterations
            )


if __name__ == "__main__":
    main()
