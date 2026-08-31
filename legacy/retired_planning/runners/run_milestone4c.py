"""Run the authorized Milestone-4C variable-duration CEM campaign."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from planning.artifacts import (
    create_plots,
    create_result_directory,
    final_replay_metrics,
    save_command_csv,
    save_command_npz,
    save_replay_npz,
    source_hash_manifest,
    verify_planning_model_integrity,
    write_json,
)
from planning.cem import (
    CemSeedResult,
    is_marginal_success,
    optimize_variable_duration_cem,
    select_final_seed,
)
from planning.cem_task import DEFAULT_VARIABLE_DURATION_TASK, load_variable_duration_task
from planning.results import load_planning_result
from planning.rollout import hover_preroll
from planning.variable_duration import replay_variable_trajectory
from planning.video import render_replay_video
from simulator.parameters import SimulatorSettings
from simulator.production import (
    active_model_paths,
    build_production_simulator,
    load_active_model_manifest,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT_PATH = ROOT / "MILESTONE4C_VARIABLE_DURATION_CEM_REPORT.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_figure8_source() -> dict[str, object]:
    """One bounded non-legacy source audit under the existing PHD code area."""

    search_root = ROOT.parent
    extensions = {".py", ".cpp", ".cc", ".cxx", ".hpp", ".h", ".yaml", ".yml", ".json"}
    excluded = {".git", ".venv", "data", "legacy", "__pycache__", "reports"}
    candidates: list[dict[str, object]] = []
    scanned = 0
    for path in search_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        relative_parts = {part.lower() for part in path.relative_to(search_root).parts[:-1]}
        if relative_parts & excluded or "fig8vertical_002" in path.name.lower():
            continue
        scanned += 1
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lower = text.lower()
        has_fullstate = "cmdfullstate" in lower or "cmd_full_state" in lower
        has_figure8 = "figure8" in lower or "figure_8" in lower or "figure-8" in lower
        if has_fullstate and has_figure8:
            candidates.append(
                {
                    "path": str(path),
                    "has_cmdFullState": True,
                    "has_figure8_definition_terms": True,
                    "authoritative_generator": False,
                    "reason": "Candidate requires manual authority evidence; repository references alone are not sufficient.",
                }
            )
    unambiguous = [item for item in candidates if item["authoritative_generator"]]
    status = "FOUND" if len(unambiguous) == 1 else "AMBIGUOUS" if candidates else "NOT FOUND"
    payload = {
        "schema": "figure8_source_audit_4c_v1",
        "search_root": str(search_root),
        "scope": "Relevant source/config files under the user's existing PHD project area; legacy, data, reports, environments, and protected take excluded.",
        "files_scanned": scanned,
        "candidate_files": candidates,
        "authoritative_current_generators": unambiguous,
        "status": status,
        "endpoint_semantics_unambiguous": len(unambiguous) == 1,
        "task_b_authorized": len(unambiguous) == 1,
        "protected_test_accessed": False,
    }
    write_json(ROOT / "reports" / "figure8_source_audit_4c.json", payload)
    return payload


def _seed_summary(result: CemSeedResult) -> dict[str, object]:
    metrics = result.best_success_metrics or result.best_feasible_metrics or result.best_overall_metrics
    success = result.best_success_metrics is not None
    return {
        "seed": result.seed,
        "duration_max_s": result.duration_max_s,
        "iterations": result.iterations,
        "stopped_reason": result.stopped_reason,
        "success": success,
        "best_tip_error_m": float(
            metrics["first_entry_tip_distance_m"] if success else metrics["best_event_tip_distance_m"]
        ),
        "hit_time_s": metrics["first_entry_time_s"] if success else None,
        "best_event_time_s": metrics["best_event_time_s"],
        "optimized_duration_s": metrics["optimized_duration_s"],
        "direction_error_deg": float(
            metrics["first_entry_direction_angle_deg"] if success else metrics["best_event_direction_angle_deg"]
        ),
        "directed_tip_speed_m_s": float(
            metrics["first_entry_directed_speed_m_s"] if success else metrics["best_event_directed_speed_m_s"]
        ),
        "maximum_uav_displacement_m": metrics["maximum_uav_displacement_m"],
        "maximum_uav_speed_m_s": metrics["maximum_uav_speed_m_s"],
        "runtime_s": result.total_runtime_s,
        "population_rollouts_evaluated": result.population_rollouts_evaluated,
    }


def _vector_difference(left: object, right: object) -> float:
    return max(abs(float(a) - float(b)) for a, b in zip(left, right))


def _candidate_replay_comparison(
    candidate: dict[str, object], replay: dict[str, object]
) -> dict[str, object]:
    uav = _vector_difference(candidate["final_uav_position_m"], replay["final_uav_position_m"])
    tip = _vector_difference(candidate["final_c10_position_m"], replay["final_c10_position_m"])
    distance = abs(
        float(candidate["minimum_tip_target_distance_m"])
        - float(replay["minimum_tip_target_distance_m"])
    )
    classification = (
        bool(candidate["success"]) == bool(replay["success"])
        and candidate["first_entry_marker"] == replay["first_entry_marker"]
    )
    passed = uav <= 0.001 and tip <= 0.002 and distance <= 0.002 and classification
    return {
        "sampled_cem_candidate": candidate,
        "authoritative_deterministic_replay": replay,
        "canonical_fixed_uav_evaluation_batch_size": 2048,
        "final_uav_position_difference_m": uav,
        "final_c10_position_difference_m": tip,
        "minimum_target_distance_difference_m": distance,
        "success_and_first_entry_identical": classification,
        "tolerances_m": {"final_uav": 0.001, "final_c10": 0.002, "minimum_distance": 0.002},
        "pass": passed,
    }


def _strong(metrics: dict[str, object] | None, task) -> bool:
    if metrics is None:
        return False
    return bool(
        float(metrics["first_entry_tip_distance_m"]) <= task.cem.strong_tip_error_m
        and float(metrics["first_entry_directed_speed_m_s"]) >= task.cem.strong_directed_speed_m_s
        and float(metrics["first_entry_direction_angle_deg"]) <= task.cem.strong_direction_error_deg
        and float(metrics["maximum_uav_displacement_m"]) <= task.cem.strong_uav_displacement_m
        and float(metrics["maximum_uav_speed_m_s"]) <= task.cem.strong_uav_speed_m_s
    )


def _write_report(
    *,
    report_path: Path,
    result_dir: Path,
    task,
    seed_summaries: list[dict[str, object]],
    selected_seed: int,
    selected_kind: str,
    duration_extended: bool,
    final: dict[str, object],
    consistency: dict[str, object],
    runtime: dict[str, object],
    audit: dict[str, object],
    video_path: Path,
    preflight: dict[str, object],
) -> None:
    rows = "\n".join(
        "| {seed} | {iterations} | {success} | {best_tip_error_m:.4f} | {hit} | {optimized_duration_s:.3f} | {direction_error_deg:.2f} | {directed_tip_speed_m_s:.3f} | {maximum_uav_displacement_m:.3f} | {maximum_uav_speed_m_s:.3f} | {runtime_s:.1f} |".format(
            hit="—" if item["hit_time_s"] is None else f"{float(item['hit_time_s']):.3f}",
            **item,
        )
        for item in seed_summaries
    )
    gate_rows = "\n".join(
        f"| {name} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in final["hard_success_gates"].items()
    )
    hit = "NONE" if final["hit_time_s"] is None else f"{float(final['hit_time_s']):.3f} s"
    duration_status = str(final["duration_status"])
    classification = str(final["task_classification"])
    legacy_profile = (
        task.legacy_run_online_objective.profile
        if task.legacy_run_online_objective is not None
        else ""
    )
    margin_tuned_reward = legacy_profile == "legacy_run_online_strike_margin_tuned_v4"
    objective_description = (
        "legacy `run_online` strike reward with explicit interior strike-margin tuning"
        if margin_tuned_reward
        else (
            "legacy `run_online` strike-reward port under the current CEM/task contract"
            if task.legacy_run_online_objective is not None
            else "Milestone 4C normalized strike reward"
        )
    )
    legacy_objective_section = ""
    if task.legacy_run_online_objective is not None:
        legacy = task.legacy_run_online_objective
        legacy_objective_section = f"""
## Legacy `run_online` reward used

The reward kernel and active corrected GUI-preset coefficients were read from
`legacy/current_baseline_2026-08-27/drone_mpc/mppi.py` and
`legacy/current_baseline_2026-08-27/drone_mpc/receding_mppi_gui.py`; they were
not reconstructed from a report.  At each candidate event the port evaluates:

    J_position = {legacy.position_weight:g} d^2 / (d^2 + {legacy.position_sigma_m:g}^2)
    proximity = exp(-d^2 / (2 * {legacy.velocity_gate_sigma_m:g}^2))
    J_speed = {legacy.speed_weight:g} proximity relu({legacy.directed_speed_shaping_m_s:g} - v_directed)^2
    predictive_proximity = exp(-d^2 / (2 * {legacy.predictive_velocity_gate_sigma_m:g}^2))
    J_predictive = {legacy.predictive_speed_weight:g} predictive_proximity relu({legacy.predictive_speed_ratio:g} * {legacy.directed_speed_shaping_m_s:g} - v_directed)^2
    J_direction = {legacy.direction_weight:g} proximity relu(cos({legacy.direction_shaping_error_deg:g} deg) - cos(theta))^2
    J_displacement = {legacy.drone_displacement_weight:g} ||p_uav(event) - p_uav(0)||^2

It also uses safety weight {legacy.safety_weight:g}, success bonus
`-{legacy.success_cost:g}`, effort weight {legacy.control_effort_weight:g}, and
smoothness weight {legacy.control_smoothness_weight:g}.  The predictive-speed
weight of {legacy.predictive_speed_weight:g} is the value initialized and
applied by the corrected legacy `run_online` GUI preset.

Scientific separation was preserved: this run did **not** restore the legacy
simulator, MPPI algorithm, or legacy 3.5-m/s/35-degree acceptance defaults.
It used the frozen production model, variable-duration CEM, feasibility-first
ordering, and the current hard gates of 4.0 m/s and 30 degrees.  Event and
safety bookkeeping use the current fixed-timestep production planner contract;
therefore this is a faithful reward-shape/coefficient port, not a byte-for-byte
execution of the complete legacy `_mppi_event_objective` implementation.
"""
        if margin_tuned_reward:
            legacy_objective_section += f"""

## Reward tuning decision

The original port produced a close and fast near-miss, but its direction error
was 47.104 degrees.  A first direction-weight run increased the direction
coefficient from 20 to {legacy.direction_weight:g}; it passed at 29.983 degrees,
only 0.017 degrees inside the 30-degree hard gate.  This exposed two reward
hinges that stopped supplying a refinement signal exactly at the scientific
acceptance thresholds.  The final profile therefore changes only three reward
settings relative to the legacy port:

- direction coefficient: `20 -> {legacy.direction_weight:g}`;
- direction shaping target: `30 -> {legacy.direction_shaping_error_deg:g} deg`;
- directed-speed shaping target: `4.0 -> {legacy.directed_speed_shaping_m_s:g} m/s`.

The scientific hard gates remain unchanged at 50 mm, 4.0 m/s, 30 degrees,
0.50-m UAV displacement, 3.0-m/s UAV speed, and 20-m/s^2 command acceleration.
Within the hard-feasible successful category, CEM elite selection and final
cross-seed selection use the configured reward so the added margin terms can
continue polishing a valid strike.  Legacy-profile runs retain their original
ordering and remain reproducible.

| Reward stage | Tip error | Directed speed | Direction error | Result |
|---|---:|---:|---:|:---:|
| Untuned legacy port | 5.477 mm | 4.225 m/s | 47.104 deg | FAIL |
| Direction weight only | 1.103 mm | 4.282 m/s | 29.983 deg | PASS, fragile margin |
| Final interior-margin profile | {1000.0 * float(final['reported_event_tip_position_error_m']):.3f} mm | {float(final['reported_event_directed_tip_speed_m_s']):.3f} m/s | {float(final['reported_event_direction_error_deg']):.3f} deg | {classification} |
"""
    report = f"""# Milestone 4C — Variable-Duration Offline CEM Report

## Outcome

Variable-duration CEM was implemented and evaluated against the unchanged `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`. The prior fixed 0.70-s formulation was replaced because Milestone 4B.1's best feasible event occurred at that artificial boundary. CEM jointly optimized 16 normalized-time acceleration knots and maneuver duration.

Objective profile: **{objective_description}**.

The physical model, UAV residual, cable parameters, geometry, PCG32 backend, float32 precision, three DDER substeps, and four projections were not changed or refitted. The protected test was not evaluated and no real hardware was connected or executed.

{legacy_objective_section}

## Formulation and implementation

- Decision dimension: 49 (48 acceleration components + one duration).
- Initial duration range: 0.45–1.20 s.
- One-time extension used: **{'YES' if duration_extended else 'NO'}**.
- Final allowed duration maximum: {float(final['duration_max_allowed_s']):.2f} s.
- Fixed physics timestep; candidate-specific `t <= T_i` masks exclude all post-duration states.
- A candidate terminates scientifically at its first valid strike.
- FullState position and velocity are integrated from linearly interpolated acceleration; yaw and omega commands remain zero.
- Population: 8192, evaluated as four canonical 2048-row chunks.
- Elite fraction: 5% (~410 candidates).
- Covariance: full 49×49 with positive-definite regularization and configured variance floors.
- Distribution smoothing: 30% old + 70% elite.
- Feasibility-first global elite ordering: successes, feasible near-misses, then infeasible candidates with continuous violation ordering.
- Every completed iteration was checkpointed with distribution, RNG state, best candidate, bounds, model reference, and iteration summary.

## Cheap pre-run verification

```json
{json.dumps(preflight, indent=2, sort_keys=True)}
```

## Per-seed campaign

| Seed | Iterations | Success | Tip error [m] | Hit [s] | T [s] | Direction [deg] | Directed speed [m/s] | UAV disp. [m] | UAV speed [m/s] | Runtime [s] |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

Final seed: **{selected_seed}**. Selection class: **{selected_kind}**. {'For the tuned profile, successful feasible plans were ranked by the configured strike reward, which explicitly preserves target accuracy while polishing directed-speed and direction margins.' if margin_tuned_reward else 'Successful feasible plans, when present, were ranked by target-center accuracy, then direction, then feasibility margin; duration was only a weak tie-breaker.'}

## Deterministic replay and numerical consistency

- Optimized duration: {float(final['optimized_duration_s']):.4f} s ({duration_status}).
- First valid hit: {hit}.
- Tip error: {1000.0 * float(final['reported_event_tip_position_error_m']):.3f} mm.
- Tip total speed: {float(final['reported_event_tip_total_speed_m_s']):.3f} m/s.
- Directed tip speed: {float(final['reported_event_directed_tip_speed_m_s']):.3f} m/s.
- Direction error: {float(final['reported_event_direction_error_deg']):.3f} deg.
- First target-entry marker: {final['first_target_entry_marker_label'] or 'none'}.
- UAV maximum displacement: {float(final['maximum_uav_displacement_m']):.4f} m.
- UAV maximum speed: {float(final['maximum_uav_speed_m_s']):.4f} m/s.
- Maximum command acceleration: {float(final['maximum_command_acceleration_m_s2']):.4f} m/s².
- Tip/UAV speed ratio at event: {float(final['tip_speed_to_uav_speed_ratio_at_reported_event']):.3f}.
- Candidate/replay consistency: **{'PASS' if consistency['pass'] else 'FAIL'}**.

| Hard gate | Result |
|---|:---:|
{gate_rows}

If the hit is later than 1.0 s, the artifact is explicitly labeled `LONGER_THAN_PRIMARY_VALIDATION_HORIZON`; it is not rejected in simulation solely for that reason.

## Figure-8 source audit

The bounded broader audit status is **{audit['status']}**. Figure-8 Task B was **{'authorized' if audit['task_b_authorized'] else 'NOT RUN'}** because an authoritative current generator and unambiguous endpoint convention were {'found' if audit['task_b_authorized'] else 'not found'}. See `reports/figure8_source_audit_4c.json`.

## Runtime and artifacts

- Campaign runtime: {float(runtime['campaign_runtime_s']):.2f} s.
- Candidate rollouts: {int(runtime['candidate_rollouts_evaluated'])}.
- Effective rollouts/s: {float(runtime['effective_rollouts_per_s']):.1f}.
- Peak CUDA memory: {float(runtime['peak_cuda_memory_mb']):.1f} MB.
- Result directory: `{result_dir}`.
- Video: `{video_path}`.
- Email delivery: `NOT_AVAILABLE_IN_LOCAL_ENVIRONMENT`.
- Protected test: `NOT EVALUATED`.
- Real hardware: `NOT EXECUTED`.

## Final summary

    Model:
        MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI

    Optimizer:
        Variable-Duration CEM

    Canonical task:
        target = [1.0, 0.0, 1.4]

    Duration search:
        initial range = [0.45, 1.20] s
        extended = {'YES' if duration_extended else 'NO'}
        final allowed max = {float(final['duration_max_allowed_s']):.2f} s

    Seeds executed:
        {[item['seed'] for item in seed_summaries]}

    Final selected seed:
        {selected_seed}

    Optimized maneuver duration:
        {float(final['optimized_duration_s']):.4f} s

    Hit time:
        {hit}

    Duration status:
        {duration_status}

    Tip error:
        {1000.0 * float(final['reported_event_tip_position_error_m']):.3f} mm

    Tip total speed:
        {float(final['reported_event_tip_total_speed_m_s']):.3f} m/s

    Directed tip speed:
        {float(final['reported_event_directed_tip_speed_m_s']):.3f} m/s

    Direction error:
        {float(final['reported_event_direction_error_deg']):.3f} deg

    First target-entry marker:
        {final['first_target_entry_marker_label'] or 'none'}

    UAV max displacement:
        {float(final['maximum_uav_displacement_m']):.4f} m

    UAV max speed:
        {float(final['maximum_uav_speed_m_s']):.4f} m/s

    Max command acceleration:
        {float(final['maximum_command_acceleration_m_s2']):.4f} m/s^2

    Candidate/replay consistency:
        {'PASS' if consistency['pass'] else 'FAIL'}

    VARIABLE_DURATION_CEM:
        {classification}

    Figure-8 source:
        {audit['status']}

    Figure-8 task:
        {'NOT RUN' if not audit['task_b_authorized'] else 'SEE TASK ARTIFACT'}

    Video:
        {video_path}

    Protected test:
        NOT EVALUATED

    Real hardware:
        NOT EXECUTED
"""
    report_path.write_text(report, encoding="utf-8")
    (result_dir / report_path.name).write_text(report, encoding="utf-8")


def run_campaign(task_path: Path = DEFAULT_VARIABLE_DURATION_TASK) -> dict[str, object]:
    # Artifact hashing expects project-rooted absolute paths.  Normalize here so
    # callers may safely pass either the default path or a relative config path.
    task_path = Path(task_path).resolve()
    task = load_variable_duration_task(task_path)
    if task.task_id == "canonical_whip_variable_duration_tuned_reward_v1":
        report_path = ROOT / "MILESTONE4C_TUNED_REWARD_CEM_REPORT.md"
    elif task.legacy_run_online_objective is not None:
        report_path = ROOT / "MILESTONE4C_LEGACY_RUN_ONLINE_REWARD_CEM_REPORT.md"
    else:
        report_path = DEFAULT_REPORT_PATH
    active = load_active_model_manifest()
    settings = SimulatorSettings.load(active_model_paths(active)["configuration"])
    task.validate_for_dt(settings.dt_s)
    integrity = verify_planning_model_integrity(settings, task)
    if not torch.cuda.is_available():
        raise RuntimeError("Milestone 4C requires the frozen CUDA production backend.")
    simulator = build_production_simulator(settings, device="cuda", dtype=torch.float32)
    backend = simulator.forward_backend_audit()
    if backend["damping_backend"] != "pcg32_experimental" or not backend["production_fused_forward_active"]:
        raise RuntimeError("Frozen PCG32 fused production DDER backend is not active.")
    simulator.uav_model.set_fixed_evaluation_batch_size(task.cem.fixed_uav_evaluation_batch_size)

    result_dir = create_result_directory(task.task_id)
    write_json(result_dir / "task_config_snapshot.json", task.snapshot())
    write_json(result_dir / "model_freeze_reference.json", integrity)
    write_json(result_dir / "cem_config.json", asdict(task.cem))
    audit = audit_figure8_source()
    write_json(result_dir / "figure8_source_audit_4c.json", audit)
    initial_state = hover_preroll(simulator, task)
    write_json(
        result_dir / "hover_preroll.json",
        {
            "duration_s": task.hover_preroll_s,
            "fixed_uav_evaluation_batch_size": simulator.uav_model.fixed_evaluation_batch_size,
            "residual_history_shape": list(initial_state.uav.residual_history.features.shape),
        },
    )

    preflight_path = ROOT / "reports" / "milestone4c_preflight_verification.json"
    preflight = (
        json.loads(preflight_path.read_text(encoding="utf-8"))
        if preflight_path.is_file()
        else {"status": "RUNNER_REQUIRES_PREFLIGHT_TEST_ARTIFACT", "pass": False}
    )
    if preflight.get("pass") is not True:
        raise RuntimeError("Milestone 4C preflight verification has not passed.")
    write_json(result_dir / "preflight_verification.json", preflight)

    campaign_start = time.perf_counter()
    deadline = campaign_start + task.cem.hard_stop_s
    torch.cuda.reset_peak_memory_stats(simulator.device)
    seed_results: list[CemSeedResult] = []
    duration_max = task.cem.duration_max_initial_s
    duration_extended = False

    def progress(record) -> None:
        print("CEM_PROGRESS " + json.dumps(asdict(record), sort_keys=True), flush=True)

    for seed_index, seed in enumerate(task.cem.seeds[: task.cem.maximum_seed_count]):
        if time.perf_counter() >= deadline:
            break
        seed_result = optimize_variable_duration_cem(
            simulator,
            initial_state,
            task,
            seed=seed,
            duration_max_s=duration_max,
            seed_directory=result_dir / "cem_checkpoints" / f"seed_{seed}",
            campaign_deadline=deadline,
            iteration_callback=progress,
        )
        seed_results.append(seed_result)
        write_json(
            result_dir / "cem_seed_summary.json",
            [_seed_summary(item) for item in seed_results],
        )
        if (
            task.task_id != "canonical_whip_variable_duration_legacy_reward_v1"
            and _strong(seed_result.best_success_metrics, task)
            and not is_marginal_success(seed_result.best_success_metrics)
        ):
            break
        # Three independent failures at the upper-duration boundary constitute
        # specific boundary-seeking evidence.  Preserve the six-run total cap
        # by using only remaining authorized seeds for the one extension.
        if not duration_extended and len(seed_results) >= 3 and not any(
            item.best_success_metrics is not None for item in seed_results
        ):
            feasible_durations = [
                float(item.best_feasible_metrics["optimized_duration_s"])
                for item in seed_results
                if item.best_feasible_metrics is not None
            ]
            boundary_count = sum(
                value >= task.cem.duration_max_initial_s - 0.03
                for value in feasible_durations
            )
            if feasible_durations and boundary_count / len(feasible_durations) >= 2.0 / 3.0:
                duration_max = task.cem.duration_max_extended_s
                duration_extended = True

    if not seed_results:
        raise RuntimeError("No CEM seed completed before the campaign hard stop.")
    selected_seed, decision, sampled_metrics, selected_kind = select_final_seed(
        seed_results, task
    )
    knots = decision[:-1].reshape(task.cem.knot_count, 3).to(simulator.device, simulator.dtype)
    duration = float(decision[-1])
    torch.cuda.synchronize(simulator.device)
    replay_start = time.perf_counter()
    replay = replay_variable_trajectory(simulator, initial_state, knots, duration, task)
    torch.cuda.synchronize(simulator.device)
    replay_runtime = time.perf_counter() - replay_start
    final = final_replay_metrics(replay, task, settings)
    consistency = _candidate_replay_comparison(sampled_metrics, final)
    classification = (
        "NUMERICALLY_UNTRUSTWORTHY"
        if not consistency["pass"]
        else "PASS"
        if final["success"]
        else "FAIL"
    )
    final.update(
        {
            "optimizer": "Variable-Duration CEM",
            "optimized_duration_s": duration,
            "duration_max_allowed_s": selected_seed.duration_max_s,
            "duration_status": (
                "T_MAX_BOUNDARY" if duration >= selected_seed.duration_max_s - 0.01 else "INTERIOR"
            ),
            "task_completion_time_s": final["hit_time_s"] if final["success"] else duration,
            "longer_than_primary_validation_horizon": bool(
                final["hit_time_s"] is not None and float(final["hit_time_s"]) > 1.0
            ),
            "model_validity_label": (
                "LONGER_THAN_PRIMARY_VALIDATION_HORIZON"
                if final["hit_time_s"] is not None and float(final["hit_time_s"]) > 1.0
                else "WITHIN_PRIMARY_VALIDATION_HORIZON"
            ),
            "selected_seed": selected_seed.seed,
            "selected_candidate_class": selected_kind,
            "candidate_replay_consistency_pass": consistency["pass"],
            "task_classification": classification,
            "variable_duration_cem": classification,
        }
    )

    summaries = [_seed_summary(item) for item in seed_results]
    write_json(result_dir / "cem_seed_summary.json", summaries)
    write_json(
        result_dir / "best_candidate_per_seed.json",
        [
            {
                "seed": item.seed,
                "best_success": item.best_success_metrics,
                "best_feasible": item.best_feasible_metrics,
                "best_overall": item.best_overall_metrics,
            }
            for item in seed_results
        ],
    )
    write_json(result_dir / "cem_iteration_history.json", [asdict(item) for item in selected_seed.history])
    np.savez_compressed(
        result_dir / "final_distribution.npz",
        mean=selected_seed.final_mean.numpy(),
        covariance=selected_seed.final_covariance.numpy(),
    )
    write_json(
        result_dir / "best_acceleration_knots.json",
        {"units": "m/s^2", "values": knots.detach().cpu().tolist()},
    )
    write_json(result_dir / "optimized_duration.json", {"duration_s": duration, "status": final["duration_status"]})
    write_json(
        result_dir / "best_feasible_metrics.json",
        selected_seed.best_feasible_metrics or {"available": False},
    )
    best_overall_seed = min(seed_results, key=lambda item: float(item.best_overall_metrics["task_cost"]))
    write_json(result_dir / "best_overall_metrics.json", best_overall_seed.best_overall_metrics)
    write_json(result_dir / "final_metrics.json", final)
    write_json(result_dir / "sampled_candidate_vs_replay.json", consistency)
    write_json(
        result_dir / "task_success.json",
        {
            "classification": classification,
            "success": classification == "PASS",
            "hard_success_gates": final["hard_success_gates"],
            "authorization": "SIMULATION_ONLY",
            "real_hardware_execution": "NOT EXECUTED",
        },
    )
    save_command_csv(result_dir / "planned_fullstate_command.csv", replay)
    save_command_npz(result_dir / "planned_fullstate_command.npz", replay, task)
    save_replay_npz(result_dir / "final_replay.npz", replay, task)
    create_plots(result_dir, replay, task, final)

    total_rollouts = sum(item.population_rollouts_evaluated for item in seed_results)
    campaign_runtime = time.perf_counter() - campaign_start
    runtime = {
        "campaign_runtime_s": campaign_runtime,
        "seed_runtime_s": {str(item.seed): item.total_runtime_s for item in seed_results},
        "candidate_rollouts_evaluated": total_rollouts,
        "effective_rollouts_per_s": total_rollouts / max(campaign_runtime, 1e-12),
        "final_replay_s": replay_runtime,
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(simulator.device) / 1024**2,
        "device": torch.cuda.get_device_name(simulator.device),
        "hard_stop_s": task.cem.hard_stop_s,
        "hard_stop_reached": time.perf_counter() >= deadline,
    }
    write_json(result_dir / "planning_runtime.json", runtime)
    write_json(result_dir / "email_delivery.json", {"status": "NOT_AVAILABLE_IN_LOCAL_ENVIRONMENT"})
    source_manifest = source_hash_manifest(task_config=task_path, runner=Path(__file__))
    source_manifest["task_config_sha256"] = _sha256(task_path)
    write_json(result_dir / "source_hash_manifest.json", source_manifest)
    video_path, video_metadata = render_replay_video(load_planning_result(result_dir))
    write_json(result_dir / "video_metadata.json", video_metadata)
    _write_report(
        report_path=report_path,
        result_dir=result_dir,
        task=task,
        seed_summaries=summaries,
        selected_seed=selected_seed.seed,
        selected_kind=selected_kind,
        duration_extended=duration_extended,
        final=final,
        consistency=consistency,
        runtime=runtime,
        audit=audit,
        video_path=video_path,
        preflight=preflight,
    )
    output = {
        "result_directory": str(result_dir),
        "classification": classification,
        "selected_seed": selected_seed.seed,
        "optimized_duration_s": duration,
        "video": str(video_path),
        "report": str(report_path),
        "figure8_source": audit["status"],
        "figure8_task": "NOT RUN" if not audit["task_b_authorized"] else "AUTHORIZED_BUT_NOT_IMPLEMENTED",
    }
    print("CEM_RESULT " + json.dumps(output, sort_keys=True), flush=True)
    return output


if __name__ == "__main__":
    run_campaign()
