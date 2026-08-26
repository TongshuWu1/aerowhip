"""Evaluate exact-DDER gradient guidance for the far/fast MPPI whip task.

This tool preserves the existing full DDER dynamics and hard MPPI objective.
The smooth objective is used only to obtain a local proposal direction.  It
first validates that direction around saved trajectories and then supports a
paired-seed comparison of vanilla and guided stochastic MPPI.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from drone_mpc.model import load_cable_model
from drone_mpc.problem import MpcProblem
from drone_mpc.mppi import (
    MppiPlan,
    compute_dder_guidance,
    evaluate_mppi_rollout,
    interpolate_control_knots,
    smooth_strike_surrogate,
)
from drone_mpc.perfect_model import (
    PerfectMpcSettings,
    save_perfect_mpc_result,
    solve_perfect_model_mpc,
)
from drone_mpc.simulator import WhipSimulator
from optitrack_offline.config import DEFAULT_MODEL_PATH
from research_tools.mppi_ablation import (
    resample_knots_in_time,
    structured_seed_knots,
)


SCHEMA = "dder_gradient_guided_mppi_study_v2"
DEFAULT_MINIMUM_IMPACT_SPEED_M_S = 3.5
DEFAULT_BASELINE = Path("data/drone_mpc/perfect_model_mppi_farther_faster.npz")
DEFAULT_NEAR_SUCCESS = Path(
    "data/drone_mpc/ablations/mppi_discovery_pilot/"
    "initialization_backward_forward__T2__k11__vg0_12__pw0__"
    "pilot__i6__n128__b128__seed11.npz"
)


@dataclass(frozen=True, slots=True)
class Condition:
    key: str
    label: str
    guided: bool
    initialization: str
    primary: bool


CONDITIONS = (
    Condition("A", "vanilla_zero", False, "zero", True),
    Condition("B", "guided_zero", True, "zero", True),
    Condition("C", "guided_forward_recoil", True, "forward_recoil", True),
    # C versus D is a required matched-initialization comparison.  A versus B
    # tests cold-start discovery; C versus D isolates gradient guidance from
    # the much larger effect of the structured forward-recoil initialization.
    Condition("D", "vanilla_forward_recoil", False, "forward_recoil", True),
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    names = sorted({name for row in rows for name in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _load_metadata(path: Path) -> dict[str, object]:
    sidecar = path.expanduser().resolve().with_suffix(".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Replay metadata is not an object: {sidecar}")
    return payload


def _settings_from_metadata(payload: dict[str, object]) -> PerfectMpcSettings:
    raw = payload.get("settings")
    if not isinstance(raw, dict):
        raise ValueError("Baseline replay has no settings payload.")
    known = {field.name for field in fields(PerfectMpcSettings)}
    return PerfectMpcSettings(
        **{name: value for name, value in raw.items() if name in known}
    )


def _problem_from_metadata(payload: dict[str, object]) -> MpcProblem:
    raw = payload.get("problem")
    if not isinstance(raw, dict):
        raise ValueError("Baseline replay has no problem payload.")
    return MpcProblem(**raw)


def _knots_from_metadata(
    payload: dict[str, object],
) -> tuple[np.ndarray, float]:
    controls = payload.get("control_parameterization")
    settings = payload.get("settings")
    if not isinstance(controls, dict) or not isinstance(settings, dict):
        raise ValueError("Replay has no control parameterization/settings payload.")
    if controls.get("type") != "low_frequency_acceleration_knots":
        raise ValueError("Replay does not contain MPPI acceleration knots.")
    return (
        np.asarray(controls["knots_m_s2"], dtype=np.float32),
        float(settings["horizon_s"]),
    )


def _bounded_knots(values: torch.Tensor, maximum: float) -> torch.Tensor:
    norms = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    scale = torch.clamp(maximum / torch.clamp(norms, min=1.0e-12), max=1.0)
    return values * scale


def _evaluate_knots_batch(
    simulator: WhipSimulator,
    initial_state,
    problem: MpcProblem,
    settings,
    knots: torch.Tensor,
) -> list[dict[str, object]]:
    if knots.ndim != 3 or knots.shape[2] != 3:
        raise ValueError("Batched validation knots must have shape BxMx3.")
    with torch.no_grad():
        controls = interpolate_control_knots(
            knots,
            simulator.settings.control_count,
            simulator.settings.maximum_acceleration_m_s2,
        )
        rollout = simulator.rollout(initial_state, controls, create_graph=False)
        real_cost, diagnostics, _ = evaluate_mppi_rollout(
            rollout, initial_state, problem, simulator, settings
        )
        smooth_cost, smooth_diagnostics = smooth_strike_surrogate(
            rollout, problem, settings
        )
    values: list[dict[str, object]] = []
    for index in range(knots.shape[0]):
        geometric_contact = bool(
            diagnostics["geometric_tip_contact"][index].cpu()
        )
        non_tip_safe = bool(
            diagnostics["non_tip_contact_violation"][index].cpu() == 0.0
        )
        values.append(
            {
                "smooth_surrogate_cost": float(smooth_cost[index].cpu()),
                "real_mppi_cost": float(real_cost[index].cpu()),
                "target_error_m": float(
                    diagnostics["position_error_m"][index].cpu()
                ),
                "directed_tip_speed_m_s": float(
                    diagnostics["directional_speed_m_s"][index].cpu()
                ),
                "total_tip_speed_m_s": float(
                    diagnostics["tip_speed_m_s"][index].cpu()
                ),
                "direction_error_deg": float(
                    diagnostics["direction_error_deg"][index].cpu()
                ),
                "geometric_tip_contact": geometric_contact,
                "physical_tip_contact": bool(
                    diagnostics["physical_tip_contact"][index].cpu()
                ),
                "tip_first": geometric_contact and non_tip_safe,
                "minimum_non_tip_target_distance_m": float(
                    diagnostics["minimum_non_tip_target_distance_m"][index].cpu()
                ),
                "minimum_non_tip_clearance_m": float(
                    diagnostics["minimum_non_tip_target_distance_m"][index].cpu()
                    - problem.maximum_tip_error_m
                ),
                "feasible": bool(diagnostics["feasible"][index].cpu()),
                "smooth_impact_time_s": float(
                    smooth_diagnostics["approximate_impact_time_s"][index].cpu()
                ),
            }
        )
    del rollout, controls
    return values


def _local_direction_summaries(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Pair +/- line-search samples into compact direction diagnostics."""

    summaries: list[dict[str, object]] = []
    for case_name in sorted({str(row["case"]) for row in rows}):
        ratios = sorted(
            {
                float(row["delta_sigma_ratio"])
                for row in rows
                if row["case"] == case_name
            }
        )
        for ratio in ratios:
            selected = {
                str(row["direction"]): row
                for row in rows
                if row["case"] == case_name
                and float(row["delta_sigma_ratio"]) == ratio
            }
            if set(selected) != {
                "negative_gradient",
                "nominal",
                "positive_gradient",
            }:
                continue
            negative = selected["negative_gradient"]
            nominal = selected["nominal"]
            positive = selected["positive_gradient"]

            def change(name: str) -> float:
                return float(negative[name]) - float(nominal[name])

            summaries.append(
                {
                    "case": case_name,
                    "delta_sigma_ratio": ratio,
                    "smooth_local_ordering": (
                        float(negative["smooth_surrogate_cost"])
                        < float(nominal["smooth_surrogate_cost"])
                        < float(positive["smooth_surrogate_cost"])
                    ),
                    "negative_gradient_smooth_cost_change": change(
                        "smooth_surrogate_cost"
                    ),
                    "negative_gradient_real_cost_change": change("real_mppi_cost"),
                    "negative_gradient_target_error_change_mm": 1000.0
                    * change("target_error_m"),
                    "negative_gradient_directed_speed_change_m_s": change(
                        "directed_tip_speed_m_s"
                    ),
                    "negative_gradient_direction_error_change_deg": change(
                        "direction_error_deg"
                    ),
                    "negative_gradient_clearance_change_mm": 1000.0
                    * change("minimum_non_tip_clearance_m"),
                    "negative_gradient_tip_first": negative["tip_first"],
                    "negative_gradient_feasible": negative["feasible"],
                }
            )
    return summaries


def run_local_validation(arguments: argparse.Namespace) -> Path:
    output = arguments.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline_metadata = _load_metadata(arguments.baseline_replay)
    snapshot = load_cable_model(arguments.model)
    if snapshot.sha256 != baseline_metadata.get("model_sha256"):
        raise ValueError("Baseline and selected cable-model hashes do not match.")
    problem = replace(
        _problem_from_metadata(baseline_metadata),
        minimum_impact_speed_m_s=arguments.minimum_impact_speed_m_s,
    )
    base = _settings_from_metadata(baseline_metadata)
    settings = replace(
        base,
        horizon_s=2.0,
        mppi_knot_count=16,
        mppi_gradient_guidance_fraction=0.5,
        mppi_gradient_step_sigma_ratio=arguments.gradient_step_sigma_ratio,
        mppi_smooth_softmin_temperature_m=arguments.softmin_temperature_m,
    )
    simulator = WhipSimulator(
        snapshot, settings.simulation_settings(), device=arguments.device
    )
    initial_drone = tuple(float(value) for value in problem.drone_workspace_center_m)
    state = simulator.initial_state(initial_drone)
    all_cases = (
        ("known_success_source", arguments.baseline_replay),
        ("near_success_source", arguments.near_success_replay),
    )
    cases = tuple(
        item
        for item in all_cases
        if arguments.validation_cases == "all"
        or (arguments.validation_cases == "success" and item[0].startswith("known"))
        or (arguments.validation_cases == "near" and item[0].startswith("near"))
    )
    validation_json = output / "gradient_direction_validation.json"
    rows: list[dict[str, object]] = []
    if validation_json.is_file():
        loaded = json.loads(validation_json.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            rows = loaded
    if rows and not arguments.rerun:
        incompatible = [
            row
            for row in rows
            if not math.isclose(
                float(row.get("minimum_impact_speed_m_s", math.nan)),
                problem.minimum_impact_speed_m_s,
            )
            or not math.isclose(
                float(row.get("maximum_tip_error_m", math.nan)),
                problem.maximum_tip_error_m,
            )
        ]
        if incompatible:
            raise ValueError(
                "Existing validation rows use a different strike definition. "
                "Use --rerun or a new --output-dir; results with different "
                "speed/position thresholds must not be mixed."
            )
    for row in rows:
        # Migrate checkpoints written before explicit contact/clearance fields
        # were added.  This simulator has geometric target events and no
        # separate physical-contact engine, so this value is intentionally false.
        row.setdefault("physical_tip_contact", False)
        row.setdefault(
            "minimum_non_tip_target_distance_m",
            float(row["minimum_non_tip_clearance_m"])
            + problem.maximum_tip_error_m,
        )
    completed_cases = {str(row["case"]) for row in rows}
    for case_name, source_path in cases:
        if case_name in completed_cases and not arguments.rerun:
            print(f"skip completed local validation: {case_name}")
            continue
        if arguments.rerun:
            rows = [row for row in rows if row["case"] != case_name]
        metadata = _load_metadata(source_path)
        if metadata.get("model_sha256") != snapshot.sha256:
            raise ValueError(f"Model hash mismatch for {source_path}.")
        source_knots, source_horizon = _knots_from_metadata(metadata)
        knots_np = resample_knots_in_time(source_knots, source_horizon, 2.0, 16)
        knots = torch.as_tensor(knots_np, dtype=simulator.dtype, device=simulator.device)
        guidance = compute_dder_guidance(
            simulator,
            state,
            problem,
            settings.mppi_settings(),
            knots,
            simulator.settings.control_count,
        )
        if not guidance.valid or guidance.negative_gradient_direction is None:
            raise RuntimeError(
                f"DDER gradient invalid for {case_name}: {guidance.reason}"
            )
        print(
            f"{case_name}: smooth={guidance.smooth_cost:.5g}, "
            f"|grad|={guidance.gradient_norm:.5g}, "
            f"gradient time={guidance.computation_time_s:.2f}s"
        )
        dimension_scale = math.sqrt(16 * 3)
        candidate_metadata: list[dict[str, object]] = []
        candidate_knots: list[torch.Tensor] = []
        for ratio in arguments.delta_sigma_ratios:
            delta = ratio * settings.mppi_noise_sigma_m_s2 * dimension_scale
            for sign, direction_label in (
                (1.0, "negative_gradient"),
                (0.0, "nominal"),
                (-1.0, "positive_gradient"),
            ):
                candidate = _bounded_knots(
                    knots + sign * delta * guidance.negative_gradient_direction,
                    settings.maximum_acceleration_m_s2,
                )
                candidate_metadata.append(
                    {
                        "schema": SCHEMA,
                        "case": case_name,
                        "source_replay": str(source_path.expanduser().resolve()),
                        "direction": direction_label,
                        "delta_sigma_ratio": ratio,
                        "delta_l2_m_s2": delta,
                        "gradient_norm": guidance.gradient_norm,
                        "gradient_computation_time_s": guidance.computation_time_s,
                        "softmin_temperature_m": arguments.softmin_temperature_m,
                        "minimum_impact_speed_m_s": (
                            problem.minimum_impact_speed_m_s
                        ),
                        "maximum_tip_error_m": problem.maximum_tip_error_m,
                    }
                )
                candidate_knots.append(candidate)
        evaluated = _evaluate_knots_batch(
            simulator,
            state,
            problem,
            settings.mppi_settings(),
            torch.stack(candidate_knots),
        )
        for row, metrics in zip(candidate_metadata, evaluated, strict=True):
            row.update(metrics)
            rows.append(row)
    _write_csv(output / "gradient_direction_validation.csv", rows)
    validation_json.write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    summaries = _local_direction_summaries(rows)
    _write_csv(output / "gradient_direction_summary.csv", summaries)
    (output / "gradient_direction_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output


def _wilson(successes: int, trials: int) -> tuple[float, float]:
    z = 1.959963984540054
    probability = successes / trials
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(
        probability * (1.0 - probability) / trials
        + z * z / (4.0 * trials * trials)
    ) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition.key]
        if not selected:
            continue
        successes = sum(bool(row["feasible"]) for row in selected)
        low, high = _wilson(successes, len(selected))

        def median(name: str) -> float:
            return float(np.median([float(row[name]) for row in selected]))

        first = [
            float(row["first_valid_strike_iteration"])
            for row in selected
            if row["first_valid_strike_iteration"] not in (None, "")
        ]
        result.append(
            {
                "condition": condition.key,
                "label": condition.label,
                "primary": condition.primary,
                "runs": len(selected),
                "successes": successes,
                "success_rate": successes / len(selected),
                "success_wilson_95_low": low,
                "success_wilson_95_high": high,
                "median_first_valid_strike_iteration": (
                    float(np.median(first)) if first else ""
                ),
                "median_target_error_mm": 1000.0 * median("target_error_m"),
                "median_directed_tip_speed_m_s": median("directed_tip_speed_m_s"),
                "median_total_tip_speed_m_s": median("total_tip_speed_m_s"),
                "median_direction_error_deg": median("direction_error_deg"),
                "median_impact_time_s": median("impact_time_s"),
                "median_non_tip_clearance_mm": 1000.0
                * median("minimum_non_tip_clearance_m"),
                "median_maximum_drone_speed_m_s": median(
                    "maximum_drone_speed_m_s"
                ),
                "median_maximum_drone_displacement_m": median(
                    "maximum_drone_displacement_m"
                ),
                "median_gradient_time_s": median("gradient_computation_time_s"),
                "median_total_runtime_s": median("total_runtime_s"),
            }
        )
    return result


def run_benchmark(arguments: argparse.Namespace) -> Path:
    output = arguments.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline_metadata = _load_metadata(arguments.baseline_replay)
    snapshot = load_cable_model(arguments.model)
    if snapshot.sha256 != baseline_metadata.get("model_sha256"):
        raise ValueError("Baseline and selected cable-model hashes do not match.")
    problem = replace(
        _problem_from_metadata(baseline_metadata),
        minimum_impact_speed_m_s=arguments.minimum_impact_speed_m_s,
    )
    base = _settings_from_metadata(baseline_metadata)
    initial_drone = tuple(float(value) for value in problem.drone_workspace_center_m)
    rows_path = output / "gradient_mppi_runs.json"
    rows: list[dict[str, object]] = []
    if rows_path.is_file() and not arguments.rerun:
        loaded = json.loads(rows_path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            rows = loaded
    if rows:
        incompatible = [
            row
            for row in rows
            if not math.isclose(
                float(row.get("minimum_impact_speed_m_s", math.nan)),
                problem.minimum_impact_speed_m_s,
            )
            or not math.isclose(
                float(row.get("maximum_tip_error_m", math.nan)),
                problem.maximum_tip_error_m,
            )
        ]
        if incompatible:
            raise ValueError(
                "Existing benchmark rows use a different strike definition. "
                "Use --rerun or a new --output-dir; results with different "
                "speed/position thresholds must not be mixed."
            )
    complete = {(str(row["condition"]), int(row["seed"])) for row in rows}
    for seed in arguments.seeds:
        for condition in CONDITIONS:
            key = (condition.key, seed)
            if key in complete and not arguments.rerun:
                print(f"skip completed {condition.key}, seed={seed}")
                continue
            if arguments.rerun:
                rows = [
                    row
                    for row in rows
                    if (str(row["condition"]), int(row["seed"])) != key
                ]
            settings = replace(
                base,
                horizon_s=2.0,
                mppi_knot_count=16,
                mppi_iterations=arguments.iterations,
                mppi_samples=arguments.samples,
                mppi_rollout_batch_size=min(arguments.batch_size, arguments.samples),
                mppi_gradient_guidance_fraction=(0.5 if condition.guided else 0.0),
                mppi_gradient_step_sigma_ratio=arguments.gradient_step_sigma_ratio,
                mppi_smooth_softmin_temperature_m=arguments.softmin_temperature_m,
                mppi_seed=seed,
            )
            warm = None
            if condition.initialization == "forward_recoil":
                warm = structured_seed_knots(
                    "forward_recoil",
                    problem,
                    2.0,
                    16,
                    settings.maximum_acceleration_m_s2,
                )
            print(
                f"\n{condition.key}: {condition.label}, seed={seed}, "
                f"guided={condition.guided}"
            )
            started = time.perf_counter()
            solved = solve_perfect_model_mpc(
                snapshot,
                problem,
                initial_drone,  # type: ignore[arg-type]
                settings,
                device=arguments.device,
                mppi_warm_start_knots_m_s2=warm,
                progress=(print if arguments.verbose else None),
            )
            runtime = time.perf_counter() - started
            if not isinstance(solved.plan, MppiPlan):
                raise TypeError("Gradient study requires an MPPI plan.")
            first_success = next(
                (
                    index + 1
                    for index, rate in enumerate(
                        solved.plan.sample_success_rate_history
                    )
                    if rate > 0.0
                ),
                None,
            )
            artifact = save_perfect_mpc_result(
                output / f"condition_{condition.key}_seed_{seed}.npz",
                solved,
                settings,
                problem,
            )
            row: dict[str, object] = {
                "schema": SCHEMA,
                "condition": condition.key,
                "label": condition.label,
                "primary": condition.primary,
                "seed": seed,
                "initialization": condition.initialization,
                "guided": condition.guided,
                "minimum_impact_speed_m_s": problem.minimum_impact_speed_m_s,
                "maximum_tip_error_m": problem.maximum_tip_error_m,
                "horizon_s": 2.0,
                "knot_count": 16,
                "samples": arguments.samples,
                "iterations": arguments.iterations,
                "artifact": str(artifact),
                "feasible": solved.feasible,
                "first_valid_strike_iteration": first_success,
                "target_error_m": solved.terms["position_error_m"],
                "directed_tip_speed_m_s": solved.terms["directional_speed_m_s"],
                "total_tip_speed_m_s": solved.terms["tip_speed_m_s"],
                "direction_error_deg": solved.terms["direction_error_deg"],
                "impact_time_s": solved.terms["impact_time_s"],
                "minimum_non_tip_clearance_m": (
                    solved.terms["minimum_non_tip_target_distance_m"]
                    - problem.maximum_tip_error_m
                ),
                "minimum_non_tip_target_distance_m": solved.terms[
                    "minimum_non_tip_target_distance_m"
                ],
                "physical_tip_contact": bool(
                    solved.terms["physical_tip_contact"]
                ),
                "maximum_drone_speed_m_s": solved.terms["maximum_drone_speed_m_s"],
                "maximum_drone_displacement_m": solved.terms[
                    "maximum_drone_excursion_m"
                ],
                "gradient_computation_time_s": sum(
                    solved.plan.gradient_computation_time_s_history
                ),
                "gradient_valid_iterations": sum(
                    solved.plan.gradient_valid_history
                ),
                "total_runtime_s": runtime,
            }
            rows.append(row)
            complete.add(key)
            rows_path.write_text(
                json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
            )
            _write_csv(output / "gradient_mppi_runs.csv", rows)
    summaries = _summaries(rows)
    _write_csv(output / "gradient_mppi_summary.csv", summaries)
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_sha256": snapshot.sha256,
        "baseline_replay": str(arguments.baseline_replay.expanduser().resolve()),
        "task_invariants": {
            "horizon_s": 2.0,
            "knot_count": 16,
            "minimum_impact_speed_m_s": problem.minimum_impact_speed_m_s,
            "maximum_tip_error_m": problem.maximum_tip_error_m,
            "real_cost": "unchanged hard MPPI event and safety objective",
            "conditions": [asdict(condition) for condition in CONDITIONS],
        },
        "compute": {
            "seeds": list(arguments.seeds),
            "iterations": arguments.iterations,
            "samples": arguments.samples,
            "batch_size": arguments.batch_size,
            "device": arguments.device,
        },
        "guidance": {
            "guided_fraction": 0.5,
            "gradient_step_sigma_ratio": arguments.gradient_step_sigma_ratio,
            "softmin_temperature_m": arguments.softmin_temperature_m,
        },
        "summary": summaries,
    }
    (output / "gradient_mppi_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output


def _parse_float_list(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item <= 0.0 for item in result):
        raise argparse.ArgumentTypeError("Expected positive comma-separated values.")
    return result


def _parse_int_list(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("Expected comma-separated integer seeds.")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("validation", "benchmark", "all"), default="validation"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--baseline-replay", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--near-success-replay", type=Path, default=DEFAULT_NEAR_SUCCESS
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/drone_mpc/ablations/dder_gradient_mppi"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--validation-cases",
        choices=("all", "success", "near"),
        default="all",
    )
    parser.add_argument("--softmin-temperature-m", type=float, default=0.05)
    parser.add_argument(
        "--minimum-impact-speed-m-s",
        type=float,
        default=DEFAULT_MINIMUM_IMPACT_SPEED_M_S,
        help=(
            "Directed free-tip speed required for a valid strike. The physical "
            "position tolerance remains unchanged."
        ),
    )
    parser.add_argument("--gradient-step-sigma-ratio", type=float, default=0.025)
    parser.add_argument(
        "--delta-sigma-ratios",
        type=_parse_float_list,
        default=(0.001, 0.005, 0.010, 0.025, 0.050),
    )
    parser.add_argument(
        "--seeds", type=_parse_int_list, default=tuple(range(201, 221))
    )
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.minimum_impact_speed_m_s <= 0.0:
        raise ValueError("--minimum-impact-speed-m-s must be positive.")
    if arguments.mode in {"validation", "all"}:
        run_local_validation(arguments)
    if arguments.mode in {"benchmark", "all"}:
        run_benchmark(arguments)


if __name__ == "__main__":
    main()
