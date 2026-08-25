"""Reproducible MPPI ablations for the matched-model drone-whip task.

The verified far/fast replay is treated as immutable baseline evidence.  This
tool varies one research question at a time:

* horizon length at approximately fixed knot spacing;
* initialization family at a fixed task and budget;
* near-target velocity gating, including one weak predictive-speed ablation.

The structured trajectories are optimizer initializations, not task rewards or
hard-coded controller phases.  Every final candidate is evaluated with the
same full DDER rollout and the same physical strike definition.
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

from drone_mpc.model import load_cable_model
from drone_mpc.mppi import MppiPlan
from drone_mpc.mpc import MpcProblem
from drone_mpc.perfect_model import (
    PerfectMpcSettings,
    save_perfect_mpc_result,
    solve_perfect_model_mpc,
)
from optitrack_offline.config import DEFAULT_MODEL_PATH


SCHEMA = "mppi_whip_ablation_v1"
DEFAULT_MINIMUM_IMPACT_SPEED_M_S = 3.5


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    study: str
    name: str
    horizon_s: float
    knot_count: int
    initialization: str
    velocity_gate_sigma_m: float = 0.12
    predictive_speed_weight: float = 0.0
    predictive_velocity_gate_sigma_m: float = 0.45
    predictive_speed_ratio: float = 0.25


@dataclass(frozen=True, slots=True)
class ComputeProfile:
    iterations: int
    samples: int
    batch_size: int
    seeds: tuple[int, ...]


PROFILES = {
    # Smoke verifies the complete experiment and artifact pipeline.  One seed
    # and two iterations are explicitly not statistical evidence.
    "smoke": ComputeProfile(2, 32, 32, (17,)),
    # Pilot is useful for deciding whether the expensive paper sweep is worth
    # running, but uncertainty should still be reported.
    "pilot": ComputeProfile(6, 128, 128, (11, 17, 29)),
    # Pre-registered paired-seed discovery study.  All six initialization
    # strategies use the same twenty MPPI sampling seeds.
    "discovery": ComputeProfile(15, 256, 128, tuple(range(101, 121))),
    "full": ComputeProfile(
        15,
        256,
        128,
        (5, 11, 17, 23, 29, 37, 41, 47, 53, 61),
    ),
}


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("At least one integer seed is required.")
    return seeds


def resample_knots_in_time(
    source_knots_m_s2: np.ndarray,
    source_horizon_s: float,
    target_horizon_s: float,
    target_knot_count: int,
) -> np.ndarray:
    """Preserve the absolute timing of a seed while changing the horizon."""

    source = np.asarray(source_knots_m_s2, dtype=np.float64)
    if source.ndim != 2 or source.shape[0] < 2 or source.shape[1] != 3:
        raise ValueError("source knots must have shape Mx3 with M >= 2.")
    if source_horizon_s <= 0.0 or target_horizon_s <= 0.0:
        raise ValueError("source and target horizons must be positive.")
    if target_knot_count < 2:
        raise ValueError("target_knot_count must be at least two.")
    source_time = np.linspace(0.0, source_horizon_s, source.shape[0])
    target_time = np.linspace(0.0, target_horizon_s, target_knot_count)
    return np.stack(
        tuple(
            np.interp(target_time, source_time, source[:, axis], right=0.0)
            for axis in range(3)
        ),
        axis=1,
    ).astype(np.float32)


def _target_basis(problem: MpcProblem) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = np.asarray(problem.target_position_m, dtype=np.float64) - np.asarray(
        problem.drone_workspace_center_m, dtype=np.float64
    )
    forward[2] = 0.0
    if float(np.linalg.norm(forward)) <= 1.0e-9:
        forward = np.asarray(problem.impact_direction, dtype=np.float64).copy()
        forward[2] = 0.0
    forward /= max(float(np.linalg.norm(forward)), 1.0e-12)
    up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    lateral = np.cross(up, forward)
    lateral /= max(float(np.linalg.norm(lateral)), 1.0e-12)
    return forward, lateral, up


def structured_seed_knots(
    family: str,
    problem: MpcProblem,
    horizon_s: float,
    knot_count: int,
    maximum_acceleration_m_s2: float,
) -> np.ndarray:
    """Return a simple deterministic seed family, not a prescribed policy."""

    forward, lateral, up = _target_basis(problem)
    if family == "forward_recoil":
        first, second = forward, -forward
    elif family == "backward_forward":
        first, second = -forward, forward
    elif family == "lateral":
        first, second = lateral, -lateral
    elif family == "vertical":
        first, second = up, -up
    else:
        raise ValueError(f"Unknown structured initialization family: {family}")
    times = np.linspace(0.0, horizon_s, knot_count)
    result = np.zeros((knot_count, 3), dtype=np.float64)
    first_stop = min(0.24, 0.30 * horizon_s)
    second_stop = min(0.58, 0.65 * horizon_s)
    result[times <= first_stop] = 0.60 * maximum_acceleration_m_s2 * first
    second_mask = (times > first_stop) & (times <= second_stop)
    result[second_mask] = 0.85 * maximum_acceleration_m_s2 * second
    return result.astype(np.float32)


def random_seed_knots(
    seed: int,
    knot_count: int,
    maximum_acceleration_m_s2: float,
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    values = generator.normal(size=(knot_count, 3))
    if knot_count > 2:
        padded = np.pad(values, ((1, 1), (0, 0)), mode="edge")
        values = 0.25 * padded[:-2] + 0.50 * padded[1:-1] + 0.25 * padded[2:]
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    values /= np.maximum(norms, 1.0)
    return (0.45 * maximum_acceleration_m_s2 * values).astype(np.float32)


def build_experiment_specs(study: str) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    if study in {"horizon", "all"}:
        # The verified baseline has roughly 0.20 s between its 16 knots.  Keep
        # that spacing approximately fixed so this isolates prediction horizon
        # rather than silently increasing control bandwidth.
        for horizon in (1.0, 1.5, 2.0, 3.0):
            knots = int(round(horizon / 0.20)) + 1
            specs.append(
                ExperimentSpec(
                    study="horizon",
                    name=f"horizon_{horizon:.1f}s",
                    horizon_s=horizon,
                    knot_count=knots,
                    initialization="continuation",
                )
            )
    if study in {"initialization", "all"}:
        for initialization in (
            "zero",
            "random",
            "forward_recoil",
            "backward_forward",
            "lateral",
            "continuation",
        ):
            specs.append(
                ExperimentSpec(
                    study="initialization",
                    name=f"initialization_{initialization}",
                    horizon_s=2.0,
                    knot_count=11,
                    initialization=initialization,
                )
            )
    if study in {"gate", "all"}:
        specs.extend(
            (
                ExperimentSpec(
                    study="gate",
                    name="gate_current_0.12m",
                    horizon_s=1.5,
                    knot_count=9,
                    initialization="zero",
                    velocity_gate_sigma_m=0.12,
                ),
                ExperimentSpec(
                    study="gate",
                    name="gate_broad_0.24m",
                    horizon_s=1.5,
                    knot_count=9,
                    initialization="zero",
                    velocity_gate_sigma_m=0.24,
                ),
                ExperimentSpec(
                    study="gate",
                    name="gate_current_plus_weak_predictive",
                    horizon_s=1.5,
                    knot_count=9,
                    initialization="zero",
                    velocity_gate_sigma_m=0.12,
                    predictive_speed_weight=1.0,
                    predictive_velocity_gate_sigma_m=0.45,
                    predictive_speed_ratio=0.25,
                ),
            )
        )
    return specs


def _load_baseline(
    replay_path: Path,
) -> tuple[dict[str, object], np.ndarray, float]:
    metadata = json.loads(replay_path.with_suffix(".json").read_text(encoding="utf-8"))
    settings = metadata.get("settings")
    controls = metadata.get("control_parameterization")
    if not isinstance(settings, dict) or not isinstance(controls, dict):
        raise ValueError("Baseline replay metadata is incomplete.")
    if controls.get("type") != "low_frequency_acceleration_knots":
        raise ValueError("Baseline replay is not a low-frequency MPPI plan.")
    knots = np.asarray(controls["knots_m_s2"], dtype=np.float32)
    return metadata, knots, float(settings["horizon_s"])


def _settings_from_metadata(payload: dict[str, object]) -> PerfectMpcSettings:
    known = {field.name for field in fields(PerfectMpcSettings)}
    return PerfectMpcSettings(
        **{name: value for name, value in payload.items() if name in known}
    )


def _warm_start(
    spec: ExperimentSpec,
    seed: int,
    problem: MpcProblem,
    baseline_knots: np.ndarray,
    baseline_horizon_s: float,
    maximum_acceleration_m_s2: float,
) -> np.ndarray | None:
    if spec.initialization == "zero":
        return None
    if spec.initialization == "continuation":
        return resample_knots_in_time(
            baseline_knots,
            baseline_horizon_s,
            spec.horizon_s,
            spec.knot_count,
        )
    if spec.initialization == "random":
        return random_seed_knots(seed, spec.knot_count, maximum_acceleration_m_s2)
    return structured_seed_knots(
        spec.initialization,
        problem,
        spec.horizon_s,
        spec.knot_count,
        maximum_acceleration_m_s2,
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    names = sorted({name for row in rows for name in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    keys = sorted(
        {
            (
                str(row["study"]),
                str(row["experiment"]),
                str(row["profile"]),
                int(row["iterations"]),
                int(row["samples"]),
                int(row["batch_size"]),
                float(row["horizon_s"]),
                int(row["knot_count"]),
                float(row["velocity_gate_sigma_m"]),
                float(row["predictive_speed_weight"]),
                float(row.get("minimum_impact_speed_m_s", 5.0)),
                float(row.get("maximum_tip_error_m", 0.05)),
            )
            for row in rows
        }
    )
    for (
        study,
        experiment,
        profile,
        iterations,
        samples,
        batch_size,
        horizon_s,
        knot_count,
        velocity_gate_sigma_m,
        predictive_speed_weight,
        minimum_impact_speed_m_s,
        maximum_tip_error_m,
    ) in keys:
        selected = [
            row
            for row in rows
            if row["study"] == study
            and row["experiment"] == experiment
            and row["profile"] == profile
            and int(row["iterations"]) == iterations
            and int(row["samples"]) == samples
            and int(row["batch_size"]) == batch_size
            and float(row["horizon_s"]) == horizon_s
            and int(row["knot_count"]) == knot_count
            and float(row["velocity_gate_sigma_m"]) == velocity_gate_sigma_m
            and float(row["predictive_speed_weight"]) == predictive_speed_weight
            and float(row.get("minimum_impact_speed_m_s", 5.0))
            == minimum_impact_speed_m_s
            and float(row.get("maximum_tip_error_m", 0.05))
            == maximum_tip_error_m
        ]
        success = np.asarray([bool(row["feasible"]) for row in selected])
        error = np.asarray([float(row["position_error_m"]) for row in selected])
        speed = np.asarray([float(row["directional_speed_m_s"]) for row in selected])
        angle = np.asarray([float(row["direction_error_deg"]) for row in selected])
        runtime = np.asarray([float(row["runtime_s"]) for row in selected])
        first_success = np.asarray(
            [
                float(row["first_success_iteration"])
                for row in selected
                if row.get("first_success_iteration") not in (None, "")
            ],
            dtype=np.float64,
        )
        reversal = np.asarray(
            [float(row["reversal_time_s"]) for row in selected], dtype=np.float64
        )
        clearance = np.asarray(
            [1000.0 * float(row["non_tip_clearance_margin_m"]) for row in selected],
            dtype=np.float64,
        )
        success_count = int(np.sum(success))
        probability, ci_low, ci_high = _wilson_interval(success_count, len(selected))
        summaries.append(
            {
                "study": study,
                "experiment": experiment,
                "profile": profile,
                "iterations": iterations,
                "samples": samples,
                "batch_size": batch_size,
                "horizon_s": horizon_s,
                "knot_count": knot_count,
                "velocity_gate_sigma_m": velocity_gate_sigma_m,
                "predictive_speed_weight": predictive_speed_weight,
                "minimum_impact_speed_m_s": minimum_impact_speed_m_s,
                "maximum_tip_error_m": maximum_tip_error_m,
                "runs": len(selected),
                "success_count": success_count,
                "success_rate": probability,
                "success_wilson_95_low": ci_low,
                "success_wilson_95_high": ci_high,
                "median_tip_error_mm": float(1000.0 * np.median(error)),
                "median_directed_speed_m_s": float(np.median(speed)),
                "median_direction_error_deg": float(np.median(angle)),
                "median_runtime_s": float(np.median(runtime)),
                "median_first_success_iteration": (
                    float(np.median(first_success)) if first_success.size else ""
                ),
                "median_reversal_time_s": float(np.median(reversal)),
                "median_non_tip_clearance_margin_mm": float(np.median(clearance)),
                "statistical_warning": (
                    "single-seed smoke test; not inferential evidence"
                    if len(selected) == 1
                    else ""
                ),
            }
        )
    return summaries


def _wilson_interval(successes: int, trials: int) -> tuple[float, float, float]:
    """Return the observed proportion and two-sided Wilson 95% interval."""

    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Wilson interval requires 0 <= successes <= trials.")
    probability = successes / trials
    z = 1.959963984540054
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return probability, max(0.0, center - half_width), min(1.0, center + half_width)


def _run_identifier(
    spec: ExperimentSpec,
    profile: str,
    iterations: int,
    samples: int,
    batch_size: int,
    seed: int,
    minimum_impact_speed_m_s: float,
) -> str:
    return (
        f"{spec.name}__T{spec.horizon_s:g}__k{spec.knot_count}__"
        f"vg{spec.velocity_gate_sigma_m:g}__pw{spec.predictive_speed_weight:g}__"
        f"{profile}__i{iterations}__n{samples}__b{batch_size}__"
        f"v{minimum_impact_speed_m_s:g}__seed{seed}"
    )


def _row_run_identifier(row: dict[str, object]) -> str:
    value = row.get("run_id")
    if value is not None:
        return str(value)
    return (
        f"{row['experiment']}__{row['profile']}__i{row['iterations']}__"
        f"n{row['samples']}__b{row['batch_size']}__seed{row['seed']}"
    )


def run_benchmark(arguments: argparse.Namespace) -> Path:
    output = arguments.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    replay = arguments.baseline_replay.expanduser().resolve()
    metadata, baseline_knots, baseline_horizon = _load_baseline(replay)
    snapshot = load_cable_model(arguments.model)
    if snapshot.sha256 != metadata.get("model_sha256"):
        raise ValueError("Baseline and selected cable-model hashes do not match.")

    raw_settings = metadata.get("settings")
    raw_problem = metadata.get("problem")
    if not isinstance(raw_settings, dict) or not isinstance(raw_problem, dict):
        raise ValueError("Baseline metadata has no settings/problem payload.")
    base_settings = _settings_from_metadata(raw_settings)
    problem = replace(
        MpcProblem(**raw_problem),
        minimum_impact_speed_m_s=arguments.minimum_impact_speed_m_s,
    )
    initial_drone = tuple(float(value) for value in problem.drone_workspace_center_m)
    profile = PROFILES[arguments.profile]
    seeds = arguments.seeds if arguments.seeds is not None else profile.seeds
    iterations = arguments.iterations or profile.iterations
    samples = arguments.samples or profile.samples
    batch_size = arguments.batch_size or min(profile.batch_size, samples)
    specs = build_experiment_specs(arguments.study)

    rows: list[dict[str, object]] = []
    checkpoint = output / "runs.json"
    if checkpoint.is_file() and not arguments.rerun:
        existing = json.loads(checkpoint.read_text(encoding="utf-8"))
        if isinstance(existing, list):
            rows.extend(existing)
    completed = {_row_run_identifier(row) for row in rows}

    for spec in specs:
        for seed in seeds:
            run_key = _run_identifier(
                spec,
                arguments.profile,
                iterations,
                samples,
                batch_size,
                seed,
                problem.minimum_impact_speed_m_s,
            )
            if run_key in completed and not arguments.rerun:
                print(f"skip completed {spec.name}, seed={seed}")
                continue
            if arguments.rerun:
                rows = [row for row in rows if _row_run_identifier(row) != run_key]
                completed.discard(run_key)
            settings = replace(
                base_settings,
                horizon_s=spec.horizon_s,
                mppi_knot_count=spec.knot_count,
                mppi_iterations=iterations,
                mppi_samples=samples,
                mppi_rollout_batch_size=batch_size,
                mppi_velocity_gate_sigma_m=spec.velocity_gate_sigma_m,
                mppi_predictive_speed_weight=spec.predictive_speed_weight,
                mppi_predictive_velocity_gate_sigma_m=(
                    spec.predictive_velocity_gate_sigma_m
                ),
                mppi_predictive_speed_ratio=spec.predictive_speed_ratio,
                mppi_seed=seed,
            )
            warm = _warm_start(
                spec,
                seed,
                problem,
                baseline_knots,
                baseline_horizon,
                settings.maximum_acceleration_m_s2,
            )
            print(
                f"\n[{spec.study}] {spec.name}, seed={seed}, "
                f"T={spec.horizon_s:g}s, knots={spec.knot_count}, "
                f"iterations={iterations}, samples={samples}"
            )
            started = time.perf_counter()
            result = solve_perfect_model_mpc(
                snapshot,
                problem,
                initial_drone,  # type: ignore[arg-type]
                settings,
                device=arguments.device,
                mppi_warm_start_knots_m_s2=warm,
                progress=(print if arguments.verbose else None),
            )
            runtime = time.perf_counter() - started
            run_name = _safe_name(run_key)
            replay_path = save_perfect_mpc_result(
                output / f"{run_name}.npz", result, settings, problem
            )
            row: dict[str, object] = {
                "schema": SCHEMA,
                "run_id": run_key,
                "study": spec.study,
                "experiment": spec.name,
                "seed": seed,
                "profile": arguments.profile,
                "iterations": iterations,
                "samples": samples,
                "batch_size": batch_size,
                "horizon_s": spec.horizon_s,
                "knot_count": spec.knot_count,
                "initialization": spec.initialization,
                "velocity_gate_sigma_m": spec.velocity_gate_sigma_m,
                "predictive_speed_weight": spec.predictive_speed_weight,
                "minimum_impact_speed_m_s": problem.minimum_impact_speed_m_s,
                "maximum_tip_error_m": problem.maximum_tip_error_m,
                "runtime_s": runtime,
                "artifact": str(replay_path),
                "feasible": result.feasible,
            }
            if isinstance(result.plan, MppiPlan):
                row["first_success_iteration"] = next(
                    (
                        index + 1
                        for index, rate in enumerate(
                            result.plan.sample_success_rate_history
                        )
                        if rate > 0.0
                    ),
                    None,
                )
                row["sample_success_rate_history"] = json.dumps(
                    result.plan.sample_success_rate_history
                )
            else:
                row["first_success_iteration"] = None
                row["sample_success_rate_history"] = "[]"
            row["reversal_time_s"] = result.energy.injection_end_s
            row["non_tip_clearance_margin_m"] = (
                result.terms["minimum_non_tip_target_distance_m"]
                - problem.maximum_tip_error_m
            )
            for name in (
                "position_error_m",
                "directional_speed_m_s",
                "tip_speed_m_s",
                "direction_error_deg",
                "impact_time_s",
                "drone_displacement_at_impact_m",
                "maximum_drone_excursion_m",
                "minimum_non_tip_target_distance_m",
                "mppi_objective",
            ):
                row[name] = result.terms[name]
            rows.append(row)
            completed.add(run_key)
            checkpoint.write_text(
                json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
            )
            _write_csv(output / "runs.csv", rows)

    summaries = _summarize(rows)
    _write_csv(output / "summary.csv", summaries)
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_replay": str(replay),
        "baseline_model_sha256": snapshot.sha256,
        "study": arguments.study,
        "profile": arguments.profile,
        "compute": {
            "iterations": iterations,
            "samples": samples,
            "batch_size": batch_size,
            "seeds": list(seeds),
            "device": arguments.device,
        },
        "design": [asdict(spec) for spec in specs],
        "strike_definition": {
            "minimum_impact_speed_m_s": problem.minimum_impact_speed_m_s,
            "maximum_tip_error_m": problem.maximum_tip_error_m,
            "maximum_impact_angle_deg": problem.maximum_impact_angle_deg,
        },
        "summary": summaries,
        "warning": (
            "Smoke results validate software paths only; use the full multi-seed "
            "profile before drawing research conclusions."
            if arguments.profile == "smoke"
            else ""
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"\nBenchmark: {output}")
    print(f"Runs: {output / 'runs.csv'}")
    print(f"Summary: {output / 'summary.csv'}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--baseline-replay",
        type=Path,
        default=Path("data/drone_mpc/perfect_model_mppi_farther_faster.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/drone_mpc/ablations/mppi_far_fast"),
    )
    parser.add_argument(
        "--study",
        choices=("all", "horizon", "initialization", "gate"),
        default="all",
    )
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument("--seeds", type=_parse_seeds)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--minimum-impact-speed-m-s",
        type=float,
        default=DEFAULT_MINIMUM_IMPACT_SPEED_M_S,
        help="Directed free-tip speed required for a valid strike.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.minimum_impact_speed_m_s <= 0.0:
        raise ValueError("--minimum-impact-speed-m-s must be positive.")
    run_benchmark(arguments)


if __name__ == "__main__":
    main()
