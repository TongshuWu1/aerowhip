from __future__ import annotations

import argparse
from pathlib import Path

from drone_mpc.benchmark import (
    default_mismatch_cases,
    default_resolution_cases,
    run_mismatch_benchmark,
    save_benchmark_results,
)
from drone_mpc.model import load_cable_model
from drone_mpc.mpc import MpcProblem
from drone_mpc.realtime import RealtimeSettings
from optitrack_offline.config import DEFAULT_MODEL_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark reduced-DER whip MPC against the full fitted DER plant."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/drone_mpc/controller_benchmark.json"),
    )
    parser.add_argument("--initial-iterations", type=int, default=60)
    parser.add_argument("--update-iterations", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=1.0e-4)
    parser.add_argument("--initial-time-limit", type=float, default=15.0)
    parser.add_argument("--update-time-limit", type=float, default=2.0)
    parser.add_argument(
        "--study",
        choices=("mismatch", "resolution", "both"),
        default="both",
        help="Run parameter sensitivity, controller resolution, or both.",
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    snapshot = load_cable_model(arguments.model)
    settings = RealtimeSettings(
        physics_dt_s=0.02,
        horizon_s=2.5,
        control_interval_s=0.1,
        replan_interval_s=0.4,
        mission_duration_s=2.5,
        matched_model=False,
        initial_ipopt_iterations=arguments.initial_iterations,
        ipopt_iterations=arguments.update_iterations,
        ipopt_tolerance=arguments.tolerance,
        ipopt_acceptable_tolerance=10.0 * arguments.tolerance,
        initial_ipopt_max_wall_time_s=arguments.initial_time_limit,
        ipopt_max_wall_time_s=arguments.update_time_limit,
    )
    problem = MpcProblem(
        target_position_m=(0.48, 0.0, 1.35),
        impact_direction=(1.0, 0.0, 0.0),
        minimum_impact_speed_m_s=1.5,
        drone_keepout_radius_m=0.30,
        maximum_drone_excursion_m=0.15,
        maximum_tip_error_m=0.05,
        planning_tip_error_margin_m=0.003,
        maximum_impact_angle_deg=35.0,
        planning_impact_angle_margin_deg=2.0,
    )
    cases = ()
    if arguments.study in ("mismatch", "both"):
        cases += default_mismatch_cases()
    if arguments.study in ("resolution", "both"):
        cases += default_resolution_cases(snapshot.node_count)
    results = run_mismatch_benchmark(
        snapshot,
        settings,
        problem,
        (0.0, 0.0, 1.5),
        cases=cases,
        progress=print,
    )
    output = save_benchmark_results(arguments.output, results, settings, problem)
    for result in results:
        outcome = "HIT" if result.actual_hit_feasible else "MISS"
        print(
            f"{result.case}: {outcome} error={1000.0 * result.tip_error_m:.1f}mm "
            f"speed={result.directional_speed_m_s:.2f}m/s "
            f"angle={result.direction_error_deg:.1f}deg "
            f"excursion={result.maximum_drone_excursion_m:.3f}m "
            f"controller={result.controller_node_count}nodes/"
            f"{result.controller_substeps}substeps "
            f"deadline_misses={result.controller_deadline_misses}/"
            f"{result.replan_count}"
        )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
