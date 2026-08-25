from __future__ import annotations

import argparse
from pathlib import Path

from drone_mpc.model import load_cable_model
from drone_mpc.mpc import MpcProblem
from drone_mpc.oracle import (
    ReachabilitySettings,
    optimize_reachability,
    save_reachability_result,
)
from drone_mpc.trajectory_gui import DEFAULT_OUTPUT_PATH, main as gui_main
from optitrack_offline.config import DEFAULT_MODEL_PATH


def _xyz(value: str) -> tuple[float, float, float]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated X,Y,Z values.") from error
    if len(parsed) != 3:
        raise argparse.ArgumentTypeError("Expected exactly three comma-separated values.")
    return parsed  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize and verify one offline drone-whip trajectory."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the optimizer without opening the trajectory UI.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--target", type=_xyz, default=(0.48, 0.0, 0.90))
    parser.add_argument("--initial-drone", type=_xyz, default=(0.0, 0.0, 1.50))
    parser.add_argument("--direction", type=_xyz, default=(1.0, 0.0, 0.0))
    parser.add_argument("--minimum-speed", type=float, default=1.0)
    parser.add_argument("--hit-tolerance", type=float, default=0.05)
    parser.add_argument("--maximum-angle", type=float, default=35.0)
    parser.add_argument("--maximum-excursion", type=float, default=0.15)
    parser.add_argument("--horizon", type=float, default=4.0)
    parser.add_argument("--maximum-acceleration", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _run_headless(arguments: argparse.Namespace) -> None:
    snapshot = load_cable_model(arguments.model)
    settings = ReachabilitySettings(
        horizon_s=arguments.horizon,
        iterations=arguments.iterations,
        maximum_acceleration_m_s2=arguments.maximum_acceleration,
        seed=arguments.seed,
    )
    problem = MpcProblem(
        target_position_m=arguments.target,
        impact_direction=arguments.direction,
        minimum_impact_speed_m_s=arguments.minimum_speed,
        maximum_tip_error_m=arguments.hit_tolerance,
        maximum_impact_angle_deg=arguments.maximum_angle,
        maximum_drone_excursion_m=arguments.maximum_excursion,
        drone_keepout_radius_m=0.30,
        minimum_forward_stroke_m=0.05,
        minimum_recoil_stroke_m=0.05,
    )
    result = optimize_reachability(
        snapshot,
        problem,
        arguments.initial_drone,
        settings,
        progress=print,
    )
    output = save_reachability_result(arguments.output, result, settings, problem)
    terms = result.terms
    print(
        f"exact full model: {'FEASIBLE HIT' if result.feasible else 'INFEASIBLE'}  "
        f"error={1000.0 * terms['position_error_m']:.1f} mm  "
        f"directed speed={terms['directional_speed_m_s']:.3f} m/s  "
        f"direction error={terms['direction_error_deg']:.1f} deg  "
        f"speed amplification={result.speed_amplification:.2f}x"
    )
    print(f"Saved {output}")


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.headless:
        _run_headless(arguments)
    else:
        gui_main()


if __name__ == "__main__":
    main()
