from __future__ import annotations

import argparse
from pathlib import Path

from drone_mpc.model import load_cable_model
from drone_mpc.mpc import MpcProblem
from drone_mpc.oracle import (
    REACHABILITY_RANK_TERMS,
    ReachabilityResult,
    ReachabilitySettings,
    optimize_reachability,
    reachability_result_rank,
    save_reachability_result,
)
from drone_mpc.reduced import stable_controller_model
from optitrack_offline.config import DEFAULT_MODEL_PATH


DEFAULT_TARGETS = (
    (0.55, 0.00),
    (0.60, -0.20),
    (0.60, -0.05),
    (0.65, -0.10),
    (0.80, -0.10),
    (0.92, -0.20),
    (0.98, 0.00),
    (1.00, -0.20),
    (1.05, -0.20),
    (1.05, -0.10),
    (1.10, -0.10),
    (1.15, -0.20),
)


def _radius_height(value: str) -> tuple[float, float]:
    try:
        radius, height = (float(item.strip()) for item in value.split(","))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("Expected RADIUS,HEIGHT in metres.") from error
    if radius <= 0.0:
        raise argparse.ArgumentTypeError("Target radius must be positive.")
    return radius, height


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("Expected a positive integer.") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate verified MPC anchors for goal-conditioned whip SAC."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/drone_mpc/goal_demos_generated12"),
    )
    parser.add_argument(
        "--target",
        action="append",
        type=_radius_height,
        default=[],
        help=(
            "Training anchor as RADIUS,HEIGHT in metres. Repeat for multiple "
            "anchors; the default is a balanced twelve-target near/far set."
        ),
    )
    parser.add_argument("--nodes", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--candidates", type=int, default=512)
    parser.add_argument("--minimum-speed", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--restarts",
        type=_positive_int,
        default=4,
        help="Independent deterministic MPC searches per target (default: 4).",
    )
    return parser


def _name(index: int, radius: float, height: float) -> str:
    radius_mm = int(round(1000.0 * radius))
    height_mm = int(round(1000.0 * height))
    height_token = f"p{height_mm}" if height_mm >= 0 else f"m{abs(height_mm)}"
    return f"demo_{index:02d}_r{radius_mm}_h{height_token}.npz"


def _restart_seed(base_seed: int, target_index: int, restart_index: int) -> int:
    """Stable seed schedule that does not change when restart count changes."""

    return base_seed + 10_000 * (target_index - 1) + restart_index


def _select_best_feasible(
    attempts: list[tuple[int, ReachabilitySettings, ReachabilityResult]],
) -> tuple[int, ReachabilitySettings, ReachabilityResult] | None:
    feasible = [attempt for attempt in attempts if attempt[2].feasible]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda attempt: (reachability_result_rank(attempt[2]), attempt[0]),
    )


def _attempt_provenance(
    attempts: list[tuple[int, ReachabilitySettings, ReachabilityResult]],
    *,
    target_index: int,
    base_seed: int,
    selected_restart: int,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for restart_index, settings, result in attempts:
        records.append(
            {
                "restart_index": restart_index,
                "seed": settings.seed,
                "feasible": result.feasible,
                "impact_frame": result.impact_frame,
                "canonical_rank": list(reachability_result_rank(result)),
                "terms": {name: float(value) for name, value in result.terms.items()},
            }
        )
    return {
        "generator": "python -m research_tools.goal_demos",
        "target_index": target_index,
        "base_seed": base_seed,
        "requested_restarts": len(attempts),
        "restart_seed_formula": "base_seed + 10000 * (target_index - 1) + restart_index",
        "rank_terms": list(REACHABILITY_RANK_TERMS),
        "selection": "canonical_lexicographic_rank_among_exact_feasible_restarts",
        "selected_restart_index": selected_restart,
        "attempts": records,
    }


def main() -> None:
    arguments = build_parser().parse_args()
    targets = tuple(arguments.target) if arguments.target else DEFAULT_TARGETS
    if len(set(targets)) != len(targets):
        raise ValueError("Goal-demonstration targets must be distinct.")
    output_directory = arguments.output_directory.resolve()
    outputs = tuple(
        output_directory / _name(index, radius, height)
        for index, (radius, height) in enumerate(targets, start=1)
    )
    collisions = [output for output in outputs if output.exists()]
    if collisions:
        names = "\n- ".join(str(path) for path in collisions)
        raise FileExistsError(
            "Refusing to start because demonstration outputs already exist:\n- "
            + names
        )
    raw = load_cable_model(arguments.model)
    controller = stable_controller_model(
        raw,
        simulation_dt_s=0.02,
        node_count=arguments.nodes,
        constraint_iterations=4,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    saved = 0
    for index, ((radius, height), output) in enumerate(
        zip(targets, outputs, strict=True), start=1
    ):
        problem = MpcProblem(
            target_position_m=(radius, 0.0, height),
            impact_direction=(1.0, 0.0, 0.0),
            minimum_impact_speed_m_s=arguments.minimum_speed,
            maximum_tip_error_m=0.05,
            maximum_impact_angle_deg=35.0,
            maximum_drone_excursion_m=10.0,
            drone_keepout_radius_m=0.25,
            minimum_forward_stroke_m=0.05,
            minimum_recoil_stroke_m=0.05,
        )
        print(
            f"\nAnchor {index}/{len(targets)}: target=({radius:.2f}, 0, {height:.2f})m; "
            f"restarts={arguments.restarts}"
        )
        attempts: list[tuple[int, ReachabilitySettings, ReachabilityResult]] = []
        for restart_index in range(arguments.restarts):
            settings = ReachabilitySettings(
                iterations=arguments.iterations,
                candidates=arguments.candidates,
                elite_count=max(2, arguments.candidates // 8),
                seed=_restart_seed(arguments.seed, index, restart_index),
            )
            print(
                f"  restart {restart_index + 1}/{arguments.restarts}: "
                f"seed={settings.seed}"
            )
            result = optimize_reachability(
                controller,
                problem,
                (0.0, 0.0, 0.0),
                settings,
                progress=lambda message, restart=restart_index: print(
                    f"    [{restart + 1}] {message}"
                ),
            )
            attempts.append((restart_index, settings, result))
            print(
                f"    result feasible={result.feasible} "
                f"violation={result.terms['constraint_violation']:.4g} "
                f"error={1000.0 * result.terms['position_error_m']:.1f}mm "
                f"speed={result.terms['directional_speed_m_s']:.2f}m/s"
            )
        selected = _select_best_feasible(attempts)
        if selected is None:
            best_infeasible = min(
                attempts,
                key=lambda attempt: (reachability_result_rank(attempt[2]), attempt[0]),
            )
            failures.append(
                f"({radius:.2f}, 0, {height:.2f})m violation="
                f"{best_infeasible[2].terms['constraint_violation']:.4g}"
            )
            print("Not saved: no restart produced an exact feasible rollout.")
            continue
        selected_restart, selected_settings, selected_result = selected
        save_reachability_result(
            output,
            selected_result,
            selected_settings,
            problem,
            provenance=_attempt_provenance(
                attempts,
                target_index=index,
                base_seed=arguments.seed,
                selected_restart=selected_restart,
            ),
            overwrite=False,
        )
        saved += 1
        print(
            f"Saved {output.name} from restart {selected_restart + 1}: "
            f"error={1000.0 * selected_result.terms['position_error_m']:.1f}mm "
            f"speed={selected_result.terms['directional_speed_m_s']:.2f}m/s"
        )
    if failures:
        raise RuntimeError(
            "Some training anchors have no verified demonstration:\n- "
            + "\n- ".join(failures)
        )
    print(f"\nVerified {saved} target-conditioned demonstrations in {output_directory}")


if __name__ == "__main__":
    main()
