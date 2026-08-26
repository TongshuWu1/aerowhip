"""Compare reference and exact-accelerated 11-node closed-loop DDER--MPPI.

The two modes use the same saved controller profile, seed, warm start, cable
model, perturbation, and production receding-horizon loop.  ``reference``
disables every exact acceleration through its public environment switch;
``optimized`` enables them.  Run the modes in separate Python processes so
CUDA graph and NVRTC caches cannot cross-contaminate the comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from drone_mpc.mppi import MppiSettings
from drone_mpc.receding_mppi import (
    RecedingMppiSettings,
    perturb_cable_state,
    run_receding_horizon_mppi,
)
from drone_mpc.simulator import DroneCableState, WhipSimulator
from research_tools.profile_mppi_forward import DEFAULT_PROFILE, load_workload


SCHEMA = "dder_mppi_exact_acceleration_closed_loop_equivalence_v1"

PERTURBATIONS = {
    "nominal": (math.inf, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "sway"),
    "lateral_sway": (0.0, (0.0, 0.030, 0.0), (0.0, 0.0, 0.0), "sway"),
    "hidden_interior_velocity": (
        0.0,
        (0.0, 0.0, 0.0),
        (0.0, 0.40, 0.0),
        "interior",
    ),
    "before_reversal": (
        0.30,
        (0.0, 0.0, 0.0),
        (0.0, 0.40, 0.0),
        "interior",
    ),
    "after_reversal": (
        0.50,
        (0.0, 0.0, 0.0),
        (-0.40, 0.0, 0.0),
        "interior",
    ),
}

SWITCHES = (
    "CABLE_TWIN_FUSED_FIXED_PCG",
    "CABLE_TWIN_FUSED_FIXED_DAMPING",
    "CABLE_TWIN_FUSED_FIXED_PROJECTION",
    "CABLE_TWIN_FULL_HORIZON_GRAPH",
    "DRONE_MPPI_FUSED_COST",
)


def _configure(implementation: str) -> None:
    value = "0" if implementation == "reference" else "1"
    for name in SWITCHES:
        os.environ[name] = value


def _transform(condition: str):
    event_time, position, velocity, profile = PERTURBATIONS[condition]
    applied = False

    def transform(elapsed_s: float, state: DroneCableState) -> DroneCableState:
        nonlocal applied
        if applied or elapsed_s + 1.0e-9 < event_time:
            return state
        applied = True
        return perturb_cable_state(
            state,
            position_delta_m=position,
            velocity_delta_m_s=velocity,
            profile=profile,
        )

    return transform


def _result_row(execution, *, implementation: str, condition: str, feedback: str, seed: int):
    terms = execution.cost_terms
    planning = [update.planning_wall_time_s for update in execution.updates]
    return {
        "implementation": implementation,
        "condition": condition,
        "feedback": feedback,
        "seed": seed,
        "valid_strike": execution.feasible,
        "terminal_reason": execution.terminal_reason,
        "updates": len(execution.updates),
        "target_error_m": float(terms["position_error_m"]),
        "directed_tip_speed_m_s": float(terms["directional_speed_m_s"]),
        "total_tip_speed_m_s": float(terms["tip_speed_m_s"]),
        "direction_error_deg": float(terms["direction_error_deg"]),
        "impact_time_s": execution.impact_time_s,
        "tip_first": bool(terms["geometric_tip_contact"])
        and float(terms["non_tip_contact_violation"]) == 0.0,
        "non_tip_clearance_m": float(terms["minimum_non_tip_target_distance_m"]),
        "maximum_drone_speed_m_s": float(terms["maximum_drone_speed_m_s"]),
        "maximum_drone_displacement_m": float(terms["maximum_drone_excursion_m"]),
        "cost": execution.cost,
        "mean_update_wall_s": statistics.fmean(planning) if planning else 0.0,
        "median_update_wall_s": statistics.median(planning) if planning else 0.0,
        "maximum_update_wall_s": max(planning) if planning else 0.0,
        "total_planning_wall_s": execution.total_planning_wall_time_s,
        "total_rollouts": execution.total_rollouts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--implementation", choices=("reference", "optimized"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", choices=tuple(PERTURBATIONS), default=tuple(PERTURBATIONS))
    parser.add_argument("--feedback", nargs="+", choices=("full", "endpoint"), default=("full",))
    parser.add_argument("--seeds", nargs="+", type=int, default=(11, 17, 29))
    parser.add_argument("--timeout", type=float, default=1.2)
    parser.add_argument("--replan", type=float, default=0.1)
    args = parser.parse_args()

    _configure(args.implementation)
    workload = load_workload(args.profile)
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for condition in args.conditions:
        for feedback in args.feedback:
            for seed in args.seeds:
                planner = WhipSimulator(workload.controller, workload.simulation, device="cuda")
                plant = WhipSimulator(workload.truth, workload.simulation, device="cuda")
                mppi: MppiSettings = replace(workload.mppi, seed=seed)
                state = plant.initial_state(workload.initial_xyz)
                execution = run_receding_horizon_mppi(
                    planner,
                    plant,
                    state,
                    workload.problem,
                    mppi,
                    RecedingMppiSettings(
                        replan_interval_s=args.replan,
                        timeout_s=args.timeout,
                        feedback_mode=feedback,
                    ),
                    workload.warm_knots,
                    state_transform=_transform(condition),
                )
                row = _result_row(
                    execution,
                    implementation=args.implementation,
                    condition=condition,
                    feedback=feedback,
                    seed=seed,
                )
                rows.append(row)
                print(
                    f"{args.implementation} {condition} {feedback} seed={seed}: "
                    f"valid={row['valid_strike']} error={1000.0 * row['target_error_m']:.1f}mm "
                    f"speed={row['directed_tip_speed_m_s']:.2f}m/s "
                    f"plan={row['median_update_wall_s']:.3f}s"
                )

    payload = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "implementation": args.implementation,
        "profile": str(args.profile.resolve()),
        "switches": {name: os.environ[name] for name in SWITCHES},
        "node_count": workload.controller.node_count,
        "physics_unchanged": True,
        "mppi_objective_unchanged": True,
        "elapsed_wall_s": time.perf_counter() - started,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
