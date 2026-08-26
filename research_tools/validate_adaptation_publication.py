"""Validate atomic EI/Cb publication into captured DDER--MPPI runtimes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import torch

from drone_mpc.distributed_adaptation import (
    AtomicControllerRuntimeStore,
    ParameterEstimate,
)
from drone_mpc.mppi import interpolate_control_knots
from drone_mpc.reduced import reduce_cable_model
from drone_mpc.simulator import WhipSimulator
from research_tools.profile_mppi_forward import DEFAULT_PROFILE, load_workload


SCHEMA = "dder_adaptation_atomic_publication_validation_v1"
DEFAULT_OUTPUT = Path(
    "data/drone_mpc/adaptation/adaptation_publication_validation.json"
)


def _differences(first, second) -> dict[str, float]:
    return {
        "maximum_position_difference_m": float(
            torch.amax(
                torch.abs(first.cable_positions_m - second.cable_positions_m)
            ).detach().cpu()
        ),
        "rms_position_difference_m": float(
            torch.sqrt(
                torch.mean(
                    (first.cable_positions_m - second.cable_positions_m).square()
                )
            ).detach().cpu()
        ),
        "maximum_velocity_difference_m_s": float(
            torch.amax(
                torch.abs(first.cable_velocities_m_s - second.cable_velocities_m_s)
            ).detach().cpu()
        ),
        "rms_velocity_difference_m_s": float(
            torch.sqrt(
                torch.mean(
                    (first.cable_velocities_m_s - second.cable_velocities_m_s).square()
                )
            ).detach().cpu()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ei-ratio", type=float, default=0.8)
    parser.add_argument("--cb-ratio", type=float, default=0.7)
    parser.add_argument(
        "--prewarm-mppi",
        action="store_true",
        help="Also capture the full 2048-candidate graph before publication.",
    )
    args = parser.parse_args()
    workload = load_workload(args.profile)
    estimate = ParameterEstimate(
        eta_e=math.log(args.ei_ratio),
        eta_c=math.log(args.cb_ratio),
        generation=1,
        source="publication_validation",
    )
    store = AtomicControllerRuntimeStore(
        workload.controller, workload.simulation, device="cuda"
    )
    prewarm = [(1, workload.simulation.control_count)]
    if args.prewarm_mppi:
        prewarm.insert(0, (workload.mppi.samples, workload.simulation.control_count))
    prepared = store.prepare(estimate, prewarm_batches=prewarm)
    published = store.publish(prepared)
    active = store.snapshot()
    truth = reduce_cable_model(
        workload.controller,
        node_count=11,
        substeps=workload.controller.model.parameters.substeps,
        constraint_iterations=workload.controller.model.parameters.constraint_iterations,
        bending_stiffness_scale=args.ei_ratio,
        bending_damping_scale=args.cb_ratio,
    )
    truth_simulator = WhipSimulator(truth, workload.simulation, device="cuda")
    nominal_simulator = WhipSimulator(
        workload.controller, workload.simulation, device="cuda"
    )
    initial = truth_simulator.initial_state(workload.initial_xyz)
    knots = torch.tensor(
        np.array(workload.warm_knots[None], copy=True),
        dtype=torch.float32,
        device="cuda",
    )
    controls = interpolate_control_knots(
        knots,
        workload.simulation.control_count,
        workload.simulation.maximum_acceleration_m_s2,
    )
    truth_rollout = truth_simulator.rollout(initial, controls, create_graph=False)
    adapted_rollout = active.simulator.rollout(initial, controls, create_graph=False)
    nominal_rollout = nominal_simulator.rollout(initial, controls, create_graph=False)
    torch.cuda.synchronize()
    exact = _differences(adapted_rollout, truth_rollout)
    mismatch = _differences(nominal_rollout, truth_rollout)
    payload = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "profile": str(args.profile.resolve()),
        "estimate": asdict(estimate),
        "published": published,
        "active_generation": active.estimate.generation,
        "active_ei_n_m2": active.snapshot.bending_stiffness_n_m2,
        "active_cb_n_m2_s": active.snapshot.bending_damping_n_m2_s,
        "truth_ei_n_m2": truth.bending_stiffness_n_m2,
        "truth_cb_n_m2_s": truth.bending_damping_n_m2_s,
        "model_identity_matches_truth": active.snapshot.sha256 == truth.sha256,
        "captured_parameter_semantics": (
            "EI/Cb are DderRuntimeConstants tensors captured per WhipSimulator; "
            "an accepted estimate is applied by preparing and atomically swapping "
            "a complete immutable simulator outside an MPPI solve"
        ),
        "rebuild_and_prewarm_wall_s": prepared.rebuild_wall_time_s,
        "prewarm_batches": prewarm,
        "adapted_vs_truth": exact,
        "nominal_vs_truth": mismatch,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
