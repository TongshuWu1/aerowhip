"""Locate batch-shape divergence for the saved Milestone-4A whip command."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from planning.command_parameterization import acceleration_knots_to_fullstate
from planning.rollout import clone_state_batch, hover_preroll, run_population_rollout
from planning.task import load_canonical_whip_task
from simulator.parameters import SimulatorSettings
from simulator.production import (
    active_model_paths,
    build_production_simulator,
    load_active_model_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOTS = (
    ROOT
    / "data"
    / "planning_results"
    / "canonical_whip_v1"
    / "2026-08-29T041300.459080Z"
    / "best_acceleration_knots.json"
)


def _command(knots: torch.Tensor, task, simulator):
    return acceleration_knots_to_fullstate(
        knots,
        initial_position_m=torch.tensor(
            task.initial_uav_position_m,
            device=simulator.device,
            dtype=simulator.dtype,
        ),
        initial_velocity_m_s=torch.zeros(
            3, device=simulator.device, dtype=simulator.dtype
        ),
        yaw_rad=task.initial_yaw_rad,
        horizon_s=task.mppi.horizon_s,
        dt_s=simulator.dt_s,
    ).simulator_sequence()


def _row_snapshot(state) -> dict[str, torch.Tensor]:
    return {
        "uav_position": state.uav.position_m[0].detach().cpu().clone(),
        "uav_velocity": state.uav.velocity_m_s[0].detach().cpu().clone(),
        "uav_orientation": state.uav.orientation_xyzw[0].detach().cpu().clone(),
        "uav_angular_velocity": (
            state.uav.angular_velocity_world_rad_s[0].detach().cpu().clone()
        ),
        "residual_output": (
            torch.zeros(3)
            if state.uav.residual_acceleration_m_s2 is None
            else state.uav.residual_acceleration_m_s2[0].detach().cpu().clone()
        ),
        "residual_fifo": (
            torch.empty(0)
            if state.uav.residual_history is None
            else state.uav.residual_history.features[0].detach().cpu().clone()
        ),
        "cable_position": state.cable.positions_m[0].detach().cpu().clone(),
        "cable_velocity": state.cable.velocities_m_s[0].detach().cpu().clone(),
    }


def _identical_row_spread(state) -> dict[str, float]:
    def spread(value: torch.Tensor | None) -> float:
        if value is None or value.shape[0] == 1:
            return 0.0
        return float(torch.max(torch.abs(value - value[0:1])).detach().cpu())

    return {
        "uav_position": spread(state.uav.position_m),
        "uav_velocity": spread(state.uav.velocity_m_s),
        "uav_orientation": spread(state.uav.orientation_xyzw),
        "uav_angular_velocity": spread(state.uav.angular_velocity_world_rad_s),
        "residual_output": spread(state.uav.residual_acceleration_m_s2),
        "residual_fifo": spread(
            None
            if state.uav.residual_history is None
            else state.uav.residual_history.features
        ),
        "cable_position": spread(state.cable.positions_m),
        "cable_velocity": spread(state.cable.velocities_m_s),
    }


def _trace(simulator, initial_state, task, knots: torch.Tensor, batch: int):
    batch_knots = knots[None].repeat(batch, 1, 1)
    commands = _command(batch_knots, task, simulator)
    state = clone_state_batch(initial_state, batch)
    snapshots = [_row_snapshot(state)]
    spreads = [_identical_row_spread(state)]
    with torch.no_grad():
        for index in range(commands.step_count):
            state = simulator._propagate(  # noqa: SLF001
                state,
                commands.command_at(index),
                simulator.parameters,
                create_graph=False,
            )
            snapshots.append(_row_snapshot(state))
            spreads.append(_identical_row_spread(state))
    torch.cuda.synchronize(simulator.device)
    command_row = {
        name: value[:, 0].detach().cpu()
        for name, value in {
            "position": commands.positions_m,
            "velocity": commands.velocities_m_s,
            "acceleration": commands.accelerations_m_s2,
            "orientation": commands.orientations_xyzw,
            "angular_velocity": commands.angular_velocities_body_rad_s,
        }.items()
    }
    return snapshots, spreads, command_row


def _batch_one_boundaries(simulator, initial_state, task, knots: torch.Tensor):
    commands = _command(knots[None], task, simulator)
    uav = clone_state_batch(initial_state, 1).uav
    boundaries = []
    with torch.no_grad():
        for index in range(commands.step_count):
            uav = simulator.uav_model.step(
                uav,
                commands.command_at(index),
                simulator.dt_s,
                simulator.parameters.uav,
            )
            boundaries.append(
                simulator.root_boundary.evaluate(
                    uav, simulator.cable_configuration.rest_lengths_m[0]
                ).prescribed_positions_m.detach().clone()
            )
    return boundaries


def _cable_trace_with_boundaries(
    simulator,
    initial_state,
    boundaries,
    batch: int,
):
    cable = clone_state_batch(initial_state, batch).cable
    base = simulator.cable_model.runtime_constants(cable.positions_m)
    EI, Cb = simulator.parameters.cable.tensors(cable.positions_m)
    constants = replace(
        base,
        bending_stiffness_n_m2=EI.reshape(1).expand(batch),
        bending_damping_n_m2_s=Cb.reshape(1).expand(batch),
    )
    dt = torch.full(
        (batch,), simulator.dt_s, device=simulator.device, dtype=simulator.dtype
    )
    snapshots = [
        {
            "cable_position": cable.positions_m[0].detach().cpu().clone(),
            "cable_velocity": cable.velocities_m_s[0].detach().cpu().clone(),
        }
    ]
    spreads = []
    with torch.no_grad():
        for boundary_one in boundaries:
            boundary = boundary_one.repeat(batch, 1, 1)
            cable = simulator.cable_model.step_runtime(
                cable,
                boundary,
                dt,
                constants,
                iterative_damping=True,
                damping_backend="pcg32_experimental",
                pinned_endpoints=simulator.root_boundary.pinned_endpoints,
                create_graph=False,
                functional_force_autograd=simulator.functional_force_autograd,
            )
            snapshots.append(
                {
                    "cable_position": cable.positions_m[0].detach().cpu().clone(),
                    "cable_velocity": cable.velocities_m_s[0].detach().cpu().clone(),
                }
            )
            spreads.append(
                {
                    "cable_position": float(
                        torch.max(
                            torch.abs(cable.positions_m - cable.positions_m[0:1])
                        ).detach().cpu()
                    ),
                    "cable_velocity": float(
                        torch.max(
                            torch.abs(cable.velocities_m_s - cable.velocities_m_s[0:1])
                        ).detach().cpu()
                    ),
                }
            )
    return snapshots, spreads


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--knots", type=Path, default=DEFAULT_KNOTS)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 2048])
    args = parser.parse_args()

    active = load_active_model_manifest()
    settings = SimulatorSettings.load(active_model_paths(active)["configuration"])
    simulator = build_production_simulator(settings, device="cuda", dtype=torch.float32)
    task = load_canonical_whip_task()
    task.validate_for_dt(settings.dt_s)
    initial_state = hover_preroll(simulator, task)
    simulator.uav_model.set_fixed_evaluation_batch_size(max(args.batches))
    payload = json.loads(args.knots.read_text(encoding="utf-8"))
    knots = torch.tensor(payload["values"], device=simulator.device, dtype=simulator.dtype)

    traces = {}
    spreads = {}
    commands = {}
    rollout_metrics = {}
    for batch in args.batches:
        traces[batch], spreads[batch], commands[batch] = _trace(
            simulator, initial_state, task, knots, batch
        )
        rollout_metrics[batch] = run_population_rollout(
            simulator,
            initial_state,
            knots[None].repeat(batch, 1, 1),
            task,
        ).row(0)

    boundaries = _batch_one_boundaries(simulator, initial_state, task, knots)
    cable_only = {}
    cable_only_spreads = {}
    for batch in args.batches:
        cable_only[batch], cable_only_spreads[batch] = _cable_trace_with_boundaries(
            simulator, initial_state, boundaries, batch
        )

    reference = args.batches[0]
    comparisons: dict[str, object] = {}
    for batch in args.batches[1:]:
        rows = []
        first = {}
        for step, (left, right) in enumerate(zip(traces[reference], traces[batch])):
            differences = {
                name: float(torch.max(torch.abs(left[name] - right[name])))
                for name in left
            }
            for name, value in differences.items():
                if value > 0.0 and name not in first:
                    first[name] = {
                        "step": step,
                        "time_s": step * settings.dt_s,
                        "absolute_difference": value,
                    }
            rows.append({"step": step, "time_s": step * settings.dt_s, **differences})
        command_difference = {
            name: float(torch.max(torch.abs(commands[reference][name] - commands[batch][name])))
            for name in commands[reference]
        }
        reference_metrics = rollout_metrics[reference]
        batch_metrics = rollout_metrics[batch]
        final_uav_difference = max(
            abs(float(left) - float(right))
            for left, right in zip(
                reference_metrics["final_uav_position_m"],
                batch_metrics["final_uav_position_m"],
            )
        )
        final_c10_difference = max(
            abs(float(left) - float(right))
            for left, right in zip(
                reference_metrics["final_c10_position_m"],
                batch_metrics["final_c10_position_m"],
            )
        )
        minimum_distance_difference = abs(
            float(reference_metrics["minimum_tip_target_distance_m"])
            - float(batch_metrics["minimum_tip_target_distance_m"])
        )
        classification_equal = (
            reference_metrics["success"] == batch_metrics["success"]
            and reference_metrics["first_entry_marker"]
            == batch_metrics["first_entry_marker"]
        )
        comparisons[str(batch)] = {
            "first_nonzero_difference": first,
            "per_step": rows,
            "maximum_identical_row_spread": {
                name: max(item[name] for item in spreads[batch])
                for name in spreads[batch][0]
            },
            "command_row_difference": command_difference,
            "gate_metrics": {
                "final_uav_position_difference_m": final_uav_difference,
                "final_c10_position_difference_m": final_c10_difference,
                "minimum_tip_target_distance_difference_m": (
                    minimum_distance_difference
                ),
                "success_and_first_entry_classification_identical": (
                    classification_equal
                ),
                "pass": bool(
                    final_uav_difference <= 0.001
                    and final_c10_difference <= 0.002
                    and minimum_distance_difference <= 0.002
                    and classification_equal
                ),
            },
        }

    cable_only_comparisons = {}
    for batch in args.batches[1:]:
        rows = []
        for step, (left, right) in enumerate(
            zip(cable_only[reference], cable_only[batch])
        ):
            rows.append(
                {
                    "step": step,
                    "time_s": step * settings.dt_s,
                    "cable_position": float(
                        torch.max(
                            torch.abs(
                                left["cable_position"] - right["cable_position"]
                            )
                        )
                    ),
                    "cable_velocity": float(
                        torch.max(
                            torch.abs(
                                left["cable_velocity"] - right["cable_velocity"]
                            )
                        )
                    ),
                }
            )
        cable_only_comparisons[str(batch)] = {
            "per_step": rows,
            "maximum_identical_row_spread": {
                name: max(item[name] for item in cable_only_spreads[batch])
                for name in cable_only_spreads[batch][0]
            },
        }

    overall_gate = all(
        bool(comparisons[str(batch)]["gate_metrics"]["pass"])
        for batch in args.batches[1:]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema": "mppi_batch_consistency_diagnostic_v1",
                "saved_knots": str(args.knots.resolve()),
                "batches": args.batches,
                "dt_s": settings.dt_s,
                "step_count": len(traces[reference]) - 1,
                "fixed_uav_evaluation_batch_size": max(args.batches),
                "acceptance_gate": {
                    "final_uav_position_tolerance_m": 0.001,
                    "final_c10_position_tolerance_m": 0.002,
                    "minimum_tip_target_distance_tolerance_m": 0.002,
                    "classification_must_match": True,
                    "pass": overall_gate,
                },
                "comparisons_against_batch_one": comparisons,
                "cable_only_with_identical_batch_one_boundaries": (
                    cable_only_comparisons
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
