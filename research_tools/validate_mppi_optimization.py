"""Validate an accelerated DDER--MPPI runtime against the frozen oracle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from cable_twin.shared.dder import DderState
import drone_mpc.mppi as mppi_module
from drone_mpc.mppi import interpolate_control_knots, optimize_mppi
from drone_mpc.receding_mppi import perturb_cable_state
from drone_mpc.simulator import DroneCableState, WhipSimulator
from research_tools.freeze_mppi_reference import _step_components
from research_tools.profile_mppi_forward import DEFAULT_PROFILE, load_workload


DEFAULT_REFERENCE = Path("data/drone_mpc/optimization/reference")


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    difference = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(difference * difference)))


def _relative(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.max(np.abs(np.asarray(b, dtype=np.float64))))
    return _max_abs(a, b) / max(denominator, np.finfo(np.float64).tiny)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        result = np.empty_like(order, dtype=np.float64)
        result[order] = np.arange(len(values), dtype=np.float64)
        return result

    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def _metadata(archive: np.lib.npyio.NpzFile) -> dict[str, object]:
    return json.loads(str(archive["metadata_json"]))


def validate(profile_path: Path, reference: Path, output: Path) -> Path:
    workload = load_workload(profile_path)
    simulator = WhipSimulator(workload.controller, workload.simulation, device="cuda")
    initial = simulator.initial_state(workload.initial_xyz)
    count = simulator.settings.control_count
    maximum = simulator.settings.maximum_acceleration_m_s2
    report: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference": reference.resolve().as_posix(),
        "step_cases": {},
        "rollout_cases": {},
    }

    for path in sorted(reference.glob("step__*.npz")):
        with np.load(path) as old:
            state = DroneCableState(
                torch.tensor(old["input_drone_position_m"], device="cuda"),
                torch.tensor(old["input_drone_velocity_m_s"], device="cuda"),
                DderState(
                    torch.tensor(old["input_positions_m"], device="cuda"),
                    torch.tensor(old["input_velocities_m_s"], device="cuda"),
                ),
            )
            next_drone = torch.tensor(old["next_drone_position_m"], device="cuda")
            _undamped, damping, state_out = _step_components(
                simulator, state, next_drone
            )
            torch.cuda.synchronize()
            new_q = state_out.positions_m.detach().cpu().numpy()
            new_v = state_out.velocities_m_s.detach().cpu().numpy()
            new_damping = damping.detach().cpu().numpy()
            q_ref = old["output_positions_m"]
            v_ref = old["output_velocities_m_s"]
            damping_ref = old["damping_solution_velocity_m_s"]
            case = {
                "position_max_abs_m": _max_abs(new_q, q_ref),
                "position_relative": _relative(new_q, q_ref),
                "velocity_max_abs_m_s": _max_abs(new_v, v_ref),
                "velocity_relative": _relative(new_v, v_ref),
                "damping_max_abs_m_s": _max_abs(new_damping, damping_ref),
                "damping_relative": _relative(new_damping, damping_ref),
                "attachment_position_max_abs_m": _max_abs(
                    new_q[:, 0], q_ref[:, 0]
                ),
                "attachment_velocity_max_abs_m_s": _max_abs(
                    new_v[:, 0], v_ref[:, 0]
                ),
            }
            if "edge_length_residual_m" in old:
                rest = simulator.snapshot.model.runtime_constants(
                    state.cable.positions_m
                ).rest_lengths_m
                new_edges = state_out.positions_m[:, 1:] - state_out.positions_m[:, :-1]
                new_residual = (
                    torch.linalg.vector_norm(new_edges, dim=2) - rest[None]
                ).detach().cpu().numpy()
                reference_residual = old["edge_length_residual_m"]
                case.update(
                    {
                        "edge_length_residual_max_abs_m": float(
                            np.max(np.abs(new_residual))
                        ),
                        "edge_length_residual_reference_max_abs_m": float(
                            np.max(np.abs(reference_residual))
                        ),
                        "edge_length_residual_difference_max_abs_m": _max_abs(
                            new_residual, reference_residual
                        ),
                    }
                )
            report["step_cases"][path.stem] = case

    for path in sorted(reference.glob("rollout__*.npz")):
        with np.load(path) as old:
            knots = torch.tensor(old["control_knots_m_s2"], device="cuda")
            controls = interpolate_control_knots(knots.reshape(1, -1, 3), count, maximum)
            rollout_initial = initial
            if path.stem.endswith("hidden_interior_velocity"):
                rollout_initial = perturb_cable_state(
                    initial,
                    profile="interior",
                    velocity_delta_m_s=(0.0, 0.40, 0.0),
                )
            rollout = simulator.rollout(rollout_initial, controls, create_graph=False)
            costs, diagnostics = mppi_module._mppi_event_objective(
                rollout, initial, workload.problem, simulator, workload.mppi
            )
            torch.cuda.synchronize()
            q = rollout.cable_positions_m.detach().cpu().numpy()
            v = rollout.cable_velocities_m_s.detach().cpu().numpy()
            q_ref = old["cable_positions_m"]
            v_ref = old["cable_velocities_m_s"]
            old_meta = _metadata(old)
            case: dict[str, object] = {
                "position_max_abs_m": _max_abs(q, q_ref),
                "position_rms_m": _rms(q, q_ref),
                "velocity_max_abs_m_s": _max_abs(v, v_ref),
                "velocity_rms_m_s": _rms(v, v_ref),
                "cost_abs": abs(float(costs[0].cpu()) - float(old["cost"][0])),
            }
            for key in (
                "feasible",
                "impact_frame",
                "impact_time_s",
                "position_error_m",
                "directional_speed_m_s",
                "tip_speed_m_s",
                "direction_error_deg",
                "minimum_non_tip_target_distance_m",
                "non_tip_contact_violation",
            ):
                ref_key = f"diagnostic__{key}"
                if ref_key in old:
                    new_value = diagnostics[key].detach().cpu().numpy()
                    old_value = old[ref_key]
                    if new_value.dtype == np.bool_:
                        case[f"{key}_agreement"] = bool(np.array_equal(new_value, old_value))
                    else:
                        case[f"{key}_max_abs"] = _max_abs(new_value, old_value)
            case["reference_metadata"] = old_meta
            report["rollout_cases"][path.stem] = case

    with np.load(reference / "mppi_iteration_seeded.npz") as old:
        candidates = torch.tensor(old["candidate_knots_m_s2"], device="cuda")
        controls = interpolate_control_knots(candidates, count, maximum)
        rollout = simulator.rollout(initial, controls, create_graph=False)
        costs, diagnostics = mppi_module._mppi_event_objective(
            rollout, initial, workload.problem, simulator, workload.mppi
        )
        minimum = torch.min(costs)
        importance = torch.exp(-(costs - minimum) / workload.mppi.temperature)
        weights = importance / torch.clamp(torch.sum(importance), min=1.0e-30)
        nominal = torch.tensor(old["nominal_knots_m_s2"], device="cuda")
        weighted = mppi_module._bound_vectors(
            nominal
            + torch.sum(weights[:, None, None] * (candidates - nominal[None]), dim=0),
            maximum,
        )
        torch.cuda.synchronize()
        new_costs = costs.detach().cpu().numpy()
        ref_costs = old["candidate_costs"]
        new_order = np.argsort(new_costs)
        ref_order = np.argsort(ref_costs)
        new_feasible = diagnostics["feasible"].detach().cpu().numpy()
        ref_feasible = old["diagnostic__feasible"]
        new_non_tip = (
            diagnostics["non_tip_contact_violation"].detach().cpu().numpy() > 0
        )
        ref_non_tip = old["diagnostic__non_tip_contact_violation"] > 0
        report["candidate_ranking"] = {
            "cost_max_abs": _max_abs(new_costs, ref_costs),
            "cost_rms": _rms(new_costs, ref_costs),
            "pearson": float(np.corrcoef(new_costs, ref_costs)[0, 1]),
            "spearman": _spearman(new_costs, ref_costs),
            "top1_agreement": bool(new_order[0] == ref_order[0]),
            "top5_overlap": int(len(set(new_order[:5]) & set(ref_order[:5]))),
            "top10_overlap": int(len(set(new_order[:10]) & set(ref_order[:10]))),
            "valid_classification_agreement": float(np.mean(new_feasible == ref_feasible)),
            "tip_first_classification_agreement": float(np.mean(new_non_tip == ref_non_tip)),
            "importance_weight_max_abs": _max_abs(
                weights.detach().cpu().numpy(), old["importance_weights"]
            ),
            "weighted_nominal_max_abs_m_s2": _max_abs(
                weighted.detach().cpu().numpy(), old["weighted_nominal_knots_m_s2"]
            ),
        }

    # Verify the retained winner is exactly the trajectory obtained by an
    # explicit replay of the selected knots.
    plan = optimize_mppi(
        simulator,
        initial,
        workload.problem,
        workload.mppi,
        warm_start_knots_m_s2=workload.warm_knots,
    )
    explicit_controls = interpolate_control_knots(
        torch.tensor(plan.control_knots_m_s2, device="cuda")[None], count, maximum
    )
    explicit = simulator.rollout(initial, explicit_controls, create_graph=False)
    torch.cuda.synchronize()
    report["retained_winner_vs_explicit_replay"] = {
        "cable_position_max_abs_m": _max_abs(
            plan.prediction.cable_positions_m,
            explicit.cable_positions_m[0].detach().cpu().numpy(),
        ),
        "cable_velocity_max_abs_m_s": _max_abs(
            plan.prediction.cable_velocities_m_s,
            explicit.cable_velocities_m_s[0].detach().cpu().numpy(),
        ),
        "drone_position_max_abs_m": _max_abs(
            plan.prediction.drone_positions_m,
            explicit.drone_positions_m[0].detach().cpu().numpy(),
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/drone_mpc/optimization/latest_validation.json"),
    )
    arguments = parser.parse_args()
    result = validate(arguments.profile, arguments.reference, arguments.output)
    print(result.resolve())


if __name__ == "__main__":
    main()
