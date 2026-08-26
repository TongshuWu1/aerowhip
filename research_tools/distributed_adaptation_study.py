"""Causal EI/Cb adaptation study for the accelerated 11-node DDER--MPPI.

The default ``estimator`` phase is deterministic and never feeds an estimate
back into control.  It replays exact distributed plant observations one frame
at a time, applies the production event hierarchy, and evaluates nominal,
adapted, and oracle short-horizon predictions.  The optional ``control`` phase
uses accepted between-strike estimates in the unchanged receding-horizon MPPI
loop and compares fixed, adapted, and oracle controllers with paired seeds.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from drone_mpc.distributed_adaptation import (
    ADAPTATION_SCHEMA,
    DistributedAdaptationSettings,
    DistributedObservation,
    DistributedParameterFitter,
    OnlineAdaptationMonitor,
    ParameterEstimate,
    select_informative_segments,
    snapshot_for_estimate,
)
from drone_mpc.mppi import interpolate_control_knots
from drone_mpc.receding_mppi import RecedingMppiSettings, run_receding_horizon_mppi
from drone_mpc.reduced import reduce_cable_model
from drone_mpc.simulator import SimulationSettings, TensorRollout, WhipSimulator
from research_tools.profile_mppi_forward import DEFAULT_PROFILE, Workload, load_workload


SCHEMA = "distributed_event_triggered_adaptation_study_v1"
DEFAULT_OUTPUT = Path("data/drone_mpc/adaptation/distributed_adaptation_study.json")
TRUTH_RATIOS: tuple[tuple[float, float], ...] = (
    (1.0, 1.0),
    (0.8, 1.0),
    (1.2, 1.0),
    (1.0, 0.7),
    (1.0, 1.3),
    (0.8, 0.7),
    (1.2, 1.3),
    (0.8, 1.3),
    (1.2, 0.7),
)


def _serializable(value):
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _truth_model(workload: Workload, ei_ratio: float, cb_ratio: float):
    return reduce_cable_model(
        workload.controller,
        node_count=11,
        substeps=workload.controller.model.parameters.substeps,
        constraint_iterations=workload.controller.model.parameters.constraint_iterations,
        bending_stiffness_scale=ei_ratio,
        bending_damping_scale=cb_ratio,
    )


def _two_second_controls(
    workload: Workload, simulation: SimulationSettings, device: torch.device
) -> torch.Tensor:
    strike_steps = workload.simulation.control_count
    warm = torch.tensor(
        np.array(workload.warm_knots[None], copy=True),
        dtype=torch.float32,
        device=device,
    )
    strike = interpolate_control_knots(
        warm,
        strike_steps,
        simulation.maximum_acceleration_m_s2,
    )
    controls = torch.zeros(
        (1, simulation.control_count, 3), dtype=torch.float32, device=device
    )
    controls[:, :strike_steps] = strike
    return controls


def generate_truth_rollout(
    workload: Workload, ei_ratio: float, cb_ratio: float
) -> tuple[TensorRollout, WhipSimulator, SimulationSettings]:
    simulation = replace(workload.simulation, horizon_s=2.0)
    truth = _truth_model(workload, ei_ratio, cb_ratio)
    simulator = WhipSimulator(truth, simulation, device="cuda")
    state = simulator.initial_state(workload.initial_xyz)
    controls = _two_second_controls(workload, simulation, simulator.device)
    rollout = simulator.rollout(state, controls, create_graph=False)
    torch.cuda.synchronize()
    return rollout, simulator, simulation


def observations_from_rollout(
    rollout: TensorRollout,
    workload: Workload,
    estimate: ParameterEstimate,
) -> tuple[DistributedObservation, ...]:
    position = rollout.cable_positions_m[0].detach().cpu().numpy()
    velocity = rollout.cable_velocities_m_s[0].detach().cpu().numpy()
    attachment = rollout.attachment_positions_m[0].detach().cpu().numpy()
    drone_position = rollout.drone_positions_m[0].detach().cpu().numpy()
    drone_velocity = rollout.drone_velocities_m_s[0].detach().cpu().numpy()
    controls = rollout.accelerations_m_s2[0].detach().cpu().numpy()
    time_s = rollout.time_s.detach().cpu().numpy()
    target = np.asarray(workload.problem.target_position_m, dtype=np.float64)
    result = []
    for frame in range(len(time_s)):
        attachment_velocity = (
            np.zeros(3)
            if frame == 0
            else (attachment[frame] - attachment[frame - 1])
            / (time_s[frame] - time_s[frame - 1])
        )
        contact = bool(
            np.any(
                np.linalg.norm(position[frame] - target[None], axis=1)
                <= workload.problem.maximum_tip_error_m
            )
        )
        result.append(
            DistributedObservation(
                timestamp_s=float(time_s[frame]),
                attachment_position_m=attachment[frame],
                attachment_velocity_m_s=attachment_velocity,
                cable_positions_m=position[frame],
                cable_velocities_m_s=velocity[frame],
                drone_state=np.concatenate((drone_position[frame], drone_velocity[frame])),
                executed_action_m_s2=(
                    controls[min(frame, len(controls) - 1)]
                    if len(controls)
                    else np.zeros(3)
                ),
                active_estimate=estimate,
                contact=contact,
            )
        )
    return tuple(result)


def _with_estimate(
    observation: DistributedObservation, estimate: ParameterEstimate
) -> DistributedObservation:
    return replace(observation, active_estimate=estimate)


def _prediction_rows(
    fitter: DistributedParameterFitter,
    observations: tuple[DistributedObservation, ...],
    estimate: ParameterEstimate,
    truth_eta: np.ndarray,
    impact_direction_xyz: tuple[float, float, float],
) -> list[dict[str, float]]:
    rows = []
    impact_direction = np.asarray(impact_direction_xyz, dtype=np.float64)
    impact_direction /= np.linalg.norm(impact_direction)
    dt = observations[1].timestamp_s - observations[0].timestamp_s
    for horizon in (0.1, 0.2, 0.3):
        steps = int(round(horizon / dt))
        starts = [0, max(0, int(round(0.25 / dt))), max(0, int(round(0.45 / dt)))]
        for start in starts:
            end = start + steps
            if end >= len(observations):
                continue
            frames = observations[start : end + 1]
            if any(frame.contact or frame.safety_violation for frame in frames):
                continue
            from drone_mpc.distributed_adaptation import _segment_from_observations

            segment = _segment_from_observations(frames, 0.0, "prediction")
            hypotheses = np.stack((np.zeros(2), estimate.eta, truth_eta))
            positions, velocities, _elapsed = fitter.predictor.predict(
                (segment,), hypotheses
            )
            observed_position = segment.cable_positions_m
            observed_velocity = segment.cable_velocities_m_s
            observed_peak = int(
                np.argmax(np.linalg.norm(observed_velocity[1:, -1], axis=1)) + 1
            )
            for index, label in enumerate(("nominal", "adapted", "oracle")):
                position_error = positions[0, index, -1, 1:] - observed_position[-1, 1:]
                velocity_error = velocities[0, index, -1, 1:] - observed_velocity[-1, 1:]
                rows.append(
                    {
                        "horizon_s": horizon,
                        "start_time_s": segment.start_time_s,
                        "model": label,
                        "all_node_position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
                        "all_node_velocity_rmse_m_s": float(np.sqrt(np.mean(velocity_error**2))),
                        "tip_position_error_m": float(
                            np.linalg.norm(positions[0, index, -1, -1] - observed_position[-1, -1])
                        ),
                        "tip_velocity_error_m_s": float(
                            np.linalg.norm(velocities[0, index, -1, -1] - observed_velocity[-1, -1])
                        ),
                        "directed_tip_velocity_error_m_s": float(
                            abs(
                                np.dot(
                                    velocities[0, index, -1, -1]
                                    - observed_velocity[-1, -1],
                                    impact_direction,
                                )
                            )
                        ),
                        "propagation_timing_error_s": float(
                            abs(
                                segment.time_s[
                                    int(
                                        np.argmax(
                                            np.linalg.norm(
                                                velocities[0, index, 1:, -1], axis=1
                                            )
                                        )
                                        + 1
                                    )
                                ]
                                - segment.time_s[observed_peak]
                            )
                        ),
                    }
                )
    return rows


def causal_estimator_case(
    workload: Workload,
    settings: DistributedAdaptationSettings,
    ei_ratio: float,
    cb_ratio: float,
    *,
    strike_count: int = 3,
) -> dict[str, object]:
    rollout, _truth_simulator, _simulation = generate_truth_rollout(
        workload, ei_ratio, cb_ratio
    )
    nominal = ParameterEstimate()
    observations = observations_from_rollout(rollout, workload, nominal)
    fitter = DistributedParameterFitter(workload.controller, settings, device="cuda")
    current = nominal
    fit_results = []
    trace = []
    attempts = 0
    estimates_by_strike = []
    memory_bytes = 0
    for strike_index in range(strike_count):
        monitor = OnlineAdaptationMonitor(fitter.predictor, settings)
        for original in observations:
            observation = _with_estimate(original, current)
            diagnostic = monitor.append(observation)
            if diagnostic is None:
                continue
            trace.append(
                {
                    "strike": strike_index + 1,
                    "time_s": diagnostic.timestamp_s,
                    "global_time_s": strike_index * observations[-1].timestamp_s
                    + diagnostic.timestamp_s,
                    "instantaneous_error_m2": diagnostic.instantaneous_error_m2,
                    "ema_error_m2": diagnostic.ema_error_m2,
                    "excitation": asdict(diagnostic.excitation),
                    "persistent": diagnostic.mismatch_persistent,
                    "armed": diagnostic.hysteresis_armed,
                    "reason": diagnostic.reason,
                    "eta_e": current.eta_e,
                    "eta_c": current.eta_c,
                }
            )
            if diagnostic.reason != "fit_candidate":
                continue
            fit_segments, validation_segments = select_informative_segments(
                monitor.buffer.candidate_segments(), settings
            )
            if not fit_segments or not validation_segments:
                trace[-1]["reason"] = "waiting_for_segments"
                continue
            if not monitor.claim_trigger(diagnostic):
                continue
            attempts += 1
            result = fitter.fit(fit_segments, validation_segments, current)
            fit_results.append(result)
            if result.accepted:
                current = result.candidate_estimate
        memory_bytes = max(memory_bytes, monitor.buffer.approximate_memory_bytes)
        estimates_by_strike.append(
            {
                "strike": strike_index + 1,
                "ei_ratio": current.ei_ratio,
                "cb_ratio": current.cb_ratio,
                "generation": current.generation,
            }
        )
    truth_eta = np.log(np.asarray((ei_ratio, cb_ratio), dtype=np.float64))
    prediction_rows = _prediction_rows(
        fitter,
        observations,
        current,
        truth_eta,
        workload.problem.impact_direction,
    )
    return {
        "truth_ei_ratio": ei_ratio,
        "truth_cb_ratio": cb_ratio,
        "estimate_after_first_strike": estimates_by_strike[0],
        "estimates_by_strike": estimates_by_strike,
        "final_ei_ratio": current.ei_ratio,
        "final_cb_ratio": current.cb_ratio,
        "eta_error_l2": float(np.linalg.norm(current.eta - truth_eta)),
        "fit_attempts": attempts,
        "accepted_fits": sum(result.accepted for result in fit_results),
        "rejected_fits": sum(not result.accepted for result in fit_results),
        "buffer_memory_bytes": memory_bytes,
        "trace": trace,
        "fits": [asdict(result) for result in fit_results],
        "prediction_rows": prediction_rows,
    }


def policy_efficiency_case(
    workload: Workload,
    base_settings: DistributedAdaptationSettings,
    ei_ratio: float,
    cb_ratio: float,
    policy: str,
) -> dict[str, object]:
    """Compare naive continuous, error-only, and fully gated fit scheduling."""

    rollout, _truth_simulator, _simulation = generate_truth_rollout(
        workload, ei_ratio, cb_ratio
    )
    observations = observations_from_rollout(rollout, workload, ParameterEstimate())
    if policy == "full_event_triggered":
        settings = base_settings
    elif policy == "error_triggered":
        settings = replace(
            base_settings,
            minimum_excitation=1.0e-12,
            minimum_information_eigenvalue=1.0e-14,
            maximum_information_condition=1.0e14,
        )
    elif policy == "continuous":
        settings = replace(
            base_settings,
            minimum_excitation=1.0e-12,
            minimum_information_eigenvalue=1.0e-14,
            maximum_information_condition=1.0e14,
        )
    else:
        raise ValueError(policy)
    fitter = DistributedParameterFitter(workload.controller, settings, device="cuda")
    monitor = OnlineAdaptationMonitor(fitter.predictor, settings)
    current = ParameterEstimate()
    attempts = 0
    accepted = 0
    rejected = 0
    total_fit_s = 0.0
    short_rollouts = 0
    for original in observations:
        diagnostic = monitor.append(_with_estimate(original, current))
        if diagnostic is None:
            continue
        if policy == "continuous":
            should_attempt = True
        else:
            should_attempt = diagnostic.mismatch_persistent
        if not should_attempt:
            continue
        fit_segments, validation_segments = select_informative_segments(
            monitor.buffer.candidate_segments(), settings
        )
        if not fit_segments or not validation_segments:
            continue
        if policy != "continuous" and not monitor.claim_trigger(
            replace(diagnostic, reason="fit_candidate")
        ):
            continue
        attempts += 1
        result = fitter.fit(fit_segments, validation_segments, current)
        total_fit_s += result.timing.total_s
        short_rollouts += result.timing.short_rollouts
        if result.accepted:
            accepted += 1
            current = result.candidate_estimate
        else:
            rejected += 1
    truth_eta = np.log(np.asarray((ei_ratio, cb_ratio), dtype=np.float64))
    return {
        "truth_ei_ratio": ei_ratio,
        "truth_cb_ratio": cb_ratio,
        "policy": policy,
        "fit_attempts": attempts,
        "accepted_fits": accepted,
        "rejected_fits": rejected,
        "short_dder_rollouts": short_rollouts,
        "total_fit_wall_s": total_fit_s,
        "final_ei_ratio": current.ei_ratio,
        "final_cb_ratio": current.cb_ratio,
        "eta_error_l2": float(np.linalg.norm(current.eta - truth_eta)),
    }


def _control_row(execution, condition: str, baseline: str, seed: int) -> dict[str, object]:
    terms = execution.cost_terms
    return {
        "condition": condition,
        "baseline": baseline,
        "seed": seed,
        "valid_strike": execution.feasible,
        "terminal_reason": execution.terminal_reason,
        "target_error_m": float(terms["position_error_m"]),
        "directed_tip_speed_m_s": float(terms["directional_speed_m_s"]),
        "total_tip_speed_m_s": float(terms["tip_speed_m_s"]),
        "direction_error_deg": float(terms["direction_error_deg"]),
        "impact_time_s": execution.impact_time_s,
        "tip_first": bool(terms["geometric_tip_contact"])
        and float(terms["non_tip_contact_violation"]) == 0.0,
        "non_tip_clearance_m": float(terms["minimum_non_tip_target_distance_m"]),
        "maximum_drone_displacement_m": float(terms["maximum_drone_excursion_m"]),
        "maximum_drone_speed_m_s": float(terms["maximum_drone_speed_m_s"]),
        "safety_violation": float(terms["safety_cost"]) > 0.0,
        "planning_wall_time_s": execution.total_planning_wall_time_s,
    }


def control_comparison(
    workload: Workload,
    estimator_cases: list[dict[str, object]],
    seeds: tuple[int, ...],
    ratios: tuple[tuple[float, float], ...],
) -> list[dict[str, object]]:
    rows = []
    by_truth = {
        (float(case["truth_ei_ratio"]), float(case["truth_cb_ratio"])): case
        for case in estimator_cases
    }
    receding = RecedingMppiSettings(
        replan_interval_s=0.1, timeout_s=1.2, feedback_mode="full"
    )
    for ei_ratio, cb_ratio in ratios:
        truth = _truth_model(workload, ei_ratio, cb_ratio)
        case = by_truth[(ei_ratio, cb_ratio)]
        adapted_estimate = ParameterEstimate(
            eta_e=math.log(float(case["final_ei_ratio"])),
            eta_c=math.log(float(case["final_cb_ratio"])),
            generation=1,
            source="between_strike",
        )
        first = case["estimate_after_first_strike"]
        assert isinstance(first, dict)
        first_estimate = ParameterEstimate(
            eta_e=math.log(float(first["ei_ratio"])),
            eta_c=math.log(float(first["cb_ratio"])),
            generation=1,
            source="after_first_strike",
        )
        models = {
            "fixed_nominal": workload.controller,
            "adapted_after_strike_1": snapshot_for_estimate(
                workload.controller, first_estimate
            ),
            "adapted_after_strike_3": snapshot_for_estimate(
                workload.controller, adapted_estimate
            ),
            "oracle": truth,
        }
        condition = f"ei{ei_ratio:g}_cb{cb_ratio:g}"
        for seed in seeds:
            for label, model in models.items():
                planner = WhipSimulator(model, workload.simulation, device="cuda")
                plant = WhipSimulator(truth, workload.simulation, device="cuda")
                initial_state = plant.initial_state(workload.initial_xyz)
                execution = run_receding_horizon_mppi(
                    planner,
                    plant,
                    initial_state,
                    workload.problem,
                    replace(workload.mppi, seed=seed),
                    receding,
                    workload.warm_knots,
                )
                row = _control_row(execution, condition, label, seed)
                rows.append(row)
                print(
                    f"control {condition} {label} seed={seed}: "
                    f"valid={row['valid_strike']} "
                    f"error={1000.0 * float(row['target_error_m']):.1f}mm "
                    f"speed={float(row['directed_tip_speed_m_s']):.2f}m/s"
                )
    return rows


def _write_plots(payload: dict[str, object], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cases = payload.get("estimator_cases", [])
    if not isinstance(cases, list) or not cases:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    truth_e = [float(case["truth_ei_ratio"]) for case in cases]
    estimate_e = [float(case["final_ei_ratio"]) for case in cases]
    truth_c = [float(case["truth_cb_ratio"]) for case in cases]
    estimate_c = [float(case["final_cb_ratio"]) for case in cases]
    axes[0].scatter(truth_e, estimate_e, color="#1f77b4")
    axes[0].plot((0.7, 1.3), (0.7, 1.3), "k--", linewidth=1)
    axes[0].set(xlabel="true EI / EI0", ylabel="estimated EI / EI0", title="EI recovery")
    axes[1].scatter(truth_c, estimate_c, color="#d62728")
    axes[1].plot((0.6, 1.4), (0.6, 1.4), "k--", linewidth=1)
    axes[1].set(xlabel="true Cb / Cb0", ylabel="estimated Cb / Cb0", title="Cb recovery")
    figure.tight_layout()
    figure.savefig(output.with_name(output.stem + "_parameters.png"), dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--phase", choices=("estimator", "control", "all"), default="estimator"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(11, 17, 29))
    parser.add_argument(
        "--truth",
        nargs=2,
        type=float,
        metavar=("EI_RATIO", "CB_RATIO"),
        help="Run one truth pair instead of the complete matrix.",
    )
    parser.add_argument(
        "--policy-efficiency",
        action="store_true",
        help="Also compare continuous, error-only, and fully gated fitting.",
    )
    parser.add_argument(
        "--adaptation-ablations",
        action="store_true",
        help="Run EI-only, Cb-only, joint, and joint-without-information-gate replay.",
    )
    args = parser.parse_args()
    workload = load_workload(args.profile)
    settings = DistributedAdaptationSettings()
    ratios = (tuple(args.truth),) if args.truth else TRUTH_RATIOS
    started = time.perf_counter()
    estimator_cases = []
    if args.phase in {"estimator", "all", "control"}:
        for ei_ratio, cb_ratio in ratios:
            print(f"causal replay truth EI={ei_ratio:g} Cb={cb_ratio:g}")
            case = causal_estimator_case(
                workload, settings, float(ei_ratio), float(cb_ratio)
            )
            estimator_cases.append(case)
            print(
                f"  estimate EI={case['final_ei_ratio']:.4f} "
                f"Cb={case['final_cb_ratio']:.4f} "
                f"attempts={case['fit_attempts']} accepted={case['accepted_fits']}"
            )
    control_rows = []
    if args.phase in {"control", "all"}:
        control_rows = control_comparison(
            workload,
            estimator_cases,
            tuple(args.seeds),
            tuple((float(pair[0]), float(pair[1])) for pair in ratios),
        )
    policy_rows = []
    if args.policy_efficiency:
        for ei_ratio, cb_ratio in ratios:
            for policy in ("continuous", "error_triggered", "full_event_triggered"):
                row = policy_efficiency_case(
                    workload,
                    settings,
                    float(ei_ratio),
                    float(cb_ratio),
                    policy,
                )
                policy_rows.append(row)
                print(
                    f"policy EI={ei_ratio:g} Cb={cb_ratio:g} {policy}: "
                    f"fits={row['fit_attempts']} accepted={row['accepted_fits']} "
                    f"time={row['total_fit_wall_s']:.3f}s"
                )
    ablation_rows = []
    if args.adaptation_ablations:
        ablation_settings = {
            "ei_only": replace(settings, parameter_mode="ei_only"),
            "cb_only": replace(settings, parameter_mode="cb_only"),
            "joint": settings,
            "joint_without_information_gate": replace(
                settings,
                minimum_information_eigenvalue=1.0e-14,
                maximum_information_condition=1.0e14,
            ),
        }
        for ei_ratio, cb_ratio in ratios:
            for label, variant in ablation_settings.items():
                case = causal_estimator_case(
                    workload,
                    variant,
                    float(ei_ratio),
                    float(cb_ratio),
                    strike_count=3,
                )
                ablation_rows.append(
                    {
                        "truth_ei_ratio": float(ei_ratio),
                        "truth_cb_ratio": float(cb_ratio),
                        "ablation": label,
                        "final_ei_ratio": case["final_ei_ratio"],
                        "final_cb_ratio": case["final_cb_ratio"],
                        "eta_error_l2": case["eta_error_l2"],
                        "fit_attempts": case["fit_attempts"],
                        "accepted_fits": case["accepted_fits"],
                        "rejected_fits": case["rejected_fits"],
                    }
                )
                print(
                    f"ablation EI={ei_ratio:g} Cb={cb_ratio:g} {label}: "
                    f"estimate=({case['final_ei_ratio']:.3f},"
                    f"{case['final_cb_ratio']:.3f})"
                )
    payload = {
        "schema": SCHEMA,
        "adaptation_schema": ADAPTATION_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "profile": str(args.profile.resolve()),
        "nominal_ei_n_m2": workload.controller.bending_stiffness_n_m2,
        "nominal_cb_n_m2_s": workload.controller.bending_damping_n_m2_s,
        "settings": asdict(settings),
        "physics_unchanged": True,
        "controller_objective_unchanged": True,
        "exact_distributed_state": True,
        "estimator_cases": estimator_cases,
        "control_rows": control_rows,
        "policy_efficiency_rows": policy_rows,
        "adaptation_ablation_rows": ablation_rows,
        "elapsed_wall_s": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_serializable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_plots(payload, args.output)
    print(f"saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
