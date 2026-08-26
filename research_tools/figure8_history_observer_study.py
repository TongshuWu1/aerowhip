"""Controlled full/endpoint/history-observer Figure-8 comparison.

This study keeps EI/Cb and all DDER/MPPI settings matched.  The history
observer receives only attachment motion and free-tip position history; full
plant state is used exclusively by the evaluator and full-state baseline.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from cable_twin.shared.dder import DderState, START_PINNED_FREE_END
from drone_mpc.figure8_tracking import (
    Figure8Execution,
    Figure8ExecutionSettings,
    Figure8Reference,
    TrackingCostSettings,
    run_figure8_tracking,
)
from drone_mpc.history_observer import (
    DderHistoryObserver,
    EndpointHistoryObservation,
    HistoryObserverSettings,
)
from drone_mpc.model import CableModelSnapshot, load_cable_model
from drone_mpc.mppi import MppiSettings
from drone_mpc.receding_mppi import perturb_cable_state
from drone_mpc.reduced import build_controller_and_truth_models
from drone_mpc.simulator import SimulationSettings, WhipSimulator
from optitrack_offline.config import DEFAULT_MODEL_PATH


OUTPUT = PROJECT / "reports" / "figure8_history_observer_data"


@dataclass(frozen=True, slots=True)
class StudySettings:
    duration_s: float = 2.0
    seeds: tuple[int, ...] = (17, 23, 41)
    hidden_velocity_m_s: float = 0.45
    prediction_horizons_s: tuple[float, ...] = (0.10, 0.20, 0.30)


def _initial_observation(
    timestamp_s: float,
    state: DderState,
) -> EndpointHistoryObservation:
    position = state.positions_m[0].detach().cpu().numpy()
    velocity = state.velocities_m_s[0].detach().cpu().numpy()
    return EndpointHistoryObservation(
        timestamp_s,
        position[0],
        velocity[0],
        position[-1],
    )


def _prediction_errors(
    execution: Figure8Execution,
    model: CableModelSnapshot,
    dt_s: float,
    horizons_s: tuple[float, ...],
    device: torch.device,
) -> dict[str, float]:
    results: dict[str, float] = {}
    dtype = torch.float32 if device.type == "cuda" else torch.float64
    minimum_start = int(round(0.30 / dt_s))
    stride = max(1, int(round(0.10 / dt_s)))
    for horizon_s in horizons_s:
        step_count = int(round(horizon_s / dt_s))
        starts = np.arange(
            minimum_start,
            len(execution.time_s) - step_count,
            stride,
            dtype=np.int64,
        )
        if len(starts) == 0:
            results[f"estimated_tip_prediction_rmse_{horizon_s:.2f}s_m"] = math.nan
            results[f"true_tip_prediction_rmse_{horizon_s:.2f}s_m"] = math.nan
            continue
        estimated_q = execution.estimated_cable_positions_m[starts]
        estimated_v = execution.estimated_cable_velocities_m_s[starts]
        true_q = execution.cable_positions_m[starts]
        true_v = execution.cable_velocities_m_s[starts]
        state = DderState(
            torch.as_tensor(
                np.concatenate((estimated_q, true_q)), dtype=dtype, device=device
            ),
            torch.as_tensor(
                np.concatenate((estimated_v, true_v)), dtype=dtype, device=device
            ),
        )
        constants = model.model.runtime_constants(state.positions_m)
        dt = torch.full((2 * len(starts),), dt_s, dtype=dtype, device=device)
        for offset in range(1, step_count + 1):
            boundary_single = execution.cable_positions_m[starts + offset, 0]
            boundary = torch.as_tensor(
                np.concatenate((boundary_single, boundary_single))[:, None],
                dtype=dtype,
                device=device,
            )
            state = model.model.step_runtime(
                state,
                boundary,
                dt,
                constants,
                iterative_damping=True,
                pinned_endpoints=START_PINNED_FREE_END,
            )
        predicted = state.positions_m[:, -1].detach().cpu().numpy()
        actual = execution.actual_tip_positions_m[starts + step_count]
        estimated_error = predicted[: len(starts)] - actual
        true_error = predicted[len(starts) :] - actual
        results[f"estimated_tip_prediction_rmse_{horizon_s:.2f}s_m"] = float(
            np.sqrt(np.mean(np.square(estimated_error)))
        )
        results[f"true_tip_prediction_rmse_{horizon_s:.2f}s_m"] = float(
            np.sqrt(np.mean(np.square(true_error)))
        )
    return results


def _basis_representability(
    execution: Figure8Execution,
    settings: HistoryObserverSettings,
) -> tuple[float, float]:
    node_count = execution.cable_positions_m.shape[1]
    coordinate = np.linspace(0.0, 1.0, node_count)
    basis = [coordinate]
    basis.extend(
        np.sin(index * math.pi * coordinate)
        for index in range(1, settings.spatial_mode_count)
    )
    matrix = np.stack(basis, axis=1)
    matrix[0] = 0.0
    position_error = (
        execution.cable_positions_m - execution.estimated_cable_positions_m
    )
    velocity_error = (
        execution.cable_velocities_m_s - execution.estimated_cable_velocities_m_s
    )

    def relative_residual(values: np.ndarray) -> float:
        ratios = []
        for frame in values:
            coefficients = np.linalg.lstsq(matrix, frame, rcond=None)[0]
            residual = frame - matrix @ coefficients
            denominator = np.linalg.norm(frame)
            if denominator > 1.0e-12:
                ratios.append(float(np.linalg.norm(residual) / denominator))
        return float(np.median(ratios)) if ratios else 0.0

    return relative_residual(position_error), relative_residual(velocity_error)


def _propagation_only_metrics(
    execution: Figure8Execution,
    model: CableModelSnapshot,
) -> dict[str, float]:
    dtype = torch.float32
    initial = DderState(
        torch.tensor(
            execution.estimated_cable_positions_m[0:1],
            dtype=dtype,
            device="cuda",
        ),
        torch.tensor(
            execution.estimated_cable_velocities_m_s[0:1],
            dtype=dtype,
            device="cuda",
        ),
    )
    observer = DderHistoryObserver(
        model,
        initial,
        EndpointHistoryObservation(
            float(execution.time_s[0]),
            execution.cable_positions_m[0, 0],
            execution.cable_velocities_m_s[0, 0],
            execution.actual_tip_positions_m[0],
        ),
        HistoryObserverSettings(
            history_duration_s=max(float(execution.time_s[-1]) + 1.0, 1.0),
            maximum_iterations=1,
        ),
        device="cuda",
    )
    positions = [observer.current_state.positions_m[0].detach().cpu().numpy()]
    velocities = [observer.current_state.velocities_m_s[0].detach().cpu().numpy()]
    for index in range(1, len(execution.time_s)):
        state = observer.ingest(
            EndpointHistoryObservation(
                float(execution.time_s[index]),
                execution.cable_positions_m[index, 0],
                execution.cable_velocities_m_s[index, 0],
                execution.actual_tip_positions_m[index],
            )
        )
        positions.append(state.positions_m[0].detach().cpu().numpy())
        velocities.append(state.velocities_m_s[0].detach().cpu().numpy())
    position = np.asarray(positions)
    velocity = np.asarray(velocities)
    return {
        "propagation_only_position_rmse_m": float(
            np.sqrt(
                np.mean(
                    np.square(position[:, 1:] - execution.cable_positions_m[:, 1:])
                )
            )
        ),
        "propagation_only_velocity_rmse_m_s": float(
            np.sqrt(
                np.mean(
                    np.square(
                        velocity[:, 1:] - execution.cable_velocities_m_s[:, 1:]
                    )
                )
            )
        ),
        "propagation_only_tip_history_rmse_m": float(
            np.sqrt(
                np.mean(np.square(position[:, -1] - execution.actual_tip_positions_m))
            )
        ),
    }


def _metrics(
    condition: str,
    mode: str,
    seed: int,
    execution: Figure8Execution,
    observer: DderHistoryObserver | None,
    model: CableModelSnapshot,
    simulation: SimulationSettings,
    observer_settings: HistoryObserverSettings,
) -> dict[str, object]:
    dynamic_position_error = (
        execution.estimated_cable_positions_m[:, 1:]
        - execution.cable_positions_m[:, 1:]
    )
    dynamic_velocity_error = (
        execution.estimated_cable_velocities_m_s[:, 1:]
        - execution.cable_velocities_m_s[:, 1:]
    )
    row: dict[str, object] = {
        "condition": condition,
        "mode": mode,
        "seed": seed,
        **execution.summary,
        "distributed_position_rmse_m": float(
            np.sqrt(np.mean(np.square(dynamic_position_error)))
        ),
        "distributed_velocity_rmse_m_s": float(
            np.sqrt(np.mean(np.square(dynamic_velocity_error)))
        ),
        "observer_mean_update_time_s": float(
            np.mean(execution.observer_update_times_s)
        ),
        "observer_p95_update_time_s": float(
            np.percentile(execution.observer_update_times_s, 95.0)
        ),
        "observer_accepted_updates": int(
            np.sum(execution.observer_update_accepted)
        ),
        "drone_maximum_speed_m_s": float(
            np.max(np.linalg.norm(execution.drone_velocities_m_s, axis=1))
        ),
        "drone_maximum_acceleration_m_s2": float(
            np.max(np.linalg.norm(execution.commands_m_s2, axis=1))
        ),
    }
    if observer is not None:
        ready = [update for update in observer.updates if update.ready]
        row.update(
            {
                "observer_ready_updates": len(ready),
                "observer_history_rmse_before_m": float(
                    np.nanmean([update.measurement_rmse_before_m for update in ready])
                )
                if ready
                else math.nan,
                "observer_history_rmse_after_m": float(
                    np.nanmean([update.measurement_rmse_after_m for update in ready])
                )
                if ready
                else math.nan,
                "observer_median_numerical_rank": float(
                    np.median([update.numerical_rank for update in ready])
                )
                if ready
                else math.nan,
                "observer_median_condition_number": float(
                    np.median([update.condition_number for update in ready])
                )
                if ready
                else math.nan,
                "observer_mean_finite_difference_time_s": float(
                    np.mean(
                        [update.finite_difference_rollout_time_s for update in ready]
                    )
                )
                if ready
                else math.nan,
                "observer_mean_line_search_time_s": float(
                    np.mean([update.line_search_rollout_time_s for update in ready])
                )
                if ready
                else math.nan,
                "observer_mean_linear_solve_time_s": float(
                    np.mean([update.linear_solve_time_s for update in ready])
                )
                if ready
                else math.nan,
            }
        )
    position_basis, velocity_basis = _basis_representability(
        execution, observer_settings
    )
    row["basis_position_relative_residual"] = position_basis
    row["basis_velocity_relative_residual"] = velocity_basis
    row.update(
        _prediction_errors(
            execution,
            model,
            simulation.simulation_dt_s,
            (0.10, 0.20, 0.30),
            torch.device("cuda"),
        )
    )
    if mode == "history":
        row.update(_propagation_only_metrics(execution, model))
    return row


def run_case(
    condition: str,
    mode: str,
    seed: int,
    study: StudySettings,
) -> tuple[dict[str, object], Figure8Execution, DderHistoryObserver | None]:
    source = load_cable_model(DEFAULT_MODEL_PATH)
    simulation = SimulationSettings(
        horizon_s=1.0,
        simulation_dt_s=0.02,
        control_interval_s=0.02,
        attachment_drop_m=0.10,
        maximum_acceleration_m_s2=6.0,
        maximum_speed_m_s=3.0,
    )
    controller, truth = build_controller_and_truth_models(
        source,
        simulation_dt_s=simulation.simulation_dt_s,
        node_count=11,
        truth_bending_stiffness_scale=1.0,
        truth_bending_damping_scale=1.0,
    )
    planner = WhipSimulator(controller, simulation, device="cuda")
    plant = WhipSimulator(truth, simulation, device="cuda")
    planner.require_online_acceleration()
    plant.require_online_acceleration()
    prior = plant.initial_state((0.0, 0.0, 1.45))
    plant_initial = (
        perturb_cable_state(
            prior,
            velocity_delta_m_s=(0.0, study.hidden_velocity_m_s, 0.0),
            profile="interior",
        )
        if condition == "hidden_interior_velocity"
        else prior
    )
    center = tuple(
        float(value)
        for value in prior.cable.positions_m[0, -1].detach().cpu().numpy()
    )
    reference = Figure8Reference(center, 0.7, 0.5)
    mppi = MppiSettings(
        samples=2048,
        rollout_batch_size=2048,
        iterations=2,
        knot_count=11,
        temperature=1.0,
        acceleration_noise_sigma_m_s2=3.0,
        noise_decay=0.92,
        seed=seed,
        gradient_guidance_fraction=0.0,
    )
    observer_settings = HistoryObserverSettings()
    observer = (
        DderHistoryObserver(
            controller,
            prior.cable,
            _initial_observation(0.0, plant_initial.cable),
            observer_settings,
            device="cuda",
        )
        if mode == "history"
        else None
    )
    started = time.perf_counter()
    execution = run_figure8_tracking(
        planner,
        plant,
        plant_initial,
        reference,
        mppi,
        TrackingCostSettings(),
        Figure8ExecutionSettings(
            replan_interval_s=0.10,
            observation_mode=mode,  # type: ignore[arg-type]
            duration_s=study.duration_s,
            realtime_pacing=False,
            periodic_swing_bootstrap=True,
        ),
        history_observer=observer,
        controller_initial_state=prior,
    )
    row = _metrics(
        condition,
        mode,
        seed,
        execution,
        observer,
        controller,
        simulation,
        observer_settings,
    )
    row["total_case_wall_time_s"] = time.perf_counter() - started
    return row, execution, observer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run one 1.0 s seed.")
    args = parser.parse_args()
    study = StudySettings(
        duration_s=1.0 if args.quick else 2.0,
        seeds=(17,) if args.quick else (17, 23, 41),
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for condition in ("clean", "hidden_interior_velocity"):
        for mode in ("full", "endpoint", "history"):
            for seed in study.seeds:
                print(f"running condition={condition} mode={mode} seed={seed}", flush=True)
                row, execution, observer = run_case(condition, mode, seed, study)
                rows.append(row)
                stem = f"{condition}__{mode}__seed{seed}"
                np.savez_compressed(
                    OUTPUT / f"{stem}.npz",
                    time_s=execution.time_s,
                    tracking_errors_m=execution.tracking_errors_m,
                    cable_positions_m=execution.cable_positions_m,
                    cable_velocities_m_s=execution.cable_velocities_m_s,
                    estimated_cable_positions_m=execution.estimated_cable_positions_m,
                    estimated_cable_velocities_m_s=execution.estimated_cable_velocities_m_s,
                    observer_update_times_s=execution.observer_update_times_s,
                    observer_history_rmse_before_m=execution.observer_history_rmse_before_m,
                    observer_history_rmse_after_m=execution.observer_history_rmse_after_m,
                    observer_finite_difference_times_s=np.asarray(
                        [
                            update.finite_difference_rollout_time_s
                            for update in (observer.updates if observer is not None else ())
                        ],
                        dtype=np.float64,
                    ),
                    observer_line_search_times_s=np.asarray(
                        [
                            update.line_search_rollout_time_s
                            for update in (observer.updates if observer is not None else ())
                        ],
                        dtype=np.float64,
                    ),
                    observer_linear_solve_times_s=np.asarray(
                        [
                            update.linear_solve_time_s
                            for update in (observer.updates if observer is not None else ())
                        ],
                        dtype=np.float64,
                    ),
                )
                print(
                    f"  RMSE={1000.0 * float(row['tip_position_rmse_m']):.1f}mm "
                    f"state={1000.0 * float(row['distributed_position_rmse_m']):.1f}mm "
                    f"wall={float(row['total_case_wall_time_s']):.1f}s",
                    flush=True,
                )
    payload = {
        "schema": "figure8_dder_history_observer_study_v1",
        "study_settings": asdict(study),
        "observer_settings": asdict(HistoryObserverSettings()),
        "fixed_physics": True,
        "parameter_adaptation": False,
        "rows": rows,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"saved {OUTPUT / 'summary.json'}")


if __name__ == "__main__":
    main()
