"""Generalization study for the frozen accelerated endpoint-history observer."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Literal

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
from drone_mpc.hidden_state_disturbance import (
    RANDOM_SMOOTH_SEED,
    HiddenVelocityDisturbance,
    apply_hidden_velocity_disturbance,
    basis_representability,
    disturbance_diagnostics,
    primary_hidden_velocity_disturbances,
)
from drone_mpc.history_observer import (
    DderHistoryObserver,
    EndpointHistoryObservation,
    HistoryObserverSettings,
    spatial_correction_basis,
)
from drone_mpc.model import CableModelSnapshot, load_cable_model
from drone_mpc.mppi import MppiSettings
from drone_mpc.reduced import build_controller_and_truth_models
from drone_mpc.simulator import DroneCableState, SimulationSettings, WhipSimulator
from optitrack_offline.config import DEFAULT_MODEL_PATH


OUTPUT = PROJECT / "reports" / "figure8_history_observer_generalization_data"
Mode = Literal["full", "endpoint", "history"]


@dataclass(frozen=True, slots=True)
class GeneralizationStudySettings:
    duration_s: float = 2.0
    seeds: tuple[int, ...] = (17, 23, 41)
    samples: int = 1024
    iterations: int = 2
    horizon_s: float = 1.0
    knot_count: int = 11
    physics_dt_s: float = 0.02
    replan_interval_s: float = 0.10
    midrun_injection_time_s: float = 0.80
    random_smooth_seed: int = RANDOM_SMOOTH_SEED


@dataclass(frozen=True, slots=True)
class ExperimentCase:
    case_id: str
    disturbance: HiddenVelocityDisturbance
    injection_time_s: float
    category: str


def _initial_observation(
    timestamp_s: float, state: DderState
) -> EndpointHistoryObservation:
    position = state.positions_m[0].detach().cpu().numpy()
    velocity = state.velocities_m_s[0].detach().cpu().numpy()
    return EndpointHistoryObservation(
        timestamp_s, position[0], velocity[0], position[-1]
    )


def _prediction_errors_from_state_history(
    execution: Figure8Execution,
    model: CableModelSnapshot,
    positions_m: np.ndarray,
    velocities_m_s: np.ndarray,
    *,
    prefix: str,
    horizons_s: tuple[float, ...] = (0.10, 0.20, 0.30),
) -> dict[str, float]:
    results: dict[str, float] = {}
    dt_s = float(np.median(np.diff(execution.time_s)))
    minimum_start = int(round(0.30 / dt_s))
    stride = max(1, int(round(0.10 / dt_s)))
    dtype = torch.float32
    for horizon_s in horizons_s:
        step_count = int(round(horizon_s / dt_s))
        starts = np.arange(
            minimum_start,
            len(execution.time_s) - step_count,
            stride,
            dtype=np.int64,
        )
        if len(starts) == 0:
            results[f"{prefix}_tip_prediction_rmse_{horizon_s:.2f}s_m"] = math.nan
            continue
        state = DderState(
            torch.as_tensor(positions_m[starts], dtype=dtype, device="cuda"),
            torch.as_tensor(velocities_m_s[starts], dtype=dtype, device="cuda"),
        )
        constants = model.model.runtime_constants(state.positions_m)
        dt = torch.full((len(starts),), dt_s, dtype=dtype, device="cuda")
        for offset in range(1, step_count + 1):
            boundary = torch.as_tensor(
                execution.cable_positions_m[starts + offset, 0, None],
                dtype=dtype,
                device="cuda",
            )
            state = model.model.step_runtime(
                state,
                boundary,
                dt,
                constants,
                iterative_damping=True,
                pinned_endpoints=START_PINNED_FREE_END,
            )
        predicted_tip = state.positions_m[:, -1].detach().cpu().numpy()
        actual_tip = execution.actual_tip_positions_m[starts + step_count]
        results[f"{prefix}_tip_prediction_rmse_{horizon_s:.2f}s_m"] = float(
            np.sqrt(np.mean(np.square(predicted_tip - actual_tip)))
        )
    return results


def _propagation_only_history(
    execution: Figure8Execution,
    model: CableModelSnapshot,
    prior: DroneCableState,
) -> tuple[np.ndarray, np.ndarray]:
    observer = DderHistoryObserver(
        model,
        prior.cable,
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
    return np.asarray(positions), np.asarray(velocities)


def _state_metrics(
    execution: Figure8Execution,
    positions_m: np.ndarray,
    velocities_m_s: np.ndarray,
    *,
    prefix: str,
) -> dict[str, float]:
    position_error = positions_m[:, 1:] - execution.cable_positions_m[:, 1:]
    velocity_error = velocities_m_s[:, 1:] - execution.cable_velocities_m_s[:, 1:]
    return {
        f"{prefix}_position_rmse_m": float(
            np.sqrt(np.mean(np.square(position_error)))
        ),
        f"{prefix}_velocity_rmse_m_s": float(
            np.sqrt(np.mean(np.square(velocity_error)))
        ),
    }


def _observer_metrics(
    observer: DderHistoryObserver | None,
) -> dict[str, object]:
    if observer is None:
        return {}
    ready = [update for update in observer.updates if update.ready]
    if not ready:
        return {
            "observer_ready_updates": 0,
            "observer_accepted_updates": 0,
        }
    significant_smallest: list[float] = []
    largest: list[float] = []
    alphas: list[float] = []
    for update in ready:
        values = np.asarray(update.singular_values, dtype=np.float64)
        if len(values):
            largest.append(float(values[0]))
        if update.numerical_rank > 0 and len(values) >= update.numerical_rank:
            significant_smallest.append(float(values[update.numerical_rank - 1]))
        alphas.extend(update.selected_line_search_alphas)
    times = np.asarray([update.total_time_s for update in ready])
    return {
        "observer_ready_updates": len(ready),
        "observer_accepted_updates": int(sum(update.accepted for update in ready)),
        "observer_rejected_updates": int(sum(not update.accepted for update in ready)),
        "observer_mean_iterations": float(
            np.mean([update.iterations for update in ready])
        ),
        "observer_mean_correction_norm": float(
            np.mean([update.correction_norm for update in ready])
        ),
        "observer_history_rmse_before_m": float(
            np.mean([update.measurement_rmse_before_m for update in ready])
        ),
        "observer_history_rmse_after_m": float(
            np.mean([update.measurement_rmse_after_m for update in ready])
        ),
        "observer_median_numerical_rank": float(
            np.median([update.numerical_rank for update in ready])
        ),
        "observer_median_condition_number": float(
            np.median([update.condition_number for update in ready])
        ),
        "observer_median_smallest_significant_singular_value": (
            float(np.median(significant_smallest))
            if significant_smallest
            else math.nan
        ),
        "observer_median_largest_singular_value": (
            float(np.median(largest)) if largest else math.nan
        ),
        "observer_trust_region_saturation_count": int(
            sum(update.trust_region_saturations for update in ready)
        ),
        "observer_component_bound_saturation_count": int(
            sum(update.component_bound_saturations for update in ready)
        ),
        "observer_line_search_alphas": alphas,
        "observer_mean_update_time_s": float(np.mean(times)),
        "observer_median_update_time_s": float(np.median(times)),
        "observer_p95_update_time_s": float(np.percentile(times, 95.0)),
        "observer_maximum_update_time_s": float(np.max(times)),
    }


def _case_metrics(
    case: ExperimentCase,
    mode: Mode,
    seed: int,
    execution: Figure8Execution,
    observer: DderHistoryObserver | None,
    model: CableModelSnapshot,
    prior: DroneCableState,
    basis_metrics: dict[str, float | np.ndarray],
) -> tuple[dict[str, object], tuple[np.ndarray, np.ndarray] | None]:
    row: dict[str, object] = {
        "case_id": case.case_id,
        "category": case.category,
        "disturbance_name": case.disturbance.name,
        "disturbance_coefficients": list(case.disturbance.coefficients),
        "disturbance_direction": list(case.disturbance.direction),
        "injection_time_s": case.injection_time_s,
        "mode": mode,
        "seed": seed,
        "basis_relative_residual": float(basis_metrics["relative_residual"]),
        "basis_explained_fraction": float(basis_metrics["explained_fraction"]),
        **disturbance_diagnostics(case.disturbance, model.node_count),
        **execution.summary,
        **_state_metrics(
            execution,
            execution.estimated_cable_positions_m,
            execution.estimated_cable_velocities_m_s,
            prefix="estimated",
        ),
        **_prediction_errors_from_state_history(
            execution,
            model,
            execution.estimated_cable_positions_m,
            execution.estimated_cable_velocities_m_s,
            prefix="estimated",
        ),
        **_observer_metrics(observer),
    }
    propagation: tuple[np.ndarray, np.ndarray] | None = None
    if mode == "history":
        propagation = _propagation_only_history(execution, model, prior)
        propagation_position, propagation_velocity = propagation
        row.update(
            _state_metrics(
                execution,
                propagation_position,
                propagation_velocity,
                prefix="propagation_only",
            )
        )
        row.update(
            _prediction_errors_from_state_history(
                execution,
                model,
                propagation_position,
                propagation_velocity,
                prefix="propagation_only",
            )
        )
    return row, propagation


def _injection_hook(
    disturbance: HiddenVelocityDisturbance, injection_time_s: float
):
    injected = False

    def hook(elapsed_s: float, state: DroneCableState) -> DroneCableState:
        nonlocal injected
        if not injected and elapsed_s + 1.0e-9 >= injection_time_s:
            injected = True
            return apply_hidden_velocity_disturbance(state, disturbance)
        return state

    return hook


def run_case(
    case: ExperimentCase,
    mode: Mode,
    seed: int,
    settings: GeneralizationStudySettings,
    planner: WhipSimulator,
    plant: WhipSimulator,
    model: CableModelSnapshot,
    basis_metrics: dict[str, float | np.ndarray],
) -> tuple[
    dict[str, object],
    Figure8Execution,
    DderHistoryObserver | None,
    tuple[np.ndarray, np.ndarray] | None,
]:
    prior = plant.initial_state((0.0, 0.0, 1.45))
    initial_injection = case.injection_time_s <= 1.0e-12
    plant_initial = (
        apply_hidden_velocity_disturbance(prior, case.disturbance)
        if initial_injection and case.disturbance.name != "clean"
        else prior
    )
    center = tuple(
        float(value)
        for value in prior.cable.positions_m[0, -1].detach().cpu().numpy()
    )
    observer_settings = HistoryObserverSettings()
    observer = (
        DderHistoryObserver(
            model,
            prior.cable,
            _initial_observation(0.0, plant_initial.cable),
            observer_settings,
            device="cuda",
        )
        if mode == "history"
        else None
    )
    if observer is not None:
        observer.prewarm(settings.physics_dt_s)
    mppi = MppiSettings(
        samples=settings.samples,
        rollout_batch_size=settings.samples,
        iterations=settings.iterations,
        knot_count=settings.knot_count,
        temperature=1.0,
        acceleration_noise_sigma_m_s2=3.0,
        noise_decay=0.92,
        seed=seed,
        gradient_guidance_fraction=0.0,
    )
    hook = (
        _injection_hook(case.disturbance, case.injection_time_s)
        if not initial_injection and case.disturbance.name != "clean"
        else None
    )
    started = time.perf_counter()
    execution = run_figure8_tracking(
        planner,
        plant,
        plant_initial,
        Figure8Reference(center, 0.7, 0.5),
        mppi,
        TrackingCostSettings(),
        Figure8ExecutionSettings(
            replan_interval_s=settings.replan_interval_s,
            observation_mode=mode,
            duration_s=settings.duration_s,
            realtime_pacing=False,
            periodic_swing_bootstrap=True,
        ),
        history_observer=observer,
        controller_initial_state=prior,
        plant_state_hook=hook,
    )
    row, propagation = _case_metrics(
        case, mode, seed, execution, observer, model, prior, basis_metrics
    )
    row["total_case_wall_time_s"] = time.perf_counter() - started
    return row, execution, observer, propagation


def _save_case(
    case: ExperimentCase,
    mode: Mode,
    seed: int,
    execution: Figure8Execution,
    observer: DderHistoryObserver | None,
    propagation: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    stem = f"{case.case_id}__{mode}__seed{seed}"
    payload: dict[str, np.ndarray] = {
        "time_s": execution.time_s,
        "reference_tip_positions_m": execution.reference_tip_positions_m,
        "tracking_errors_m": execution.tracking_errors_m,
        "cable_positions_m": execution.cable_positions_m,
        "cable_velocities_m_s": execution.cable_velocities_m_s,
        "estimated_cable_positions_m": execution.estimated_cable_positions_m,
        "estimated_cable_velocities_m_s": execution.estimated_cable_velocities_m_s,
        "observer_update_times_s": execution.observer_update_times_s,
        "planning_wall_times_s": execution.planning_wall_times_s,
        "sequential_update_wall_times_s": execution.sequential_update_wall_times_s,
    }
    if propagation is not None:
        payload["propagation_only_positions_m"] = propagation[0]
        payload["propagation_only_velocities_m_s"] = propagation[1]
    if observer is not None:
        payload["observer_rank"] = np.asarray(
            [update.numerical_rank for update in observer.updates]
        )
        payload["observer_condition"] = np.asarray(
            [update.condition_number for update in observer.updates]
        )
        payload["observer_rmse_before_m"] = np.asarray(
            [update.measurement_rmse_before_m for update in observer.updates]
        )
        payload["observer_rmse_after_m"] = np.asarray(
            [update.measurement_rmse_after_m for update in observer.updates]
        )
    np.savez_compressed(OUTPUT / f"{stem}.npz", **payload)


def _primary_cases(settings: GeneralizationStudySettings) -> list[ExperimentCase]:
    return [
        ExperimentCase(item.name, item, 0.0, "primary_initial_y")
        for item in primary_hidden_velocity_disturbances(
            random_seed=settings.random_smooth_seed
        )
    ]


def _direction_cases(settings: GeneralizationStudySettings) -> list[ExperimentCase]:
    selected = {
        item.name: item
        for item in primary_hidden_velocity_disturbances(
            random_seed=settings.random_smooth_seed
        )
    }
    directions = {
        "x": (1.0, 0.0, 0.0),
        "xy": (1.0, 1.0, 0.0),
    }
    result: list[ExperimentCase] = []
    for name in ("sin1", "sin4", "mixed_span"):
        for label, direction in directions.items():
            disturbance = selected[name].with_direction(
                direction, name=f"{name}_{label}"
            )
            result.append(
                ExperimentCase(
                    disturbance.name,
                    disturbance,
                    0.0,
                    "direction_initial",
                )
            )
    return result


def _midrun_cases(settings: GeneralizationStudySettings) -> list[ExperimentCase]:
    selected = {
        item.name: item
        for item in primary_hidden_velocity_disturbances(
            random_seed=settings.random_smooth_seed
        )
    }
    return [
        ExperimentCase(
            f"{name}_midrun",
            selected[name],
            settings.midrun_injection_time_s,
            "midrun_y",
        )
        for name in ("sin1", "sin4", "mixed_span")
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("primary", "directions", "midrun", "all"), default="all"
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--case", action="append", dest="case_ids")
    args = parser.parse_args()
    settings = GeneralizationStudySettings(
        duration_s=1.0 if args.quick else 2.0,
        seeds=(17,) if args.quick else (17, 23, 41),
    )
    simulation = SimulationSettings(
        horizon_s=settings.horizon_s,
        simulation_dt_s=settings.physics_dt_s,
        control_interval_s=settings.physics_dt_s,
        attachment_drop_m=0.10,
        maximum_acceleration_m_s2=6.0,
        maximum_speed_m_s=3.0,
    )
    controller, truth = build_controller_and_truth_models(
        load_cable_model(DEFAULT_MODEL_PATH),
        simulation_dt_s=simulation.simulation_dt_s,
        node_count=11,
        truth_bending_stiffness_scale=1.0,
        truth_bending_damping_scale=1.0,
    )
    planner = WhipSimulator(controller, simulation, device="cuda")
    plant = WhipSimulator(truth, simulation, device="cuda")
    planner.require_online_acceleration()
    plant.require_online_acceleration()
    basis = spatial_correction_basis(
        11, 4, dtype=torch.float64, device=torch.device("cpu")
    ).numpy()
    cases: list[ExperimentCase] = []
    if args.stage in ("primary", "all"):
        cases.extend(_primary_cases(settings))
    if args.stage in ("directions", "all"):
        cases.extend(_direction_cases(settings))
    if args.stage in ("midrun", "all"):
        cases.extend(_midrun_cases(settings))
    if args.case_ids:
        requested = set(args.case_ids)
        cases = [case for case in cases if case.case_id in requested]
        missing = requested - {case.case_id for case in cases}
        if missing:
            raise ValueError(f"Unknown/unstaged case ids: {sorted(missing)}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    case_metadata: dict[str, object] = {}
    total = len(cases) * 3 * len(settings.seeds)
    completed = 0
    for case in cases:
        representability = basis_representability(case.disturbance, basis)
        case_metadata[case.case_id] = {
            "disturbance": asdict(case.disturbance),
            "injection_time_s": case.injection_time_s,
            "category": case.category,
            "basis_relative_residual": float(
                representability["relative_residual"]
            ),
            "basis_explained_fraction": float(
                representability["explained_fraction"]
            ),
            **disturbance_diagnostics(case.disturbance, controller.node_count),
        }
        for mode in ("full", "endpoint", "history"):
            for seed in settings.seeds:
                completed += 1
                print(
                    f"[{completed}/{total}] case={case.case_id} mode={mode} seed={seed}",
                    flush=True,
                )
                row, execution, observer, propagation = run_case(
                    case,
                    mode,  # type: ignore[arg-type]
                    seed,
                    settings,
                    planner,
                    plant,
                    controller,
                    representability,
                )
                rows.append(row)
                _save_case(
                    case,
                    mode,  # type: ignore[arg-type]
                    seed,
                    execution,
                    observer,
                    propagation,
                )
                print(
                    f"  tracking={1000.0 * float(row['tip_position_rmse_m']):.2f}mm "
                    f"state={1000.0 * float(row['estimated_position_rmse_m']):.2f}mm "
                    f"wall={float(row['total_case_wall_time_s']):.2f}s",
                    flush=True,
                )
        partial = {
            "schema": "figure8_history_observer_generalization_v1",
            "settings": asdict(settings),
            "observer_settings": asdict(HistoryObserverSettings()),
            "case_metadata": case_metadata,
            "rows": rows,
        }
        (OUTPUT / "summary_partial.json").write_text(
            json.dumps(partial, indent=2, sort_keys=True), encoding="utf-8"
        )

    payload = {
        "schema": "figure8_history_observer_generalization_v1",
        "settings": asdict(settings),
        "observer_settings": asdict(HistoryObserverSettings()),
        "physics_matched": True,
        "parameter_adaptation": False,
        "sensor_noise": False,
        "case_metadata": case_metadata,
        "rows": rows,
    }
    (OUTPUT / f"summary_{args.stage}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"saved {OUTPUT / f'summary_{args.stage}.json'}")


if __name__ == "__main__":
    main()
