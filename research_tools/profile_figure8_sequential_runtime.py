"""Measure current observer -> MPPI sequential Figure-8 update latency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from drone_mpc.figure8_tracking import (
    Figure8ExecutionSettings,
    Figure8Reference,
    TrackingCostSettings,
    run_figure8_tracking,
)
from drone_mpc.hidden_state_disturbance import (
    apply_hidden_velocity_disturbance,
    primary_hidden_velocity_disturbances,
)
from drone_mpc.history_observer import (
    DderHistoryObserver,
    EndpointHistoryObservation,
    HistoryObserverSettings,
)
from drone_mpc.model import load_cable_model
from drone_mpc.mppi import MppiSettings
from drone_mpc.reduced import build_controller_and_truth_models
from drone_mpc.simulator import SimulationSettings, WhipSimulator
from optitrack_offline.config import DEFAULT_MODEL_PATH


def _summary_seconds(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean_ms": 1.0e3 * statistics.fmean(array),
        "median_ms": 1.0e3 * float(np.median(array)),
        "p95_ms": 1.0e3 * float(np.percentile(array, 95.0)),
        "maximum_ms": 1.0e3 * float(np.max(array)),
    }


def run_profile(duration_s: float) -> dict[str, object]:
    simulation = SimulationSettings(
        horizon_s=1.0,
        simulation_dt_s=0.02,
        control_interval_s=0.02,
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
    prior = plant.initial_state((0.0, 0.0, 1.45))
    sin1 = {item.name: item for item in primary_hidden_velocity_disturbances()}[
        "sin1"
    ]
    plant_initial = apply_hidden_velocity_disturbance(prior, sin1)
    initial_position = plant_initial.cable.positions_m[0].detach().cpu().numpy()
    initial_velocity = plant_initial.cable.velocities_m_s[0].detach().cpu().numpy()
    observer_settings = HistoryObserverSettings()
    observer = DderHistoryObserver(
        controller,
        prior.cable,
        EndpointHistoryObservation(
            0.0,
            initial_position[0],
            initial_velocity[0],
            initial_position[-1],
        ),
        observer_settings,
        device="cuda",
    )
    observer.prewarm(simulation.simulation_dt_s)
    center = tuple(
        float(value)
        for value in prior.cable.positions_m[0, -1].detach().cpu().numpy()
    )
    mppi = MppiSettings(
        samples=1024,
        rollout_batch_size=1024,
        iterations=2,
        knot_count=11,
        temperature=1.0,
        acceleration_noise_sigma_m_s2=3.0,
        noise_decay=0.92,
        seed=17,
        gradient_guidance_fraction=0.0,
    )
    execution = run_figure8_tracking(
        planner,
        plant,
        plant_initial,
        Figure8Reference(center, 0.7, 0.5),
        mppi,
        TrackingCostSettings(),
        Figure8ExecutionSettings(
            replan_interval_s=0.10,
            observation_mode="history",
            duration_s=duration_s,
            realtime_pacing=False,
            periodic_swing_bootstrap=True,
        ),
        history_observer=observer,
        controller_initial_state=prior,
    )
    observer_times = np.asarray(execution.observer_update_times_s)
    active = observer_times > 0.0
    observer_active = observer_times[active]
    mppi_active = np.asarray(execution.planning_wall_times_s)[active]
    total_active = np.asarray(execution.sequential_update_wall_times_s)[active]
    unattributed = total_active - observer_active - mppi_active
    active_frame_counts = np.asarray(
        [update.frame_count for update in observer.updates], dtype=np.int64
    )[active]
    frame_count_histogram = {
        str(value): int(np.sum(active_frame_counts == value))
        for value in np.unique(active_frame_counts)
    }
    return {
        "device": torch.cuda.get_device_name(0),
        "configuration": {
            "mppi_horizon_s": simulation.horizon_s,
            "mppi_candidates": mppi.samples,
            "mppi_iterations": mppi.iterations,
            "acceleration_knots": mppi.knot_count,
            "physics_rate_hz": 1.0 / simulation.simulation_dt_s,
            "acceleration_control_rate_hz": 1.0 / simulation.control_interval_s,
            "requested_replanning_rate_hz": 10.0,
            "observer_correction_rate_hz": 10.0,
            "observer_history_s": observer_settings.history_duration_s,
            "observer_frames": 16,
        },
        "duration_s": duration_s,
        "all_updates": int(len(observer_times)),
        "active_warmed_updates": int(np.sum(active)),
        "active_frame_count_histogram": frame_count_histogram,
        "observer": _summary_seconds(observer_active),
        "mppi": _summary_seconds(mppi_active),
        "total_sequential": _summary_seconds(total_active),
        "unattributed": _summary_seconds(unattributed),
        "achievable_sequential_rate_hz_from_mean": (
            1.0 / float(np.mean(total_active))
        ),
        "achievable_sequential_rate_hz_from_p95": (
            1.0 / float(np.percentile(total_active, 95.0))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_profile(args.duration)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
