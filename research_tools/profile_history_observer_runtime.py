"""Profile the fixed-history DDER observer without changing its estimator.

The benchmark uses the production 11-node, 0.30 s, four-mode, two-iteration
configuration and a hidden interior-velocity perturbation.  It reports CUDA
capture/prewarm separately from repeated active correction latency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cable_twin.shared.dder import DderState, START_PINNED_FREE_END
from drone_mpc.history_observer import (
    DderHistoryObserver,
    EndpointHistoryObservation,
    HistoryObserverSettings,
)
from drone_mpc.model import load_cable_model
from drone_mpc.receding_mppi import perturb_cable_state
from drone_mpc.reduced import build_controller_and_truth_models
from drone_mpc.simulator import SimulationSettings, WhipSimulator
from optitrack_offline.config import DEFAULT_MODEL_PATH


def _observation(timestamp_s: float, state: DderState) -> EndpointHistoryObservation:
    return EndpointHistoryObservation(
        timestamp_s,
        state.positions_m[0, 0].detach().cpu().numpy(),
        state.velocities_m_s[0, 0].detach().cpu().numpy(),
        state.positions_m[0, -1].detach().cpu().numpy(),
    )


def _summary(values: list[float]) -> dict[str, float]:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    return {
        "mean_ms": 1.0e3 * statistics.fmean(values),
        "median_ms": 1.0e3 * float(np.median(ordered)),
        "minimum_ms": 1.0e3 * float(ordered[0]),
        "p95_ms": 1.0e3 * float(np.quantile(ordered, 0.95)),
        "maximum_ms": 1.0e3 * float(ordered[-1]),
        "standard_deviation_ms": (
            1.0e3 * statistics.pstdev(values) if len(values) > 1 else 0.0
        ),
    }


def profile(update_count: int) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("This runtime profile requires CUDA.")
    device = torch.device("cuda")
    dt_s = 0.02
    simulation = SimulationSettings(
        horizon_s=0.10,
        simulation_dt_s=dt_s,
        control_interval_s=dt_s,
    )
    snapshot, _truth = build_controller_and_truth_models(
        load_cable_model(DEFAULT_MODEL_PATH),
        simulation_dt_s=dt_s,
        node_count=11,
        truth_bending_stiffness_scale=1.0,
        truth_bending_damping_scale=1.0,
    )
    simulator = WhipSimulator(snapshot, simulation, device=device)
    initial = simulator.initial_state((0.0, 0.0, 1.45))
    truth = perturb_cable_state(
        initial,
        velocity_delta_m_s=(0.0, 0.45, 0.0),
        profile="interior",
    ).cable
    settings = HistoryObserverSettings()
    observer = DderHistoryObserver(
        snapshot,
        initial.cable,
        _observation(0.0, truth),
        settings,
        device=device,
    )

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    observer.prewarm(dt_s)
    prewarm_s = time.perf_counter() - started
    prewarm_peak_bytes = int(torch.cuda.max_memory_allocated(device))

    recursive_eager_times: list[float] = []
    recursive_graph_times: list[float] = []
    assert observer._recursive_step is not None
    for _ in range(update_count):
        started = time.perf_counter()
        eager = observer._step(initial.cable, observer._root_positions[0], dt_s)
        eager.positions_m.clone()
        eager.velocities_m_s.clone()
        torch.cuda.synchronize(device)
        recursive_eager_times.append(time.perf_counter() - started)

        started = time.perf_counter()
        captured = observer._recursive_step(
            initial.cable, observer._root_positions[0], dt_s
        )
        captured.positions_m.clone()
        captured.velocities_m_s.clone()
        torch.cuda.synchronize(device)
        recursive_graph_times.append(time.perf_counter() - started)

    constants = snapshot.model.runtime_constants(truth.positions_m)
    frame_count = int(round(settings.history_duration_s / dt_s)) + 1
    for index in range(1, frame_count):
        truth = snapshot.model.step_runtime(
            truth,
            truth.positions_m[:, :1],
            torch.full((1,), dt_s, dtype=torch.float32, device=device),
            constants,
            iterative_damping=True,
            pinned_endpoints=START_PINNED_FREE_END,
        )
        observer.ingest(_observation(index * dt_s, truth))

    frozen_states = [
        DderState(state.positions_m.clone(), state.velocities_m_s.clone())
        for state in observer._states
    ]
    total_times: list[float] = []
    fd_times: list[float] = []
    line_times: list[float] = []
    accepted = 0
    last_update = None
    for _ in range(update_count):
        observer._states = [
            DderState(state.positions_m.clone(), state.velocities_m_s.clone())
            for state in frozen_states
        ]
        update = observer.correct()
        last_update = update
        total_times.append(update.total_time_s)
        fd_times.append(update.finite_difference_rollout_time_s)
        line_times.append(update.line_search_rollout_time_s)
        accepted += int(update.accepted)

    assert last_update is not None
    return {
        "device": torch.cuda.get_device_name(device),
        "node_count": snapshot.node_count,
        "history_duration_s": settings.history_duration_s,
        "history_frames": observer.frame_count,
        "spatial_modes": settings.spatial_mode_count,
        "correction_dimension": observer.correction_dimension,
        "finite_difference_candidates": 1 + 2 * observer.correction_dimension,
        "line_search_candidates": len(settings.line_search_steps),
        "maximum_gn_iterations": settings.maximum_iterations,
        "prewarm_ms": 1.0e3 * prewarm_s,
        "prewarm_peak_memory_mib": prewarm_peak_bytes / (1024.0 * 1024.0),
        "recursive_eager_step": _summary(recursive_eager_times),
        "recursive_captured_step": _summary(recursive_graph_times),
        "updates": update_count,
        "accepted_updates": accepted,
        "reference_correction": {
            "measurement_rmse_before_mm": (
                1.0e3 * last_update.measurement_rmse_before_m
            ),
            "measurement_rmse_after_mm": (
                1.0e3 * last_update.measurement_rmse_after_m
            ),
            "objective_before": last_update.objective_before,
            "objective_after": last_update.objective_after,
            "correction_norm": last_update.correction_norm,
            "numerical_rank": last_update.numerical_rank,
        },
        "total": _summary(total_times),
        "finite_difference_rollouts": _summary(fd_times),
        "line_search_rollouts": _summary(line_times),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.updates < 1:
        raise ValueError("--updates must be positive")
    result = profile(args.updates)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
