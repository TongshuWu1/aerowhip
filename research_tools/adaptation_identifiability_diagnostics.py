"""Finite-difference stability and phase-wise EI/Cb loss landscapes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np

from drone_mpc.distributed_adaptation import (
    DistributedAdaptationSettings,
    DistributedParameterFitter,
    ParameterEstimate,
    _information,
    _residual_vector,
    _segment_from_observations,
)
from research_tools.distributed_adaptation_study import (
    generate_truth_rollout,
    observations_from_rollout,
)
from research_tools.profile_mppi_forward import DEFAULT_PROFILE, load_workload


SCHEMA = "distributed_adaptation_identifiability_diagnostics_v1"
DEFAULT_OUTPUT = Path(
    "data/drone_mpc/adaptation/adaptation_identifiability_diagnostics.json"
)
PHASE_STARTS = {
    "stroke": 0.08,
    "reversal": 0.28,
    "distal_lash": 0.50,
    "weak_late_motion": 1.52,
}


def _serialize(value):
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _phase_segment(observations, start_s: float, duration_s: float):
    frames = [
        observation
        for observation in observations
        if start_s - 1.0e-9
        <= observation.timestamp_s
        <= start_s + duration_s + 1.0e-9
    ]
    if len(frames) < 2:
        raise RuntimeError(f"No segment was available at t={start_s:g}s.")
    return _segment_from_observations(frames, 0.0, "phase_diagnostic")


def _finite_difference_columns(fitter, segment, eta, step_e, step_c):
    hypotheses = np.repeat(eta[None], 5, axis=0)
    hypotheses[1, 0] += step_e
    hypotheses[2, 0] -= step_e
    hypotheses[3, 1] += step_c
    hypotheses[4, 1] -= step_c
    positions, _velocities, elapsed = fitter.predictor.predict((segment,), hypotheses)
    observed = segment.cable_positions_m[None]
    column_e = (
        _residual_vector(positions[:, 1][0:1], observed, fitter.nominal_model.cable_length_m)
        - _residual_vector(positions[:, 2][0:1], observed, fitter.nominal_model.cable_length_m)
    ) / (2.0 * step_e)
    column_c = (
        _residual_vector(positions[:, 3][0:1], observed, fitter.nominal_model.cable_length_m)
        - _residual_vector(positions[:, 4][0:1], observed, fitter.nominal_model.cable_length_m)
    ) / (2.0 * step_c)
    return column_e, column_c, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--truth-ei", type=float, default=0.8)
    parser.add_argument("--truth-cb", type=float, default=0.7)
    parser.add_argument("--grid-size", type=int, default=17)
    args = parser.parse_args()
    workload = load_workload(args.profile)
    settings = DistributedAdaptationSettings()
    rollout, _truth_simulator, _simulation = generate_truth_rollout(
        workload, args.truth_ei, args.truth_cb
    )
    observations = observations_from_rollout(
        rollout, workload, ParameterEstimate()
    )
    fitter = DistributedParameterFitter(workload.controller, settings, device="cuda")
    steps = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08)
    grid_e = np.linspace(0.6, 1.4, args.grid_size)
    grid_c = np.linspace(0.5, 1.5, args.grid_size)
    grid_eta = np.asarray(
        [(math.log(e), math.log(c)) for e in grid_e for c in grid_c],
        dtype=np.float64,
    )
    phase_rows = []
    started = time.perf_counter()
    for phase, start_s in PHASE_STARTS.items():
        segment = _phase_segment(observations, start_s, settings.segment_duration_s)
        reference_e, reference_c, _elapsed = _finite_difference_columns(
            fitter, segment, np.zeros(2), 0.03, 0.03
        )
        finite_rows = []
        for step in steps:
            column_e, column_c, elapsed = _finite_difference_columns(
                fitter, segment, np.zeros(2), step, step
            )
            finite_rows.append(
                {
                    "log_step": step,
                    "ei_norm": float(np.linalg.norm(column_e)),
                    "cb_norm": float(np.linalg.norm(column_c)),
                    "ei_direction_cosine_vs_0.03": float(
                        np.dot(column_e, reference_e)
                        / (np.linalg.norm(column_e) * np.linalg.norm(reference_e) + 1.0e-30)
                    ),
                    "cb_direction_cosine_vs_0.03": float(
                        np.dot(column_c, reference_c)
                        / (np.linalg.norm(column_c) * np.linalg.norm(reference_c) + 1.0e-30)
                    ),
                    "forward_time_s": elapsed,
                }
            )
        jacobian = np.stack((reference_e, reference_c), axis=1)
        information = _information(jacobian, settings)
        positions, _velocities, grid_elapsed = fitter.predictor.predict(
            (segment,), grid_eta
        )
        observed = segment.cable_positions_m
        losses = np.mean(
            (
                (
                    positions[0, :, 1:, 1:, :]
                    - observed[None, 1:, 1:, :]
                )
                / workload.controller.cable_length_m
            )
            ** 2,
            axis=(1, 2, 3),
        )
        minimum_index = int(np.argmin(losses))
        phase_rows.append(
            {
                "phase": phase,
                "start_time_s": segment.start_time_s,
                "duration_s": segment.time_s[-1] - segment.time_s[0],
                "information": asdict(information),
                "finite_difference_study": finite_rows,
                "loss_grid": {
                    "ei_ratios": grid_e,
                    "cb_ratios": grid_c,
                    "loss": losses.reshape(len(grid_e), len(grid_c)),
                    "minimum_ei_ratio": float(math.exp(grid_eta[minimum_index, 0])),
                    "minimum_cb_ratio": float(math.exp(grid_eta[minimum_index, 1])),
                    "minimum_loss": float(losses[minimum_index]),
                    "forward_time_s": grid_elapsed,
                },
            }
        )
        print(
            f"{phase}: lambda_min={information.minimum_eigenvalue:.3g} "
            f"condition={information.condition:.2f} rho={information.sensitivity_correlation:.3f} "
            f"grid_min=({math.exp(grid_eta[minimum_index, 0]):.3f},"
            f"{math.exp(grid_eta[minimum_index, 1]):.3f})"
        )
    payload = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "profile": str(args.profile.resolve()),
        "truth_ei_ratio": args.truth_ei,
        "truth_cb_ratio": args.truth_cb,
        "settings": asdict(settings),
        "phases": phase_rows,
        "elapsed_wall_s": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_serialize(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
