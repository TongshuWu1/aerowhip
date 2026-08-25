from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np

from ..shared.observation_data import (
    load_observation,
    registered_depth_standard_deviation,
    save_trajectory,
)

from .config import DEFAULT_CONFIG_PATH, load_cable_settings, load_settings
from .filter import DderParticleFilter, ParticleFilterEstimate
from ..shared.model_artifact import load_dder_artifact


TIMING_NAMES = ("boundary_ms", "dder_transition_ms", "measurement_ms", "total_ms")


def _gravity(sequence) -> tuple[float, float, float]:
    finite = np.all(np.isfinite(sequence.gravity_camera_m_s2), axis=1)
    if not np.any(finite):
        raise ValueError("Observation contains no synchronized ZED IMU gravity.")
    value = np.median(sequence.gravity_camera_m_s2[finite], axis=0)
    magnitude = float(np.linalg.norm(value))
    if not 8.5 <= magnitude <= 10.8:
        raise ValueError(f"Implausible camera-frame gravity magnitude {magnitude:.3f}m/s2.")
    return tuple(float(item) for item in value)


def _atomic_save(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.npz")
    try:
        np.savez_compressed(temporary, **values)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(arguments: argparse.Namespace) -> tuple[Path, Path]:
    sequence = load_observation(arguments.observation)
    artifact = load_dder_artifact(arguments.model)
    settings = load_settings(arguments.config)
    cable_settings = load_cable_settings(arguments.config)
    if artifact.cable_identity != sequence.cable_identity:
        raise ValueError("DDER model and observation cable identities differ.")
    model = artifact.make_rescaled_bare_model(
        _gravity(sequence),
        cable_length_m=cable_settings.length_m,
        cable_diameter_m=cable_settings.diameter_m,
        node_count=cable_settings.node_count,
        substeps=cable_settings.solver_substeps,
        constraint_iterations=cable_settings.constraint_iterations,
    )
    tracker = DderParticleFilter(model, settings)
    metric_sensor_sigma = registered_depth_standard_deviation(
        sequence.metric_points_m,
        sequence.metric_valid,
        sequence.depth_spread_m,
        sequence.depth_support,
    )
    metric_sigma = np.sqrt(
        metric_sensor_sigma**2 + settings.metric_curve_noise_m**2
    )
    endpoint_sensor_sigma = registered_depth_standard_deviation(
        sequence.metric_endpoint_points_m,
        sequence.metric_endpoint_valid,
        sequence.metric_endpoint_depth_spread_m,
        sequence.metric_endpoint_support,
    )
    endpoint_sigma = np.sqrt(
        endpoint_sensor_sigma**2 + settings.metric_endpoint_noise_m**2
    )
    intrinsics = np.asarray(
        sequence.metadata["source"]["calibration"]["left_intrinsics"],
        dtype=np.float64,
    )
    frame_count = sequence.frame_count
    node_count = model.parameters.node_count
    estimates: list[ParticleFilterEstimate | None] = []
    initialized_frame = -1
    started = time.perf_counter()
    for frame in range(frame_count):
        if not tracker.initialized:
            try:
                estimate = tracker.initialize(
                    timestamp_ns=int(sequence.timestamps_ns[frame]),
                    metric_points_m=sequence.metric_points_m[frame],
                    metric_valid=sequence.metric_valid[frame],
                    segment_ids=sequence.route_segment_id[frame],
                    endpoints_m=sequence.metric_endpoint_points_m[frame],
                    endpoint_valid=sequence.metric_endpoint_valid[frame],
                    metric_sigma_m=metric_sigma[frame],
                    endpoint_sigma_m=endpoint_sigma[frame],
                )
            except ValueError:
                estimates.append(None)
                continue
            initialized_frame = frame
            print(f"PF initialized at frame {frame + 1}/{frame_count}", flush=True)
        else:
            estimate = tracker.update(
                timestamp_ns=int(sequence.timestamps_ns[frame]),
                endpoints_m=sequence.metric_endpoint_points_m[frame],
                endpoint_valid=sequence.metric_endpoint_valid[frame],
                endpoint_sigma_m=endpoint_sigma[frame],
                image_points_xy=sequence.route_xy[frame],
                image_point_valid=sequence.route_valid[frame],
                image_endpoints_xy=sequence.endpoint_centers_xy[frame],
                image_endpoint_valid=sequence.endpoint_valid[frame],
                left_intrinsics=intrinsics,
            )
        estimates.append(estimate)
        if frame % 30 == 0 or frame + 1 == frame_count:
            print(
                f"PF frame={frame + 1}/{frame_count} ESS={estimate.ess:.1f} "
                f"residual={estimate.body_residual_px:.2f}px "
                f"time={estimate.timing_ms[-1]:.1f}ms",
                flush=True,
            )
    if initialized_frame < 0:
        raise ValueError("No frame provided a complete causal PF initialization.")

    positions = np.full((frame_count, node_count, 3), np.nan, dtype=np.float32)
    velocities = np.full_like(positions, np.nan)
    posterior_mean = np.full_like(positions, np.nan)
    covariance = np.full((frame_count, node_count, 3, 3), np.nan, dtype=np.float32)
    weights = np.full((frame_count, settings.particle_count), np.nan, dtype=np.float32)
    ess = np.full(frame_count, np.nan, dtype=np.float32)
    residual = np.full(frame_count, np.nan, dtype=np.float32)
    body_updated = np.zeros(frame_count, dtype=bool)
    endpoint_count = np.zeros(frame_count, dtype=np.int8)
    prediction_only = np.zeros(frame_count, dtype=bool)
    resampled = np.zeros(frame_count, dtype=bool)
    timing = np.full((frame_count, len(TIMING_NAMES)), np.nan, dtype=np.float32)
    for frame, estimate in enumerate(estimates):
        if estimate is None:
            continue
        positions[frame] = estimate.positions_m
        velocities[frame] = estimate.velocities_m_s
        posterior_mean[frame] = estimate.posterior_mean_m
        covariance[frame] = estimate.node_covariance_m2
        weights[frame] = estimate.weights
        ess[frame] = estimate.ess
        residual[frame] = estimate.body_residual_px
        body_updated[frame] = estimate.body_updated
        endpoint_count[frame] = estimate.endpoint_count
        prediction_only[frame] = estimate.prediction_only
        resampled[frame] = estimate.resampled
        timing[frame] = estimate.timing_ms

    output = arguments.output.expanduser().resolve()
    trajectory = output.with_name(output.stem + ".trajectory.npz")
    metadata = {
        "schema": "dder_particle_filter_v1",
        "observation": str(sequence.path),
        "model": str(artifact.path),
        "config": str(Path(arguments.config).expanduser().resolve()),
        "random_seed": settings.random_seed,
        "particle_count": settings.particle_count,
        "deployment_cable": {
            "length_m": model.parameters.cable_length_m,
            "mass_kg": model.parameters.cable_mass_kg,
            "diameter_m": model.parameters.cable_diameter_m,
            "node_count": model.parameters.node_count,
            "substeps": model.parameters.substeps,
            "constraint_iterations": model.parameters.constraint_iterations,
        },
        "initialized_frame": initialized_frame,
        "elapsed_s": time.perf_counter() - started,
    }
    _atomic_save(
        output,
        metadata_json=np.asarray(json.dumps(metadata, separators=(",", ":"))),
        timestamp_ns=sequence.timestamps_ns,
        source_position=sequence.source_positions,
        positions_m=positions,
        velocities_m_s=velocities,
        posterior_mean_m=posterior_mean,
        node_covariance_m2=covariance,
        weights=weights,
        ess=ess,
        projected_body_residual_px=residual,
        body_updated=body_updated,
        endpoint_count=endpoint_count,
        prediction_only=prediction_only,
        resampled=resampled,
        timing_names=np.asarray(TIMING_NAMES),
        timings_ms=timing,
    )
    save_trajectory(
        trajectory,
        sequence=sequence,
        positions_m=positions,
        velocities_m_s=velocities,
        metrics={
            "trajectory_kind": "causal_dder_particle_filter_map",
            "particle_count": settings.particle_count,
            "initialized_frame": initialized_frame,
        },
        model_path=artifact.path,
    )
    valid_timing = timing[initialized_frame + 1 :, -1]
    valid_timing = valid_timing[np.isfinite(valid_timing)]
    print(
        f"PF saved: {output} | trajectory: {trajectory} | "
        f"median={np.median(valid_timing):.1f}ms p95={np.percentile(valid_timing, 95):.1f}ms",
        flush=True,
    )
    if arguments.svo is not None:
        from ..shared.replay import run as replay

        replay(arguments.svo.expanduser().resolve(), sequence.path, trajectory)
    return output, trajectory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the causal DDER cable particle filter.")
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--svo",
        type=Path,
        default=None,
        help="Optionally replay the saved PF centerline in the ZED point-cloud viewer.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser


def main() -> None:
    try:
        run(build_parser().parse_args())
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"DDER_PF_FAILED: {error}", flush=True)
        raise SystemExit(2) from None
