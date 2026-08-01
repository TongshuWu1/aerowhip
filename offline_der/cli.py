"""Command-line extraction and free-cable DDER system identification."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import time

import torch

from cable_twin.cable_observation import CableObservationBuilder
from cable_twin.config import DEFAULT_CONFIG_PATH, load_settings
from cable_twin.der import save_dder_model
from cable_twin.perception import PerceptionRuntime
from cable_twin.recording import (
    exact_frame_timestamps_ns,
    load_recording_manifest,
    sha256_file,
)
from cable_twin.replay import validate_replay_manifest
from cable_twin.zed_source import SvoZedSource

from .fitting import IdentificationSettings, identify_free_cable
from .reference import (
    ReferenceAccumulator,
    ReferenceExtractionSettings,
    load_reference_dataset,
    save_reference_dataset,
)


def _cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Offline DDER extraction/identification requires CUDA")
    return torch.device("cuda")


def _selected_cables(value: str) -> tuple[int, ...]:
    return (0, 1) if value == "both" else (int(value),)


def _extract(args: argparse.Namespace) -> None:
    settings = load_settings(Path(args.config).expanduser().resolve())
    device = _cuda_device()
    perception = PerceptionRuntime(settings.pidnet.runtime_config)
    warmup = perception.warmup_1080p()
    print(
        f"PIDNet ready device={perception.device} "
        f"checkpoint={perception.checkpoint_sha256[:12]} "
        f"warmup={warmup.total_ms:.1f}ms"
    )
    reference_settings = ReferenceExtractionSettings(
        node_count=args.nodes,
        minimum_depth_valid_fraction=args.minimum_depth_valid_fraction,
        maximum_missing_arc_m=args.maximum_missing_arc_m,
        projection_iterations=args.projection_iterations,
        projection_tolerance_m=args.projection_tolerance_m,
        route_continuity_sigma_m=args.route_continuity_sigma_m,
        minimum_sequence_frames=args.minimum_sequence_frames,
    )
    output_directory = Path(args.output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    for svo_value in args.svo:
        svo_path = Path(svo_value).expanduser().resolve()
        manifest = load_recording_manifest(svo_path)
        source = SvoZedSource(
            svo_path,
            settings.camera,
            playback_realtime=False,
            exact_timestamps_ns=exact_frame_timestamps_ns(manifest),
        )
        try:
            validate_replay_manifest(
                manifest,
                settings=settings,
                perception=perception,
                source=source,
            )
            accumulator = ReferenceAccumulator(
                reference_settings,
                settings.observation.cable_lengths_m,
                device=device,
            )
            observation_builder = CableObservationBuilder(settings.observation)
            started = time.perf_counter()
            frame_count = 0
            while True:
                frame = source.read(timeout_s=None)
                if frame is None:
                    break
                perception_frame = perception.infer(frame)
                frame_count += 1
                cable_observation = observation_builder.build(
                    perception_frame,
                    source.descriptor.calibration,
                )
                accumulator.add(cable_observation)
                if frame_count == 1 or frame_count % 30 == 0:
                    print(
                        f"extract frame={frame.key.sequence_index} "
                        f"pidnet={perception_frame.timing.total_ms:.1f}ms "
                        f"observation={cable_observation.timing.total_ms:.1f}ms "
                        f"routes={[len(value) for value in cable_observation.routes_by_cable]}"
                    )
            elapsed = time.perf_counter() - started
            metadata = {
                "source_svo": str(svo_path),
                "source_svo_size_bytes": svo_path.stat().st_size,
                "recording_id": manifest.get("recording_id"),
                "recording_manifest": str(Path(f"{svo_path}.json")),
                "recording_manifest_sha256": sha256_file(Path(f"{svo_path}.json")),
                "camera_calibration": asdict(source.descriptor.calibration),
                "pidnet_checkpoint_sha256": perception.checkpoint_sha256,
                "pidnet_runtime_config_sha256": perception.runtime_config_sha256,
                "observation_settings": asdict(settings.observation),
                "extraction_device": str(device),
                "torch_version": torch.__version__,
            }
            for cable_id in _selected_cables(args.cable):
                dataset = accumulator.finish(cable_id, metadata=metadata)
                destination = output_directory / (
                    f"{svo_path.stem}_cable{cable_id}_dder_reference.npz"
                )
                save_reference_dataset(destination, dataset)
                print(
                    f"saved {destination} frames={dataset.frame_count} "
                    f"sequences={len(set(dataset.sequence_id_i32.tolist()))} "
                    f"nodes={dataset.node_count}"
                )
            print(
                f"extracted {frame_count} frames from {svo_path.name} "
                f"in {elapsed:.1f}s"
            )
        finally:
            source.close()


def _fit(args: argparse.Namespace) -> None:
    device = _cuda_device()
    reference_paths = tuple(Path(value).expanduser().resolve() for value in args.reference)
    datasets = tuple(load_reference_dataset(path) for path in reference_paths)
    settings = IdentificationSettings(
        epochs=args.epochs,
        rollout_steps=args.rollout_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        observation_sigma_floor_m=args.observation_sigma_floor_m,
        minimum_shape_excitation_rms_m=args.minimum_shape_excitation_rms_m,
        bending_stiffness_initial_n_m2=args.bending_stiffness_initial_n_m2,
        bending_stiffness_min_n_m2=args.bending_stiffness_min_n_m2,
        bending_stiffness_max_n_m2=args.bending_stiffness_max_n_m2,
        velocity_damping_initial_s_inv=args.velocity_damping_initial_s_inv,
        velocity_damping_max_s_inv=args.velocity_damping_max_s_inv,
        coarse_stiffness_samples=args.coarse_stiffness_samples,
        coarse_damping_samples=args.coarse_damping_samples,
        substeps=args.substeps,
        constraint_iterations=args.constraint_iterations,
        seed=args.seed,
    )
    result = identify_free_cable(
        datasets,
        linear_density_kg_m=args.linear_density_kg_m,
        cable_radius_m=args.cable_radius_m,
        gravity_camera_m_s2=tuple(args.gravity_camera_m_s2),
        settings=settings,
        device=device,
        reference_paths=reference_paths,
    )
    output = Path(args.output).expanduser().resolve()
    save_dder_model(output, result.parameters, identification=result.metadata)
    print(f"saved deployment DDER model: {output}")
    print(
        f"EI={result.parameters.bending_stiffness_n_m2:.8g} N m^2 "
        f"damping={result.parameters.velocity_damping_s_inv:.6g} s^-1 "
        f"rmse={result.metadata['final_observed_rmse_m'] * 1000.0:.3f} mm"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline free-cable DDER reference extraction and identification",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser(
        "extract",
        help="Replay SVO observations into fixed-length reference trajectories",
    )
    extract.add_argument("--svo", type=Path, nargs="+", required=True)
    extract.add_argument("--output-directory", type=Path, default=Path("data/dder_references"))
    extract.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    extract.add_argument("--cable", choices=("0", "1", "both"), default="both")
    extract.add_argument("--nodes", type=int, default=24)
    extract.add_argument("--minimum-depth-valid-fraction", type=float, default=0.65)
    extract.add_argument("--maximum-missing-arc-m", type=float, default=0.075)
    extract.add_argument("--projection-iterations", type=int, default=20)
    extract.add_argument("--projection-tolerance-m", type=float, default=2.0e-4)
    extract.add_argument("--route-continuity-sigma-m", type=float, default=0.025)
    extract.add_argument("--minimum-sequence-frames", type=int, default=8)
    extract.set_defaults(run=_extract)

    fit = subparsers.add_parser(
        "fit",
        help="Identify free-cable bending and damping and save a deployment model",
    )
    fit.add_argument("--reference", type=Path, nargs="+", required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--linear-density-kg-m", type=float, required=True)
    fit.add_argument("--gravity-camera-m-s2", type=float, nargs=3, required=True)
    fit.add_argument("--cable-radius-m", type=float, default=0.0045)
    fit.add_argument("--epochs", type=int, default=80)
    fit.add_argument("--rollout-steps", type=int, default=5)
    fit.add_argument("--batch-size", type=int, default=32)
    fit.add_argument("--learning-rate", type=float, default=0.01)
    fit.add_argument("--observation-sigma-floor-m", type=float, default=0.003)
    fit.add_argument("--minimum-shape-excitation-rms-m", type=float, default=0.005)
    fit.add_argument("--bending-stiffness-initial-n-m2", type=float, default=1.0e-4)
    fit.add_argument("--bending-stiffness-min-n-m2", type=float, default=1.0e-7)
    fit.add_argument("--bending-stiffness-max-n-m2", type=float, default=1.0e-2)
    fit.add_argument("--velocity-damping-initial-s-inv", type=float, default=2.0)
    fit.add_argument("--velocity-damping-max-s-inv", type=float, default=30.0)
    fit.add_argument("--coarse-stiffness-samples", type=int, default=13)
    fit.add_argument("--coarse-damping-samples", type=int, default=9)
    fit.add_argument("--substeps", type=int, default=2)
    fit.add_argument("--constraint-iterations", type=int, default=8)
    fit.add_argument("--seed", type=int, default=1729)
    fit.set_defaults(run=_fit)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    arguments.run(arguments)


if __name__ == "__main__":
    main()
