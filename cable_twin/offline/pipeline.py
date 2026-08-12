"""One-command planar PIDNet analysis followed by EI/Cb fitting."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .analyze import analyze_svo
from .config import DEFAULT_CONFIG_PATH, load_settings
from .planar_data import validate_planar_observation
from ..shared.observation_data import trajectory_path


def _valid_archive(path: Path, *, cable_length_m: float) -> bool:
    if not path.is_file():
        return False
    with np.load(path, allow_pickle=False) as data:
        metadata = validate_planar_observation(data)
    archived_length = float(metadata["image_plane_mapping"]["cable_length_m"])
    return bool(np.isclose(archived_length, cable_length_m, rtol=1.0e-9, atol=1.0e-12))


def run(arguments: argparse.Namespace) -> Path:
    settings = load_settings(arguments.config)
    svo_paths = tuple(Path(value).expanduser().resolve() for value in arguments.svo)
    observations = tuple(Path(value).expanduser().resolve() for value in arguments.observation)
    if len(svo_paths) != len(observations) or not svo_paths:
        raise ValueError("Provide one planar observation path for every SVO2.")
    for svo, observation in zip(svo_paths, observations, strict=True):
        if not svo.is_file():
            raise FileNotFoundError(f"SVO2 recording not found: {svo}")
        needs_analysis = True
        if observation.is_file():
            try:
                needs_analysis = not _valid_archive(
                    observation,
                    cable_length_m=settings.cable.length_m,
                )
            except ValueError:
                needs_analysis = True
        if needs_analysis:
            trajectory_path(observation).unlink(missing_ok=True)
            print(f"Extracting image-plane PIDNet route: {svo.name}", flush=True)
            cable_identity = int(svo.stem.rsplit("_cable", 1)[-1])
            analyze_svo(
                svo,
                observation,
                config_path=arguments.config,
                cable_identity=cable_identity,
                headless=True,
                replace_existing=observation.exists(),
            )
    from .optimize import run as run_fit

    return run_fit(
        argparse.Namespace(
            svo=svo_paths,
            observation=observations,
            output=arguments.output,
            config=arguments.config,
            iterations=arguments.iterations,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze planar SVO2 data and fit EI/Cb.")
    parser.add_argument("--svo", type=Path, nargs="+", required=True)
    parser.add_argument("--observation", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--iterations", type=int, default=None)
    return parser


def main() -> None:
    try:
        run(build_parser().parse_args())
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"CABLE_MODEL_IDENTIFICATION_FAILED: {error}", flush=True)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
