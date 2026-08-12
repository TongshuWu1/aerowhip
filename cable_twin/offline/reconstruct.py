"""Extract, constrain, and display one image-plane cable trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .analyze import analyze_svo
from .config import DEFAULT_CONFIG_PATH
from .optimize import prepare_recording
from .planar_data import validate_planar_observation
from .replay import run as replay
from ..shared.observation_data import trajectory_path


def _current(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            validate_planar_observation(data)
    except (OSError, ValueError):
        return False
    return True


def run(arguments: argparse.Namespace) -> Path:
    svo = arguments.svo.expanduser().resolve()
    observation = arguments.observation.expanduser().resolve()
    if not svo.is_file():
        raise FileNotFoundError(f"SVO2 recording not found: {svo}")
    if not _current(observation):
        analyze_svo(
            svo,
            observation,
            config_path=arguments.config,
            cable_identity=arguments.cable,
            replace_existing=observation.exists(),
            headless=True,
        )
    prepared = prepare_recording(
        svo,
        observation,
        config_path=arguments.config,
    )
    print(
        f"PIDNet node observations: valid={np.count_nonzero(prepared.valid)}/"
        f"{len(prepared.valid)} DDER_initialization_shift="
        f"{1000.0 * prepared.initialization_shift_m:.3f}mm",
        flush=True,
    )
    replay(svo, observation)
    return trajectory_path(prepared.observation_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process one image-plane 2D cable recording.")
    parser.add_argument("--svo", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--cable", type=int, choices=(1, 2), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser


def main() -> None:
    try:
        run(build_parser().parse_args())
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"2D_PROCESSING_FAILED: {error}", flush=True)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
