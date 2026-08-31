"""Run the frozen Milestone 3C decomposed full-episode identification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from fitting.config import DEFAULT_PATH as DEFAULT_FIT_CONFIG, load_fit_configuration
from fitting.dataset import DEFAULT_MANIFEST, DEFAULT_PROCESSED_ROOT, load_dataset
from fitting.decomposed import (
    run_decomposed_identification,
    run_remeasured_geometry_pre_mppi,
)
from simulator.parameters import SimulatorSettings


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("decomposed-full", "cable-refit-pre-mppi"),
        default="decomposed-full",
        help="Select the frozen production workflow to execute.",
    )
    parser.add_argument(
        "--sim-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "default.json",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume a stopped decomposed-fit directory without rerunning completed UAV fitting.",
    )
    parser.add_argument("--fit-config", type=Path, default=DEFAULT_FIT_CONFIG)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--wall-clock-limit-s",
        type=float,
        default=None,
        help="Hard total limit; completed stages remain saved if it expires.",
    )
    arguments = parser.parse_args()
    results_root = PROJECT_ROOT / "data" / "fit_results_decomposed"
    before = set(results_root.iterdir()) if results_root.exists() else set()
    try:
        dataset = load_dataset(arguments.processed_root, arguments.manifest)
        fit_config = load_fit_configuration(arguments.fit_config)
        settings = SimulatorSettings.load(arguments.sim_config)
        if arguments.stage == "cable-refit-pre-mppi":
            if arguments.resume is not None:
                parser.error("The PRE_MPPI cable-refit workflow does not accept --resume.")
            output = run_remeasured_geometry_pre_mppi(
                dataset,
                fit_config,
                settings,
                root=PROJECT_ROOT,
                wall_clock_limit_s=(
                    1200.0
                    if arguments.wall_clock_limit_s is None
                    else arguments.wall_clock_limit_s
                ),
            )
        else:
            output = run_decomposed_identification(
                dataset,
                fit_config,
                settings,
                root=PROJECT_ROOT,
                wall_clock_limit_s=(
                    5400.0
                    if arguments.wall_clock_limit_s is None
                    else arguments.wall_clock_limit_s
                ),
                resume_output=arguments.resume,
            )
    except Exception as error:
        after = set(results_root.iterdir()) if results_root.exists() else set()
        created = sorted(after - before, key=lambda path: path.stat().st_mtime)
        if arguments.resume is not None:
            created = [arguments.resume.resolve()]
        if created:
            status_path = created[-1] / "execution_status.json"
            status = (
                json.loads(status_path.read_text(encoding="utf-8"))
                if status_path.exists()
                else {}
            )
            status.update(
                {
                    "status": "stopped_with_error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            status_path.write_text(
                json.dumps(status, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        raise
    print(f"Milestone 3C workflow complete: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
