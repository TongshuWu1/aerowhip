"""Launch the PVA research UI (or a historical headless constant-force rollout)."""

from __future__ import annotations

import argparse
from pathlib import Path

from experimental_data.io import deterministic_npz
from simulator.rollout import load_json, resolve_device, simulate_constant_force


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "config" / "model.json"
DEFAULT_TASK_PATH = ROOT / "config" / "task.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the desktop UI and print a JSON summary.",
    )
    parser.add_argument(
        "--force",
        nargs=3,
        type=float,
        metavar=("FX", "FY", "FZ"),
        help="Constant world-frame point force in newtons; default is hanging hover.",
    )
    parser.add_argument("--duration", type=float, help="Simulation duration in seconds.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--save", type=Path, help="Optional output NPZ trajectory path.")
    arguments = parser.parse_args()
    if not arguments.headless:
        from simulator.gui.app import main as run_gui

        return run_gui()

    arrays, summary = simulate_constant_force(
        load_json(DEFAULT_MODEL_PATH),
        load_json(DEFAULT_TASK_PATH),
        force_world_n=tuple(arguments.force) if arguments.force is not None else None,
        duration_s=arguments.duration,
        device=resolve_device(arguments.device),
    )
    if arguments.save is not None:
        destination = arguments.save.expanduser().resolve()
        deterministic_npz(destination, arrays)
        summary["trajectory_path"] = str(destination)
    import json

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
