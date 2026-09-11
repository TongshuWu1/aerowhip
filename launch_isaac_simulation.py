"""Launch the standalone PhysX process using its separate Isaac Lab interpreter.

Run with the ordinary project Python. ISAACLAB_PYTHON or --isaac-python selects
the dedicated environment. All other arguments go to run_isaac_simulation.py.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--isaac-python", type=Path)
    args, forward = parser.parse_known_args()
    explicit = args.isaac_python or os.environ.get("ISAACLAB_PYTHON")
    if explicit and not Path(explicit).expanduser().is_file():
        raise SystemExit(f"Configured Isaac Lab Python does not exist: {explicit}")
    candidates = (
        [explicit]
        if explicit
        else [
            Path.home() / "env_isaaclab" / "Scripts" / "python.exe",
            Path.home() / "env_isaaclab" / "bin" / "python",
        ]
    )
    python = next(
        (
            Path(p).expanduser().resolve()
            for p in candidates
            if p and Path(p).expanduser().is_file()
        ),
        None,
    )
    if python is None:
        raise SystemExit(
            "Set ISAACLAB_PYTHON or pass --isaac-python pointing to the Isaac Lab environment Python."
        )
    root = Path(__file__).resolve().parent
    command = [str(python), str(root / "run_isaac_simulation.py"), *forward]
    print(
        "Starting separate Isaac Lab process:",
        subprocess.list2cmdline(command),
        flush=True,
    )
    return subprocess.call(command, cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
