"""Launch the Isaac Lab drone--cable plant with the correct import order."""

from __future__ import annotations

import argparse
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _link_count(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("link count must be at least two")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 6-DoF force/torque drone with the identified passive cable "
            "in one or many cloned Isaac Lab environments."
        )
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "optitrack_offline" / "models" / "cable_model.json",
        help="Fitted cable-model artifact (default: canonical offline model).",
    )
    parser.add_argument(
        "--drone-config",
        type=Path,
        default=PROJECT_ROOT / "isaac_whip" / "drone_config.json",
        help="Explicit drone physical assumptions JSON.",
    )
    parser.add_argument(
        "--link-count", type=_link_count, default=20, help="Rigid cable links (default: 20)."
    )
    parser.add_argument(
        "--num-envs", type=_positive_int, default=1, help="Parallel cloned plants (default: 1)."
    )
    parser.add_argument(
        "--env-spacing", type=_positive_float, default=2.0, help="Clone grid spacing in metres."
    )
    parser.add_argument(
        "--physics-dt",
        type=_positive_float,
        default=0.002,
        help="Physics time step in seconds (default: 0.002).",
    )
    parser.add_argument(
        "--render-hz", type=_positive_float, default=60.0, help="Requested viewer refresh rate."
    )
    parser.add_argument(
        "--control-hz", type=_positive_float, default=50.0, help="Geometric hover controller rate."
    )
    parser.add_argument(
        "--duration-s",
        type=_nonnegative_float,
        default=None,
        help="Run duration; defaults to continuous in GUI and 8 s headless. Use 0 for continuous.",
    )
    parser.add_argument(
        "--mode",
        choices=("hover", "excite", "free"),
        default="hover",
        help="Hover check, smooth root excitation, or uncontrolled free fall.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "data" / "isaac_whip" / "last_run.json",
        help="Physics/run report path.",
    )
    parser.add_argument(
        "--export-usd",
        type=Path,
        default=None,
        help="Combined USD output; defaults to data/isaac_whip/drone_cable_<N>link.usda.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    args = _parser().parse_args()
    launcher = AppLauncher(args)
    simulation_app = launcher.app
    try:
        # Kit/pxr/Isaac Lab simulation imports are deliberately delayed until
        # after AppLauncher has created the application.
        from isaac_whip.runtime import RuntimeOptions, run

        duration_s = args.duration_s
        if duration_s is None:
            duration_s = 8.0 if args.headless else 0.0
        export_usd = args.export_usd
        if export_usd is None:
            export_usd = (
                PROJECT_ROOT
                / "data"
                / "isaac_whip"
                / f"drone_cable_{args.link_count}link.usda"
            )
        result = run(
            RuntimeOptions(
                model_path=args.model_path.expanduser().resolve(),
                drone_config_path=args.drone_config.expanduser().resolve(),
                link_count=args.link_count,
                num_envs=args.num_envs,
                env_spacing_m=args.env_spacing,
                physics_dt_s=args.physics_dt,
                render_hz=args.render_hz,
                control_hz=args.control_hz,
                duration_s=duration_s,
                mode=args.mode,
                device=args.device,
                headless=args.headless,
                output_json=args.output_json.expanduser().resolve(),
                export_usd=export_usd.expanduser().resolve(),
            ),
            simulation_app,
        )
        return 0 if result["status"] == "PASS" else 2
    except Exception:
        # Immediate Kit cleanup can otherwise swallow the Python traceback on
        # Windows.  Print it before shutting the framework down.
        traceback.print_exc()
        raise
    finally:
        # Isaac Sim 5.1 can spend minutes waiting on unused Replicator cleanup.
        # Immediate cleanup is a documented SimulationApp mode and keeps both
        # finite GUI checks and headless validation deterministic on Windows.
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    raise SystemExit(main())
