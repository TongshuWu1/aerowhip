"""Separate Isaac Lab / PhysX drone+cable runner (use the Isaac Lab Python)."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config/isaac_physx/rig_153g_paracord.json",
    )
    parser.add_argument(
        "--trajectory",
        choices=["hover", "circle", "figure8", "vertical8"],
        default="figure8",
    )
    parser.add_argument("--period", type=float)
    parser.add_argument("--radius", type=float)
    parser.add_argument("--hold", type=float)
    parser.add_argument("--cycles", type=int)
    parser.add_argument(
        "--duration",
        type=float,
        help="Optional shorter test duration in simulation seconds",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--exit-after-trajectory", action="store_true")
    parser.add_argument(
        "--screenshot",
        action="store_true",
        help="Save a viewport image near the end (enables rendering)",
    )
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    # One small articulation is substantially faster on CPU; RTX rendering
    # remains on the GPU. --device cuda:0 retains the GPU PhysX option.
    parser.set_defaults(device="cpu")
    args = parser.parse_args()
    if args.screenshot:
        args.enable_cameras = True
    launcher = AppLauncher(args, fast_shutdown=True)
    app = launcher.app
    try:
        from isaac_simulation.runner import run

        return run(app, args)
    except Exception:
        import traceback
        import omni.kit.app

        traceback.print_exc()
        omni.kit.app.get_app().post_quit(1)
        raise
    finally:
        from isaaclab.sim import SimulationContext

        # Release Lab's STOP subscription before closing the Kit stage. This
        # installation otherwise keeps rendering a stopped standalone scene.
        SimulationContext.clear_instance()
        print("[PhysX] Closing Isaac application.", flush=True)
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
