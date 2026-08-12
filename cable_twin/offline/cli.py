from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record image-plane 2D cable motion and fit the elastic rod."
    )
    parser.parse_args()
    from .gui import run_gui

    run_gui()
