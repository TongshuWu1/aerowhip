from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def test_recordings_retain_motion_but_no_force_measurement() -> None:
    processed = ROOT / "data/processed_takes"
    take_paths = sorted(processed.glob("*/take.npz"))
    assert len(take_paths) == 8
    with np.load(take_paths[0], allow_pickle=False) as take:
        assert take["uav_position_m"].shape[1:] == (3,)
        assert take["uav_orientation_xyzw"].shape[1:] == (4,)
        assert take["cable_marker_positions_m"].shape[1:] == (10, 3)
        assert "command_acceleration_mps2" in take
        assert not any(
            name in take.files
            for name in ("force_world_n", "motor_thrust_n", "motor_rpm")
        )


def test_dataset_roles_preserve_an_untouched_test() -> None:
    manifest = json.loads(
        (ROOT / "data/dataset_manifest.json").read_text(encoding="utf-8")
    )
    roles = {name: row["role"] for name, row in manifest["takes"].items()}
    assert sum(role == "training" for role in roles.values()) == 5
    assert sum(role == "validation" for role in roles.values()) == 2
    assert roles["fig8vertical_002"] == "untouched_test"
