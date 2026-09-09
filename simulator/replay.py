"""Durable saved-run library for simulator replays."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from experimental_data.io import atomic_json, deterministic_npz


REQUIRED_TRAJECTORY_FIELDS = (
    "time_s",
    "cable_node_position_world_m",
    "cable_node_velocity_world_m_s",
    "commanded_force_world_n",
    "effective_cable_reaction_on_point_world_n",
)


def _timestamp() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    return (
        now.strftime("%Y-%m-%dT%H%M%S.") + f"{now.microsecond:06d}Z",
        now.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
    )


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", value.strip()).strip("-").lower()
    return cleaned[:48] or "run"


def validate_replay_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    missing = [name for name in REQUIRED_TRAJECTORY_FIELDS if name not in arrays]
    if missing:
        raise ValueError(f"Replay is missing fields: {', '.join(missing)}")
    time = np.asarray(arrays["time_s"])
    positions = np.asarray(arrays["cable_node_position_world_m"])
    velocities = np.asarray(arrays["cable_node_velocity_world_m_s"])
    commands = np.asarray(arrays["commanded_force_world_n"])
    reactions = np.asarray(arrays["effective_cable_reaction_on_point_world_n"])
    if time.ndim != 1 or len(time) < 2:
        raise ValueError("Replay time_s must contain at least two frames.")
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("Replay cable positions must have shape TxNx3.")
    if velocities.shape != positions.shape:
        raise ValueError("Replay cable velocity shape must match cable positions.")
    if positions.shape[0] != len(time):
        raise ValueError("Replay position count must match time_s.")
    expected_step_shape = (len(time) - 1, 3)
    if commands.shape != expected_step_shape or reactions.shape != expected_step_shape:
        raise ValueError("Replay force arrays must have shape (frames - 1)x3.")
    if not all(np.isfinite(np.asarray(arrays[name])).all() for name in REQUIRED_TRAJECTORY_FIELDS):
        raise ValueError("Replay contains non-finite trajectory values.")


def save_replay(
    project_root: Path,
    arrays: Mapping[str, np.ndarray],
    summary: Mapping[str, Any],
    *,
    name: str = "",
) -> Path:
    validate_replay_arrays(arrays)
    stamp, local_time = _timestamp()
    source = str(summary.get("source", "simulation")).replace("_", " ").title()
    display_name = name.strip() or f"{source} — {local_time}"
    directory = project_root / "runs" / "replays" / f"{stamp}_{_slug(display_name)}"
    directory.mkdir(parents=True, exist_ok=False)
    deterministic_npz(
        directory / "trajectory.npz",
        {key: np.asarray(value) for key, value in arrays.items()},
    )
    atomic_json(
        directory / "replay.json",
        {
            "schema": "point_force_saved_replay_v1",
            "display_name": display_name,
            "saved_at_utc": stamp,
            "trajectory": "trajectory.npz",
            "summary": dict(summary),
        },
    )
    return directory


def list_saved_replays(project_root: Path) -> list[dict[str, Any]]:
    root = project_root / "runs" / "replays"
    entries: list[dict[str, Any]] = []
    if not root.is_dir():
        return entries
    for metadata_path in root.glob("*/replay.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            trajectory = metadata_path.parent / str(metadata["trajectory"])
            if metadata.get("schema") != "point_force_saved_replay_v1":
                continue
            if not trajectory.is_file():
                continue
            entries.append(
                {
                    **metadata,
                    "directory": str(metadata_path.parent),
                    "trajectory_path": str(trajectory),
                }
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return sorted(entries, key=lambda item: str(item["saved_at_utc"]), reverse=True)


def load_replay(directory: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    metadata_path = directory / "replay.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != "point_force_saved_replay_v1":
        raise ValueError("Unsupported saved replay schema.")
    trajectory = directory / str(metadata["trajectory"])
    with np.load(trajectory, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    validate_replay_arrays(arrays)
    summary = dict(metadata.get("summary", {}))
    summary["saved_replay_name"] = str(metadata.get("display_name", directory.name))
    summary["saved_replay_directory"] = str(directory)
    return arrays, summary
