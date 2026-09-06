"""Named training-run and checkpoint library for point-force PPO."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any
from uuid import uuid4

import torch

from experimental_data.io import atomic_json


CHECKPOINT_LABELS = {
    "terminal.pt": "Terminal",
    "latest.pt": "Latest",
    "best_validation.pt": "Best validation",
}
MANUAL_CHECKPOINT_GLOB = "manual_*.pt"
CURRENT_SUCCESS_DIRECTION_KEY = (
    "maximum_tip_velocity_to_desired_direction_error_deg"
)


def _manual_checkpoint_sort_ns(path: Path) -> int:
    """Return the creation token embedded in a manual checkpoint filename."""

    try:
        return int(path.stem.split("_")[2])
    except (IndexError, ValueError):
        return path.stat().st_mtime_ns


def _checkpoint_candidates(artifact: Path) -> list[tuple[Path, str]]:
    checkpoint_dir = artifact / "checkpoints"
    candidates = [
        (checkpoint_dir / filename, kind)
        for filename, kind in CHECKPOINT_LABELS.items()
    ]
    candidates.extend(
        (path, "Manual snapshot")
        for path in checkpoint_dir.glob(MANUAL_CHECKPOINT_GLOB)
    )
    return candidates


def save_current_policy_snapshot(project_root: Path, artifact: Path) -> Path:
    """Hard-link the newest completed PPO update into the run's library.

    ``latest.pt`` is atomically replaced by the trainer only after an update has
    completed. Creating another hard link in the same directory therefore pins
    exactly one published policy without copying a file while it may be replaced.
    """

    runs_root = (project_root / "runs" / "ppo").resolve()
    target = artifact.resolve()
    if target.parent != runs_root or target == runs_root:
        raise ValueError("Training run is outside the PPO run library.")
    if not target.is_dir():
        raise FileNotFoundError(f"Training run does not exist: {target}")

    try:
        status = json.loads((target / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("Training status is unavailable.") from exc
    state = str(status.get("status", "")).upper()
    if state not in {"STARTING", "RUNNING"}:
        raise RuntimeError("Policy snapshots are available only while PPO is running.")
    if (target / "STOP_REQUESTED").exists():
        raise RuntimeError("Policy snapshots are unavailable while PPO is stopping.")

    source = target / "checkpoints" / "latest.pt"
    if not source.is_file():
        raise FileNotFoundError(
            "The current policy is not available until the first PPO batch completes."
        )

    last_error: BaseException | None = None
    linked: Path | None = None
    for _ in range(8):
        created_ns = time.time_ns()
        linked = source.parent / f".manual_{created_ns}_{uuid4().hex[:8]}.tmp"
        try:
            # os.link resolves the source to one inode atomically. If the trainer
            # replaces latest.pt concurrently, this name still refers to the
            # complete old or complete new publication.
            os.link(source, linked)
        except (FileExistsError, FileNotFoundError, PermissionError) as exc:
            last_error = exc
            time.sleep(0.01)
            continue
        break
    if linked is None or not linked.is_file():
        raise RuntimeError("Could not pin the current PPO policy checkpoint.") from last_error

    try:
        try:
            checkpoint = torch.load(linked, map_location="cpu", weights_only=False)
            if checkpoint.get("schema") != "force_ppo_checkpoint_v1":
                raise ValueError("Current policy has an incompatible checkpoint schema.")
            episodes = int(checkpoint.get("episodes", 0))
            int(checkpoint.get("gradient_updates", 0))
        except (
            AttributeError,
            EOFError,
            OSError,
            pickle.UnpicklingError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError("Current policy checkpoint is invalid.") from exc

        destination = source.parent / (
            f"manual_{episodes:09d}_{created_ns}_{uuid4().hex[:8]}.pt"
        )
        os.replace(linked, destination)
        linked = None
        return destination.resolve()
    finally:
        if linked is not None:
            linked.unlink(missing_ok=True)


def _active_artifact(project_root: Path) -> Path | None:
    pointer = project_root / "runs" / "ppo" / "ACTIVE_RUN.txt"
    if not pointer.is_file():
        return None
    try:
        value = Path(pointer.read_text(encoding="utf-8-sig").strip())
        return (value if value.is_absolute() else project_root / value).resolve()
    except OSError:
        return None


def _run_name(artifact: Path) -> str:
    metadata_path = artifact / "run.json"
    if metadata_path.is_file():
        try:
            value = json.loads(metadata_path.read_text(encoding="utf-8"))
            name = str(value.get("display_name", "")).strip()
            if name:
                return name
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return artifact.name


def _has_current_success_contract(artifact: Path) -> bool:
    try:
        task = json.loads((artifact / "task.json").read_text(encoding="utf-8"))
        success = task["success"]
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return (
        "minimum_directed_tip_speed_m_s" in success
        and CURRENT_SUCCESS_DIRECTION_KEY in success
    )


def list_policy_checkpoints(project_root: Path) -> list[dict[str, Any]]:
    runs_root = project_root / "runs" / "ppo"
    active = _active_artifact(project_root)
    entries: list[dict[str, Any]] = []
    if not runs_root.is_dir():
        return entries
    for artifact in runs_root.iterdir():
        if not artifact.is_dir():
            continue
        status: dict[str, Any] = {}
        try:
            status = json.loads((artifact / "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        for path, kind in _checkpoint_candidates(artifact):
            if not path.is_file():
                continue
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                if checkpoint.get("schema") != "force_ppo_checkpoint_v1":
                    continue
                episodes = int(checkpoint.get("episodes", status.get("episodes", 0)))
                updates = int(checkpoint.get("gradient_updates", 0))
            except (OSError, pickle.UnpicklingError, RuntimeError, ValueError, KeyError):
                continue
            run_name = _run_name(artifact)
            is_active = active == artifact.resolve()
            compatible = _has_current_success_contract(artifact)
            entries.append(
                {
                    "label": (
                        f"{run_name} · {kind} · {episodes:,} episodes"
                        + (" · INCOMPATIBLE" if not compatible else "")
                        + (" · CURRENT" if is_active else "")
                    ),
                    "run_name": run_name,
                    "kind": kind,
                    "episodes": episodes,
                    "gradient_updates": updates,
                    "path": str(path.resolve()),
                    "artifact": str(artifact.resolve()),
                    "status": str(status.get("status", "UNKNOWN")),
                    "active": is_active,
                    "compatible": compatible,
                    "modified_ns": (
                        _manual_checkpoint_sort_ns(path)
                        if kind == "Manual snapshot"
                        else path.stat().st_mtime_ns
                    ),
                }
            )
    return sorted(
        entries,
        key=lambda item: (int(item["modified_ns"]), int(item["episodes"])),
        reverse=True,
    )


def rename_training_run(artifact: Path, display_name: str) -> None:
    name = display_name.strip()
    if not name:
        raise ValueError("Training run name cannot be empty.")
    metadata_path = artifact / "run.json"
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            metadata = {}
    metadata.update(
        {
            "schema": "point_force_training_run_v1",
            "display_name": name,
        }
    )
    atomic_json(metadata_path, metadata)


def write_active_run(project_root: Path, artifact: Path) -> None:
    pointer = project_root / "runs" / "ppo" / "ACTIVE_RUN.txt"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=pointer.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(str(artifact.resolve()) + "\n")
    os.replace(temporary, pointer)


def delete_training_run(project_root: Path, artifact: Path) -> None:
    runs_root = (project_root / "runs" / "ppo").resolve()
    target = artifact.resolve()
    if target.parent != runs_root or target == runs_root:
        raise ValueError("Training run is outside the PPO run library.")
    try:
        status = json.loads((target / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        status = {}
    if str(status.get("status")) in {"STARTING", "RUNNING"}:
        raise RuntimeError("A running training job cannot be deleted.")
    if not target.is_dir():
        raise FileNotFoundError(f"Training run does not exist: {target}")
    was_active = _active_artifact(project_root) == target
    shutil.rmtree(target)
    if was_active:
        remaining = list_policy_checkpoints(project_root)
        pointer = runs_root / "ACTIVE_RUN.txt"
        if remaining:
            write_active_run(project_root, Path(str(remaining[0]["artifact"])))
        else:
            pointer.unlink(missing_ok=True)
