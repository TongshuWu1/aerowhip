"""Read-only processed-take and take-level role access."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed_takes"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "dataset_manifest.json"
DATASET_ROLES = frozenset({"training", "validation", "untouched_test", "ignore"})
REQUIRED_ARRAYS = frozenset(
    {
        "time_s", "motive_source_time_s", "motive_frame", "command_position_m", "command_velocity_mps",
        "command_acceleration_mps2", "command_orientation_xyzw", "command_yaw",
        "command_angular_velocity", "command_source_ros_time", "command_age_s",
        "command_valid", "uav_position_m", "uav_orientation_xyzw", "uav_valid",
        "cable_marker_positions_m", "cable_marker_valid", "auto_frame_valid",
    }
)


@dataclass(frozen=True, slots=True)
class ProcessedTake:
    take_id: str
    path: Path
    arrays: dict[str, np.ndarray]
    metadata: dict[str, object]
    sync_report: dict[str, object]
    role: str
    enabled: bool
    note: str
    segments: tuple[dict[str, object], ...]
    episode_breaks_s: tuple[float, ...] = ()

    @property
    def duration_s(self) -> float:
        return float(self.arrays["time_s"][-1])

    @property
    def frame_count(self) -> int:
        return int(len(self.arrays["time_s"]))

    @property
    def fit_ready(self) -> bool:
        return bool(self.metadata.get("fit_ready", False))


@dataclass(frozen=True, slots=True)
class Dataset:
    takes: tuple[ProcessedTake, ...]
    manifest: dict[str, object]

    def with_role(self, role: str) -> tuple[ProcessedTake, ...]:
        return tuple(take for take in self.takes if take.enabled and take.role == role)

    @property
    def training(self) -> tuple[ProcessedTake, ...]:
        return self.with_role("training")

    @property
    def validation(self) -> tuple[ProcessedTake, ...]:
        return self.with_role("validation")

    @property
    def untouched_test(self) -> tuple[ProcessedTake, ...]:
        """Protected one-shot evaluation takes; never eligible for fitting."""

        return self.with_role("untouched_test")

    @property
    def fitting_takes(self) -> tuple[ProcessedTake, ...]:
        """The only whole-take roles allowed into fitting/normalization."""

        return self.training + self.validation

    def fitting_gate(self) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if not self.training:
            reasons.append("Assign at least one complete fit-ready take to Training.")
        if not self.validation:
            reasons.append("Assign a physically independent complete take to Validation.")
        for take in self.fitting_takes:
            if not take.fit_ready:
                reasons.append(f"{take.take_id} is NOT_FIT_READY: " + "; ".join(take.metadata.get("warnings", [])))
        return not reasons, reasons


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, object]:
    source = Path(path)
    if not source.exists():
        return {"schema": "aerial_cable_dataset_manifest_v1", "takes": {}}
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != "aerial_cable_dataset_manifest_v1":
        raise ValueError("Unsupported dataset manifest schema.")
    return payload


def load_dataset(
    processed_root: str | Path = DEFAULT_PROCESSED_ROOT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> Dataset:
    root = Path(processed_root)
    manifest = load_manifest(manifest_path)
    decisions = manifest.get("takes", {})
    assert isinstance(decisions, dict)
    takes: list[ProcessedTake] = []
    if not root.exists():
        return Dataset((), manifest)
    for folder in sorted(item for item in root.iterdir() if item.is_dir()):
        paths = [folder / name for name in ("take.npz", "metadata.json", "sync_report.json")]
        if not all(path.exists() for path in paths):
            continue
        with np.load(paths[0], allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
        missing = REQUIRED_ARRAYS.difference(arrays)
        if missing:
            raise ValueError(f"{folder.name} processed take is missing arrays: {sorted(missing)}")
        metadata = json.loads(paths[1].read_text(encoding="utf-8"))
        sync = json.loads(paths[2].read_text(encoding="utf-8"))
        decision = decisions.get(folder.name, {})
        role = str(decision.get("role", "ignore"))
        if role not in DATASET_ROLES:
            raise ValueError(f"Invalid whole-take role {role!r} for {folder.name}.")
        takes.append(
            ProcessedTake(
                take_id=folder.name,
                path=folder,
                arrays=arrays,
                metadata=metadata,
                sync_report=sync,
                role=role,
                enabled=bool(decision.get("enabled", False)),
                note=str(decision.get("note", "")),
                segments=tuple(decision.get("segments", [])),
                episode_breaks_s=tuple(
                    float(value) for value in decision.get("episode_breaks_s", [])
                ),
            )
        )
    return Dataset(tuple(takes), manifest)
