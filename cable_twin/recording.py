"""Lossless SVO recording identities and reproducibility sidecars."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence
import uuid

from .frames import RecordingStatus, SourceDescriptor


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def new_recording_path(directory: Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    identifier = uuid.uuid4().hex[:8]
    path = directory / f"cable_twin_{timestamp}_{identifier}.svo2"
    if path.exists():
        raise FileExistsError(path)
    return path


def git_identity(project_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        revision = run("rev-parse", "HEAD")
        status = run("status", "--porcelain=v1")
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "working_tree_dirty": None}
    return {
        "revision": revision,
        "working_tree_dirty": bool(status),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def load_recording_manifest(svo_path: Path) -> dict[str, Any]:
    """Load the canonical JSON sidecar for one project recording."""

    path = Path(svo_path).expanduser().resolve()
    sidecar = Path(f"{path}.json")
    if not sidecar.is_file():
        raise FileNotFoundError(
            f"Canonical SVO replay requires its recording sidecar: {sidecar}"
        )
    with sidecar.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported recording manifest: {sidecar}")
    if payload.get("svo_path") != path.name:
        raise ValueError(
            f"Manifest SVO identity does not match {path.name}: "
            f"{payload.get('svo_path')!r}"
        )
    return payload


def exact_frame_timestamps_ns(manifest: Mapping[str, Any]) -> tuple[int, ...]:
    """Return the exact live camera timestamp assigned to every SVO frame."""

    recording = manifest.get("recording")
    if not isinstance(recording, Mapping):
        raise ValueError("Recording manifest has no recording section")
    values = recording.get("frame_timestamps_ns")
    if not isinstance(values, list):
        raise ValueError("Recording manifest has no exact per-frame timestamp timeline")
    timestamps = tuple(int(value) for value in values)
    expected_count = int(recording.get("frame_count", -1))
    if len(timestamps) != expected_count:
        raise ValueError(
            "Recording timestamp count does not match frame_count: "
            f"timestamps={len(timestamps)}, frame_count={expected_count}"
        )
    if any(value <= 0 for value in timestamps):
        raise ValueError("Recording timestamps must be positive")
    if any(
        current <= previous
        for previous, current in zip(timestamps, timestamps[1:])
    ):
        raise ValueError("Recording timestamps must be strictly increasing")
    return timestamps


class RecordingManifestWriter:
    """Atomically maintains one sidecar beside an SVO2 recording."""

    def __init__(
        self,
        svo_path: Path,
        source: SourceDescriptor,
        static_metadata: Mapping[str, Any],
        project_root: Path,
    ) -> None:
        self.svo_path = Path(svo_path).resolve()
        self.sidecar_path = Path(f"{self.svo_path}.json")
        if self.sidecar_path.exists():
            raise FileExistsError(self.sidecar_path)
        self._created_utc = datetime.now(timezone.utc).isoformat()
        self._base = {
            "schema_version": 1,
            "recording_id": self.svo_path.stem,
            "svo_path": self.svo_path.name,
            "source": {
                "source_id": source.source_id,
                "kind": source.kind,
                "label": source.label,
                "calibration": asdict(source.calibration),
            },
            "created_utc": self._created_utc,
            "git": git_identity(project_root),
            "configuration": _jsonable(static_metadata),
            "depth_reproducibility": {
                "stored_in_svo": False,
                "method": "recomputed_from_lossless_stereo_on_playback",
                "require_pinned_sdk_and_depth_configuration": True,
            },
        }

    def write(
        self,
        status: RecordingStatus,
        *,
        completed: bool,
        error: str | None = None,
        frame_timestamps_ns: Sequence[int] = (),
    ) -> None:
        timestamps = tuple(int(value) for value in frame_timestamps_ns)
        if len(timestamps) != status.frame_count:
            raise ValueError(
                "Exact timestamp count must match recorded frame count: "
                f"timestamps={len(timestamps)}, frames={status.frame_count}"
            )
        if any(value <= 0 for value in timestamps):
            raise ValueError("Exact frame timestamps must be positive")
        if any(
            current <= previous
            for previous, current in zip(timestamps, timestamps[1:])
        ):
            raise ValueError("Exact frame timestamps must be strictly increasing")
        payload = dict(self._base)
        payload["recording"] = {
            "active": bool(status.active),
            "completed": bool(completed),
            "compression": status.compression,
            "frame_count": int(status.frame_count),
            "first_timestamp_ns": status.first_timestamp_ns,
            "last_timestamp_ns": status.last_timestamp_ns,
            "frame_timestamps_ns": list(timestamps),
            "timestamp_source": "zed_image_timestamp_at_live_grab",
            "error": error,
        }
        payload["updated_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = self.sidecar_path.with_suffix(self.sidecar_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.sidecar_path)
