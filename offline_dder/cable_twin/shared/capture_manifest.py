from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid
from typing import Any, Mapping

from .zed_source import ZedStereoSource


CAPTURE_MANIFEST_SCHEMA = "zed_stereo_capture_v3"


def capture_manifest_path(svo_path: str | Path) -> Path:
    source = Path(svo_path).expanduser().resolve()
    if source.suffix.lower() != ".svo2":
        raise ValueError("Capture manifests require an .svo2 source path.")
    return source.with_suffix(".capture.json")


def build_capture_manifest(
    source: ZedStereoSource,
    svo_path: str | Path,
    frame_count: int,
    *,
    cable_identity: int,
) -> dict[str, Any]:
    descriptor = source.descriptor
    calibration = descriptor.calibration
    settings = source.capture_settings
    if descriptor.kind != "live" or settings is None:
        raise ValueError("A capture manifest can only be built from a live ZED source.")
    if int(frame_count) < 1:
        raise ValueError("A capture manifest requires at least one recorded frame.")
    if int(cable_identity) not in (1, 2):
        raise ValueError("Cable identity must be 1 or 2.")
    path = Path(svo_path).expanduser().resolve()
    return {
        "schema": CAPTURE_MANIFEST_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "svo_filename": path.name,
        "frame_count": int(frame_count),
        "experiment": {
            "cable_identity": int(cable_identity),
        },
        "recording": {
            "container": "SVO2",
            "compression": "H265_LOSSLESS",
            "timestamp_source": "ZED_IMAGE",
        },
        "camera": {
            "model": calibration.camera_model,
            "serial_number": calibration.serial_number,
            "width_px": calibration.width_px,
            "height_px": calibration.height_px,
            "fps": calibration.fps,
        },
        "capture_settings": asdict(settings),
    }


def write_capture_manifest(
    svo_path: str | Path,
    manifest: Mapping[str, Any],
) -> Path:
    target = capture_manifest_path(svo_path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite capture manifest: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(dict(manifest), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            raise FileExistsError(
                f"Refusing to overwrite capture manifest: {target}"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_capture_manifest(svo_path: str | Path) -> dict[str, Any] | None:
    source = Path(svo_path).expanduser().resolve()
    path = capture_manifest_path(source)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Capture manifest is unreadable: {path}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != CAPTURE_MANIFEST_SCHEMA
        or payload.get("svo_filename") != source.name
    ):
        raise ValueError(f"Capture manifest does not match its SVO2 source: {path}")
    return payload
