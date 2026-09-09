"""File classification and deterministic artifact I/O."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from simulator.artifact_io import replace_with_retry


LOGGER_REQUIRED_FIELDS = frozenset(
    {
        "ros_time",
        "motive_time",
        "natnet_frame",
        "drone_x",
        "drone_qx",
        "cmd_ros_time",
        "cmd_valid",
        "cmd_x",
        "cmd_vx",
        "cmd_ax",
    }
)
MOTIVE_METADATA_FIELDS = frozenset(
    {
        "Format Version",
        "Take Name",
        "Capture Frame Rate",
        "Export Frame Rate",
        "Rotation Type",
        "Length Units",
        "Coordinate Space",
    }
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_csv(path: str | Path) -> str:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", errors="replace") as handle:
        first = handle.readline().rstrip("\r\n")
    cells = [value.strip() for value in first.split(",")]
    cell_set = set(cells)
    if LOGGER_REQUIRED_FIELDS.issubset(cell_set):
        return "logger"
    metadata_keys = set(cells[0::2])
    if MOTIVE_METADATA_FIELDS.issubset(metadata_keys):
        return "motive"
    return "unknown"


def resolve_raw_pair(directory: str | Path) -> tuple[Path, Path]:
    folder = Path(directory)
    classified: dict[str, list[Path]] = {"logger": [], "motive": [], "unknown": []}
    for path in sorted(folder.glob("*.csv")):
        classified[classify_csv(path)].append(path)
    if len(classified["logger"]) != 1 or len(classified["motive"]) != 1:
        detail = ", ".join(
            f"{kind}={[item.name for item in values]}"
            for kind, values in classified.items()
        )
        raise ValueError(
            "Each raw take requires exactly one logger and one Motive CSV; " + detail
        )
    return classified["logger"][0], classified["motive"][0]


def atomic_json(path: str | Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    replace_with_retry(temporary, destination)


def deterministic_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an NPZ with fixed member order/timestamps for reproducible bytes."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as raw:
        temporary = Path(raw.name)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO()
                np.lib.format.write_array(
                    buffer, np.ascontiguousarray(arrays[name]), allow_pickle=False
                )
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
        replace_with_retry(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
