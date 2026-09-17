"""Parser for Motive multi-row Take CSV exports."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _number(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def normalize_quaternions_xyzw(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quaternion = np.asarray(values, dtype=np.float64).copy()
    norm = np.linalg.norm(quaternion, axis=1)
    valid = np.isfinite(quaternion).all(axis=1) & (norm > 1.0e-12)
    quaternion[valid] /= norm[valid, None]
    quaternion[~valid] = np.nan
    previous: np.ndarray | None = None
    for index in range(len(quaternion)):
        if not valid[index]:
            continue
        if previous is not None and float(np.dot(previous, quaternion[index])) < 0.0:
            quaternion[index] *= -1.0
        previous = quaternion[index]
    return quaternion, valid


@dataclass(frozen=True, slots=True)
class MotiveTake:
    metadata: dict[str, str]
    frame: np.ndarray
    source_time_s: np.ndarray
    uav_position_m: np.ndarray
    uav_orientation_xyzw: np.ndarray
    uav_valid: np.ndarray
    cable_marker_positions_m: np.ndarray
    cable_marker_valid: np.ndarray
    field_mapping: dict[str, object]


def parse_motive_take(
    path: str | Path,
    *,
    uav_label: str,
    cable_labels: list[str],
) -> MotiveTake:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 8:
        raise ValueError("Motive CSV is missing its multi-row header or data.")
    metadata_row = rows[0]
    metadata = {
        metadata_row[index].strip(): metadata_row[index + 1].strip()
        for index in range(0, len(metadata_row) - 1, 2)
        if metadata_row[index].strip()
    }
    header_index = next(
        (
            index
            for index, row in enumerate(rows)
            if len(row) >= 2
            and row[0].strip() == "Frame"
            and row[1].strip().startswith("Time")
        ),
        None,
    )
    if header_index is None or header_index < 4:
        raise ValueError("Could not locate Motive component header row.")
    width = max(len(row) for row in rows[: header_index + 1])
    header_rows = [
        row + [""] * (width - len(row)) for row in rows[header_index - 4 : header_index + 1]
    ]
    object_types, labels, _identifiers, measurements, components = header_rows

    def columns(label: str, measurement: str, names: tuple[str, ...]) -> list[int]:
        result: list[int] = []
        for component in names:
            matches = [
                index
                for index in range(width)
                if labels[index].strip() == label
                and measurements[index].strip() == measurement
                and components[index].strip() == component
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Motive field {label}/{measurement}/{component} resolved to {matches}."
                )
            result.append(matches[0])
        return result

    quaternion_columns = columns(uav_label, "Rotation", ("X", "Y", "Z", "W"))
    position_columns = columns(uav_label, "Position", ("X", "Y", "Z"))
    marker_columns = {
        label: columns(label, "Position", ("X", "Y", "Z")) for label in cable_labels
    }
    data_rows = [row for row in rows[header_index + 1 :] if row and row[0].strip()]
    frame = np.asarray([int(row[0]) for row in data_rows], dtype=np.int64)
    if len(frame) == 0 or np.any(np.diff(frame) <= 0):
        raise ValueError("Motive frame numbers must be strictly increasing.")
    source_time = np.asarray([_number(row[1]) for row in data_rows], dtype=np.float64)

    def values(indices: list[int]) -> np.ndarray:
        return np.asarray(
            [
                [_number(row[index]) if index < len(row) else np.nan for index in indices]
                for row in data_rows
            ],
            dtype=np.float64,
        )

    uav_position = values(position_columns)
    uav_orientation, quaternion_valid = normalize_quaternions_xyzw(
        values(quaternion_columns)
    )
    uav_valid = np.isfinite(uav_position).all(axis=1) & quaternion_valid
    markers = np.stack([values(marker_columns[label]) for label in cable_labels], axis=1)
    marker_valid = np.isfinite(markers).all(axis=2)
    markers[~marker_valid] = np.nan
    return MotiveTake(
        metadata=metadata,
        frame=frame,
        source_time_s=source_time,
        uav_position_m=uav_position,
        uav_orientation_xyzw=uav_orientation,
        uav_valid=uav_valid,
        cable_marker_positions_m=markers,
        cable_marker_valid=marker_valid,
        field_mapping={
            "uav_label": uav_label,
            "uav_position_columns": position_columns,
            "uav_quaternion_columns_xyzw": quaternion_columns,
            "cable_position_columns_xyz": marker_columns,
            "ignored_unlabeled_columns": int(
                sum(label.startswith("Unlabeled ") for label in labels)
            ),
            "rotation_type": metadata.get("Rotation Type"),
        },
    )
