"""Strict Motive reader for one driven attachment and one free cable tip.

The measurement contract is one rigid-body pivot at the driven material end,
followed by ten labelled moving markers ``c1...c10``.  Marker ``c10`` is the
free tip.  The rigid-body orientation is deliberately not part of the cable
boundary condition.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


_CABLE_LABEL = re.compile(r"^(?P<asset>[^:]+):c(?P<index>[0-9]+)$", re.IGNORECASE)
_UNIT_SCALE_TO_METRES = {
    "meter": 1.0,
    "meters": 1.0,
    "metre": 1.0,
    "metres": 1.0,
    "centimeter": 0.01,
    "centimeters": 0.01,
    "centimetre": 0.01,
    "centimetres": 0.01,
    "millimeter": 0.001,
    "millimeters": 0.001,
    "millimetre": 0.001,
    "millimetres": 0.001,
}
_MOTIVE_TO_PROJECT = np.asarray(
    ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
)


@dataclass(frozen=True, slots=True)
class MotiveCableTake:
    source_path: Path
    take_name: str
    asset_name: str
    capture_rate_hz: float
    export_rate_hz: float
    coordinate_space: str
    source_axis_convention: str
    axis_convention: str
    frame_numbers: np.ndarray
    timestamps_s: np.ndarray
    marker_names: tuple[str, ...]
    positions_m: np.ndarray
    observed: np.ndarray
    attachment_rigid_body_name: str
    attachment_observed: np.ndarray
    attachment_error_m: np.ndarray

    def __post_init__(self) -> None:
        frame_numbers = np.asarray(self.frame_numbers, dtype=np.int64)
        timestamps = np.asarray(self.timestamps_s, dtype=np.float64)
        positions = np.asarray(self.positions_m, dtype=np.float64)
        observed = np.asarray(self.observed, dtype=bool)
        frame_count = len(frame_numbers)
        marker_count = len(self.marker_names)
        if frame_count < 1 or marker_count < 2:
            raise ValueError("A cable take requires frames and at least two vertices.")
        if timestamps.shape != (frame_count,):
            raise ValueError("Motive timestamps have an invalid shape.")
        if positions.shape != (frame_count, marker_count, 3):
            raise ValueError("Cable positions must have shape TxNx3.")
        if observed.shape != (frame_count, marker_count):
            raise ValueError("Cable visibility must have shape TxN.")
        if np.any(np.diff(frame_numbers) <= 0):
            raise ValueError("Motive frame numbers must increase strictly.")
        if np.any(~np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0.0):
            raise ValueError("Motive timestamps must increase strictly.")
        if np.any(np.all(np.isfinite(positions), axis=2) != observed):
            raise ValueError("Each cable vertex must contain either XYZ or no measurement.")
        if self.capture_rate_hz <= 0.0 or self.export_rate_hz <= 0.0:
            raise ValueError("Motive frame rates must be positive.")

        attachment_observed = np.asarray(self.attachment_observed, dtype=bool)
        attachment_error = np.asarray(self.attachment_error_m, dtype=np.float64)
        if attachment_observed.shape != (frame_count,):
            raise ValueError("Attachment visibility must have shape T.")
        if attachment_error.shape != (frame_count,):
            raise ValueError("Attachment rigid-body error must have shape T.")
        if np.any(attachment_observed != observed[:, 0]):
            raise ValueError("Attachment visibility must match material site zero.")
        if np.any(np.isfinite(attachment_error) != attachment_observed):
            raise ValueError(
                "Each observed attachment requires Motive Error Per Marker."
            )
        if np.any(attachment_error[attachment_observed] < 0.0):
            raise ValueError("Attachment rigid-body error cannot be negative.")
        if not self.attachment_rigid_body_name.strip():
            raise ValueError("The attachment rigid body must have a name.")

        for value in (
            frame_numbers,
            timestamps,
            positions,
            observed,
            attachment_observed,
            attachment_error,
        ):
            value.setflags(write=False)
        object.__setattr__(self, "frame_numbers", frame_numbers)
        object.__setattr__(self, "timestamps_s", timestamps)
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "attachment_observed", attachment_observed)
        object.__setattr__(self, "attachment_error_m", attachment_error)

    @property
    def frame_count(self) -> int:
        return len(self.frame_numbers)

    @property
    def marker_count(self) -> int:
        """Number of directly observed rod vertices, including the attachment."""

        return len(self.marker_names)

    @property
    def moving_marker_count(self) -> int:
        return self.marker_count - 1

    @property
    def has_attachment(self) -> bool:
        return True

    @property
    def duration_s(self) -> float:
        return float(self.timestamps_s[-1] - self.timestamps_s[0])

    @property
    def max_marker_displacement_m(self) -> float:
        largest = 0.0
        for marker_index in range(self.marker_count):
            positions = self.positions_m[self.observed[:, marker_index], marker_index]
            if len(positions) > 1:
                largest = max(
                    largest,
                    float(np.max(np.linalg.norm(positions - positions[0], axis=1))),
                )
        return largest


def _metadata(row: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for index in range(0, len(row) - 1, 2):
        key = row[index].strip()
        if key:
            values[key] = row[index + 1].strip()
    return values


def _data_header_index(rows: list[list[str]]) -> int:
    for index, row in enumerate(rows):
        if len(row) >= 2 and row[0].strip().casefold() == "frame":
            if row[1].strip().casefold().startswith("time"):
                return index
    raise ValueError("The Motive Frame/Time header was not found.")


def _float(value: str, *, field: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"Invalid {field}: {value!r}") from error
    if not np.isfinite(result):
        raise ValueError(f"Non-finite {field}: {value!r}")
    return result


def _motive_y_up_to_project_z_up(positions_m: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions_m, dtype=np.float64)
    return np.einsum("ij,...j->...i", _MOTIVE_TO_PROJECT, positions)


def _read_vector(
    row: list[str],
    columns: dict[str, int],
    axes: tuple[str, ...],
    *,
    field: str,
    scale: float = 1.0,
) -> np.ndarray | None:
    values = [row[columns[axis]].strip() for axis in axes]
    present = [bool(value) for value in values]
    if any(present) and not all(present):
        raise ValueError(f"Partial {field} measurement.")
    if not all(present):
        return None
    return np.asarray([_float(value, field=field) * scale for value in values])


def load_motive_cable_csv(
    path: str | Path,
    *,
    asset_name: str | None = None,
) -> MotiveCableTake:
    """Load one attachment pivot plus moving markers c1...c10."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Motive CSV does not exist: {source}")
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if len(rows) < 8:
        raise ValueError("Motive CSV is incomplete.")

    metadata = _metadata(rows[0])
    header_index = _data_header_index(rows)
    if header_index < 4:
        raise ValueError("Motive trajectory header is incomplete.")
    kinds = rows[header_index - 4]
    names = rows[header_index - 3]
    properties = rows[header_index - 1]
    axes = rows[header_index]
    width = max(len(kinds), len(names), len(properties), len(axes))
    for row in (kinds, names, properties, axes):
        row += [""] * (width - len(row))

    marker_assets: dict[str, dict[int, dict[str, int]]] = {}
    canonical_asset: dict[str, str] = {}
    rigid_bodies: dict[str, dict[str, dict[str, int]]] = {}
    rigid_body_error_columns: dict[str, int] = {}
    for column in range(2, width):
        name = names[column].strip()
        kind = kinds[column].strip().casefold().replace(" ", "")
        prop = properties[column].strip().casefold()
        axis = axes[column].strip().casefold()
        match = _CABLE_LABEL.fullmatch(name)
        if match is not None and prop == "position" and axis in {"x", "y", "z"}:
            asset = match.group("asset")
            key = asset.casefold()
            marker_index = int(match.group("index"))
            marker_axes = marker_assets.setdefault(key, {}).setdefault(marker_index, {})
            if axis in marker_axes:
                raise ValueError(f"Duplicate {asset}:c{marker_index} {axis.upper()} column.")
            marker_axes[axis] = column
            canonical_asset.setdefault(key, asset)
        if kind == "rigidbody" and name and prop == "position":
            if axis not in {"x", "y", "z"}:
                continue
            columns = rigid_bodies.setdefault(name, {}).setdefault(prop, {})
            if axis in columns:
                raise ValueError(f"Duplicate rigid body {name!r} {prop} {axis.upper()} column.")
            columns[axis] = column
        if kind == "rigidbody" and name and prop == "error per marker":
            if name in rigid_body_error_columns:
                raise ValueError(f"Duplicate rigid body {name!r} Error Per Marker column.")
            rigid_body_error_columns[name] = column

    if asset_name is None:
        if len(marker_assets) != 1:
            choices = ", ".join(sorted(canonical_asset.values())) or "none"
            raise ValueError(f"Expected one cable marker asset; found {choices}.")
        asset_key = next(iter(marker_assets))
    else:
        asset_key = asset_name.casefold()
        if asset_key not in marker_assets:
            choices = ", ".join(sorted(canonical_asset.values())) or "none"
            raise ValueError(f"Cable asset {asset_name!r} was not found; found {choices}.")
    marker_columns = marker_assets[asset_key]
    marker_indices = sorted(marker_columns)
    for marker_index, marker_axes in marker_columns.items():
        if set(marker_axes) != {"x", "y", "z"}:
            raise ValueError(f"Cable marker c{marker_index} lacks complete XYZ columns.")

    units = metadata.get("Length Units", "").casefold()
    if units not in _UNIT_SCALE_TO_METRES:
        raise ValueError(f"Unsupported or missing Motive length unit: {units or 'missing'}.")
    scale = _UNIT_SCALE_TO_METRES[units]
    coordinate_space = metadata.get("Coordinate Space", "")
    if coordinate_space.casefold() != "global":
        raise ValueError("Motive CSV must use global/world coordinates.")

    if marker_indices != list(range(1, 11)):
        raise ValueError(
            "One-attachment cable takes must contain exactly c1...c10; c10 is "
            "the free-tip marker. Legacy c0...c10 or c1...c9 exports are not supported."
        )
    if len(rigid_bodies) != 1:
        legacy = " The former two-holder format is not compatible." if len(rigid_bodies) == 2 else ""
        raise ValueError(
            "One-attachment cable takes require exactly one exported rigid-body "
            f"attachment pivot; found {len(rigid_bodies)}.{legacy}"
        )
    attachment_name = next(iter(rigid_bodies))
    attachment_fields = rigid_bodies[attachment_name]
    if set(attachment_fields.get("position", {})) != {"x", "y", "z"}:
        raise ValueError(
            f"Attachment rigid body {attachment_name!r} lacks complete XYZ position columns."
        )
    if attachment_name not in rigid_body_error_columns:
        raise ValueError(
            f"Attachment rigid body {attachment_name!r} lacks Error Per Marker. "
            "Re-export with Quality Statistics enabled."
        )

    frame_numbers: list[int] = []
    timestamps: list[float] = []
    marker_values: list[np.ndarray] = []
    attachment_positions: list[np.ndarray] = []
    attachment_errors: list[float] = []
    for row_number, row in enumerate(rows[header_index + 1 :], start=header_index + 2):
        if not row or not row[0].strip():
            continue
        row += [""] * max(0, width - len(row))
        try:
            frame = int(row[0])
        except ValueError as error:
            raise ValueError(f"Invalid Motive frame number on CSV row {row_number}.") from error
        timestamp = _float(row[1], field=f"timestamp on CSV row {row_number}")
        markers = np.full((len(marker_indices), 3), np.nan, dtype=np.float64)
        for output_index, marker_index in enumerate(marker_indices):
            value = _read_vector(
                row,
                marker_columns[marker_index],
                ("x", "y", "z"),
                field=f"c{marker_index} position on CSV row {row_number}",
                scale=scale,
            )
            if value is not None:
                markers[output_index] = value
        attachment_position = _read_vector(
            row,
            attachment_fields["position"],
            ("x", "y", "z"),
            field=f"{attachment_name} position on CSV row {row_number}",
            scale=scale,
        )
        attachment_error_text = row[rigid_body_error_columns[attachment_name]].strip()
        if attachment_position is not None and not attachment_error_text:
            raise ValueError(
                f"Attachment rigid body {attachment_name!r} has a position without "
                f"Error Per Marker on CSV row {row_number}."
            )
        if attachment_position is None and attachment_error_text:
            raise ValueError(
                f"Attachment rigid body {attachment_name!r} has Error Per Marker "
                f"without a position on CSV row {row_number}."
            )
        if attachment_position is None:
            attachment_position = np.full(3, np.nan, dtype=np.float64)
            attachment_error = float("nan")
        else:
            attachment_error = _float(
                attachment_error_text,
                field=f"{attachment_name} Error Per Marker on CSV row {row_number}",
            ) * scale
        frame_numbers.append(frame)
        timestamps.append(timestamp)
        marker_values.append(markers)
        attachment_positions.append(attachment_position)
        attachment_errors.append(attachment_error)

    if not marker_values:
        raise ValueError("Motive CSV contains no data rows.")
    source_markers = np.stack(marker_values)
    marker_observed = np.all(np.isfinite(source_markers), axis=2)
    project_markers = _motive_y_up_to_project_z_up(source_markers)

    source_attachment = np.stack(attachment_positions)
    attachment_observed = np.all(np.isfinite(source_attachment), axis=1)
    project_attachment = _motive_y_up_to_project_z_up(source_attachment)
    positions = np.concatenate((project_attachment[:, None], project_markers), axis=1)
    observed = np.concatenate((attachment_observed[:, None], marker_observed), axis=1)
    marker_names = ("Attachment", *(f"c{index}" for index in marker_indices))
    attachment_error = np.asarray(attachment_errors, dtype=np.float64)

    return MotiveCableTake(
        source_path=source,
        take_name=metadata.get("Take Name", source.stem),
        asset_name=canonical_asset[asset_key],
        capture_rate_hz=_float(metadata.get("Capture Frame Rate", ""), field="capture rate"),
        export_rate_hz=_float(metadata.get("Export Frame Rate", ""), field="export rate"),
        coordinate_space=coordinate_space,
        source_axis_convention="Motive right-handed Y-up",
        axis_convention="Project right-handed Z-up",
        frame_numbers=np.asarray(frame_numbers, dtype=np.int64),
        timestamps_s=np.asarray(timestamps, dtype=np.float64),
        marker_names=marker_names,
        positions_m=positions,
        observed=observed,
        attachment_rigid_body_name=attachment_name,
        attachment_observed=attachment_observed,
        attachment_error_m=attachment_error,
    )
