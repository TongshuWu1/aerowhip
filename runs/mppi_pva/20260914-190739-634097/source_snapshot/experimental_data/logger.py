"""Parser for the paired ROS/NatNet/FullState logger CSV."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import LOGGER_REQUIRED_FIELDS


def _number(value: str | None) -> float:
    try:
        return float(value) if value is not None else float("nan")
    except ValueError:
        return float("nan")


@dataclass(frozen=True, slots=True)
class LoggerTake:
    fields: tuple[str, ...]
    ros_time_s: np.ndarray
    motive_time_s: np.ndarray
    natnet_frame: np.ndarray
    uav_position_m: np.ndarray
    uav_orientation_xyzw: np.ndarray
    uav_valid: np.ndarray
    command_source_ros_time_s: np.ndarray
    command_age_source_s: np.ndarray
    command_valid: np.ndarray
    command_position_m: np.ndarray
    command_velocity_m_s: np.ndarray
    command_acceleration_m_s2: np.ndarray
    command_orientation_xyzw: np.ndarray
    command_yaw: np.ndarray
    command_angular_velocity_raw: np.ndarray


def parse_logger(path: str | Path) -> LoggerTake:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        if not LOGGER_REQUIRED_FIELDS.issubset(fields):
            missing = sorted(LOGGER_REQUIRED_FIELDS.difference(fields))
            raise ValueError(f"Logger CSV is missing fields: {missing}")
        rows = list(reader)

    def scalar(name: str) -> np.ndarray:
        return np.asarray([_number(row.get(name)) for row in rows], dtype=np.float64)

    def vector(prefix: str, suffixes: tuple[str, ...]) -> np.ndarray:
        return np.column_stack([scalar(prefix + suffix) for suffix in suffixes])

    ros_time = scalar("ros_time")
    motive_time = scalar("motive_time")
    natnet_float = scalar("natnet_frame")
    if not np.isfinite(natnet_float).all():
        raise ValueError("Logger natnet_frame contains missing/nonfinite values.")
    natnet = natnet_float.astype(np.int64)
    if len(natnet) == 0 or np.any(natnet < 0):
        raise ValueError("Logger natnet_frame must be nonempty and nonnegative.")
    # NatNet/Motive can reset its frame counter while this independent ROS
    # logger remains alive.  Preserve chronological row order and allow that
    # explicit discontinuity, but reject ambiguous duplicate frame IDs because
    # exact Motive-frame matching would otherwise select an arbitrary epoch.
    if len(np.unique(natnet)) != len(natnet):
        raise ValueError("Logger natnet_frame contains duplicate IDs across epochs.")
    return LoggerTake(
        fields=fields,
        ros_time_s=ros_time,
        motive_time_s=motive_time,
        natnet_frame=natnet,
        uav_position_m=vector("drone_", ("x", "y", "z")),
        uav_orientation_xyzw=vector("drone_q", ("x", "y", "z", "w")),
        uav_valid=scalar("drone_valid") == 1.0,
        command_source_ros_time_s=scalar("cmd_ros_time"),
        command_age_source_s=scalar("cmd_age"),
        command_valid=scalar("cmd_valid") == 1.0,
        command_position_m=vector("cmd_", ("x", "y", "z")),
        command_velocity_m_s=vector("cmd_v", ("x", "y", "z")),
        command_acceleration_m_s2=vector("cmd_a", ("x", "y", "z")),
        command_orientation_xyzw=vector("cmd_q", ("x", "y", "z", "w")),
        command_yaw=scalar("cmd_yaw"),
        command_angular_velocity_raw=vector("cmd_omega_", ("x", "y", "z")),
    )
