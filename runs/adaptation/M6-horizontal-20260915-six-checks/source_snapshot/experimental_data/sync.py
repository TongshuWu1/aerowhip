"""Robust affine clock fitting and zero-order-held command reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .logger import LoggerTake
from .motive import MotiveTake, normalize_quaternions_xyzw


@dataclass(frozen=True, slots=True)
class AffineClock:
    slope: float
    offset: float
    rms_residual_s: float
    maximum_residual_s: float
    sample_count: int
    rejected_count: int

    def map(self, value: np.ndarray) -> np.ndarray:
        return self.slope * np.asarray(value, dtype=np.float64) + self.offset

    def as_dict(self) -> dict[str, float | int]:
        return {
            "slope": self.slope,
            "offset": self.offset,
            "rms_residual_s": self.rms_residual_s,
            "maximum_residual_s": self.maximum_residual_s,
            "sample_count": self.sample_count,
            "rejected_count": self.rejected_count,
        }


def robust_affine_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    sigma: float,
    maximum_iterations: int,
    minimum_samples: int,
) -> AffineClock:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(finite)) < minimum_samples:
        raise ValueError("Too few finite samples for clock synchronization.")
    active = finite.copy()
    for _ in range(maximum_iterations):
        xv, yv = x[active], y[active]
        x_center, y_center = float(np.mean(xv)), float(np.mean(yv))
        denominator = float(np.dot(xv - x_center, xv - x_center))
        if denominator <= 0.0:
            raise ValueError("Clock samples have no time span.")
        slope = float(np.dot(xv - x_center, yv - y_center) / denominator)
        offset = y_center - slope * x_center
        residual = y - (slope * x + offset)
        centered = residual[active] - np.median(residual[active])
        mad = float(np.median(np.abs(centered)))
        robust_scale = max(1.4826 * mad, 1.0e-9)
        next_active = finite & (np.abs(residual - np.median(residual[active])) <= sigma * robust_scale)
        if int(np.count_nonzero(next_active)) < minimum_samples:
            break
        if np.array_equal(next_active, active):
            active = next_active
            break
        active = next_active
    residual = y[active] - (slope * x[active] + offset)
    return AffineClock(
        slope=slope,
        offset=offset,
        rms_residual_s=float(np.sqrt(np.mean(np.square(residual)))),
        maximum_residual_s=float(np.max(np.abs(residual))),
        sample_count=int(np.count_nonzero(active)),
        rejected_count=int(np.count_nonzero(finite) - np.count_nonzero(active)),
    )


def matched_frame_indices(
    logger: LoggerTake, motive: MotiveTake
) -> tuple[np.ndarray, np.ndarray]:
    logger_lookup = {int(frame): index for index, frame in enumerate(logger.natnet_frame)}
    pairs = [
        (logger_lookup[int(frame)], motive_index)
        for motive_index, frame in enumerate(motive.frame)
        if int(frame) in logger_lookup
    ]
    if not pairs:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return (
        np.asarray([pair[0] for pair in pairs], dtype=np.int64),
        np.asarray([pair[1] for pair in pairs], dtype=np.int64),
    )


def missing_runs(values: np.ndarray) -> list[dict[str, int]]:
    values = np.asarray(values, dtype=np.int64)
    if len(values) < 2:
        return []
    runs: list[dict[str, int]] = []
    for left, right in zip(values[:-1], values[1:], strict=True):
        if right > left + 1:
            runs.append(
                {"after_frame": int(left), "before_frame": int(right), "count": int(right-left-1)}
            )
    return runs


def frame_resets(values: np.ndarray, motive_time_s: np.ndarray) -> list[dict[str, float | int]]:
    """Return chronological NatNet counter resets without altering evidence."""

    values = np.asarray(values, dtype=np.int64)
    motive_time_s = np.asarray(motive_time_s, dtype=np.float64)
    return [
        {
            "row_before": int(index),
            "frame_before": int(values[index]),
            "frame_after": int(values[index + 1]),
            "motive_time_before_s": float(motive_time_s[index]),
            "motive_time_after_s": float(motive_time_s[index + 1]),
        }
        for index in np.flatnonzero(np.diff(values) <= 0)
    ]


def build_clocks(
    logger: LoggerTake,
    motive: MotiveTake,
    config: dict[str, object],
) -> tuple[AffineClock, AffineClock, np.ndarray, np.ndarray]:
    clock = config["clock"]
    assert isinstance(clock, dict)
    kwargs = {
        "sigma": float(clock["robust_sigma"]),
        "maximum_iterations": int(clock["maximum_iterations"]),
        "minimum_samples": int(clock["minimum_samples"]),
    }
    ros_to_logger = robust_affine_fit(
        logger.ros_time_s, logger.motive_time_s, **kwargs
    )
    logger_index, motive_index = matched_frame_indices(logger, motive)
    logger_to_take = robust_affine_fit(
        logger.motive_time_s[logger_index], motive.source_time_s[motive_index], **kwargs
    )
    return ros_to_logger, logger_to_take, logger_index, motive_index


def reconstruct_commands(
    logger: LoggerTake,
    motive: MotiveTake,
    ros_to_logger: AffineClock,
    logger_to_take: AffineClock,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    candidate = logger.command_valid & np.isfinite(logger.command_source_ros_time_s)
    candidate &= np.isfinite(logger.command_position_m).all(axis=1)
    candidate &= np.isfinite(logger.command_velocity_m_s).all(axis=1)
    candidate &= np.isfinite(logger.command_acceleration_m_s2).all(axis=1)
    candidate &= np.isfinite(logger.command_orientation_xyzw).all(axis=1)
    candidate &= np.isfinite(logger.command_yaw)
    candidate &= np.isfinite(logger.command_angular_velocity_raw).all(axis=1)
    indices = np.flatnonzero(candidate)
    event_by_time: dict[float, int] = {}
    conflicts = 0
    fields = (
        logger.command_position_m,
        logger.command_velocity_m_s,
        logger.command_acceleration_m_s2,
        logger.command_orientation_xyzw,
        logger.command_yaw[:, None],
        logger.command_angular_velocity_raw,
    )
    for index in indices:
        key = float(logger.command_source_ros_time_s[index])
        previous = event_by_time.get(key)
        if previous is not None and any(
            not np.allclose(value[previous], value[index], rtol=0.0, atol=1.0e-12)
            for value in fields
        ):
            conflicts += 1
        event_by_time[key] = int(index)
    event_ros = np.asarray(sorted(event_by_time), dtype=np.float64)
    event_indices = np.asarray([event_by_time[float(value)] for value in event_ros], dtype=np.int64)
    event_take = logger_to_take.map(ros_to_logger.map(event_ros))
    observation_time = motive.source_time_s
    selected = np.searchsorted(event_take, observation_time, side="right") - 1
    has_event = selected >= 0
    safe_selected = np.clip(selected, 0, max(len(event_indices) - 1, 0))
    selected_logger = event_indices[safe_selected] if len(event_indices) else np.zeros(len(selected), dtype=np.int64)

    frame_lookup = {int(frame): index for index, frame in enumerate(logger.natnet_frame)}
    availability = np.asarray(
        [
            logger.command_valid[frame_lookup[int(frame)]]
            if int(frame) in frame_lookup
            else False
            for frame in motive.frame
        ],
        dtype=bool,
    )
    valid = has_event & availability

    def select(value: np.ndarray, width: int | None = None) -> np.ndarray:
        shape = (len(observation_time),) if width is None else (len(observation_time), width)
        output = np.full(shape, np.nan, dtype=np.float64)
        if len(event_indices):
            output[valid] = value[selected_logger[valid]]
        return output

    orientation, orientation_valid = normalize_quaternions_xyzw(
        select(logger.command_orientation_xyzw, 4)
    )
    valid &= orientation_valid
    command_source = select(logger.command_source_ros_time_s)
    command_age = np.full(len(observation_time), np.nan, dtype=np.float64)
    command_age[valid] = observation_time[valid] - event_take[selected[valid]]
    arrays = {
        "command_position_m": select(logger.command_position_m, 3),
        "command_velocity_mps": select(logger.command_velocity_m_s, 3),
        "command_acceleration_mps2": select(logger.command_acceleration_m_s2, 3),
        "command_orientation_xyzw": orientation,
        "command_yaw": select(logger.command_yaw),
        "command_angular_velocity": select(logger.command_angular_velocity_raw, 3),
        "command_source_ros_time": command_source,
        "command_age_s": command_age,
        "command_valid": valid,
    }
    if len(event_take) > 1:
        rate = (len(event_take) - 1) / (event_take[-1] - event_take[0])
    else:
        rate = 0.0
    report = {
        "unique_command_count": int(len(event_take)),
        "duplicate_value_conflicts": int(conflicts),
        "command_event_rate_hz": float(rate),
        "command_coverage_fraction": float(np.mean(valid)),
        "first_command_take_time_s": None if not len(event_take) else float(event_take[0]),
        "last_command_take_time_s": None if not len(event_take) else float(event_take[-1]),
        "reconstruction": "zero_order_hold_latest_unique_valid_cmd_ros_time",
        "availability": "logger cmd_valid at matched NatNet/Motive frame",
        "raw_command_diagnostics_no_unit_inference": {
            "horizontal_acceleration_max": float(
                np.max(np.linalg.norm(logger.command_acceleration_m_s2[event_indices, :2], axis=1))
            ) if len(event_indices) else None,
            "command_quaternion_xy_norm_max": float(
                np.max(np.linalg.norm(logger.command_orientation_xyzw[event_indices, :2], axis=1))
            ) if len(event_indices) else None,
            "command_quaternion_xy_norm_median": float(
                np.median(np.linalg.norm(logger.command_orientation_xyzw[event_indices, :2], axis=1))
            ) if len(event_indices) else None,
            "command_yaw_raw_min": float(np.min(logger.command_yaw[event_indices])) if len(event_indices) else None,
            "command_yaw_raw_max": float(np.max(logger.command_yaw[event_indices])) if len(event_indices) else None,
            "command_omega_raw_absolute_max": float(
                np.max(np.abs(logger.command_angular_velocity_raw[event_indices]))
            ) if len(event_indices) else None,
        },
    }
    return arrays, report
