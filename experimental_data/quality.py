"""Measurement/synchronization quality flags without rejecting high dynamics."""

from __future__ import annotations

import numpy as np


def quaternion_to_rotation_matrix_xyzw(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack(
        (
            1 - 2 * (y*y + z*z), 2 * (x*y-z*w), 2 * (x*z+y*w),
            2 * (x*y+z*w), 1 - 2 * (x*x+z*z), 2 * (y*z-x*w),
            2 * (x*z-y*w), 2 * (y*z+x*w), 1 - 2 * (x*x+y*y),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def _transition_frames(values: np.ndarray) -> np.ndarray:
    output = np.zeros(len(values) + 1, dtype=bool)
    indices = np.flatnonzero(values)
    output[indices] = True
    output[indices + 1] = True
    return output


def _intervals(mask: np.ndarray, time_s: np.ndarray, reason: str) -> list[dict[str, object]]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    result: list[dict[str, object]] = []
    start = previous = int(indices[0])
    for raw in indices[1:]:
        index = int(raw)
        if index != previous + 1:
            result.append(
                {"start_s": float(time_s[start]), "end_s": float(time_s[previous]), "reason": reason}
            )
            start = index
        previous = index
    result.append(
        {"start_s": float(time_s[start]), "end_s": float(time_s[previous]), "reason": reason}
    )
    return result


def evaluate_quality(
    arrays: dict[str, np.ndarray],
    *,
    quality_config: dict[str, object],
    attachment_offset_body_m: tuple[float, float, float],
    interval_lengths_m: tuple[float, ...],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    time = arrays["time_s"]
    uav_valid = arrays["uav_valid"].astype(bool)
    marker_valid = arrays["cable_marker_valid"].astype(bool)
    uav = arrays["uav_position_m"]
    marker = arrays["cable_marker_positions_m"]
    uav_transition = uav_valid[:-1] & uav_valid[1:]
    uav_jump = _transition_frames(
        uav_transition
        & (np.linalg.norm(np.diff(uav, axis=0), axis=1) > float(quality_config["maximum_uav_position_jump_m"]))
    )
    common_marker = marker_valid[:-1] & marker_valid[1:]
    marker_delta = np.linalg.norm(np.diff(marker, axis=0), axis=2)
    marker_jump = _transition_frames(
        np.any(
            common_marker
            & (marker_delta > float(quality_config["maximum_marker_position_jump_m"])),
            axis=1,
        )
    )
    rotation = quaternion_to_rotation_matrix_xyzw(arrays["uav_orientation_xyzw"])
    offset = np.asarray(attachment_offset_body_m, dtype=np.float64)
    connector = uav + np.einsum("tij,j->ti", rotation, offset)
    sites = np.concatenate((connector[:, None], marker), axis=1)
    site_valid = np.concatenate((uav_valid[:, None], marker_valid), axis=1)
    interval_valid = site_valid[:, :-1] & site_valid[:, 1:]
    measured_interval = np.linalg.norm(np.diff(sites, axis=1), axis=2)
    interval_error = np.abs(measured_interval - np.asarray(interval_lengths_m)[None])
    geometry_invalid = np.any(
        interval_valid
        & (interval_error > float(quality_config["maximum_interval_length_error_m"])),
        axis=1,
    )
    marker_dropout = ~np.all(marker_valid, axis=1)
    command_invalid = ~arrays["command_valid"].astype(bool)
    command_stale = (
        arrays["command_valid"].astype(bool)
        & np.isfinite(arrays["command_age_s"])
        & (arrays["command_age_s"] > float(quality_config["maximum_command_age_s"]))
    )
    critical = (~uav_valid) | uav_jump | marker_jump | geometry_invalid
    flags = {
        "quality_uav_invalid": ~uav_valid,
        "quality_uav_jump": uav_jump,
        "quality_marker_dropout": marker_dropout,
        "quality_marker_jump": marker_jump,
        "quality_geometry_invalid": geometry_invalid,
        "quality_command_invalid": command_invalid,
        "quality_command_stale": command_stale,
        "auto_frame_valid": ~critical,
    }
    intervals: list[dict[str, object]] = []
    for name, mask in flags.items():
        if name != "auto_frame_valid":
            intervals.extend(_intervals(mask, time, name.removeprefix("quality_")))
    problem_intervals = sorted(intervals, key=lambda item: (item["start_s"], item["reason"]))
    long_dropout_threshold = float(quality_config["long_dropout_duration_s"])
    median_dt = float(np.median(np.diff(time))) if len(time) > 1 else 0.0
    report = {
        "uav_valid_fraction": float(np.mean(uav_valid)),
        "cable_valid_fraction": float(np.mean(marker_valid)),
        "marker_valid_fraction": [float(np.mean(marker_valid[:, index])) for index in range(marker_valid.shape[1])],
        "automatic_problem_regions": problem_intervals,
        "long_dropout_intervals": [
            item for item in problem_intervals
            if item["reason"] == "marker_dropout"
            and float(item["end_s"]) - float(item["start_s"]) + median_dt >= long_dropout_threshold
        ],
        "flagged_frame_counts": {name: int(np.count_nonzero(mask)) for name, mask in flags.items() if name != "auto_frame_valid"},
        "maximum_measured_interval_error_m": float(np.nanmax(np.where(interval_valid, interval_error, np.nan))),
    }
    return flags, report
