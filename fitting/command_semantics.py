"""Dataset-backed command-semantics checks for Milestone 3A.

The flight computer is external to this repository.  These diagnostics make
only the user-confirmed stock/default assumption and the facts directly
present in the four recordings.  They do not emulate unavailable firmware.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from simulator.parameters import UAVResponseParameters
from simulator.uav.quaternion import (
    desired_rotation_from_acceleration_and_yaw,
    quaternion_to_rotation_matrix_xyzw,
    yaw_from_quaternion_xyzw,
)

from .dataset import Dataset, ProcessedTake


SEMANTICS_ASSUMPTION = {
    "flight_stack_location": "external_computer_not_available_in_repository",
    "firmware": "stock_unmodified_user_confirmed_exact_version_not_archived_yet",
    "configuration": "default_user_confirmed_exact_external_files_not_archived_yet",
    "controller": "default_Mellinger_user_confirmed",
    "estimator": "default_Kalman_user_confirmed",
    "vehicle_type": "medium",
    "command_quaternion_interpretation": "heading_yaw_reference_for_current_dataset",
    "command_angular_velocity_semantics": "body_frame_rad_s_standard_cmdFullState_contract",
    "current_dataset_angular_velocity_command": "identically_zero_frame_invariant",
    "model_scope": "effective_closed_loop_recorded_command_to_measured_uav_motion",
    "future_reproducibility_action": "archive_exact_external_firmware_and_configuration_when_access_is_available",
}


def _rotation_to_euler_xyz_rad(rotation: np.ndarray) -> np.ndarray:
    """Return extrinsic XYZ / roll-pitch-yaw angles for body-to-world R."""

    pitch = np.arcsin(np.clip(-rotation[..., 2, 0], -1.0, 1.0))
    roll = np.arctan2(rotation[..., 2, 1], rotation[..., 2, 2])
    yaw = np.arctan2(rotation[..., 1, 0], rotation[..., 0, 0])
    return np.stack((roll, pitch, yaw), axis=-1)


def _wrapped_angle_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(left - right), np.cos(left - right))


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    finite = np.isfinite(left) & np.isfinite(right)
    if int(np.count_nonzero(finite)) < 3:
        return None
    left_valid, right_valid = left[finite], right[finite]
    if float(np.std(left_valid)) < 1.0e-12 or float(np.std(right_valid)) < 1.0e-12:
        return None
    return float(np.corrcoef(left_valid, right_valid)[0, 1])


def _best_attitude_lag(
    desired_rp: np.ndarray,
    measured_rp: np.ndarray,
    dt_s: float,
    maximum_lag_s: float = 0.25,
) -> tuple[float, float]:
    """Find diagnostic lag; positive means measured attitude occurs later."""

    maximum = int(round(maximum_lag_s / dt_s))
    best_lag, best_score = 0, -math.inf
    for lag in range(-maximum, maximum + 1):
        if lag > 0:
            desired, measured = desired_rp[:-lag], measured_rp[lag:]
        elif lag < 0:
            desired, measured = desired_rp[-lag:], measured_rp[:lag]
        else:
            desired, measured = desired_rp, measured_rp
        correlations = [
            value
            for axis in range(2)
            if (value := _safe_correlation(desired[:, axis], measured[:, axis]))
            is not None
        ]
        score = float(np.mean(correlations)) if correlations else -math.inf
        if score > best_score:
            best_lag, best_score = lag, score
    return float(best_lag * dt_s), float(best_score)


def instantaneous_desired_attitude(
    take: ProcessedTake,
    parameters: UAVResponseParameters,
    gravity_m_s2: float = 9.80665,
) -> dict[str, np.ndarray]:
    """Evaluate the corrected attitude construction on measured current state.

    The retrospective velocity derivative is used only for this structural
    diagnostic.  It is never used to initialize or correct a fitting rollout.
    """

    arrays = take.arrays
    time = arrays["time_s"]
    measured_velocity = np.gradient(arrays["uav_position_m"], time, axis=0)
    K_p, K_v, k_a = (
        float(parameters.K_p),
        float(parameters.K_v),
        float(parameters.k_a),
    )
    acceleration = (
        K_p * (arrays["command_position_m"] - arrays["uav_position_m"])
        + K_v * (arrays["command_velocity_mps"] - measured_velocity)
        + k_a * arrays["command_acceleration_mps2"]
    )
    command_q = torch.as_tensor(arrays["command_orientation_xyzw"], dtype=torch.float64)
    desired_rotation = desired_rotation_from_acceleration_and_yaw(
        torch.as_tensor(acceleration, dtype=torch.float64),
        yaw_from_quaternion_xyzw(command_q),
        gravity_m_s2,
    ).numpy()
    measured_rotation = quaternion_to_rotation_matrix_xyzw(
        torch.as_tensor(arrays["uav_orientation_xyzw"], dtype=torch.float64)
    ).numpy()
    return {
        "desired_acceleration_m_s2": acceleration,
        "desired_rotation": desired_rotation,
        "measured_rotation": measured_rotation,
        "desired_euler_rad": _rotation_to_euler_xyz_rad(desired_rotation),
        "measured_euler_rad": _rotation_to_euler_xyz_rad(measured_rotation),
    }


def structural_diagnostic_for_take(
    take: ProcessedTake,
    parameters: UAVResponseParameters,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    data = instantaneous_desired_attitude(take, parameters)
    desired = data["desired_euler_rad"]
    measured = data["measured_euler_rad"]
    command_valid = take.arrays["command_valid"].astype(bool)
    usable = command_valid & take.arrays["uav_valid"].astype(bool)
    desired, measured = desired[usable], measured[usable]
    error = _wrapped_angle_difference(desired, measured)
    dt_s = float(np.median(np.diff(take.arrays["time_s"])))
    lag_s, lag_correlation = _best_attitude_lag(desired[:, :2], measured[:, :2], dt_s)
    horizontal_force = np.linalg.norm(
        data["desired_acceleration_m_s2"][usable, :2], axis=1
    )
    desired_tilt = np.linalg.norm(desired[:, :2], axis=1)
    measured_tilt = np.linalg.norm(measured[:, :2], axis=1)
    desired_step_rotation = np.einsum(
        "...ji,...jk->...ik",
        data["desired_rotation"][usable][:-1],
        data["desired_rotation"][usable][1:],
    )
    step_angle = np.arccos(
        np.clip((np.trace(desired_step_rotation, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    )
    report: dict[str, object] = {
        "frames": int(np.count_nonzero(usable)),
        "q_cmd_xy_max": float(
            np.max(np.linalg.norm(take.arrays["command_orientation_xyzw"][usable, :2], axis=1))
        ),
        "omega_cmd_abs_max": float(
            np.max(np.abs(take.arrays["command_angular_velocity"][usable]))
        ),
        "horizontal_acceleration_command_rms_m_s2": float(
            np.sqrt(np.mean(np.sum(take.arrays["command_acceleration_mps2"][usable, :2] ** 2, axis=1)))
        ),
        "horizontal_acceleration_command_max_m_s2": float(
            np.max(np.linalg.norm(take.arrays["command_acceleration_mps2"][usable, :2], axis=1))
        ),
        "roll_rmse_deg": float(np.degrees(np.sqrt(np.mean(error[:, 0] ** 2)))),
        "pitch_rmse_deg": float(np.degrees(np.sqrt(np.mean(error[:, 1] ** 2)))),
        "yaw_rmse_deg": float(np.degrees(np.sqrt(np.mean(error[:, 2] ** 2)))),
        "roll_correlation": _safe_correlation(desired[:, 0], measured[:, 0]),
        "pitch_correlation": _safe_correlation(desired[:, 1], measured[:, 1]),
        "horizontal_force_vs_desired_tilt_correlation": _safe_correlation(horizontal_force, desired_tilt),
        "horizontal_force_vs_measured_tilt_correlation": _safe_correlation(horizontal_force, measured_tilt),
        "best_measured_lag_s": lag_s,
        "best_lag_roll_pitch_mean_correlation": lag_correlation,
        "maximum_desired_attitude_step_deg": float(np.degrees(np.max(step_angle))) if len(step_angle) else 0.0,
    }
    return report, data


def run_structural_diagnostic(
    dataset: Dataset,
    parameters: UAVResponseParameters,
) -> tuple[dict[str, object], dict[str, dict[str, np.ndarray]]]:
    reports: dict[str, object] = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    # Model-structure diagnostics are part of the fitting contract.  Ignored
    # and protected Untouched Test takes must not affect model readiness or a
    # previously frozen scientific interpretation.
    for take in dataset.fitting_takes:
        report, data = structural_diagnostic_for_take(take, parameters)
        reports[take.take_id] = report
        arrays[take.take_id] = data

    q_yaw_only = all(
        float(item["q_cmd_xy_max"]) == 0.0
        for item in reports.values()  # type: ignore[union-attr]
    )
    omega_zero = all(
        float(item["omega_cmd_abs_max"]) == 0.0
        for item in reports.values()  # type: ignore[union-attr]
    )
    # Direction checks are deliberately structural, not accuracy gates.  The
    # low-roll-amplitude osc take is judged on its excited pitch axis.
    directional = True
    for take_id, item in reports.items():
        assert isinstance(item, dict)
        correlations = (
            [item["pitch_correlation"]]
            if take_id == "osc_001"
            else [item["roll_correlation"], item["pitch_correlation"]]
        )
        directional &= all(value is not None and float(value) > 0.25 for value in correlations)
    summary = {
        "schema": "milestone3a_command_structure_diagnostic_v1",
        "assumption": SEMANTICS_ASSUMPTION,
        "nominal_parameters_used_only_for_structure": {
            name: float(getattr(parameters, name))
            for name in ("K_p", "K_v", "k_a", "K_R", "K_omega")
        },
        "takes": reports,
        "checks": {
            "all_q_cmd_xy_exactly_zero": q_yaw_only,
            "all_omega_cmd_exactly_zero": omega_zero,
            "excited_axis_tilt_direction_consistent": bool(directional),
            "structural_diagnostic_passed": bool(q_yaw_only and omega_zero and directional),
        },
        "limitations": [
            "All four takes were inspected while correcting model structure; fig8_003 remains held out from parameter fitting, not from structural model selection.",
            "Retrospective measured velocity is used only in this diagnostic and never enters an open-loop fit rollout.",
            "The standard cmdFullState angular-rate command is body-frame rad/s; all current values are exactly zero, so this clarification does not change any fitted rollout.",
            "The exact firmware commit and external configuration files have not yet been archived and must be captured later for final reproducibility.",
        ],
    }
    return summary, arrays
