"""Timestamp-aware rigid cube state filter separated from raw RGB-D measurement."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from cube_tracker import (
    CubeTrackingResult,
    choose_equivalent_rotation,
    rotation_matrix_to_quaternion,
)


@dataclass(frozen=True)
class CubeStateFilterConfig:
    """Constant-velocity SE(3) filter parameters."""

    linear_acceleration_std_mps2: float = 1.5
    angular_acceleration_std_rps2: float = 4.0
    initial_linear_velocity_std_mps: float = 0.20
    initial_angular_velocity_std_rps: float = 0.80
    maximum_prediction_age_s: float = 0.50
    maximum_prediction_step_s: float = 0.10

    @classmethod
    def from_mapping(cls, values: dict | None) -> "CubeStateFilterConfig":
        source = values or {}
        defaults = cls()
        config = cls(
            linear_acceleration_std_mps2=float(
                source.get(
                    "linear_acceleration_std_mps2",
                    defaults.linear_acceleration_std_mps2,
                )
            ),
            angular_acceleration_std_rps2=float(
                source.get(
                    "angular_acceleration_std_rps2",
                    defaults.angular_acceleration_std_rps2,
                )
            ),
            initial_linear_velocity_std_mps=float(
                source.get(
                    "initial_linear_velocity_std_mps",
                    defaults.initial_linear_velocity_std_mps,
                )
            ),
            initial_angular_velocity_std_rps=float(
                source.get(
                    "initial_angular_velocity_std_rps",
                    defaults.initial_angular_velocity_std_rps,
                )
            ),
            maximum_prediction_age_s=float(
                source.get(
                    "maximum_prediction_age_s",
                    defaults.maximum_prediction_age_s,
                )
            ),
            maximum_prediction_step_s=float(
                source.get(
                    "maximum_prediction_step_s",
                    defaults.maximum_prediction_step_s,
                )
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.linear_acceleration_std_mps2 <= 0.0:
            raise ValueError("linear_acceleration_std_mps2 must be positive.")
        if self.angular_acceleration_std_rps2 <= 0.0:
            raise ValueError("angular_acceleration_std_rps2 must be positive.")
        if self.initial_linear_velocity_std_mps <= 0.0:
            raise ValueError("initial_linear_velocity_std_mps must be positive.")
        if self.initial_angular_velocity_std_rps <= 0.0:
            raise ValueError("initial_angular_velocity_std_rps must be positive.")
        if self.maximum_prediction_age_s <= 0.0:
            raise ValueError("maximum_prediction_age_s must be positive.")
        if not 0.0 < self.maximum_prediction_step_s <= self.maximum_prediction_age_s:
            raise ValueError(
                "maximum_prediction_step_s must be positive and no greater than "
                "maximum_prediction_age_s."
            )


@dataclass(frozen=True)
class CubeStateEstimate:
    """Filtered rigid state; the raw CubeTrackingResult remains unchanged.

    Pose covariance order is camera-frame left perturbation
    [position xyz, rotation-vector xyz]. State covariance appends camera-frame
    linear velocity xyz and angular velocity xyz.
    """

    valid: bool
    initialized: bool
    measurement_used: bool
    reason: str
    cube_side_m: float
    timestamp: float
    measurement_age_s: float
    center_m: np.ndarray | None
    rotation: np.ndarray | None
    quaternion_xyzw: np.ndarray | None
    linear_velocity_mps: np.ndarray | None
    angular_velocity_rps: np.ndarray | None
    pose_covariance: np.ndarray | None
    state_covariance: np.ndarray | None
    processing_ms: float


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array(
        (
            (0.0, -z, y),
            (z, 0.0, -x),
            (-y, x, 0.0),
        ),
        dtype=np.float64,
    )


def _rotation_exp(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(value))
    skew = _skew(value)
    if angle < 1.0e-8:
        return np.eye(3, dtype=np.float64) + skew + 0.5 * (skew @ skew)
    first = math.sin(angle) / angle
    second = (1.0 - math.cos(angle)) / (angle * angle)
    return np.eye(3, dtype=np.float64) + first * skew + second * (skew @ skew)


def _so3_left_jacobian(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(value))
    skew = _skew(value)
    if angle < 1.0e-8:
        return np.eye(3, dtype=np.float64) + 0.5 * skew + (skew @ skew) / 6.0
    second = (1.0 - math.cos(angle)) / (angle * angle)
    third = (angle - math.sin(angle)) / (angle**3)
    return np.eye(3, dtype=np.float64) + second * skew + third * (skew @ skew)


def _rotation_log(rotation: np.ndarray) -> np.ndarray:
    quaternion = rotation_matrix_to_quaternion(rotation)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    vector = quaternion[:3]
    length = float(np.linalg.norm(vector))
    if length < 1.0e-10:
        return 2.0 * vector
    angle = 2.0 * math.atan2(length, float(quaternion[3]))
    return (angle / length) * vector


def _symmetric_positive_covariance(covariance: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (
        np.asarray(covariance, dtype=np.float64)
        + np.asarray(covariance, dtype=np.float64).T
    )
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.maximum(eigenvalues, 1.0e-12)
    return (eigenvectors * eigenvalues[None, :]) @ eigenvectors.T


class RigidCubeStateFilter:
    """One-thread error-state Kalman filter on position, rotation, and velocity."""

    def __init__(self, config: CubeStateFilterConfig):
        config.validate()
        self.config = config
        self.center_m: np.ndarray | None = None
        self.rotation: np.ndarray | None = None
        self.linear_velocity_mps = np.zeros(3, dtype=np.float64)
        self.angular_velocity_rps = np.zeros(3, dtype=np.float64)
        self.covariance: np.ndarray | None = None
        self.cube_side_m = float("nan")
        self.last_timestamp: float | None = None
        self.last_measurement_timestamp: float | None = None

    @property
    def initialized(self) -> bool:
        return (
            self.center_m is not None
            and self.rotation is not None
            and self.covariance is not None
        )

    @staticmethod
    def _has_refined_measurement(measurement: CubeTrackingResult | None) -> bool:
        return bool(
            measurement is not None
            and measurement.valid
            and measurement.refinement_valid
            and measurement.refined_center_m is not None
            and measurement.refined_rotation is not None
            and measurement.pose_covariance is not None
        )

    def _initialize(
        self,
        measurement: CubeTrackingResult,
        timestamp: float,
    ) -> None:
        assert measurement.refined_center_m is not None
        assert measurement.refined_rotation is not None
        assert measurement.pose_covariance is not None
        self.center_m = np.asarray(
            measurement.refined_center_m,
            dtype=np.float64,
        ).reshape(3).copy()
        self.rotation = np.asarray(
            measurement.refined_rotation,
            dtype=np.float64,
        ).reshape(3, 3).copy()
        self.linear_velocity_mps.fill(0.0)
        self.angular_velocity_rps.fill(0.0)
        covariance = np.zeros((12, 12), dtype=np.float64)
        covariance[:6, :6] = _symmetric_positive_covariance(
            measurement.pose_covariance
        )
        covariance[6:9, 6:9] = (
            self.config.initial_linear_velocity_std_mps**2
        ) * np.eye(3)
        covariance[9:12, 9:12] = (
            self.config.initial_angular_velocity_std_rps**2
        ) * np.eye(3)
        self.covariance = covariance
        self.cube_side_m = float(measurement.cube_side_m)
        self.last_timestamp = timestamp
        self.last_measurement_timestamp = timestamp

    def _predict_step(self, dt: float) -> None:
        assert self.center_m is not None
        assert self.rotation is not None
        assert self.covariance is not None
        self.center_m += dt * self.linear_velocity_mps
        rotation_vector = dt * self.angular_velocity_rps
        rotation_increment = _rotation_exp(rotation_vector)
        rotation_jacobian = _so3_left_jacobian(rotation_vector)
        self.rotation = rotation_increment @ self.rotation

        transition = np.eye(12, dtype=np.float64)
        transition[:3, 6:9] = dt * np.eye(3)
        transition[3:6, 3:6] = rotation_increment
        transition[3:6, 9:12] = dt * rotation_jacobian
        process = np.zeros((12, 12), dtype=np.float64)
        linear_noise_map = np.zeros((12, 3), dtype=np.float64)
        linear_noise_map[:3] = 0.5 * dt**2 * np.eye(3)
        linear_noise_map[6:9] = dt * np.eye(3)
        process += (
            self.config.linear_acceleration_std_mps2**2
        ) * (linear_noise_map @ linear_noise_map.T)
        angular_noise_map = np.zeros((12, 3), dtype=np.float64)
        angular_noise_map[3:6] = 0.5 * dt**2 * rotation_jacobian
        angular_noise_map[9:12] = dt * np.eye(3)
        process += (
            self.config.angular_acceleration_std_rps2**2
        ) * (angular_noise_map @ angular_noise_map.T)
        self.covariance = _symmetric_positive_covariance(
            transition @ self.covariance @ transition.T + process
        )

    def _predict_to(self, timestamp: float) -> None:
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
            return
        interval = timestamp - self.last_timestamp
        if interval < -1.0e-9:
            raise ValueError("Cube-state timestamps must be monotonic.")
        remaining = max(0.0, interval)
        while remaining > 0.0:
            dt = min(remaining, self.config.maximum_prediction_step_s)
            self._predict_step(dt)
            remaining -= dt
        self.last_timestamp = timestamp

    def _measurement_update(self, measurement: CubeTrackingResult) -> None:
        assert self.center_m is not None
        assert self.rotation is not None
        assert self.covariance is not None
        assert measurement.refined_center_m is not None
        assert measurement.refined_rotation is not None
        assert measurement.pose_covariance is not None

        measured_center = np.asarray(
            measurement.refined_center_m,
            dtype=np.float64,
        ).reshape(3)
        measured_rotation = choose_equivalent_rotation(
            np.asarray(measurement.refined_rotation, dtype=np.float64).reshape(3, 3),
            self.rotation,
        )
        residual = np.concatenate(
            (
                measured_center - self.center_m,
                _rotation_log(measured_rotation @ self.rotation.T),
            )
        )
        observation = np.zeros((6, 12), dtype=np.float64)
        observation[:, :6] = np.eye(6)
        measurement_covariance = _symmetric_positive_covariance(
            measurement.pose_covariance
        )
        innovation_covariance = (
            observation @ self.covariance @ observation.T
            + measurement_covariance
        )
        covariance_observation_t = self.covariance @ observation.T
        gain = np.linalg.solve(
            innovation_covariance,
            covariance_observation_t.T,
        ).T
        increment = gain @ residual
        self.center_m += increment[:3]
        self.rotation = _rotation_exp(increment[3:6]) @ self.rotation
        self.linear_velocity_mps += increment[6:9]
        self.angular_velocity_rps += increment[9:12]

        identity = np.eye(12, dtype=np.float64)
        correction = identity - gain @ observation
        posterior_covariance = (
            correction @ self.covariance @ correction.T
            + gain @ measurement_covariance @ gain.T
        )
        reset_jacobian = np.eye(12, dtype=np.float64)
        # The filter uses a left orientation error:
        # R_true = Exp(delta_theta) R_nominal. After injecting the estimated
        # increment on the left, the first-order reset Jacobian has a positive
        # half-skew term.
        reset_jacobian[3:6, 3:6] += 0.5 * _skew(increment[3:6])
        self.covariance = _symmetric_positive_covariance(
            reset_jacobian @ posterior_covariance @ reset_jacobian.T
        )

    def _estimate(
        self,
        *,
        timestamp: float,
        measurement_used: bool,
        reason: str,
        processing_ms: float,
    ) -> CubeStateEstimate:
        age = (
            timestamp - self.last_measurement_timestamp
            if self.last_measurement_timestamp is not None
            else float("inf")
        )
        valid = bool(
            self.initialized and age <= self.config.maximum_prediction_age_s
        )
        if not self.initialized:
            return CubeStateEstimate(
                valid=False,
                initialized=False,
                measurement_used=False,
                reason=reason,
                cube_side_m=float("nan"),
                timestamp=timestamp,
                measurement_age_s=float("inf"),
                center_m=None,
                rotation=None,
                quaternion_xyzw=None,
                linear_velocity_mps=None,
                angular_velocity_rps=None,
                pose_covariance=None,
                state_covariance=None,
                processing_ms=processing_ms,
            )

        assert self.center_m is not None
        assert self.rotation is not None
        assert self.covariance is not None
        return CubeStateEstimate(
            valid=valid,
            initialized=True,
            measurement_used=measurement_used,
            reason=reason,
            cube_side_m=self.cube_side_m,
            timestamp=timestamp,
            measurement_age_s=max(0.0, age),
            center_m=self.center_m.copy(),
            rotation=self.rotation.copy(),
            quaternion_xyzw=rotation_matrix_to_quaternion(self.rotation),
            linear_velocity_mps=self.linear_velocity_mps.copy(),
            angular_velocity_rps=self.angular_velocity_rps.copy(),
            pose_covariance=self.covariance[:6, :6].copy(),
            state_covariance=self.covariance.copy(),
            processing_ms=processing_ms,
        )

    def update(
        self,
        measurement: CubeTrackingResult | None,
        timestamp: float,
    ) -> CubeStateEstimate:
        started = time.perf_counter()
        timestamp = float(timestamp)
        has_measurement = self._has_refined_measurement(measurement)

        if not self.initialized:
            if not has_measurement:
                return self._estimate(
                    timestamp=timestamp,
                    measurement_used=False,
                    reason="waiting for a complete refined cube measurement",
                    processing_ms=(time.perf_counter() - started) * 1000.0,
                )
            assert measurement is not None
            self._initialize(measurement, timestamp)
            return self._estimate(
                timestamp=timestamp,
                measurement_used=True,
                reason="initialized from refined cube measurement",
                processing_ms=(time.perf_counter() - started) * 1000.0,
            )

        self._predict_to(timestamp)
        age_before_update = (
            timestamp - self.last_measurement_timestamp
            if self.last_measurement_timestamp is not None
            else float("inf")
        )
        if has_measurement and age_before_update > self.config.maximum_prediction_age_s:
            assert measurement is not None
            self._initialize(measurement, timestamp)
            reason = "reinitialized after expired cube prediction"
            measurement_used = True
        elif has_measurement:
            assert measurement is not None
            self._measurement_update(measurement)
            self.last_measurement_timestamp = timestamp
            reason = "refined cube measurement update"
            measurement_used = True
        else:
            measurement_used = False
            age = max(0.0, age_before_update)
            if age <= self.config.maximum_prediction_age_s:
                reason = f"cube prediction only ({age * 1000.0:.1f} ms old)"
            else:
                reason = f"cube prediction expired ({age * 1000.0:.1f} ms old)"

        return self._estimate(
            timestamp=timestamp,
            measurement_used=measurement_used,
            reason=reason,
            processing_ms=(time.perf_counter() - started) * 1000.0,
        )
