"""Deterministic checks for passive cable--cube contact inference."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parent.parent
MAIN_SOURCE_DIR = PROJECT_DIR / "ZED_segmentation_viewer" / "source"
if str(MAIN_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_SOURCE_DIR))

from contact_estimator import CableCubeContactEstimator, ContactEstimatorConfig
from cube_state_filter import CubeStateEstimate
from cube_tracker import CubeTrackingResult, query_cube_surface


PARTICLE_COUNT = 32
NODE_COUNT = 6
CABLE_LENGTH_M = 0.120
CABLE_RADIUS_M = 0.0045
CUBE_SIDE_M = 0.150


class _Posterior:
    def __init__(
        self,
        particles: torch.Tensor,
        velocities: torch.Tensor,
        weights: torch.Tensor,
    ):
        self.tensors = (particles, velocities, weights)

    def posterior_tensors(self):
        return self.tensors


def _inputs(
    device: torch.device,
    *,
    cable_velocity_mps: float = 0.0,
) -> _Posterior:
    particles = torch.zeros(
        (2, PARTICLE_COUNT, NODE_COUNT, 3),
        dtype=torch.float32,
        device=device,
    )
    particles[0, :, :, 0] = torch.linspace(
        -0.060, 0.060, NODE_COUNT, device=device
    )
    particles[0, :, :, 1] = 0.5 * CUBE_SIDE_M + CABLE_RADIUS_M
    particles[1, :, :, 0] = torch.linspace(
        -0.060, 0.060, NODE_COUNT, device=device
    )
    particles[1, :, :, 1] = 0.5 * CUBE_SIDE_M + 0.030
    velocities = torch.zeros_like(particles)
    velocities[0, :, :, 0] = float(cable_velocity_mps)
    weights = torch.full(
        (2, PARTICLE_COUNT),
        1.0 / PARTICLE_COUNT,
        dtype=torch.float32,
        device=device,
    )
    return _Posterior(particles, velocities, weights)


def _frame(*, cable_measurements: tuple[bool, bool] = (True, True)):
    return SimpleNamespace(
        initialized=(True, True),
        measurement_used=cable_measurements,
        cable_radius_m=CABLE_RADIUS_M,
    )


def _cube_state(
    timestamp: float,
    *,
    measurement_used: bool = True,
    linear_velocity_mps: float = 0.0,
) -> CubeStateEstimate:
    covariance = np.zeros((12, 12), dtype=np.float64)
    covariance[:3, :3] = np.eye(3) * 0.001**2
    covariance[3:6, 3:6] = np.eye(3) * np.deg2rad(0.5) ** 2
    covariance[6:9, 6:9] = np.eye(3) * 0.010**2
    covariance[9:12, 9:12] = np.eye(3) * 0.020**2
    return CubeStateEstimate(
        valid=True,
        initialized=True,
        measurement_used=measurement_used,
        reason="synthetic cube state",
        cube_side_m=CUBE_SIDE_M,
        timestamp=float(timestamp),
        measurement_age_s=0.0,
        center_m=np.zeros(3, dtype=np.float64),
        rotation=np.eye(3, dtype=np.float64),
        quaternion_xyzw=np.array((0.0, 0.0, 0.0, 1.0)),
        linear_velocity_mps=np.array(
            (linear_velocity_mps, 0.0, 0.0), dtype=np.float64
        ),
        angular_velocity_rps=np.zeros(3, dtype=np.float64),
        pose_covariance=covariance[:6, :6].copy(),
        state_covariance=covariance,
        processing_ms=0.1,
    )


def _cube_measurement() -> CubeTrackingResult:
    return CubeTrackingResult(
        valid=True,
        reason="synthetic cube measurement",
        cube_side_m=CUBE_SIDE_M,
        mask=np.zeros((4, 4), dtype=np.uint8),
        center_m=np.zeros(3, dtype=np.float64),
        rotation=np.eye(3, dtype=np.float64),
        quaternion_xyzw=np.array((0.0, 0.0, 0.0, 1.0)),
        refinement_valid=True,
        refinement_reason="synthetic refinement",
        refined_center_m=np.zeros(3, dtype=np.float64),
        refined_rotation=np.eye(3, dtype=np.float64),
        refined_quaternion_xyzw=np.array((0.0, 0.0, 0.0, 1.0)),
        pose_covariance=np.eye(6, dtype=np.float64) * 1.0e-6,
        candidate_plane_count=3,
        face_count=3,
        yellow_pixels=1000,
        sampled_mask_pixels=1000,
        valid_depth_points=1000,
        depth_coverage=1.0,
        fit_point_count=1000,
        surface_rms_m=0.001,
        refined_surface_rms_m=0.001,
        observed_span_m=CUBE_SIDE_M,
        refinement_iterations=3,
        refinement_ms=0.1,
        processing_ms=1.0,
        face_pixels=(),
    )


def _estimator(device: torch.device) -> CableCubeContactEstimator:
    return CableCubeContactEstimator(
        ContactEstimatorConfig(),
        cable_lengths_m=(CABLE_LENGTH_M, CABLE_LENGTH_M),
        particle_count=PARTICLE_COUNT,
        node_count=NODE_COUNT,
        dense_samples_per_segment=3,
        device=device,
    )


class ContactEstimatorTests(unittest.TestCase):
    def test_cuda_ready_box_query_matches_canonical_cube_geometry(self):
        angle = 0.47
        rotation = np.array(
            (
                (np.cos(angle), -np.sin(angle), 0.0),
                (np.sin(angle), np.cos(angle), 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        center = np.array((0.04, -0.03, -0.8), dtype=np.float64)
        local_points = np.array(
            (
                (0.090, 0.020, -0.010),
                (0.010, 0.020, 0.010),
                (-0.075, 0.010, 0.030),
            ),
            dtype=np.float64,
        )
        points = center + local_points @ rotation.T
        expected = query_cube_surface(points, center, rotation, CUBE_SIDE_M)
        actual = CableCubeContactEstimator._box_query(
            torch.as_tensor(points, dtype=torch.float64),
            torch.as_tensor(center, dtype=torch.float64),
            torch.as_tensor(rotation, dtype=torch.float64),
            CUBE_SIDE_M,
        )
        np.testing.assert_allclose(actual[0].numpy(), expected.signed_distance_m)
        np.testing.assert_allclose(actual[1].numpy(), expected.closest_points_m)
        np.testing.assert_allclose(actual[2].numpy(), expected.normals)

    def test_repeated_evidence_establishes_contact_but_one_frame_does_not(self):
        device = torch.device("cpu")
        estimator = _estimator(device)
        posterior = _inputs(device)
        measurement = _cube_measurement()

        outputs = []
        for frame_index in range(12):
            timestamp = frame_index * 0.05
            outputs.append(
                estimator.update(
                    posterior,
                    _frame(),
                    _cube_state(timestamp),
                    measurement,
                    timestamp,
                )
            )

        first = outputs[0].cables[0]
        last_contact = outputs[-1].cables[0]
        last_free = outputs[-1].cables[1]
        self.assertLess(first.contact_probability, 0.10)
        self.assertGreater(last_contact.contact_probability, 0.80)
        self.assertLess(last_free.contact_probability, 0.01)
        self.assertAlmostEqual(last_contact.minimum_gap_m, 0.0, places=6)
        self.assertAlmostEqual(last_free.minimum_gap_m, 0.0255, places=5)
        self.assertAlmostEqual(last_contact.arc_interval_m[0], 0.0, places=6)
        self.assertAlmostEqual(
            last_contact.arc_interval_m[1], CABLE_LENGTH_M, places=6
        )
        np.testing.assert_allclose(
            last_contact.surface_normal,
            np.array((0.0, 1.0, 0.0)),
            atol=1.0e-6,
        )

    def test_prediction_only_frame_cannot_add_contact_evidence(self):
        device = torch.device("cpu")
        estimator = _estimator(device)
        posterior = _inputs(device)
        measurement = _cube_measurement()
        previous = None
        for frame_index in range(12):
            timestamp = frame_index * 0.05
            previous = estimator.update(
                posterior,
                _frame(),
                _cube_state(timestamp),
                measurement,
                timestamp,
            )
        assert previous is not None
        predicted = estimator.update(
            posterior,
            _frame(cable_measurements=(False, True)),
            _cube_state(0.60),
            measurement,
            0.60,
        )
        self.assertFalse(predicted.cables[0].evidence_used)
        self.assertIn("cable measurement unavailable", predicted.cables[0].reason)
        self.assertLess(
            predicted.cables[0].contact_probability,
            previous.cables[0].contact_probability,
        )

    def test_observer_does_not_modify_particle_filter_tensors(self):
        device = torch.device("cpu")
        posterior = _inputs(device)
        before = tuple(value.clone() for value in posterior.tensors)
        _estimator(device).update(
            posterior,
            _frame(),
            _cube_state(0.0),
            _cube_measurement(),
            0.0,
        )
        for expected, actual in zip(before, posterior.tensors):
            self.assertTrue(torch.equal(expected, actual))

    def test_comotion_is_positive_evidence_only_when_motion_is_excited(self):
        device = torch.device("cpu")
        measurement = _cube_measurement()
        stationary = _estimator(device).update(
            _inputs(device),
            _frame(),
            _cube_state(0.0),
            measurement,
            0.0,
        )
        moving = _estimator(device).update(
            _inputs(device, cable_velocity_mps=0.10),
            _frame(),
            _cube_state(0.0, linear_velocity_mps=0.10),
            measurement,
            0.0,
        )
        self.assertAlmostEqual(stationary.cables[0].comotion_score, 0.0, places=6)
        self.assertGreater(moving.cables[0].comotion_score, 0.95)
        self.assertGreater(
            moving.cables[0].contact_probability,
            stationary.cables[0].contact_probability,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_path_matches_contact_geometry(self):
        device = torch.device("cuda")
        output = _estimator(device).update(
            _inputs(device),
            _frame(),
            _cube_state(0.0),
            _cube_measurement(),
            0.0,
        )
        self.assertTrue(output.cables[0].geometry_valid)
        self.assertAlmostEqual(output.cables[0].minimum_gap_m, 0.0, places=6)
        self.assertGreaterEqual(output.gpu_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
