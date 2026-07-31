"""Deterministic synthetic checks for the isolated cube geometry."""

from __future__ import annotations

from dataclasses import replace
import unittest
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent
MAIN_SOURCE_DIR = PROJECT_DIR / "ZED_segmentation_viewer" / "source"
if str(MAIN_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_SOURCE_DIR))

from cube_tracker import (
    CubeTrackingResult,
    CubeTrackerConfig,
    CUBE_SYMMETRIES,
    cube_pose_from_faces,
    extract_plane_candidates,
    point_to_cube_surface_distance,
    query_cube_surface,
    refine_cube_pose,
    remove_cable_pixels,
    segment_yellow,
    select_orthogonal_pair,
    select_orthogonal_faces,
    two_face_span_is_valid,
)
from cube_state_filter import CubeStateFilterConfig, RigidCubeStateFilter


def axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    skew = np.array(
        (
            (0.0, -axis[2], axis[1]),
            (axis[2], 0.0, -axis[0]),
            (-axis[1], axis[0], 0.0),
        )
    )
    return (
        np.eye(3)
        + np.sin(angle) * skew
        + (1.0 - np.cos(angle)) * (skew @ skew)
    )


def refined_measurement(
    center: np.ndarray,
    rotation: np.ndarray,
    covariance: np.ndarray,
) -> CubeTrackingResult:
    return CubeTrackingResult(
        valid=True,
        reason="synthetic refined measurement",
        cube_side_m=0.150,
        mask=np.zeros((4, 4), dtype=np.uint8),
        center_m=np.asarray(center, dtype=np.float64).copy(),
        rotation=np.asarray(rotation, dtype=np.float64).copy(),
        quaternion_xyzw=np.array((0.0, 0.0, 0.0, 1.0)),
        refinement_valid=True,
        refinement_reason="synthetic joint refinement",
        refined_center_m=np.asarray(center, dtype=np.float64).copy(),
        refined_rotation=np.asarray(rotation, dtype=np.float64).copy(),
        refined_quaternion_xyzw=np.array((0.0, 0.0, 0.0, 1.0)),
        pose_covariance=np.asarray(covariance, dtype=np.float64).copy(),
        candidate_plane_count=3,
        face_count=3,
        yellow_pixels=1000,
        sampled_mask_pixels=1000,
        valid_depth_points=1000,
        depth_coverage=1.0,
        fit_point_count=1000,
        surface_rms_m=0.001,
        refined_surface_rms_m=0.001,
        observed_span_m=float("nan"),
        refinement_iterations=3,
        refinement_ms=0.1,
        processing_ms=1.0,
        face_pixels=(),
    )


class CubeGeometryTests(unittest.TestCase):
    def test_cube_has_24_proper_symmetries(self) -> None:
        self.assertEqual(len(CUBE_SYMMETRIES), 24)
        for symmetry in CUBE_SYMMETRIES:
            np.testing.assert_allclose(symmetry.T @ symmetry, np.eye(3), atol=1.0e-12)
            self.assertAlmostEqual(float(np.linalg.det(symmetry)), 1.0, places=12)

    def test_surface_distance_is_exact_for_cube_faces(self) -> None:
        center = np.array((0.12, -0.03, -0.85))
        rotation = axis_angle_rotation(np.array((0.3, 0.8, 0.5)), 0.61)
        half_side = 0.075
        local = np.array(
            (
                (half_side, 0.02, -0.01),
                (-0.03, -half_side, 0.04),
                (0.01, 0.02, half_side),
            )
        )
        points = center + local @ rotation.T
        distance = point_to_cube_surface_distance(points, center, rotation, 0.150)
        np.testing.assert_allclose(distance, 0.0, atol=1.0e-12)

    def test_signed_surface_query_returns_closest_points_and_normals(self) -> None:
        center = np.array((0.04, -0.02, -0.80))
        rotation = axis_angle_rotation(np.array((0.2, 0.9, 0.3)), 0.48)
        half_side = 0.075
        local = np.array(
            (
                (half_side + 0.010, 0.020, -0.030),
                (half_side - 0.020, 0.000, 0.000),
                (half_side, -0.010, 0.025),
            )
        )
        points = center + local @ rotation.T
        query = query_cube_surface(points, center, rotation, 0.150)

        np.testing.assert_allclose(
            query.signed_distance_m,
            (0.010, -0.020, 0.0),
            atol=1.0e-12,
        )
        expected_closest = local.copy()
        expected_closest[:, 0] = half_side
        np.testing.assert_allclose(
            query.closest_points_m,
            center + expected_closest @ rotation.T,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            query.normals,
            np.repeat(rotation[:, 0][None, :], 3, axis=0),
            atol=1.0e-12,
        )

    def test_ransac_recovers_three_orthogonal_faces(self) -> None:
        rng = np.random.default_rng(9)
        config = CubeTrackerConfig(
            maximum_points=9000,
            plane_inlier_threshold_m=0.003,
            ransac_iterations=160,
            minimum_face_points=250,
            minimum_face_fraction=0.03,
        )
        rotation = axis_angle_rotation(np.array((0.2, 0.7, 0.4)), 0.72)
        center = np.array((0.08, 0.04, -0.95))
        half_side = 0.5 * config.cube_side_m
        camera_direction = -center / np.linalg.norm(center)
        points: list[np.ndarray] = []
        expected_normals: list[np.ndarray] = []
        for axis_index in range(3):
            axis = rotation[:, axis_index]
            normal = axis if np.dot(axis, camera_direction) >= 0.0 else -axis
            expected_normals.append(normal)
            tangents = [
                rotation[:, index] for index in range(3) if index != axis_index
            ]
            coordinates = rng.uniform(-half_side, half_side, size=(1200, 2))
            face = (
                center
                + half_side * normal
                + coordinates[:, :1] * tangents[0]
                + coordinates[:, 1:] * tangents[1]
            )
            face += rng.normal(0.0, 0.0006, size=face.shape)
            points.append(face)
        cloud = np.vstack(points)
        candidates = extract_plane_candidates(cloud, config, rng)
        selected = select_orthogonal_faces(
            candidates,
            config.orthogonality_tolerance_deg,
        )
        self.assertIsNotNone(selected)
        assert selected is not None
        recovered = np.stack(tuple(plane.normal for plane in selected))
        agreement = np.abs(recovered @ np.stack(expected_normals).T)
        self.assertGreater(float(np.min(np.max(agreement, axis=1))), 0.995)
        self.assertLess(max(plane.rms_m for plane in selected), 0.001)

    def test_two_faces_recover_center_from_shared_edge_span(self) -> None:
        rng = np.random.default_rng(19)
        config = CubeTrackerConfig(
            maximum_points=6000,
            plane_inlier_threshold_m=0.003,
            ransac_iterations=160,
            minimum_face_points=250,
            minimum_face_fraction=0.03,
        )
        rotation = axis_angle_rotation(np.array((0.4, 0.6, 0.2)), 0.67)
        center = np.array((-0.06, 0.03, -0.82))
        half_side = 0.5 * config.cube_side_m
        camera_direction = -center / np.linalg.norm(center)
        points: list[np.ndarray] = []
        for axis_index in (0, 1):
            axis = rotation[:, axis_index]
            normal = axis if np.dot(axis, camera_direction) >= 0.0 else -axis
            tangents = [
                rotation[:, index] for index in range(3) if index != axis_index
            ]
            coordinates = rng.uniform(-half_side, half_side, size=(1600, 2))
            face = (
                center
                + half_side * normal
                + coordinates[:, :1] * tangents[0]
                + coordinates[:, 1:] * tangents[1]
            )
            face += rng.normal(0.0, 0.0005, size=face.shape)
            points.append(face)
        cloud = np.vstack(points)
        candidates = extract_plane_candidates(cloud, config, rng)
        selected = select_orthogonal_pair(
            candidates,
            config.orthogonality_tolerance_deg,
        )
        self.assertIsNotNone(selected)
        assert selected is not None
        estimated_center, estimated_rotation, observed_span = cube_pose_from_faces(
            cloud,
            selected,
            config.cube_side_m,
        )
        self.assertTrue(two_face_span_is_valid(observed_span, config))
        self.assertLess(abs(observed_span - config.cube_side_m), 0.006)
        self.assertLess(float(np.linalg.norm(estimated_center - center)), 0.003)
        agreement = np.abs(rotation.T @ estimated_rotation)
        self.assertGreater(float(np.min(np.max(agreement, axis=1))), 0.995)
        refined = refine_cube_pose(
            cloud,
            selected,
            estimated_center,
            estimated_rotation,
            config,
        )
        self.assertLess(float(np.linalg.norm(refined.center_m - center)), 0.004)
        self.assertTrue(np.isfinite(refined.surface_rms_m))
        self.assertGreater(
            float(np.min(np.linalg.eigvalsh(refined.covariance))),
            0.0,
        )

    def test_joint_refinement_reduces_pose_surface_residual_and_returns_covariance(
        self,
    ) -> None:
        rng = np.random.default_rng(27)
        config = CubeTrackerConfig(
            maximum_points=9000,
            plane_inlier_threshold_m=0.003,
            ransac_iterations=180,
            minimum_face_points=250,
            minimum_face_fraction=0.03,
            pose_refinement_huber_m=0.0025,
        )
        true_rotation = axis_angle_rotation(np.array((0.3, 0.5, 0.8)), 0.59)
        true_center = np.array((0.06, 0.025, -0.88))
        half_side = 0.5 * config.cube_side_m
        camera_direction = -true_center / np.linalg.norm(true_center)
        clouds: list[np.ndarray] = []
        for axis_index in range(3):
            axis = true_rotation[:, axis_index]
            normal = axis if np.dot(axis, camera_direction) >= 0.0 else -axis
            tangents = [
                true_rotation[:, index]
                for index in range(3)
                if index != axis_index
            ]
            coordinates = rng.uniform(-half_side, half_side, size=(1000, 2))
            face = (
                true_center
                + half_side * normal
                + coordinates[:, :1] * tangents[0]
                + coordinates[:, 1:] * tangents[1]
            )
            face += rng.normal(0.0, 0.0007, size=face.shape)
            clouds.append(face)
        cloud = np.vstack(clouds)
        candidates = extract_plane_candidates(cloud, config, rng)
        faces = select_orthogonal_faces(
            candidates,
            config.orthogonality_tolerance_deg,
        )
        self.assertIsNotNone(faces)
        assert faces is not None

        initial_center, initial_rotation, _ = cube_pose_from_faces(
            cloud,
            faces,
            config.cube_side_m,
        )
        perturbed_center = initial_center + np.array((0.004, -0.003, 0.002))
        perturbed_rotation = (
            axis_angle_rotation(np.array((0.6, -0.2, 0.4)), 0.035)
            @ initial_rotation
        )
        fit_indices = np.unique(
            np.concatenate(tuple(face.point_indices for face in faces))
        )
        before = point_to_cube_surface_distance(
            cloud[fit_indices],
            perturbed_center,
            perturbed_rotation,
            config.cube_side_m,
        )
        refined = refine_cube_pose(
            cloud,
            faces,
            perturbed_center,
            perturbed_rotation,
            config,
        )

        self.assertLess(
            refined.surface_rms_m,
            float(np.sqrt(np.mean(np.square(before)))),
        )
        self.assertLess(
            float(np.linalg.norm(refined.center_m - true_center)),
            float(np.linalg.norm(perturbed_center - true_center)),
        )
        np.testing.assert_allclose(
            refined.covariance,
            refined.covariance.T,
            atol=1.0e-12,
        )
        self.assertGreater(float(np.min(np.linalg.eigvalsh(refined.covariance))), 0.0)

    def test_incomplete_two_face_span_is_rejected(self) -> None:
        config = CubeTrackerConfig()
        incomplete_span = 0.70 * config.cube_side_m
        self.assertFalse(two_face_span_is_valid(incomplete_span, config))

    def test_opencv_hsv_yellow_defaults_include_bright_yellow(self) -> None:
        bgr = np.uint8([[[0, 255, 255]]])
        hue = int(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0, 0])
        config = CubeTrackerConfig()
        self.assertGreaterEqual(hue, config.hue_min)
        self.assertLessEqual(hue, config.hue_max)

    def test_yellow_fragments_remain_separate_but_are_pooled(self) -> None:
        image = np.zeros((180, 240, 3), dtype=np.uint8)
        image[20:160, 30:210] = (0, 255, 255)
        image[20:160, 110:130] = 0
        mask = segment_yellow(
            image,
            CubeTrackerConfig(minimum_mask_area_px=1000),
        )

        self.assertEqual(int(mask[80, 70]), 255)
        self.assertEqual(int(mask[80, 170]), 255)
        self.assertEqual(int(mask[80, 120]), 0)
        component_count, _ = cv2.connectedComponents(mask, connectivity=8)
        self.assertEqual(component_count - 1, 2)

    def test_cable_exclusion_removes_a_dilated_boundary(self) -> None:
        yellow = np.full((40, 60), 255, dtype=np.uint8)
        cable = np.zeros_like(yellow)
        cable[:, 30] = 255
        cleaned = remove_cable_pixels(yellow, cable, exclusion_radius_px=3)

        self.assertTrue(np.all(cleaned[:, 27:34] == 0))
        self.assertTrue(np.all(cleaned[:, :26] == 255))
        self.assertTrue(np.all(cleaned[:, 35:] == 255))
        self.assertTrue(np.all(yellow == 255))
        self.assertEqual(int(np.count_nonzero(cable)), 40)

    def test_temporal_cube_filter_estimates_motion_and_expires_prediction(
        self,
    ) -> None:
        config = CubeStateFilterConfig(
            initial_linear_velocity_std_mps=0.5,
            initial_angular_velocity_std_rps=1.0,
            maximum_prediction_age_s=0.25,
            maximum_prediction_step_s=0.05,
        )
        state_filter = RigidCubeStateFilter(config)
        covariance = np.diag(
            (
                0.001**2,
                0.001**2,
                0.001**2,
                np.deg2rad(0.5) ** 2,
                np.deg2rad(0.5) ** 2,
                np.deg2rad(0.5) ** 2,
            )
        )
        first = state_filter.update(
            refined_measurement(np.array((0.0, 0.0, -0.8)), np.eye(3), covariance),
            1.0,
        )
        self.assertTrue(first.valid)
        self.assertTrue(first.measurement_used)

        second_rotation = axis_angle_rotation(np.array((0.0, 1.0, 0.0)), 0.04)
        second = state_filter.update(
            refined_measurement(
                np.array((0.010, 0.0, -0.8)),
                second_rotation,
                covariance,
            ),
            1.1,
        )
        self.assertTrue(second.valid)
        assert second.linear_velocity_mps is not None
        assert second.angular_velocity_rps is not None
        self.assertGreater(float(second.linear_velocity_mps[0]), 0.01)
        self.assertGreater(float(second.angular_velocity_rps[1]), 0.01)
        assert second.rotation is not None
        accepted_rotation = second.rotation.copy()

        raw_only = replace(
            refined_measurement(
                np.array((0.020, 0.0, -0.8)),
                second_rotation,
                covariance,
            ),
            refinement_valid=False,
            refinement_reason="synthetic refinement rejection",
        )
        predicted = state_filter.update(raw_only, 1.2)
        self.assertTrue(predicted.valid)
        self.assertFalse(predicted.measurement_used)
        assert predicted.center_m is not None
        assert second.center_m is not None
        self.assertGreater(float(predicted.center_m[0]), float(second.center_m[0]))
        assert predicted.rotation is not None
        np.testing.assert_allclose(predicted.rotation, accepted_rotation, atol=1.0e-12)

        expired = state_filter.update(None, 1.4)
        self.assertFalse(expired.valid)
        self.assertTrue(expired.initialized)

    def test_temporal_cube_filter_resolves_symmetry_before_angular_velocity(
        self,
    ) -> None:
        state_filter = RigidCubeStateFilter(CubeStateFilterConfig())
        covariance = np.diag(
            (
                0.001**2,
                0.001**2,
                0.001**2,
                np.deg2rad(0.5) ** 2,
                np.deg2rad(0.5) ** 2,
                np.deg2rad(0.5) ** 2,
            )
        )
        rotation = axis_angle_rotation(np.array((0.3, 0.7, 0.2)), 0.35)
        first = state_filter.update(
            refined_measurement(
                np.array((0.0, 0.0, -0.8)),
                rotation,
                covariance,
            ),
            1.0,
        )
        equivalent_measurement = rotation @ CUBE_SYMMETRIES[1]
        second = state_filter.update(
            refined_measurement(
                np.array((0.0, 0.0, -0.8)),
                equivalent_measurement,
                covariance,
            ),
            1.1,
        )

        self.assertTrue(second.measurement_used)
        assert first.rotation is not None
        assert second.rotation is not None
        assert second.angular_velocity_rps is not None
        np.testing.assert_allclose(second.rotation, first.rotation, atol=1.0e-10)
        np.testing.assert_allclose(
            second.angular_velocity_rps,
            np.zeros(3),
            atol=1.0e-10,
        )

    def test_temporal_cube_filter_rejects_pose_innovation_outlier(self) -> None:
        state_filter = RigidCubeStateFilter(CubeStateFilterConfig())
        covariance = np.eye(6, dtype=np.float64) * 1.0e-6
        initial = state_filter.update(
            refined_measurement(
                np.array((0.0, 0.0, -0.8)),
                np.eye(3),
                covariance,
            ),
            1.0,
        )
        self.assertTrue(initial.measurement_used)

        rejected = state_filter.update(
            refined_measurement(
                np.array((0.5, 0.0, -0.8)),
                np.eye(3),
                covariance,
            ),
            1.1,
        )
        self.assertTrue(rejected.valid)
        self.assertFalse(rejected.measurement_used)
        self.assertIn("innovation chi2", rejected.reason)
        assert rejected.center_m is not None
        self.assertLess(abs(float(rejected.center_m[0])), 0.05)


if __name__ == "__main__":
    unittest.main()
