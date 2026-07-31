"""Deterministic synthetic checks for the isolated cube geometry."""

from __future__ import annotations

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
    CubeTrackerConfig,
    CUBE_SYMMETRIES,
    cube_pose_from_faces,
    extract_plane_candidates,
    point_to_cube_surface_distance,
    remove_cable_pixels,
    segment_yellow,
    select_orthogonal_pair,
    select_orthogonal_faces,
    two_face_span_is_valid,
)


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


if __name__ == "__main__":
    unittest.main()
