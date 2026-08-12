from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

from cable_twin.shared.observation_data import (
    RECONSTRUCTION_KINDS,
    MetricCurveSequence,
    initial_metric_trajectory,
    load_stored_trajectory,
    save_trajectory,
)
from cable_twin.shared.contracts import (
    POINT_CLOUD_CURVE_METHOD,
    PartialCurveObservation,
    StereoFrame,
    ViewObservation,
)
from cable_twin.shared.config import RouteSettings
from cable_twin.shared.metric_curve import (
    MetricCurveObservation,
    lift_partial_curve_from_registered_depth,
)
from cable_twin.shared.saving import ObservationSequenceWriter, validate_observation_archive
from cable_twin.shared.routes import extract_partial_curve


class PointCloudEvidenceTests(unittest.TestCase):
    @staticmethod
    def _metadata() -> dict:
        intrinsics = [[80.0, 0.0, 24.0], [0.0, 80.0, 16.0], [0.0, 0.0, 1.0]]
        return {
            "method": POINT_CLOUD_CURVE_METHOD,
            "settings": {"cable_identity": 1},
            "source": {
                "calibration": {
                    "left_intrinsics": intrinsics,
                    "right_intrinsics": intrinsics,
                    "baseline_m": 0.12,
                },
                "capture_manifest": {"experiment": {"cable_identity": 1}},
            },
        }

    def test_archive_contains_registered_depth_curve(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "observation.npz"
            writer = ObservationSequenceWriter(path, metadata=self._metadata(), route_samples=24)
            image = np.zeros((32, 48, 3), dtype=np.uint8)
            depth = np.ones((32, 48), dtype=np.float32)
            mask = np.zeros((32, 48), dtype=np.uint8)
            points = np.stack((np.linspace(5.0, 40.0, 24), np.linspace(8.0, 22.0, 24)), axis=1)
            curve = PartialCurveObservation(
                points, np.ones(24, bool), np.zeros(24, np.int16),
                points[[0, -1]], np.ones(2, bool), 2, 1,
            )
            view = ViewObservation(mask, mask, curve)
            xyz = np.column_stack(((points[:, 0] - 24.0) / 80.0, (points[:, 1] - 16.0) / 80.0, np.ones(24)))
            metric = MetricCurveObservation(
                xyz, np.ones(24, bool), np.zeros(24), np.full(24, 3),
                xyz[[0, -1]], np.ones(2, bool), np.zeros(2), np.full(2, 3),
            )
            writer.append(
                StereoFrame(0, 0, 1_000_000_000, image, depth, gravity_camera_m_s2=np.asarray((0.0, 9.8, 0.0))),
                view,
                metric,
                timings_ms={"pidnet_ms": 10.0},
            )
            writer.close()
            with np.load(path, allow_pickle=False) as data:
                metadata = validate_observation_archive(data)
                self.assertEqual(metadata["method"], POINT_CLOUD_CURVE_METHOD)
                self.assertEqual(data["metric_points_m"].shape, (1, 24, 3))
                self.assertEqual(data["route_valid"].shape, (1, 24))
                self.assertEqual(data["metric_endpoint_points_m"].shape, (1, 2, 3))
                self.assertNotIn("projected_right_xy", data.files)

    def test_depth_lift_rejects_background_island(self) -> None:
        height, width, samples = 64, 96, 32
        mask = np.zeros((height, width), dtype=np.uint8)
        route_xy = np.column_stack((np.linspace(10, 85, samples), np.full(samples, 32.0)))
        for x, y in np.rint(route_xy).astype(int):
            mask[y - 2 : y + 3, x - 2 : x + 3] = 255
        depth = np.full((height, width), np.nan, dtype=np.float32)
        depth[mask != 0] = 1.0
        depth[29:36, 45:52] = 1.45
        curve = PartialCurveObservation(
            route_xy, np.ones(samples, bool), np.zeros(samples, np.int16),
            route_xy[[0, -1]], np.ones(2, bool), 2, 1,
        )
        view = ViewObservation(mask, mask, curve)
        metric = lift_partial_curve_from_registered_depth(
            view, depth, np.asarray(self._metadata()["source"]["calibration"]["left_intrinsics"]),
            radius_px=3, minimum_support=3, maximum_cluster_span_m=0.025,
            depth_min_m=0.2, depth_max_m=4.0,
        )
        self.assertGreater(np.count_nonzero(metric.valid), 20)
        self.assertLess(float(np.nanmax(metric.points_camera_m[:, 2])), 1.05)

    def test_observed_endpoints_cap_disconnected_visible_segments(self) -> None:
        body = np.zeros((80, 120), dtype=np.uint8)
        body[39:42, 5:45] = 255
        body[39:42, 55:105] = 255
        endpoints = np.zeros_like(body)
        endpoints[38:43, 13:18] = 255
        endpoints[38:43, 93:98] = 255
        observation = extract_partial_curve(
            body,
            endpoints,
            body_component_count=2,
            settings=RouteSettings(
                endpoint_min_area_px=2,
                dense_samples=64,
                maximum_segments=4,
                minimum_segment_length_px=10.0,
            ),
        )
        curve = observation.curve
        ranges = sorted(
            (
                float(curve.points_xy[selected, 0].min()),
                float(curve.points_xy[selected, 0].max()),
            )
            for segment_id in np.unique(curve.segment_ids[curve.point_valid])
            for selected in [curve.point_valid & (curve.segment_ids == segment_id)]
        )
        self.assertEqual(len(ranges), 2)
        self.assertAlmostEqual(ranges[0][0], 15.0, places=6)
        self.assertLessEqual(ranges[0][1], 45.0)
        self.assertGreaterEqual(ranges[1][0], 54.0)
        self.assertAlmostEqual(ranges[1][1], 95.0, places=6)

    def test_initializer_uses_smooth_metric_depth_and_fixed_curve_order(self) -> None:
        frames, samples = 8, 64
        route = np.column_stack((np.linspace(100.0, 520.0, samples), 200.0 + 35.0 * np.sin(np.linspace(0, np.pi, samples))))
        z = 1.0 + 0.03 * np.sin(np.linspace(0, np.pi, samples))
        k = np.asarray([[800.0, 0.0, 320.0], [0.0, 800.0, 180.0], [0.0, 0.0, 1.0]])
        xyz = np.column_stack(((route[:, 0] - 320.0) * z / 800.0, (route[:, 1] - 180.0) * z / 800.0, z))
        metadata = self._metadata()
        metadata["source"]["calibration"]["left_intrinsics"] = k.tolist()
        sequence = MetricCurveSequence(
            Path("synthetic.npz"), metadata,
            1_000_000_000 + np.arange(frames) * 33_333_333,
            np.arange(frames), np.repeat(route[None], frames, axis=0),
            np.ones((frames, samples), bool), np.zeros((frames, samples), np.int16),
            np.repeat(route[[0, -1]][None], frames, axis=0), np.ones((frames, 2), bool),
            np.repeat(xyz[None], frames, axis=0), np.ones((frames, samples), bool),
            np.repeat(xyz[[0, -1]][None], frames, axis=0), np.ones((frames, 2), bool),
            np.full((frames, samples), 0.001), np.full((frames, samples), 5),
            np.full((frames, 2), 0.001), np.full((frames, 2), 5),
            np.tile((0.0, 9.80665, 0.0), (frames, 1)),
        )
        positions, usable = initial_metric_trajectory(sequence, 24, 0.518)
        self.assertEqual(positions.shape, (frames, 24, 3))
        self.assertTrue(np.all(usable))
        self.assertTrue(np.all(np.isfinite(positions)))

    def test_prediction_trajectory_cannot_be_reused_as_reconstruction(self) -> None:
        frames, samples, nodes = 3, 24, 8
        metadata = self._metadata()
        route = np.zeros((frames, samples, 2), dtype=np.float64)
        metric = np.zeros((frames, samples, 3), dtype=np.float64)
        with tempfile.TemporaryDirectory() as temporary:
            observation = Path(temporary) / "observation.npz"
            observation.write_bytes(b"exact observation identity")
            sequence = MetricCurveSequence(
                observation, metadata, np.arange(frames, dtype=np.int64) + 1,
                np.arange(frames, dtype=np.int64), route,
                np.ones((frames, samples), bool), np.zeros((frames, samples), np.int16),
                np.zeros((frames, 2, 2)), np.ones((frames, 2), bool), metric,
                np.ones((frames, samples), bool), np.zeros((frames, 2, 3)),
                np.ones((frames, 2), bool), np.zeros((frames, samples)),
                np.ones((frames, samples), np.int16), np.zeros((frames, 2)),
                np.ones((frames, 2), np.int16), np.tile((0.0, 9.8, 0.0), (frames, 1)),
            )
            trajectory = Path(temporary) / "prediction.npz"
            state = np.zeros((frames, nodes, 3), dtype=np.float32)
            save_trajectory(
                trajectory,
                sequence=sequence,
                positions_m=state,
                velocities_m_s=state,
                metrics={"trajectory_kind": "held_out_recursive_prediction"},
                model_path=None,
            )
            with self.assertRaisesRegex(ValueError, "not a reconstruction cache"):
                load_stored_trajectory(
                    trajectory,
                    sequence=sequence,
                    node_count=nodes,
                    accepted_kinds=RECONSTRUCTION_KINDS,
                )


if __name__ == "__main__":
    unittest.main()
