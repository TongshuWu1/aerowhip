from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest

import numpy as np

from cable_twin.offline.optimize import _contiguous_runs, _local_quadratic
from cable_twin.offline.planar_data import (
    PLANAR_OBSERVATION_METHOD,
    PlanarSequenceWriter,
    load_planar_sequence,
)
from cable_twin.shared.config import RouteSettings
from cable_twin.shared.contracts import StereoFrame
from cable_twin.shared.model_artifact import load_dder_artifact
from cable_twin.shared.routes import extract_complete_endpoint_route


class PlanarObservationTests(unittest.TestCase):
    def test_endpoint_route_is_complete_and_ordered(self) -> None:
        height, width = 120, 180
        body = np.zeros((height, width), dtype=np.uint8)
        endpoint = np.zeros_like(body)
        skeleton = np.zeros_like(body, dtype=bool)
        x = np.arange(25, 156)
        y = np.rint(45.0 + 0.0025 * (x - 90) ** 2).astype(np.int32)
        body[y, x] = 255
        skeleton[y, x] = True
        for offset in (-2, -1, 1, 2):
            body[np.clip(y + offset, 0, height - 1), x] = 255
        endpoint[y[0] - 2 : y[0] + 3, x[0] - 2 : x[0] + 3] = 255
        endpoint[y[-1] - 2 : y[-1] + 3, x[-1] - 2 : x[-1] + 3] = 255
        result = extract_complete_endpoint_route(
            body,
            endpoint,
            1,
            RouteSettings(2, 64, 1, 20.0),
            skeleton=skeleton,
        )
        self.assertIsNone(result.failure)
        self.assertTrue(bool(np.all(result.curve.point_valid)))
        self.assertTrue(bool(np.all(result.curve.endpoint_valid)))
        np.testing.assert_allclose(
            result.curve.points_xy[[0, -1]],
            result.curve.endpoint_centers_xy,
        )

    def test_complete_route_rejects_extra_endpoint_component(self) -> None:
        height, width = 120, 180
        body = np.zeros((height, width), dtype=np.uint8)
        endpoints = np.zeros_like(body)
        skeleton = np.zeros_like(body, dtype=bool)
        skeleton[60, 25:156] = True
        body[58:63, 25:156] = 255
        endpoints[56:65, 21:30] = 255
        endpoints[56:65, 151:160] = 255
        endpoints[15:24, 85:94] = 255
        result = extract_complete_endpoint_route(
            body,
            endpoints,
            1,
            RouteSettings(2, 64, 1, 20.0),
            skeleton=skeleton,
        )
        self.assertEqual(
            result.failure,
            "Planar observation requires exactly two endpoint components.",
        )

    def test_planar_archive_round_trip(self) -> None:
        routes_px = (
            np.column_stack((np.linspace(10.0, 90.0, 24), np.full(24, 30.0))),
            np.column_stack((np.linspace(0.0, 100.0, 24), np.full(24, 30.0))),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "planar.npz"
            writer = PlanarSequenceWriter(
                path,
                metadata={"method": PLANAR_OBSERVATION_METHOD},
                route_samples=24,
                cable_length_m=0.45,
                image_width_px=100,
                image_height_px=60,
            )
            for index, route_px in enumerate(routes_px):
                curve = SimpleNamespace(
                    points_xy=route_px,
                    endpoint_centers_xy=route_px[[0, -1]],
                )
                observed = SimpleNamespace(
                    view=SimpleNamespace(curve=curve, failure=None),
                    complete=True,
                    timings_ms={
                        "pidnet_ms": 1.0,
                        "mask_postprocess_ms": 2.0,
                        "route_ms": 3.0,
                        "total_ms": 6.0,
                    },
                )
                frame = StereoFrame(
                    sequence_index=index,
                    source_position=index,
                    timestamp_ns=1_000_000_000 + index * 33_000_000,
                    left_bgr=np.zeros((60, 100, 3), dtype=np.uint8),
                    depth_m=None,
                )
                writer.append(frame, observed)
            writer.close()
            loaded = load_planar_sequence(path)
        self.assertEqual(loaded.frame_count, 2)
        self.assertTrue(bool(np.all(loaded.complete)))
        scale = float(loaded.metadata["image_plane_mapping"]["scale_m_per_px"])
        self.assertAlmostEqual(scale, 0.45 / 90.0)
        metric_lengths = np.linalg.norm(
            np.diff(loaded.route_xz_m, axis=1), axis=2
        ).sum(axis=1)
        np.testing.assert_allclose(metric_lengths, (0.4, 0.5), atol=1.0e-6)

    def test_planar_fit_artifact_loads_shared_dder_model(self) -> None:
        payload = {
            "schema": "planar_rgb_reference_dder_v2",
            "cable_identity": 1,
            "measured": {
                "node_count": 24,
                "length_m": 0.518,
                "mass_kg": 0.03,
                "diameter_m": 0.004,
            },
            "optimized": {
                "bending_stiffness_n_m2": 8.0e-6,
                "bending_damping_n_m2_s": 2.0e-8,
            },
            "solver": {"substeps": 8, "constraint_iterations": 8},
            "fit": {"status": "completed"},
            "observation_model": {
                "degrees_of_freedom": 4.0,
                "unresolved_curve_scale_m": 0.002,
                "unresolved_endpoint_scale_m": 0.001,
            },
            "process_model": {
                "interior_acceleration_sigma_m_s2": 0.2,
                "unobserved_endpoint_acceleration_sigma_m_s2": 0.1,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            artifact = load_dder_artifact(path)
            model = artifact.make_model((0.0, 0.0, -9.81))
        self.assertEqual(model.parameters.node_count, 24)
        self.assertEqual(model.parameters.bending_damping_n_m2_s, 2.0e-8)

    def test_local_quadratic_recovers_velocity_without_frame_difference_noise(self) -> None:
        timestamp = np.arange(11, dtype=np.int64) * 33_333_333 + 1_000_000_000
        time = (timestamp - timestamp[5]) * 1.0e-9
        values = np.zeros((11, 2, 2), dtype=np.float64)
        values[:, :, 0] = (0.2 + 0.4 * time + 0.1 * np.square(time))[:, None]
        values[:, :, 1] = (-0.1 + 0.25 * time)[:, None]
        smooth, velocity = _local_quadratic(
            values,
            timestamp,
            np.ones(11, dtype=bool),
            window=5,
            maximum_dt_s=0.034,
        )
        np.testing.assert_allclose(smooth[5], values[5], atol=1.0e-12)
        np.testing.assert_allclose(velocity[5, :, 0], 0.4, atol=1.0e-10)
        np.testing.assert_allclose(velocity[5, :, 1], 0.25, atol=1.0e-10)

    def test_missing_frame_splits_rollout_windows(self) -> None:
        timestamp = np.arange(8, dtype=np.int64) * 33_000_000 + 1_000_000_000
        valid = np.ones(8, dtype=bool)
        valid[4] = False
        runs = _contiguous_runs(valid, timestamp, 0.034)
        self.assertEqual([run.tolist() for run in runs], [[0, 1, 2, 3], [5, 6, 7]])


if __name__ == "__main__":
    unittest.main()
