from __future__ import annotations

import csv
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from optitrack_offline.config import (
    CONFIG_SCHEMA,
    DEFAULT_CONFIG,
    OptitrackFitConfig,
    ValidationSettings,
    load_config,
    save_config,
)
from optitrack_offline.data import MotiveCableTake, load_motive_cable_csv
from optitrack_offline.fitting import (
    _causal_quadratic_velocity,
    _local_quadratic_velocity,
    marker_positions_to_rod,
    marker_positions_to_feasible_rod,
)


class MotiveCsvTests(unittest.TestCase):
    def test_refined_initialization_preserves_sites_and_exact_edge_lengths(self) -> None:
        cable = DEFAULT_CONFIG.cable
        markers = np.zeros((cable.marker_count, 3), dtype=np.float64)
        coordinates = np.asarray(cable.marker_material_coordinates_m)
        markers[:, 0] = 0.92 * coordinates
        markers[:, 2] = 0.025 * np.sin(np.pi * coordinates / cable.length_m)
        rod, correction = marker_positions_to_feasible_rod(markers, cable)
        np.testing.assert_allclose(
            rod[np.asarray(cable.marker_node_indices)],
            markers,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            np.linalg.vector_norm(np.diff(rod, axis=0), axis=1),
            cable.rod_rest_lengths_m,
            atol=3.0e-9,
        )
        self.assertAlmostEqual(float(correction), 0.0, delta=1.0e-12)

    def test_default_instrumented_cable_matches_measurements(self) -> None:
        cable = DEFAULT_CONFIG.cable
        self.assertEqual(cable.marker_count, 11)
        self.assertAlmostEqual(cable.length_m, 0.943)
        self.assertAlmostEqual(sum(cable.vertex_masses_kg), cable.total_dynamic_mass_kg)
        self.assertAlmostEqual(cable.bare_cable_mass_kg, 0.007)
        self.assertEqual(len(cable.moving_marker_masses_kg), 10)
        self.assertTrue(
            all(
                math.isclose(value, 0.010 / 11.0)
                for value in cable.moving_marker_masses_kg
            )
        )
        self.assertEqual(cable.moving_marker_count, 10)
        self.assertAlmostEqual(cable.total_dynamic_mass_kg, 0.007 + 0.100 / 11.0)
        self.assertEqual(cable.node_count, 21)
        self.assertEqual(cable.marker_node_indices, tuple(range(0, 21, 2)))
        self.assertEqual(DEFAULT_CONFIG.fit.window_frames, 100)
        self.assertEqual(DEFAULT_CONFIG.fit.window_stride, 100)
        self.assertEqual(DEFAULT_CONFIG.fit.substeps, 3)
        moving_nodes = np.asarray(cable.marker_node_indices[1:])
        bare_only_mass = cable.bare_cable_mass_kg / cable.length_m
        cable_vertex_mass = np.asarray(cable.vertex_masses_kg).copy()
        for node, marker_mass in zip(moving_nodes, cable.moving_marker_masses_kg):
            cable_vertex_mass[node] -= marker_mass
        self.assertAlmostEqual(float(np.sum(cable_vertex_mass)), bare_only_mass * cable.length_m)

    def test_marker_positions_select_exact_nodes_of_refined_rod(self) -> None:
        cable = DEFAULT_CONFIG.cable
        markers = np.zeros((cable.marker_count, 3), dtype=np.float64)
        markers[:, 0] = np.cumsum((0.0, *(0.98 * np.asarray(cable.rest_lengths_m))))
        rod = marker_positions_to_rod(markers, DEFAULT_CONFIG)
        self.assertEqual(rod.shape, (cable.node_count, 3))
        np.testing.assert_allclose(rod[list(cable.marker_node_indices)], markers)
        np.testing.assert_allclose(
            rod[1::2, 0],
            0.5 * (markers[:-1, 0] + markers[1:, 0]),
            atol=1.0e-12,
        )
        self.assertFalse(np.shares_memory(rod, markers))

    def test_simulated_node_count_is_tunable_by_uniform_refinement(self) -> None:
        cable = replace(
            DEFAULT_CONFIG.cable,
            rod_segments_per_marker_interval=3,
        )
        self.assertEqual(cable.marker_count, 11)
        self.assertEqual(cable.node_count, 31)
        self.assertEqual(cable.marker_node_indices, tuple(range(0, 31, 3)))

    def test_local_quadratic_velocity_uses_positions_only_for_velocity(self) -> None:
        timestamps = np.arange(7, dtype=np.float64) * 0.01
        positions = np.zeros((7, 2, 3), dtype=np.float64)
        positions[:, :, 0] = timestamps[:, None] + 2.0 * timestamps[:, None] ** 2
        complete = np.ones(7, dtype=bool)
        velocity = _local_quadratic_velocity(
            positions,
            timestamps,
            complete,
            window_frames=5,
            maximum_dt_s=0.02,
        )
        np.testing.assert_allclose(
            velocity[:, :, 0],
            np.repeat((1.0 + 4.0 * timestamps)[:, None], 2, axis=1),
            atol=1.0e-10,
        )

    def test_causal_velocity_does_not_use_future_positions(self) -> None:
        timestamps = np.arange(8, dtype=np.float64) * 0.01
        positions = np.zeros((8, 2, 3), dtype=np.float64)
        positions[:, :, 0] = timestamps[:, None] + timestamps[:, None] ** 2
        complete = np.ones(8, dtype=bool)
        baseline = _causal_quadratic_velocity(
            positions,
            timestamps,
            complete,
            window_frames=5,
            maximum_dt_s=0.02,
        )
        changed = positions.copy()
        changed[5:] = 1000.0
        repeated = _causal_quadratic_velocity(
            changed,
            timestamps,
            complete,
            window_frames=5,
            maximum_dt_s=0.02,
        )
        np.testing.assert_allclose(repeated[4], baseline[4])

    def test_identification_and_validation_settings_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = OptitrackFitConfig(
                cable=DEFAULT_CONFIG.cable,
                fit=DEFAULT_CONFIG.fit,
                model_path=root / "model.json",
                validation=ValidationSettings(history_frames=7),
            )
            path = root / "config.json"
            save_config(expected, path)
            serialized = json.loads(path.read_text(encoding="utf-8"))
            actual = load_config(path)
        self.assertEqual(actual.fit, expected.fit)
        self.assertEqual(actual.fit.optimizer_iterations, 20)
        self.assertEqual(actual.fit.initializer_candidates, 24)
        self.assertEqual(actual.fit.optimizer_batch_windows, 64)
        self.assertAlmostEqual(actual.fit.optimizer_learning_rate, 0.02)
        self.assertEqual(actual.fit.constraint_iterations, 4)
        self.assertEqual(actual.validation, expected.validation)
        self.assertEqual(actual.model_path, expected.model_path)
        self.assertEqual(serialized["schema"], CONFIG_SCHEMA)
        self.assertEqual(serialized["validation"], {"history_frames": 7})
        self.assertNotIn("gj_min_n_m2", serialized["fit"])
        self.assertNotIn("gj_max_n_m2", serialized["fit"])
        self.assertNotIn("maximum_endpoint_angular_speed_rad_s", serialized["fit"])
        self.assertIn("maximum_attachment_speed_m_s", serialized["fit"])
        self.assertNotIn("maximum_endpoint_speed_m_s", serialized["fit"])
        self.assertNotIn("warm_start_from_model", serialized["fit"])

    def test_two_endpoint_config_is_rejected_instead_of_migrated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cable = asdict(DEFAULT_CONFIG.cable)
            cable["rod_segments_per_marker_interval"] = 2
            fit = asdict(DEFAULT_CONFIG.fit)
            fit.pop("optimizer_iterations")
            fit["search_refinement_levels"] = 3
            fit["optimizer_learning_rate"] = 0.15
            fit["warm_start_from_model"] = True
            path = root / "legacy.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "optitrack_cable_fit_config_v3",
                        "cable": cable,
                        "fit": fit,
                        "validation": asdict(DEFAULT_CONFIG.validation),
                        "model_path": "model.json",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "one-attachment/free-tip"):
                load_config(path)

    @staticmethod
    def _motive_rows(rigid_body_names: tuple[str, ...]) -> list[list[str]]:
        marker_names = [f"cable1:c{index}" for index in range(1, 11)]
        blocks = [
            block
            for body_name in rigid_body_names
            for block in (
                ("Rigid Body", body_name, "Position", ("X", "Y", "Z")),
                ("Rigid Body", body_name, "Error Per Marker", ("Mean",)),
            )
        ]
        blocks.extend(
            ("Marker", name, "Position", ("X", "Y", "Z"))
            for name in marker_names
        )
        kinds = ["", ""]
        names = ["", ""]
        properties = ["", ""]
        axes = ["Frame", "Time (Seconds)"]
        for kind, name, prop, block_axes in blocks:
            kinds.extend([kind] * len(block_axes))
            names.extend([name] * len(block_axes))
            properties.extend([prop] * len(block_axes))
            axes.extend(block_axes)
        rows = [
            [
                "Format Version", "1.22", "Take Name", "one_free_tip",
                "Capture Frame Rate", "100", "Export Frame Rate", "100",
                "Length Units", "Meters", "Coordinate Space", "Global",
            ],
            [],
            kinds,
            names,
            ["", "", *(["id"] * (len(axes) - 2))],
            properties,
            axes,
        ]
        body_values = [value for index, _name in enumerate(rigid_body_names) for value in (index, 1.0, 0.0, 0.0004)]
        marker_values = [
            value
            for index in range(1, 11)
            for value in (0.09 * index, 1.0, 0.0)
        ]
        rows.append(["0", "0.000", *body_values, *marker_values])
        return rows

    def test_loads_one_position_only_attachment_and_c1_through_c10(self) -> None:
        rows = self._motive_rows(("drone_attachment",))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "take.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerows(rows)
            take = load_motive_cable_csv(path)

        self.assertTrue(take.has_attachment)
        self.assertEqual(take.attachment_rigid_body_name, "drone_attachment")
        self.assertEqual(
            take.marker_names,
            ("Attachment", *(f"c{index}" for index in range(1, 11))),
        )
        self.assertEqual(take.positions_m.shape, (1, 11, 3))
        np.testing.assert_array_equal(take.observed, np.ones((1, 11), dtype=bool))
        np.testing.assert_array_equal(take.attachment_observed, (True,))
        self.assertAlmostEqual(float(take.attachment_error_m[0]), 0.0004)
        np.testing.assert_allclose(take.positions_m[0, 0], (0.0, 0.0, 1.0))
        np.testing.assert_allclose(take.positions_m[0, -1], (0.9, 0.0, 1.0))
        self.assertEqual(take.axis_convention, "Project right-handed Z-up")

    def test_marker_only_export_is_rejected(self) -> None:
        rows = self._motive_rows(())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "marker_only.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerows(rows)
            with self.assertRaisesRegex(ValueError, "exactly one exported rigid-body"):
                load_motive_cable_csv(path)

    def test_former_two_holder_export_is_rejected(self) -> None:
        rows = self._motive_rows(("endpoint1", "endpoint2"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "two_holders.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerows(rows)
            with self.assertRaisesRegex(ValueError, "former two-holder format"):
                load_motive_cable_csv(path)


if __name__ == "__main__":
    unittest.main()
