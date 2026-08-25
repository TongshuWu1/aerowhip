from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cable_twin.shared.dder import START_PINNED_FREE_END
from drone_mpc.adaptation import (
    SyntheticExcitationSettings,
    TipAdaptationSettings,
    TipOnlyMeasurements,
    estimate_tip_only_parameters,
    fixed_solver_model,
    make_synthetic_tip_measurements,
)
from drone_mpc.model import CableModelSnapshot, load_cable_model
from optitrack_offline.fitting import MODEL_SCHEMA


class DroneAdaptationTests(unittest.TestCase):
    def test_measurement_contract_contains_tip_and_boundary_only(self) -> None:
        names = tuple(field.name for field in fields(TipOnlyMeasurements))

        self.assertEqual(
            names,
            (
                "time_s",
                "attachment_positions_m",
                "free_tip_positions_m",
                "commanded_accelerations_m_s2",
                "source",
            ),
        )
        self.assertFalse(any("state" in name or "cable" in name for name in names))
        measurements = TipOnlyMeasurements(
            time_s=np.array([0.0, 0.01, 0.02]),
            attachment_positions_m=np.zeros((3, 3)),
            free_tip_positions_m=np.zeros((3, 3)),
            commanded_accelerations_m_s2=np.zeros((2, 3)),
        )
        self.assertFalse(measurements.time_s.flags.writeable)
        self.assertFalse(measurements.attachment_positions_m.flags.writeable)
        self.assertFalse(measurements.free_tip_positions_m.flags.writeable)

    def test_synthetic_measurements_are_deterministic_and_tip_only(self) -> None:
        model = self._small_model()
        settings = SyntheticExcitationSettings(
            duration_s=0.08,
            dt_s=0.02,
            maximum_acceleration_m_s2=2.0,
            tip_noise_std_m=0.001,
            noise_seed=73,
        )

        first = make_synthetic_tip_measurements(model, settings, device="cpu")
        second = make_synthetic_tip_measurements(model, settings, device="cpu")

        np.testing.assert_array_equal(first.time_s, second.time_s)
        np.testing.assert_array_equal(
            first.attachment_positions_m, second.attachment_positions_m
        )
        np.testing.assert_array_equal(
            first.free_tip_positions_m, second.free_tip_positions_m
        )
        np.testing.assert_array_equal(
            first.commanded_accelerations_m_s2,
            second.commanded_accelerations_m_s2,
        )
        self.assertEqual(first.frame_count, 5)
        self.assertEqual(first.free_tip_positions_m.shape, (5, 3))

    def test_stationary_record_freezes_at_the_prior(self) -> None:
        model = self._small_model()
        measurements = make_synthetic_tip_measurements(
            model,
            SyntheticExcitationSettings(
                duration_s=0.08,
                dt_s=0.02,
                maximum_acceleration_m_s2=0.0,
            ),
            device="cpu",
        )

        result = estimate_tip_only_parameters(
            model,
            measurements,
            TipAdaptationSettings(optimizer_iterations=2),
            device="cpu",
        )

        self.assertFalse(result.update_applied)
        self.assertIn("insufficient tip sensitivity", result.freeze_reason or "")
        self.assertEqual(result.estimated_ei_scale, 1.0)
        self.assertEqual(result.estimated_cb_scale, 1.0)
        self.assertEqual(result.estimated_ei_n_m2, result.nominal_ei_n_m2)
        self.assertEqual(result.estimated_cb_n_m2_s, result.nominal_cb_n_m2_s)
        self.assertEqual(result.best_iteration, 0)
        self.assertEqual(result.evaluations, 1)

    def test_fixed_solver_is_shared_and_unstable_search_box_is_rejected(self) -> None:
        source = self._snapshot()
        nominal = fixed_solver_model(
            source,
            node_count=6,
            substeps=2,
            constraint_iterations=3,
        )
        hidden = fixed_solver_model(
            source,
            node_count=6,
            substeps=2,
            constraint_iterations=3,
            bending_stiffness_scale=1.2,
            bending_damping_scale=0.8,
        )

        self.assertEqual(nominal.node_count, hidden.node_count)
        self.assertEqual(nominal.model.parameters.substeps, 2)
        self.assertEqual(hidden.model.parameters.substeps, 2)
        self.assertEqual(nominal.model.parameters.constraint_iterations, 3)
        self.assertEqual(hidden.model.parameters.constraint_iterations, 3)
        measurements = make_synthetic_tip_measurements(
            nominal,
            SyntheticExcitationSettings(duration_s=0.04, dt_s=0.02),
            device="cpu",
        )
        maximum_ei = nominal.model.maximum_stable_bending_stiffness(
            0.02,
            pinned_endpoints=START_PINNED_FREE_END,
        )
        unstable_upper_scale = 1.01 * maximum_ei / nominal.bending_stiffness_n_m2
        with self.assertRaisesRegex(ValueError, "fixed adaptation solver is unstable"):
            estimate_tip_only_parameters(
                nominal,
                measurements,
                TipAdaptationSettings(
                    optimizer_iterations=1,
                    ei_scale_bounds=(0.5, unstable_upper_scale),
                ),
                device="cpu",
            )

    def test_short_informative_record_recovers_toward_hidden_parameters(self) -> None:
        source = self._snapshot()
        nominal = fixed_solver_model(
            source,
            node_count=6,
            substeps=2,
            constraint_iterations=4,
        )
        hidden_ei_scale = 1.35
        hidden_cb_scale = 0.70
        hidden = fixed_solver_model(
            source,
            node_count=6,
            substeps=2,
            constraint_iterations=4,
            bending_stiffness_scale=hidden_ei_scale,
            bending_damping_scale=hidden_cb_scale,
        )
        measurements = make_synthetic_tip_measurements(
            hidden,
            SyntheticExcitationSettings(
                duration_s=0.60,
                dt_s=0.02,
                maximum_acceleration_m_s2=5.0,
            ),
            device="cpu",
        )

        result = estimate_tip_only_parameters(
            nominal,
            measurements,
            TipAdaptationSettings(
                optimizer_iterations=6,
                # The synthetic trace is noiseless.  A tight, explicit noise
                # model makes this a parameter-recovery test rather than a
                # test of prior shrinkage under sub-noise model differences.
                measurement_noise_std_m=1.0e-6,
                prior_log_scale_std=3.0,
                minimum_relative_information_eigenvalue=1.0e-8,
                maximum_information_condition=1.0e12,
                sensitivity_log_step=0.005,
            ),
            device="cpu",
        )

        self.assertTrue(result.update_applied)
        self.assertLess(result.final_tip_rmse_m, result.initial_tip_rmse_m)
        self.assertLess(
            abs(np.log(result.estimated_ei_scale / hidden_ei_scale)),
            abs(np.log(1.0 / hidden_ei_scale)),
        )
        self.assertLess(
            abs(np.log(result.estimated_cb_scale / hidden_cb_scale)),
            abs(np.log(1.0 / hidden_cb_scale)),
        )
        self.assertTrue(np.all(np.isfinite(result.predicted_tip_positions_m)))
        self.assertTrue(np.all(np.isfinite(result.relative_information_eigenvalues)))

    @classmethod
    def _small_model(cls) -> CableModelSnapshot:
        return fixed_solver_model(
            cls._snapshot(),
            node_count=6,
            substeps=2,
            constraint_iterations=3,
        )

    @classmethod
    def _snapshot(cls) -> CableModelSnapshot:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(cls._artifact()), encoding="utf-8")
            return load_cable_model(path)

    @staticmethod
    def _artifact() -> dict[str, object]:
        rest_lengths = [0.05] * 10
        material_coordinates = np.r_[0.0, np.cumsum(rest_lengths)].tolist()
        vertex_masses = [0.002] * 11
        return {
            "schema": MODEL_SCHEMA,
            "measured": {
                "marker_count": 11,
                "node_count": 11,
                "rod_segments_per_marker_interval": 1,
                "marker_node_indices": list(range(11)),
                "marker_interval_lengths_m": rest_lengths,
                "rest_lengths_m": rest_lengths,
                "marker_material_coordinates_m": material_coordinates,
                "rod_material_coordinates_m": material_coordinates,
                "length_m": 0.5,
                "bare_cable_mass_kg": 0.012,
                "moving_marker_count": 10,
                "moving_marker_masses_kg": [0.001] * 10,
                "vertex_masses_kg": vertex_masses,
                "total_dynamic_mass_kg": sum(vertex_masses),
                "diameter_m": 0.0035,
            },
            "optimized": {
                "bending_stiffness_n_m2": 1.0e-6,
                "bending_damping_n_m2_s": 1.0e-6,
            },
            "fit": {"status": "completed"},
            "solver": {
                "gravity_m_s2": [0.0, 0.0, -9.80665],
                "substeps": 2,
                "constraint_iterations": 4,
            },
            "boundary_condition": {
                "prescribed_vertices": [0],
                "attachment": "prescribed position",
                "distal_terminal": "dynamic and free",
            },
        }


if __name__ == "__main__":
    unittest.main()
