"""Unit checks for the identified-cable to Isaac articulation mapping.

These tests deliberately avoid importing Isaac Sim.  They pin the physical
artifact used by the first plant and check the continuum-to-joint conversion
before the more expensive PhysX validation is launched.
"""

from __future__ import annotations

import math
import unittest

from isaac_whip.config import (
    DEFAULT_MODEL_PATH,
    build_cable_spec,
    load_drone_config,
)


CANONICAL_MODEL_SHA256 = (
    "0eece55cf9c2b07fcd699c1e7d9f5079d01240aabda2a7237a54b8c3c4fbc827"
)


class IsaacWhipConfigurationTests(unittest.TestCase):
    def test_regridding_conserves_length_and_mass_exactly(self) -> None:
        """A resolution change must not change the physical cable."""

        for link_count in (10, 20, 30):
            with self.subTest(link_count=link_count):
                snapshot, spec = build_cable_spec(link_count=link_count)

                self.assertEqual(spec.link_count, link_count)
                self.assertEqual(spec.node_count, link_count + 1)
                self.assertEqual(len(spec.rest_lengths_m), link_count)
                self.assertEqual(len(spec.vertex_masses_kg), link_count + 1)
                self.assertEqual(len(spec.link_masses_kg), link_count)
                self.assertTrue(all(value > 0.0 for value in spec.rest_lengths_m))
                self.assertTrue(all(value > 0.0 for value in spec.link_masses_kg))
                self.assertAlmostEqual(
                    sum(spec.rest_lengths_m), snapshot.cable_length_m, delta=1.0e-12
                )
                self.assertAlmostEqual(
                    spec.total_mass_kg,
                    sum(snapshot.model.parameters.vertex_masses_kg),
                    delta=1.0e-12,
                )
                self.assertAlmostEqual(spec.material_coordinates_m[0], 0.0, delta=1.0e-15)
                self.assertAlmostEqual(
                    spec.material_coordinates_m[-1], spec.length_m, delta=1.0e-12
                )

    def test_canonical_artifact_identity_and_transferred_values_are_pinned(self) -> None:
        """A silent refit or path change must be visible in test results."""

        snapshot, spec = build_cable_spec()

        self.assertEqual(spec.source_path, DEFAULT_MODEL_PATH.resolve())
        self.assertEqual(spec.source_sha256, CANONICAL_MODEL_SHA256)
        self.assertEqual(snapshot.sha256, CANONICAL_MODEL_SHA256)
        self.assertEqual(spec.source_schema, "optitrack_twist_aware_rod_v5")
        self.assertTrue(snapshot.provisional)
        self.assertTrue(spec.provisional)
        self.assertIn("two-holder", spec.provenance_note)
        self.assertAlmostEqual(spec.length_m, 0.9610000000000001, delta=1.0e-15)
        self.assertAlmostEqual(spec.diameter_m, 0.0035, delta=1.0e-15)
        self.assertAlmostEqual(spec.total_mass_kg, 0.01609091, delta=1.0e-12)
        self.assertAlmostEqual(
            spec.bending_stiffness_n_m2, 0.00010154787051679222, delta=1.0e-18
        )
        self.assertAlmostEqual(
            spec.bending_damping_n_m2_s,
            1.4999999999999987e-05,
            delta=1.0e-18,
        )

    def test_ei_and_cb_map_to_dual_length_hinge_gains(self) -> None:
        """Discrete bending energy uses k=EI/h and c=Cb/h at each hinge.

        PhysX angular drives are parameterized per degree, so an additional
        pi/180 factor is required relative to the SI per-radian gains.
        """

        _, spec = build_cable_spec()
        self.assertEqual(len(spec.hinge_dual_lengths_m), spec.link_count - 1)
        radians_per_degree = math.pi / 180.0

        for index, dual_length in enumerate(spec.hinge_dual_lengths_m):
            with self.subTest(hinge=index):
                stiffness_rad = spec.bending_stiffness_n_m2 / dual_length
                damping_rad = spec.bending_damping_n_m2_s / dual_length
                self.assertAlmostEqual(
                    spec.hinge_stiffness_n_m_rad[index], stiffness_rad, delta=1.0e-15
                )
                self.assertAlmostEqual(
                    spec.hinge_damping_n_m_s_rad[index], damping_rad, delta=1.0e-15
                )
                self.assertAlmostEqual(
                    spec.usd_hinge_stiffness_n_m_deg[index],
                    stiffness_rad * radians_per_degree,
                    delta=1.0e-15,
                )
                self.assertAlmostEqual(
                    spec.usd_hinge_damping_n_m_s_deg[index],
                    damping_rad * radians_per_degree,
                    delta=1.0e-15,
                )

    def test_manifest_records_the_intended_boundary_conditions(self) -> None:
        """The flight plant is clamped at the drone and mechanically free at its tip."""

        snapshot, spec = build_cable_spec()
        manifest = spec.to_manifest()

        self.assertEqual(snapshot.model.parameters.torsional_stiffness_n_m2, 0.0)
        self.assertEqual(
            manifest["internal_twist_mode"],
            "free; no stiffness, damping, friction, or limit",
        )
        self.assertEqual(
            manifest["root_boundary"], "fixed material frame at drone attachment"
        )
        self.assertEqual(manifest["distal_boundary"], "mechanically free")
        self.assertNotIn("torsional_stiffness_n_m2", manifest)

    def test_provisional_drone_has_explicit_hover_thrust_margin(self) -> None:
        """The assumed actuator must support the complete suspended system."""

        drone = load_drone_config()
        _, cable = build_cable_spec()
        total_mass = drone.mass_kg + cable.total_mass_kg
        system_weight_n = total_mass * 9.80665
        thrust_to_weight = drone.maximum_collective_thrust_n / system_weight_n

        self.assertTrue(drone.provisional)
        self.assertIn("Replace", drone.note)
        self.assertGreater(drone.maximum_total_mass_kg, total_mass)
        self.assertGreater(drone.maximum_collective_thrust_n, system_weight_n)
        self.assertGreater(thrust_to_weight, 2.0)
        self.assertAlmostEqual(thrust_to_weight, 2.247871396358349, places=12)


if __name__ == "__main__":
    unittest.main()
