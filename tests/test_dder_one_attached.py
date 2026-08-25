from __future__ import annotations

import unittest

import torch

from cable_twin.shared.dder import (
    DderModel,
    DderParameters,
    START_PINNED_FREE_END,
)


class OneAttachedDderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = DderModel(
            DderParameters(
                node_count=8,
                cable_length_m=0.518,
                cable_mass_kg=0.030,
                cable_diameter_m=0.004,
                bending_stiffness_n_m2=1.0e-5,
                bending_damping_n_m2_s=1.0e-8,
                gravity_camera_m_s2=(0.0, 9.80665, 0.0),
                substeps=4,
                constraint_iterations=8,
            )
        )

    def test_one_attached_step_keeps_only_start_fixed(self) -> None:
        positions = torch.zeros((1, 8, 3), dtype=torch.float64)
        positions[:, :, 0] = torch.linspace(0.0, 0.518, 8)
        state = self.model.initial_state(positions)
        dt = 1.0 / 60.0
        stable_ei = self.model.maximum_stable_bending_stiffness(
            dt,
            pinned_endpoints=START_PINNED_FREE_END,
        )
        stiffness = torch.tensor(
            0.25 * stable_ei,
            dtype=torch.float64,
            requires_grad=True,
        )
        damping = torch.tensor(1.0e-5, dtype=torch.float64, requires_grad=True)

        result = self.model.step(
            state,
            positions[:, :1],
            dt,
            create_graph=True,
            bending_stiffness_n_m2=stiffness,
            bending_damping_n_m2_s=damping,
            pinned_endpoints=START_PINNED_FREE_END,
        )

        torch.testing.assert_close(
            result.positions_m[:, :1],
            positions[:, :1],
            atol=0.0,
            rtol=0.0,
        )
        self.assertGreater(
            float(
                torch.linalg.vector_norm(
                    result.positions_m[:, -1] - positions[:, -1]
                ).detach()
            ),
            1.0e-6,
        )
        self.assertGreater(
            float(
                torch.linalg.vector_norm(result.velocities_m_s[:, -1]).detach()
            ),
            1.0e-6,
        )
        self.assertLess(
            float(self.model.maximum_segment_error_m(result.positions_m).detach()),
            1.0e-4,
        )

        result.positions_m[:, 1:].square().sum().backward()
        self.assertTrue(bool(torch.isfinite(stiffness.grad)))
        self.assertTrue(bool(torch.isfinite(damping.grad)))

    def test_one_attached_velocity_projection_does_not_clamp_free_tip(self) -> None:
        positions = torch.zeros((1, 8, 3), dtype=torch.float64)
        positions[:, :, 0] = torch.linspace(0.0, 0.518, 8)
        velocities = torch.zeros_like(positions)
        velocities[:, -1, 2] = 0.2

        projected = self.model.project_velocities(
            positions,
            velocities,
            torch.zeros((1, 1, 3), dtype=torch.float64),
            pinned_endpoints=START_PINNED_FREE_END,
        )

        torch.testing.assert_close(
            projected[:, :1],
            torch.zeros_like(projected[:, :1]),
            atol=0.0,
            rtol=0.0,
        )
        self.assertAlmostEqual(float(projected[0, -1, 2]), 0.2, 12)
        tangents = positions[:, 1:] - positions[:, :-1]
        tangents = tangents / torch.linalg.vector_norm(
            tangents,
            dim=-1,
            keepdim=True,
        )
        constraint_speed = torch.sum(
            tangents * (projected[:, 1:] - projected[:, :-1]),
            dim=-1,
        )
        self.assertLess(float(torch.amax(torch.abs(constraint_speed))), 1.0e-10)

    def test_one_attached_position_only_path_rejects_torsion(self) -> None:
        positions = torch.zeros((1, 8, 3), dtype=torch.float64)
        positions[:, :, 0] = torch.linspace(0.0, 0.518, 8)
        with self.assertRaisesRegex(ValueError, "zero torsional stiffness"):
            self.model.step(
                self.model.initial_state(positions),
                positions[:, :1],
                1.0 / 60.0,
                torsional_stiffness_n_m2=1.0e-7,
                pinned_endpoints=START_PINNED_FREE_END,
            )


if __name__ == "__main__":
    unittest.main()
