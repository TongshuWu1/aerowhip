from __future__ import annotations

import unittest

import torch

from drone_mpc.mpc import MpcProblem
from drone_mpc.oracle import (
    ReachabilitySettings,
    _basis_controls,
    _bounded_acceleration,
    _candidate_order,
    _vertical_target_plane,
)


class DroneOracleTests(unittest.TestCase):
    def test_settings_define_independent_control_knots(self) -> None:
        settings = ReachabilitySettings(
            horizon_s=2.0,
            control_interval_s=0.1,
            candidates=32,
            elite_count=4,
            iterations=3,
        )
        self.assertEqual(settings.control_count, 20)

    def test_acceleration_projection_preserves_direction_and_norm_bound(self) -> None:
        controls = torch.tensor(
            [[[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float64
        )
        projected = _bounded_acceleration(controls, 2.0)
        torch.testing.assert_close(
            projected[0, 0], torch.tensor([1.2, 1.6, 0.0], dtype=torch.float64)
        )
        torch.testing.assert_close(projected[0, 1], controls[0, 1])

    def test_basis_is_bounded_and_has_no_lateral_leakage(self) -> None:
        settings = ReachabilitySettings(
            horizon_s=2.0,
            control_interval_s=0.1,
            basis_count=3,
            candidates=32,
            elite_count=4,
            iterations=3,
            maximum_acceleration_m_s2=1.0,
        )
        coefficients = torch.zeros((1, 3, 3), dtype=torch.float64)
        coefficients[0, 1, 0] = 0.2
        controls = _basis_controls(coefficients, settings)
        self.assertEqual(controls.shape, (1, 20, 3))
        self.assertLessEqual(
            float(torch.max(torch.linalg.vector_norm(controls, dim=2))), 1.0
        )
        torch.testing.assert_close(
            controls[0, :, 1:], torch.zeros((20, 2), dtype=torch.float64)
        )

    def test_candidate_order_is_constraint_then_energy_lexicographic(self) -> None:
        terms = {
            "safety_violation": torch.tensor([0.0, 0.0, 0.1]),
            "hit_violation": torch.tensor([0.0, 0.2, 0.0]),
            "negative_directional_tip_energy_j": torch.tensor([-0.1, -1.0, -2.0]),
            "impact_time": torch.tensor([2.0, 1.0, 0.5]),
            "regularization": torch.tensor([1.0, 0.0, 0.0]),
        }
        self.assertEqual(_candidate_order(terms).tolist(), [0, 1, 2])

    def test_vertical_target_plane_removes_only_lateral_component(self) -> None:
        coefficients = torch.tensor(
            [[[1.0, 2.0, 3.0]]], dtype=torch.float64
        )
        projected = _vertical_target_plane(
            coefficients,
            (1.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
        )
        torch.testing.assert_close(
            projected, torch.tensor([[[1.0, 0.0, 3.0]]], dtype=torch.float64)
        )

    def test_problem_retains_physical_whip_constraints(self) -> None:
        problem = MpcProblem(
            target_position_m=(0.48, 0.0, 1.35),
            impact_direction=(1.0, 0.0, 0.0),
            minimum_forward_stroke_m=0.075,
            minimum_recoil_stroke_m=0.075,
        )
        self.assertEqual(problem.minimum_forward_stroke_m, 0.075)
        self.assertEqual(problem.minimum_recoil_stroke_m, 0.075)


if __name__ == "__main__":
    unittest.main()
