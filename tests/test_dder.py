from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

import torch

from cable_twin.online.config import ParticleFilterSettings
from cable_twin.online.filter import DderParticleFilter
from cable_twin.shared.dder import (
    DderModel,
    DderParameters,
    DderState,
    curvature_binormals,
    curvature_binormal_rates,
    solve_symmetric_tridiagonal,
)


class StraightCableDderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = DderParameters(
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
        self.model = DderModel(self.parameters)

    def test_exact_bending_energy_is_rigid_motion_invariant(self) -> None:
        parameter = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
        curve = torch.stack(
            (
                0.45 * parameter,
                0.05 * torch.sin(math.pi * parameter),
                0.03 * torch.sin(2.0 * math.pi * parameter),
            ),
            dim=-1,
        ).unsqueeze(0)
        angle = 0.7
        rotation = torch.tensor(
            (
                (math.cos(angle), -math.sin(angle), 0.0),
                (math.sin(angle), math.cos(angle), 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=torch.float64,
        )
        moved = curve @ rotation.T + torch.tensor((0.3, -0.2, 0.4))
        torch.testing.assert_close(
            self.model.bending_energy(curve),
            self.model.bending_energy(moved),
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        self.assertEqual(curvature_binormals(curve).shape, (1, 6, 3))

    def test_step_is_differentiable_and_preserves_length(self) -> None:
        positions = torch.zeros((1, 8, 3), dtype=torch.float64)
        positions[:, :, 0] = torch.linspace(0.0, 0.518, 8)
        state = self.model.initial_state(positions)
        stable_ei = self.model.maximum_stable_bending_stiffness(1.0 / 30.0)
        stiffness = torch.tensor(0.5 * stable_ei, dtype=torch.float64, requires_grad=True)
        stable_cb = self.model.maximum_stable_bending_damping(1.0 / 30.0)
        damping = torch.tensor(0.5 * stable_cb, dtype=torch.float64, requires_grad=True)
        result = self.model.step(
            state,
            positions[:, (0, -1)],
            1.0 / 30.0,
            create_graph=True,
            bending_stiffness_n_m2=stiffness,
            bending_damping_n_m2_s=damping,
        )
        self.assertLess(
            float(self.model.maximum_segment_error_m(result.positions_m).detach()),
            1.0e-4,
        )
        result.positions_m.square().sum().backward()
        self.assertTrue(bool(torch.isfinite(stiffness.grad)))
        self.assertTrue(bool(torch.isfinite(damping.grad)))

    def test_curvature_damping_ignores_rigid_translation(self) -> None:
        parameter = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
        positions = torch.stack(
            (
                0.45 * parameter,
                torch.zeros_like(parameter),
                0.06 * torch.sin(math.pi * parameter),
            ),
            dim=-1,
        ).unsqueeze(0)
        velocity = torch.tensor((0.3, -0.2, 0.1), dtype=torch.float64)[None, None]
        velocity = velocity.expand_as(positions).clone()
        rate = curvature_binormal_rates(positions, velocity)
        torch.testing.assert_close(rate, torch.zeros_like(rate), atol=1.0e-12, rtol=0.0)
        force = self.model.damping_force(
            positions,
            velocity,
            create_graph=False,
        )
        torch.testing.assert_close(force, torch.zeros_like(force), atol=1.0e-12, rtol=0.0)

    def test_curvature_damping_never_adds_kinetic_energy(self) -> None:
        parameter = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
        positions = torch.stack(
            (
                0.45 * parameter,
                torch.zeros_like(parameter),
                0.05 * torch.sin(math.pi * parameter),
            ),
            dim=-1,
        ).unsqueeze(0)
        velocity = torch.zeros_like(positions)
        velocity[0, :, 2] = torch.sin(2.0 * math.pi * parameter)
        force = self.model.damping_force(positions, velocity, create_graph=False)
        self.assertLessEqual(float(torch.sum(force * velocity)), 1.0e-12)

    def test_velocity_projection_satisfies_inextensibility(self) -> None:
        positions = torch.zeros((2, 8, 3), dtype=torch.float32)
        positions[:, :, 0] = torch.linspace(0.0, 0.518, 8)
        velocity = torch.randn(positions.shape, generator=torch.Generator().manual_seed(4))
        projected = self.model.project_velocities(
            positions, velocity, torch.zeros((2, 2, 3), dtype=torch.float32)
        )
        tangent = positions[:, 1:] - positions[:, :-1]
        tangent /= torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
        constraint_speed = torch.sum(
            tangent * (projected[:, 1:] - projected[:, :-1]), dim=-1
        )
        self.assertLess(float(torch.max(torch.abs(constraint_speed))), 6.0e-6)

    def test_structured_constraint_solver_matches_dense_solve(self) -> None:
        generator = torch.Generator().manual_seed(12)
        off = 0.1 * torch.rand((5, 7), generator=generator, dtype=torch.float64)
        diagonal = 1.0 + torch.nn.functional.pad(off, (1, 0)) + torch.nn.functional.pad(off, (0, 1))
        rhs = torch.randn((5, 8), generator=generator, dtype=torch.float64)
        system = torch.diag_embed(diagonal)
        system += torch.diag_embed(off, offset=1)
        system += torch.diag_embed(off, offset=-1)
        expected = torch.linalg.solve(system, rhs.unsqueeze(-1)).squeeze(-1)
        actual = solve_symmetric_tridiagonal(diagonal, off, rhs)
        torch.testing.assert_close(actual, expected, rtol=1.0e-11, atol=1.0e-12)

    def test_runtime_step_is_numerically_identical(self) -> None:
        positions = torch.zeros((3, 8, 3), dtype=torch.float32)
        positions[:, :, 0] = torch.linspace(0.0, 0.518, 8)
        velocity = torch.zeros_like(positions)
        boundaries = positions[:, (0, -1)]
        reference = self.model.step(
            DderState(positions, velocity), boundaries, 1.0 / 30.0
        )
        runtime = self.model.step_runtime(
            DderState(positions, velocity),
            boundaries,
            torch.full((3,), 1.0 / 30.0),
            self.model.runtime_constants(positions),
        )
        torch.testing.assert_close(runtime.positions_m, reference.positions_m)
        torch.testing.assert_close(runtime.velocities_m_s, reference.velocities_m_s)

    def test_free_length_projection_does_not_fix_noisy_stereo_endpoints(self) -> None:
        parameter = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
        positions = torch.stack(
            (
                0.35 * parameter,
                0.04 * torch.sin(math.pi * parameter),
                1.2 + 0.03 * torch.sin(2.0 * math.pi * parameter),
            ),
            dim=-1,
        ).unsqueeze(0).requires_grad_(True)
        projected = self.model.project_free_lengths(positions)
        self.assertLess(
            float(self.model.maximum_segment_error_m(projected).detach()),
            1.0e-7,
        )
        projected.square().sum().backward()
        self.assertTrue(bool(torch.all(torch.isfinite(positions.grad))))

    def test_unobserved_pf_endpoints_follow_zero_velocity_process_prior(self) -> None:
        settings = ParticleFilterSettings(
            device="cpu", particle_count=16, random_seed=0,
            maximum_dt_s=0.034, maximum_gap_s=0.2,
            ess_resample_fraction=0.5,
        )
        artifact = SimpleNamespace(
            endpoint_acceleration_sigma_m_s2=0.2,
            process_acceleration_sigma_m_s2=0.4,
            student_t_degrees_of_freedom=4.0,
        )
        tracker = DderParticleFilter(self.model, settings, artifact)
        positions = torch.zeros((16, 8, 3), dtype=torch.float32)
        positions[:, :, 0] = torch.linspace(0.0, 0.518, 8)
        tracker.positions = positions
        tracker.velocities = torch.zeros_like(positions)
        tracker.log_weights = torch.full((16,), -math.log(16.0))
        boundary, log_evidence = tracker._sample_endpoint_boundary(
            torch.zeros((2, 3)),
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2),
            0.03,
        )
        torch.testing.assert_close(boundary, positions[:, (0, -1)], atol=0.001, rtol=0.0)
        torch.testing.assert_close(log_evidence, torch.zeros(16))

    def test_endpoint_predictive_evidence_favors_consistent_particles(self) -> None:
        settings = ParticleFilterSettings(
            device="cpu", particle_count=16, random_seed=0,
            maximum_dt_s=0.034, maximum_gap_s=0.2,
            ess_resample_fraction=0.5,
        )
        artifact = SimpleNamespace(
            endpoint_acceleration_sigma_m_s2=0.2,
            process_acceleration_sigma_m_s2=0.4,
            student_t_degrees_of_freedom=4.0,
        )
        tracker = DderParticleFilter(self.model, settings, artifact)
        positions = torch.zeros((16, 8, 3), dtype=torch.float32)
        positions[:, :, 0] = torch.linspace(0.0, 0.518, 8)
        positions[:, :, 1] = torch.linspace(0.0, 0.15, 16)[:, None]
        tracker.positions = positions
        tracker.velocities = torch.zeros_like(positions)
        tracker.log_weights = torch.full((16,), -math.log(16.0))
        _boundary, log_evidence = tracker._sample_endpoint_boundary(
            torch.tensor(((0.0, 0.0, 0.0), (float("nan"),) * 3)),
            torch.tensor((True, False)),
            torch.tensor((0.005, float("nan"))),
            0.03,
        )
        self.assertTrue(torch.all(torch.isfinite(log_evidence)))
        self.assertGreater(float(log_evidence[0]), float(log_evidence[-1]))


if __name__ == "__main__":
    unittest.main()
