from __future__ import annotations

import math
import unittest

import torch

from cable_twin.online.config import ParticleFilterSettings
from cable_twin.online.filter import DderParticleFilter
from cable_twin.shared.dder import (
    continuous_endpoint_twist_angles,
    DderModel,
    DderParameters,
    DderState,
    curvature_binormals,
    curvature_binormal_rates,
    _interpolate_rotations,
    solve_symmetric_tridiagonal,
    solve_symmetric_tridiagonal_dense,
)


class StraightCableDderTests(unittest.TestCase):
    def test_endpoint_rotations_are_interpolated_on_so3(self) -> None:
        start = torch.eye(3, dtype=torch.float64).repeat(1, 2, 1, 1)
        finish = start.clone()
        angle = math.pi / 2.0
        finish[:, :, :2, :2] = torch.tensor(
            ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))),
            dtype=torch.float64,
        )
        midpoint, rate = _interpolate_rotations(
            start,
            finish,
            0.5,
            torch.tensor((0.02,), dtype=torch.float64),
        )
        identity = torch.eye(3, dtype=torch.float64).expand(1, 2, -1, -1)
        torch.testing.assert_close(
            midpoint.transpose(-1, -2) @ midpoint,
            identity,
            atol=1.0e-12,
            rtol=0.0,
        )
        self.assertTrue(bool(torch.all(torch.isfinite(rate))))
        self.assertAlmostEqual(float(midpoint[0, 0, 0, 0]), math.sqrt(0.5), 12)

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
        self.assertTrue(
            math.isinf(self.model.maximum_stable_bending_damping(1.0 / 30.0))
        )
        damping = torch.tensor(1.0e-5, dtype=torch.float64, requires_grad=True)
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

    def test_curvature_damping_ignores_rigid_rotation(self) -> None:
        parameter = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
        positions = torch.stack(
            (
                0.45 * parameter,
                0.04 * torch.sin(2.0 * math.pi * parameter),
                0.06 * torch.sin(math.pi * parameter),
            ),
            dim=-1,
        ).unsqueeze(0)
        angular_velocity = torch.tensor((0.7, -0.4, 0.2), dtype=torch.float64)
        velocity = torch.linalg.cross(
            angular_velocity[None, None].expand_as(positions),
            positions,
            dim=-1,
        )
        force = self.model.damping_force(positions, velocity, create_graph=False)
        torch.testing.assert_close(force, torch.zeros_like(force), atol=1.0e-11, rtol=0.0)
        self.assertLessEqual(abs(float(torch.sum(force * velocity))), 1.0e-12)

        tangent = positions[:, (1, -1)] - positions[:, (0, -2)]
        tangent = tangent / torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
        reference = torch.tensor((0.0, 1.0, 0.0), dtype=torch.float64)[None, None]
        material_y = reference - tangent * torch.sum(reference * tangent, dim=-1, keepdim=True)
        material_y = material_y / torch.linalg.vector_norm(
            material_y, dim=-1, keepdim=True
        )
        material_z = torch.linalg.cross(tangent, material_y, dim=-1)
        orientations = torch.stack((tangent, material_y, material_z), dim=-1)
        wx, wy, wz = angular_velocity
        omega = torch.stack(
            (
                torch.stack((wx * 0.0, -wz, wy)),
                torch.stack((wz, wx * 0.0, -wx)),
                torch.stack((-wy, wx, wx * 0.0)),
            )
        )
        orientation_rates = omega[None, None] @ orientations
        terminal_force = self.model.damping_force(
            positions,
            velocity,
            create_graph=False,
            endpoint_orientations=orientations,
            endpoint_orientation_rates=orientation_rates,
        )
        torch.testing.assert_close(
            terminal_force,
            torch.zeros_like(terminal_force),
            atol=1.0e-11,
            rtol=0.0,
        )

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

    def test_implicit_damping_remains_finite_above_old_explicit_limit(self) -> None:
        model = DderModel(
            DderParameters(
                node_count=8,
                cable_length_m=0.518,
                cable_mass_kg=0.03,
                cable_diameter_m=0.0035,
                bending_stiffness_n_m2=1.0e-12,
                bending_damping_n_m2_s=1.0e-3,
                gravity_camera_m_s2=(0.0, 0.0, 0.0),
                substeps=2,
                constraint_iterations=8,
            )
        )
        positions = torch.zeros((1, 8, 3), dtype=torch.float64)
        positions[0, :, 0] = torch.linspace(0.0, 0.518, 8)
        velocity = torch.zeros_like(positions)
        velocity[0, 1:-1, 2] = torch.tensor(
            (0.1, -0.1, 0.1, -0.1, 0.1, -0.1), dtype=torch.float64
        )
        velocity = model.project_velocities(
            positions, velocity, velocity[:, (0, -1)]
        )
        initial_energy = torch.sum(velocity.square())
        result = model.step(
            DderState(positions, velocity),
            positions[:, (0, -1)],
            1.0 / 30.0,
        )
        self.assertTrue(bool(torch.all(torch.isfinite(result.positions_m))))
        self.assertLess(
            float(torch.sum(result.velocities_m_s.square())),
            float(initial_energy),
        )

    def test_terminal_twist_is_unwrapped_across_multiple_turns(self) -> None:
        sample_angles = torch.linspace(0.0, 2.5 * math.pi, 26, dtype=torch.float64)
        positions = torch.zeros((len(sample_angles), 8, 3), dtype=torch.float64)
        positions[:, :, 0] = torch.linspace(0.0, 0.518, 8)
        orientations = torch.eye(3, dtype=torch.float64).repeat(
            len(sample_angles), 2, 1, 1
        )
        cosine, sine = torch.cos(sample_angles), torch.sin(sample_angles)
        orientations[:, 1, 1, 1] = cosine
        orientations[:, 1, 1, 2] = -sine
        orientations[:, 1, 2, 1] = sine
        orientations[:, 1, 2, 2] = cosine

        unwrapped = continuous_endpoint_twist_angles(positions, orientations)
        torch.testing.assert_close(unwrapped, sample_angles, rtol=1.0e-12, atol=1.0e-12)

        stiffness = 2.0e-5
        energy = self.model.twist_energy(
            positions[-1:],
            orientations[-1:],
            stiffness,
            endpoint_twist_reference_rad=unwrapped[-1:],
        )
        expected = 0.5 * stiffness * sample_angles[-1].square() / 0.518
        torch.testing.assert_close(energy[0], expected, rtol=1.0e-12, atol=1.0e-12)

    def test_dynamics_state_carries_twist_branch_through_pi(self) -> None:
        parameters = DderParameters(
            node_count=8,
            cable_length_m=0.518,
            cable_mass_kg=0.030,
            cable_diameter_m=0.004,
            bending_stiffness_n_m2=1.0e-5,
            bending_damping_n_m2_s=1.0e-8,
            torsional_stiffness_n_m2=1.0e-7,
            gravity_camera_m_s2=(0.0, 0.0, 0.0),
            substeps=4,
            constraint_iterations=8,
        )
        model = DderModel(parameters)
        positions = torch.zeros((1, 8, 3), dtype=torch.float64)
        positions[:, :, 0] = torch.linspace(0.0, 0.518, 8)
        start = torch.eye(3, dtype=torch.float64).repeat(1, 2, 1, 1)
        finish = start.clone()
        for orientations, angle in ((start, 0.9 * math.pi), (finish, 1.1 * math.pi)):
            orientations[0, 1, 1, 1] = math.cos(angle)
            orientations[0, 1, 1, 2] = -math.sin(angle)
            orientations[0, 1, 2, 1] = math.sin(angle)
            orientations[0, 1, 2, 2] = math.cos(angle)
        state = DderState(
            positions,
            torch.zeros_like(positions),
            start,
            torch.tensor((0.9 * math.pi,), dtype=torch.float64),
        )
        result = model.step(
            state,
            positions[:, (0, -1)],
            0.01,
            boundary_orientations_next=finish,
        )
        self.assertIsNotNone(result.endpoint_twist_rad)
        assert result.endpoint_twist_rad is not None
        self.assertGreater(float(result.endpoint_twist_rad[0]), math.pi)

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
        parallel = solve_symmetric_tridiagonal_dense(diagonal, off, rhs)
        torch.testing.assert_close(parallel, expected, rtol=1.0e-11, atol=1.0e-12)

    def test_parallel_constraint_solver_matches_structured_gradients(self) -> None:
        generator = torch.Generator().manual_seed(21)
        base_off = 0.08 * torch.rand((4, 6), generator=generator, dtype=torch.float64)
        base_diagonal = (
            1.0
            + torch.nn.functional.pad(base_off, (1, 0))
            + torch.nn.functional.pad(base_off, (0, 1))
        )
        base_rhs = torch.randn((4, 7), generator=generator, dtype=torch.float64)

        gradients = []
        outputs = []
        for solver in (
            solve_symmetric_tridiagonal,
            solve_symmetric_tridiagonal_dense,
        ):
            diagonal = base_diagonal.detach().clone().requires_grad_(True)
            off = base_off.detach().clone().requires_grad_(True)
            rhs = base_rhs.detach().clone().requires_grad_(True)
            output = solver(diagonal, off, rhs)
            output.square().sum().backward()
            outputs.append(output.detach())
            gradients.append((diagonal.grad, off.grad, rhs.grad))

        torch.testing.assert_close(outputs[0], outputs[1], rtol=1.0e-11, atol=1.0e-12)
        for structured, parallel in zip(gradients[0], gradients[1]):
            torch.testing.assert_close(
                structured,
                parallel,
                rtol=2.0e-10,
                atol=2.0e-11,
            )

    def test_nonuniform_marker_mass_model_uses_exact_physical_discretization(self) -> None:
        rest_lengths = (0.078, 0.085, 0.100, 0.100, 0.098, 0.100, 0.100, 0.100, 0.100, 0.100)
        masses = tuple(0.017 / 11.0 for _ in range(11))
        parameters = DderParameters(
            node_count=11,
            cable_length_m=sum(rest_lengths),
            cable_mass_kg=sum(masses),
            cable_diameter_m=0.0035,
            bending_stiffness_n_m2=1.0e-5,
            bending_damping_n_m2_s=1.0e-8,
            gravity_camera_m_s2=(0.0, 0.0, -9.80665),
            rest_lengths_m=rest_lengths,
            vertex_masses_kg=masses,
            substeps=4,
            constraint_iterations=12,
        )
        model = DderModel(parameters)
        positions = torch.zeros((1, 11, 3), dtype=torch.float64)
        positions[0, 1:, 0] = torch.cumsum(
            torch.tensor(rest_lengths, dtype=torch.float64), dim=0
        )
        constants = model.runtime_constants(positions)
        torch.testing.assert_close(
            constants.rest_lengths_m,
            torch.tensor(rest_lengths, dtype=torch.float64),
        )
        torch.testing.assert_close(
            constants.masses_kg,
            torch.tensor(masses, dtype=torch.float64),
        )
        self.assertLess(float(model.maximum_segment_error_m(positions)), 1.0e-12)

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

    def test_capture_safe_runtime_damping_matches_direct_smooth_step(self) -> None:
        positions = torch.zeros((3, 8, 3), dtype=torch.float32)
        parameter = torch.linspace(0.0, 1.0, 8)
        positions[:, :, 0] = 0.45 * parameter
        positions[:, :, 2] = 0.03 * torch.sin(math.pi * parameter)
        positions = self.model.project_lengths(
            positions, positions[:, (0, -1)]
        )
        velocity = torch.zeros_like(positions)
        boundaries = positions[:, (0, -1)]
        dt = torch.full((3,), 1.0 / 30.0)
        constants = self.model.runtime_constants(positions)
        direct = self.model.step_runtime(
            DderState(positions, velocity), boundaries, dt, constants
        )
        iterative = self.model.step_runtime(
            DderState(positions, velocity),
            boundaries,
            dt,
            constants,
            iterative_damping=True,
        )
        torch.testing.assert_close(
            iterative.positions_m,
            direct.positions_m,
            rtol=0.0,
            atol=1.0e-6,
        )

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
            endpoint_acceleration_sigma_m_s2=0.2,
            process_acceleration_sigma_m_s2=0.4,
        )
        tracker = DderParticleFilter(self.model, settings)
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
            endpoint_acceleration_sigma_m_s2=0.2,
            process_acceleration_sigma_m_s2=0.4,
        )
        tracker = DderParticleFilter(self.model, settings)
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

    def test_projected_body_likelihood_favors_image_aligned_particle(self) -> None:
        settings = ParticleFilterSettings(
            device="cpu", particle_count=16, random_seed=0,
            maximum_dt_s=0.034, maximum_gap_s=0.2,
            ess_resample_fraction=0.5,
            body_pixel_sigma_px=4.0,
        )
        tracker = DderParticleFilter(self.model, settings)
        positions = torch.zeros((16, 8, 3), dtype=torch.float32)
        positions[:, :, 0] = torch.linspace(0.0, 0.518, 8)
        positions[:, :, 1] = torch.linspace(0.0, 0.15, 16)[:, None]
        positions[:, :, 2] = 1.0
        tracker.positions = positions
        tracker.velocities = torch.zeros_like(positions)
        tracker.log_weights = torch.full((16,), -math.log(16.0))
        observed = torch.stack(
            (
                torch.linspace(320.0, 579.0, 32),
                torch.full((32,), 240.0),
            ),
            dim=1,
        )
        body_updated, endpoint_count, residual_px = tracker._measurement_update(
            observed,
            torch.ones(32, dtype=torch.bool),
            torch.full((2, 2), float("nan")),
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
            torch.tensor((500.0, 500.0, 320.0, 240.0)),
            torch.zeros(16),
        )
        weights = torch.softmax(tracker.log_weights, dim=0)
        self.assertTrue(body_updated)
        self.assertEqual(endpoint_count, 0)
        self.assertTrue(math.isfinite(residual_px))
        self.assertGreater(float(weights[0]), float(weights[-1]))


if __name__ == "__main__":
    unittest.main()
