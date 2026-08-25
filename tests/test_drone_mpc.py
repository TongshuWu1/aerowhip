from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from cable_twin.shared.dder import DderState
from drone_mpc.model import load_cable_model
from drone_mpc.mpc import (
    casting_controls,
    CastingPhaseSchedule,
    CostWeights,
    MpcProblem,
    OptimizerSettings,
    optimize_controls,
    run_receding_horizon_mpc,
    variable_impact_rollout_cost_terms,
)
from drone_mpc.reduced import (
    reduce_cable_model,
    stable_controller_model,
    transfer_dder_state,
)
from drone_mpc.realtime import RealtimeSettings
from drone_mpc.simulator import (
    DroneCableState,
    SimulationSettings,
    TensorRollout,
    WhipSimulator,
)
from optitrack_offline.fitting import MODEL_SCHEMA


class DroneMpcTests(unittest.TestCase):
    def test_casting_primitive_is_a_bounded_two_sweep_control_family(self) -> None:
        base = self._simulator()
        simulator = WhipSimulator(
            base.snapshot,
            SimulationSettings(
                horizon_s=1.0,
                simulation_dt_s=0.02,
                control_interval_s=0.10,
                attachment_drop_m=0.10,
                maximum_acceleration_m_s2=6.0,
                maximum_speed_m_s=3.0,
            ),
            device="cpu",
        )
        state = simulator.initial_state((0.0, 0.0, 1.0))
        problem = MpcProblem(
            target_position_m=(1.0, 0.0, 0.5),
            impact_direction=(1.0, 0.0, 0.0),
            drone_keepout_radius_m=0.01,
            maximum_drone_excursion_m=0.15,
        )
        normalized = torch.tensor(
            [[0.5, 0.5, 1.0, 1.0, 0.5, 1.0]], dtype=torch.float64
        )

        controls = casting_controls(
            normalized,
            state,
            problem,
            simulator,
            simulator.settings.control_count,
        )

        self.assertEqual(controls.shape, (1, 10, 3))
        self.assertLess(float(torch.min(controls[0, :, 0])), 0.0)
        self.assertGreater(float(torch.max(controls[0, :, 0])), 0.0)
        torch.testing.assert_close(
            controls[0, :, 1], torch.zeros(10, dtype=torch.float64)
        )
        self.assertLessEqual(
            float(torch.max(torch.linalg.vector_norm(controls, dim=2))),
            simulator.settings.maximum_acceleration_m_s2 * (1.0 + 1.0e-12),
        )

    def test_online_phase_schedule_locks_plane_and_ignores_replan_timing_reset(self) -> None:
        base = self._simulator()
        simulator = WhipSimulator(
            base.snapshot,
            SimulationSettings(
                horizon_s=1.0,
                simulation_dt_s=0.02,
                control_interval_s=0.10,
                attachment_drop_m=0.10,
                maximum_acceleration_m_s2=6.0,
                maximum_speed_m_s=3.0,
            ),
            device="cpu",
        )
        state = simulator.initial_state((0.0, 0.0, 1.0))
        problem = MpcProblem(
            target_position_m=(1.0, 0.0, 0.5),
            impact_direction=(1.0, 0.0, 0.0),
            drone_keepout_radius_m=0.01,
            maximum_drone_excursion_m=0.15,
        )
        # The candidates differ only in recoil-plane deflection and their old
        # relative timing variables. A persistent online schedule must make
        # those differences irrelevant.
        normalized = torch.tensor(
            (
                (0.5, 0.0, 1.0, 1.0, 0.0, 0.2),
                (0.5, 1.0, 1.0, 1.0, 1.0, 1.0),
            ),
            dtype=torch.float64,
        )
        schedule = CastingPhaseSchedule(
            elapsed_s=0.2,
            injection_end_s=0.5,
            motion_end_s=1.0,
        )

        controls = casting_controls(
            normalized,
            state,
            problem,
            simulator,
            simulator.settings.control_count,
            schedule,
        )

        torch.testing.assert_close(controls[0], controls[1])
        torch.testing.assert_close(
            controls[:, :, 1], torch.zeros((2, 10), dtype=torch.float64)
        )

    def test_completed_online_phase_schedule_commands_zero_acceleration(self) -> None:
        simulator = self._simulator()
        state = simulator.initial_state((0.0, 0.0, 1.0))
        normalized = torch.full((1, 6), 0.5, dtype=torch.float64)
        schedule = CastingPhaseSchedule(
            elapsed_s=1.0,
            injection_end_s=0.4,
            motion_end_s=0.8,
        )

        controls = casting_controls(
            normalized,
            state,
            self._problem(),
            simulator,
            simulator.settings.control_count,
            schedule,
        )

        torch.testing.assert_close(controls, torch.zeros_like(controls))

    def test_loader_is_strictly_one_attachment_and_gj_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            payload = self._artifact()
            path.write_text(json.dumps(payload), encoding="utf-8")
            snapshot = load_cable_model(path)

            self.assertEqual(snapshot.node_count, 11)
            self.assertEqual(snapshot.model.parameters.torsional_stiffness_n_m2, 0.0)
            self.assertFalse(snapshot.provisional)

            payload["schema"] = "unsupported_old_model"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "one-attachment|supported"):
                load_cable_model(path)

    def test_latest_two_holder_fit_is_an_explicit_provisional_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old_model.json"
            payload = self._artifact()
            payload["schema"] = "optitrack_twist_aware_rod_v5"
            measured = payload["measured"]
            optimized = payload["optimized"]
            assert isinstance(measured, dict)
            assert isinstance(optimized, dict)
            measured.pop("moving_marker_masses_kg")
            measured["interior_marker_count"] = 9
            measured["interior_marker_masses_kg"] = [0.001] * 9
            optimized["torsional_stiffness_n_m2"] = 3.0e-5
            path.write_text(json.dumps(payload), encoding="utf-8")

            snapshot = load_cable_model(path)

            self.assertTrue(snapshot.provisional)
            self.assertIn("two-holder", snapshot.provenance_note)
            self.assertEqual(snapshot.model.parameters.torsional_stiffness_n_m2, 0.0)
            self.assertAlmostEqual(snapshot.model.parameters.cable_mass_kg, 0.023)

    def test_rollout_prescribes_only_attachment_and_preserves_length(self) -> None:
        simulator = self._simulator()
        state = simulator.initial_state((0.0, 0.0, 1.0))
        controls = torch.zeros((1, 2, 3), dtype=torch.float64)
        controls[:, 0, 0] = 1.0
        rollout = simulator.rollout(state, controls, create_graph=False)

        np.testing.assert_allclose(
            rollout.cable_positions_m[:, :, 0].detach().cpu().numpy(),
            rollout.attachment_positions_m.detach().cpu().numpy(),
            atol=1.0e-12,
        )
        residual = simulator.snapshot.model.maximum_segment_error_m(
            rollout.cable_positions_m[:, -1]
        )
        self.assertLess(float(torch.max(residual)), 1.0e-9)
        self.assertGreater(
            float(
                torch.linalg.vector_norm(
                    rollout.cable_positions_m[0, -1, -1]
                    - rollout.cable_positions_m[0, 0, -1]
                )
            ),
            0.0,
        )

    def test_runtime_rollout_matches_exact_dder_forward_model(self) -> None:
        simulator = self._simulator()
        state = simulator.initial_state((0.0, 0.0, 1.0))
        controls = torch.tensor(
            [[[0.8, -0.2, 0.1], [-0.4, 0.3, -0.1]]],
            dtype=torch.float64,
        )
        runtime = simulator.rollout(state, controls, create_graph=False)
        exact = simulator.rollout(state, controls, create_graph=True)
        torch.testing.assert_close(
            runtime.cable_positions_m,
            exact.cable_positions_m,
            rtol=1.0e-8,
            atol=1.0e-10,
        )
        torch.testing.assert_close(
            runtime.cable_velocities_m_s,
            exact.cable_velocities_m_s,
            rtol=1.0e-8,
            atol=1.0e-9,
        )

    def test_reduced_der_conserves_mass_and_material_parameters(self) -> None:
        source = self._simulator().snapshot
        reduced = reduce_cable_model(
            source,
            node_count=6,
            bending_stiffness_scale=1.25,
            bending_damping_scale=0.75,
        )
        self.assertEqual(reduced.node_count, 6)
        self.assertAlmostEqual(
            sum(reduced.model.parameters.vertex_masses_kg),
            sum(source.model.parameters.vertex_masses_kg),
            places=12,
        )
        self.assertEqual(
            reduced.bending_stiffness_n_m2,
            1.25 * source.bending_stiffness_n_m2,
        )
        self.assertEqual(
            reduced.bending_damping_n_m2_s,
            0.75 * source.bending_damping_n_m2_s,
        )

    def test_full_resolution_controller_preserves_the_fitted_grid(self) -> None:
        source = self._simulator().snapshot
        controller = reduce_cable_model(source, node_count=source.node_count)

        self.assertEqual(
            controller.rod_material_coordinates_m,
            source.rod_material_coordinates_m,
        )
        self.assertEqual(
            controller.model.parameters.rest_lengths_m,
            source.model.parameters.rest_lengths_m,
        )
        self.assertEqual(
            controller.model.parameters.vertex_masses_kg,
            source.model.parameters.vertex_masses_kg,
        )

    def test_controller_uses_the_minimum_stable_substep_count(self) -> None:
        source = self._simulator().snapshot
        controller = stable_controller_model(
            source,
            simulation_dt_s=0.02,
            node_count=source.node_count,
            bending_stiffness_scale=100.0,
        )

        self.assertEqual(controller.model.parameters.substeps, 2)
        self.assertLessEqual(
            controller.bending_stiffness_n_m2,
            controller.model.maximum_stable_bending_stiffness(
                0.02, pinned_endpoints=(True, False)
            ),
        )

    def test_full_state_transfer_preserves_attachment_and_controller_constraints(self) -> None:
        source = self._simulator().snapshot
        controller = reduce_cable_model(source, node_count=6)
        angles = torch.linspace(0.0, 0.8, source.node_count - 1, dtype=torch.float64)
        directions = torch.stack(
            (torch.sin(angles), torch.zeros_like(angles), -torch.cos(angles)),
            dim=1,
        )
        lengths = torch.tensor(
            source.model.parameters.rest_lengths_m, dtype=torch.float64
        )
        edges = lengths[:, None] * directions
        positions = torch.cat(
            (
                torch.tensor([[[0.1, -0.2, 1.4]]], dtype=torch.float64),
                torch.tensor([[[0.1, -0.2, 1.4]]], dtype=torch.float64)
                + torch.cumsum(edges, dim=0)[None],
            ),
            dim=1,
        )
        velocities = torch.linspace(
            0.0, 0.3, source.node_count, dtype=torch.float64
        )[None, :, None] * torch.tensor([[[1.0, -0.2, 0.1]]], dtype=torch.float64)
        state = transfer_dder_state(
            DderState(positions, velocities), source, controller
        )

        torch.testing.assert_close(state.positions_m[:, 0], positions[:, 0])
        torch.testing.assert_close(state.velocities_m_s[:, 0], velocities[:, 0])
        self.assertLess(
            float(controller.model.maximum_segment_error_m(state.positions_m)),
            1.0e-10,
        )
        edge = state.positions_m[:, 1:] - state.positions_m[:, :-1]
        relative_velocity = (
            state.velocities_m_s[:, 1:] - state.velocities_m_s[:, :-1]
        )
        self.assertLess(
            float(torch.max(torch.abs(torch.sum(edge * relative_velocity, dim=2)))),
            1.0e-10,
        )

    def test_later_feasible_hit_always_beats_earlier_infeasible_event(self) -> None:
        simulator = self._simulator()
        rollout, initial = self._event_rollout(simulator)
        # Frame 1 is at the target but is too slow.  Frame 3 is the first event
        # satisfying the position, direction, and speed constraints.
        rollout.cable_velocities_m_s[0, 1, -1, 0] = 0.50
        rollout.cable_velocities_m_s[0, 3:, -1, 0] = 1.25

        terms, impact_frame = variable_impact_rollout_cost_terms(
            rollout,
            initial,
            self._problem(),
            simulator,
            CostWeights(),
        )

        self.assertEqual(int(impact_frame[0]), 3)
        self.assertTrue(bool(terms["feasible"][0]))
        self.assertEqual(float(terms["constraint_violation"][0]), 0.0)

    def test_highest_energy_feasible_event_is_selected(self) -> None:
        simulator = self._simulator()
        rollout, initial = self._event_rollout(simulator)
        rollout.cable_velocities_m_s[0, 2:, -1, 0] = 1.25
        rollout.cable_velocities_m_s[0, 4, -1, 0] = 1.75

        terms, impact_frame = variable_impact_rollout_cost_terms(
            rollout,
            initial,
            self._problem(),
            simulator,
            CostWeights(),
        )

        self.assertEqual(int(impact_frame[0]), 4)
        self.assertTrue(bool(terms["feasible"][0]))
        self.assertGreater(float(terms["directional_tip_energy_j"][0]), 0.0)

    def test_whip_impact_requires_forward_stroke_then_recoil(self) -> None:
        simulator = self._simulator()
        rollout, initial = self._event_rollout(simulator)
        rollout.cable_velocities_m_s[0, 1:, -1, 0] = 1.25

        terms, impact_frame = variable_impact_rollout_cost_terms(
            rollout,
            initial,
            self._problem(),
            simulator,
            CostWeights(),
        )

        self.assertEqual(int(impact_frame[0]), 3)
        self.assertTrue(bool(terms["feasible"][0]))
        self.assertGreaterEqual(
            float(terms["maximum_forward_stroke_m"][0]),
            self._problem().minimum_forward_stroke_m,
        )
        self.assertGreaterEqual(
            float(terms["recoil_stroke_m"][0]),
            self._problem().minimum_recoil_stroke_m,
        )

    def test_planning_margin_tightens_only_the_predicted_hit_constraint(self) -> None:
        simulator = self._simulator()
        rollout, initial = self._event_rollout(simulator)
        rollout.cable_positions_m[:, 1:, -1, 0] = 0.04
        rollout.cable_velocities_m_s[:, 1:, -1, 0] = 1.25
        problem = MpcProblem(
            target_position_m=(0.0, 0.0, 0.50),
            impact_direction=(1.0, 0.0, 0.0),
            maximum_tip_error_m=0.05,
            planning_tip_error_margin_m=0.02,
            drone_keepout_radius_m=0.10,
            maximum_drone_excursion_m=0.25,
        )

        terms, _ = variable_impact_rollout_cost_terms(
            rollout,
            initial,
            problem,
            simulator,
            CostWeights(),
        )

        self.assertAlmostEqual(problem.planning_tip_error_limit_m, 0.03)
        self.assertFalse(bool(terms["feasible"][0]))
        self.assertGreater(float(terms["position_violation"][0]), 0.0)
        self.assertAlmostEqual(
            float(terms["planning_tip_error_limit_m"][0]), 0.03
        )

    def test_planning_margin_must_leave_a_positive_internal_tolerance(self) -> None:
        with self.assertRaisesRegex(ValueError, "planning margin"):
            MpcProblem(
                target_position_m=(0.0, 0.0, 0.50),
                impact_direction=(1.0, 0.0, 0.0),
                maximum_tip_error_m=0.05,
                planning_tip_error_margin_m=0.05,
            )

    def test_planning_angle_margin_tightens_only_the_predicted_cone(self) -> None:
        simulator = self._simulator()
        rollout, initial = self._event_rollout(simulator)
        angle = np.deg2rad(25.0)
        rollout.cable_velocities_m_s[:, 1:, -1, 0] = np.cos(angle) * 1.6
        rollout.cable_velocities_m_s[:, 1:, -1, 1] = np.sin(angle) * 1.6
        problem = MpcProblem(
            target_position_m=(0.0, 0.0, 0.50),
            impact_direction=(1.0, 0.0, 0.0),
            minimum_impact_speed_m_s=1.0,
            maximum_impact_angle_deg=30.0,
            planning_impact_angle_margin_deg=10.0,
            drone_keepout_radius_m=0.10,
            maximum_drone_excursion_m=0.25,
        )

        terms, _ = variable_impact_rollout_cost_terms(
            rollout, initial, problem, simulator, CostWeights()
        )

        self.assertAlmostEqual(problem.planning_impact_angle_limit_deg, 20.0)
        self.assertFalse(bool(terms["feasible"][0]))
        self.assertGreater(float(terms["direction_violation"][0]), 0.0)
        self.assertAlmostEqual(
            float(terms["planning_impact_angle_limit_deg"][0]), 20.0
        )

    def test_impact_event_is_evaluated_between_control_boundaries(self) -> None:
        simulator = self._simulator()
        rollout, initial = self._event_rollout(simulator)
        # Physics frames occur every 10 ms, while controls change every 20 ms.
        # Only frame 1, halfway through the first control, is a valid hit.
        rollout.drone_positions_m[0, 0, 0] = 0.10
        rollout.drone_positions_m[0, 1:, 0] = 0.0
        rollout.cable_velocities_m_s[0, 1, -1, 0] = 1.25
        rollout.cable_positions_m[0, 2:, -1, 1] = 0.10
        problem = MpcProblem(
            target_position_m=(0.0, 0.0, 0.50),
            impact_direction=(1.0, 0.0, 0.0),
            minimum_impact_speed_m_s=1.0,
            maximum_tip_error_m=0.01,
            maximum_impact_angle_deg=10.0,
            drone_keepout_radius_m=0.10,
            maximum_drone_excursion_m=0.25,
            minimum_forward_stroke_m=0.075,
            minimum_recoil_stroke_m=0.075,
            drone_workspace_center_m=(0.0, 0.0, 1.50),
        )

        terms, impact_frame = variable_impact_rollout_cost_terms(
            rollout,
            initial,
            problem,
            simulator,
            CostWeights(),
        )

        self.assertEqual(int(impact_frame[0]), 1)
        self.assertTrue(bool(terms["feasible"][0]))
        self.assertAlmostEqual(float(rollout.time_s[impact_frame[0]]), 0.01)

    def test_impact_direction_and_speed_use_world_frame_tip_velocity(self) -> None:
        simulator = self._simulator()
        rollout, initial = self._event_rollout(simulator)
        # Relative-to-drone velocity would incorrectly accept frame 1 and
        # reject frame 2.  The stationary target must use world tip velocity.
        rollout.cable_velocities_m_s[0, 1, -1, 0] = 0.50
        rollout.drone_velocities_m_s[0, 1, 0] = -2.0
        rollout.cable_velocities_m_s[0, 2, -1, 0] = 1.50
        rollout.drone_velocities_m_s[0, 2, 0] = 1.50
        rollout.cable_velocities_m_s[0, 3, -1, 0] = 1.50
        rollout.drone_velocities_m_s[0, 3, 0] = 1.50
        problem = MpcProblem(
            target_position_m=(0.0, 0.0, 0.50),
            impact_direction=(1.0, 0.0, 0.0),
            minimum_impact_speed_m_s=1.0,
            maximum_tip_error_m=0.01,
            maximum_impact_angle_deg=10.0,
            drone_keepout_radius_m=0.10,
            maximum_drone_excursion_m=0.25,
        )

        terms, impact_frame = variable_impact_rollout_cost_terms(
            rollout,
            initial,
            problem,
            simulator,
            CostWeights(),
        )

        self.assertEqual(int(impact_frame[0]), 2)
        self.assertTrue(bool(terms["feasible"][0]))
        self.assertAlmostEqual(float(terms["directional_speed_m_s"][0]), 1.50)

    def test_keepout_excursion_and_speed_limit_each_invalidate_hit(self) -> None:
        simulator = self._simulator()
        rollout, initial = self._event_rollout(simulator, batch=3)
        rollout.cable_velocities_m_s[:, 1:, -1, 0] = 1.25

        target = torch.tensor(self._problem().target_position_m, dtype=torch.float64)
        rollout.drone_positions_m[0, 1] = target  # target keepout violation
        rollout.drone_positions_m[1, 1, 0] = 0.30  # 0.25 m excursion limit
        rollout.drone_velocities_m_s[2, 1, 0] = 2.50  # 2.0 m/s speed limit

        terms, _ = variable_impact_rollout_cost_terms(
            rollout,
            initial,
            self._problem(),
            simulator,
            CostWeights(),
        )

        self.assertEqual(terms["feasible"].tolist(), [False, False, False])
        self.assertTrue(bool(torch.all(terms["safety_violation"] > 0.0)))
        self.assertLess(
            float(terms["minimum_drone_clearance_m"][0]),
            self._problem().drone_keepout_radius_m,
        )
        self.assertGreater(
            float(terms["maximum_drone_excursion_m"][1]),
            self._problem().maximum_drone_excursion_m,
        )
        self.assertGreater(
            float(terms["maximum_drone_speed_m_s"][2]),
            simulator.settings.maximum_speed_m_s,
        )

    def test_optimizer_returns_explicit_infeasible_plan_when_no_hit_exists(self) -> None:
        simulator = self._simulator()
        state = simulator.initial_state((0.0, 0.0, 1.0))
        impossible = MpcProblem(
            target_position_m=(100.0, 0.0, 0.0),
            impact_direction=(1.0, 0.0, 0.0),
            minimum_impact_speed_m_s=100.0,
            maximum_tip_error_m=1.0e-4,
            maximum_impact_angle_deg=1.0,
            drone_keepout_radius_m=0.10,
            maximum_drone_excursion_m=0.25,
        )

        plan = optimize_controls(
            simulator,
            state,
            impossible,
            OptimizerSettings(iterations=1, maximum_wall_time_s=1.0),
            optimize_impact_time=True,
        )

        self.assertFalse(plan.feasible)
        self.assertGreater(plan.constraint_violation, 0.0)
        self.assertFalse(bool(plan.cost_terms["feasible"]))
        self.assertGreater(plan.cost_terms["constraint_violation"], 0.0)
        self.assertTrue(np.isfinite(plan.cost))

    def test_realtime_defaults_define_twenty_step_matched_model_mpc(self) -> None:
        settings = RealtimeSettings()
        self.assertEqual(settings.physics_dt_s, 0.01)
        self.assertEqual(settings.horizon_s, 0.4)
        self.assertEqual(settings.control_interval_s, 0.02)
        self.assertEqual(settings.replan_interval_s, 0.1)
        self.assertEqual(settings.horizon_steps, 20)
        self.assertEqual(settings.injection_steps, 10)
        self.assertEqual(settings.apply_steps, 5)
        self.assertTrue(settings.matched_model)
        self.assertEqual(settings.initial_ipopt_iterations, 40)
        self.assertEqual(settings.ipopt_iterations, 8)
        self.assertAlmostEqual(settings.ipopt_tolerance, 1.0e-4)
        self.assertEqual(settings.maximum_acceleration_m_s2, 20.0)
        with self.assertRaisesRegex(ValueError, "Injection steps"):
            RealtimeSettings(injection_steps=20)

    def test_optimizer_and_receding_horizon_return_bounded_controls(self) -> None:
        simulator = self._simulator()
        state = simulator.initial_state((0.0, 0.0, 1.0))
        problem = MpcProblem(
            target_position_m=(0.0, 0.0, 0.40),
            impact_direction=(0.0, 0.0, -1.0),
            minimum_impact_speed_m_s=1.0e-9,
            maximum_tip_error_m=10.0,
            maximum_impact_angle_deg=89.0,
            drone_keepout_radius_m=0.01,
            maximum_drone_excursion_m=0.25,
            minimum_forward_stroke_m=0.01,
            minimum_recoil_stroke_m=0.01,
        )
        settings = OptimizerSettings(
            iterations=2,
            maximum_wall_time_s=1.0,
            replan_interval_s=0.02,
        )
        plan = optimize_controls(simulator, state, problem, settings)
        self.assertTrue(np.isfinite(plan.cost))
        self.assertGreaterEqual(len(plan.history), 1)
        self.assertTrue(np.all(np.diff(np.asarray(plan.history)) <= 0.0))
        self.assertEqual(plan.controls_m_s2.shape, (2, 3))
        self.assertLessEqual(
            plan.casting_action.forward_excursion_m,
            problem.maximum_drone_excursion_m,
        )
        self.assertLessEqual(
            plan.casting_action.recoil_excursion_m,
            problem.maximum_drone_excursion_m,
        )
        self.assertLessEqual(
            float(np.max(np.linalg.norm(plan.controls_m_s2, axis=1))),
            simulator.settings.maximum_acceleration_m_s2 * (1.0 + 1.0e-9),
        )

        execution = run_receding_horizon_mpc(
            simulator,
            state,
            problem,
            OptimizerSettings(
                iterations=1,
                maximum_wall_time_s=1.0,
                replan_interval_s=0.02,
            ),
        )
        self.assertEqual(execution.result.frame_count, 5)
        self.assertEqual(execution.executed_controls_m_s2.shape, (2, 3))
        self.assertEqual(len(execution.replanning_costs), 2)

    @staticmethod
    def _problem() -> MpcProblem:
        return MpcProblem(
            target_position_m=(0.0, 0.0, 0.50),
            impact_direction=(1.0, 0.0, 0.0),
            minimum_impact_speed_m_s=1.0,
            maximum_tip_error_m=0.01,
            maximum_impact_angle_deg=10.0,
            drone_keepout_radius_m=0.10,
            maximum_drone_excursion_m=0.25,
            minimum_forward_stroke_m=0.075,
            minimum_recoil_stroke_m=0.075,
        )

    @staticmethod
    def _event_rollout(
        simulator: WhipSimulator,
        *,
        batch: int = 1,
    ) -> tuple[TensorRollout, DroneCableState]:
        frames = 5
        nodes = simulator.snapshot.node_count
        drone_positions = torch.zeros((batch, frames, 3), dtype=torch.float64)
        drone_positions[:, :, 2] = 1.50
        drone_positions[:, :, 0] = torch.tensor(
            [0.0, 0.08, 0.10, 0.02, 0.0], dtype=torch.float64
        )
        drone_velocities = torch.zeros_like(drone_positions)
        cable_positions = torch.zeros(
            (batch, frames, nodes, 3), dtype=torch.float64
        )
        cable_positions[:, :, :, 2] = 0.50
        cable_velocities = torch.zeros_like(cable_positions)
        rollout = TensorRollout(
            time_s=torch.arange(frames, dtype=torch.float64) * 0.01,
            drone_positions_m=drone_positions,
            drone_velocities_m_s=drone_velocities,
            attachment_positions_m=drone_positions.clone(),
            cable_positions_m=cable_positions,
            cable_velocities_m_s=cable_velocities,
            accelerations_m_s2=torch.zeros((batch, 2, 3), dtype=torch.float64),
        )
        initial = DroneCableState(
            drone_positions[:, 0],
            drone_velocities[:, 0],
            DderState(cable_positions[:, 0], cable_velocities[:, 0]),
        )
        return rollout, initial

    @staticmethod
    def _artifact() -> dict[str, object]:
        rest = [0.05] * 10
        coordinates = np.r_[0.0, np.cumsum(rest)].tolist()
        masses = [0.002] * 11
        return {
            "schema": MODEL_SCHEMA,
            "measured": {
                "marker_count": 11,
                "node_count": 11,
                "rod_segments_per_marker_interval": 1,
                "marker_node_indices": list(range(11)),
                "marker_interval_lengths_m": rest,
                "rest_lengths_m": rest,
                "marker_material_coordinates_m": coordinates,
                "rod_material_coordinates_m": coordinates,
                "length_m": 0.5,
                "bare_cable_mass_kg": 0.012,
                "moving_marker_count": 10,
                "moving_marker_masses_kg": [0.001] * 10,
                "vertex_masses_kg": masses,
                "total_dynamic_mass_kg": sum(masses),
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

    def _simulator(self) -> WhipSimulator:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(self._artifact()), encoding="utf-8")
            snapshot = load_cable_model(path)
        return WhipSimulator(
            snapshot,
            SimulationSettings(
                horizon_s=0.04,
                simulation_dt_s=0.01,
                control_interval_s=0.02,
                attachment_drop_m=0.10,
                maximum_acceleration_m_s2=3.0,
                maximum_speed_m_s=2.0,
            ),
            device="cpu",
        )


if __name__ == "__main__":
    unittest.main()
