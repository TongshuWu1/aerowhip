from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cable_twin.shared.dder import DderState
from drone_mpc.model import load_cable_model
from drone_mpc.mppi import (
    MppiSettings,
    compute_dder_guidance,
    evaluate_mppi_rollout,
    interpolate_control_knots,
    optimize_mppi,
    smooth_strike_surrogate,
)
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
    build_controller_and_truth_models,
    reduce_cable_model,
    stable_controller_model,
    transfer_dder_state,
)
from drone_mpc.realtime import RealtimeSettings
from drone_mpc.receding_mppi import (
    RecedingMppiSettings,
    endpoint_conditioned_state,
    run_receding_horizon_mppi,
    save_receding_mppi_execution,
    shift_control_knots,
)
from drone_mpc.receding_mppi_gui import (
    FIXED_OBJECTIVE,
    SETTINGS_PROFILE_FIELDS,
    SETTINGS_PROFILE_SCHEMA,
    TruthModelSettings,
    model_pair_provenance,
    normalize_settings_profile,
    resolve_run_seed,
)
from drone_mpc.simulator import (
    DroneCableState,
    SimulationSettings,
    TensorRollout,
    WhipSimulator,
)
from optitrack_offline.fitting import MODEL_SCHEMA
from research_tools.mppi_ablation import (
    _wilson_interval,
    build_experiment_specs,
    resample_knots_in_time,
)
from research_tools.mppi_gradient_study import (
    CONDITIONS as GRADIENT_STUDY_CONDITIONS,
    build_parser as build_gradient_study_parser,
)
from research_tools.mppi_propagation import compute_propagation_diagnostics


class DroneMpcTests(unittest.TestCase):
    def test_settings_profile_requires_every_versioned_field(self) -> None:
        payload = {
            "schema": SETTINGS_PROFILE_SCHEMA,
            "fixed_objective": dict(FIXED_OBJECTIVE),
            **{
                section: {field: "1" for field in fields}
                for section, fields in SETTINGS_PROFILE_FIELDS.items()
            },
        }
        normalized = normalize_settings_profile(payload)
        self.assertEqual(
            normalized["plant_truth"], {"ei_scale": "1", "cb_scale": "1"}
        )

        del payload["controller"]["samples"]
        with self.assertRaisesRegex(ValueError, "controller.samples"):
            normalize_settings_profile(payload)

    def test_settings_profile_rejects_an_unknown_schema(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            normalize_settings_profile({"schema": "future_schema"})

    def test_zero_ui_seed_resolves_once_and_positive_seed_is_preserved(self) -> None:
        with patch(
            "drone_mpc.receding_mppi_gui.secrets.randbelow", return_value=41
        ):
            self.assertEqual(resolve_run_seed(0), 42)
        self.assertEqual(resolve_run_seed(17), 17)
        with self.assertRaisesRegex(ValueError, "zero or a positive"):
            resolve_run_seed(-1)

    def test_shift_control_knots_advances_and_holds_tail(self) -> None:
        knots = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (2.0, 4.0, 6.0)),
            dtype=np.float32,
        )
        shifted = shift_control_knots(knots, 0.5, 2.0, 20.0)
        np.testing.assert_allclose(
            shifted,
            np.asarray(
                ((0.5, 1.0, 1.5), (1.5, 3.0, 4.5), (2.0, 4.0, 6.0)),
                dtype=np.float32,
            ),
            atol=1.0e-6,
        )

    def test_endpoint_observer_does_not_read_true_interior_nodes(self) -> None:
        simulator = self._simulator()
        predicted = simulator.initial_state((0.0, 0.0, 1.0))
        observed = simulator.initial_state((0.1, 0.0, 1.0))
        positions = observed.cable.positions_m.clone()
        velocities = observed.cable.velocities_m_s.clone()
        positions[:, 1:-1, 1] += 0.25
        velocities[:, 1:-1, 0] -= 0.50
        hidden_interior_changed = DroneCableState(
            observed.drone_position_m,
            observed.drone_velocity_m_s,
            DderState(positions, velocities),
        )
        first = endpoint_conditioned_state(predicted, observed, 0.10)
        second = endpoint_conditioned_state(
            predicted, hidden_interior_changed, 0.10
        )
        torch.testing.assert_close(first.cable.positions_m, second.cable.positions_m)
        torch.testing.assert_close(first.cable.velocities_m_s, second.cable.velocities_m_s)
        torch.testing.assert_close(
            first.cable.positions_m[:, -1], observed.cable.positions_m[:, -1]
        )

    def test_receding_mppi_uses_shifted_warm_start_and_current_state(self) -> None:
        planner = self._simulator()
        plant = self._simulator()
        state = plant.initial_state((0.0, 0.0, 1.0))
        problem = MpcProblem(
            target_position_m=(0.0, 0.0, 0.40),
            impact_direction=(0.0, 0.0, -1.0),
            minimum_impact_speed_m_s=1.0e-9,
            maximum_tip_error_m=10.0,
            maximum_impact_angle_deg=89.0,
            drone_keepout_radius_m=0.01,
            maximum_drone_excursion_m=10.0,
            minimum_forward_stroke_m=0.0,
            minimum_recoil_stroke_m=0.0,
        )
        mppi_settings = MppiSettings(
            iterations=1,
            samples=4,
            rollout_batch_size=4,
            knot_count=2,
            acceleration_noise_sigma_m_s2=0.1,
        )
        execution_settings = RecedingMppiSettings(
            replan_interval_s=0.02,
            timeout_s=0.04,
            feedback_mode="full",
        )
        live_updates = []
        execution = run_receding_horizon_mppi(
            planner,
            plant,
            state,
            problem,
            mppi_settings,
            execution_settings,
            np.zeros((2, 3), dtype=np.float32),
            live_update=live_updates.append,
        )
        self.assertGreaterEqual(len(execution.updates), 1)
        self.assertEqual(execution.updates[0].nominal_knots_m_s2.shape, (2, 3))
        self.assertGreaterEqual(execution.result.frame_count, 3)
        self.assertGreaterEqual(execution.total_rollouts, 4)
        self.assertEqual(len(live_updates), len(execution.updates))
        self.assertEqual(
            live_updates[-1].realized.frame_count, execution.result.frame_count
        )
        self.assertEqual(
            live_updates[-1].update.index, execution.updates[-1].index
        )

        with tempfile.TemporaryDirectory() as directory:
            output = save_receding_mppi_execution(
                Path(directory) / "execution.npz",
                execution,
                problem,
                mppi_settings,
                execution_settings,
                planner.settings,
                warm_start_source="seed.npz",
                model_provenance={"matched": True},
            )
            with np.load(output) as archive:
                self.assertIn("cable_positions_m", archive)
                self.assertEqual(
                    int(np.asarray(archive["simulation_node_count"]).item()),
                    planner.snapshot.node_count,
                )
                self.assertIn("predicted_cable_positions_m", archive)
                self.assertIn("optimized_knots_m_s2", archive)
                self.assertIn("executed_control_counts", archive)
                self.assertEqual(
                    str(np.asarray(archive["controller_model_sha256"]).item()),
                    planner.snapshot.sha256,
                )
                self.assertEqual(
                    str(np.asarray(archive["plant_model_sha256"]).item()),
                    plant.snapshot.sha256,
                )
            metadata = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(
                metadata["schema"], "receding_horizon_dder_mppi_execution_v1"
            )
            self.assertEqual(metadata["feedback_mode"], "full")
            self.assertEqual(
                metadata["simulation_node_count"], planner.snapshot.node_count
            )
            self.assertEqual(metadata["mppi_settings"]["samples"], 4)
            self.assertEqual(metadata["model_provenance"], {"matched": True})

    def test_public_mppi_can_disable_legacy_workspace_radius(self) -> None:
        simulator = self._simulator()
        rollout, initial = self._event_rollout(simulator)
        rollout.cable_velocities_m_s[:, 1:, -1, 0] = 1.25
        rollout.drone_positions_m[:, 1:, 0] = 0.40
        problem = self._problem()

        _, terms, _ = evaluate_mppi_rollout(
            rollout,
            initial,
            problem,
            simulator,
            MppiSettings(
                iterations=1,
                samples=4,
                rollout_batch_size=4,
                knot_count=2,
                enforce_workspace_limit=False,
            ),
        )

        self.assertGreater(
            float(terms["maximum_drone_excursion_m"][0]),
            problem.maximum_drone_excursion_m,
        )
        self.assertEqual(float(terms["workspace_violation"][0]), 0.0)
        self.assertTrue(bool(terms["feasible"][0]))

    def test_dder_guidance_is_a_finite_local_descent_direction(self) -> None:
        simulator = self._simulator()
        state = simulator.initial_state((0.0, 0.0, 1.0))
        problem = MpcProblem(
            target_position_m=(0.20, 0.0, 0.50),
            impact_direction=(1.0, 0.0, 0.0),
            minimum_impact_speed_m_s=0.10,
            maximum_tip_error_m=0.10,
            maximum_impact_angle_deg=45.0,
            drone_keepout_radius_m=0.01,
            maximum_drone_excursion_m=0.25,
        )
        settings = MppiSettings(
            iterations=1,
            samples=4,
            rollout_batch_size=2,
            knot_count=2,
            gradient_guidance_fraction=0.5,
        )
        nominal = torch.zeros((2, 3), dtype=simulator.dtype)

        guidance = compute_dder_guidance(
            simulator,
            state,
            problem,
            settings,
            nominal,
            simulator.settings.control_count,
        )

        self.assertTrue(guidance.valid, guidance.reason)
        self.assertTrue(math.isfinite(guidance.gradient_norm))
        self.assertGreater(guidance.gradient_norm, 0.0)
        self.assertIsNotNone(guidance.negative_gradient_direction)
        costs = []
        for sign in (-1.0, 0.0, 1.0):
            candidate = nominal + (
                sign * 1.0e-4 * guidance.negative_gradient_direction
            )
            controls = interpolate_control_knots(
                candidate[None],
                simulator.settings.control_count,
                simulator.settings.maximum_acceleration_m_s2,
            )
            rollout = simulator.rollout(state, controls, create_graph=False)
            cost, _ = smooth_strike_surrogate(rollout, problem, settings)
            costs.append(float(cost[0]))
        # Moving along -gradient (sign +1 because the helper stores -g_hat)
        # must improve the differentiable objective locally.
        self.assertLess(costs[2], costs[1])
        self.assertGreater(costs[0], costs[1])

    def test_gradient_guided_mppi_records_guidance_without_replacing_sampling(self) -> None:
        simulator = self._simulator()
        state = simulator.initial_state((0.0, 0.0, 1.0))
        problem = MpcProblem(
            target_position_m=(0.20, 0.0, 0.50),
            impact_direction=(1.0, 0.0, 0.0),
            minimum_impact_speed_m_s=0.10,
            maximum_tip_error_m=0.10,
            maximum_impact_angle_deg=45.0,
            drone_keepout_radius_m=0.01,
            maximum_drone_excursion_m=0.25,
        )

        plan = optimize_mppi(
            simulator,
            state,
            problem,
            MppiSettings(
                iterations=1,
                samples=6,
                rollout_batch_size=3,
                knot_count=2,
                acceleration_noise_sigma_m_s2=1.0,
                gradient_guidance_fraction=0.5,
            ),
        )

        self.assertEqual(plan.gradient_valid_history, (True,))
        self.assertEqual(len(plan.gradient_computation_time_s_history), 1)
        self.assertGreater(plan.gradient_computation_time_s_history[0], 0.0)
        self.assertTrue(np.isfinite(plan.cost))

    def test_gradient_guidance_fraction_must_leave_standard_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "standard-MPPI"):
            MppiSettings(gradient_guidance_fraction=1.0)

    def test_low_frequency_mppi_interpolation_is_bounded_and_three_dimensional(self) -> None:
        knots = torch.tensor(
            (((10.0, 0.0, 0.0), (0.0, -10.0, 0.0), (0.0, 0.0, 10.0)),),
            dtype=torch.float64,
        )

        controls = interpolate_control_knots(knots, 11, 3.0)

        self.assertEqual(controls.shape, (1, 11, 3))
        self.assertLessEqual(
            float(torch.max(torch.linalg.vector_norm(controls, dim=2))),
            3.0 * (1.0 + 1.0e-12),
        )
        self.assertGreater(float(torch.max(torch.abs(controls[:, :, 1]))), 0.0)
        self.assertGreater(float(torch.max(torch.abs(controls[:, :, 2]))), 0.0)

    def test_low_frequency_mppi_returns_a_replayable_bounded_plan(self) -> None:
        simulator = self._simulator()
        state = simulator.initial_state((0.0, 0.0, 1.0))
        problem = MpcProblem(
            target_position_m=(0.0, 0.0, 0.40),
            impact_direction=(0.0, 0.0, -1.0),
            minimum_impact_speed_m_s=1.0e-6,
            maximum_tip_error_m=0.10,
            maximum_impact_angle_deg=89.0,
            drone_keepout_radius_m=0.01,
            maximum_drone_excursion_m=0.25,
        )

        plan = optimize_mppi(
            simulator,
            state,
            problem,
            MppiSettings(
                iterations=2,
                samples=4,
                rollout_batch_size=2,
                knot_count=2,
                acceleration_noise_sigma_m_s2=1.0,
            ),
        )

        self.assertEqual(plan.controls_m_s2.shape[1], 3)
        self.assertGreaterEqual(plan.controls_m_s2.shape[0], 1)
        self.assertLessEqual(plan.controls_m_s2.shape[0], 2)
        self.assertEqual(plan.control_knots_m_s2.shape, (2, 3))
        self.assertEqual(len(plan.history), 2)
        self.assertTrue(np.all(np.diff(np.asarray(plan.history)) <= 0.0))
        self.assertLessEqual(
            float(np.max(np.linalg.norm(plan.controls_m_s2, axis=1))),
            simulator.settings.maximum_acceleration_m_s2 * (1.0 + 1.0e-9),
        )
        self.assertTrue(np.isfinite(plan.cost))

    def test_mppi_uses_first_geometric_tip_contact_and_bounded_position_cost(self) -> None:
        simulator = self._simulator()
        rollout, initial = self._event_rollout(simulator)
        rollout.cable_positions_m[:, :, :-1, 0] = 1.0
        rollout.cable_positions_m[0, 1:, -1, 0] = torch.tensor(
            (0.10, 0.04, 0.00, 0.02), dtype=torch.float64
        )
        rollout.cable_velocities_m_s[0, 2, -1, 0] = 1.5
        problem = MpcProblem(
            target_position_m=(0.0, 0.0, 0.50),
            impact_direction=(1.0, 0.0, 0.0),
            minimum_impact_speed_m_s=1.0,
            maximum_tip_error_m=0.05,
            maximum_impact_angle_deg=20.0,
            drone_keepout_radius_m=0.10,
            maximum_drone_excursion_m=0.25,
        )
        settings = MppiSettings(
            iterations=1,
            samples=4,
            rollout_batch_size=2,
            knot_count=2,
            objective_stage="full",
        )

        cost, terms, impact = evaluate_mppi_rollout(
            rollout, initial, problem, simulator, settings
        )

        self.assertEqual(int(impact[0]), 2)
        self.assertAlmostEqual(float(terms["position_error_m"][0]), 0.04)
        self.assertTrue(bool(terms["geometric_tip_contact"][0]))
        self.assertFalse(bool(terms["physical_tip_contact"][0]))
        self.assertTrue(bool(terms["feasible"][0]))
        self.assertLessEqual(float(terms["position_cost"][0]), settings.position_weight)
        self.assertTrue(torch.isfinite(cost[0]))

    def test_fast_tip_three_hundred_mm_from_target_is_not_a_hit(self) -> None:
        simulator = self._simulator()
        rollout, initial = self._event_rollout(simulator)
        rollout.cable_positions_m[:, :, :-1, 0] = 1.0
        rollout.cable_positions_m[:, :, -1, 0] = 0.30
        rollout.cable_velocities_m_s[:, :, -1, 0] = 4.0
        problem = MpcProblem(
            target_position_m=(0.0, 0.0, 0.50),
            impact_direction=(1.0, 0.0, 0.0),
            minimum_impact_speed_m_s=3.5,
            maximum_tip_error_m=0.05,
            maximum_impact_angle_deg=35.0,
            drone_keepout_radius_m=0.10,
            maximum_drone_excursion_m=0.25,
        )

        _, terms, _ = evaluate_mppi_rollout(
            rollout,
            initial,
            problem,
            simulator,
            MppiSettings(
                iterations=1,
                samples=4,
                rollout_batch_size=2,
                knot_count=2,
                objective_stage="full",
            ),
        )

        self.assertAlmostEqual(float(terms["position_error_m"][0]), 0.30)
        self.assertGreater(float(terms["directional_speed_m_s"][0]), 3.5)
        self.assertFalse(bool(terms["geometric_tip_contact"][0]))
        self.assertFalse(bool(terms["feasible"][0]))

    def test_gradient_study_defaults_to_three_point_five_m_s_and_matched_controls(self) -> None:
        arguments = build_gradient_study_parser().parse_args([])
        conditions = {condition.key: condition for condition in GRADIENT_STUDY_CONDITIONS}

        self.assertEqual(arguments.minimum_impact_speed_m_s, 3.5)
        self.assertTrue(conditions["D"].primary)
        self.assertEqual(conditions["C"].initialization, "forward_recoil")
        self.assertEqual(conditions["D"].initialization, "forward_recoil")
        self.assertTrue(conditions["C"].guided)
        self.assertFalse(conditions["D"].guided)

    def test_mppi_diagnostic_stages_change_success_definition_not_physics(self) -> None:
        simulator = self._simulator()
        rollout, initial = self._event_rollout(simulator)
        rollout.cable_positions_m[:, :, :-1, 0] = 1.0
        rollout.cable_velocities_m_s[:, :, -1, 0] = -2.0
        problem = MpcProblem(
            target_position_m=(0.0, 0.0, 0.50),
            impact_direction=(1.0, 0.0, 0.0),
            minimum_impact_speed_m_s=1.0,
            maximum_tip_error_m=0.05,
            maximum_impact_angle_deg=20.0,
            drone_keepout_radius_m=0.10,
            maximum_drone_excursion_m=0.25,
        )

        _, position_terms, _ = evaluate_mppi_rollout(
            rollout,
            initial,
            problem,
            simulator,
            MppiSettings(
                iterations=1,
                samples=4,
                rollout_batch_size=2,
                knot_count=2,
                objective_stage="position",
            ),
        )
        _, full_terms, _ = evaluate_mppi_rollout(
            rollout,
            initial,
            problem,
            simulator,
            MppiSettings(
                iterations=1,
                samples=4,
                rollout_batch_size=2,
                knot_count=2,
                objective_stage="full",
            ),
        )

        self.assertTrue(bool(position_terms["feasible"][0]))
        self.assertFalse(bool(full_terms["feasible"][0]))
        self.assertEqual(float(position_terms["speed_cost"][0]), 0.0)
        self.assertEqual(float(position_terms["direction_cost"][0]), 0.0)
        self.assertGreater(float(full_terms["speed_cost"][0]), 0.0)
        self.assertGreater(float(full_terms["direction_cost"][0]), 0.0)

    def test_predictive_speed_ablation_is_opt_in(self) -> None:
        simulator = self._simulator()
        rollout, initial = self._event_rollout(simulator)
        rollout.cable_positions_m[:, :, :-1, 0] = 1.0
        rollout.cable_positions_m[:, 1:, -1, 0] = 0.35
        rollout.cable_velocities_m_s[:, :, -1] = 0.0
        problem = MpcProblem(
            target_position_m=(0.0, 0.0, 0.50),
            impact_direction=(1.0, 0.0, 0.0),
            minimum_impact_speed_m_s=2.0,
            maximum_tip_error_m=0.05,
            maximum_impact_angle_deg=20.0,
            drone_keepout_radius_m=0.10,
            maximum_drone_excursion_m=0.25,
        )

        base_cost, base_terms, _ = evaluate_mppi_rollout(
            rollout,
            initial,
            problem,
            simulator,
            MppiSettings(iterations=1, samples=4, rollout_batch_size=2, knot_count=2),
        )
        ablation_cost, ablation_terms, _ = evaluate_mppi_rollout(
            rollout,
            initial,
            problem,
            simulator,
            MppiSettings(
                iterations=1,
                samples=4,
                rollout_batch_size=2,
                knot_count=2,
                predictive_speed_weight=1.0,
                predictive_velocity_gate_sigma_m=0.45,
                predictive_speed_ratio=0.25,
            ),
        )

        self.assertEqual(float(base_terms["predictive_speed_cost"][0]), 0.0)
        self.assertGreater(float(ablation_terms["predictive_speed_cost"][0]), 0.0)
        self.assertGreater(float(ablation_cost[0]), float(base_cost[0]))

    def test_ablation_knot_resampling_preserves_absolute_time(self) -> None:
        source = np.asarray(((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)))

        target = resample_knots_in_time(source, 2.0, 1.0, 3)

        np.testing.assert_allclose(target[:, 0], (1.0, 1.5, 2.0))
        np.testing.assert_allclose(target[:, 1:], 0.0)

    def test_discovery_study_is_paired_at_fixed_two_second_horizon(self) -> None:
        specifications = build_experiment_specs("initialization")

        self.assertEqual(len(specifications), 6)
        self.assertTrue(all(spec.horizon_s == 2.0 for spec in specifications))
        self.assertTrue(all(spec.knot_count == 11 for spec in specifications))
        self.assertEqual(
            {spec.initialization for spec in specifications},
            {
                "zero",
                "random",
                "forward_recoil",
                "backward_forward",
                "lateral",
                "continuation",
            },
        )

    def test_wilson_interval_is_finite_at_zero_and_complete_success(self) -> None:
        zero = _wilson_interval(0, 20)
        complete = _wilson_interval(20, 20)

        self.assertEqual(zero[0], 0.0)
        self.assertEqual(complete[0], 1.0)
        self.assertGreater(zero[2], 0.0)
        self.assertLess(complete[1], 1.0)

    def test_propagation_diagnostics_resolve_node_energy_and_events(self) -> None:
        time_s = np.asarray((0.0, 0.1, 0.2))
        drone_position = np.asarray(((0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.05, 0.0, 1.0)))
        drone_velocity = np.zeros((3, 3))
        cable_position = np.zeros((3, 4, 3))
        cable_position[:, :, 2] = np.asarray((1.0, 0.9, 0.8, 0.7))[None]
        cable_velocity = np.zeros_like(cable_position)
        cable_velocity[1, 1, 0] = 1.0
        cable_velocity[2, 3, 0] = 3.0

        diagnostics = compute_propagation_diagnostics(
            time_s=time_s,
            drone_positions_m=drone_position,
            drone_velocities_m_s=drone_velocity,
            cable_positions_m=cable_position,
            cable_velocities_m_s=cable_velocity,
            impact_frame=2,
            vertex_masses_kg=np.ones(4),
            rest_lengths_m=np.full(3, 0.1),
            forward_direction=np.asarray((1.0, 0.0, 0.0)),
        )

        self.assertEqual(diagnostics.peak_forward_frame, 1)
        self.assertEqual(diagnostics.peak_tip_speed_frame, 2)
        self.assertEqual(diagnostics.relative_speed_m_s.shape, (3, 4))
        self.assertAlmostEqual(diagnostics.distal_energy_fraction_at_impact, 1.0)

    def test_mppi_non_tip_contact_is_checked_strictly_before_tip_impact(self) -> None:
        simulator = self._simulator()
        rollout, initial = self._event_rollout(simulator)
        rollout.cable_positions_m[:, :, :-1, 0] = 1.0
        rollout.cable_positions_m[0, 1:, -1, 0] = torch.tensor(
            (0.10, 0.04, 0.00, 0.02), dtype=torch.float64
        )
        rollout.cable_velocities_m_s[0, 2, -1, 0] = 1.5
        # The penultimate material point shares the target region on the tip's
        # first-contact frame.  This is not an earlier non-tip strike.
        rollout.cable_positions_m[0, 2, -2, 0] = 0.0
        problem = MpcProblem(
            target_position_m=(0.0, 0.0, 0.50),
            impact_direction=(1.0, 0.0, 0.0),
            minimum_impact_speed_m_s=1.0,
            maximum_tip_error_m=0.05,
            maximum_impact_angle_deg=20.0,
            drone_keepout_radius_m=0.10,
            maximum_drone_excursion_m=0.25,
        )
        settings = MppiSettings(
            iterations=1,
            samples=4,
            rollout_batch_size=2,
            knot_count=2,
            objective_stage="full",
        )

        _, same_frame_terms, impact = evaluate_mppi_rollout(
            rollout, initial, problem, simulator, settings
        )
        self.assertEqual(int(impact[0]), 2)
        self.assertEqual(float(same_frame_terms["non_tip_contact_violation"][0]), 0.0)
        self.assertTrue(bool(same_frame_terms["feasible"][0]))

        # Moving the same non-tip contact one frame earlier must disqualify it.
        rollout.cable_positions_m[0, 1, -2, 0] = 0.0
        _, earlier_terms, _ = evaluate_mppi_rollout(
            rollout, initial, problem, simulator, settings
        )
        self.assertGreater(float(earlier_terms["non_tip_contact_violation"][0]), 0.0)
        self.assertFalse(bool(earlier_terms["feasible"][0]))

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

    def test_truth_model_mismatch_changes_only_ei_and_cb(self) -> None:
        source = self._simulator().snapshot
        controller, truth = build_controller_and_truth_models(
            source,
            simulation_dt_s=0.01,
            node_count=6,
            truth_bending_stiffness_scale=1.4,
            truth_bending_damping_scale=0.6,
        )

        self.assertNotEqual(controller.sha256, truth.sha256)
        self.assertEqual(controller.node_count, truth.node_count)
        self.assertEqual(
            controller.rod_material_coordinates_m,
            truth.rod_material_coordinates_m,
        )
        controller_parameters = controller.model.parameters
        truth_parameters = truth.model.parameters
        self.assertEqual(
            controller_parameters.rest_lengths_m, truth_parameters.rest_lengths_m
        )
        self.assertEqual(
            controller_parameters.vertex_masses_kg, truth_parameters.vertex_masses_kg
        )
        self.assertEqual(controller_parameters.substeps, truth_parameters.substeps)
        self.assertEqual(
            controller_parameters.constraint_iterations,
            truth_parameters.constraint_iterations,
        )
        self.assertEqual(
            controller_parameters.gravity_camera_m_s2,
            truth_parameters.gravity_camera_m_s2,
        )
        self.assertEqual(
            truth.bending_stiffness_n_m2,
            1.4 * controller.bending_stiffness_n_m2,
        )
        self.assertEqual(
            truth.bending_damping_n_m2_s,
            0.6 * controller.bending_damping_n_m2_s,
        )

        provenance = model_pair_provenance(
            source,
            controller,
            truth,
            TruthModelSettings(1.4, 0.6),
        )
        self.assertFalse(provenance["matched"])
        self.assertEqual(
            provenance["controlled_mismatch"], "EI and Cb only"
        )

    def test_unit_truth_scales_are_an_exact_matched_model(self) -> None:
        source = self._simulator().snapshot
        controller, truth = build_controller_and_truth_models(
            source,
            simulation_dt_s=0.01,
            node_count=source.node_count,
        )

        self.assertIs(controller, truth)
        self.assertEqual(controller.sha256, source.sha256)

    def test_receding_mppi_accepts_an_ei_cb_truth_mismatch(self) -> None:
        baseline = self._simulator()
        controller_snapshot, plant_snapshot = build_controller_and_truth_models(
            baseline.snapshot,
            simulation_dt_s=baseline.settings.simulation_dt_s,
            node_count=baseline.snapshot.node_count,
            truth_bending_stiffness_scale=1.2,
            truth_bending_damping_scale=0.8,
        )
        planner = WhipSimulator(controller_snapshot, baseline.settings, device="cpu")
        plant = WhipSimulator(plant_snapshot, baseline.settings, device="cpu")
        problem = MpcProblem(
            target_position_m=(0.0, 0.0, 0.40),
            impact_direction=(0.0, 0.0, -1.0),
            minimum_impact_speed_m_s=1.0e-9,
            maximum_tip_error_m=10.0,
            maximum_impact_angle_deg=89.0,
            drone_keepout_radius_m=0.01,
            maximum_drone_excursion_m=10.0,
            minimum_forward_stroke_m=0.0,
            minimum_recoil_stroke_m=0.0,
        )
        execution = run_receding_horizon_mppi(
            planner,
            plant,
            plant.initial_state((0.0, 0.0, 1.0)),
            problem,
            MppiSettings(
                iterations=1,
                samples=4,
                rollout_batch_size=4,
                knot_count=2,
                acceleration_noise_sigma_m_s2=0.1,
            ),
            RecedingMppiSettings(
                replan_interval_s=0.02,
                timeout_s=0.02,
                feedback_mode="full",
            ),
            np.zeros((2, 3), dtype=np.float32),
        )

        self.assertEqual(execution.controller_model_sha256, controller_snapshot.sha256)
        self.assertEqual(execution.plant_model_sha256, plant_snapshot.sha256)
        self.assertNotEqual(
            execution.controller_model_sha256, execution.plant_model_sha256
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
