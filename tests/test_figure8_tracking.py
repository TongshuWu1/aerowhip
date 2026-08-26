from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from cable_twin.shared.dder import DderState
from drone_mpc.figure8_canvas import causal_rolling_mean
from drone_mpc.history_observer import (
    DderHistoryObserver,
    EndpointHistoryObservation,
    HistoryObserverSettings,
)
from drone_mpc.figure8_tracking import (
    Figure8ExecutionSettings,
    Figure8Reference,
    TrackingCostSettings,
    endpoint_only_observation_state,
    path_following_metrics,
    periodic_swing_seed_bank,
    run_figure8_tracking,
    tracking_rollout_objective,
)
from drone_mpc.model import load_cable_model
from drone_mpc.mppi import MppiSettings
from drone_mpc.receding_mppi import endpoint_conditioned_state
from drone_mpc.reduced import build_controller_and_truth_models
from drone_mpc.simulator import (
    DroneCableState,
    SimulationSettings,
    TensorRollout,
    WhipSimulator,
)
from optitrack_offline.config import DEFAULT_MODEL_PATH


class Figure8TrackingTests(unittest.TestCase):
    def test_error_plot_rolling_mean_is_causal_and_time_based(self) -> None:
        times = np.asarray((0.0, 1.0, 2.0, 3.0))
        errors = np.asarray((1.0, 3.0, 5.0, 7.0))
        smoothed = causal_rolling_mean(times, errors, window_s=2.0)
        self.assertTrue(np.allclose(smoothed, (1.0, 2.0, 3.0, 5.0)))

    def test_reference_is_a_flat_closed_geometric_path(self) -> None:
        reference = Figure8Reference((0.2, -0.1, 0.7), 1.0, 0.8)
        positions = reference.path_numpy(2001)
        self.assertTrue(np.allclose(positions[0], positions[-1], atol=1.0e-12))
        self.assertTrue(np.allclose(positions[:, 2], 0.7))
        self.assertAlmostEqual(
            float(np.max(np.abs(positions[:, 0] - 0.2))), 1.0, places=5
        )
        self.assertAlmostEqual(
            float(np.max(np.abs(positions[:, 1] + 0.1))), 0.8, places=5
        )

    def test_local_projection_tracks_unwrapped_branch_without_timing(self) -> None:
        reference = Figure8Reference((0.0, 0.0, 0.4), 1.0, 0.8)
        true_progress = np.asarray((0.01, 0.03, 0.06, 0.10))
        positions = reference.point_numpy(true_progress)
        projected, progress, error = reference.project_sequence_numpy(
            positions, 0.0
        )
        self.assertTrue(np.all(np.diff(progress) >= 0.0))
        self.assertTrue(np.allclose(projected, positions, atol=0.012))
        self.assertLess(float(np.max(error)), 0.012)

        crossing = reference.point_numpy(0.5)[None]
        _projected, branch_progress, _error = reference.project_sequence_numpy(
            crossing, 0.49
        )
        self.assertGreater(float(branch_progress[0]), 0.45)

    def test_gpu_cost_projection_can_follow_more_than_quarter_cycle(self) -> None:
        reference = Figure8Reference((0.0, 0.0, 0.8), 1.0, 0.8)
        expected_progress = np.linspace(0.0, 0.8, 81)
        positions = torch.as_tensor(
            reference.point_numpy(expected_progress)[None], dtype=torch.float64
        )
        error, progress, projected = path_following_metrics(
            positions, reference, 0.0
        )
        self.assertGreater(float(progress[0, -1]), 0.79)
        self.assertLess(float(torch.max(error)), 0.012)
        self.assertTrue(torch.allclose(projected, positions, atol=0.012))

    def test_periodic_seed_bank_contains_zero_and_bounded_recoil_strokes(self) -> None:
        seeds, labels = periodic_swing_seed_bank(
            13,
            1.0,
            6.0,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        self.assertEqual(seeds.shape, (49, 13, 3))
        self.assertEqual(len(labels), 49)
        self.assertTrue(torch.equal(seeds[0], torch.zeros_like(seeds[0])))
        self.assertLessEqual(
            float(torch.max(torch.linalg.vector_norm(seeds, dim=2))),
            6.0 + 1.0e-6,
        )
        times = torch.linspace(0.0, 1.0, 13)
        integrated_acceleration = torch.trapezoid(seeds[1:], times, dim=1)
        self.assertTrue(
            torch.allclose(
                integrated_acceleration,
                torch.zeros_like(integrated_acceleration),
                atol=1.0e-6,
            )
        )
        long_seeds, _labels = periodic_swing_seed_bank(
            21,
            2.0,
            6.0,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        self.assertTrue(torch.equal(long_seeds[:, 11:], torch.zeros_like(long_seeds[:, 11:])))
        self.assertTrue(torch.equal(long_seeds[:, 0], torch.zeros_like(long_seeds[:, 0])))

    def test_minimal_objective_does_not_reward_a_prescribed_swing(self) -> None:
        source = load_cable_model(DEFAULT_MODEL_PATH)
        simulation = SimulationSettings(
            horizon_s=0.02,
            simulation_dt_s=0.02,
            control_interval_s=0.02,
        )
        model, _truth = build_controller_and_truth_models(
            source,
            simulation_dt_s=0.02,
            node_count=11,
            truth_bending_stiffness_scale=1.0,
            truth_bending_damping_scale=1.0,
        )
        simulator = WhipSimulator(model, simulation, device="cpu")
        state = simulator.initial_state((0.0, 0.0, 1.45))
        cable_positions = state.cable.positions_m[:, None].repeat(1, 2, 1, 1)
        drone_positions = state.drone_position_m[:, None].repeat(1, 2, 1)
        tangent = Figure8Reference((0.0, 0.0, 0.0)).unit_tangent_torch(
            torch.tensor(0.0)
        )

        def rollout(root_speed: torch.Tensor, tip_speed: torch.Tensor) -> TensorRollout:
            velocities = torch.zeros_like(cable_positions)
            velocities[:, :, 0] = root_speed
            velocities[:, :, -1] = tip_speed
            return TensorRollout(
                time_s=torch.tensor((0.0, 0.02)),
                drone_positions_m=drone_positions,
                drone_velocities_m_s=root_speed.reshape(1, 1, 3).repeat(1, 2, 1),
                attachment_positions_m=cable_positions[:, :, 0],
                cable_positions_m=cable_positions,
                cable_velocities_m_s=velocities,
                accelerations_m_s2=torch.zeros((1, 1, 3)),
            )

        reference = Figure8Reference(
            tuple(float(value) for value in state.cable.positions_m[0, -1]),
            1.0,
            0.8,
        )
        carried = rollout(tangent, tangent)
        swinging = rollout(torch.zeros(3), tangent)
        carried_cost, carried_terms = tracking_rollout_objective(
            carried, state, reference, 0.0, simulator, TrackingCostSettings()
        )
        swinging_cost, swinging_terms = tracking_rollout_objective(
            swinging, state, reference, 0.0, simulator, TrackingCostSettings()
        )
        self.assertTrue(torch.allclose(carried_cost, swinging_cost))
        self.assertNotIn("swing_tangent_velocity_reward", carried_terms)
        self.assertNotIn("normal_tip_velocity_cost", carried_terms)
        self.assertNotIn("tangent_acceleration_cost", carried_terms)

        stationary_tip = rollout(-tangent, torch.zeros(3))
        stationary_cost, stationary_terms = tracking_rollout_objective(
            stationary_tip, state, reference, 0.0, simulator, TrackingCostSettings()
        )
        self.assertTrue(torch.allclose(carried_cost, stationary_cost))
        self.assertNotIn("swing_tangent_velocity_reward", stationary_terms)

        accelerating_tip_velocity = swinging.cable_velocities_m_s.clone()
        accelerating_tip_velocity[:, 0, -1] = 0.0
        accelerating_tip = TensorRollout(
            time_s=swinging.time_s,
            drone_positions_m=swinging.drone_positions_m,
            drone_velocities_m_s=swinging.drone_velocities_m_s,
            attachment_positions_m=swinging.attachment_positions_m,
            cable_positions_m=swinging.cable_positions_m,
            cable_velocities_m_s=accelerating_tip_velocity,
            accelerations_m_s2=swinging.accelerations_m_s2,
        )
        accelerating_cost, accelerating_terms = tracking_rollout_objective(
            accelerating_tip,
            state,
            reference,
            0.0,
            simulator,
            TrackingCostSettings(),
        )
        self.assertGreater(float(accelerating_cost), float(swinging_cost))
        self.assertGreater(
            float(accelerating_terms["tip_motion_smoothness_cost"]), 0.0
        )

    def test_endpoint_conditioning_does_not_read_true_interior(self) -> None:
        dtype = torch.float64
        root = torch.tensor(((0.0, 0.0, 1.0),), dtype=dtype)
        root_velocity = torch.tensor(((0.1, 0.0, 0.0),), dtype=dtype)
        coordinate = torch.linspace(0.0, 1.0, 11, dtype=dtype)
        predicted_positions = torch.zeros((1, 11, 3), dtype=dtype)
        predicted_positions[0, :, 2] = 0.9 - coordinate
        predicted_velocities = torch.zeros_like(predicted_positions)
        predicted = DroneCableState(
            root,
            root_velocity,
            DderState(predicted_positions, predicted_velocities),
        )
        observed_a_positions = predicted_positions.clone()
        observed_b_positions = predicted_positions.clone()
        observed_b_positions[:, 1:-1, 0] += 0.4 * torch.sin(
            math.pi * coordinate[1:-1]
        )
        observed_a_velocities = predicted_velocities.clone()
        observed_b_velocities = predicted_velocities.clone()
        observed_b_velocities[:, 1:-1, 1] += 2.0
        observed_a = DroneCableState(
            root,
            root_velocity,
            DderState(observed_a_positions, observed_a_velocities),
        )
        observed_b = DroneCableState(
            root,
            root_velocity,
            DderState(observed_b_positions, observed_b_velocities),
        )
        endpoint_a = endpoint_only_observation_state(predicted, observed_a)
        endpoint_b = endpoint_only_observation_state(predicted, observed_b)
        self.assertTrue(
            torch.equal(endpoint_a.cable.positions_m, endpoint_b.cable.positions_m)
        )
        self.assertTrue(
            torch.equal(
                endpoint_a.cable.velocities_m_s,
                endpoint_b.cable.velocities_m_s,
            )
        )
        conditioned_a = endpoint_conditioned_state(predicted, endpoint_a, 0.1)
        conditioned_b = endpoint_conditioned_state(predicted, endpoint_b, 0.1)
        self.assertTrue(
            torch.equal(
                conditioned_a.cable.positions_m,
                conditioned_b.cable.positions_m,
            )
        )
        self.assertTrue(
            torch.equal(
                conditioned_a.cable.velocities_m_s,
                conditioned_b.cable.velocities_m_s,
            )
        )

    def test_short_cpu_smoke_uses_identical_matched_eleven_node_models(self) -> None:
        source = load_cable_model(DEFAULT_MODEL_PATH)
        simulation = SimulationSettings(
            horizon_s=0.04,
            simulation_dt_s=0.02,
            control_interval_s=0.02,
            attachment_drop_m=0.10,
            maximum_acceleration_m_s2=6.0,
            maximum_speed_m_s=3.0,
        )
        controller, truth = build_controller_and_truth_models(
            source,
            simulation_dt_s=simulation.simulation_dt_s,
            node_count=11,
            truth_bending_stiffness_scale=1.0,
            truth_bending_damping_scale=1.0,
        )
        self.assertEqual(controller.node_count, 11)
        self.assertEqual(controller.sha256, truth.sha256)
        mppi = MppiSettings(
            iterations=1,
            samples=4,
            rollout_batch_size=4,
            knot_count=3,
            acceleration_noise_sigma_m_s2=0.2,
            seed=3,
        )
        summaries: dict[str, dict[str, float]] = {}
        for mode in ("full", "endpoint", "history"):
            planner = WhipSimulator(controller, simulation, device="cpu")
            plant = WhipSimulator(truth, simulation, device="cpu")
            initial = plant.initial_state((0.0, 0.0, 1.45))
            center = tuple(
                float(value)
                for value in initial.cable.positions_m[0, -1].detach().cpu().numpy()
            )
            initial_position = initial.cable.positions_m[0].detach().cpu().numpy()
            initial_velocity = initial.cable.velocities_m_s[0].detach().cpu().numpy()
            history_observer = (
                DderHistoryObserver(
                    controller,
                    initial.cable,
                    EndpointHistoryObservation(
                        0.0,
                        initial_position[0],
                        initial_velocity[0],
                        initial_position[-1],
                    ),
                    HistoryObserverSettings(
                        history_duration_s=0.04,
                        maximum_iterations=1,
                    ),
                    device="cpu",
                )
                if mode == "history"
                else None
            )
            execution = run_figure8_tracking(
                planner,
                plant,
                initial,
                Figure8Reference(center, 0.03, 0.02),
                mppi,
                TrackingCostSettings(),
                Figure8ExecutionSettings(
                    replan_interval_s=0.02,
                    observation_mode=mode,  # type: ignore[arg-type]
                    duration_s=0.02,
                ),
                history_observer=history_observer,
            )
            self.assertEqual(execution.observation_mode, mode)
            self.assertEqual(execution.cable_positions_m.shape[1:], (11, 3))
            self.assertEqual(
                execution.controller_model_sha256,
                execution.plant_model_sha256,
            )
            summaries[mode] = execution.summary
        self.assertEqual(set(summaries), {"full", "endpoint", "history"})

    def test_observation_mode_can_switch_between_replanning_updates(self) -> None:
        source = load_cable_model(DEFAULT_MODEL_PATH)
        simulation = SimulationSettings(
            horizon_s=0.04,
            simulation_dt_s=0.02,
            control_interval_s=0.02,
            attachment_drop_m=0.10,
            maximum_acceleration_m_s2=6.0,
            maximum_speed_m_s=3.0,
        )
        controller, truth = build_controller_and_truth_models(
            source,
            simulation_dt_s=simulation.simulation_dt_s,
            node_count=11,
            truth_bending_stiffness_scale=1.0,
            truth_bending_damping_scale=1.0,
        )
        planner = WhipSimulator(controller, simulation, device="cpu")
        plant = WhipSimulator(truth, simulation, device="cpu")
        initial = plant.initial_state((0.0, 0.0, 1.45))
        center = tuple(
            float(value)
            for value in initial.cable.positions_m[0, -1].detach().cpu().numpy()
        )
        requested_modes = iter(("full", "endpoint"))
        execution = run_figure8_tracking(
            planner,
            plant,
            initial,
            Figure8Reference(center, 0.03, 0.02),
            MppiSettings(
                iterations=1,
                samples=4,
                rollout_batch_size=4,
                knot_count=3,
                acceleration_noise_sigma_m_s2=0.2,
                seed=5,
            ),
            TrackingCostSettings(),
            Figure8ExecutionSettings(
                replan_interval_s=0.02,
                observation_mode="full",
                duration_s=0.04,
            ),
            observation_mode_provider=lambda: next(requested_modes),  # type: ignore[arg-type]
        )
        self.assertEqual(execution.observation_mode, "mixed")
        self.assertEqual(
            execution.observation_modes.tolist(),
            ["full", "full", "endpoint"],
        )


if __name__ == "__main__":
    unittest.main()
