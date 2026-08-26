from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np
import torch

from drone_mpc.distributed_adaptation import ParameterEstimate
from drone_mpc.model import load_cable_model
from drone_mpc.mpc import MpcProblem
from drone_mpc.mppi import MppiSettings, evaluate_mppi_rollout
from drone_mpc.online_adaptation import observations_from_execution
from drone_mpc.reduced import build_controller_and_truth_models
from drone_mpc.receding_mppi import RecedingMppiExecution
from drone_mpc.simulator import (
    SimulationResult,
    SimulationSettings,
    WhipSimulator,
)
from optitrack_offline.config import DEFAULT_MODEL_PATH


class OnlineAccelerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = load_cable_model(DEFAULT_MODEL_PATH)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_arbitrary_node_full_horizon_matches_reference_step_replay(self) -> None:
        simulation = SimulationSettings(
            horizon_s=0.10,
            simulation_dt_s=0.02,
            control_interval_s=0.02,
            maximum_acceleration_m_s2=20.0,
        )
        model, _ = build_controller_and_truth_models(
            self.source,
            simulation_dt_s=simulation.simulation_dt_s,
            node_count=6,
        )
        controls = torch.tensor(
            [
                [[2.0, 0.0, 0.0], [-1.0, 0.5, 0.0], [0.0, 0.0, 0.0],
                 [0.5, -0.2, 0.1], [0.0, 0.0, 0.0]],
                [[-1.0, 0.3, 0.0], [1.5, 0.0, -0.2], [0.0, 0.0, 0.0],
                 [0.2, 0.4, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        with patch.dict(os.environ, {"CABLE_TWIN_FULL_HORIZON_GRAPH": "1"}):
            fast = WhipSimulator(model, simulation, device="cuda")
            self.assertTrue(fast.require_online_acceleration().online_ready)
            self.assertFalse(fast.runtime_acceleration.fused_mechanics)
            accelerated = fast.rollout(
                fast.initial_state((0.0, 0.0, 1.5)), controls, create_graph=False
            )
            torch.cuda.synchronize()
        with patch.dict(os.environ, {"CABLE_TWIN_FULL_HORIZON_GRAPH": "0"}):
            reference = WhipSimulator(model, simulation, device="cuda")
            reference_rollout = reference.rollout(
                reference.initial_state((0.0, 0.0, 1.5)),
                controls,
                create_graph=False,
            )
            torch.cuda.synchronize()
        torch.testing.assert_close(
            accelerated.cable_positions_m,
            reference_rollout.cable_positions_m,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            accelerated.cable_velocities_m_s,
            reference_rollout.cable_velocities_m_s,
            rtol=0.0,
            atol=0.0,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_fused_mppi_cost_accepts_non_11_node_rollout(self) -> None:
        simulation = SimulationSettings(
            horizon_s=0.10,
            simulation_dt_s=0.02,
            control_interval_s=0.02,
            maximum_acceleration_m_s2=20.0,
        )
        model, _ = build_controller_and_truth_models(
            self.source,
            simulation_dt_s=simulation.simulation_dt_s,
            node_count=6,
        )
        simulator = WhipSimulator(model, simulation, device="cuda")
        initial = simulator.initial_state((0.0, 0.0, 1.5))
        controls = torch.zeros((4, 5, 3), dtype=torch.float32, device="cuda")
        rollout = simulator.rollout(initial, controls, create_graph=False)
        problem = MpcProblem(
            target_position_m=(1.0, 0.0, 1.4),
            impact_direction=(1.0, 0.0, 0.0),
            maximum_drone_excursion_m=1.0,
        )
        settings = MppiSettings(
            iterations=1,
            samples=4,
            rollout_batch_size=4,
            knot_count=2,
            enforce_workspace_limit=False,
        )
        with patch.dict(os.environ, {"DRONE_MPPI_FUSED_COST": "1"}):
            fused = evaluate_mppi_rollout(
                rollout, initial, problem, simulator, settings
            )
        with patch.dict(os.environ, {"DRONE_MPPI_FUSED_COST": "0"}):
            reference = evaluate_mppi_rollout(
                rollout, initial, problem, simulator, settings
            )
        torch.testing.assert_close(fused[0], reference[0], rtol=2.0e-5, atol=2.0e-5)
        self.assertTrue(torch.equal(fused[2], reference[2]))
        for name in reference[1]:
            if torch.all(torch.isinf(reference[1][name])):
                self.assertTrue(torch.all(fused[1][name] > 1.0e30))
                continue
            torch.testing.assert_close(
                fused[1][name], reference[1][name], rtol=2.0e-5, atol=2.0e-5
            )

    def test_execution_observations_preserve_distributed_state_and_actions(self) -> None:
        simulation = SimulationSettings(
            horizon_s=0.1,
            simulation_dt_s=0.01,
            control_interval_s=0.02,
        )
        frames = 5
        nodes = 6
        result = SimulationResult(
            time_s=np.arange(frames, dtype=np.float64) * 0.01,
            drone_positions_m=np.zeros((frames, 3)),
            drone_velocities_m_s=np.ones((frames, 3)),
            attachment_positions_m=np.zeros((frames, 3)),
            cable_positions_m=np.zeros((frames, nodes, 3)),
            cable_velocities_m_s=np.ones((frames, nodes, 3)),
            accelerations_m_s2=np.asarray(((1.0, 0.0, 0.0), (2.0, 0.0, 0.0))),
            target_position_m=np.zeros(3),
            impact_direction=np.asarray((1.0, 0.0, 0.0)),
            model_sha256="test",
        )
        execution = RecedingMppiExecution(
            result=result,
            updates=(),
            cost=0.0,
            cost_terms={"geometric_tip_contact": 0.0},
            impact_time_s=0.04,
            feasible=False,
            terminal_reason="timeout",
            total_rollouts=0,
            total_planning_wall_time_s=0.0,
            feedback_mode="full",
        )
        observations = observations_from_execution(
            execution, simulation, estimate=ParameterEstimate()
        )
        self.assertEqual(len(observations), frames)
        self.assertEqual(observations[-1].cable_positions_m.shape, (nodes, 3))
        np.testing.assert_allclose(observations[1].executed_action_m_s2, (1, 0, 0))
        np.testing.assert_allclose(observations[3].executed_action_m_s2, (2, 0, 0))


if __name__ == "__main__":
    unittest.main()
