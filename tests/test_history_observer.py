from __future__ import annotations

from dataclasses import fields
import unittest

import numpy as np
import torch

from cable_twin.shared.dder import DderState, START_PINNED_FREE_END
from drone_mpc.history_observer import (
    DderHistoryObserver,
    EndpointHistoryObservation,
    HistoryObserverSettings,
    spatial_correction_basis,
)
from drone_mpc.model import load_cable_model
from drone_mpc.receding_mppi import perturb_cable_state
from drone_mpc.reduced import build_controller_and_truth_models
from drone_mpc.simulator import SimulationSettings, WhipSimulator
from optitrack_offline.config import DEFAULT_MODEL_PATH


class HistoryObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = load_cable_model(DEFAULT_MODEL_PATH)
        simulation = SimulationSettings(
            horizon_s=0.10,
            simulation_dt_s=0.02,
            control_interval_s=0.02,
        )
        cls.snapshot, _truth = build_controller_and_truth_models(
            source,
            simulation_dt_s=simulation.simulation_dt_s,
            node_count=11,
            truth_bending_stiffness_scale=1.0,
            truth_bending_damping_scale=1.0,
        )
        cls.simulator = WhipSimulator(cls.snapshot, simulation, device="cpu")
        cls.initial = cls.simulator.initial_state((0.0, 0.0, 1.45))
        cls.dt = simulation.simulation_dt_s

    @staticmethod
    def _observation(timestamp_s: float, state: DderState) -> EndpointHistoryObservation:
        return EndpointHistoryObservation(
            timestamp_s=timestamp_s,
            root_position_m=state.positions_m[0, 0].detach().cpu().numpy(),
            root_velocity_m_s=state.velocities_m_s[0, 0].detach().cpu().numpy(),
            tip_position_m=state.positions_m[0, -1].detach().cpu().numpy(),
        )

    @classmethod
    def _step(cls, state: DderState) -> DderState:
        boundary = state.positions_m[:, :1]
        return cls.snapshot.model.step_runtime(
            state,
            boundary,
            torch.full((1,), cls.dt, dtype=state.positions_m.dtype),
            cls.snapshot.model.runtime_constants(state.positions_m),
            iterative_damping=True,
            pinned_endpoints=START_PINNED_FREE_END,
        )

    def test_sparse_observation_cannot_contain_interior_truth(self) -> None:
        self.assertEqual(
            {field.name for field in fields(EndpointHistoryObservation)},
            {
                "timestamp_s",
                "root_position_m",
                "root_velocity_m_s",
                "tip_position_m",
            },
        )

    def test_spatial_basis_fixes_root_and_contains_distal_mode(self) -> None:
        basis = spatial_correction_basis(
            11, 4, dtype=torch.float64, device=torch.device("cpu")
        ).numpy()
        self.assertTrue(np.array_equal(basis[0], np.zeros(4)))
        self.assertAlmostEqual(float(basis[-1, 0]), 1.0)
        self.assertTrue(np.allclose(basis[-1, 1:], 0.0, atol=1.0e-12))

    def test_clean_recursive_propagation_matches_identical_truth(self) -> None:
        truth = self.initial.cable
        observer = DderHistoryObserver(
            self.snapshot,
            self.initial.cable,
            self._observation(0.0, truth),
            HistoryObserverSettings(history_duration_s=0.04, maximum_iterations=1),
            device="cpu",
        )
        for index in range(1, 4):
            truth = self._step(truth)
            observer.ingest(self._observation(index * self.dt, truth))
        estimate = observer.current_state
        self.assertTrue(
            torch.allclose(estimate.positions_m, truth.positions_m, atol=1.0e-9)
        )
        self.assertTrue(
            torch.allclose(estimate.velocities_m_s, truth.velocities_m_s, atol=1.0e-8)
        )

    def test_endpoint_history_reduces_hidden_velocity_history_residual(self) -> None:
        hidden = perturb_cable_state(
            self.initial,
            velocity_delta_m_s=(0.0, 0.45, 0.0),
            profile="interior",
        ).cable
        observer = DderHistoryObserver(
            self.snapshot,
            self.initial.cable,
            self._observation(0.0, hidden),
            HistoryObserverSettings(
                history_duration_s=0.10,
                spatial_mode_count=2,
                maximum_iterations=1,
                prior_weight=1.0e-6,
                lm_damping=1.0e-5,
            ),
            device="cpu",
        )
        truth = hidden
        for index in range(1, 6):
            truth = self._step(truth)
            observer.ingest(self._observation(index * self.dt, truth))
        update = observer.correct()
        self.assertTrue(update.ready)
        self.assertTrue(update.accepted)
        self.assertLess(
            update.measurement_rmse_after_m,
            update.measurement_rmse_before_m,
        )

    def test_fixed_history_replay_matches_reference_python_replay(self) -> None:
        truth = self.initial.cable
        observer = DderHistoryObserver(
            self.snapshot,
            self.initial.cable,
            self._observation(0.0, truth),
            HistoryObserverSettings(
                history_duration_s=0.10,
                spatial_mode_count=2,
                maximum_iterations=1,
            ),
            device="cpu",
        )
        for index in range(1, 6):
            truth = self._step(truth)
            observer.ingest(self._observation(index * self.dt, truth))
        rng = np.random.default_rng(41)
        q_values = rng.normal(
            0.0, 0.1, size=(3, observer.correction_dimension)
        )
        history_tensors = observer._history_tensors()
        optimized = observer._replay(q_values, history_tensors)

        q_tensor = torch.as_tensor(q_values, dtype=observer.dtype)
        position_delta, velocity_delta = observer._correction(q_tensor)
        anchor = observer._states[0]
        candidate = DderState(
            anchor.positions_m.expand(3, -1, -1).clone() + position_delta,
            anchor.velocities_m_s.expand(3, -1, -1).clone() + velocity_delta,
        )
        state = observer._project_state(
            candidate,
            observer._root_positions[0],
            observer._root_velocities[0],
            batch_size=3,
        )
        positions = [state.positions_m]
        velocities = [state.velocities_m_s]
        for index in range(1, observer.frame_count):
            dt = (
                observer._observations[index].timestamp_s
                - observer._observations[index - 1].timestamp_s
            )
            state = observer._step(state, observer._root_positions[index], dt)
            positions.append(state.positions_m)
            velocities.append(state.velocities_m_s)
        reference_position = torch.stack(positions, dim=1)
        reference_velocity = torch.stack(velocities, dim=1)
        self.assertTrue(torch.equal(optimized.positions_m, reference_position))
        self.assertTrue(torch.equal(optimized.velocities_m_s, reference_velocity))
        self.assertTrue(
            np.array_equal(
                optimized.tip_positions_m,
                reference_position[:, :, -1].numpy(),
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_captured_history_replay_matches_eager_gpu_execution(self) -> None:
        device = torch.device("cuda")
        state = DderState(
            self.initial.cable.positions_m.to(device=device, dtype=torch.float32),
            self.initial.cable.velocities_m_s.to(device=device, dtype=torch.float32),
        )
        observer = DderHistoryObserver(
            self.snapshot,
            state,
            self._observation(0.0, self.initial.cable),
            HistoryObserverSettings(
                history_duration_s=0.04,
                spatial_mode_count=2,
                maximum_iterations=1,
            ),
            device=device,
        )
        truth = state
        constants = self.snapshot.model.runtime_constants(state.positions_m)
        for index in range(1, 4):
            truth = self.snapshot.model.step_runtime(
                truth,
                truth.positions_m[:, :1],
                torch.full((1,), self.dt, dtype=torch.float32, device=device),
                constants,
                iterative_damping=True,
                pinned_endpoints=START_PINNED_FREE_END,
            )
            estimate = observer.ingest(
                EndpointHistoryObservation(
                    index * self.dt,
                    truth.positions_m[0, 0].cpu().numpy(),
                    truth.velocities_m_s[0, 0].cpu().numpy(),
                    truth.positions_m[0, -1].cpu().numpy(),
                )
            )
            self.assertTrue(torch.equal(estimate.positions_m, truth.positions_m))
            self.assertTrue(
                torch.allclose(
                    estimate.velocities_m_s,
                    truth.velocities_m_s,
                    rtol=0.0,
                    atol=2.0e-10,
                )
            )
        q_values = np.zeros((3, observer.correction_dimension), dtype=np.float64)
        captured = observer._replay(q_values, observer._history_tensors())
        captured_position = captured.positions_m.clone()
        captured_velocity = captured.velocities_m_s.clone()
        operator = observer._replay_operators[(3, observer.frame_count)]
        eager_position, eager_velocity = operator._execute()
        torch.cuda.synchronize(device)
        self.assertTrue(torch.equal(captured_position, eager_position))
        self.assertTrue(torch.equal(captured_velocity, eager_velocity))


if __name__ == "__main__":
    unittest.main()
