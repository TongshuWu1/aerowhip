from __future__ import annotations

import unittest

import numpy as np
import torch

from drone_mpc.figure8_tracking import endpoint_only_observation_state
from drone_mpc.hidden_state_disturbance import (
    RANDOM_SMOOTH_SEED,
    apply_hidden_velocity_disturbance,
    basis_representability,
    primary_hidden_velocity_disturbances,
    reference_velocity_rms_m_s,
)
from drone_mpc.history_observer import spatial_correction_basis
from drone_mpc.history_observer import (
    DderHistoryObserver,
    EndpointHistoryObservation,
    HistoryObserverSettings,
)
from drone_mpc.model import load_cable_model
from drone_mpc.reduced import build_controller_and_truth_models
from drone_mpc.simulator import SimulationSettings, WhipSimulator
from optitrack_offline.config import DEFAULT_MODEL_PATH


class HiddenStateDisturbanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = load_cable_model(DEFAULT_MODEL_PATH)
        simulation = SimulationSettings(
            horizon_s=0.10,
            simulation_dt_s=0.02,
            control_interval_s=0.02,
        )
        snapshot, _ = build_controller_and_truth_models(
            source,
            simulation_dt_s=0.02,
            node_count=11,
            truth_bending_stiffness_scale=1.0,
            truth_bending_damping_scale=1.0,
        )
        cls.snapshot = snapshot
        cls.initial = WhipSimulator(snapshot, simulation, device="cpu").initial_state(
            (0.0, 0.0, 1.45)
        )
        cls.disturbances = {
            item.name: item for item in primary_hidden_velocity_disturbances()
        }

    def test_all_hidden_modes_are_exactly_zero_at_root_and_tip(self) -> None:
        for disturbance in self.disturbances.values():
            delta = disturbance.normalized_velocity_m_s(11)
            self.assertTrue(np.array_equal(delta[0], np.zeros(3)))
            self.assertTrue(np.array_equal(delta[-1], np.zeros(3)))

    def test_rms_normalization_is_consistent(self) -> None:
        expected = reference_velocity_rms_m_s(11)
        for name, disturbance in self.disturbances.items():
            if name == "clean":
                continue
            delta = disturbance.normalized_velocity_m_s(11)
            actual = float(
                np.sqrt(np.mean(np.sum(np.square(delta[1:]), axis=1)))
            )
            self.assertAlmostEqual(actual, expected, places=13)

    def test_disturbance_changes_only_plant_cable_velocity(self) -> None:
        prior_positions = self.initial.cable.positions_m.clone()
        prior_velocities = self.initial.cable.velocities_m_s.clone()
        disturbed = apply_hidden_velocity_disturbance(
            self.initial, self.disturbances["mixed_span"]
        )
        self.assertTrue(torch.equal(disturbed.drone_position_m, self.initial.drone_position_m))
        self.assertTrue(torch.equal(disturbed.drone_velocity_m_s, self.initial.drone_velocity_m_s))
        self.assertTrue(torch.equal(disturbed.cable.positions_m, prior_positions))
        self.assertTrue(torch.equal(self.initial.cable.positions_m, prior_positions))
        self.assertTrue(torch.equal(self.initial.cable.velocities_m_s, prior_velocities))
        self.assertFalse(torch.equal(disturbed.cable.velocities_m_s, prior_velocities))
        self.assertTrue(
            torch.equal(disturbed.cable.velocities_m_s[:, 0], prior_velocities[:, 0])
        )
        self.assertTrue(
            torch.equal(disturbed.cable.velocities_m_s[:, -1], prior_velocities[:, -1])
        )

    def test_endpoint_observation_does_not_receive_disturbed_interior(self) -> None:
        disturbed = apply_hidden_velocity_disturbance(
            self.initial, self.disturbances["sin1"]
        )
        sanitized = endpoint_only_observation_state(self.initial, disturbed)
        self.assertTrue(
            torch.equal(
                sanitized.cable.velocities_m_s[:, 1:-1],
                self.initial.cable.velocities_m_s[:, 1:-1],
            )
        )
        self.assertTrue(
            torch.equal(
                sanitized.cable.velocities_m_s[:, -1],
                disturbed.cable.velocities_m_s[:, -1],
            )
        )

    def test_history_observer_prior_is_not_modified_by_hidden_injection(self) -> None:
        disturbed = apply_hidden_velocity_disturbance(
            self.initial, self.disturbances["sin1"]
        )
        observation = EndpointHistoryObservation(
            0.0,
            disturbed.cable.positions_m[0, 0].numpy(),
            disturbed.cable.velocities_m_s[0, 0].numpy(),
            disturbed.cable.positions_m[0, -1].numpy(),
        )
        observer = DderHistoryObserver(
            self.snapshot,
            self.initial.cable,
            observation,
            HistoryObserverSettings(history_duration_s=0.04),
            device="cpu",
        )
        self.assertTrue(
            torch.equal(
                observer.current_state.velocities_m_s,
                self.initial.cable.velocities_m_s,
            )
        )
        self.assertFalse(
            torch.equal(
                observer.current_state.velocities_m_s[:, 1:-1],
                disturbed.cable.velocities_m_s[:, 1:-1],
            )
        )

    def test_sparse_history_observation_is_identical_for_hidden_interior_states(self) -> None:
        disturbed = apply_hidden_velocity_disturbance(
            self.initial, self.disturbances["mixed_span"]
        )

        def sparse(state):
            return EndpointHistoryObservation(
                0.0,
                state.cable.positions_m[0, 0].numpy(),
                state.cable.velocities_m_s[0, 0].numpy(),
                state.cable.positions_m[0, -1].numpy(),
            )

        prior_observation = sparse(self.initial)
        disturbed_observation = sparse(disturbed)
        self.assertEqual(
            prior_observation.timestamp_s, disturbed_observation.timestamp_s
        )
        self.assertTrue(
            np.array_equal(
                prior_observation.root_position_m,
                disturbed_observation.root_position_m,
            )
        )
        self.assertTrue(
            np.array_equal(
                prior_observation.root_velocity_m_s,
                disturbed_observation.root_velocity_m_s,
            )
        )
        self.assertTrue(
            np.array_equal(
                prior_observation.tip_position_m,
                disturbed_observation.tip_position_m,
            )
        )

    def test_full_state_reference_retains_true_disturbed_interior(self) -> None:
        disturbed = apply_hidden_velocity_disturbance(
            self.initial, self.disturbances["sin3"]
        )
        full_state_observation = disturbed
        self.assertTrue(
            torch.equal(
                full_state_observation.cable.velocities_m_s[:, 1:-1],
                disturbed.cable.velocities_m_s[:, 1:-1],
            )
        )
        self.assertFalse(
            torch.equal(
                full_state_observation.cable.velocities_m_s[:, 1:-1],
                self.initial.cable.velocities_m_s[:, 1:-1],
            )
        )

    def test_basis_residual_separates_in_span_and_out_of_span_modes(self) -> None:
        basis = spatial_correction_basis(
            11, 4, dtype=torch.float64, device=torch.device("cpu")
        ).numpy()
        for name in ("sin1", "sin2", "sin3"):
            result = basis_representability(self.disturbances[name], basis)
            self.assertLess(float(result["relative_residual"]), 1.0e-12)
        for name in ("sin4", "sin5"):
            result = basis_representability(self.disturbances[name], basis)
            self.assertGreater(float(result["relative_residual"]), 0.5)

    def test_seeded_random_mixture_is_reproducible(self) -> None:
        first = primary_hidden_velocity_disturbances(
            random_seed=RANDOM_SMOOTH_SEED
        )[-1]
        second = primary_hidden_velocity_disturbances(
            random_seed=RANDOM_SMOOTH_SEED
        )[-1]
        different = primary_hidden_velocity_disturbances(
            random_seed=RANDOM_SMOOTH_SEED + 1
        )[-1]
        self.assertEqual(first.coefficients, second.coefficients)
        self.assertNotEqual(first.coefficients, different.coefficients)


if __name__ == "__main__":
    unittest.main()
