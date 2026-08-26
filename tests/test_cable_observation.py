from __future__ import annotations

import unittest

import numpy as np

from drone_mpc.cable_observation import (
    CableObservation,
    CausalCableStateEstimator,
    SimulatedOptiTrackSource,
)


def _observation(
    time_s: float,
    sequence: int,
    positions: np.ndarray,
    *,
    arrival_s: float | None = None,
    validity: np.ndarray | None = None,
) -> CableObservation:
    return CableObservation(
        sample_timestamp_s=time_s,
        arrival_timestamp_s=time_s if arrival_s is None else arrival_s,
        sequence_number=sequence,
        root_position_m=positions[0],
        marker_positions_m=positions[1:],
        marker_validity_mask=(
            np.ones(10, dtype=bool) if validity is None else validity
        ),
    )


class CableObservationContractTests(unittest.TestCase):
    def test_measurement_contract_has_positions_but_no_truth_velocity(self) -> None:
        observation = _observation(0.0, 0, np.zeros((11, 3)))
        self.assertFalse(hasattr(observation, "cable_velocities_m_s"))
        self.assertFalse(hasattr(observation, "truth_state"))
        self.assertEqual(observation.marker_positions_m.shape, (10, 3))

    def test_quadratic_history_recovers_newest_position_and_velocity_causally(self) -> None:
        estimator = CausalCableStateEstimator(polynomial_degree=2, history_length=7)
        latest = None
        for sequence, time_s in enumerate((0.0, 0.011, 0.021, 0.032, 0.044, 0.055, 0.067)):
            position = np.zeros((11, 3), dtype=np.float64)
            position[:, 0] = 0.4 + 0.8 * time_s + 0.5 * 1.2 * time_s**2
            latest = estimator.update(_observation(time_s, sequence, position))
        assert latest is not None
        expected_position = 0.4 + 0.8 * 0.067 + 0.5 * 1.2 * 0.067**2
        expected_velocity = 0.8 + 1.2 * 0.067
        np.testing.assert_allclose(latest.cable_positions_m[:, 0], expected_position, atol=1e-12)
        np.testing.assert_allclose(latest.cable_velocities_m_s[:, 0], expected_velocity, atol=1e-12)

    def test_past_estimate_is_immutable_when_future_sample_arrives(self) -> None:
        estimator = CausalCableStateEstimator(history_length=3)
        first = estimator.update(_observation(0.0, 0, np.zeros((11, 3))))
        frozen = first.cable_positions_m.copy()
        future = np.ones((11, 3)) * 100.0
        estimator.update(_observation(0.01, 1, future))
        np.testing.assert_array_equal(first.cable_positions_m, frozen)

    def test_dropout_is_imputed_but_not_marked_measured(self) -> None:
        estimator = CausalCableStateEstimator(history_length=3)
        for sequence, time_s in enumerate((0.0, 0.01, 0.02)):
            positions = np.zeros((11, 3))
            positions[:, 0] = time_s
            validity = np.ones(10, dtype=bool)
            if sequence == 2:
                validity[4] = False
            estimate = estimator.update(
                _observation(time_s, sequence, positions, validity=validity)
            )
        self.assertFalse(estimate.measurement_validity_mask[5])
        self.assertTrue(estimate.imputed_mask[5])
        self.assertTrue(estimate.state_validity_mask[5])

    def test_source_releases_samples_only_after_arrival(self) -> None:
        time_s = np.asarray((0.0, 0.02, 0.04))
        root = np.zeros((3, 3))
        cable = np.zeros((3, 11, 3))
        source = SimulatedOptiTrackSource(
            time_s,
            root,
            cable,
            measurement_rate_hz=100.0,
            latency_s=0.02,
        )
        self.assertEqual(source.observations_arrived_by(0.019), ())
        first = source.observations_arrived_by(0.02)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].sample_timestamp_s, 0.0)
        later = source.observations_arrived_by(0.041)
        self.assertEqual([item.sample_timestamp_s for item in later], [0.01, 0.02])


if __name__ == "__main__":
    unittest.main()

