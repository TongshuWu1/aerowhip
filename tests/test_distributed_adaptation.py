from __future__ import annotations

from dataclasses import replace
import math
import unittest

import numpy as np

from drone_mpc.distributed_adaptation import (
    AdaptationSegment,
    AtomicParameterStore,
    DistributedAdaptationSettings,
    DistributedObservation,
    ParameterEstimate,
    RollingDistributedBuffer,
    _residual_vector,
    curvature_binormals_numpy,
    select_informative_segments,
)


def _observation(time_s: float, *, curved: bool = False) -> DistributedObservation:
    z = np.linspace(0.0, -1.0, 11)
    positions = np.zeros((11, 3), dtype=np.float64)
    positions[:, 2] = z
    if curved:
        positions[:, 0] = 0.08 * np.sin(np.linspace(0.0, math.pi, 11))
    velocities = np.zeros_like(positions)
    velocities[:, 0] = np.linspace(0.0, 1.0, 11)
    return DistributedObservation(
        timestamp_s=time_s,
        attachment_position_m=positions[0],
        attachment_velocity_m_s=np.zeros(3),
        cable_positions_m=positions,
        cable_velocities_m_s=velocities,
        drone_state=np.zeros(6),
        executed_action_m_s2=np.zeros(3),
        active_estimate=ParameterEstimate(),
    )


class DistributedAdaptationContractTests(unittest.TestCase):
    def test_atomic_parameter_publication_rejects_stale_fit(self) -> None:
        store = AtomicParameterStore()
        first = ParameterEstimate(0.1, -0.1, 1, 1.0, 0.1, "fit")
        self.assertTrue(store.publish_if_current(first, expected_generation=0))
        stale = ParameterEstimate(0.2, -0.2, 1, 2.0, 0.05, "stale")
        self.assertFalse(store.publish_if_current(stale, expected_generation=0))
        self.assertEqual(store.snapshot(), first)

    def test_curvature_is_zero_for_a_straight_chain(self) -> None:
        positions = np.zeros((11, 3), dtype=np.float64)
        positions[:, 2] = np.linspace(0.0, -1.0, 11)
        np.testing.assert_allclose(curvature_binormals_numpy(positions), 0.0)

    def test_recent_fifo_expires_but_informative_cache_remains(self) -> None:
        settings = replace(
            DistributedAdaptationSettings(),
            buffer_duration_s=0.20,
            segment_duration_s=0.10,
            window_duration_s=0.20,
            cache_size=3,
        )
        buffer = RollingDistributedBuffer(settings)
        for index in range(11):
            buffer.append(_observation(0.05 * index, curved=index < 5))
        self.assertGreater(buffer.recent[0].timestamp_s, 0.0)
        self.assertGreaterEqual(len(buffer.cache), 1)
        self.assertLessEqual(len(buffer.cache), 3)

    def test_segment_selection_preserves_a_held_out_set(self) -> None:
        settings = replace(
            DistributedAdaptationSettings(),
            fit_segment_count=3,
            validation_segment_count=2,
            minimum_excitation=0.01,
        )
        base = _observation(0.0, curved=True)
        candidates = []
        for index in range(7):
            times = np.asarray((index * 0.2, index * 0.2 + 0.1))
            candidates.append(
                AdaptationSegment(
                    time_s=times,
                    attachment_positions_m=np.stack(
                        (base.attachment_position_m, base.attachment_position_m)
                    ),
                    cable_positions_m=np.stack(
                        (base.cable_positions_m, base.cable_positions_m)
                    ),
                    cable_velocities_m_s=np.stack(
                        (base.cable_velocities_m_s, base.cable_velocities_m_s)
                    ),
                    excitation_score=0.1 + index,
                    source="test",
                    start_time_s=float(times[0]),
                )
            )
        fit, validation = select_informative_segments(candidates, settings)
        self.assertEqual(len(fit), 3)
        self.assertEqual(len(validation), 2)
        self.assertLess(max(item.start_time_s for item in fit), min(item.start_time_s for item in validation))

    def test_imputed_marker_is_excluded_from_parameter_residual(self) -> None:
        observed = np.zeros((1, 3, 11, 3), dtype=np.float64)
        predicted = observed.copy()
        predicted[0, 1:, 5, :] = 100.0
        validity = np.ones((1, 3, 11), dtype=bool)
        validity[0, 1:, 5] = False
        residual = _residual_vector(predicted, observed, 1.0, validity)
        np.testing.assert_array_equal(residual, 0.0)


if __name__ == "__main__":
    unittest.main()
