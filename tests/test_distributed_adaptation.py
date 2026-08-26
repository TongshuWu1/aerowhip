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
    OnlineAdaptationMonitor,
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


class _OffsetPredictor:
    """Deterministic monitor-only predictor with configurable model error."""

    def __init__(self, offset_m: float) -> None:
        self.offset_m = offset_m

    def predict(self, segments, eta):
        hypothesis_count = len(eta)
        positions = np.stack(
            [segment.cable_positions_m for segment in segments], axis=0
        )[:, None]
        positions = np.repeat(positions, hypothesis_count, axis=1)
        positions[:, :, :, 1:, 0] += self.offset_m
        velocities = np.stack(
            [segment.cable_velocities_m_s for segment in segments], axis=0
        )[:, None]
        velocities = np.repeat(velocities, hypothesis_count, axis=1)
        return positions, velocities, 0.0


def _monitor_settings(**changes) -> DistributedAdaptationSettings:
    values = {
        "buffer_duration_s": 1.0,
        "health_horizon_s": 0.10,
        "health_evaluation_interval_s": 0.05,
        "health_ema_alpha": 0.0,
        "error_high_m2": 1.0e-6,
        "error_low_m2": 1.0e-8,
        "persistence_s": 0.10,
        "cooldown_s": 0.30,
        "minimum_excitation": 1.0e-6,
        "window_duration_s": 0.20,
        "segment_duration_s": 0.10,
    }
    values.update(changes)
    return replace(DistributedAdaptationSettings(), **values)


def _append_monitor_range(
    monitor: OnlineAdaptationMonitor, start: float, stop: float
):
    diagnostics = []
    count = int(round((stop - start) / 0.05))
    for index in range(count + 1):
        diagnostic = monitor.append(
            _observation(round(start + 0.05 * index, 10), curved=True)
        )
        if diagnostic is not None:
            diagnostics.append(diagnostic)
    return diagnostics


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

        cache_before = buffer.cache
        buffer.start_new_recording()
        self.assertEqual(buffer.recent, ())
        self.assertEqual(buffer.cache, cache_before)
        # A new strike may restart from another state but must continue on a
        # monotonic session clock and must not form a cross-strike segment.
        buffer.append(_observation(1.0, curved=True))
        self.assertEqual(len(buffer.recent), 1)
        self.assertEqual(buffer.cache, cache_before)

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

    def test_rejected_fit_rearms_after_cooldown_without_low_error_crossing(self) -> None:
        monitor = OnlineAdaptationMonitor(_OffsetPredictor(0.01), _monitor_settings())
        first = _append_monitor_range(monitor, 0.0, 0.20)
        self.assertEqual(first[-1].reason, "fit_candidate")
        self.assertTrue(monitor.claim_trigger(first[-1]))
        cached = monitor.buffer.cache

        monitor.complete_fit(published=False)
        monitor.start_new_recording()
        second = _append_monitor_range(monitor, 0.25, 0.50)

        self.assertEqual(monitor.buffer.cache[: len(cached)], cached)
        self.assertIn("cooldown", [item.reason for item in second])
        self.assertEqual(second[-1].reason, "fit_candidate")
        self.assertGreater(second[-1].ema_error_m2, monitor.settings.error_high_m2)

    def test_published_partial_update_rebaselines_then_allows_next_strike_fit(self) -> None:
        monitor = OnlineAdaptationMonitor(_OffsetPredictor(0.01), _monitor_settings())
        first = _append_monitor_range(monitor, 0.0, 0.20)
        self.assertTrue(monitor.claim_trigger(first[-1]))
        cached = monitor.buffer.cache

        monitor.complete_fit(published=True)
        self.assertEqual(monitor.ema_error_m2, 0.0)
        monitor.start_new_recording()
        second = _append_monitor_range(monitor, 0.25, 0.50)

        self.assertEqual(monitor.buffer.cache[: len(cached)], cached)
        self.assertEqual(second[0].reason, "persistence_pending")
        self.assertIn("cooldown", [item.reason for item in second])
        self.assertEqual(second[-1].reason, "fit_candidate")

    def test_monitor_reports_healthy_persistence_and_excitation_states(self) -> None:
        healthy = OnlineAdaptationMonitor(_OffsetPredictor(0.0), _monitor_settings())
        healthy_diagnostics = _append_monitor_range(healthy, 0.0, 0.10)
        self.assertEqual(healthy_diagnostics[-1].reason, "healthy")

        pending = OnlineAdaptationMonitor(_OffsetPredictor(0.01), _monitor_settings())
        pending_diagnostics = _append_monitor_range(pending, 0.0, 0.10)
        self.assertEqual(pending_diagnostics[-1].reason, "persistence_pending")

        blocked = OnlineAdaptationMonitor(
            _OffsetPredictor(0.01),
            _monitor_settings(minimum_excitation=1.0e12),
        )
        blocked_diagnostics = _append_monitor_range(blocked, 0.0, 0.20)
        self.assertEqual(blocked_diagnostics[-1].reason, "insufficient_excitation")


if __name__ == "__main__":
    unittest.main()
