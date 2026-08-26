from __future__ import annotations

import unittest

import numpy as np

from drone_mpc.receding_mppi_gui import warm_start_compatibility_issues


class WarmStartCompatibilityTests(unittest.TestCase):
    def _metadata(self) -> dict[str, object]:
        return {
            "model_sha256": "model-a",
            "problem": {
                "target_position_m": [1.0, 0.0, 1.4],
                "impact_direction": [1.0, 0.0, 0.0],
                "minimum_impact_speed_m_s": 3.5,
                "maximum_tip_error_m": 0.05,
                "maximum_impact_angle_deg": 35.0,
            },
            "simulation_settings": {
                "horizon_s": 0.7,
                "control_interval_s": 0.02,
                "maximum_acceleration_m_s2": 15.0,
            },
        }

    def _issues(self, metadata: dict[str, object]) -> tuple[str, ...]:
        return warm_start_compatibility_issues(
            metadata,
            source_model_sha256="model-a",
            target_position_m=np.asarray((1.0, 0.0, 1.4)),
            impact_direction=np.asarray((1.0, 0.0, 0.0)),
            minimum_impact_speed_m_s=3.5,
            target_radius_m=0.05,
            maximum_impact_angle_deg=35.0,
            horizon_s=0.7,
            control_interval_s=0.02,
            maximum_acceleration_m_s2=15.0,
        )

    def test_matching_sidecar_is_compatible(self) -> None:
        self.assertEqual(self._issues(self._metadata()), ())

    def test_task_model_and_timing_differences_are_reported(self) -> None:
        metadata = self._metadata()
        metadata["model_sha256"] = "model-b"
        metadata["problem"]["target_position_m"] = [0.8, 0.2, 1.3]  # type: ignore[index]
        metadata["problem"]["minimum_impact_speed_m_s"] = 5.0  # type: ignore[index]
        metadata["simulation_settings"]["horizon_s"] = 2.0  # type: ignore[index]
        issues = self._issues(metadata)
        self.assertIn("source model differs", issues)
        self.assertIn("target differs", issues)
        self.assertIn("speed requirement differs", issues)
        self.assertIn("horizon differs", issues)

    def test_missing_sidecar_is_explicitly_unverified(self) -> None:
        self.assertEqual(self._issues({}), ("no JSON sidecar provenance",))


if __name__ == "__main__":
    unittest.main()
