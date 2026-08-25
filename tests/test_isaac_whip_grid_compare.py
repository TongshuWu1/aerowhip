from __future__ import annotations

import copy
import math
import unittest

from isaac_whip.compare_grid import compare


def _report(link_count: int) -> dict[str, object]:
    return {
        "schema": "isaac_whip_physics_run_v1",
        "root_reference_schema": "isaac_whip_root_reference_v1",
        "status": "PASS",
        "finite": True,
        "mode": "excite",
        "num_envs": 1,
        "device": "cuda:0",
        "physics_dt_s": 0.002,
        "achieved_control_hz": 50.0,
        "simulated_time_s": 0.2,
        "cable": {
            "link_count": link_count,
            "source_sha256": "model-hash",
            "length_m": 0.961,
            "diameter_m": 0.0035,
            "total_mass_kg": 0.016,
            "bending_stiffness_n_m2": 1.0e-4,
            "bending_damping_n_m2_s": 1.5e-5,
        },
        "drone": {"config_sha256": "drone-hash"},
        "trace_env_0_10_hz": [
            {
                "time_s": 0.1,
                "tip_position_m": [0.0, 0.0, 0.0],
                "drone_position_m": [0.0, 0.0, 0.0],
            },
            {
                "time_s": 0.2,
                "tip_position_m": [0.0, 0.0, 0.0],
                "drone_position_m": [0.0, 0.0, 0.0],
            },
        ],
    }


class GridComparisonTests(unittest.TestCase):
    def test_metrics_use_common_time_vector_distances(self) -> None:
        left = _report(20)
        right = _report(30)
        right["trace_env_0_10_hz"][0]["tip_position_m"] = [0.003, 0.0, 0.0]
        right["trace_env_0_10_hz"][1]["tip_position_m"] = [0.004, 0.0, 0.0]
        result = compare(left, right, 0.2)
        self.assertAlmostEqual(result["tip"]["rmse_m"], math.sqrt(12.5e-6), places=12)
        self.assertAlmostEqual(result["tip"]["maximum_m"], 0.004, places=12)

    def test_mismatched_experiment_is_rejected(self) -> None:
        left = _report(20)
        right = _report(30)
        right["mode"] = "hover"
        with self.assertRaisesRegex(ValueError, "mode differs"):
            compare(left, right, 0.2)

    def test_roundoff_in_derived_physical_scalars_is_accepted(self) -> None:
        left = _report(20)
        right = _report(30)
        right["cable"]["total_mass_kg"] = left["cable"]["total_mass_kg"] - 4.0e-18

        result = compare(left, right, 0.2)

        self.assertEqual(result["sample_count"], 2)

    def test_partial_requested_horizon_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not cover"):
            compare(_report(20), _report(30), 0.3)

    def test_mismatched_trace_times_are_rejected(self) -> None:
        left = _report(20)
        right = copy.deepcopy(_report(30))
        right["trace_env_0_10_hz"][1]["time_s"] = 0.15
        with self.assertRaisesRegex(ValueError, "identical 10 Hz trace times"):
            compare(left, right, 0.2)


if __name__ == "__main__":
    unittest.main()
