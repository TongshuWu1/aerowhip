from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from drone_mpc.sac import SAC_CHECKPOINT_SCHEMA
from research_tools.sac_multiseed import build_summary


class DroneSacMultiseedTests(unittest.TestCase):
    def _write_run(
        self,
        directory: Path,
        seed: int,
        rate: float,
        *,
        test_seed: int = 10042,
    ) -> Path:
        path = directory / f"seed{seed}.json"
        path.with_suffix(".pt").write_bytes(f"policy-{seed}".encode())
        payload = {
            "schema": SAC_CHECKPOINT_SCHEMA,
            "source_model_sha256": "source",
            "controller_model_sha256": "controller",
            "evaluation_model_sha256": "controller",
            "source_model_provisional": True,
            "algorithm": "sac_with_symmetric_prior_replay",
            "task_distribution": "continuous_goal_conditioned_target_distribution",
            "prior_replay": {"verified_transition_count": 79},
            "settings": {"seed": seed, "batch_size": 512},
            "task": {"hit_tolerance_m": 0.05},
            "checkpoint_validation_seed": 1042,
            "final_test_seed": test_seed,
            "best_transitions": 100 + seed,
            "evaluation": {
                "episodes": 10.0,
                "success_rate": rate,
                "within_initial_reach_episodes": 6.0,
                "within_initial_reach_success_rate": rate,
                "beyond_initial_reach_episodes": 4.0,
                "beyond_initial_reach_success_rate": rate,
                "mean_minimum_error_m": 1.0 - rate,
                "mean_directional_speed_m_s": 1.0 + rate,
                "mean_hit_drone_displacement_m": 0.5,
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_summary_uses_exact_counts_and_sample_standard_deviation(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            paths = [
                self._write_run(directory, 42, 0.0),
                self._write_run(directory, 43, 0.5),
                self._write_run(directory, 44, 1.0),
            ]
            summary = build_summary(paths)
        overall = summary["aggregate"]["overall_success"]
        self.assertEqual(overall["pooled_successes"], 15)
        self.assertEqual(overall["pooled_episodes"], 30)
        self.assertAlmostEqual(overall["rate_mean"], 0.5)
        self.assertAlmostEqual(overall["rate_sample_std"], 0.5)
        self.assertEqual([run["training_seed"] for run in summary["runs"]], [42, 43, 44])

    def test_summary_rejects_different_held_out_target_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            first = self._write_run(directory, 42, 0.0)
            second = self._write_run(directory, 43, 0.5, test_seed=10043)
            with self.assertRaisesRegex(ValueError, "common target seeds"):
                build_summary([first, second])


if __name__ == "__main__":
    unittest.main()
