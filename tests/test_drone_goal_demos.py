from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from drone_mpc.mpc import MpcProblem
from drone_mpc.oracle import (
    ReachabilityResult,
    ReachabilitySettings,
    save_reachability_result,
)
from drone_mpc.simulator import SimulationResult
from research_tools.goal_demos import (
    _restart_seed,
    _select_best_feasible,
)


class DroneGoalDemoTests(unittest.TestCase):
    def test_restart_seeds_are_stable_and_best_feasible_uses_oracle_rank(self) -> None:
        self.assertEqual(_restart_seed(42, 1, 0), 42)
        self.assertEqual(_restart_seed(42, 2, 0), 10_042)
        self.assertEqual(_restart_seed(42, 2, 3), 10_045)

        settings = ReachabilitySettings(candidates=16, elite_count=2, iterations=1)
        attempts = [
            (0, settings, self._result(feasible=False, energy_j=10.0)),
            (1, settings, self._result(feasible=True, energy_j=0.5)),
            (2, settings, self._result(feasible=True, energy_j=1.0)),
        ]
        selected = _select_best_feasible(attempts)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected[0], 2)

    def test_strict_save_preserves_restart_provenance_and_refuses_overwrite(self) -> None:
        result = self._result(feasible=True, energy_j=1.0)
        settings = ReachabilitySettings(candidates=16, elite_count=2, iterations=1)
        problem = MpcProblem(
            target_position_m=(0.8, 0.0, -0.1),
            impact_direction=(1.0, 0.0, 0.0),
        )
        provenance = {"requested_restarts": 4, "selected_restart_index": 2}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "demo.npz"
            save_reachability_result(
                output,
                result,
                settings,
                problem,
                provenance=provenance,
                overwrite=False,
            )
            original = output.read_bytes()
            with np.load(output, allow_pickle=False) as archive:
                saved = json.loads(str(archive["provenance_json"].item()))
            self.assertEqual(saved, provenance)
            with self.assertRaises(FileExistsError):
                save_reachability_result(
                    output,
                    result,
                    settings,
                    problem,
                    overwrite=False,
                )
            self.assertEqual(output.read_bytes(), original)

    @staticmethod
    def _result(*, feasible: bool, energy_j: float) -> ReachabilityResult:
        time_s = np.asarray((0.0, 0.02), dtype=np.float64)
        vectors = np.zeros((2, 3), dtype=np.float64)
        cable = np.zeros((2, 2, 3), dtype=np.float64)
        prediction = SimulationResult(
            time_s=time_s,
            drone_positions_m=vectors.copy(),
            drone_velocities_m_s=vectors.copy(),
            attachment_positions_m=vectors.copy(),
            cable_positions_m=cable.copy(),
            cable_velocities_m_s=cable.copy(),
            accelerations_m_s2=np.zeros((1, 3), dtype=np.float64),
            target_position_m=np.asarray((0.8, 0.0, -0.1), dtype=np.float64),
            impact_direction=np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
            model_sha256="model",
        )
        terms = {
            "feasible": float(feasible),
            "safety_violation": 0.0 if feasible else 1.0,
            "hit_violation": 0.0,
            "constraint_violation": 0.0 if feasible else 1.0,
            "negative_directional_tip_energy_j": -energy_j,
            "directional_tip_energy_j": energy_j,
            "impact_time": 0.1,
            "impact_time_s": 0.4,
            "regularization": 0.2,
            "position_error_m": 0.01,
            "directional_speed_m_s": 2.0,
            "maximum_drone_speed_m_s": 1.0,
        }
        return ReachabilityResult(
            basis_coefficients_m=np.zeros((6, 3), dtype=np.float64),
            controls_m_s2=np.zeros((40, 3), dtype=np.float64),
            prediction=prediction,
            terms=terms,
            impact_frame=1,
            search_history=(1.0,),
            refinement_history=(0.5,),
            source_model_sha256="model",
            search_model_sha256="search",
            source_node_count=15,
            search_node_count=7,
            source_model_provisional=True,
        )


if __name__ == "__main__":
    unittest.main()
