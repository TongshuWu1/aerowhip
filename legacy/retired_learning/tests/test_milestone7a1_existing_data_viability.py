"""Focused guards for the existing-data-only Milestone-7A.1 runner."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_milestone7a1.py"
CONFIG = ROOT / "config/learning/amortized_cem_diffusion_existing_data_viability_v1.json"


def test_viability_configuration_prohibits_new_experiments() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["schema"] == "amortized_cem_diffusion_existing_data_viability_v1"
    assert payload["model_freeze"] == "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI"
    assert payload["prohibitions"] == {
        "new_cem_solves": True,
        "sac": True,
        "aggregation": True,
        "architecture_change": True,
        "theta_randomization": True,
        "final_test": True,
        "protected_test": "fig8vertical_002",
        "hardware": True,
    }


def test_runner_has_no_cem_optimizer_call() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "optimize_production_cem" not in called
    assert "run_teacher_generation" not in called


def test_runner_uses_final_models_without_alternate_architecture() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "ConditionalActionDiffusion" in source
    assert "ManeuverOutcomeScorer" in source
    assert "PCAController" not in source
    assert "LatentDiffusion" not in source
    assert "fixed_noise_bank.npy" in source


def test_runner_keeps_final_test_out_of_evaluation_records() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '"final_7a_test_used": False' in source
    assert '"seven_a_test_state_ids_used_for_gradients": 0' in source
    assert "test_evaluation" not in source
