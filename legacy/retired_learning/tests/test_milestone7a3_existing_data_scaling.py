from __future__ import annotations

import ast
import json
from pathlib import Path

from learning.action_diffusion import ddim_timestep_schedule
import run_milestone7a3 as milestone


ROOT = Path(__file__).resolve().parents[1]


def test_milestone7a3_keeps_repaired_final_diffusion_contract() -> None:
    config = json.loads(
        (ROOT / "config/learning/diffusion_existing_data_scaling_v1.json").read_text(
            encoding="utf-8"
        )
    )
    milestone._validate_config(config)
    assert ddim_timestep_schedule().tolist()[0] == 95
    assert ddim_timestep_schedule().tolist()[-1] == 0
    assert len(ddim_timestep_schedule()) == 25


def test_milestone7a3_runner_has_no_optimizer_or_scorer_training_call() -> None:
    tree = ast.parse((ROOT / "run_milestone7a3.py").read_text(encoding="utf-8"))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "optimize_production_cem" not in called
    assert "run_production_cem" not in called
    assert "train_scorer" not in called


def test_milestone7a3_prohibitions_are_locked() -> None:
    config = json.loads(
        (ROOT / "config/learning/diffusion_existing_data_scaling_v1.json").read_text(
            encoding="utf-8"
        )
    )
    required = config["prohibitions"]
    assert required["new_cem_solves"]
    assert required["scorer_training"]
    assert required["final_test"]
    assert required["protected_test"] == "fig8vertical_002"
    assert required["hardware"]
