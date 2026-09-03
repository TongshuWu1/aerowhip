from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

from run_ppo_simulation import DEFAULT_CHECKPOINT, DEFAULT_CONFIG
from run_simple_ppo import DEFAULT_CONFIG as DEFAULT_TRAINING_CONFIG, _load_config
from simulator.production import (
    PROJECT_ROOT,
    active_model_paths,
    resolve_portable_artifact_reference,
)


SELECTED = (
    PROJECT_ROOT
    / "results"
    / "ppo"
    / "policies"
    / "PPO_WHIP_FORWARD_REVERSE_RELEASE_D50_V1"
)


def test_fresh_clone_contains_runtime_model_and_selected_policy() -> None:
    paths = active_model_paths()
    assert (paths["uav_residual_freeze"] / "residual_weights.pt").is_file()
    assert (paths["development_cable_fit"] / "fitted_cable_parameters.json").is_file()
    assert (
        PROJECT_ROOT
        / "data"
        / "model_freezes"
        / "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI"
        / "manifest.json"
    ).is_file()
    assert DEFAULT_CONFIG == SELECTED / "config.json"
    assert DEFAULT_CHECKPOINT == SELECTED / "checkpoints" / "terminal.pt"
    assert DEFAULT_CONFIG.is_file()
    assert DEFAULT_CHECKPOINT.is_file()


def test_legacy_absolute_freeze_reference_relocates_inside_clone() -> None:
    relative = Path(
        "data/fit_results_decomposed/"
        "2026-08-28T200529.563705+0000_e828f6b1"
    )
    legacy = Path("Z:/retired-machine/old-project") / relative
    assert resolve_portable_artifact_reference(legacy) == (PROJECT_ROOT / relative).resolve()


def test_portable_training_config_has_only_tracked_relative_dependencies() -> None:
    config = _load_config(DEFAULT_TRAINING_CONFIG)
    assert config["experiment_id"] == "whip_ppo_portable_continuation_v1"
    dependencies = (
        config["simulator_config"],
        config["task_config"],
        config["context_normalizer"],
        config["initialization"]["policy_checkpoint"],
        config["training_initial_states"]["training_bank"],
        config["training_initial_states"]["training_bank_manifest"],
        config["training_initial_states"]["validation_bank"],
        config["training_initial_states"]["validation_bank_manifest"],
    )
    for dependency in dependencies:
        path = Path(str(dependency))
        assert not path.is_absolute()
        assert (PROJECT_ROOT / path).is_file(), dependency


def test_committed_pycharm_run_configurations_are_portable() -> None:
    expected = {
        "01 Workstation Preflight.run.xml": "verify_workstation.py",
        "02 Simulator UI.run.xml": "run_simulator.py",
        "03 PPO Replay - Selected D50.run.xml": "run_ppo_simulation.py",
        "04 PPO Training Preflight.run.xml": "run_simple_ppo.py",
        "05 PPO Training - Portable Continuation.run.xml": "run_simple_ppo.py",
        "06 Regression Tests.run.xml": "run_tests.py",
    }
    for filename, script in expected.items():
        source = PROJECT_ROOT / ".run" / filename
        root = ET.parse(source).getroot()
        configuration = root.find("configuration")
        assert configuration is not None
        options = {
            item.attrib["name"]: item.attrib.get("value", "")
            for item in configuration.findall("option")
        }
        assert options["WORKING_DIRECTORY"] == "$PROJECT_DIR$"
        assert options["SCRIPT_NAME"] == f"$PROJECT_DIR$/{script}"
        assert options["IS_MODULE_SDK"] == "true"


def test_requirements_cannot_replace_cuda_pytorch_wheel() -> None:
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    package_names = {
        line.split("==", 1)[0].strip().casefold()
        for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "torch" not in package_names


def test_selected_freeze_manifest_checkpoint_matches_portable_file() -> None:
    manifest = json.loads((SELECTED / "freeze_manifest.json").read_text(encoding="utf-8"))
    assert manifest["selected_checkpoint"] == "checkpoints/terminal.pt"
    assert (SELECTED / manifest["selected_checkpoint"]).is_file()
