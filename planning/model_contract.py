"""Fail-closed verification of the selected production planning model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experimental_data.io import sha256_file
from simulator.parameters import SimulatorSettings
from simulator.production import (
    PROJECT_ROOT,
    load_active_model_manifest,
    verify_active_geometry,
    verify_active_parameters,
)

from .task import CanonicalWhipTask


def verify_planning_model_integrity(
    settings: SimulatorSettings,
    task: CanonicalWhipTask,
) -> dict[str, Any]:
    """Verify the explicit model freeze, every pinned hash, and test seal."""

    active = load_active_model_manifest()
    required_freeze = "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI"
    if active.get("status") != "MODEL_FROZEN_FOR_MPPI":
        raise RuntimeError("Active model status is not MODEL_FROZEN_FOR_MPPI.")
    if active.get("ready_for_mppi") is not True:
        raise RuntimeError("Active model is not explicitly ready_for_mppi.")
    if active.get("model_integrity") != "Verified":
        raise RuntimeError("Active model integrity is not Verified.")
    freeze_relative = Path(str(active.get("production_freeze", "")))
    if freeze_relative.name != required_freeze or task.model_freeze != required_freeze:
        raise RuntimeError("Active planning freeze does not match canonical task.")
    if active.get("protected_test_predictively_evaluated") is not False:
        raise RuntimeError("Protected-test seal is not intact.")
    verify_active_geometry(settings, active)
    verify_active_parameters(settings, active)

    freeze = (PROJECT_ROOT / freeze_relative).resolve()
    manifest_path = freeze / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_status") != "MODEL_FROZEN_FOR_MPPI":
        raise RuntimeError("Pinned production freeze is not ready for planning.")
    if manifest.get("protected_test_predictively_evaluated") is not False:
        raise RuntimeError("Pinned production freeze has lost protected-test seal.")
    verified: dict[str, str] = {}
    for relative, expected in manifest["artifact_hashes"].items():
        artifact = freeze / relative
        observed = sha256_file(artifact)
        if observed != expected:
            raise RuntimeError(f"Frozen artifact hash mismatch: {artifact}")
        verified[relative] = observed
    acceptance = json.loads((freeze / "pre_mppi_acceptance.json").read_text(encoding="utf-8"))
    if acceptance.get("pass") is not True:
        raise RuntimeError("Frozen planning acceptance artifact is not PASS.")
    frozen_active = json.loads(
        (freeze / "active_model_manifest.json").read_text(encoding="utf-8")
    )
    keys = (
        "geometry_version",
        "simulator_configuration",
        "uav_residual_freeze",
        "development_cable_fit",
        "damping_backend",
        "precision",
    )
    if any(active.get(key) != frozen_active.get(key) for key in keys):
        raise RuntimeError("Current active model differs from the frozen active snapshot.")
    return {
        "verified": True,
        "active_model": active,
        "freeze_path": str(freeze),
        "freeze_manifest_sha256": sha256_file(manifest_path),
        "verified_artifact_hashes": verified,
        "protected_test_predictively_evaluated": False,
    }
