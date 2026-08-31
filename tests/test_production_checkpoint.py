from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from fitting.production_status import (
    get_active_model_freeze,
    get_active_model_summary,
    get_dataset_role_summary,
)
from simulator.parameters import SimulatorSettings
from simulator.production import (
    ACTIVE_MODEL_MANIFEST,
    build_production_simulator,
    load_active_model_manifest,
    verify_active_geometry,
    verify_active_parameters,
)
from simulator.uav.state import FullStateCommand


ROOT = Path(__file__).resolve().parents[1]
SETTINGS = SimulatorSettings.load(ROOT / "config" / "default.json")


def test_active_model_is_explicit_and_frozen_for_mppi() -> None:
    manifest = load_active_model_manifest()
    assert ACTIVE_MODEL_MANIFEST == ROOT / "config" / "active_model.json"
    assert manifest["selection_policy"] == "explicit_manifest_only_never_newest_timestamp"
    assert manifest["status"] == "MODEL_FROZEN_FOR_MPPI"
    assert manifest["ready_for_mppi"] is True
    assert manifest["production_freeze"].endswith(
        "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI"
    )
    assert manifest["development_cable_fit_geometry_version"] == manifest["geometry_version"]
    assert manifest["protected_take_id"] == "fig8vertical_002"
    assert manifest["protected_test_predictively_evaluated"] is False
    assert manifest["damping_backend"] == "pcg32_experimental"
    verify_active_geometry(SETTINGS, manifest)
    verify_active_parameters(SETTINGS, manifest)


def test_status_api_exposes_current_topology_and_sealed_take() -> None:
    active = get_active_model_summary(SETTINGS)
    assert active["node_count"] == 12
    assert active["edge_count"] == 11
    assert active["cable_length_m"] == pytest.approx(0.9525)
    assert active["marker_node_mapping"] == list(range(2, 12))
    assert active["residual_active"] is True
    rows = {row["take_id"]: row for row in get_dataset_role_summary()}
    protected = rows["fig8vertical_002"]
    assert protected["role"] == "Protected Test"
    assert protected["cable_status"] == "PROTECTED — NOT EVALUATED"
    freeze = get_active_model_freeze()
    assert freeze["ready_for_mppi"] is True
    assert freeze["model_integrity"] == "Verified"
    assert freeze["protected_test_status"].endswith("PROTECTED / NOT EVALUATED")


def test_identification_page_contains_no_superseded_controls() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QLabel
    from simulator.gui.fit_widget import FitValidateWidget

    application = QApplication.instance() or QApplication([])
    widget = FitValidateWidget(SETTINGS)
    assert "MODEL_FROZEN_FOR_MPPI" in widget.production_status.text()
    assert "12-node" in widget.active_model_label.text()
    assert "0.9525 m" in widget.active_model_label.text()
    assert "100 ms causal" in widget.residual_label.text()
    assert "VALIDATED PRODUCTION VALUES" in widget.cable_label.text()
    assert "PASS / READY FOR MPPI" in widget.end_to_end_label.text()
    assert "Model integrity: Verified" in widget.freeze_label.text()
    assert "fig8vertical_002" in widget.freeze_label.text()
    page_text = " ".join(label.text() for label in widget.findChildren(QLabel))
    for forbidden in (
        "Physics Baseline",
        "Fixed Delay Ablation",
        "multiple-shooting",
        "21-node production",
        "Run Protected",
    ):
        assert forbidden not in page_text
    widget.close()
    application.processEvents()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="production backend requires CUDA")
def test_current_pr_12node_dder_executes_short_finite_rollout() -> None:
    simulator = build_production_simulator(SETTINGS)
    assert simulator.device.type == "cuda"
    assert simulator.dtype == torch.float32
    assert simulator.uav_model.residual_enabled is True
    simulator.reset(torch.tensor(SETTINGS.initial_uav_position_m, device=simulator.device))
    zeros = torch.zeros((1, 3), device=simulator.device, dtype=simulator.dtype)
    command = FullStateCommand(
        torch.tensor([SETTINGS.initial_uav_position_m], device=simulator.device),
        zeros,
        zeros,
        torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=simulator.device),
        zeros,
    )
    for _ in range(3):
        simulator.step(command, create_graph=False)
    torch.cuda.synchronize()
    assert simulator.last_dder_execution == "production_cuda_fused"
    assert bool(torch.isfinite(simulator.state.cable.positions_m).all())
    assert simulator.state.cable.positions_m.shape == (1, 12, 3)
