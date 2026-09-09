from __future__ import annotations

import json
from pathlib import Path

import torch

from simulator.cable import (
    START_PINNED_FREE_END,
    CableConfiguration,
    DderModel,
    hanging_cable_state,
)


ROOT = Path(__file__).resolve().parents[2]


def _model() -> tuple[dict[str, object], CableConfiguration, DderModel]:
    payload = json.loads((ROOT / "config/model.json").read_text(encoding="utf-8"))
    cable = CableConfiguration.from_mapping(payload["cable"])
    parameters = cable.dder_parameters(
        EI=float(payload["cable"]["EI_n_m2"]),
        Cb=float(payload["cable"]["Cb_n_m2_s"]),
    )
    return payload, cable, DderModel(parameters)


def test_model_contract_uses_measured_point_and_cable_mass() -> None:
    payload, cable, _ = _model()
    assert payload["point_mass"]["mass_kg"] == 0.157
    assert payload["point_force_controller"]["dimensions"] == 3
    assert payload["point_force_controller"]["application_node"] == 0
    assert payload["cable"]["attachment"] == "dynamic_shared_node_0_free_pivot"
    assert cable.node_count == 12
    assert cable.length_m == 0.9525
    assert abs(cable.total_dynamic_mass_kg - 0.018) < 1.0e-12
    # Calibration coefficients are editable/versioned; they are not topology constants.
    assert payload["cable"]["EI_n_m2"] > 0
    assert payload["cable"]["Cb_n_m2_s"] >= 0


def test_retained_dder_cable_step_is_finite_and_constrained() -> None:
    payload, cable, model = _model()
    root = torch.tensor([[0.0, 0.0, 1.5]], dtype=torch.float64)
    state = hanging_cable_state(model, root, cable)
    for _ in range(10):
        state = model.step(
            state,
            root[:, None],
            float(payload["simulation"]["dt_s"]),
            pinned_endpoints=START_PINNED_FREE_END,
        )
    assert torch.isfinite(state.positions_m).all()
    assert torch.isfinite(state.velocities_m_s).all()
    assert float(model.maximum_segment_error_m(state.positions_m).max()) < 1.0e-8
