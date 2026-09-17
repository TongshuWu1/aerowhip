from __future__ import annotations

import json
from pathlib import Path
from copy import deepcopy

import torch

from run_simulation import simulate_constant_force


ROOT = Path(__file__).resolve().parents[2]


def test_default_runner_executes_point_force_hover() -> None:
    model = json.loads((ROOT / "config/model.json").read_text(encoding="utf-8"))
    task = json.loads((ROOT / "config/task.json").read_text(encoding="utf-8"))
    arrays, summary = simulate_constant_force(
        model, task, duration_s=0.05, device=torch.device("cpu")
    )
    assert arrays["cable_node_position_world_m"].shape == (6, 12, 3)
    assert arrays["commanded_force_world_n"].shape == (5, 3)
    assert summary["finite"] is True
    assert summary["steps"] == 5
    assert summary["maximum_segment_error_m"] < 1.0e-10
    # Drag operator splitting permits sub-micrometre drift at this short horizon.
    torch.testing.assert_close(torch.tensor(summary['final_root_position_world_m'],dtype=torch.float64),
                               torch.tensor(task['initial_root_position_m'],dtype=torch.float64),atol=1e-6,rtol=0.)
