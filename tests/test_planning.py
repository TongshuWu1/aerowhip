from __future__ import annotations

from pathlib import Path

import pytest
import torch

from planning.command_parameterization import (
    acceleration_knots_to_fullstate,
    command_consistency_errors,
    project_acceleration_knots,
)
from planning.metrics import feasibility_first_cost, synthetic_hard_success
from planning.model_contract import verify_planning_model_integrity
from planning.rollout import clone_state_batch, hover_preroll
from planning.task import load_canonical_whip_task
from simulator.parameters import SimulatorSettings
from simulator.production import build_production_simulator


ROOT = Path(__file__).resolve().parents[1]
TASK = load_canonical_whip_task()
SETTINGS = SimulatorSettings.load(ROOT / "config" / "default.json")


def test_canonical_task_and_model_freeze_are_explicit() -> None:
    TASK.validate_for_dt(SETTINGS.dt_s)
    integrity = verify_planning_model_integrity(SETTINGS, TASK)
    assert integrity["verified"] is True
    assert len(integrity["verified_artifact_hashes"]) == 13
    assert integrity["protected_test_predictively_evaluated"] is False


def test_fullstate_acceleration_parameterization_is_consistent() -> None:
    knots = TASK.nominal_knots(device=torch.device("cpu"), dtype=torch.float64)
    command = acceleration_knots_to_fullstate(
        knots,
        initial_position_m=torch.tensor(TASK.initial_uav_position_m),
        initial_velocity_m_s=torch.zeros(3),
        yaw_rad=TASK.initial_yaw_rad,
        horizon_s=TASK.mppi.horizon_s,
        dt_s=SETTINGS.dt_s,
    )
    assert command.positions_m.shape == (71, 1, 3)
    assert command.simulator_sequence().step_count == 70
    errors = command_consistency_errors(command)
    assert errors["maximum_velocity_consistency_error"] < 1.0e-12
    assert errors["maximum_position_consistency_error"] < 1.0e-12
    assert torch.count_nonzero(command.orientations_xyzw[..., :3]) == 0
    assert torch.all(command.orientations_xyzw[..., 3] == 1)
    assert torch.count_nonzero(command.angular_velocities_body_rad_s) == 0


def test_acceleration_projection_uses_vector_norm() -> None:
    raw = torch.tensor([[20.0, 20.0, 20.0], [-40.0, 0.0, 0.0]])
    projected = project_acceleration_knots(raw, 20.0)
    assert torch.all(torch.linalg.vector_norm(projected, dim=-1) <= 20.0 + 1.0e-6)
    assert projected[0, 0] < 20.0


def test_synthetic_hard_task_gates() -> None:
    target = torch.tensor(TASK.target_position_m)
    kwargs = dict(
        tip_position_m=target,
        tip_velocity_m_s=torch.tensor([4.5, 0.0, 0.0]),
        first_entry_marker=10,
        maximum_uav_displacement_m=0.2,
        maximum_uav_speed_m_s=2.0,
        maximum_command_acceleration_m_s2=19.0,
    )
    assert synthetic_hard_success(TASK, **kwargs)
    assert not synthetic_hard_success(
        TASK, **{**kwargs, "tip_velocity_m_s": torch.tensor([-4.5, 0.0, 0.0])}
    )
    assert not synthetic_hard_success(
        TASK, **{**kwargs, "tip_velocity_m_s": torch.tensor([3.9, 0.0, 0.0])}
    )
    assert not synthetic_hard_success(TASK, **{**kwargs, "first_entry_marker": 5})
    assert not synthetic_hard_success(
        TASK, **{**kwargs, "maximum_uav_displacement_m": 0.51}
    )


def test_feasible_near_miss_outranks_strongly_illegal_close_hit() -> None:
    # Row zero has the much better task objective but violates the physical
    # envelope. Row one is a legal near-miss and must rank first.
    ranked = feasibility_first_cost(
        torch.tensor([0.01, 12.0, 20.0], dtype=torch.float64),
        feasible=torch.tensor([False, True, True]),
        violation=torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64),
        finite=torch.tensor([True, True, True]),
        invalid_cost=1.0e9,
    )
    assert int(torch.argmin(ranked)) == 1
    assert float(ranked[0]) > float(torch.max(ranked[1:]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="production planning requires CUDA")
def test_post_hover_state_clone_includes_independent_residual_fifo() -> None:
    simulator = build_production_simulator(SETTINGS, device="cuda", dtype=torch.float32)
    state = hover_preroll(simulator, TASK)
    cloned = clone_state_batch(state, 4)
    assert cloned.uav.position_m.shape == (4, 3)
    assert cloned.cable.positions_m.shape == (4, 12, 3)
    assert cloned.uav.residual_history.features.shape == (4, 10, 9)
    assert cloned.uav.position_m.data_ptr() != state.uav.position_m.data_ptr()
    assert cloned.cable.positions_m.data_ptr() != state.cable.positions_m.data_ptr()
    assert (
        cloned.uav.residual_history.features.data_ptr()
        != state.uav.residual_history.features.data_ptr()
    )
