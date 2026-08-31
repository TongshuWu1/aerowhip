from __future__ import annotations

import math
from dataclasses import fields
from pathlib import Path

import pytest
import torch

from learning.policy_action import decode_policy_action, encode_physical_action
from planning.cem_task import load_variable_duration_task
from planning.metrics import PopulationRolloutResult
from planning.production_cem import ProductionCemSettings, event_segment
from planning.selection import feasibility_elite_order
from planning.variable_duration import variable_duration_fullstate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = (
    PROJECT_ROOT
    / "config"
    / "tasks"
    / "canonical_whip_variable_duration_tuned_reward_v1.json"
)


def _task():
    return load_variable_duration_task(TASK_PATH)


def _selection_result(count: int) -> PopulationRolloutResult:
    boolean = {"feasible", "success", "finite"}
    vectors = {"final_uav_position_m", "final_c10_position_m"}
    payload = {}
    for item in fields(PopulationRolloutResult):
        if item.name in boolean:
            payload[item.name] = torch.zeros(count, dtype=torch.bool)
        elif item.name == "first_entry_marker":
            payload[item.name] = torch.zeros(count, dtype=torch.int64)
        elif item.name in vectors:
            payload[item.name] = torch.zeros(count, 3)
        else:
            payload[item.name] = torch.zeros(count)
    payload["finite"][:] = True
    payload["task_cost"][:] = 10.0
    payload["best_event_tip_distance_m"][:] = 1.0
    payload["best_event_direction_angle_deg"][:] = 90.0
    payload["first_entry_tip_distance_m"][:] = 1.0
    payload["first_entry_direction_angle_deg"][:] = 90.0
    return PopulationRolloutResult(**payload)


def test_production_selection_is_success_then_feasible_then_infeasible():
    result = _selection_result(4)
    result.feasible[1] = True
    result.task_cost[1] = 1.0
    result.feasibility_violation[0] = 1.0
    result.feasibility_violation[2] = 0.01
    result.success[3] = True
    result.feasible[3] = True
    result.first_entry_tip_distance_m[3] = 0.04
    result.first_entry_direction_angle_deg[3] = 20.0
    assert feasibility_elite_order(result, _task()).tolist() == [3, 1, 2, 0]


def test_terminal_settle_has_the_specified_boundary_values_and_hold_position():
    duration = 0.60
    settle = 0.30
    dt = 0.01
    knots = torch.tensor(
        [[[1.5, -0.4, 0.25]] * 16],
        dtype=torch.float64,
    )
    initial_position = torch.tensor([[0.2, -0.1, 1.5]], dtype=torch.float64)
    initial_velocity = torch.tensor([[0.3, 0.2, -0.1]], dtype=torch.float64)
    command = variable_duration_fullstate(
        knots,
        torch.tensor([duration], dtype=torch.float64),
        initial_position_m=initial_position,
        initial_velocity_m_s=initial_velocity,
        yaw_rad=torch.tensor([0.37], dtype=torch.float64),
        maximum_time_s=1.20,
        dt_s=dt,
        settle_duration_s=settle,
    )

    active_index = round(duration / dt)
    hold_index = round((duration + settle) / dt)
    a_terminal = knots[0, -1]
    v_terminal = initial_velocity[0] + duration * a_terminal
    p_terminal = initial_position[0] + duration * initial_velocity[0] + 0.5 * duration**2 * a_terminal
    p_hold = p_terminal + 0.5 * settle * v_terminal + (settle**2 / 12.0) * a_terminal

    assert torch.allclose(command.positions_m[active_index, 0], p_terminal, atol=1e-12, rtol=0)
    assert torch.allclose(command.velocities_m_s[active_index, 0], v_terminal, atol=1e-12, rtol=0)
    assert torch.allclose(command.accelerations_m_s2[active_index, 0], a_terminal, atol=1e-12, rtol=0)
    assert torch.allclose(command.positions_m[hold_index, 0], p_hold, atol=1e-12, rtol=0)
    assert torch.equal(command.velocities_m_s[hold_index, 0], torch.zeros(3, dtype=torch.float64))
    assert torch.equal(command.accelerations_m_s2[hold_index, 0], torch.zeros(3, dtype=torch.float64))
    assert torch.allclose(
        command.positions_m[hold_index:, 0],
        p_hold.expand(command.positions_m.shape[0] - hold_index, 3),
        atol=1e-12,
        rtol=0,
    )
    assert torch.allclose(torch.diff(command.times_s), torch.full((120,), dt, dtype=torch.float64))
    assert torch.isfinite(command.positions_m).all()
    assert torch.isfinite(command.velocities_m_s).all()
    assert torch.isfinite(command.accelerations_m_s2).all()


def test_normalized_complete_action_is_a_lossless_authoritative_round_trip():
    task = _task()
    settings = ProductionCemSettings()
    generator = torch.Generator().manual_seed(6042)
    knots = torch.randn((64, 16, 3), generator=generator, dtype=torch.float64)
    norms = torch.linalg.vector_norm(knots, dim=-1, keepdim=True)
    knots = 19.5 * knots / torch.clamp(norms, min=1.0e-12)
    durations = torch.linspace(settings.duration_min_s, settings.duration_max_s, 64, dtype=torch.float64)
    encoded = encode_physical_action(
        knots,
        durations,
        task,
        duration_max_s=settings.duration_max_s,
    )
    decoded = decode_policy_action(encoded, task, duration_max_s=settings.duration_max_s)
    reencoded = encode_physical_action(
        decoded.acceleration_knots_local_m_s2,
        decoded.duration_s,
        task,
        duration_max_s=settings.duration_max_s,
    )

    assert encoded.shape == (64, 49)
    assert torch.allclose(decoded.normalized_action, encoded, atol=2e-15, rtol=0)
    assert torch.allclose(reencoded, encoded, atol=2e-15, rtol=0)
    assert torch.allclose(decoded.acceleration_knots_local_m_s2, knots, atol=3e-14, rtol=0)
    assert torch.allclose(decoded.duration_s, durations, atol=2e-15, rtol=0)


def test_decoder_canonicalizes_out_of_ball_vectors_before_rollout():
    task = _task()
    action = torch.zeros((1, 49), dtype=torch.float64)
    action[0, :3] = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    decoded = decode_policy_action(action, task, duration_max_s=1.80)
    knot_norm = torch.linalg.vector_norm(decoded.normalized_action[0, :3])
    assert knot_norm.item() == pytest.approx(1.0, abs=1e-12)
    assert torch.linalg.vector_norm(decoded.acceleration_knots_local_m_s2[0, 0]).item() == pytest.approx(
        20.0, abs=1e-11
    )


def test_event_segments_and_duration_envelopes_are_distinct():
    assert event_segment(1.10, 1.20, 0.30) == "ACTIVE"
    assert event_segment(1.35, 1.20, 0.30) == "SETTLE"
    assert event_segment(1.75, 1.20, 0.30) == "HOLD"
    assert event_segment(math.nan, 1.20, 0.30) is None
    settings = ProductionCemSettings()
    assert settings.duration_min_s == 0.45
    assert settings.duration_max_s == 1.80
    assert settings.settle_duration_s == 0.30
    assert settings.evaluation_time_s == 2.40
