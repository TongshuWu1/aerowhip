from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from learning.policy_action import decode_policy_action
from planning.cem_task import load_variable_duration_task
from planning.flick_primitive import (
    DIRECTED_FLICK_PARAMETER_DIM,
    FLICK_PARAMETER_DIM,
    FlickCemSettings,
    FlickPrimitiveBounds,
    directed_flick_parameters_to_knots,
    encode_flick_as_production_action,
    flick_parameters_to_knots,
    project_flick_parameters,
    scientific_support_elite_order,
)
from planning.production_cem import ProductionCemSettings


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "config" / "tasks" / "canonical_whip_variable_duration_tuned_reward_v1.json"


def test_flick_parameter_projection_is_finite_bounded_and_periodic() -> None:
    bounds = FlickPrimitiveBounds()
    raw = torch.tensor([[3.0 * math.pi, 4.0, -2.0, 30.0, 3.0]])
    projected = project_flick_parameters(raw, bounds)
    assert projected.shape == (1, FLICK_PARAMETER_DIM)
    assert float(projected[0, 0]) == pytest.approx(-math.pi, abs=2.0e-7)
    assert float(projected[0, 1]) == pytest.approx(bounds.elevation_max_rad, abs=1.0e-7)
    assert projected[0, 2:].tolist() == pytest.approx([0.0, 20.0, 1.8], abs=1.0e-7)


def test_flick_knots_are_smooth_bounded_and_target_relative() -> None:
    parameters = torch.tensor(
        [[0.0, 0.0, 12.0, 8.0, 1.1], [math.pi / 2.0, 0.0, 12.0, 8.0, 1.1]],
        dtype=torch.float64,
    )
    knots, duration = flick_parameters_to_knots(
        parameters,
        torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64),
    )
    assert knots.shape == (2, 16, 3)
    assert duration.tolist() == [1.1, 1.1]
    assert torch.equal(knots[:, 0], torch.zeros((2, 3), dtype=torch.float64))
    assert torch.allclose(knots[:, -1], torch.zeros((2, 3), dtype=torch.float64), atol=1e-28, rtol=0)
    assert float(torch.linalg.vector_norm(knots, dim=-1).max()) <= 12.0 + 1.0e-12
    # Zero azimuth follows target +X.  A +90-degree offset follows local +Y.
    assert float(knots[0, :, 1:].abs().max()) == 0.0
    assert float(knots[1, :, (0, 2)].abs().max()) < 1.0e-12


def test_flick_uses_the_lossless_normalized_production_action_codec() -> None:
    task = load_variable_duration_task(TASK)
    settings = ProductionCemSettings(population=2048)
    parameters = torch.tensor([[0.3, -0.2, 14.0, 11.0, 1.15]], dtype=torch.float64)
    action = encode_flick_as_production_action(
        parameters,
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        task,
        settings,
    )
    decoded = decode_policy_action(action, task, duration_max_s=settings.duration_max_s)
    expected_knots, expected_duration = flick_parameters_to_knots(
        parameters, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    )
    assert action.shape == (1, 49)
    assert torch.allclose(decoded.acceleration_knots_local_m_s2, expected_knots, atol=2e-14, rtol=0)
    assert torch.allclose(decoded.duration_s, expected_duration, atol=2e-15, rtol=0)
    assert float(torch.linalg.vector_norm(decoded.acceleration_knots_local_m_s2, dim=-1).max()) <= 20.0


def test_flick_cem_contract_is_exactly_five_dimensional_and_fixed_batch() -> None:
    settings = FlickCemSettings()
    assert settings.population == 2048
    assert len(settings.initial_mean) == FLICK_PARAMETER_DIM == 5


def test_seven_parameter_repair_gives_the_reverse_pulse_an_independent_axis() -> None:
    # First pulse follows +X; second pulse follows +Y rather than being tied to -X.
    parameters = torch.tensor(
        [[0.0, 0.0, math.pi / 2.0, 0.0, 12.0, 9.0, 1.1]],
        dtype=torch.float64,
    )
    knots, duration = directed_flick_parameters_to_knots(
        parameters, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    )
    assert parameters.shape[1] == DIRECTED_FLICK_PARAMETER_DIM == 7
    assert duration.tolist() == [1.1]
    assert float(knots[0, :8, 1].abs().max()) == 0.0
    assert float(knots[0, 8:, 0].abs().max()) < 1.0e-12
    assert float(knots[0, :8, 0].max()) > 10.0
    assert float(knots[0, 8:, 1].max()) > 8.0


def test_seven_parameter_family_losslessly_contains_the_five_parameter_near_hit() -> None:
    shared = torch.tensor(
        [[-0.3777563431036288, -0.6887993678991313, 5.9648291281477235,
          11.236645159991072, 0.7490965877323066]],
        dtype=torch.float64,
    )
    directed = torch.tensor(
        [[shared[0, 0], shared[0, 1], shared[0, 0] + math.pi,
          -shared[0, 1], shared[0, 2], shared[0, 3], shared[0, 4]]],
        dtype=torch.float64,
    )
    direction = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    shared_knots, shared_duration = flick_parameters_to_knots(shared, direction)
    directed_knots, directed_duration = directed_flick_parameters_to_knots(
        directed, direction
    )
    assert torch.allclose(directed_knots, shared_knots, atol=2.0e-14, rtol=0)
    assert torch.equal(directed_duration, shared_duration)


def test_support_order_preserves_a_tip_entry_over_a_farther_legacy_cost_winner() -> None:
    task = load_variable_duration_task(TASK)
    rows = [
        {
            "finite": True, "feasible": True, "success": False,
            "first_entry_marker": 10, "first_entry_tip_distance_m": 0.026,
            "first_entry_directed_speed_m_s": 0.91,
            "first_entry_direction_angle_deg": 75.0,
            "minimum_tip_target_distance_m": 0.026,
            "best_event_direction_angle_deg": 33.0,
            "best_event_directed_speed_m_s": 2.5,
            "task_cost": 33.35, "feasibility_violation": 0.0,
        },
        {
            "finite": True, "feasible": True, "success": False,
            "first_entry_marker": 0, "first_entry_tip_distance_m": float("inf"),
            "first_entry_directed_speed_m_s": 0.0,
            "first_entry_direction_angle_deg": 180.0,
            "minimum_tip_target_distance_m": 0.36,
            "best_event_direction_angle_deg": 25.0,
            "best_event_directed_speed_m_s": 3.5,
            "task_cost": 33.15, "feasibility_violation": 0.0,
        },
    ]
    population = SimpleNamespace(
        **{
            name: torch.tensor([row[name] for row in rows])
            for name in (
                "success", "finite", "feasible", "first_entry_marker",
                "first_entry_tip_distance_m", "first_entry_directed_speed_m_s",
                "first_entry_direction_angle_deg", "minimum_tip_target_distance_m",
                "best_event_direction_angle_deg", "best_event_directed_speed_m_s",
                "task_cost", "feasibility_violation",
            )
        }
    )
    assert scientific_support_elite_order(population, task).tolist() == [0, 1]
    directed_flick_parameters_to_knots,
