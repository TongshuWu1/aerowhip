from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from learning.amortized_policy import AmortizedTrajectoryActor, imitation_losses
from planning.cem_task import load_variable_duration_task
from planning.teacher_cem import TeacherCemSettings
from planning.variable_duration import variable_duration_fullstate


ROOT = Path(__file__).resolve().parents[1]
TASK = load_variable_duration_task(
    ROOT / "config" / "tasks" / "canonical_whip_variable_duration_tuned_reward_v1.json"
)


def test_amortized_actor_is_one_context_to_one_complete_maneuver() -> None:
    actor = AmortizedTrajectoryActor()
    output = actor(torch.zeros((7, 83), dtype=torch.float32))
    assert output.shape == (7, 49)
    assert bool((output.abs() <= 1.0).all())
    target = torch.zeros_like(output)
    loss, parts = imitation_losses(output, target)
    assert bool(torch.isfinite(loss))
    assert set(parts) == {"acceleration_mse", "duration_mse"}


def test_post_maneuver_command_smoothly_settles_then_holds() -> None:
    duration = torch.tensor([0.60], dtype=torch.float32)
    knots = torch.zeros((1, 16, 3), dtype=torch.float32)
    knots[:, :, 0] = 2.0
    command = variable_duration_fullstate(
        knots,
        duration,
        initial_position_m=torch.tensor([0.0, 0.0, 1.0]),
        initial_velocity_m_s=torch.zeros(3),
        yaw_rad=0.0,
        maximum_time_s=1.50,
        dt_s=0.01,
    )
    hold = command.times_s >= duration[0] + 0.30 - 1.0e-6
    positions = command.positions_m[hold, 0]
    assert torch.allclose(positions, positions[0].expand_as(positions), atol=2.0e-6)
    assert torch.allclose(command.velocities_m_s[hold, 0], torch.zeros_like(positions), atol=2.0e-6)
    assert torch.allclose(command.accelerations_m_s2[hold, 0], torch.zeros_like(positions), atol=2.0e-6)
    expected_hold_x = (
        0.5 * 2.0 * 0.60**2
        + 0.5 * 0.30 * (2.0 * 0.60)
        + (0.30**2 / 12.0) * 2.0
    )
    assert float(positions[0, 0]) == pytest.approx(expected_hold_x, abs=2.0e-6)


def test_teacher_contract_keeps_maneuver_and_evaluation_time_separate() -> None:
    settings = TeacherCemSettings()
    assert settings.duration_max_s == 1.20
    assert settings.evaluation_time_s == 1.50
    config = json.loads(
        (ROOT / "config" / "learning" / "amortized_cem_nominal_pilot_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["time_contract"]["maneuver_duration_max_s"] == 1.20
    assert config["time_contract"]["evaluation_time_s"] == 1.50
    assert config["artifact_policy"]["sac_used"] is False
    assert config["artifact_policy"]["cem_used_online"] is False
