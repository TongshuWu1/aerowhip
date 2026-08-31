"""Regression guards for the diagnosed DDIM terminal-cliff defect."""

from __future__ import annotations

import torch

from learning.action_diffusion import cosine_alpha_bar_schedule, ddim_timestep_schedule


def test_default_ddim_schedule_excludes_cosine_terminal_cliff() -> None:
    alpha = cosine_alpha_bar_schedule(100)
    schedule = ddim_timestep_schedule(100, 25)
    assert schedule.tolist() == [
        95, 91, 87, 83, 79, 75, 71, 67, 63, 59, 55, 51, 48,
        44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 4, 0,
    ]
    assert float(alpha[99]) < 1.0e-6
    assert float(alpha[schedule[0]]) > 1.0e-3
    assert float(torch.rsqrt(alpha[99])) > 2_000.0
    assert float(torch.rsqrt(alpha[schedule[0]])) < 20.0


def test_explicit_terminal_timestep_remains_available_for_diagnostics() -> None:
    schedule = ddim_timestep_schedule(100, 25, maximum_timestep=99)
    assert schedule[0] == 99
    assert schedule[-1] == 0
    assert torch.unique(schedule).numel() == 25
