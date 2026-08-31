from __future__ import annotations

import torch

from learning.figure8_sac_env import (
    Figure8Reference,
    LegacyFigure8RewardWeights,
    figure8_episode_success,
    figure8_interval_reward,
)


def test_figure8_reference_is_closed_and_has_forward_tangent() -> None:
    reference = Figure8Reference((0.1, -0.2, 0.4), 0.7, 0.5)
    progress = torch.tensor([0.0, 1.0])
    points = reference.point(progress)
    assert torch.allclose(points[0], points[1], atol=1.0e-6)
    tangent = reference.tangent(torch.tensor([0.0]))
    assert tangent.shape == (1, 3)
    assert torch.allclose(torch.linalg.vector_norm(tangent, dim=-1), torch.ones(1))


def test_legacy_figure8_reward_rewards_progress_and_penalizes_tracking_error() -> None:
    zeros = torch.zeros(1)
    weights = LegacyFigure8RewardWeights()
    progressing = figure8_interval_reward(
        zeros,
        torch.tensor([0.10]),
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        torch.tensor([False]),
        acceleration_scale_m_s2=10.0,
        weights=weights,
    )
    inaccurate = figure8_interval_reward(
        torch.tensor([0.01]),
        torch.tensor([0.10]),
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        torch.tensor([False]),
        acceleration_scale_m_s2=10.0,
        weights=weights,
    )
    assert float(progressing) == 0.5
    assert float(inaccurate) < float(progressing)


def test_figure8_reported_success_is_one_safe_accurate_cycle() -> None:
    assert bool(
        figure8_episode_success(
            torch.tensor([1.1]),
            torch.tensor([0.08]),
            torch.tensor([2.5]),
            torch.tensor([0.03]),
            torch.tensor([False]),
        )[0]
    )
    assert not bool(
        figure8_episode_success(
            torch.tensor([0.9]),
            torch.tensor([0.08]),
            torch.tensor([2.5]),
            torch.tensor([0.03]),
            torch.tensor([False]),
        )[0]
    )
