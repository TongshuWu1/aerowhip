from __future__ import annotations

import math

import torch

from learning.constrained_sac import (
    ConstrainedReplayBatch,
    ConstrainedSacAgent,
    LocalSpectralActor,
    local_frequency_base,
)
from learning.spectral_sac import spectral_coefficients_to_normalized_action


def test_local_actor_reconstructs_center_and_keeps_inactive_modes_fixed() -> None:
    torch.manual_seed(6)
    center = 0.2 * torch.randn(3, 16)
    actor = LocalSpectralActor(center)
    context = torch.randn(17, 83)
    deterministic = actor(context, deterministic=True)
    expected = spectral_coefficients_to_normalized_action(center[None]).expand(17, -1)
    assert torch.allclose(deterministic.deterministic_mean_action, expected, atol=1e-7)
    stochastic = actor(context)
    assert stochastic.normalized_action.shape == (17, 49)
    assert stochastic.log_prob.shape == (17, 1)
    assert torch.allclose(stochastic.spectral_coefficients[:, :, 4:], center[None, :, 4:])
    assert torch.all(stochastic.normalized_action[:, -1] == 1.0)


def test_local_standard_deviation_stays_in_audited_bounds() -> None:
    actor = LocalSpectralActor(torch.zeros(3, 16))
    _, initial = actor.statistics(torch.randn(8, 83))
    base = local_frequency_base().repeat(3)
    assert torch.allclose(initial, 0.05 * base[None], atol=1e-7)
    with torch.no_grad():
        actor.std_head.bias.fill_(100.0)
    _, maximum = actor.statistics(torch.randn(8, 83))
    assert bool((maximum <= 0.10 * base[None] + 1e-7).all())
    with torch.no_grad():
        actor.std_head.bias.fill_(-100.0)
    _, minimum = actor.statistics(torch.randn(8, 83))
    assert bool((minimum >= 0.005 * base[None] - 1e-7).all())


def test_local_action_log_probability_and_backward_are_finite() -> None:
    torch.manual_seed(7)
    actor = LocalSpectralActor(0.1 * torch.randn(3, 16))
    sample = actor(torch.randn(32, 83))
    knots = sample.normalized_action[:, :48].reshape(-1, 16, 3)
    assert bool(torch.isfinite(sample.log_prob).all())
    assert bool((torch.linalg.vector_norm(knots, dim=-1) < 1.0 + 1e-6).all())
    loss = sample.log_prob.mean() + sample.normalized_action.square().mean()
    loss.backward()
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in actor.parameters()
    )


def test_constrained_terminal_sac_update_is_finite_and_lambda_nonnegative() -> None:
    torch.manual_seed(8)
    agent = ConstrainedSacAgent.create(
        torch.zeros(3, 16), device="cpu", target_entropy=-20.0
    )
    batch = ConstrainedReplayBatch(
        context=torch.randn(64, 83),
        action=torch.cat((0.3 * torch.randn(64, 48), torch.ones(64, 1)), dim=1),
        reward=torch.randn(64, 1),
        safety_cost=torch.randint(0, 2, (64, 1)).float(),
    )
    metrics = agent.update(batch)
    assert all(math.isfinite(value) for value in metrics.values())
    assert float(agent.lambda_safe.detach()) >= 0.0
    assert metrics["reward_critic1_loss"] >= 0.0
    assert metrics["safety_critic1_bce"] >= 0.0
