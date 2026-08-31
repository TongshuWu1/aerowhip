from __future__ import annotations

import math

import torch

from learning.sac import terminal_critic_target
from learning.spectral_sac import (
    SpectralOneShotActor,
    alpha_explore_schedule,
    initial_frequency_std,
    inverse_radial_squash,
    orthonormal_idct_matrix,
    spectral_to_temporal,
    spectral_coefficients_to_normalized_action,
    temporal_to_spectral,
)
from learning.sac import radial_squash


def test_orthonormal_dct_is_full_rank_energy_preserving_and_invertible() -> None:
    basis = orthonormal_idct_matrix(dtype=torch.float64)
    identity = torch.eye(16, dtype=torch.float64)
    assert torch.allclose(basis.T @ basis, identity, atol=1e-12, rtol=1e-12)
    assert int(torch.linalg.matrix_rank(basis)) == 16
    temporal = torch.randn(7, 16, 3, dtype=torch.float64)
    coefficients = temporal_to_spectral(temporal)
    reconstructed = spectral_to_temporal(coefficients)
    assert torch.allclose(reconstructed, temporal, atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        coefficients.square().sum(dim=(1, 2)),
        temporal.square().sum(dim=(1, 2)),
        atol=1e-11,
        rtol=1e-11,
    )


def test_initial_spectrum_has_all_frequencies_and_declines() -> None:
    std = initial_frequency_std(dtype=torch.float64)
    assert std.shape == (16,)
    assert bool((std > 0).all())
    assert math.isclose(float(std[0]), 1.0)
    assert bool((std[1:] < std[:-1]).all())


def test_inverse_radial_and_spectral_center_reconstruct_normalized_action() -> None:
    torch.manual_seed(19)
    raw = 0.6 * torch.randn(11, 16, 3, dtype=torch.float64)
    normalized, _ = radial_squash(raw)
    recovered_raw = inverse_radial_squash(normalized)
    coefficients = temporal_to_spectral(recovered_raw)
    reconstructed = spectral_coefficients_to_normalized_action(coefficients)
    assert torch.allclose(recovered_raw, raw, atol=1e-11, rtol=1e-11)
    assert torch.allclose(
        reconstructed[:, :48].reshape(-1, 16, 3),
        normalized,
        atol=1e-11,
        rtol=1e-11,
    )


def test_spectral_actor_bounds_log_probability_and_gradients() -> None:
    torch.manual_seed(4)
    actor = SpectralOneShotActor()
    context = torch.randn(32, 83)
    sample = actor(context)
    knots = sample.normalized_action[:, :48].reshape(-1, 16, 3)
    assert sample.normalized_action.shape == (32, 49)
    assert sample.log_prob.shape == (32, 1)
    assert torch.all(sample.normalized_action[:, -1] == 1.0)
    assert bool((torch.linalg.vector_norm(knots, dim=-1) < 1.0 + 1e-6).all())
    assert bool(torch.isfinite(sample.log_prob).all())
    loss = (sample.log_prob + sample.normalized_action.square().mean(dim=1, keepdim=True)).mean()
    loss.backward()
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in actor.parameters()
    )


def test_exploration_schedule_exact_and_monotone() -> None:
    expected = {
        0: 0.25,
        74_999: 0.25,
        75_000: 0.25,
        187_500: 0.125,
        300_000: 0.0,
        400_000: 0.0,
    }
    for episodes, value in expected.items():
        assert math.isclose(alpha_explore_schedule(episodes), value, abs_tol=1e-12)
    values = [alpha_explore_schedule(k) for k in range(0, 500_001, 1000)]
    assert all(0.0 <= value <= 0.25 for value in values)
    assert all(right <= left for left, right in zip(values, values[1:]))


def test_terminal_critic_target_is_stationary_across_exploration_schedule() -> None:
    physical_reward = torch.tensor([[-1.5], [2.75]])
    targets = [
        terminal_critic_target(physical_reward) + 0.0 * alpha_explore_schedule(k)
        for k in (0, 100_000, 400_000)
    ]
    assert all(torch.equal(target, physical_reward) for target in targets)
