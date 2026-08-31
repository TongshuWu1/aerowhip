from __future__ import annotations

import torch

from learning.cem_residual_policy import DeterministicCemResidualPolicy
from learning.trajectory_primitive import (
    CompleteActionCodec,
    FixedCubicSplineActionCodec,
    canonicalize_normalized_action,
    open_uniform_bspline_basis,
)


def test_open_uniform_spline_basis_has_endpoint_and_partition_contract():
    for controls in (4, 6, 8):
        basis = open_uniform_bspline_basis(16, controls)
        assert basis.shape == (16, controls)
        assert torch.allclose(basis.sum(dim=1), torch.ones(16, dtype=basis.dtype), atol=1e-12)
        assert torch.equal(basis[0], torch.nn.functional.one_hot(torch.tensor(0), controls).double())
        assert torch.equal(
            basis[-1], torch.nn.functional.one_hot(torch.tensor(controls - 1), controls).double()
        )


def test_spline_codec_preserves_duration_and_production_bounds():
    generator = torch.Generator().manual_seed(712)
    actions = torch.randn((9, 49), generator=generator, dtype=torch.float64)
    actions[:, -1] = torch.linspace(-1.0, 1.0, 9)
    codec = FixedCubicSplineActionCodec(6)
    latent = codec.encode(actions)
    reconstructed = codec.decode(latent)
    assert latent.shape == (9, 19)
    assert reconstructed.shape == (9, 49)
    assert torch.equal(reconstructed[:, -1], canonicalize_normalized_action(actions)[:, -1])
    knot_norms = torch.linalg.vector_norm(reconstructed[:, :48].reshape(-1, 16, 3), dim=-1)
    assert bool((knot_norms <= 1.0 + 1e-12).all())
    assert torch.isfinite(reconstructed).all()


def test_complete_codec_and_zero_initialized_residual_policy_contract():
    action = canonicalize_normalized_action(torch.linspace(-1.2, 1.2, 49))
    codec = CompleteActionCodec()
    assert torch.equal(codec.decode(codec.encode(action)), action)
    policy = DeterministicCemResidualPolicy(codec.latent_dimension, hidden_dimension=32)
    context = torch.randn((3, 83))
    center = action.float().repeat(3, 1)
    assert torch.equal(policy(context, center), center)
