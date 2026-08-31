from __future__ import annotations

import torch

from learning.iterative_residual import (
    BINARY_OUTCOME_DIM,
    CONTINUOUS_OUTCOME_DIM,
    OBSERVED_OUTCOME_DIM,
    IterativeResidualOutcomeModel,
    OutcomeNormalizer,
    deterministic_compact_candidates,
    fit_compact_residual_basis,
    outcome_arrays_from_rows,
    predicted_candidate_score,
)


def test_iterative_residual_model_has_the_production_context_and_action_contract() -> None:
    model = IterativeResidualOutcomeModel(hidden_dimension=64)
    continuous, binary = model(
        torch.randn(7, 83),
        torch.rand(7, 49) * 2.0 - 1.0,
        torch.randn(7, OBSERVED_OUTCOME_DIM),
        torch.randn(7, 49) * 0.01,
    )
    assert continuous.shape == (7, CONTINUOUS_OUTCOME_DIM)
    assert binary.shape == (7, BINARY_OUTCOME_DIM)
    loss = continuous.square().mean() + binary.square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_compact_residual_basis_and_candidates_are_deterministic_and_bounded() -> None:
    generator = torch.Generator().manual_seed(7)
    initial = torch.randn((40, 49), generator=generator) * 0.1
    teacher = initial + torch.randn((40, 49), generator=generator) * 0.01
    basis = fit_compact_residual_basis(initial, teacher, rank=13)
    assert basis.shape == (13, 49)
    assert torch.allclose(basis @ basis.T, torch.eye(13), atol=1.0e-5, rtol=0.0)

    first, first_delta = deterministic_compact_candidates(
        initial[0],
        basis,
        candidate_count=128,
        coordinate_rms_scales=(0.01, 0.005),
        seed=11,
    )
    second, second_delta = deterministic_compact_candidates(
        initial[0],
        basis,
        candidate_count=128,
        coordinate_rms_scales=(0.01, 0.005),
        seed=11,
    )
    assert torch.equal(first, second)
    assert torch.equal(first_delta, second_delta)
    assert torch.equal(first[0], initial[0])
    assert float(first.abs().max()) <= 1.0
    knot_norms = torch.linalg.vector_norm(first[:, :48].reshape(-1, 16, 3), dim=-1)
    assert float(knot_norms.max()) <= 1.0 + 1.0e-6


def test_scientific_gate_score_prefers_a_predicted_pass() -> None:
    normalizer = OutcomeNormalizer(torch.zeros(6), torch.ones(6))
    # Transform-space values: log distance, speed, cosine, displacement,
    # UAV speed, command acceleration.
    transformed = torch.tensor(
        [
            [-2.0, 2.0, 0.0, torch.log1p(torch.tensor(0.8)), torch.log1p(torch.tensor(4.0)), torch.log1p(torch.tensor(20.0))],
            [-4.0, 5.0, 0.95, torch.log1p(torch.tensor(0.3)), torch.log1p(torch.tensor(2.0)), torch.log1p(torch.tensor(10.0))],
        ]
    )
    logits = torch.tensor([[-3.0, -3.0, -3.0], [3.0, 3.0, 3.0]])
    score, diagnostics = predicted_candidate_score(transformed, logits, normalizer)
    assert int(torch.argmax(score)) == 1
    assert float(diagnostics["success_probability"][1]) > 0.9


def test_no_entry_marker_is_a_valid_failed_observation() -> None:
    continuous, binary = outcome_arrays_from_rows(
        [[{
            "best_event_tip_distance_m": 0.2,
            "best_event_directed_speed_m_s": 1.0,
            "best_event_direction_angle_deg": 90.0,
            "maximum_uav_displacement_m": 0.1,
            "maximum_uav_speed_m_s": 1.0,
            "maximum_command_acceleration_m_s2": 10.0,
            "first_entry_marker": None,
            "feasible": True,
            "success": False,
        }]]
    )
    assert continuous.shape == (1, 6)
    assert binary.tolist() == [[0.0, 1.0, 0.0]]
