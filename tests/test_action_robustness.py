from __future__ import annotations

import numpy as np
import torch

from learning.action_robustness import build_action_perturbation_bank
from learning.policy_action import canonicalize_normalized_action


def test_action_perturbation_bank_is_deterministic_bounded_and_complete():
    teacher = torch.linspace(-0.8, 0.8, 49)
    arguments = dict(
        context_id="C_JOINT_017",
        acceleration_sigmas=(0.001, 0.01),
        duration_offsets_ms=(1.0, 5.0),
        samples_per_level=8,
        base_seed=42,
    )
    first, labels = build_action_perturbation_bank(teacher, **arguments)
    second, second_labels = build_action_perturbation_bank(teacher, **arguments)
    assert np.array_equal(first, second)
    assert labels == second_labels
    assert first.shape == (1 + 2 * 2 * 8 + 2 * 8, 49)
    assert np.allclose(first[0], canonicalize_normalized_action(teacher).numpy(), atol=1e-7)
    assert np.isfinite(first).all()
    assert np.max(np.abs(first)) <= 1.0
    norms = np.linalg.norm(first[:, :48].reshape(-1, 16, 3), axis=-1)
    assert np.max(norms) <= 1.0 + 1e-6


def test_duration_only_rows_do_not_change_acceleration_coordinates():
    teacher = np.zeros(49, dtype=np.float32)
    actions, labels = build_action_perturbation_bank(
        teacher,
        context_id="duration",
        acceleration_sigmas=(0.01,),
        duration_offsets_ms=(2.0,),
        samples_per_level=8,
        base_seed=7,
    )
    duration_indices = [index for index, row in enumerate(labels) if row["family"] == "DURATION_ONLY"]
    assert np.array_equal(actions[duration_indices, :48], np.zeros((8, 48), dtype=np.float32))
    offsets = actions[duration_indices, 48]
    assert set(np.sign(offsets).tolist()) == {-1.0, 1.0}
