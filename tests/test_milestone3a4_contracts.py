from __future__ import annotations

from dataclasses import replace

import numpy as np

from experimental_data.contracts import audit_take_contract
from fitting.dataset import Dataset, load_dataset


TAKES = ("osc_001", "fig8_001", "fig8_002", "fig8_003")


def test_motive_manual_trim_is_exact_scientific_timeline() -> None:
    for take_id in TAKES:
        audit = audit_take_contract(take_id)
        assert audit["timeline_contract"] == "PASS"
        assert audit["exact_time_array_match"] is True
        assert audit["exact_source_time_array_match"] is True
        assert audit["exact_frame_array_match"] is True
        assert audit["maximum_time_difference_s"] == 0.0
        assert audit["logger_pre_history_available_s"] > 0.0
        assert audit["logger_post_history_excluded_s"] > 0.0


def test_untouched_role_is_not_a_fitting_or_normalization_role() -> None:
    dataset = load_dataset()
    protected = replace(dataset.takes[0], role="untouched_test", enabled=True)
    training = replace(dataset.takes[1], role="training", enabled=True)
    validation = replace(dataset.takes[2], role="validation", enabled=True)
    isolated = Dataset((protected, training, validation), dataset.manifest)
    assert isolated.untouched_test == (protected,)
    assert isolated.training == (training,)
    assert isolated.validation == (validation,)
    assert protected not in isolated.fitting_takes


def test_command_contract_metadata_matches_observed_current_takes() -> None:
    for take in load_dataset().takes:
        semantics = take.metadata["command_semantics"]
        assert semantics["externally_commanded"] == [
            "position_world_m",
            "velocity_world_mps",
            "acceleration_world_mps2",
            "yaw_rad",
        ]
        assert semantics["externally_not_commanded"][:2] == ["roll", "pitch"]
        assert semantics["angular_velocity_current_data"] == (
            "identically_zero_not_independently_excited"
        )
        valid = take.arrays["command_valid"]
        assert np.max(np.abs(take.arrays["command_angular_velocity"][valid])) == 0.0
