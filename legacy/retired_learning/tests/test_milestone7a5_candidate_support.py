from __future__ import annotations

import numpy as np

from run_milestone7a5 import PREFIX_COUNTS, _chunks_for_count, _prefix_summary


def _synthetic_outcomes() -> dict[str, np.ndarray]:
    shape = (3, 1024)
    values: dict[str, np.ndarray] = {
        "success": np.zeros(shape, dtype=np.bool_),
        "feasible": np.ones(shape, dtype=np.bool_),
        "finite": np.ones(shape, dtype=np.bool_),
        "first_entry_marker": np.full(shape, 10, dtype=np.int16),
        "best_event_tip_distance_m": np.full(shape, 0.04, dtype=np.float32),
        "best_event_directed_speed_m_s": np.full(shape, 4.2, dtype=np.float32),
        "best_event_direction_angle_deg": np.full(shape, 20.0, dtype=np.float32),
        "first_entry_tip_distance_m": np.full(shape, 0.04, dtype=np.float32),
        "first_entry_directed_speed_m_s": np.full(shape, 4.2, dtype=np.float32),
        "first_entry_direction_angle_deg": np.full(shape, 20.0, dtype=np.float32),
        "maximum_uav_displacement_m": np.full(shape, 0.4, dtype=np.float32),
        "maximum_uav_speed_m_s": np.full(shape, 2.0, dtype=np.float32),
        "maximum_command_acceleration_m_s2": np.full(shape, 10.0, dtype=np.float32),
        "task_cost": np.zeros(shape, dtype=np.float32),
        "maneuver_duration_s": np.ones(shape, dtype=np.float32),
        "first_entry_time_s": np.ones(shape, dtype=np.float32),
        "hit_segment": np.ones(shape, dtype=np.int8),
    }
    values["success"][0, 0] = True
    values["success"][1, 100] = True
    values["success"][2, 900] = True
    return values


def test_nested_chunk_plan_is_an_exact_partition_of_every_prefix() -> None:
    for count in PREFIX_COUNTS:
        chunks = _chunks_for_count(count)
        assert sum(chunks) == count
        assert all(value > 0 for value in chunks)
        assert chunks == (32, 32, 64, 128, 256, 512)[: len(chunks)]


def test_oracle_support_is_monotonic_for_literal_prefixes() -> None:
    outcomes = _synthetic_outcomes()
    values = [_prefix_summary(outcomes, count)["oracle_success_rate"] for count in PREFIX_COUNTS]
    assert values == sorted(values)
    assert values[0] == 1.0 / 3.0
    assert values[2] == 2.0 / 3.0
    assert values[-1] == 1.0
