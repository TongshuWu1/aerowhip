from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from learning.ppo_validation import append_manual_validation_result, write_training_plots
from run_ppo_manual_validation import run_manual_validation


def test_manual_validation_history_is_separate_durable_and_count_aware(
    tmp_path: Path,
) -> None:
    result = {
        "request_id": "manual-1",
        "completed_utc": "2026-09-01T00:00:00+00:00",
        "checkpoint_path": "checkpoint.pt",
        "checkpoint_episodes": 123_456,
        "validation_episodes": 20,
        "validation_successes": 20,
        "validation_success_rate": 1.0,
        "validation_endpoint_successes": 19,
        "validation_endpoint_success_rate": 0.95,
        "validation_legacy_scientific_successes": 0,
        "validation_legacy_scientific_success_rate": 0.0,
        "median_minimum_tip_distance_m": 0.03,
        "mean_maximum_uav_displacement_m": 0.8,
        "numerical_failures": 0,
        "tip_first_count": 19,
        "mean_maximum_uav_speed_m_s": 2.0,
        "mean_maximum_command_acceleration_m_s2": 10.0,
        "successful_first_entry_time_s": {"median": 1.5},
        "successful_tip_distance_m": {"median": 0.03},
        "successful_directed_speed_m_s": {"median": 6.0},
        "successful_direction_error_deg": {"median": 12.0},
    }

    append_manual_validation_result(tmp_path, result)

    with (tmp_path / "manual_validation_history.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["validation_episodes"] == "20"
    assert rows[0]["validation_successes"] == "20"
    assert rows[0]["successful_directed_speed_m_s_median"] == "6.0"
    latest = json.loads((tmp_path / "manual_validation_latest.json").read_text())
    assert latest["request_id"] == "manual-1"
    assert not (tmp_path / "validation_history.csv").exists()

    with (tmp_path / "training_log.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "episodes",
                "successes",
                "total_success_rate",
                "batch_success_rate",
                "total_endpoint_success_rate",
                "batch_endpoint_success_rate",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "episodes": 123456,
                "successes": 100000,
                "total_success_rate": 0.81,
                "batch_success_rate": 0.9,
                "total_endpoint_success_rate": 0.9,
                "batch_endpoint_success_rate": 0.95,
            }
        )
    with (tmp_path / "validation_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "checkpoint_episodes",
                "validation_success_rate",
                "validation_endpoint_success_rate",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "checkpoint_episodes": 123456,
                "validation_success_rate": 0.95,
                "validation_endpoint_success_rate": 1.0,
            }
        )
    write_training_plots(tmp_path)
    assert (tmp_path / "validation_success_vs_episodes.png").stat().st_size > 0
    assert not (tmp_path / "training_reward_vs_episodes.png").exists()


def test_manual_validation_rejects_unsupported_count_before_loading_artifacts(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match=r"\[1, 128\]"):
        run_manual_validation(tmp_path, count=129, request_id="too-many")
