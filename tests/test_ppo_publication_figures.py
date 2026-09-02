from __future__ import annotations

import csv
import json
from pathlib import Path

from learning.ppo_publication import export_ppo_publication_figures


def _write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_publication_export_writes_vector_raster_captions_and_provenance(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "ppo_run"
    artifact.mkdir()
    _write_csv(
        artifact / "training_log.csv",
        [
            {
                "episodes": 2048,
                "successes": 1024,
                "batch_success_rate": 0.5,
                "total_success_rate": 0.5,
                "mean_episode_reward": 10.0,
                "mean_maximum_uav_displacement_m": 0.8,
                "entropy": 4.0,
                "approximate_kl": 0.01,
                "clip_fraction": 0.1,
            },
            {
                "episodes": 4096,
                "successes": 1536,
                "batch_success_rate": 0.75,
                "total_success_rate": 0.625,
                "mean_episode_reward": 20.0,
                "mean_maximum_uav_displacement_m": 0.7,
                "entropy": 3.9,
                "approximate_kl": 0.015,
                "clip_fraction": 0.08,
            },
        ],
    )
    _write_csv(
        artifact / "validation_history.csv",
        [
            {
                "checkpoint_episodes": 0,
                "validation_episodes": 64,
                "validation_success_rate": 0.5,
                "mean_maximum_uav_displacement_m": 0.75,
            },
            {
                "checkpoint_episodes": 4096,
                "validation_episodes": 64,
                "validation_success_rate": 0.75,
                "mean_maximum_uav_displacement_m": 0.65,
            },
        ],
    )
    summary = {
        "checkpoint_episodes": 4096,
        "successful_tip_distance_m": {
            "minimum": 0.01,
            "median": 0.03,
            "maximum": 0.049,
        },
        "successful_directed_speed_m_s": {
            "minimum": 4.1,
            "median": 5.0,
            "maximum": 7.0,
        },
        "successful_direction_error_deg": {
            "minimum": 4.0,
            "median": 12.0,
            "maximum": 25.0,
        },
        "successful_first_entry_time_s": {
            "minimum": 1.0,
            "median": 1.5,
            "maximum": 2.0,
        },
    }
    (artifact / "validation_latest.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    (artifact / "status.json").write_text(
        json.dumps({"status": "COMPLETE"}), encoding="utf-8"
    )
    (artifact / "config.json").write_text(
        json.dumps(
            {
                "initialization": {
                    "source_mean_maximum_uav_displacement_m": 0.9
                }
            }
        ),
        encoding="utf-8",
    )

    manifest = export_ppo_publication_figures(artifact)

    output = artifact / "publication_figures"
    assert manifest["figure_status"] == "FINAL"
    assert manifest["latest_training_episodes"] == 4096
    assert manifest["evaluation_scope"]["target"] == "canonical only"
    assert manifest["task_success_contract"]["uav_limits_are_binary_success_gates"] is False
    assert len(manifest["figures"]) == 4
    for figure in manifest["figures"]:
        assert {Path(name).suffix for name in figure["files"]} == {
            ".png",
            ".pdf",
            ".svg",
        }
        assert all((output / name).stat().st_size > 0 for name in figure["files"])
    captions = (output / "FIGURE_CAPTIONS.md").read_text(encoding="utf-8")
    assert "Evaluation scope" in captions
    assert "Lower excursion is preferable" in captions
    persisted = json.loads((output / "figure_manifest.json").read_text())
    assert persisted["source_sha256"]["training_log.csv"]
