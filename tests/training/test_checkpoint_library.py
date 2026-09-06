from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from learning import POINT_FORCE_OBSERVATION_DIM, SimplePPOAgent
from learning.checkpoint_library import (
    delete_training_run,
    list_policy_checkpoints,
    rename_training_run,
    save_current_policy_snapshot,
    write_active_run,
)


def _make_run(project: Path) -> Path:
    artifact = project / "runs" / "ppo" / "run_a"
    checkpoints = artifact / "checkpoints"
    checkpoints.mkdir(parents=True)
    (artifact / "status.json").write_text(
        json.dumps({"status": "COMPLETED", "episodes": 64}), encoding="utf-8"
    )
    (artifact / "task.json").write_text(
        json.dumps(
            {
                "success": {
                    "minimum_directed_tip_speed_m_s": 4.0,
                    "maximum_tip_velocity_to_desired_direction_error_deg": 45.0,
                }
            }
        ),
        encoding="utf-8",
    )
    agent = SimplePPOAgent(
        POINT_FORCE_OBSERVATION_DIM,
        3,
        device=torch.device("cpu"),
        hidden_dim=16,
    )
    torch.save(
        {
            **agent.checkpoint(),
            "observation_dim": POINT_FORCE_OBSERVATION_DIM,
            "action_dim": 3,
            "episodes": 64,
        },
        checkpoints / "terminal.pt",
    )
    write_active_run(project, artifact)
    return artifact


def test_checkpoint_library_lists_renames_and_deletes_runs(tmp_path) -> None:
    artifact = _make_run(tmp_path)
    entries = list_policy_checkpoints(tmp_path)
    assert len(entries) == 1
    assert entries[0]["episodes"] == 64
    assert entries[0]["active"] is True
    assert entries[0]["compatible"] is True
    rename_training_run(artifact, "Professor demo policy")
    entries = list_policy_checkpoints(tmp_path)
    assert entries[0]["run_name"] == "Professor demo policy"
    delete_training_run(tmp_path, artifact)
    assert list_policy_checkpoints(tmp_path) == []
    assert not artifact.exists()


def test_checkpoint_library_marks_old_hit_contract_incompatible(tmp_path) -> None:
    artifact = _make_run(tmp_path)
    (artifact / "task.json").write_text(
        json.dumps(
            {
                "success": {
                    "minimum_directed_hit_speed_m_s": 4.0,
                    "maximum_cable_extended_strike_direction_error_deg": 45.0,
                }
            }
        ),
        encoding="utf-8",
    )
    entry = list_policy_checkpoints(tmp_path)[0]
    assert entry["compatible"] is False
    assert "INCOMPATIBLE" in entry["label"]


def test_current_policy_snapshot_pins_exact_published_checkpoint(tmp_path) -> None:
    artifact = _make_run(tmp_path)
    latest = artifact / "checkpoints" / "latest.pt"
    terminal = artifact / "checkpoints" / "terminal.pt"
    checkpoint = torch.load(terminal, map_location="cpu", weights_only=False)
    torch.save(checkpoint, latest)
    (artifact / "status.json").write_text(
        json.dumps({"status": "RUNNING", "episodes": 999}), encoding="utf-8"
    )

    snapshot = save_current_policy_snapshot(tmp_path, artifact)
    second_snapshot = save_current_policy_snapshot(tmp_path, artifact)
    assert snapshot.name.startswith("manual_")
    assert second_snapshot != snapshot
    assert os.path.samefile(snapshot, latest)
    assert os.path.samefile(second_snapshot, latest)

    replacement = dict(checkpoint)
    replacement["episodes"] = 128
    temporary = latest.with_suffix(".tmp")
    torch.save(replacement, temporary)
    os.replace(temporary, latest)
    assert not os.path.samefile(snapshot, latest)
    saved = torch.load(snapshot, map_location="cpu", weights_only=False)
    assert saved["episodes"] == 64

    manual = next(
        entry
        for entry in list_policy_checkpoints(tmp_path)
        if entry["path"] == str(snapshot)
    )
    assert manual["kind"] == "Manual snapshot"
    assert manual["episodes"] == 64


def test_current_policy_snapshot_rejects_stopped_or_external_runs(tmp_path) -> None:
    artifact = _make_run(tmp_path)
    with pytest.raises(RuntimeError, match="only while PPO is running"):
        save_current_policy_snapshot(tmp_path, artifact)

    outside = tmp_path / "external"
    outside.mkdir()
    with pytest.raises(ValueError, match="outside the PPO run library"):
        save_current_policy_snapshot(tmp_path, outside)


def test_current_policy_snapshot_rejects_stopping_or_unpublished_policy(tmp_path) -> None:
    artifact = _make_run(tmp_path)
    (artifact / "status.json").write_text(
        json.dumps({"status": "RUNNING", "episodes": 0}), encoding="utf-8"
    )
    (artifact / "STOP_REQUESTED").touch()
    with pytest.raises(RuntimeError, match="while PPO is stopping"):
        save_current_policy_snapshot(tmp_path, artifact)
    (artifact / "STOP_REQUESTED").unlink()

    with pytest.raises(FileNotFoundError, match="first PPO batch completes"):
        save_current_policy_snapshot(tmp_path, artifact)
    assert list((artifact / "checkpoints").glob(".manual_*.tmp")) == []
