"""Run a user-requested current-checkpoint PPO validation in simulation only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback
from typing import Any

import torch

from learning.ppo_validation import (
    FixedMildStateValidationPanel,
    append_manual_validation_result,
    write_training_plots,
)
from run_simple_ppo import _build_agent, _load_config


FIXED_NUMERICAL_BATCH_SIZE = 2048


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_manual_validation(
    artifact: Path,
    *,
    count: int,
    request_id: str,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    artifact = artifact.resolve()
    config_path = artifact / "config.json"
    checkpoint_path = (
        artifact / "checkpoints" / "latest.pt"
        if checkpoint_path is None
        else checkpoint_path.resolve()
    )
    if not 1 <= count <= 128:
        raise ValueError("Manual validation count must be in [1, 128].")
    config = _load_config(config_path)
    panel = FixedMildStateValidationPanel(
        config,
        count_override=count,
        fixed_evaluation_batch_size=FIXED_NUMERICAL_BATCH_SIZE,
    )
    device = panel.environment.simulator.device
    agent = _build_agent(config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent.policy.load_state_dict(checkpoint["policy"])
    agent.value.load_state_dict(checkpoint["value"])
    checkpoint_episodes = int(checkpoint.get("episodes", 0))
    result = panel.evaluate(agent, checkpoint_episodes=checkpoint_episodes)
    result.update(
        {
            "schema": "ppo_manual_validation_v1",
            "request_id": request_id,
            "completed_utc": _utc_now(),
            "checkpoint_path": str(checkpoint_path),
            "fixed_uav_residual_evaluation_batch_size": FIXED_NUMERICAL_BATCH_SIZE,
            "new_training": False,
            "protected_test_evaluated": False,
            "hardware_executed": False,
        }
    )
    append_manual_validation_result(artifact, result)
    write_training_plots(artifact)
    _write_json(
        artifact / "manual_validation_runs" / f"{request_id}.json",
        {"result": result, "state_manifest": panel.manifest},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-directory", required=True, type=Path)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--checkpoint", type=Path)
    arguments = parser.parse_args()
    artifact = arguments.artifact_directory.resolve()
    status_path = artifact / "manual_validation_status.json"
    _write_json(
        status_path,
        {
            "status": "RUNNING",
            "request_id": arguments.request_id,
            "validation_episodes": arguments.count,
            "started_utc": _utc_now(),
            "process_id": os.getpid(),
        },
    )
    try:
        result = run_manual_validation(
            artifact,
            count=arguments.count,
            request_id=arguments.request_id,
            checkpoint_path=arguments.checkpoint,
        )
        _write_json(status_path, {"status": "COMPLETE", **result})
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as error:
        _write_json(
            status_path,
            {
                "status": "FAILED",
                "request_id": arguments.request_id,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
