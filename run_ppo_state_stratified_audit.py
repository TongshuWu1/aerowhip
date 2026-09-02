"""Audit one frozen PPO checkpoint across canonical and state-distance strata."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from learning.ppo_validation import FixedMildStateValidationPanel
from run_simple_ppo import _build_agent, _load_config
from run_simple_sac import _build_environment


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _summary(values: torch.Tensor, mask: torch.Tensor) -> dict[str, float | int | None]:
    selected = values[mask & torch.isfinite(values)].detach().cpu().double().numpy()
    if selected.size == 0:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "count": int(selected.size),
        "mean": float(np.mean(selected)),
        "median": float(np.median(selected)),
        "minimum": float(np.min(selected)),
        "maximum": float(np.max(selected)),
    }


def _group(environment: Any, mask: torch.Tensor) -> dict[str, Any]:
    count = int(mask.sum().cpu())
    success = environment.episode_success & mask
    entered = (environment.episode_first_entry_marker > 0) & mask
    return {
        "contexts": count,
        "successes": int(success.sum().cpu()),
        "success_rate": float(success.sum().cpu()) / max(count, 1),
        "endpoint_entries": int((environment.episode_endpoint_success & mask).sum().cpu()),
        "tip_first_entries": int(((environment.episode_first_entry_marker == 10) & mask).sum().cpu()),
        "numerical_failures": int((environment.failed & mask).sum().cpu()),
        "minimum_tip_distance_m": _summary(environment.episode_minimum_tip_distance, mask),
        "successful_tip_distance_m": _summary(environment.episode_success_tip_distance, success),
        "first_entry_directed_speed_m_s": _summary(
            environment.episode_first_entry_directed_speed, entered
        ),
        "successful_directed_speed_m_s": _summary(
            environment.episode_success_directed_speed, success
        ),
        "first_entry_direction_error_deg": _summary(
            environment.episode_first_entry_direction_error_deg, entered
        ),
        "successful_direction_error_deg": _summary(
            environment.episode_success_direction_error_deg, success
        ),
        "successful_strike_time_s": _summary(
            environment.episode_first_entry_time_s, success
        ),
        "maximum_uav_displacement_m": _summary(
            environment.episode_maximum_displacement, mask
        ),
        "terminal_uav_displacement_m": _summary(
            environment.episode_terminal_displacement, mask
        ),
        "maximum_uav_speed_m_s": _summary(environment.episode_maximum_uav_speed, mask),
    }


@torch.no_grad()
def _execute(environment: Any, agent: Any) -> None:
    observation = environment.reset()
    for _ in range(environment.control_step_count):
        observation = environment.step(agent.deterministic_action(observation)).next_observation


def audit(artifact: Path) -> dict[str, Any]:
    config_path = artifact / "config.json"
    checkpoint_path = artifact / "checkpoints" / "best_validation.pt"
    best_metadata_path = artifact / "best_validation.json"
    if not checkpoint_path.is_file() or not best_metadata_path.is_file():
        raise FileNotFoundError("The artifact has no retained best-validation checkpoint.")
    config = _load_config(config_path)
    panel = FixedMildStateValidationPanel(config)
    agent = _build_agent(config, panel.environment.simulator.device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=panel.environment.simulator.device,
        weights_only=False,
    )
    agent.policy.load_state_dict(checkpoint["policy"])
    agent.value.load_state_dict(checkpoint["value"])
    agent.policy.eval()
    agent.value.eval()
    _execute(panel.environment, agent)

    distances = torch.as_tensor(
        panel.manifest["state_distances"],
        dtype=torch.float32,
        device=panel.environment.simulator.device,
    )
    lower = torch.quantile(distances, 1.0 / 3.0)
    upper = torch.quantile(distances, 2.0 / 3.0)
    validation_groups = {
        "all_validation": _group(panel.environment, torch.ones_like(distances, dtype=torch.bool)),
        "low_state_distance": _group(panel.environment, distances <= lower),
        "medium_state_distance": _group(
            panel.environment, (distances > lower) & (distances <= upper)
        ),
        "high_state_distance": _group(panel.environment, distances > upper),
    }

    canonical_environment, canonical_device = _build_environment(config, batch_size=1)
    canonical_agent = _build_agent(config, canonical_device)
    canonical_agent.policy.load_state_dict(checkpoint["policy"])
    canonical_agent.value.load_state_dict(checkpoint["value"])
    canonical_agent.policy.eval()
    canonical_agent.value.eval()
    _execute(canonical_environment, canonical_agent)
    canonical = _group(
        canonical_environment,
        torch.ones(1, dtype=torch.bool, device=canonical_device),
    )

    best_metadata = json.loads(best_metadata_path.read_text(encoding="utf-8"))
    result = {
        "schema": "ppo_best_validation_state_stratified_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": str(artifact),
        "checkpoint": str(checkpoint_path),
        "checkpoint_selection": "best deterministic 512-state validation success, then lower mean UAV displacement",
        "checkpoint_episodes": int(best_metadata["checkpoint_episodes"]),
        "model": "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI",
        "policy": "deterministic 10 Hz closed-loop PPO",
        "physics": "production UAV + causal residual + 12-node DDER; fixed numerical batch 2048",
        "validation_contexts": int(distances.numel()),
        "state_distance_strata": {
            "definition": "tertiles of the immutable validation-bank state-distance metric",
            "lower_threshold": float(lower.cpu()),
            "upper_threshold": float(upper.cpu()),
        },
        "canonical": canonical,
        "validation": validation_groups,
        "protected_test_evaluated": False,
        "hardware_executed": False,
    }
    _atomic_json(artifact / "best_validation_state_stratified_audit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.artifact.resolve()), indent=2))


if __name__ == "__main__":
    main()
