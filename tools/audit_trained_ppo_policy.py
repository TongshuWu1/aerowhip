"""Authoritative deterministic canonical and 512-state audit for trained PPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learning.state_bank import InitialStateBank
from run_ppo_open_loop_compiler_audit import _environment, _feedback_rollout, _metric_rows
from run_simple_ppo import _build_agent, _load_config
from run_simple_sac import _build_environment
from simulator.parameters import SimulatorSettings
from simulator.production import build_production_simulator


def _statistics(values: torch.Tensor) -> dict[str, float]:
    finite = values[torch.isfinite(values)].detach().cpu().double().numpy()
    if finite.size == 0:
        return {}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "p95": float(np.quantile(finite, 0.95)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
    }


def _summary(metrics: Any) -> dict[str, Any]:
    successes = metrics.task_success
    entered = metrics.first_entry_marker > 0
    count = metrics.batch_size
    return {
        "contexts": count,
        "task_successes": int(successes.sum().cpu()),
        "task_success_rate": float(successes.float().mean().cpu()),
        "endpoint_successes": int(metrics.endpoint_success.sum().cpu()),
        "endpoint_success_rate": float(metrics.endpoint_success.float().mean().cpu()),
        "legacy_hard_gate_successes": int(metrics.legacy_scientific_success.sum().cpu()),
        "legacy_hard_gate_success_rate": float(
            metrics.legacy_scientific_success.float().mean().cpu()
        ),
        "finite_rate": float(metrics.finite.float().mean().cpu()),
        "tip_first_count": int((metrics.first_entry_marker == 10).sum().cpu()),
        "first_entry_marker_histogram": {
            str(marker): int((metrics.first_entry_marker == marker).sum().cpu())
            for marker in range(11)
        },
        "successful_entry_time_s": _statistics(metrics.first_entry_time_s[successes]),
        "successful_tip_distance_m": _statistics(
            metrics.first_entry_tip_distance_m[successes]
        ),
        "successful_directed_speed_m_s": _statistics(
            metrics.first_entry_directed_speed_m_s[successes]
        ),
        "successful_direction_error_deg": _statistics(
            metrics.first_entry_direction_error_deg[successes]
        ),
        "entered_directed_speed_m_s": _statistics(
            metrics.first_entry_directed_speed_m_s[entered]
        ),
        "entered_direction_error_deg": _statistics(
            metrics.first_entry_direction_error_deg[entered]
        ),
        "minimum_tip_distance_m": _statistics(metrics.minimum_tip_distance_m),
        "maximum_uav_displacement_m": _statistics(metrics.maximum_uav_displacement_m),
        "maximum_uav_speed_m_s": _statistics(metrics.maximum_uav_speed_m_s),
        "maximum_command_acceleration_m_s2": _statistics(
            metrics.maximum_command_acceleration_m_s2
        ),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@torch.no_grad()
def audit(
    config_path: Path,
    checkpoint_path: Path,
    state_bank_root: Path,
    output_directory: Path,
) -> dict[str, Any]:
    config = _load_config(config_path)
    settings = SimulatorSettings.load(ROOT / config["simulator_config"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    fixed_batch_size = int(config["collection_batch"])

    bank = InitialStateBank.load(
        state_bank_root / "validation_state_bank.npz",
        state_bank_root / "validation_state_bank_manifest.json",
    )
    simulator = build_production_simulator(settings)
    simulator.uav_model.set_fixed_evaluation_batch_size(fixed_batch_size)
    indices = torch.arange(len(bank), dtype=torch.int64)
    selected = bank.select(indices, device=simulator.device)
    agent = _build_agent(config, simulator.device)
    agent.policy.load_state_dict(checkpoint["policy"])
    agent.value.load_state_dict(checkpoint["value"])
    state_metrics = _feedback_rollout(
        _environment(simulator, config, selected.state), agent
    )

    canonical_environment, _device = _build_environment(config, batch_size=1)
    canonical_metrics = _feedback_rollout(canonical_environment, agent)

    result = {
        "schema": "trained_ppo_authoritative_audit_v1",
        "policy": "deterministic tanh(mean)",
        "checkpoint": str(checkpoint_path.resolve()),
        "config": str(config_path.resolve()),
        "model_freeze": config["model_freeze"],
        "episode_duration_s": float(config["episode_duration_s"]),
        "fixed_uav_residual_evaluation_batch_size": fixed_batch_size,
        "new_training": False,
        "protected_test_evaluated": False,
        "hardware_executed": False,
        "canonical": _summary(canonical_metrics),
        "state_bank_512": _summary(state_metrics),
    }
    _write_json(output_directory / "canonical_metrics.json", result["canonical"])
    _write_json(output_directory / "state_bank_512_summary.json", result["state_bank_512"])
    _write_json(
        output_directory / "state_bank_512_rows.json",
        _metric_rows(state_metrics, indices),
    )
    _write_json(output_directory / "audit_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state-bank-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit(
        arguments.config.resolve(),
        arguments.checkpoint.resolve(),
        arguments.state_bank_root.resolve(),
        arguments.output_directory.resolve(),
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
