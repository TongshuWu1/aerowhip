"""Evaluate first-success physical metrics for a saved simple-PPO checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_simple_ppo import _build_agent, _load_config
from run_simple_sac import _build_environment


def _summary(values: torch.Tensor) -> dict[str, float]:
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


def _safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@torch.no_grad()
def evaluate(
    config_path: Path,
    checkpoint_path: Path,
    *,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    config = _load_config(config_path)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    environment, device = _build_environment(config, batch_size=batch_size)
    agent = _build_agent(config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent.policy.load_state_dict(checkpoint["policy"])
    agent.value.load_state_dict(checkpoint["value"])
    agent.policy.eval()
    agent.value.eval()
    observation = environment.reset()
    for _ in range(environment.control_step_count):
        action, _log_probability, _value = agent.act(observation)
        observation = environment.step(action).next_observation
    success = environment.episode_success
    endpoint_success = environment.episode_endpoint_success
    scientific_success = environment.episode_scientific_success
    success_count = int(success.sum().cpu())
    return {
        "schema": "sequential_ppo_checkpoint_success_diagnostic_v2",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_episodes": int(checkpoint.get("episodes", -1)),
        "evaluation_seed": int(seed),
        "evaluation_episodes": int(batch_size),
        "successes": success_count,
        "success_rate": success_count / batch_size,
        "success_mode": environment.success_mode,
        "endpoint_successes": int(endpoint_success.sum().cpu()),
        "endpoint_success_rate": float(endpoint_success.float().mean().cpu()),
        "scientific_successes": int(scientific_success.sum().cpu()),
        "scientific_success_rate": float(scientific_success.float().mean().cpu()),
        "tip_first_count": int(
            (environment.episode_first_entry_marker == 10).sum().cpu()
        ),
        "first_entry_marker_histogram": {
            str(marker): int(
                (environment.episode_first_entry_marker == marker).sum().cpu()
            )
            for marker in range(0, 11)
        },
        "first_entry_physics_step": _summary(
            environment.episode_first_entry_physics_step[
                environment.episode_first_entry_marker > 0
            ].float()
        ),
        "tip_first_entry_physics_step": _summary(
            environment.episode_first_entry_physics_step[
                environment.episode_first_entry_marker == 10
            ].float()
        ),
        "success_timestep_gate": "NONE; first-entry timestep is diagnostic only",
        "first_success_tip_distance_m": _summary(
            environment.episode_success_tip_distance[success]
        ),
        "first_success_tip_speed_m_s": _summary(
            environment.episode_success_tip_speed[success]
        ),
        "first_success_directed_speed_m_s": _summary(
            environment.episode_success_directed_speed[success]
        ),
        "first_success_direction_error_deg": _summary(
            environment.episode_success_direction_error_deg[success]
        ),
        "episode_minimum_tip_distance_m": _summary(
            environment.episode_minimum_tip_distance
        ),
        "maximum_uav_displacement_m": _summary(
            environment.episode_maximum_displacement
        ),
        "maximum_uav_speed_m_s": _summary(environment.episode_maximum_uav_speed),
        "maximum_command_acceleration_m_s2": _summary(
            environment.episode_maximum_command_acceleration
        ),
        "numerical_failures": int(environment.failed.sum().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7042)
    arguments = parser.parse_args()
    result = evaluate(
        arguments.config.resolve(),
        arguments.checkpoint.resolve(),
        batch_size=int(arguments.batch_size),
        seed=int(arguments.seed),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, default=_safe) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, default=_safe), flush=True)


if __name__ == "__main__":
    main()
