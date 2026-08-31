"""Artifact-based and analytic audit helpers for ``rl_whip_reward_v1``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from planning.rl_reward import (
    RLWhipRewardComponents,
    RLWhipRewardConfig,
    compose_rl_terminal_reward,
    rl_event_reward,
)


def _load_knots(directory: Path, replay: np.lib.npyio.NpzFile) -> torch.Tensor:
    artifact = directory / "best_acceleration_knots.json"
    if artifact.is_file():
        values = json.loads(artifact.read_text(encoding="utf-8"))["values"]
        return torch.tensor(values, dtype=torch.float64)
    acceleration = np.asarray(replay["a_cmd_m_s2"], dtype=np.float64)
    time = np.asarray(replay["time_s"], dtype=np.float64)
    sample_times = np.linspace(float(time[0]), float(time[-1]), 16)
    values = np.stack(
        [np.interp(sample_times, time, acceleration[:, axis]) for axis in range(3)],
        axis=-1,
    )
    return torch.from_numpy(values)


def _component_row(components: RLWhipRewardComponents) -> dict[str, float]:
    return {
        "R_progress": float(components.progress),
        "R_strike": float(components.strike),
        "R_success": float(components.success),
        "R_safety": float(components.safety),
        "R_non_tip": float(components.non_tip),
        "R_control": float(components.control),
        "R_time": float(components.time),
        "R_RL": float(components.total),
        "progress": float(components.normalized_progress),
        "d0_m": float(components.initial_tip_distance_m),
        "d_min_m": float(components.minimum_tip_distance_m),
    }


def audit_saved_replay(
    directory: str | Path,
    *,
    label: str,
    config: RLWhipRewardConfig,
) -> dict[str, Any]:
    """Evaluate one authoritative saved replay without rerunning physics."""

    source = Path(directory)
    replay_path = source / "final_replay.npz"
    if not replay_path.is_file():
        replay_path = source / "canonical_final_replay.npz"
    metrics_path = source / "final_metrics.json"
    if not metrics_path.is_file():
        metrics_path = source / "canonical_final_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    with np.load(replay_path, allow_pickle=False) as replay:
        distance = torch.from_numpy(
            np.asarray(replay["tip_target_distance_m"], dtype=np.float64)
        )
        directed = torch.from_numpy(
            np.asarray(replay["directed_tip_speed_m_s"], dtype=np.float64)
        )
        tip_speed = torch.from_numpy(
            np.asarray(replay["tip_speed_m_s"], dtype=np.float64)
        )
        cosine = directed / tip_speed.clamp_min(config.zero_speed_epsilon_m_s)
        event = rl_event_reward(distance, directed, tip_speed, cosine, config)
        valid_event = event[1:] if event.numel() > 1 else event
        strike = valid_event.max()
        knots = _load_knots(source, replay)

    knot_norm = torch.linalg.vector_norm(knots, dim=-1)
    effort = (knot_norm / 20.0).square().mean()
    smoothness = (
        torch.linalg.vector_norm(knots[1:] - knots[:-1], dim=-1) / 20.0
    ).square().mean()
    success = bool(metrics.get("success", False))
    marker_value = metrics.get("first_entry_marker")
    marker = 0 if marker_value is None else int(marker_value)
    hit_time_value = metrics.get("hit_time_s")
    hit_time = 0.0 if hit_time_value is None else float(hit_time_value)
    components = compose_rl_terminal_reward(
        strike_reward=strike.reshape(1),
        success=torch.tensor([success]),
        first_entry_marker=torch.tensor([marker]),
        first_hit_time_s=torch.tensor([hit_time], dtype=torch.float64),
        maximum_uav_displacement_m=torch.tensor(
            [metrics["maximum_uav_displacement_m"]], dtype=torch.float64
        ),
        maximum_uav_speed_m_s=torch.tensor(
            [metrics["maximum_uav_speed_m_s"]], dtype=torch.float64
        ),
        maximum_command_acceleration_m_s2=torch.tensor(
            [metrics["maximum_command_acceleration_m_s2"]], dtype=torch.float64
        ),
        effort_cost=effort.reshape(1),
        smoothness_cost=smoothness.reshape(1),
        displacement_limit_m=0.50,
        speed_limit_m_s=3.0,
        acceleration_limit_m_s2=20.0,
        config=config,
        initial_tip_distance_m=distance[:1],
        minimum_tip_distance_m=distance.min().reshape(1),
    )
    hard = metrics.get("hard_success_gates", {})
    return {
        "label": label,
        "source_directory": str(source),
        "source_replay": str(replay_path),
        "hard_success": success,
        "feasible": bool(metrics.get("feasible", False)),
        "tip_error_m": float(metrics["minimum_tip_target_distance_m"]),
        "directed_speed_m_s": float(metrics["reported_event_directed_tip_speed_m_s"]),
        "direction_error_deg": float(metrics["reported_event_direction_error_deg"]),
        "tip_first": bool(hard.get("tip_first", marker == 10)),
        "first_entry_marker": None if marker == 0 else marker,
        "maximum_uav_displacement_m": float(metrics["maximum_uav_displacement_m"]),
        "maximum_uav_speed_m_s": float(metrics["maximum_uav_speed_m_s"]),
        "maximum_command_acceleration_m_s2": float(
            metrics["maximum_command_acceleration_m_s2"]
        ),
        "legacy_reference_reward": -float(
            metrics.get("task_cost", metrics.get("cost", float("nan")))
        ),
        "control_effort": float(effort),
        "control_smoothness": float(smoothness),
        **{name: value for name, value in _component_row(components).items()},
    }


def analytic_reward_slices(config: RLWhipRewardConfig) -> dict[str, np.ndarray]:
    """Return deterministic analytic slices used both for plots and tests."""

    distance = torch.linspace(0.0, 1.2, 241, dtype=torch.float64)
    useful_speed = torch.full_like(distance, 4.5)
    useful_cosine = torch.full_like(distance, np.cos(np.deg2rad(20.0)))
    distance_score = rl_event_reward(
        distance, useful_speed, useful_speed, useful_cosine, config
    )
    directed_speed = torch.linspace(-5.0, 8.0, 261, dtype=torch.float64)
    speed_norm = torch.abs(directed_speed).clamp_min(1.0e-8)
    speed_curves = []
    for d in (0.02, 0.20, 0.80):
        speed_curves.append(
            rl_event_reward(
                torch.full_like(directed_speed, d),
                directed_speed,
                speed_norm,
                directed_speed / speed_norm,
                config,
            )
        )
    angle = torch.linspace(0.0, 180.0, 361, dtype=torch.float64)
    cosine = torch.cos(torch.deg2rad(angle))
    direction_curves = []
    for d in (0.02, 0.20, 0.80):
        direction_curves.append(
            rl_event_reward(
                torch.full_like(angle, d),
                torch.full_like(angle, 4.5),
                torch.full_like(angle, 4.5),
                cosine,
                config,
            )
        )
    displacement = torch.linspace(0.0, 1.5, 301, dtype=torch.float64)
    speed = torch.linspace(0.0, 9.0, 301, dtype=torch.float64)
    displacement_penalty = -config.safety_weight * torch.relu(
        (displacement - 0.5) / 0.5
    ).square().clamp(max=config.normalized_safety_clip)
    speed_penalty = -config.safety_weight * torch.relu(
        (speed - 3.0) / 3.0
    ).square().clamp(max=config.normalized_safety_clip)
    return {
        "distance_m": distance.numpy(),
        "distance_score": distance_score.numpy(),
        "directed_speed_m_s": directed_speed.numpy(),
        "speed_scores": torch.stack(speed_curves).numpy(),
        "direction_angle_deg": angle.numpy(),
        "direction_scores": torch.stack(direction_curves).numpy(),
        "uav_displacement_m": displacement.numpy(),
        "displacement_penalty": displacement_penalty.numpy(),
        "uav_speed_m_s": speed.numpy(),
        "speed_penalty": speed_penalty.numpy(),
    }
