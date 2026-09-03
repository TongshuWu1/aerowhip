"""Audit whether a PPO whip's forward/reverse release occurs at impact."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from learning.ppo_validation import FixedMildStateValidationPanel
from run_simple_ppo import _build_agent, _load_config


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p90": float(np.percentile(finite, 90)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
    }


def _release_quality(
    peak_forward_m: np.ndarray,
    uav_forward_speed_m_s: np.ndarray,
    relative_tip_forward_speed_m_s: np.ndarray,
    *,
    excursion_scale_m: float,
    backward_speed_scale_m_s: float,
    tip_speed_scale_m_s: float,
) -> np.ndarray:
    forward = 1.0 - np.exp(-np.maximum(peak_forward_m, 0.0) / excursion_scale_m)
    backward = 1.0 - np.exp(
        -np.maximum(-uav_forward_speed_m_s, 0.0) / backward_speed_scale_m_s
    )
    tip = 1.0 - np.exp(
        -np.maximum(relative_tip_forward_speed_m_s, 0.0) / tip_speed_scale_m_s
    )
    return forward * backward * tip


def _stratified_rows(success_rows: np.ndarray, distances: np.ndarray, count: int) -> np.ndarray:
    ordered = success_rows[np.argsort(distances[success_rows])]
    if ordered.size <= count:
        return ordered
    positions = np.linspace(0, ordered.size - 1, count).round().astype(np.int64)
    return ordered[positions]


def _aligned(values: np.ndarray, hit_steps: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    result = np.full((values.shape[0], offsets.size), np.nan, dtype=np.float64)
    for row, hit in enumerate(hit_steps):
        source = hit + offsets
        valid = (source >= 0) & (source < values.shape[1])
        result[row, valid] = values[row, source[valid]]
    return result


def _plot(
    destination: Path,
    *,
    dt_s: float,
    hit_steps: np.ndarray,
    forward_displacement_m: np.ndarray,
    uav_forward_speed_m_s: np.ndarray,
    relative_tip_forward_speed_m_s: np.ndarray,
    tip_distance_m: np.ndarray,
    release_quality: np.ndarray,
) -> None:
    # Do not request pre-episode samples for the earliest successful strike.
    offsets = np.arange(-min(120, int(np.min(hit_steps))), 1, dtype=np.int64)
    relative_time = offsets * dt_s
    series = (
        (forward_displacement_m, "UAV forward displacement", "m"),
        (uav_forward_speed_m_s, "UAV target-axis velocity", "m/s"),
        (relative_tip_forward_speed_m_s, "Tip velocity relative to attachment", "m/s"),
        (tip_distance_m, "Tip-target distance", "m"),
        (release_quality, "Release quality", "0-1"),
    )
    fig, axes = plt.subplots(len(series), 1, figsize=(10, 12), sharex=True)
    for axis, (values, title, units) in zip(axes, series, strict=True):
        aligned = _aligned(values, hit_steps, offsets)
        for row in aligned:
            axis.plot(relative_time, row, color="#4c78a8", alpha=0.10, linewidth=0.7)
        median = np.nanmedian(aligned, axis=0)
        lower = np.nanpercentile(aligned, 25, axis=0)
        upper = np.nanpercentile(aligned, 75, axis=0)
        axis.fill_between(relative_time, lower, upper, color="#4c78a8", alpha=0.22)
        axis.plot(relative_time, median, color="#174a7e", linewidth=2.0)
        axis.axvline(0.0, color="#d62728", linestyle="--", linewidth=1.2)
        axis.axhline(0.0, color="#808080", linewidth=0.6)
        axis.set_ylabel(units)
        axis.set_title(title, loc="left", fontsize=10, fontweight="bold")
        axis.grid(alpha=0.18)
    axes[-1].set_xlabel("Time relative to successful tip strike (s)")
    fig.suptitle("PPO whip mechanism audit: median, IQR, and 32 held-out successes")
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def audit(artifact: Path, checkpoint_name: str) -> dict[str, Any]:
    config_path = artifact / "config.json"
    checkpoint_path = artifact / "checkpoints" / checkpoint_name
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("Run config or requested checkpoint is missing.")
    config = _load_config(config_path)
    panel = FixedMildStateValidationPanel(config, record_state_trajectory=True)
    environment = panel.environment
    agent = _build_agent(config, environment.simulator.device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=environment.simulator.device,
        weights_only=False,
    )
    agent.policy.load_state_dict(checkpoint["policy"])
    agent.value.load_state_dict(checkpoint["value"])
    agent.policy.eval()
    agent.value.eval()

    observation = environment.reset()
    for _ in range(environment.control_step_count):
        observation = environment.step(
            agent.deterministic_action(observation)
        ).next_observation

    trajectory = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in environment.recorded_state_trajectory(clone=False).items()
    }
    success = environment.episode_success.detach().cpu().numpy().astype(bool)
    hit_steps = environment.episode_first_entry_physics_step.detach().cpu().numpy()
    distances = np.asarray(panel.manifest["state_distances"], dtype=np.float64)
    direction = environment._direction.detach().cpu().numpy()
    target = environment._target.detach().cpu().numpy()

    uav_position = trajectory["uav_position_m"]
    uav_velocity = trajectory["uav_velocity_m_s"]
    cable_position = trajectory["cable_positions_m"]
    cable_velocity = trajectory["cable_velocities_m_s"]
    displacement = uav_position - uav_position[0:1]
    forward_displacement = np.einsum("tbi,bi->bt", displacement, direction)
    uav_forward_speed = np.einsum("tbi,bi->bt", uav_velocity, direction)
    relative_tip_velocity = cable_velocity[:, :, -1] - cable_velocity[:, :, 0]
    relative_tip_forward_speed = np.einsum("tbi,bi->bt", relative_tip_velocity, direction)
    tip_distance = np.linalg.norm(cable_position[:, :, -1] - target[None], axis=-1).T
    peak_forward = np.maximum.accumulate(forward_displacement, axis=1)
    reward = config["reward"]
    release = _release_quality(
        peak_forward,
        uav_forward_speed,
        relative_tip_forward_speed,
        excursion_scale_m=float(reward["forward_excursion_scale_m"]),
        backward_speed_scale_m_s=float(reward["uav_backward_speed_scale_m_s"]),
        tip_speed_scale_m_s=float(reward["relative_tip_forward_speed_scale_m_s"]),
    )

    successful_rows = np.flatnonzero(success)
    rows: list[dict[str, Any]] = []
    for row in successful_rows:
        hit = int(hit_steps[row])
        before_hit = release[row, : hit + 1]
        release_step = int(np.argmax(before_hit))
        rows.append(
            {
                "validation_row": int(row),
                "state_bank_index": int(panel.indices[row]),
                "state_distance": float(distances[row]),
                "hit_step": hit,
                "hit_time_s": hit * environment.simulator.dt_s,
                "peak_release_step": release_step,
                "peak_release_time_s": release_step * environment.simulator.dt_s,
                "peak_release_minus_hit_s": (release_step - hit) * environment.simulator.dt_s,
                "peak_forward_displacement_m": float(peak_forward[row, hit]),
                "forward_displacement_at_hit_m": float(forward_displacement[row, hit]),
                "return_distance_at_hit_m": float(
                    peak_forward[row, hit] - forward_displacement[row, hit]
                ),
                "uav_forward_speed_at_hit_m_s": float(uav_forward_speed[row, hit]),
                "relative_tip_forward_speed_at_hit_m_s": float(
                    relative_tip_forward_speed[row, hit]
                ),
                "peak_release_quality": float(release[row, release_step]),
                "impact_release_quality": float(release[row, hit]),
                "tip_distance_at_hit_m": float(tip_distance[row, hit]),
            }
        )

    output = artifact / "mechanism_audit"
    output.mkdir(exist_ok=True)
    _atomic_json(output / "mechanism_audit_rows.json", rows)
    chosen = _stratified_rows(successful_rows, distances, 32)
    np.savez_compressed(
        output / "mechanism_timeseries.npz",
        time_s=np.arange(forward_displacement.shape[1]) * environment.simulator.dt_s,
        validation_rows=chosen,
        state_bank_indices=np.asarray(panel.indices)[chosen],
        hit_steps=hit_steps[chosen],
        forward_displacement_m=forward_displacement[chosen],
        uav_forward_speed_m_s=uav_forward_speed[chosen],
        relative_tip_forward_speed_m_s=relative_tip_forward_speed[chosen],
        tip_distance_m=tip_distance[chosen],
        release_quality=release[chosen],
    )
    _plot(
        output / "mechanism_audit.png",
        dt_s=environment.simulator.dt_s,
        hit_steps=hit_steps[chosen],
        forward_displacement_m=forward_displacement[chosen],
        uav_forward_speed_m_s=uav_forward_speed[chosen],
        relative_tip_forward_speed_m_s=relative_tip_forward_speed[chosen],
        tip_distance_m=tip_distance[chosen],
        release_quality=release[chosen],
    )

    if not rows:
        raise RuntimeError("The audited checkpoint produced no successful validation rows.")
    values = {
        key: np.asarray([row[key] for row in rows])
        for key in rows[0]
        if key not in {"validation_row", "state_bank_index"}
    }
    release_delta = values["peak_release_minus_hit_s"]
    impact_to_peak = values["impact_release_quality"] / np.maximum(
        values["peak_release_quality"], 1.0e-12
    )
    stored = environment.episode_success_release_quality.detach().cpu().numpy()[success]
    calculated = values["peak_release_quality"]
    summary = {
        "schema": "ppo_whip_mechanism_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifact": str(artifact),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "config_sha256": _sha256(config_path),
        "validation_contexts": int(success.size),
        "successful_contexts": int(success.sum()),
        "success_rate": float(success.mean()),
        "recording_dt_s": environment.simulator.dt_s,
        "production_physics": "UAV + causal residual + 12-node DDER; fixed residual batch 2048",
        "release_definition": "peak forward activation * backward UAV target-axis velocity * forward tip target-axis velocity relative to DDER root attachment",
        "release_peak_timing_relative_to_hit_s": _stats(release_delta),
        "peak_release_within_0p10_s_of_hit_fraction": float(np.mean(np.abs(release_delta) <= 0.10 + 1.0e-9)),
        "peak_release_within_0p20_s_of_hit_fraction": float(np.mean(np.abs(release_delta) <= 0.20 + 1.0e-9)),
        "peak_forward_displacement_m": _stats(values["peak_forward_displacement_m"]),
        "forward_displacement_at_hit_m": _stats(values["forward_displacement_at_hit_m"]),
        "return_distance_at_hit_m": _stats(values["return_distance_at_hit_m"]),
        "uav_forward_speed_at_hit_m_s": _stats(values["uav_forward_speed_at_hit_m_s"]),
        "relative_tip_forward_speed_at_hit_m_s": _stats(values["relative_tip_forward_speed_at_hit_m_s"]),
        "peak_release_quality": _stats(values["peak_release_quality"]),
        "impact_release_quality": _stats(values["impact_release_quality"]),
        "impact_to_peak_release_quality_ratio": _stats(impact_to_peak),
        "stored_vs_recomputed_peak_release_max_abs_error": float(
            np.max(np.abs(stored - calculated))
        ),
        "interpretation": (
            "IMPACT_COUPLING_MISSING"
            if float(np.median(impact_to_peak)) < 0.75
            or float(np.mean(np.abs(release_delta) <= 0.10 + 1.0e-9)) < 0.75
            else "CURRENT_RELEASE_IS_IMPACT_COUPLED"
        ),
        "new_cem_solves": 0,
        "hardware_executed": False,
        "protected_data_accessed": False,
    }
    _atomic_json(output / "mechanism_audit_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best_validation.pt")
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.artifact.resolve(), arguments.checkpoint), indent=2))


if __name__ == "__main__":
    main()
