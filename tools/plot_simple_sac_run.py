"""Render compact postmortem figures from a simple sequential SAC training log."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Training log is empty.")
    return {
        key: np.asarray(
            [float(row[key]) if row[key].strip() else np.nan for row in rows],
            dtype=np.float64,
        )
        for key in rows[0]
    }


def _rolling_rate(episodes: np.ndarray, successes: np.ndarray, window: int = 50) -> np.ndarray:
    result = np.zeros_like(episodes)
    for index in range(len(episodes)):
        start = max(0, index - window)
        previous_episodes = 0.0 if start == 0 else episodes[start - 1]
        previous_successes = 0.0 if start == 0 else successes[start - 1]
        denominator = episodes[index] - previous_episodes
        result[index] = (successes[index] - previous_successes) / max(denominator, 1.0)
    return result


def _rolling_mean(values: np.ndarray, window: int = 20) -> np.ndarray:
    kernel = np.ones(window, dtype=np.float64)
    numerator = np.convolve(values, kernel, mode="same")
    denominator = np.convolve(np.ones_like(values), kernel, mode="same")
    return numerator / denominator


def render(artifact: Path) -> None:
    values = _read(artifact / "training_log.csv")
    figures = artifact / "figures"
    figures.mkdir(exist_ok=True)
    episodes = values["episodes"]
    successes = values["successes"]
    episode_millions = episodes / 1.0e6
    increments = np.diff(np.concatenate(([0.0], successes)))
    success_rows = np.flatnonzero(increments > 0.0)
    last_success_episode = int(episodes[success_rows[-1]]) if len(success_rows) else None
    rolling = _rolling_rate(episodes, successes)

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    axes[0, 0].plot(episode_millions, 100.0 * values["total_success_rate"], color="#b42318", lw=2)
    axes[0, 0].set(title="Cumulative endpoint success rate", ylabel="Success (%)")
    axes[0, 1].plot(episode_millions, 100.0 * rolling, color="#175cd3", lw=1.6)
    axes[0, 1].set(title="Rolling success rate (~100k episodes)", ylabel="Success (%)")
    axes[1, 0].step(episode_millions, successes, where="post", color="#067647", lw=2)
    axes[1, 0].set(title="Cumulative successes", xlabel="Episodes (millions)", ylabel="Count")
    axes[1, 1].plot(episode_millions, values["episodes_per_second"], color="#6941c6", lw=1.5)
    axes[1, 1].set(title="Simulation throughput", xlabel="Episodes (millions)", ylabel="Episodes/s")
    if last_success_episode is not None:
        marker = last_success_episode / 1.0e6
        for axis in axes.flat[:3]:
            axis.axvline(marker, color="black", ls="--", lw=1, alpha=0.65)
        axes[0, 0].annotate(
            f"last success-containing batch\n{last_success_episode:,} episodes",
            xy=(marker, 100.0 * values["total_success_rate"][success_rows[-1]]),
            xytext=(8, 12),
            textcoords="offset points",
            fontsize=8,
        )
    figure.suptitle("Simple sequential SAC — stopped run", fontsize=14, fontweight="bold")
    figure.savefig(figures / "01_success_and_throughput.png", dpi=180)
    plt.close(figure)

    diagnostic_names = (
        ("mean_episode_reward", "Mean episode reward", "Reward", False),
        ("median_minimum_tip_distance_m", "Median minimum tip distance", "Distance (m)", False),
        ("mean_maximum_uav_displacement_m", "Mean maximum UAV displacement", "Displacement (m)", False),
        ("alpha", "SAC entropy temperature", "Alpha (log scale)", True),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    colors = ("#b42318", "#175cd3", "#b54708", "#6941c6")
    for axis, (key, title, ylabel, logarithmic), color in zip(
        axes.flat, diagnostic_names, colors, strict=True
    ):
        axis.plot(episode_millions, values[key], color=color, alpha=0.25, lw=0.7)
        axis.plot(episode_millions, _rolling_mean(values[key]), color=color, lw=1.8)
        axis.set(title=title, xlabel="Episodes (millions)", ylabel=ylabel)
        if logarithmic:
            axis.set_yscale("log")
        if last_success_episode is not None:
            axis.axvline(last_success_episode / 1.0e6, color="black", ls="--", lw=1, alpha=0.65)
    figure.suptitle("Learning diagnostics (raw + rolling mean)", fontsize=14, fontweight="bold")
    figure.savefig(figures / "02_learning_diagnostics.png", dpi=180)
    plt.close(figure)

    summary = {
        "schema": "simple_sequential_sac_stopped_run_summary_v1",
        "episodes_completed": int(episodes[-1]),
        "target_episodes": 3_000_000,
        "completion_fraction": float(episodes[-1] / 3_000_000),
        "successes": int(successes[-1]),
        "final_success_rate": float(values["total_success_rate"][-1]),
        "peak_cumulative_success_rate": float(np.max(values["total_success_rate"])),
        "last_success_containing_batch_episode": last_success_episode,
        "episodes_without_a_new_success_at_stop": (
            None if last_success_episode is None else int(episodes[-1] - last_success_episode)
        ),
        "final_episodes_per_second": float(values["episodes_per_second"][-1]),
        "final_alpha": float(values["alpha"][-1]),
        "final_median_minimum_tip_distance_m": float(values["median_minimum_tip_distance_m"][-1]),
        "final_mean_maximum_uav_displacement_m": float(values["mean_maximum_uav_displacement_m"][-1]),
        "numerical_failure_rate": float(values["numerical_failure_rate"][-1]),
        "figures": [
            "figures/01_success_and_throughput.png",
            "figures/02_learning_diagnostics.png",
        ],
    }
    (artifact / "stopped_run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    render(args.artifact.expanduser().resolve())


if __name__ == "__main__":
    main()
