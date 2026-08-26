"""Render the frozen Figure-8 history-observer study from saved NPZ data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "reports" / "figure8_history_observer_data"
OUTPUT = PROJECT / "reports" / "FIGURE8_DDER_HISTORY_OBSERVER_RESULTS.png"
SEEDS = (17, 23, 41)
MODES = ("full", "endpoint", "history")
LABELS = {
    "full": "Full distributed state",
    "endpoint": "Instantaneous endpoint",
    "history": "Endpoint history + DDER",
}
COLORS = {"full": "#1b9e77", "endpoint": "#d95f02", "history": "#386cb0"}


def load(condition: str, mode: str, seed: int) -> np.lib.npyio.NpzFile:
    return np.load(DATA / f"{condition}__{mode}__seed{seed}.npz")


def mean_band(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stacked = np.stack(values)
    return np.mean(stacked, axis=0), np.min(stacked, axis=0), np.max(stacked, axis=0)


def main() -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 7.6), constrained_layout=True)

    tracking_axis = axes[0, 0]
    for mode in MODES:
        runs = [load("hidden_interior_velocity", mode, seed) for seed in SEEDS]
        time_s = runs[0]["time_s"]
        mean, low, high = mean_band(
            [1000.0 * run["tracking_errors_m"] for run in runs]
        )
        tracking_axis.plot(time_s, mean, color=COLORS[mode], label=LABELS[mode])
        tracking_axis.fill_between(time_s, low, high, color=COLORS[mode], alpha=0.14)
    tracking_axis.axvline(0.30, color="0.35", linestyle="--", linewidth=1.0)
    tracking_axis.text(0.315, 3.0, "history ready", fontsize=8, color="0.3")
    tracking_axis.set_title("Hidden-state test: free-tip tracking")
    tracking_axis.set_xlabel("Time (s)")
    tracking_axis.set_ylabel("Tip error (mm)")
    tracking_axis.grid(alpha=0.25)
    tracking_axis.legend(fontsize=8)

    state_axis = axes[0, 1]
    for mode in ("endpoint", "history"):
        values: list[np.ndarray] = []
        for seed in SEEDS:
            run = load("hidden_interior_velocity", mode, seed)
            error = (
                run["estimated_cable_positions_m"][:, 1:]
                - run["cable_positions_m"][:, 1:]
            )
            values.append(1000.0 * np.sqrt(np.mean(np.square(error), axis=(1, 2))))
        mean, low, high = mean_band(values)
        state_axis.plot(time_s, mean, color=COLORS[mode], label=LABELS[mode])
        state_axis.fill_between(time_s, low, high, color=COLORS[mode], alpha=0.14)
    state_axis.axvline(0.30, color="0.35", linestyle="--", linewidth=1.0)
    state_axis.set_title("Hidden-state reconstruction")
    state_axis.set_xlabel("Time (s)")
    state_axis.set_ylabel("Distributed position RMSE (mm)")
    state_axis.grid(alpha=0.25)
    state_axis.legend(fontsize=8)

    residual_axis = axes[1, 0]
    before_runs: list[np.ndarray] = []
    after_runs: list[np.ndarray] = []
    for seed in SEEDS:
        run = load("hidden_interior_velocity", "history", seed)
        before_runs.append(1000.0 * run["observer_history_rmse_before_m"])
        after_runs.append(1000.0 * run["observer_history_rmse_after_m"])
    update_time = 0.10 * np.arange(len(before_runs[0]))
    before_stack = np.stack(before_runs)
    after_stack = np.stack(after_runs)
    valid = np.any(np.isfinite(before_stack), axis=0)
    before = np.full(before_stack.shape[1], np.nan)
    after = np.full(after_stack.shape[1], np.nan)
    before[valid] = np.nanmean(before_stack[:, valid], axis=0)
    after[valid] = np.nanmean(after_stack[:, valid], axis=0)
    residual_axis.plot(
        update_time[valid], before[valid], "o-", color="#7570b3", label="Before correction"
    )
    residual_axis.plot(
        update_time[valid], after[valid], "o-", color=COLORS["history"], label="After correction"
    )
    residual_axis.set_title("Endpoint-history fit at each observer update")
    residual_axis.set_xlabel("Time (s)")
    residual_axis.set_ylabel("History residual RMSE (mm)")
    residual_axis.grid(alpha=0.25)
    residual_axis.legend(fontsize=8)

    prediction_axis = axes[1, 1]
    horizons = (0.10, 0.20, 0.30)
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))

    def prediction_values(mode: str) -> tuple[float, ...]:
        rows = [
            row
            for row in summary["rows"]
            if row["condition"] == "hidden_interior_velocity" and row["mode"] == mode
        ]
        return tuple(
            1000.0
            * float(
                np.mean(
                    [
                        row[f"estimated_tip_prediction_rmse_{horizon:.2f}s_m"]
                        for row in rows
                    ]
                )
            )
            for horizon in horizons
        )

    endpoint = prediction_values("endpoint")
    history = prediction_values("history")
    indices = np.arange(len(horizons))
    width = 0.36
    prediction_axis.bar(
        indices - width / 2,
        endpoint,
        width,
        color=COLORS["endpoint"],
        label=LABELS["endpoint"],
    )
    prediction_axis.bar(
        indices + width / 2,
        history,
        width,
        color=COLORS["history"],
        label=LABELS["history"],
    )
    prediction_axis.set_xticks(indices, [f"{value:.1f}" for value in horizons])
    prediction_axis.set_title("Future tip prediction from estimated state")
    prediction_axis.set_xlabel("Prediction horizon (s)")
    prediction_axis.set_ylabel("Tip-position RMSE (mm)")
    prediction_axis.grid(axis="y", alpha=0.25)
    prediction_axis.legend(fontsize=8)

    figure.suptitle(
        "DDER moving-history observer: matched physics, hidden interior velocity",
        fontsize=13,
    )
    figure.savefig(OUTPUT, dpi=180)
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
