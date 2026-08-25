"""Plot the paired-seed MPPI maneuver-discovery study."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


STRATEGY_ORDER = (
    "initialization_zero",
    "initialization_random",
    "initialization_forward_recoil",
    "initialization_backward_forward",
    "initialization_lateral",
    "initialization_continuation",
)

DISPLAY_NAMES = {
    "initialization_zero": "zero",
    "initialization_random": "random smooth",
    "initialization_forward_recoil": "forward–recoil",
    "initialization_backward_forward": "backward–forward",
    "initialization_lateral": "lateral",
    "initialization_continuation": "continuation",
}


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _wilson(successes: int, trials: int) -> tuple[float, float, float]:
    probability = successes / trials
    z = 1.959963984540054
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    half = (
        z
        * np.sqrt(
            probability * (1.0 - probability) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return probability, max(0.0, center - half), min(1.0, center + half)


def load_discovery_rows(
    path: str | Path,
    *,
    profile: str,
    horizon_s: float,
) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if row.get("study") == "initialization"
        and row.get("profile") == profile
        and abs(float(row["horizon_s"]) - horizon_s) <= 1.0e-9
        and row.get("experiment") in STRATEGY_ORDER
    ]
    if not selected:
        raise ValueError(
            f"No initialization rows for profile={profile!r}, horizon={horizon_s:g}s."
        )
    budgets = {
        (row["iterations"], row["samples"], row["batch_size"]) for row in selected
    }
    if len(budgets) != 1:
        raise ValueError(
            "The plot input mixes MPPI compute budgets; select a separate profile/output."
        )
    return selected


def render_discovery_figure(
    rows: list[dict[str, str]],
    output_path: str | Path,
) -> Path:
    import matplotlib.pyplot as plt

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    grouped = {
        strategy: [row for row in rows if row["experiment"] == strategy]
        for strategy in STRATEGY_ORDER
    }
    grouped = {key: value for key, value in grouped.items() if value}
    strategies = list(grouped)
    labels = [DISPLAY_NAMES[value] for value in strategies]
    colors = plt.cm.tab10(np.linspace(0.0, 0.85, len(strategies)))
    x = np.arange(len(strategies))
    speed_thresholds = {
        float(row.get("minimum_impact_speed_m_s", 5.0)) for row in rows
    }
    position_thresholds = {
        float(row.get("maximum_tip_error_m", 0.05)) for row in rows
    }
    if len(speed_thresholds) != 1 or len(position_thresholds) != 1:
        raise ValueError("Plot input mixes different strike definitions.")
    speed_threshold = speed_thresholds.pop()
    position_threshold_mm = 1000.0 * position_thresholds.pop()

    figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), constrained_layout=True)

    probabilities = []
    lower = []
    upper = []
    for strategy in strategies:
        values = grouped[strategy]
        successes = sum(_as_bool(row["feasible"]) for row in values)
        probability, low, high = _wilson(successes, len(values))
        probabilities.append(probability)
        lower.append(max(0.0, probability - low))
        upper.append(max(0.0, high - probability))
    axes[0, 0].bar(x, probabilities, color=colors)
    axes[0, 0].errorbar(
        x,
        probabilities,
        yerr=np.asarray((lower, upper)),
        fmt="none",
        color="black",
        capsize=4,
    )
    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 0].set_ylabel("Successful final plan probability")
    axes[0, 0].set_title("Primary outcome (Wilson 95% CI)")

    successful_iterations = [
        [
            float(row["first_success_iteration"])
            for row in grouped[strategy]
            if row.get("first_success_iteration", "") not in {"", "None"}
        ]
        for strategy in strategies
    ]
    for index, values in enumerate(successful_iterations):
        if values:
            axes[0, 1].scatter(
                np.full(len(values), index), values, color=colors[index], alpha=0.75
            )
            axes[0, 1].plot(index, np.median(values), "k_", markersize=15)
    axes[0, 1].set_ylabel("Iteration of first successful sampled rollout")
    axes[0, 1].set_title("Discovery speed among successful trials")

    for index, strategy in enumerate(strategies):
        values = grouped[strategy]
        error = 1000.0 * np.asarray([float(row["position_error_m"]) for row in values])
        speed = np.asarray([float(row["directional_speed_m_s"]) for row in values])
        success = np.asarray([_as_bool(row["feasible"]) for row in values])
        axes[0, 2].scatter(
            error[~success],
            speed[~success],
            marker="x",
            color=colors[index],
            alpha=0.65,
            label=DISPLAY_NAMES[strategy],
        )
        axes[0, 2].scatter(
            error[success], speed[success], marker="o", color=colors[index], alpha=0.85
        )
    axes[0, 2].axvline(
        position_threshold_mm, color="#555555", linestyle="--", linewidth=1
    )
    axes[0, 2].axhline(
        speed_threshold, color="#555555", linestyle="--", linewidth=1
    )
    axes[0, 2].set_xlabel("Impact/closest-approach error (mm)")
    axes[0, 2].set_ylabel("Directed tip speed (m/s)")
    axes[0, 2].set_title("Task components (circle=success, x=failure)")
    axes[0, 2].legend(fontsize=7, loc="best")

    for index, strategy in enumerate(strategies):
        values = grouped[strategy]
        reversal = np.asarray([float(row["reversal_time_s"]) for row in values])
        impact = np.asarray([float(row["impact_time_s"]) for row in values])
        axes[1, 0].scatter(reversal, impact, color=colors[index], alpha=0.70)
    limits = axes[1, 0].get_xlim()
    axes[1, 0].plot(limits, limits, color="#aaaaaa", linewidth=1)
    axes[1, 0].set_xlabel("Inferred reversal time (s)")
    axes[1, 0].set_ylabel("Candidate impact time (s)")
    axes[1, 0].set_title("Maneuver timing")

    clearance = [
        [1000.0 * float(row["non_tip_clearance_margin_m"]) for row in grouped[strategy]]
        for strategy in strategies
    ]
    axes[1, 1].boxplot(clearance, positions=x, widths=0.6, showfliers=True)
    axes[1, 1].axhline(0.0, color="#b52b2b", linestyle="--", linewidth=1)
    axes[1, 1].set_ylabel("Minimum pre-impact non-tip clearance margin (mm)")
    axes[1, 1].set_title("Tip-first robustness margin")

    final_cost = [
        [float(row["mppi_objective"]) for row in grouped[strategy]]
        for strategy in strategies
    ]
    axes[1, 2].boxplot(final_cost, positions=x, widths=0.6, showfliers=True)
    axes[1, 2].set_ylabel("Final MPPI objective")
    axes[1, 2].set_title("Local optimum reached")

    for axis in axes.flat:
        axis.grid(True, color="#e1e1e1", linewidth=0.7, zorder=0)
    for axis in (axes[0, 0], axes[0, 1], axes[1, 1], axes[1, 2]):
        axis.set_xticks(x, labels, rotation=28, ha="right")
    budget = rows[0]
    seed_count = min(len(grouped[strategy]) for strategy in strategies)
    figure.suptitle(
        "MPPI maneuver-discovery reliability at fixed 2.0 s horizon\n"
        f"paired seeds per strategy={seed_count}, iterations={budget['iterations']}, "
        f"samples/iteration={budget['samples']}",
        fontsize=14,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("data/drone_mpc/ablations/mppi_discovery/runs.csv"),
    )
    parser.add_argument("--profile", default="discovery")
    parser.add_argument("--horizon", type=float, default=2.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/drone_mpc/ablations/mppi_discovery/discovery.png"),
    )
    arguments = parser.parse_args()
    rows = load_discovery_rows(
        arguments.runs, profile=arguments.profile, horizon_s=arguments.horizon
    )
    output = render_discovery_figure(rows, arguments.output)
    print(f"Figure: {output}")


if __name__ == "__main__":
    main()
