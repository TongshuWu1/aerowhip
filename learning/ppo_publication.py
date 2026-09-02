"""Reproducible, paper-ready figures from a durable PPO artifact."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .ppo_validation import rolling_success_rate


PUBLICATION_ROLLING_WINDOW_EPISODES = 5_000
FIGURE_FORMATS = ("png", "pdf", "svg")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_figure(figure: Any, output: Path, stem: str) -> list[str]:
    written: list[str] = []
    for suffix in FIGURE_FORMATS:
        path = output / f"{stem}.{suffix}"
        options: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix == "png":
            options["dpi"] = 300
        figure.savefig(path, **options)
        written.append(path.name)
    return written


def _finish_axis(axis: Any) -> None:
    axis.grid(True, color="#cbd5e1", alpha=0.55, linewidth=0.65)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _summary_value(summary: dict[str, Any], name: str) -> float | None:
    value = summary.get(name)
    return None if value is None else float(value)


def export_ppo_publication_figures(
    artifact: str | Path, output_directory: str | Path | None = None
) -> dict[str, Any]:
    """Export scientific learning, compactness, strike, and optimization figures."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifact = Path(artifact).resolve()
    output = (
        artifact / "publication_figures"
        if output_directory is None
        else Path(output_directory).resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    training_path = artifact / "training_log.csv"
    validation_path = artifact / "validation_history.csv"
    if not training_path.is_file():
        raise FileNotFoundError(f"Training history not found: {training_path}")
    training = _read_csv(training_path)
    if not training:
        raise ValueError("Training history is empty.")
    validation = _read_csv(validation_path) if validation_path.is_file() else []
    latest = _read_json(artifact / "validation_latest.json")
    status = _read_json(artifact / "status.json")
    config = _read_json(artifact / "config.json")

    episodes = np.asarray([float(row["episodes"]) for row in training])
    episode_millions = episodes / 1.0e6
    successes = np.asarray([float(row["successes"]) for row in training])
    batch_success = 100.0 * np.asarray(
        [float(row["batch_success_rate"]) for row in training]
    )
    cumulative_success = 100.0 * np.asarray(
        [float(row["total_success_rate"]) for row in training]
    )
    if "rolling_success_rate" in training[0]:
        rolling_success = 100.0 * np.asarray(
            [float(row["rolling_success_rate"]) for row in training]
        )
        publication_rolling_window_episodes = int(
            float(training[-1]["rolling_window_episodes"])
        )
    elif "rolling_5000_success_rate" in training[0]:
        rolling_success = 100.0 * np.asarray(
            [float(row["rolling_5000_success_rate"]) for row in training]
        )
        publication_rolling_window_episodes = 5_000
    else:
        rolling_success = 100.0 * rolling_success_rate(
            episodes,
            successes,
            window_episodes=PUBLICATION_ROLLING_WINDOW_EPISODES,
        )
        publication_rolling_window_episodes = PUBLICATION_ROLLING_WINDOW_EPISODES
    validation_episodes = np.asarray(
        [float(row["checkpoint_episodes"]) for row in validation]
    )
    validation_rate = 100.0 * np.asarray(
        [float(row["validation_success_rate"]) for row in validation]
    )

    style = {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    figures: list[dict[str, Any]] = []
    with plt.rc_context(style):
        figure, axis = plt.subplots(figsize=(7.1, 3.75))
        axis.plot(
            episode_millions,
            batch_success,
            color="#93c5fd",
            linewidth=0.65,
            alpha=0.42,
            label="collection batch (2,048 episodes)",
        )
        axis.plot(
            episode_millions,
            rolling_success,
            color="#2563eb",
            linewidth=2.0,
            label=(
                f"rolling task success ({publication_rolling_window_episodes:,} episodes)"
            ),
        )
        axis.plot(
            episode_millions,
            cumulative_success,
            color="#64748b",
            linewidth=1.1,
            linestyle=":",
            label="cumulative task success",
        )
        if validation:
            axis.plot(
                validation_episodes / 1.0e6,
                validation_rate,
                color="#0f766e",
                marker="o",
                markersize=3.4,
                linewidth=1.5,
                label="deterministic state-bank validation",
            )
        axis.set(
            xlabel="Training episodes (millions)",
            ylabel="Task success (%)",
            ylim=(0.0, 104.0),
            title="PPO whip learning and held-out initial-state validation",
        )
        _finish_axis(axis)
        axis.legend(loc="lower right", frameon=False, ncol=2)
        figure.tight_layout()
        files = _save_figure(figure, output, "ppo_learning_curve")
        plt.close(figure)
        figures.append(
            {
                "id": "ppo_learning_curve",
                "files": files,
                "purpose": "Sample efficiency and state-bank generalization progression.",
            }
        )

        training_displacement = np.asarray(
            [float(row["mean_maximum_uav_displacement_m"]) for row in training]
        )
        displacement_window = min(5, len(training_displacement))
        smoothed_displacement = np.convolve(
            training_displacement,
            np.ones(displacement_window) / displacement_window,
            mode="valid",
        )
        displacement_x = episode_millions[displacement_window - 1 :]
        validation_displacement = np.asarray(
            [float(row["mean_maximum_uav_displacement_m"]) for row in validation]
        )
        figure, axes = plt.subplots(2, 1, figsize=(7.1, 5.2), sharex=True)
        axes[0].plot(
            validation_episodes / 1.0e6,
            validation_rate,
            color="#2563eb",
            marker="o",
            markersize=3.5,
            linewidth=1.6,
        )
        axes[0].set(ylabel="Validation success (%)", ylim=(0.0, 104.0))
        axes[0].set_title("PPO success–UAV-excursion trade-off")
        _finish_axis(axes[0])
        axes[1].plot(
            displacement_x,
            smoothed_displacement,
            color="#f59e0b",
            linewidth=1.7,
            label=f"training rolling mean ({displacement_window} batches)",
        )
        if validation:
            axes[1].plot(
                validation_episodes / 1.0e6,
                validation_displacement,
                color="#b45309",
                marker="o",
                markersize=3.5,
                linewidth=1.35,
                label="deterministic validation mean",
            )
        source_displacement = config.get("initialization", {}).get(
            "source_mean_maximum_uav_displacement_m"
        )
        if source_displacement is not None:
            axes[1].axhline(
                float(source_displacement),
                color="#64748b",
                linestyle="--",
                linewidth=1.0,
                label="continuation starting policy",
            )
        axes[1].set(
            xlabel="Training episodes (millions)",
            ylabel="Mean maximum UAV excursion (m)",
        )
        _finish_axis(axes[1])
        axes[1].legend(loc="best", frameon=False)
        figure.tight_layout()
        files = _save_figure(figure, output, "ppo_compactness_progression")
        plt.close(figure)
        figures.append(
            {
                "id": "ppo_compactness_progression",
                "files": files,
                "purpose": "Joint validation-success and UAV-excursion progression.",
            }
        )

        figure, axes = plt.subplots(2, 2, figsize=(7.1, 5.2))
        diagnostic_specs = (
            ("mean_episode_reward", "Mean episodic reward", None),
            ("entropy", "Policy entropy", None),
            ("approximate_kl", "Approximate KL", 0.02),
            ("clip_fraction", "PPO clip fraction", None),
        )
        for axis, (field, label, reference) in zip(axes.flat, diagnostic_specs):
            values = np.asarray([float(row[field]) for row in training])
            axis.plot(episode_millions, values, color="#475569", linewidth=1.15)
            if reference is not None:
                axis.axhline(
                    reference,
                    color="#dc2626",
                    linestyle="--",
                    linewidth=1.0,
                    label="configured target",
                )
                axis.legend(frameon=False)
            axis.set(xlabel="Episodes (millions)", ylabel=label)
            _finish_axis(axis)
        figure.suptitle("PPO optimization diagnostics", y=1.01)
        figure.tight_layout()
        files = _save_figure(figure, output, "ppo_optimization_diagnostics")
        plt.close(figure)
        figures.append(
            {
                "id": "ppo_optimization_diagnostics",
                "files": files,
                "purpose": "Supplementary optimization-health evidence.",
            }
        )

        strike_specs = (
            (
                "successful_tip_distance_m",
                "Tip error",
                1000.0,
                "mm",
                50.0,
                "≤ 50 mm",
            ),
            (
                "successful_directed_speed_m_s",
                "Directed tip speed",
                1.0,
                "m/s",
                4.0,
                "≥ 4 m/s",
            ),
            (
                "successful_direction_error_deg",
                "Direction error",
                1.0,
                "deg",
                30.0,
                "≤ 30 deg",
            ),
            (
                "successful_first_entry_time_s",
                "Strike time",
                1.0,
                "s",
                None,
                "diagnostic",
            ),
        )
        if latest and all(isinstance(latest.get(spec[0]), dict) for spec in strike_specs):
            figure, axes = plt.subplots(1, 4, figsize=(7.1, 2.7))
            for axis, (field, title, scale, unit, threshold, threshold_label) in zip(
                axes, strike_specs
            ):
                summary = latest[field]
                median = scale * float(summary["median"])
                minimum = scale * float(summary["minimum"])
                maximum = scale * float(summary["maximum"])
                axis.errorbar(
                    [0],
                    [median],
                    yerr=[[median - minimum], [maximum - median]],
                    color="#2563eb",
                    marker="o",
                    markersize=5,
                    capsize=4,
                    linewidth=1.4,
                )
                if threshold is not None:
                    axis.axhline(
                        threshold,
                        color="#dc2626",
                        linestyle="--",
                        linewidth=1.0,
                    )
                axis.set_xticks([])
                axis.set_title(title)
                axis.set_ylabel(unit)
                axis.text(
                    0.5,
                    -0.12,
                    threshold_label,
                    transform=axis.transAxes,
                    ha="center",
                    va="top",
                    color="#64748b",
                    fontsize=7,
                )
                _finish_axis(axis)
            checkpoint = int(latest.get("checkpoint_episodes", 0))
            figure.suptitle(
                f"Deterministic successful-strike distribution at {checkpoint:,} episodes",
                y=1.03,
            )
            figure.tight_layout()
            files = _save_figure(figure, output, "ppo_validation_strike_distribution")
            plt.close(figure)
            figures.append(
                {
                    "id": "ppo_validation_strike_distribution",
                    "files": files,
                    "purpose": "Median and observed range of successful physical strike metrics.",
                }
            )

    captions = """# Publication figure captions

1. **PPO learning curve.** Task-whip training success (raw batches, the configured rolling episode window, and cumulative success) together with deterministic validation on fixed physically propagated initial states. Success requires a tip-first 50-mm target entry, at least 4 m/s directed tip speed, and at most 30 degrees direction error. UAV numerical limits are not binary success gates.

2. **UAV-excursion trade-off.** Deterministic state-bank success and mean maximum UAV excursion from the initial position during compactness continuation. The dashed reference is the policy used to initialize continuation when available. Lower excursion is preferable; this panel deliberately exposes regressions rather than implying compactness improved.

3. **Optimization diagnostics.** Mean episodic reward, policy entropy, PPO approximate KL divergence, and clipping fraction. This is supplementary training-health evidence, not a task-performance result.

4. **Strike distribution.** Median and observed range across successful deterministic validation rollouts for tip error, target-directed tip speed, direction error, and strike time. Red dashed lines denote task thresholds; strike time is diagnostic only.

**Evaluation scope.** These figures describe nominal-physics simulation at the canonical target. Validation uses the run's fixed bank of mildly perturbed, physically propagated initial states. It is not target-generalization, broad state-distribution, hardware, or protected-test evidence. A figure generated before the run completes is marked `INTERIM_DO_NOT_PUBLISH` in `figure_manifest.json`.
"""
    (output / "FIGURE_CAPTIONS.md").write_text(captions, encoding="utf-8")
    sources = {
        path.name: _sha256(path)
        for path in (training_path, validation_path, artifact / "validation_latest.json")
        if path.is_file()
    }
    manifest = {
        "schema": "ppo_publication_figures_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifact": str(artifact),
        "run_status": status.get("status", "UNKNOWN"),
        "figure_status": (
            "FINAL" if status.get("status") == "COMPLETE" else "INTERIM_DO_NOT_PUBLISH"
        ),
        "latest_training_episodes": int(episodes[-1]),
        "rolling_window_episodes": publication_rolling_window_episodes,
        "validation_contexts": (
            int(validation[-1]["validation_episodes"]) if validation else 0
        ),
        "task_success_contract": {
            "tip_first": True,
            "tip_target_distance_m": 0.05,
            "minimum_directed_tip_speed_m_s": 4.0,
            "maximum_direction_error_deg": 30.0,
            "uav_limits_are_binary_success_gates": False,
        },
        "evaluation_scope": {
            "physics": "nominal frozen production simulator",
            "target": "canonical only",
            "initial_state_validation": "fixed mildly perturbed physically propagated state bank",
            "target_generalization_evaluated": False,
            "hardware_evaluated": False,
            "protected_test_evaluated": False,
        },
        "figures": figures,
        "source_sha256": sources,
    }
    (output / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
