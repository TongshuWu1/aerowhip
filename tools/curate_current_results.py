"""Build the small, durable PPO/SAC/CEM result bundle kept in the repository.

This intentionally copies only the checkpoints, logs, plots, summaries, and
minimal shared inputs needed to understand or reproduce the currently useful
methods.  Full historical run directories are archived outside the repository.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ARCHIVE = ROOT.parent / "particle_filter_cable_project_archive_2026-08-31"


def available(relative: str) -> Path:
    """Resolve a source from either the live tree or the dated history archive."""
    live = ROOT / relative
    if live.exists():
        return live
    archived = ARCHIVE / relative
    if archived.exists():
        return archived
    raise FileNotFoundError(f"Missing curated source in repository and archive: {relative}")

PPO_RUN = available(
    "data/policy_training/whip_ppo_directional_displacement15_v1/"
    "2026-08-31T104735.951178Z"
)
PPO_COMPILER = available(
    "data/policy_training/ppo_open_loop_compiler_v1/2026-08-31T162836.928988Z"
)
SAC_RUN = available(
    "data/policy_training/simple_sequential_sac_10s_v1/2026-08-30T224351.697920Z"
)
COMMON_RUN = available(
    "data/policy_training/oneshot_sac_nominal_v1/2026-08-29T192059.907360Z"
)
CEM_RUN = available(
    "data/planning/production_cem_benchmark_v1/2026-08-30T072758.266436Z"
)
CEM_REFERENCE = available(
    "data/planning_results/canonical_whip_variable_duration_tuned_reward_v1/"
    "2026-08-29T160423.856049Z"
)
CEM_TASK_WARM_START = available(
    "data/planning_results/canonical_whip_variable_duration_tuned_reward_v1/"
    "2026-08-29T153444.863451Z/best_acceleration_knots.json"
)
CEM_GUI_REPLAY = available(
    "data/planning_results/canonical_whip_v1/2026-08-29T060719.197533Z"
)


def copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_named(source_root: Path, destination_root: Path, names: list[str]) -> None:
    for name in names:
        copy(source_root / name, destination_root / name)


def plot_cem(rows_path: Path, plot_root: Path) -> None:
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["group"])].append(row)

    labels = list(groups)
    first = [100.0 * np.mean([r["first_seed_success"] for r in groups[g]]) for g in labels]
    restart = [
        100.0 * np.mean([r["success_with_restarts"] for r in groups[g]]) for g in labels
    ]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width / 2, first, width, label="First seed")
    ax.bar(x + width / 2, restart, width, label="Up to 3 seeds")
    ax.set_ylabel("Authoritative scientific success (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_root / "benchmark_success_by_group.png", dpi=180)
    plt.close(fig)

    runtimes = np.asarray([r["planning_runtime_s"] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(runtimes, bins=28, color="#4c78a8", alpha=0.9)
    ax.axvline(np.median(runtimes), color="#e45756", linestyle="--", label="Median")
    ax.axvline(np.percentile(runtimes, 95), color="#f2cf5b", linestyle="--", label="P95")
    ax.set_xlabel("Planning time per context (s)")
    ax.set_ylabel("Contexts")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_root / "planning_runtime_distribution.png", dpi=180)
    plt.close(fig)


def main() -> None:
    for directory in (
        RESULTS / "common",
        RESULTS / "ppo/plots",
        RESULTS / "ppo/data",
        RESULTS / "ppo/checkpoints",
        RESULTS / "ppo/compiler_audit",
        RESULTS / "ppo/reports",
        RESULTS / "sac/plots",
        RESULTS / "sac/data",
        RESULTS / "sac/checkpoints",
        RESULTS / "cem/plots",
        RESULTS / "cem/data",
        RESULTS / "cem/reference",
        RESULTS / "cem/reports",
        RESULTS / "cem/replays/canonical_whip_v1/current",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    copy_named(
        COMMON_RUN,
        RESULTS / "common",
        [
            "context_normalizer.json",
            "context_schema.json",
            "training_state_bank.npz",
            "training_state_bank_manifest.json",
            "validation_state_bank.npz",
            "validation_state_bank_manifest.json",
        ],
    )

    copy_named(
        PPO_RUN,
        RESULTS / "ppo/data",
        [
            "best_validation.json",
            "config.json",
            "final_summary.json",
            "RUN_CONTRACT.md",
            "source_hash_manifest.json",
            "status.json",
            "training_log.csv",
            "validation_history.csv",
            "validation_latest.json",
            "validation_state_manifest.json",
        ],
    )
    for name in (
        "success_rate_vs_episodes.png",
        "training_success_vs_episodes.png",
        "validation_success_vs_episodes.png",
    ):
        copy(PPO_RUN / name, RESULTS / "ppo/plots" / name)
    copy(PPO_RUN / "checkpoints/latest.pt", RESULTS / "ppo/checkpoints/terminal.pt")
    copy(
        PPO_RUN / "checkpoints/best_validation.pt",
        RESULTS / "ppo/checkpoints/best_validation.pt",
    )
    copy_named(
        PPO_COMPILER,
        RESULTS / "ppo/compiler_audit",
        [
            "architecture_decision.json",
            "compilation_latency.json",
            "config.json",
            "exact_replay_verification.json",
            "final_summary.json",
            "model_mismatch_results.json",
            "nominal_mode_comparison.json",
            "nominal_state_rows.json",
            "post_planning_disturbance_results.json",
            "source_hash_manifest.json",
        ],
    )
    copy(
        available(
            "repository_history/retired_files/PPO_WHIP_COMPLETE_TECHNICAL_REPORT.md"
        ),
        RESULTS / "ppo/reports/PPO_WHIP_COMPLETE_TECHNICAL_REPORT.md",
    )
    copy(
        available(
            "repository_history/retired_files/"
            "PPO_FEEDBACK_DEPENDENCE_AND_OPEN_LOOP_COMPILER_REPORT.md"
        ),
        RESULTS / "ppo/reports/PPO_FEEDBACK_DEPENDENCE_AND_OPEN_LOOP_COMPILER_REPORT.md",
    )

    copy_named(
        SAC_RUN,
        RESULTS / "sac/data",
        [
            "config.json",
            "RUN_CONTRACT.md",
            "source_hash_manifest.json",
            "status.json",
            "stopped_run_summary.json",
            "training_log.csv",
        ],
    )
    copy(
        SAC_RUN / "figures/01_success_and_throughput.png",
        RESULTS / "sac/plots/success_and_throughput.png",
    )
    copy(
        SAC_RUN / "figures/02_learning_diagnostics.png",
        RESULTS / "sac/plots/learning_diagnostics.png",
    )
    copy(SAC_RUN / "checkpoints/latest.pt", RESULTS / "sac/checkpoints/last_checkpoint.pt")

    cem_data_files = [
        "action_codec_audit.json",
        "authoritative_actions.npz",
        "benchmark_config.json",
        "benchmark_context_manifest.json",
        "benchmark_rows.json",
        "benchmark_summary.json",
        "canonical_cem_config.json",
        "canonical_seed_results.json",
        "duration_hit_segment_summary.json",
        "replay_consistency_diagnostic.json",
        "replay_difference_summary.json",
        "robustness_margin_summary.json",
        "runtime_summary.json",
        "source_hash_manifest.json",
        "terminal_settle_contract.json",
        "terminal_settle_verification.json",
    ]
    copy_named(CEM_RUN, RESULTS / "cem/data", cem_data_files)
    copy(
        available(
            "repository_history/retired_files/"
            "MILESTONE6A_PRODUCTION_CEM_BENCHMARK_REPORT.md"
        ),
        RESULTS / "cem/reports/MILESTONE6A_PRODUCTION_CEM_BENCHMARK_REPORT.md",
    )
    copy_named(
        CEM_REFERENCE,
        RESULTS / "cem/reference",
        ["best_acceleration_knots.json", "optimized_duration.json"],
    )
    copy(
        CEM_TASK_WARM_START,
        RESULTS / "cem/reference/tuned_reward_initial_nominal_knots.json",
    )
    for source in CEM_GUI_REPLAY.iterdir():
        if source.is_file():
            copy(source, RESULTS / "cem/replays/canonical_whip_v1/current" / source.name)
    replay_root = RESULTS / "cem/replays/canonical_whip_v1/current"
    video_metadata_path = replay_root / "video_metadata.json"
    if video_metadata_path.is_file():
        video_metadata = json.loads(video_metadata_path.read_text(encoding="utf-8"))
        video_metadata["source_replay_artifact"] = str(replay_root / "final_replay.npz")
        video_metadata["video_path"] = str(replay_root / "canonical_whip_v1_final_replay.mp4")
        video_metadata_path.write_text(
            json.dumps(video_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    plot_cem(RESULTS / "cem/data/benchmark_rows.json", RESULTS / "cem/plots")

    print(f"Curated current results at {RESULTS}")


if __name__ == "__main__":
    main()
