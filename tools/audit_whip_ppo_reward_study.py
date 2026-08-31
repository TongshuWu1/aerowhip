"""Audit completed PPO whip reward variants using fixed state-bank validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learning.ppo_validation import FixedMildStateValidationPanel
from run_simple_ppo import _build_agent


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@torch.no_grad()
def _broad_validation(artifact: Path) -> dict[str, Any]:
    config = _read_json(artifact / "config.json")
    config["validation"] = dict(config["validation"])
    config["validation"].update(
        {
            "episodes": 64,
            "state_selection_seed": 1742,
            "maximum_distance_quantile": 1.0,
        }
    )
    panel = FixedMildStateValidationPanel(config)
    device = panel.environment.simulator.device
    agent = _build_agent(config, device)
    best_path = artifact / "checkpoints" / "best_validation.pt"
    checkpoint_path = (
        best_path if best_path.is_file() else artifact / "checkpoints" / "latest.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent.policy.load_state_dict(checkpoint["policy"])
    if "value" in checkpoint:
        agent.value.load_state_dict(checkpoint["value"])
    best = _read_json(artifact / "best_validation.json")
    result = panel.evaluate(
        agent, checkpoint_episodes=int(best["checkpoint_episodes"])
    )
    result["checkpoint"] = str(checkpoint_path)
    result["validation_scope"] = "64 fixed physically propagated states across the complete existing bank"
    result["state_manifest"] = panel.manifest
    return result


def _plot(suite: Path, variants: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(10.0, 11.0), sharex=True)
    for variant in variants:
        name = variant["name"]
        artifact = Path(variant["artifact"])
        training = _read_csv(artifact / "training_log.csv")
        validation = _read_csv(artifact / "validation_history.csv")
        x = np.asarray([float(row["episodes"]) for row in training])
        axes[0].plot(
            x,
            100.0 * np.asarray([float(row["batch_success_rate"]) for row in training]),
            alpha=0.55,
            label=name,
        )
        vx = np.asarray([float(row["checkpoint_episodes"]) for row in validation])
        axes[1].plot(
            vx,
            100.0
            * np.asarray([float(row["validation_success_rate"]) for row in validation]),
            "o-",
            label=name,
        )
        axes[2].plot(
            vx,
            np.asarray(
                [float(row["mean_maximum_uav_displacement_m"]) for row in validation]
            ),
            "o-",
            label=name,
        )
    axes[0].set_ylabel("Batch task success (%)")
    axes[1].set_ylabel("Fixed validation success (%)")
    axes[2].set_ylabel("Validation max displacement (m)")
    axes[2].set_xlabel("Training episodes")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(suite / "reward_study_comparison.png", dpi=170)
    plt.close(figure)


def audit(suite: Path) -> dict[str, Any]:
    manifest = _read_json(suite / "study_manifest.json")
    rows: list[dict[str, Any]] = []
    for variant in manifest["variants"]:
        artifact = Path(variant["artifact"])
        status = _read_json(artifact / "status.json")
        validation = _read_csv(artifact / "validation_history.csv")
        best = _read_json(artifact / "best_validation.json")
        broad = _broad_validation(artifact)
        _atomic_json(artifact / "final_broad_validation.json", broad)
        rows.append(
            {
                **variant,
                "status": status["status"],
                "episodes": status["episodes"],
                "training_success_rate": status["success_rate"],
                "best_fixed_validation": best,
                "last_fixed_validation": validation[-1],
                "broad_validation": broad,
            }
        )
    _plot(suite, rows)
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["broad_validation"]["validation_success_rate"]),
            -float(row["broad_validation"]["mean_maximum_uav_displacement_m"]),
            float(
                row["broad_validation"][
                    "validation_legacy_scientific_success_rate"
                ]
            ),
        ),
        reverse=True,
    )
    selected = ranked[0]
    result = {
        "schema": "ppo_whip_reward_study_audit_v1",
        "new_cem_solves": 0,
        "success_definition": "single first target entry is c10 and satisfies distance/speed/direction; no time gate",
        "episode_horizon_physics_steps": 1000,
        "numerical_uav_limits": "continuous reward costs and legacy diagnostics, not task failure gates",
        "variants": rows,
        "selected_variant": selected["name"],
        "selection_rule": "highest 64-state task success, then lowest displacement, then legacy diagnostic success",
    }
    _atomic_json(suite / "final_audit.json", result)
    lines = [
        "# PPO whip reward study — final audit",
        "",
        "The study kept the full production UAV, causal residual, and 12-node DDER model. "
        "Episodes always ran for 1,000 physics steps (10 s), including after a strike.",
        "",
        "Task success was a single c10-first target entry at ≤50 mm, "
        "≥4 m/s directed speed, and ≤30° direction error. UAV displacement, UAV speed, "
        "and command acceleration were optimized continuously and reported separately; "
        "they were not binary task-failure gates.",
        "",
        "| Reward | Episodes | Best 10-state validation | 64-state validation | Mean max displacement | Legacy numerical-gate diagnostic |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        broad = row["broad_validation"]
        best = row["best_fixed_validation"]
        lines.append(
            f"| {row['name']} | {int(row['episodes']):,} | "
            f"{100*float(best['validation_success_rate']):.1f}% | "
            f"{100*float(broad['validation_success_rate']):.1f}% | "
            f"{float(broad['mean_maximum_uav_displacement_m']):.3f} m | "
            f"{100*float(broad['validation_legacy_scientific_success_rate']):.1f}% |"
        )
    lines.extend(
        [
            "",
            f"Selected reward: **{selected['name']}**.",
            "",
            "The selected policy is a simulation result only. Protected data and hardware were not used.",
        ]
    )
    (suite / "PPO_WHIP_REWARD_STUDY_FINAL_AUDIT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-directory", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit(arguments.suite_directory.resolve())
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
