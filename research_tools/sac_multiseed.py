from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev

from cable_twin.shared.observation_data import sha256_file
from drone_mpc.sac import SAC_CHECKPOINT_SCHEMA, is_supported_sac_checkpoint_schema


SUMMARY_SCHEMA = "drone_whip_sac_multiseed_summary_v1"
METRICS = (
    "success_rate",
    "within_initial_reach_success_rate",
    "beyond_initial_reach_success_rate",
    "mean_minimum_error_m",
    "mean_directional_speed_m_s",
    "mean_hit_drone_displacement_m",
)


def _success_count(rate: float, episodes: int, name: str) -> int:
    successes = int(round(rate * episodes))
    if not 0 <= successes <= episodes or not math.isclose(
        rate,
        successes / episodes,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(f"{name} is not an exact success fraction.")
    return successes


def _load_run(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    source = path.resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not is_supported_sac_checkpoint_schema(payload.get("schema")):
        raise ValueError(f"Unsupported SAC log schema: {source}")
    evaluation = payload.get("evaluation")
    settings = payload.get("settings")
    task = payload.get("task")
    if not isinstance(evaluation, dict) or not isinstance(settings, dict) or not isinstance(task, dict):
        raise ValueError(f"Incomplete SAC log: {source}")
    policy = source.with_suffix(".pt")
    if not policy.is_file():
        raise FileNotFoundError(f"Selected SAC policy is missing: {policy}")
    episodes = int(evaluation["episodes"])
    within_episodes = int(evaluation["within_initial_reach_episodes"])
    beyond_episodes = int(evaluation["beyond_initial_reach_episodes"])
    if episodes < 1 or within_episodes + beyond_episodes != episodes:
        raise ValueError(f"Invalid held-out episode counts in {source}")
    values = {name: float(evaluation[name]) for name in METRICS}
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError(f"Non-finite held-out metric in {source}")
    run = {
        "training_seed": int(settings["seed"]),
        "checkpoint_validation_seed": int(payload["checkpoint_validation_seed"]),
        "final_test_seed": int(payload["final_test_seed"]),
        "log_path": str(source),
        "policy_path": str(policy),
        "policy_sha256": sha256_file(policy),
        "best_transitions": int(payload["best_transitions"]),
        "overall_success": {
            "successes": _success_count(values["success_rate"], episodes, "overall success"),
            "episodes": episodes,
            "rate": values["success_rate"],
        },
        "within_initial_reach": {
            "successes": _success_count(
                values["within_initial_reach_success_rate"],
                within_episodes,
                "within-reach success",
            ),
            "episodes": within_episodes,
            "rate": values["within_initial_reach_success_rate"],
        },
        "beyond_initial_reach": {
            "successes": _success_count(
                values["beyond_initial_reach_success_rate"],
                beyond_episodes,
                "beyond-reach success",
            ),
            "episodes": beyond_episodes,
            "rate": values["beyond_initial_reach_success_rate"],
        },
        "mean_minimum_error_m": values["mean_minimum_error_m"],
        "mean_directional_speed_m_s": values["mean_directional_speed_m_s"],
        "mean_hit_drone_displacement_m": values[
            "mean_hit_drone_displacement_m"
        ],
    }
    invariant = {
        "schema": payload["schema"],
        "source_model_sha256": payload["source_model_sha256"],
        "controller_model_sha256": payload["controller_model_sha256"],
        "evaluation_model_sha256": payload["evaluation_model_sha256"],
        "source_model_provisional": payload["source_model_provisional"],
        "algorithm": payload["algorithm"],
        "task_distribution": payload["task_distribution"],
        "prior_replay": payload["prior_replay"],
        "task": task,
        "settings_except_seed": deepcopy(settings),
    }
    invariant["settings_except_seed"].pop("seed", None)
    return run, invariant


def _aggregate_scalar(runs: list[dict[str, object]], name: str) -> dict[str, float]:
    values = [float(run[name]) for run in runs]
    return {"mean": mean(values), "sample_std": stdev(values)}


def _aggregate_success(
    runs: list[dict[str, object]], name: str
) -> dict[str, float | int]:
    blocks = [run[name] for run in runs]
    rates = [float(block["rate"]) for block in blocks]
    successes = sum(int(block["successes"]) for block in blocks)
    episodes = sum(int(block["episodes"]) for block in blocks)
    return {
        "pooled_successes": successes,
        "pooled_episodes": episodes,
        "pooled_rate": successes / episodes,
        "rate_mean": mean(rates),
        "rate_sample_std": stdev(rates),
    }


def build_summary(log_paths: list[str | Path] | tuple[str | Path, ...]) -> dict[str, object]:
    if len(log_paths) < 2:
        raise ValueError("A multi-seed summary requires at least two SAC logs.")
    loaded = [_load_run(Path(path)) for path in log_paths]
    runs = [item[0] for item in loaded]
    reference = loaded[0][1]
    if any(invariant != reference for _, invariant in loaded[1:]):
        raise ValueError("SAC logs do not share the same model, task, and settings.")
    training_seeds = [int(run["training_seed"]) for run in runs]
    if len(set(training_seeds)) != len(training_seeds):
        raise ValueError("SAC training seeds must be distinct.")
    validation_seeds = {int(run["checkpoint_validation_seed"]) for run in runs}
    test_seeds = {int(run["final_test_seed"]) for run in runs}
    if len(validation_seeds) != 1 or len(test_seeds) != 1:
        raise ValueError("SAC logs were not evaluated on common target seeds.")
    episode_strata = {
        (
            int(run["overall_success"]["episodes"]),
            int(run["within_initial_reach"]["episodes"]),
            int(run["beyond_initial_reach"]["episodes"]),
        )
        for run in runs
    }
    if len(episode_strata) != 1:
        raise ValueError("SAC logs do not contain the same held-out target strata.")
    runs.sort(key=lambda run: int(run["training_seed"]))
    return {
        "schema": SUMMARY_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_seed_count": len(runs),
        "checkpoint_validation_seed": next(iter(validation_seeds)),
        "final_test_seed": next(iter(test_seeds)),
        "invariants": reference,
        "runs": runs,
        "aggregate": {
            "overall_success": _aggregate_success(runs, "overall_success"),
            "within_initial_reach": _aggregate_success(
                runs, "within_initial_reach"
            ),
            "beyond_initial_reach": _aggregate_success(
                runs, "beyond_initial_reach"
            ),
            "mean_minimum_error_m": _aggregate_scalar(
                runs, "mean_minimum_error_m"
            ),
            "mean_directional_speed_m_s": _aggregate_scalar(
                runs, "mean_directional_speed_m_s"
            ),
            "mean_hit_drone_displacement_m": _aggregate_scalar(
                runs, "mean_hit_drone_displacement_m"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize SAC policies evaluated on one fixed target set."
    )
    parser.add_argument("--log", type=Path, action="append", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/drone_mpc/sac_policy_goals_demo12_multiseed.json"),
    )
    arguments = parser.parse_args()
    summary = build_summary(arguments.log)
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, output)
    overall = summary["aggregate"]["overall_success"]
    within = summary["aggregate"]["within_initial_reach"]
    beyond = summary["aggregate"]["beyond_initial_reach"]
    print(
        f"Saved {output} | overall={100.0 * overall['rate_mean']:.1f}% "
        f"+/- {100.0 * overall['rate_sample_std']:.1f}%, "
        f"within={100.0 * within['rate_mean']:.1f}%, "
        f"beyond={100.0 * beyond['rate_mean']:.1f}%"
    )


if __name__ == "__main__":
    main()
