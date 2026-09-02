"""Read-only discovery and loading of completed planning artifacts.

This module is the boundary between the GUI/replay tools and planner output.
It never runs MPPI or physics.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from simulator.production import PROJECT_ROOT


PLANNING_RESULTS_ROOT = PROJECT_ROOT / "results" / "cem" / "replays"
SUPPORTED_TASKS = {
    "canonical_whip_v1": "Single Target Whip",
    "figure8_endpoint_whip_v1": "Figure-8 Endpoint Whip",
    "canonical_whip_variable_duration_v1": "Variable-Duration Single Target Whip",
    "canonical_whip_variable_duration_legacy_reward_v1": "Variable-Duration Whip — Legacy Reward",
    "canonical_whip_variable_duration_tuned_reward_v1": "Variable-Duration Whip — Tuned Reward",
    "figure8_endpoint_whip_variable_duration_v1": "Variable-Duration Figure-8 Endpoint Whip",
}


@dataclass(frozen=True, slots=True)
class PlanningResult:
    task_id: str
    task_label: str
    directory: Path
    task_config: dict[str, Any]
    metrics: dict[str, Any]
    iteration_history: tuple[dict[str, Any], ...]

    @property
    def replay_path(self) -> Path:
        return self.directory / "final_replay.npz"

    @property
    def success(self) -> bool:
        classification = self.metrics.get("task_classification")
        if classification is not None:
            return classification == "PASS"
        return bool(self.metrics.get("success", False))

    @property
    def status(self) -> str:
        return str(
            self.metrics.get(
                "task_classification", "PASS" if self.success else "FAIL"
            )
        )

    @property
    def video_path(self) -> Path:
        return self.directory / f"{self.task_id}_final_replay.mp4"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_planning_result(directory: str | Path) -> PlanningResult:
    path = Path(directory).resolve()
    required = ("task_config_snapshot.json", "final_metrics.json", "final_replay.npz")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete planning result {path}: {missing}")
    task = _load_json(path / "task_config_snapshot.json")
    task_id = str(task["task_id"])
    return PlanningResult(
        task_id=task_id,
        task_label=SUPPORTED_TASKS.get(task_id, task_id),
        directory=path,
        task_config=task,
        metrics=_load_json(path / "final_metrics.json"),
        iteration_history=tuple(
            _load_json(
                path
                / (
                    "cem_iteration_history.json"
                    if (path / "cem_iteration_history.json").is_file()
                    else "mppi_iteration_history.json"
                )
            )
        ),
    )


def list_planning_results(task_id: str) -> tuple[PlanningResult, ...]:
    root = PLANNING_RESULTS_ROOT / task_id
    if not root.is_dir():
        return ()
    results: list[PlanningResult] = []
    for path in sorted((item for item in root.iterdir() if item.is_dir()), reverse=True):
        try:
            results.append(load_planning_result(path))
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
            continue
    return tuple(results)


def latest_planning_result(task_id: str) -> PlanningResult | None:
    results = list_planning_results(task_id)
    return results[0] if results else None


def load_replay_arrays(result: PlanningResult) -> dict[str, np.ndarray]:
    """Load a saved deterministic replay without invoking the simulator."""

    with np.load(result.replay_path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}
