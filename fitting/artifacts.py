"""Immutable fit snapshots and result artifact helpers."""

from __future__ import annotations

import csv
from pathlib import Path
import json

from experimental_data.io import atomic_json, canonical_json_hash, utc_now

from .dataset import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_ROOT / "data" / "fit_results"


def dataset_snapshot(dataset: Dataset, fit_config: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "aerial_cable_dataset_snapshot_v1",
        "created_utc": utc_now(),
        "manifest": dataset.manifest,
        "takes": {
            take.take_id: {
                "role": take.role,
                "enabled": take.enabled,
                "segments": list(take.segments),
                "episode_breaks_s": list(take.episode_breaks_s),
                "processed_take_sha256": take.metadata["processed_take_sha256"],
                "source_sha256": take.metadata["source_sha256"],
            }
            for take in dataset.takes
        },
        "fit_configuration": fit_config,
    }


def create_fit_directory(dataset: Dataset, fit_config: dict[str, object]) -> tuple[Path, dict[str, object]]:
    snapshot = dataset_snapshot(dataset, fit_config)
    fit_id = utc_now().replace(":", "").replace("+00:00", "Z") + "_" + canonical_json_hash(snapshot)[:8]
    directory = DEFAULT_RESULTS / fit_id
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory / "dataset_snapshot.json", snapshot)
    atomic_json(directory / "fit_config.json", fit_config)
    return directory, snapshot


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
