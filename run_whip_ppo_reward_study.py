"""Run the fixed three-variant PPO whip reward study sequentially."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parent
CONFIGS = (
    ("balanced", ROOT / "config" / "learning" / "whip_ppo_once_balanced_v1.json"),
    ("compact", ROOT / "config" / "learning" / "whip_ppo_once_compact_v1.json"),
    ("strike", ROOT / "config" / "learning" / "whip_ppo_once_strike_v1.json"),
)


def _stamp() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H%M%S.") + f"{now.microsecond:06d}Z"


def _read(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _write(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(suite: Path) -> None:
    suite.mkdir(parents=True, exist_ok=False)
    variants: list[dict[str, str]] = []
    for name, config_path in CONFIGS:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        artifact = (
            ROOT
            / "data"
            / "policy_training"
            / str(config["experiment_id"])
            / suite.name
        )
        variants.append(
            {"name": name, "config": str(config_path), "artifact": str(artifact)}
        )
    manifest = {
        "schema": "ppo_whip_reward_study_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "variants": variants,
        "new_cem_solves": 0,
        "protected_test": "NOT EVALUATED",
        "hardware": "NOT EXECUTED",
    }
    _write(suite / "study_manifest.json", manifest)
    started = time.perf_counter()
    for index, variant in enumerate(variants, start=1):
        if (suite / "STOP_REQUESTED").exists():
            _write(suite / "status.json", {"status": "STOPPED_BY_USER", "completed_variants": index - 1})
            return
        artifact = Path(variant["artifact"])
        stdout_path = suite / f"{index:02d}_{variant['name']}_stdout.log"
        stderr_path = suite / f"{index:02d}_{variant['name']}_stderr.log"
        command = [
            sys.executable,
            str(ROOT / "run_simple_ppo.py"),
            "--config",
            variant["config"],
            "--artifact-directory",
            str(artifact),
            "--train",
        ]
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
            while process.poll() is None:
                child = _read(artifact / "status.json") or {}
                _write(
                    suite / "status.json",
                    {
                        "status": "RUNNING",
                        "active_variant": variant["name"],
                        "variant_index": index,
                        "variant_count": len(variants),
                        "artifact": str(artifact),
                        "episodes": int(child.get("episodes", 0)),
                        "requested_episodes": int(child.get("requested_episodes", 1_000_000)),
                        "success_rate": float(child.get("success_rate", 0.0)),
                        "episodes_per_second": float(child.get("episodes_per_second", 0.0)),
                        "elapsed_s": time.perf_counter() - started,
                    },
                )
                if (suite / "STOP_REQUESTED").exists() and artifact.is_dir():
                    (artifact / "STOP_REQUESTED").touch()
                time.sleep(10.0)
        if process.returncode != 0:
            _write(
                suite / "status.json",
                {
                    "status": "FAILED",
                    "failed_variant": variant["name"],
                    "return_code": process.returncode,
                    "stderr": str(stderr_path),
                },
            )
            return
        child = _read(artifact / "status.json") or {}
        if child.get("status") != "COMPLETE":
            _write(
                suite / "status.json",
                {"status": "STOPPED_BY_USER", "active_variant": variant["name"]},
            )
            return
    audit_stdout = suite / "audit_stdout.log"
    audit_stderr = suite / "audit_stderr.log"
    with audit_stdout.open("w", encoding="utf-8") as stdout, audit_stderr.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "audit_whip_ppo_reward_study.py"),
                "--suite-directory",
                str(suite),
            ],
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    _write(
        suite / "status.json",
        {
            "status": "COMPLETE" if completed.returncode == 0 else "AUDIT_FAILED",
            "completed_variants": len(variants),
            "elapsed_s": time.perf_counter() - started,
            "final_audit": str(suite / "final_audit.json"),
            "audit_return_code": completed.returncode,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-directory", type=Path)
    arguments = parser.parse_args()
    suite = (
        arguments.suite_directory.resolve()
        if arguments.suite_directory is not None
        else ROOT / "data" / "policy_training" / "whip_ppo_reward_study_v1" / _stamp()
    )
    run(suite)


if __name__ == "__main__":
    main()
