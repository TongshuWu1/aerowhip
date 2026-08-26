"""Pure workflow and artifact catalog used by the research console.

This module deliberately imports neither Tk nor Torch.  It is safe to inspect
from tests and from the lightweight launcher process.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable


REPOSITORY_DIRECTORY = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class WorkflowDefinition:
    workflow_id: str
    stage: int
    title: str
    short_title: str
    objective: str
    method: str
    inputs: tuple[str, ...]
    output: str
    script_name: str
    required_artifacts: tuple[str, ...] = ()
    primary: bool = True


@dataclass(frozen=True, slots=True)
class LaunchSpec:
    workflow_id: str
    command: tuple[str, ...]
    cwd: Path


@dataclass(frozen=True, slots=True)
class ArtifactStatus:
    artifact_id: str
    label: str
    path: Path
    exists: bool
    kind: str
    item_count: int
    size_bytes: int
    modified_text: str
    sha256: str | None
    schema: str | None
    compatibility: str | None = None

    @property
    def short_hash(self) -> str:
        return "--" if self.sha256 is None else self.sha256[:10]

    @property
    def status_label(self) -> str:
        """A research-safe inventory label, not a claim of pipeline readiness."""

        if not self.exists:
            return "MISSING"
        if self.compatibility == "compatible":
            return "COMPATIBLE"
        if self.compatibility == "mismatch":
            return "MISMATCH"
        if self.artifact_id == "cable_model" and self.schema == "optitrack_twist_aware_rod_v5":
            return "PROVISIONAL"
        return "AVAILABLE"


WORKFLOWS = (
    WorkflowDefinition(
        workflow_id="identify",
        stage=1,
        title="Physical model identification",
        short_title="Identification",
        objective="Estimate a reproducible one-attached cable model from OptiTrack trajectories.",
        method="Review takes, assign train/validation roles, fit EI and Cb, then validate a free-tip rollout.",
        inputs=("Ordered Motive CSV takes", "Cable geometry and moving masses"),
        output="Canonical fitted cable-model JSON with fit and validation provenance",
        script_name="run_offline_fitting.py",
    ),
    WorkflowDefinition(
        workflow_id="online",
        stage=2,
        title="Online receding-horizon MPC",
        short_title="Online MPC",
        objective=(
            "Evaluate receding-horizon cable-strike control and simulation-only "
            "between-strike EI/Cb adaptation."
        ),
        method=(
            "Exact distributed-state feedback, accelerated DDER propagation, "
            "CUDA-batched MPPI, short-prefix execution, shifted warm starts, and "
            "optional held-out-validated EI/Cb publication between strikes."
        ),
        inputs=("Nominal cable model", "Whip target and impact direction"),
        output=(
            "Replayable MPPI trial with timing, distributed trajectory, contact, "
            "model-generation, and adaptation diagnostics"
        ),
        script_name="run_online.py",
        required_artifacts=("cable_model",),
    ),
)


_ARTIFACT_PATHS = {
    "optitrack_csv": (
        "OptiTrack CSV collection",
        REPOSITORY_DIRECTORY / "optitrack_offline" / "csv",
    ),
    "cable_model": (
        "Nominal cable model",
        REPOSITORY_DIRECTORY / "optitrack_offline" / "models" / "cable_model.json",
    ),
    "online_results": (
        "DDER-MPPI experiment results",
        REPOSITORY_DIRECTORY / "data" / "drone_mpc" / "receding_mppi",
    ),
}


def workflow_by_id(workflow_id: str) -> WorkflowDefinition:
    for workflow in WORKFLOWS:
        if workflow.workflow_id == workflow_id:
            return workflow
    raise KeyError(f"Unknown research workflow: {workflow_id}")


def build_launch_spec(workflow_id: str) -> LaunchSpec:
    workflow = workflow_by_id(workflow_id)
    script = (REPOSITORY_DIRECTORY / workflow.script_name).resolve()
    if not script.is_file():
        raise FileNotFoundError(f"Workflow launcher is missing: {script}")
    return LaunchSpec(
        workflow_id=workflow_id,
        command=(str(Path(sys.executable).resolve()), str(script)),
        cwd=REPOSITORY_DIRECTORY.resolve(),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_files(paths: Iterable[Path], suffixes: tuple[str, ...]) -> list[Path]:
    result: list[Path] = []
    for directory in paths:
        if directory.is_dir():
            result.extend(
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in suffixes
            )
    return sorted(result, key=lambda path: str(path).lower())


def _directory_digest(files: Iterable[Path]) -> str | None:
    paths = tuple(files)
    if not paths:
        return None
    digest = hashlib.sha256()
    for path in paths:
        try:
            identity = path.resolve().relative_to(REPOSITORY_DIRECTORY.resolve())
        except ValueError:
            identity = path.resolve()
        digest.update(identity.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _schema(path: Path) -> str | None:
    if path.suffix.lower() != ".json":
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "invalid JSON"
    value = payload.get("schema") if isinstance(payload, dict) else None
    return str(value) if value is not None else "schema missing"


def _time_text(timestamp: float | None) -> str:
    if timestamp is None:
        return "--"
    return datetime.fromtimestamp(timestamp).astimezone().strftime("%Y-%m-%d %H:%M")


def _status_for_path(artifact_id: str, label: str, path: Path) -> ArtifactStatus:
    if path.is_file():
        stat = path.stat()
        return ArtifactStatus(
            artifact_id=artifact_id,
            label=label,
            path=path.resolve(),
            exists=True,
            kind="file",
            item_count=1,
            size_bytes=stat.st_size,
            modified_text=_time_text(stat.st_mtime),
            sha256=_sha256_file(path),
            schema=_schema(path),
        )
    files = _directory_files((path,), (".csv", ".npz", ".json"))
    return ArtifactStatus(
        artifact_id=artifact_id,
        label=label,
        path=path.resolve(),
        exists=bool(files),
        kind="directory",
        item_count=len(files),
        size_bytes=sum(item.stat().st_size for item in files),
        modified_text=_time_text(max((item.stat().st_mtime for item in files), default=None)),
        sha256=_directory_digest(files),
        schema=None,
    )


def inspect_artifacts() -> tuple[ArtifactStatus, ...]:
    return tuple(
        _status_for_path(artifact_id, label, path)
        for artifact_id, (label, path) in _ARTIFACT_PATHS.items()
    )
