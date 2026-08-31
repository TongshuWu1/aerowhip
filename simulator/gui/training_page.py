"""PPO training control, status, and live train/validation curves."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from simulator.production import PROJECT_ROOT


CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "learning"
    / "whip_ppo_once_balanced_v1.json"
)
REWARD_STUDY_ROOTS = tuple(
    PROJECT_ROOT / "data" / "policy_training" / name
    for name in (
        "whip_ppo_reward_study_balanced_v1",
        "whip_ppo_reward_study_compact_v1",
        "whip_ppo_reward_study_strike_v1",
    )
)
VALIDATED_ROOT = (
    PROJECT_ROOT / "data" / "policy_training" / "simple_sequential_ppo_10s_validated_v1"
)
PILOT_ROOT = PROJECT_ROOT / "data" / "policy_training" / "simple_sequential_ppo_10s_v1"
ACTIVE_STATUSES = {"STARTING", "RESUMING", "RUNNING"}


def _utc_stamp() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H%M%S.") + f"{now.microsecond:06d}Z"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _latest_artifact() -> Path | None:
    candidates: list[Path] = []
    for root in (*REWARD_STUDY_ROOTS, VALIDATED_ROOT, PILOT_ROOT):
        if root.is_dir():
            candidates.extend(path for path in root.iterdir() if path.is_dir())
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


class TrainingCurves(FigureCanvasQTAgg):
    def __init__(self, parent: QWidget | None = None) -> None:
        self.figure = Figure(figsize=(9.5, 6.2), tight_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.training_axis, self.validation_axis = self.figure.subplots(2, 1, sharex=True)
        self.show_empty()

    def show_empty(self) -> None:
        for axis in (self.training_axis, self.validation_axis):
            axis.clear()
            axis.grid(True, alpha=0.22)
            axis.set_ylim(-2.0, 102.0)
        self.training_axis.set_ylabel("Training success (%)")
        self.validation_axis.set_ylabel("Validation success (%)")
        self.validation_axis.set_xlabel("Training episodes")
        self.training_axis.text(
            0.5,
            0.5,
            "No PPO run selected",
            transform=self.training_axis.transAxes,
            ha="center",
            va="center",
            color="#64748b",
        )
        self.draw_idle()

    def refresh_from(self, artifact: Path) -> None:
        training_path = artifact / "training_log.csv"
        validation_path = artifact / "validation_history.csv"
        try:
            with training_path.open(newline="", encoding="utf-8") as stream:
                training = list(csv.DictReader(stream))
        except (FileNotFoundError, OSError, ValueError):
            self.show_empty()
            return
        if not training:
            self.show_empty()
            return
        episodes = np.asarray([float(row["episodes"]) for row in training])
        cumulative = 100.0 * np.asarray(
            [float(row["total_success_rate"]) for row in training]
        )
        batch = 100.0 * np.asarray([float(row["batch_success_rate"]) for row in training])
        self.training_axis.clear()
        self.training_axis.plot(episodes, cumulative, linewidth=2.0, label="cumulative")
        self.training_axis.plot(
            episodes, batch, linewidth=1.0, alpha=0.45, label="latest 2,048 episodes"
        )
        if "total_endpoint_success_rate" in training[0]:
            endpoint = 100.0 * np.asarray(
                [float(row["total_endpoint_success_rate"]) for row in training]
            )
            self.training_axis.plot(
                episodes,
                endpoint,
                linewidth=1.5,
                linestyle="--",
                label="endpoint-only cumulative",
            )
        if "total_legacy_scientific_success_rate" in training[0]:
            legacy = 100.0 * np.asarray(
                [
                    float(row["total_legacy_scientific_success_rate"])
                    for row in training
                ]
            )
            self.training_axis.plot(
                episodes,
                legacy,
                linewidth=1.2,
                linestyle=":",
                label="legacy numerical-gate diagnostic",
            )
        self.training_axis.set_ylabel("Training success (%)")
        self.training_axis.set_ylim(-2.0, 102.0)
        self.training_axis.grid(True, alpha=0.22)
        self.training_axis.legend(loc="best")

        self.validation_axis.clear()
        validation: list[dict[str, str]] = []
        try:
            with validation_path.open(newline="", encoding="utf-8") as stream:
                validation = list(csv.DictReader(stream))
        except (FileNotFoundError, OSError, ValueError):
            pass
        if validation:
            validation_episodes = np.asarray(
                [float(row["checkpoint_episodes"]) for row in validation]
            )
            validation_rate = 100.0 * np.asarray(
                [float(row["validation_success_rate"]) for row in validation]
            )
            self.validation_axis.plot(
                validation_episodes,
                validation_rate,
                "o-",
                linewidth=2.0,
                label="10 fixed mild starts",
            )
            if "validation_endpoint_success_rate" in validation[0]:
                endpoint_rate = 100.0 * np.asarray(
                    [float(row["validation_endpoint_success_rate"]) for row in validation]
                )
                self.validation_axis.plot(
                    validation_episodes,
                    endpoint_rate,
                    "o--",
                    linewidth=1.5,
                    label="endpoint-only",
                )
            if "validation_legacy_scientific_success_rate" in validation[0]:
                legacy_rate = 100.0 * np.asarray(
                    [
                        float(row["validation_legacy_scientific_success_rate"])
                        for row in validation
                    ]
                )
                self.validation_axis.plot(
                    validation_episodes,
                    legacy_rate,
                    "o:",
                    linewidth=1.2,
                    label="legacy numerical gates",
                )
            self.validation_axis.legend(loc="best")
        else:
            self.validation_axis.text(
                0.5,
                0.5,
                "Validation begins at 51,200 training episodes",
                transform=self.validation_axis.transAxes,
                ha="center",
                va="center",
                color="#64748b",
            )
        self.validation_axis.set_ylabel("Validation success (%)")
        self.validation_axis.set_xlabel("Training episodes")
        self.validation_axis.set_ylim(-2.0, 102.0)
        self.validation_axis.grid(True, alpha=0.22)
        self.draw_idle()


class TrainingPage(QWidget):
    """Read/control the single authorized validated PPO training process."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("trainingPage")
        self.setStyleSheet("#trainingPage { background: #f8fafc; }")
        self.current_artifact: Path | None = _latest_artifact()
        self._build_ui()
        self.timer = QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 16)
        title = QLabel("PPO TRAINING")
        title.setObjectName("pageTitle")
        title.setStyleSheet("font-size: 22px; font-weight: 800; color: white;")
        layout.addWidget(title)
        contract = QGroupBox("Fixed experiment contract")
        form = QFormLayout(contract)
        form.addRow("Learner", QLabel("Clipped PPO + GAE, random initialization"))
        form.addRow("Training start", QLabel("Canonical settled state"))
        form.addRow("Success", QLabel("Single first entry: c10, 50 mm, 4 m/s, 30 deg; no time gate"))
        form.addRow("UAV limits", QLabel("Continuous reward costs + diagnostics; not binary failure gates"))
        form.addRow("Validation", QLabel("10 deterministic rollouts from fixed mild propagated states"))
        form.addRow("Validation cadence", QLabel("Every 51,200 training episodes"))
        form.addRow("Physics", QLabel("Production UAV + causal residual + 12-node DDER"))
        form.addRow("Horizon / action", QLabel("10.0 s / acceleration + roll, pitch, yaw rates"))
        layout.addWidget(contract)

        controls = QHBoxLayout()
        self.start_button = QPushButton("START NEW PPO RUN")
        self.stop_button = QPushButton("STOP AND CHECKPOINT")
        self.resume_button = QPushButton("RESUME")
        self.folder_button = QPushButton("OPEN RUN FOLDER")
        self.start_button.clicked.connect(self.start_new)
        self.stop_button.clicked.connect(self.request_stop)
        self.resume_button.clicked.connect(self.resume)
        self.folder_button.clicked.connect(self.open_folder)
        for button in (
            self.start_button,
            self.stop_button,
            self.resume_button,
            self.folder_button,
        ):
            controls.addWidget(button)
        layout.addLayout(controls)

        status_group = QGroupBox("Live status")
        status_form = QFormLayout(status_group)
        self.status_value = QLabel("NO RUN")
        self.episodes_value = QLabel("—")
        self.training_success_value = QLabel("—")
        self.endpoint_success_value = QLabel("—")
        self.validation_success_value = QLabel("—")
        self.speed_value = QLabel("—")
        self.run_path_value = QLabel("—")
        self.run_path_value.setWordWrap(True)
        status_form.addRow("Status", self.status_value)
        status_form.addRow("Episodes", self.episodes_value)
        status_form.addRow("Task-whip success", self.training_success_value)
        status_form.addRow("Endpoint-only success", self.endpoint_success_value)
        status_form.addRow("Validation success", self.validation_success_value)
        status_form.addRow("Throughput", self.speed_value)
        status_form.addRow("Artifact", self.run_path_value)
        layout.addWidget(status_group)
        self.curves = TrainingCurves(self)
        layout.addWidget(self.curves, 1)

    def _status(self) -> dict[str, Any] | None:
        return None if self.current_artifact is None else _read_json(
            self.current_artifact / "status.json"
        )

    def refresh(self) -> None:
        if self.current_artifact is None:
            self.current_artifact = _latest_artifact()
        status = self._status()
        if self.current_artifact is None:
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.resume_button.setEnabled(False)
            self.folder_button.setEnabled(False)
            self.curves.show_empty()
            return
        self.run_path_value.setText(str(self.current_artifact))
        self.folder_button.setEnabled(True)
        if status is None:
            state = "LAUNCHING"
            episodes = 0
            requested = 1_000_000
            success_rate = 0.0
            speed = 0.0
        else:
            state = str(status.get("status", "UNKNOWN"))
            episodes = int(status.get("episodes", 0))
            requested = int(status.get("requested_episodes", 1_000_000))
            success_rate = float(status.get("success_rate", 0.0))
            speed = float(status.get("episodes_per_second", 0.0))
        active = state in ACTIVE_STATUSES or state == "LAUNCHING"
        self.status_value.setText(state)
        self.status_value.setStyleSheet(
            "font-size: 18px; font-weight: 800; color: "
            + ("#15803d;" if active else "#475569;")
        )
        self.episodes_value.setText(f"{episodes:,} / {requested:,}")
        self.training_success_value.setText(f"{100.0 * success_rate:.2f}% cumulative")
        endpoint_rate = 0.0 if status is None else float(
            status.get("endpoint_success_rate", success_rate)
        )
        self.endpoint_success_value.setText(f"{100.0 * endpoint_rate:.2f}% cumulative")
        self.speed_value.setText(f"{speed:.1f} episodes/s" if speed else "—")
        validation = _read_json(self.current_artifact / "validation_latest.json")
        if validation is None:
            self.validation_success_value.setText("Not run yet")
        else:
            self.validation_success_value.setText(
                f"{int(validation['validation_successes'])}/"
                f"{int(validation['validation_episodes'])} "
                f"task, {int(validation.get('validation_endpoint_successes', validation['validation_successes']))}/"
                f"{int(validation['validation_episodes'])} endpoint at "
                f"{int(validation['checkpoint_episodes']):,} episodes"
            )
        self.start_button.setEnabled(not active)
        self.stop_button.setEnabled(active)
        self.resume_button.setEnabled(
            state == "STOPPED_BY_USER"
            and (self.current_artifact / "checkpoints" / "latest.pt").is_file()
            and (self.current_artifact / "config.json").is_file()
        )
        self.curves.refresh_from(self.current_artifact)

    @staticmethod
    def _launch(arguments: list[str], *, cwd: Path) -> None:
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        subprocess.Popen(
            [sys.executable, *arguments],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
            close_fds=True,
        )

    def start_new(self) -> None:
        status = self._status()
        if status is not None and str(status.get("status")) in ACTIVE_STATUSES:
            QMessageBox.warning(self, "PPO already running", "Stop the active run first.")
            return
        artifact = REWARD_STUDY_ROOTS[0] / _utc_stamp()
        artifact.parent.mkdir(parents=True, exist_ok=True)
        self._launch(
            [
                str(PROJECT_ROOT / "run_simple_ppo.py"),
                "--train",
                "--config",
                str(CONFIG_PATH),
                "--artifact-directory",
                str(artifact),
            ],
            cwd=PROJECT_ROOT,
        )
        self.current_artifact = artifact
        self.refresh()

    def request_stop(self) -> None:
        if self.current_artifact is None:
            return
        (self.current_artifact / "STOP_REQUESTED").touch()
        self.status_value.setText("STOP REQUESTED — checkpointing after current batch")

    def resume(self) -> None:
        if self.current_artifact is None:
            return
        self._launch(
            [
                str(PROJECT_ROOT / "run_simple_ppo.py"),
                "--resume",
                "--artifact-directory",
                str(self.current_artifact),
            ],
            cwd=PROJECT_ROOT,
        )
        self.refresh()

    def open_folder(self) -> None:
        if self.current_artifact is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_artifact)))
