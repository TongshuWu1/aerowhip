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
    QCheckBox,
    QFormLayout,
    QDoubleSpinBox,
    QFileDialog,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from simulator.production import PROJECT_ROOT
from learning.ppo_validation import rolling_episode_mean
from .theme import MetricCard, set_status_badge


CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "learning"
    / "whip_ppo_dense_return_release_100_continuation_v1.json"
)
ACTIVE_RUN_ROOT = (
    PROJECT_ROOT
    / "data"
    / "policy_training"
    / "whip_ppo_dense_return_release_100_continuation_v1"
)
CURATED_RUN = PROJECT_ROOT / "results" / "ppo" / "data"
ACTIVE_STATUSES = {"STARTING", "RESUMING", "RUNNING"}


def _utc_stamp() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H%M%S.") + f"{now.microsecond:06d}Z"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _latest_artifact() -> Path | None:
    candidates: list[Path] = [CURATED_RUN] if (CURATED_RUN / "status.json").is_file() else []
    if ACTIVE_RUN_ROOT.is_dir():
        candidates.extend(path for path in ACTIVE_RUN_ROOT.iterdir() if path.is_dir())
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


class TrainingCurves(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("trainingPlotTabs")
        layout.addWidget(self.tabs)

        (
            self.training_figure,
            self.training_canvas,
            self.training_axis,
        ) = self._add_plot_tab("TRAINING SUCCESS")
        (
            self.reward_figure,
            self.reward_canvas,
            self.reward_axis,
        ) = self._add_plot_tab("EPISODE REWARD")
        (
            self.validation_figure,
            self.validation_canvas,
            self.validation_axis,
        ) = self._add_plot_tab("VALIDATION")
        self._last_source_signature: tuple[object, ...] | None = None
        self.show_empty()

    def _add_plot_tab(
        self, label: str
    ) -> tuple[Figure, FigureCanvasQTAgg, object]:
        figure = Figure(figsize=(10.5, 4.8), tight_layout=True)
        figure.set_facecolor("#ffffff")
        canvas = FigureCanvasQTAgg(figure)
        axis = figure.subplots(1, 1)
        self.tabs.addTab(canvas, label)
        return figure, canvas, axis

    def _draw_all(self) -> None:
        self.training_canvas.draw_idle()
        self.reward_canvas.draw_idle()
        self.validation_canvas.draw_idle()

    def show_empty(self) -> None:
        for axis in (self.training_axis, self.reward_axis, self.validation_axis):
            axis.clear()
            axis.grid(True, alpha=0.22)
        self.training_axis.set_ylim(-2.0, 102.0)
        self.validation_axis.set_ylim(-2.0, 102.0)
        self.training_axis.set_ylabel("Training success (%)")
        self.training_axis.set_xlabel("Training episodes")
        self.reward_axis.set_ylabel("Rolling reward")
        self.reward_axis.set_xlabel("Training episodes")
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
        self._draw_all()

    def refresh_from(self, artifact: Path) -> None:
        training_path = artifact / "training_log.csv"
        validation_path = artifact / "validation_history.csv"
        manual_validation_path = artifact / "manual_validation_history.csv"

        def file_signature(path: Path) -> tuple[int, int] | None:
            try:
                stat = path.stat()
            except OSError:
                return None
            return stat.st_mtime_ns, stat.st_size

        signature: tuple[object, ...] = (
            str(artifact.resolve()),
            file_signature(training_path),
            file_signature(validation_path),
            file_signature(manual_validation_path),
        )
        if signature == self._last_source_signature:
            return
        self._last_source_signature = signature
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
        if "rolling_success_rate" in training[0]:
            rolling = 100.0 * np.asarray(
                [float(row["rolling_success_rate"]) for row in training]
            )
            rolling_window = int(float(training[-1]["rolling_window_episodes"]))
        elif "rolling_5000_success_rate" in training[0]:
            rolling = 100.0 * np.asarray(
                [float(row["rolling_5000_success_rate"]) for row in training]
            )
            rolling_window = 5_000
        else:
            successes = np.asarray([float(row["successes"]) for row in training])
            from learning.ppo_validation import rolling_success_rate

            rolling = 100.0 * rolling_success_rate(
                episodes, successes, window_episodes=5_000
            )
            rolling_window = 5_000
        self.training_axis.clear()
        self.training_axis.plot(
            episodes,
            rolling,
            color="#2563eb",
            linewidth=2.2,
            label=f"rolling {rolling_window:,} episodes",
        )
        self.training_axis.plot(
            episodes,
            cumulative,
            color="#64748b",
            linewidth=1.3,
            linestyle=":",
            label="cumulative",
        )
        self.training_axis.set_ylabel("Training success (%)")
        self.training_axis.set_xlabel("Training episodes")
        self.training_axis.set_ylim(-2.0, 102.0)
        self.training_axis.grid(True, alpha=0.22)
        self.training_axis.legend(loc="best")

        self.reward_axis.clear()
        reward_available = "mean_episode_reward" in training[0] and all(
            row.get("mean_episode_reward", "") != "" for row in training
        )
        if reward_available:
            batch_reward = np.asarray(
                [float(row["mean_episode_reward"]) for row in training]
            )
            rolling_reward = rolling_episode_mean(
                episodes,
                batch_reward,
                window_episodes=rolling_window,
            )
            self.reward_axis.plot(
                episodes,
                batch_reward,
                color="#94a3b8",
                linewidth=0.9,
                alpha=0.5,
                label="batch mean",
            )
            self.reward_axis.plot(
                episodes,
                rolling_reward,
                color="#d97706",
                linewidth=2.1,
                label=f"rolling {rolling_window:,} episodes",
            )
            self.reward_axis.legend(loc="best")
        else:
            self.reward_axis.text(
                0.5,
                0.5,
                "Episode reward was not logged for this run",
                transform=self.reward_axis.transAxes,
                ha="center",
                va="center",
                color="#64748b",
            )
        self.reward_axis.set_ylabel("Mean episodic reward")
        self.reward_axis.set_xlabel("Training episodes")
        self.reward_axis.grid(True, alpha=0.22)

        self.validation_axis.clear()
        validation: list[dict[str, str]] = []
        try:
            with validation_path.open(newline="", encoding="utf-8") as stream:
                validation = list(csv.DictReader(stream))
        except (FileNotFoundError, OSError, ValueError):
            pass
        manual_validation: list[dict[str, str]] = []
        try:
            with manual_validation_path.open(newline="", encoding="utf-8") as stream:
                manual_validation = list(csv.DictReader(stream))
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
                color="#2563eb",
                linewidth=2.0,
                label="task success on fixed state bank",
            )
        if manual_validation:
            manual_episodes = np.asarray(
                [float(row["checkpoint_episodes"]) for row in manual_validation]
            )
            manual_successes = np.asarray(
                [float(row["validation_successes"]) for row in manual_validation]
            )
            manual_counts = np.asarray(
                [float(row["validation_episodes"]) for row in manual_validation]
            )
            manual_rates = 100.0 * manual_successes / np.maximum(manual_counts, 1.0)
            z = 1.959963984540054
            denominator = 1.0 + z * z / manual_counts
            center = (
                manual_rates / 100.0 + z * z / (2.0 * manual_counts)
            ) / denominator
            radius = (
                z
                * np.sqrt(
                    (manual_rates / 100.0)
                    * (1.0 - manual_rates / 100.0)
                    / manual_counts
                    + z * z / (4.0 * manual_counts * manual_counts)
                )
                / denominator
            )
            low = 100.0 * np.maximum(0.0, center - radius)
            high = 100.0 * np.minimum(1.0, center + radius)
            self.validation_axis.errorbar(
                manual_episodes,
                manual_rates,
                yerr=(
                    np.maximum(0.0, manual_rates - low),
                    np.maximum(0.0, high - manual_rates),
                ),
                fmt="s",
                color="#f59e0b",
                markeredgecolor="#b45309",
                capsize=3,
                markersize=5,
                linewidth=1.2,
                label="manual current-policy check (95% Wilson CI)",
            )
            self.validation_axis.annotate(
                f"n={int(manual_counts[-1])}",
                (manual_episodes[-1], manual_rates[-1]),
                xytext=(5, -12),
                textcoords="offset points",
                fontsize=7,
                color="#92400e",
            )
        if validation or manual_validation:
            self.validation_axis.legend(loc="best")
        else:
            self.validation_axis.text(
                0.5,
                0.5,
                "Validation appears after the first scheduled checkpoint",
                transform=self.validation_axis.transAxes,
                ha="center",
                va="center",
                color="#64748b",
            )
        self.validation_axis.set_ylabel("Validation success (%)")
        self.validation_axis.set_xlabel("Training episodes")
        self.validation_axis.set_ylim(-2.0, 102.0)
        self.validation_axis.grid(True, alpha=0.22)
        self._draw_all()


class TrainingPage(QWidget):
    """Read/control the single authorized validated PPO training process."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("trainingPage")
        self.setStyleSheet("#trainingPage { background: #f8fafc; }")
        self.current_artifact: Path | None = _latest_artifact()
        self._manual_validation_process: subprocess.Popen[bytes] | None = None
        self._build_ui()
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 18)
        layout.setSpacing(10)
        base_config = _read_json(CONFIG_PATH) or {}
        self._configuration_template = json.loads(json.dumps(base_config))
        reward_defaults = base_config.get("reward", {})
        ppo_defaults = base_config.get("ppo", {})
        validation_defaults = base_config.get("validation", {})
        logging_defaults = base_config.get("logging", {})
        initial_state_defaults = base_config.get(
            "training_initial_states", {"mode": "canonical"}
        )
        early_stopping_defaults = base_config.get("early_stopping", {})

        toolbar = QFrame()
        toolbar.setObjectName("toolbarCard")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(13, 9, 11, 9)
        run_heading = QVBoxLayout()
        run_heading.setSpacing(0)
        run_label = QLabel("ACTIVE EXPERIMENT")
        run_label.setStyleSheet(
            "color: #64748b; font-size: 8pt; font-weight: 800; letter-spacing: 0.8px;"
        )
        run_name = QLabel("Target-aligned prepare-and-strike PPO")
        run_name.setStyleSheet("font-weight: 750; color: #1e293b;")
        run_heading.addWidget(run_label)
        run_heading.addWidget(run_name)
        toolbar_layout.addLayout(run_heading)
        toolbar_layout.addStretch(1)
        self.status_value = QLabel("NO RUN")
        set_status_badge(self.status_value, "NO RUN", "neutral")
        toolbar_layout.addWidget(self.status_value)
        self.start_button = QPushButton("START NEW PPO RUN")
        self.start_button.setObjectName("primaryButton")
        self.stop_button = QPushButton("STOP AND CHECKPOINT")
        self.stop_button.setObjectName("dangerButton")
        self.resume_button = QPushButton("RESUME")
        self.export_button = QPushButton("EXPORT FIGURES")
        self.folder_button = QPushButton("OPEN FOLDER")
        self.start_button.clicked.connect(self.start_new)
        self.stop_button.clicked.connect(self.request_stop)
        self.resume_button.clicked.connect(self.resume)
        self.export_button.clicked.connect(self.export_figures)
        self.folder_button.clicked.connect(self.open_folder)
        for button in (
            self.start_button,
            self.stop_button,
            self.resume_button,
            self.export_button,
            self.folder_button,
        ):
            toolbar_layout.addWidget(button)
        layout.addWidget(toolbar)

        self.episode_progress = QProgressBar()
        self.episode_progress.setRange(0, 10_000)
        self.episode_progress.setTextVisible(False)
        layout.addWidget(self.episode_progress)

        metrics = QHBoxLayout()
        metrics.setSpacing(9)
        self.progress_card = MetricCard("Episodes", "—", "requested training budget")
        self.training_card = MetricCard("Training success", "—", "latest collection batch")
        self.validation_card = MetricCard("Validation", "—", "fixed propagated states")
        self.displacement_card = MetricCard("UAV excursion", "—", "validation mean maximum")
        self.throughput_card = MetricCard("Throughput", "—", "estimated time remaining")
        for card in (
            self.progress_card,
            self.training_card,
            self.validation_card,
            self.displacement_card,
            self.throughput_card,
        ):
            metrics.addWidget(card, 1)
        layout.addLayout(metrics)

        self.configuration_group = QGroupBox("Run configuration — click to edit")
        self.configuration_group.setCheckable(True)
        self.configuration_group.setChecked(False)
        configuration_group_layout = QVBoxLayout(self.configuration_group)
        preset_bar = QHBoxLayout()
        preset_bar.addWidget(QLabel("Configuration preset"))
        self.loaded_config_label = QLabel(CONFIG_PATH.name)
        self.loaded_config_label.setStyleSheet("color: #64748b;")
        preset_bar.addWidget(self.loaded_config_label)
        preset_bar.addStretch(1)
        self.load_config_button = QPushButton("LOAD CONFIG")
        self.load_config_button.setObjectName("loadPpoConfigButton")
        self.save_config_button = QPushButton("SAVE CONFIG")
        self.save_config_button.setObjectName("savePpoConfigButton")
        self.load_config_button.clicked.connect(self.load_configuration)
        self.save_config_button.clicked.connect(self.save_configuration)
        preset_bar.addWidget(self.load_config_button)
        preset_bar.addWidget(self.save_config_button)
        configuration_group_layout.addLayout(preset_bar)
        self.configuration_content = QWidget()
        configuration_layout = QHBoxLayout(self.configuration_content)
        configuration_layout.setContentsMargins(2, 2, 2, 2)
        configuration_layout.setSpacing(10)
        self.config_inputs: list[QWidget] = []

        def integer_input(
            value: int, minimum: int, maximum: int, step: int = 1
        ) -> QSpinBox:
            control = QSpinBox()
            control.setRange(minimum, maximum)
            control.setSingleStep(step)
            control.setValue(value)
            control.setGroupSeparatorShown(True)
            self.config_inputs.append(control)
            return control

        def decimal_input(
            value: float,
            minimum: float,
            maximum: float,
            step: float,
            decimals: int = 3,
            suffix: str = "",
        ) -> QDoubleSpinBox:
            control = QDoubleSpinBox()
            control.setRange(minimum, maximum)
            control.setSingleStep(step)
            control.setDecimals(decimals)
            control.setValue(value)
            if suffix:
                control.setSuffix(suffix)
            self.config_inputs.append(control)
            return control

        run_group = QGroupBox("Run")
        run_form = QFormLayout(run_group)
        self.initialization_mode_input = QComboBox()
        self.initialization_mode_input.setObjectName("ppoInitializationModeInput")
        self.initialization_mode_input.addItem(
            "Continue from selected checkpoint", "continue"
        )
        self.initialization_mode_input.addItem("Fresh random policy", "fresh")
        if not bool(
            base_config.get("initialization", {}).get(
                "uses_previous_policy_checkpoint", False
            )
        ):
            self.initialization_mode_input.setCurrentIndex(1)
        self.initialization_mode_input.setToolTip(
            "Fresh starts policy, value network, and optimizer from random initialization."
        )
        self.config_inputs.append(self.initialization_mode_input)
        self.action_mode_input = QComboBox()
        self.action_mode_input.setObjectName("ppoActionModeInput")
        self.action_mode_input.addItem(
            "Target-aligned sagittal (3-D)", "target_aligned_sagittal_3d"
        )
        self.action_mode_input.addItem("Unrestricted acceleration/rates (6-D)", "full_6d")
        configured_action_mode = str(
            base_config.get("action", {}).get("mode", "full_6d")
        )
        action_mode_index = self.action_mode_input.findData(configured_action_mode)
        self.action_mode_input.setCurrentIndex(max(action_mode_index, 0))
        self.action_mode_input.setToolTip(
            "Target-aligned mode removes lateral acceleration and outputs forward/back, "
            "vertical acceleration, and pitch rate. Changing action width requires fresh training."
        )
        self.config_inputs.append(self.action_mode_input)
        self.training_state_mode_input = QComboBox()
        self.training_state_mode_input.setObjectName("ppoTrainingStateModeInput")
        self.training_state_mode_input.addItem(
            "Mixed physical TRAIN bank", "mixed_state_bank"
        )
        self.training_state_mode_input.addItem("Canonical state only", "canonical")
        training_state_mode = str(
            initial_state_defaults.get("mode", "canonical")
        )
        training_state_index = self.training_state_mode_input.findData(
            training_state_mode
        )
        self.training_state_mode_input.setCurrentIndex(max(training_state_index, 0))
        self.training_state_mode_input.setToolTip(
            "Mixed mode samples only the existing physically propagated TRAIN bank; "
            "the disjoint VALIDATION bank is never used for gradients."
        )
        self.config_inputs.append(self.training_state_mode_input)
        self.canonical_fraction_input = decimal_input(
            100.0 * float(initial_state_defaults.get("canonical_fraction", 0.0)),
            0.0,
            100.0,
            5.0,
            1,
            " %",
        )
        self.canonical_fraction_input.setObjectName("ppoCanonicalFractionInput")
        self.canonical_fraction_input.setToolTip(
            "Exact fraction of each 2,048-episode training batch initialized from "
            "the settled canonical state; remaining rows come from the TRAIN bank."
        )
        self.episode_budget_input = integer_input(
            int(base_config.get("requested_episodes", 1_000_000)),
            2_048,
            20_000_000,
            2_048,
        )
        self.episode_budget_input.setObjectName("ppoEpisodeBudgetInput")
        self.episode_horizon_input = decimal_input(
            float(base_config.get("episode_duration_s", 7.0)),
            0.5,
            20.0,
            0.1,
            1,
            " s",
        )
        self.episode_horizon_input.setObjectName("ppoEpisodeHorizonInput")
        self.validation_interval_input = integer_input(
            int(validation_defaults.get("every_episodes", 10_240)),
            2_048,
            2_000_000,
            2_048,
        )
        self.scheduled_validation_count_input = integer_input(
            int(validation_defaults.get("episodes", 64)), 10, 512, 1
        )
        self.rolling_window_input = integer_input(
            int(
                logging_defaults.get(
                    "training_success_rolling_window_episodes", 5_000
                )
            ),
            100,
            2_000_000,
            100,
        )
        self.rolling_window_input.setObjectName("ppoRollingSuccessWindowInput")
        self.seed_input = integer_input(int(base_config.get("seed", 444)), 0, 2_147_483_647)
        self.early_stopping_input = QCheckBox("Use held-out validation early stopping")
        self.early_stopping_input.setChecked(
            bool(early_stopping_defaults.get("enabled", False))
        )
        self.early_stopping_input.setToolTip(
            "Stops after the configured number of scheduled validation checks without "
            "improvement. The best deterministic validation checkpoint is retained."
        )
        self.config_inputs.append(self.early_stopping_input)
        self.early_stopping_minimum_input = integer_input(
            int(early_stopping_defaults.get("minimum_episodes", 250_000)),
            0,
            20_000_000,
            2_048,
        )
        self.early_stopping_patience_input = integer_input(
            int(
                early_stopping_defaults.get(
                    "validation_patience_evaluations", 30
                )
            ),
            1,
            500,
            1,
        )
        run_form.addRow("Initialization", self.initialization_mode_input)
        run_form.addRow("Policy action", self.action_mode_input)
        run_form.addRow("Training states", self.training_state_mode_input)
        run_form.addRow("Canonical fraction", self.canonical_fraction_input)
        run_form.addRow("Training episodes", self.episode_budget_input)
        run_form.addRow("Timeout", self.episode_horizon_input)
        run_form.addRow("Validate every", self.validation_interval_input)
        run_form.addRow("Validation trials", self.scheduled_validation_count_input)
        run_form.addRow("Success rolling window", self.rolling_window_input)
        run_form.addRow(self.early_stopping_input)
        run_form.addRow("Early-stop minimum", self.early_stopping_minimum_input)
        run_form.addRow("Validation patience", self.early_stopping_patience_input)
        run_form.addRow("Seed", self.seed_input)
        configuration_layout.addWidget(run_group, 1)

        task_reward_group = QGroupBox("Positive success reward")
        task_reward_form = QFormLayout(task_reward_group)
        self.progress_weight_input = decimal_input(
            float(reward_defaults.get("progress_weight", 20.0)), 0.0, 500.0, 1.0, 2
        )
        self.progress_reference_input = QComboBox()
        self.progress_reference_input.setObjectName("ppoProgressReferenceInput")
        self.progress_reference_input.addItem(
            "Cable span from initial attachment",
            "attachment_compensated_tip",
        )
        self.progress_reference_input.addItem(
            "Blend world and cable-span progress",
            "blended_world_attachment",
        )
        self.progress_reference_input.addItem("World-frame cable tip", "world_tip")
        configured_progress_reference = str(
            reward_defaults.get("progress_shaping_reference", "world_tip")
        )
        progress_reference_index = self.progress_reference_input.findData(
            configured_progress_reference
        )
        self.progress_reference_input.setCurrentIndex(
            max(progress_reference_index, 0)
        )
        self.progress_reference_input.setToolTip(
            "Attachment-compensated progress credits cable-span motion, not carrying "
            "the whole UAV/cable assembly toward the target. Real success remains world-frame."
        )
        self.config_inputs.append(self.progress_reference_input)
        self.progress_attachment_fraction_input = decimal_input(
            float(
                reward_defaults.get(
                    "progress_attachment_compensation_fraction", 0.5
                )
            ),
            0.0,
            1.0,
            0.05,
            2,
        )
        self.progress_attachment_fraction_input.setToolTip(
            "For blended progress: 0 is entirely world-frame and 1 is entirely "
            "attachment-compensated cable-span progress."
        )
        self.speed_reward_weight_input = decimal_input(
            float(reward_defaults.get("directed_speed_near_target_weight", 15.0)),
            0.0,
            500.0,
            1.0,
            2,
        )
        self.speed_cap_input = decimal_input(
            float(reward_defaults.get("directed_speed_reward_cap_m_s", 4.0)),
            0.5,
            20.0,
            0.5,
            2,
            " m/s",
        )
        self.speed_cap_input.setToolTip(
            "Speed shaping becomes flat here; the success threshold remains 4.0 m/s."
        )
        self.speed_reference_input = QComboBox()
        self.speed_reference_input.setObjectName("ppoSpeedReferenceInput")
        self.speed_reference_input.addItem(
            "Cable relative to attachment", "attachment_relative"
        )
        self.speed_reference_input.addItem("World-frame cable tip", "world_tip")
        configured_speed_reference = str(
            reward_defaults.get("directed_speed_shaping_reference", "world_tip")
        )
        speed_reference_index = self.speed_reference_input.findData(
            configured_speed_reference
        )
        self.speed_reference_input.setCurrentIndex(max(speed_reference_index, 0))
        self.speed_reference_input.setToolTip(
            "Attachment-relative shaping rewards cable swing rather than translating the UAV "
            "and cable together. Scientific success always uses world-frame tip velocity."
        )
        self.config_inputs.append(self.speed_reference_input)
        self.direction_reward_weight_input = decimal_input(
            float(reward_defaults.get("direction_near_target_weight", 40.0)),
            0.0,
            500.0,
            1.0,
            2,
        )
        self.direction_reward_weight_input.setToolTip(
            "Near-target alignment reward centered on the scientific 30-degree gate."
        )
        self.strike_weight_input = decimal_input(
            float(reward_defaults.get("strike_quality_improvement_weight", 30.0)),
            0.0,
            500.0,
            1.0,
            2,
        )
        self.forward_return_bonus_input = decimal_input(
            float(reward_defaults.get("success_forward_return_bonus_weight", 0.0)),
            0.0,
            500.0,
            5.0,
            2,
        )
        self.forward_return_bonus_input.setToolTip(
            "Paid only on a successful strike after genuine forward loading and "
            "subsequent target-axis return."
        )
        self.release_bonus_input = decimal_input(
            float(reward_defaults.get("success_release_bonus_weight", 0.0)),
            0.0,
            500.0,
            5.0,
            2,
        )
        self.release_bonus_input.setToolTip(
            "Paid only on success for a retreating UAV while the cable tip moves "
            "forward relative to its attachment."
        )
        self.return_release_improvement_input = decimal_input(
            float(reward_defaults.get("return_release_improvement_weight", 0.0)),
            0.0,
            500.0,
            5.0,
            2,
        )
        self.return_release_improvement_input.setToolTip(
            "Dense potential improvement for forward loading followed by UAV return "
            "while the cable moves toward a viable strike."
        )
        self.release_at_strike_input = QCheckBox("Credit release at strike only")
        self.release_at_strike_input.setChecked(
            bool(reward_defaults.get("success_release_at_strike", False))
        )
        self.release_at_strike_input.setToolTip(
            "When enabled, the success release bonus uses the release state at impact "
            "instead of the best release observed earlier in the episode."
        )
        self.config_inputs.extend(
            [self.return_release_improvement_input, self.release_at_strike_input]
        )
        self.success_bonus_input = decimal_input(
            float(reward_defaults.get("success_bonus", 100.0)), 0.0, 1000.0, 5.0, 2
        )
        self.non_tip_penalty_input = decimal_input(
            float(reward_defaults.get("non_tip_first_penalty", 25.0)),
            0.0,
            500.0,
            1.0,
            2,
        )
        task_reward_form.addRow("Progress", self.progress_weight_input)
        task_reward_form.addRow("Progress reference", self.progress_reference_input)
        task_reward_form.addRow(
            "Attachment progress fraction",
            self.progress_attachment_fraction_input,
        )
        task_reward_form.addRow("Directed speed", self.speed_reward_weight_input)
        task_reward_form.addRow("Speed cap", self.speed_cap_input)
        task_reward_form.addRow("Speed reference", self.speed_reference_input)
        task_reward_form.addRow("Direction alignment", self.direction_reward_weight_input)
        task_reward_form.addRow("Joint strike", self.strike_weight_input)
        task_reward_form.addRow("Forward-return bonus", self.forward_return_bonus_input)
        task_reward_form.addRow("Cable-release bonus", self.release_bonus_input)
        task_reward_form.addRow(
            "Dense return-release", self.return_release_improvement_input
        )
        task_reward_form.addRow("Release timing", self.release_at_strike_input)
        task_reward_form.addRow("Success bonus", self.success_bonus_input)
        task_reward_form.addRow("Non-tip-first", self.non_tip_penalty_input)
        configuration_layout.addWidget(task_reward_group, 1)

        motion_reward_group = QGroupBox("Negative reward terms")
        motion_reward_form = QFormLayout(motion_reward_group)
        self.terminal_displacement_weight_input = decimal_input(
            float(reward_defaults.get("terminal_displacement_weight", 40.0)),
            0.0,
            500.0,
            1.0,
            2,
        )
        self.terminal_displacement_success_only_input = QCheckBox(
            "Charge only after a successful strike"
        )
        self.terminal_displacement_success_only_input.setChecked(
            bool(reward_defaults.get("terminal_displacement_success_only", False))
        )
        self.terminal_displacement_success_only_input.setToolTip(
            "Prevents failed timeouts from favoring a stationary policy. Compactness "
            "becomes a secondary preference among successful strikes."
        )
        self.config_inputs.append(self.terminal_displacement_success_only_input)
        self.displacement_integral_weight_input = decimal_input(
            float(reward_defaults.get("displacement_integral_weight", 3.0)),
            0.0,
            200.0,
            0.5,
            2,
        )
        self.displacement_scale_input = decimal_input(
            float(reward_defaults.get("displacement_cost_scale_m", 0.35)),
            0.01,
            5.0,
            0.05,
            3,
            " m",
        )
        self.time_cost_input = decimal_input(
            float(reward_defaults.get("time_to_success_weight_per_s", 0.25)),
            0.0,
            100.0,
            0.05,
            3,
            "/s",
        )
        self.acceleration_effort_input = decimal_input(
            float(reward_defaults.get("acceleration_effort_weight", 0.05)),
            0.0,
            100.0,
            0.01,
            3,
        )
        self.body_rate_effort_input = decimal_input(
            float(reward_defaults.get("body_rate_effort_weight", 0.02)),
            0.0,
            100.0,
            0.01,
            3,
        )
        self.action_smoothness_input = decimal_input(
            float(reward_defaults.get("action_smoothness_weight", 0.05)),
            0.0,
            100.0,
            0.01,
            3,
        )
        motion_reward_form.addRow("Terminal displacement (-)", self.terminal_displacement_weight_input)
        motion_reward_form.addRow(
            "Terminal charge scope", self.terminal_displacement_success_only_input
        )
        motion_reward_form.addRow("Displacement integral (-)", self.displacement_integral_weight_input)
        motion_reward_form.addRow("Displacement scale", self.displacement_scale_input)
        motion_reward_form.addRow("Time deduction (-)", self.time_cost_input)
        motion_reward_form.addRow("Acceleration deduction (-)", self.acceleration_effort_input)
        motion_reward_form.addRow("Body-rate deduction (-)", self.body_rate_effort_input)
        motion_reward_form.addRow("Action-change deduction (-)", self.action_smoothness_input)
        configuration_layout.addWidget(motion_reward_group, 1)

        ppo_group = QGroupBox("PPO learning")
        ppo_form = QFormLayout(ppo_group)
        self.learning_rate_input = decimal_input(
            float(ppo_defaults.get("learning_rate", 3.0e-4)),
            1.0e-6,
            1.0e-2,
            5.0e-5,
            6,
        )
        self.gamma_input = decimal_input(
            float(ppo_defaults.get("gamma", 0.99)), 0.8, 0.9999, 0.001, 4
        )
        self.gae_lambda_input = decimal_input(
            float(ppo_defaults.get("gae_lambda", 0.95)), 0.8, 1.0, 0.005, 3
        )
        self.clip_ratio_input = decimal_input(
            float(ppo_defaults.get("clip_ratio", 0.2)), 0.01, 1.0, 0.01, 3
        )
        self.entropy_input = decimal_input(
            float(ppo_defaults.get("entropy_coefficient", 0.01)),
            0.0,
            1.0,
            0.005,
            4,
        )
        self.value_coefficient_input = decimal_input(
            float(ppo_defaults.get("value_coefficient", 0.5)),
            0.0,
            10.0,
            0.1,
            3,
        )
        self.target_kl_input = decimal_input(
            float(ppo_defaults.get("target_kl", 0.02)), 0.0001, 1.0, 0.005, 4
        )
        self.gradient_norm_input = decimal_input(
            float(ppo_defaults.get("maximum_gradient_norm", 0.5)),
            0.01,
            100.0,
            0.1,
            2,
        )
        self.update_epochs_input = integer_input(
            int(ppo_defaults.get("update_epochs", 4)), 1, 32
        )
        self.minibatch_input = integer_input(
            int(ppo_defaults.get("minibatch_transitions", 8_192)),
            256,
            262_144,
            256,
        )
        ppo_form.addRow("Learning rate", self.learning_rate_input)
        ppo_form.addRow("Gamma", self.gamma_input)
        ppo_form.addRow("GAE lambda", self.gae_lambda_input)
        ppo_form.addRow("Clip ratio", self.clip_ratio_input)
        ppo_form.addRow("Entropy", self.entropy_input)
        ppo_form.addRow("Value loss", self.value_coefficient_input)
        ppo_form.addRow("Target KL", self.target_kl_input)
        ppo_form.addRow("Gradient clip", self.gradient_norm_input)
        ppo_form.addRow("Update epochs", self.update_epochs_input)
        ppo_form.addRow("Minibatch transitions", self.minibatch_input)
        configuration_layout.addWidget(ppo_group, 1)

        configuration_group_layout.addWidget(self.configuration_content)
        self.configuration_content.setVisible(False)
        self.configuration_group.toggled.connect(self.configuration_content.setVisible)
        layout.addWidget(self.configuration_group)

        validation_bar = QFrame()
        validation_bar.setObjectName("toolbarCard")
        validation_layout = QHBoxLayout(validation_bar)
        validation_layout.setContentsMargins(13, 8, 11, 8)
        validation_heading = QVBoxLayout()
        validation_heading.setSpacing(0)
        validation_title = QLabel("CURRENT-POLICY VALIDATION")
        validation_title.setStyleSheet(
            "color: #64748b; font-size: 8pt; font-weight: 800; letter-spacing: 0.8px;"
        )
        self.manual_validation_detail = QLabel(
            "Deterministic mildly varied states • latest learned policy • canonical target"
        )
        self.manual_validation_detail.setStyleSheet(
            "font-weight: 650; color: #334155;"
        )
        validation_heading.addWidget(validation_title)
        validation_heading.addWidget(self.manual_validation_detail)
        validation_layout.addLayout(validation_heading)
        validation_layout.addStretch(1)
        self.manual_validation_status = QLabel("READY")
        set_status_badge(self.manual_validation_status, "READY", "neutral")
        validation_layout.addWidget(self.manual_validation_status)
        validation_layout.addWidget(QLabel("Trials"))
        self.validation_count_input = QSpinBox()
        self.validation_count_input.setRange(1, 128)
        self.validation_count_input.setValue(20)
        self.validation_count_input.setSingleStep(5)
        self.validation_count_input.setToolTip(
            "Number of deterministic initial-state validation rollouts."
        )
        validation_layout.addWidget(self.validation_count_input)
        self.run_validation_button = QPushButton("RUN CURRENT POLICY")
        self.run_validation_button.setObjectName("runCurrentPolicyValidationButton")
        self.run_validation_button.clicked.connect(self.run_manual_validation)
        validation_layout.addWidget(self.run_validation_button)
        layout.addWidget(validation_bar)

        self.episodes_value = self.progress_card.value_label
        self.training_success_value = self.training_card.value_label
        self.validation_success_value = self.validation_card.value_label
        self.speed_value = self.throughput_card.value_label
        self.endpoint_success_value = self.training_card.detail_label

        self.curves = TrainingCurves(self)
        layout.addWidget(self.curves, 1)

        contract = QGroupBox("Experiment contract")
        contract.setCheckable(True)
        contract.setChecked(False)
        form = QFormLayout(contract)
        form.addRow("Learner", QLabel("Clipped PPO + GAE; fresh policy"))
        form.addRow("Initialization", QLabel("Random policy; no CEM or demonstrations"))
        form.addRow("Training start", QLabel("Canonical settled state"))
        form.addRow("Success", QLabel("Single first entry: c10, 50 mm, 4 m/s, 30 deg; no time gate"))
        form.addRow("UAV excursion", QLabel("Terminal displacement cost + displacement integral; no hard gate"))
        form.addRow("Successful hit", QLabel("Terminal transition; post-hit row frozen and excluded"))
        form.addRow("Strike shaping", QLabel("Joint proximity + attachment-relative speed + direction quality"))
        form.addRow("Motion", QLabel("Target-aligned sagittal plane; no effort or smoothness reward terms"))
        form.addRow("Validation", QLabel("64 deterministic rollouts from fixed mild propagated states"))
        form.addRow("Validation cadence", QLabel("Every 10,240 training episodes"))
        form.addRow("Physics", QLabel("Production UAV + causal residual + 12-node DDER"))
        form.addRow("Horizon / action", QLabel("7.0 s / forward-back + vertical acceleration + pitch rate"))
        self.run_path_value = QLabel("—")
        self.run_path_value.setWordWrap(True)
        form.addRow("Artifact", self.run_path_value)
        contract.toggled.connect(
            lambda checked: [
                form.itemAt(index).widget().setVisible(checked)
                for index in range(form.count())
                if form.itemAt(index).widget() is not None
            ]
        )
        for index in range(form.count()):
            widget = form.itemAt(index).widget()
            if widget is not None:
                widget.setVisible(False)
        layout.addWidget(contract)

    def _status(self) -> dict[str, Any] | None:
        return None if self.current_artifact is None else _read_json(
            self.current_artifact / "status.json"
        )

    def _apply_gui_configuration(self, config: dict[str, Any]) -> None:
        collection_batch = int(config["collection_batch"])
        validation_interval = int(self.validation_interval_input.value())
        if validation_interval % collection_batch != 0:
            raise ValueError(
                f"Validation interval must be a multiple of {collection_batch:,} episodes."
            )
        config["requested_episodes"] = int(self.episode_budget_input.value())
        config["episode_duration_s"] = float(self.episode_horizon_input.value())
        config["seed"] = int(self.seed_input.value())
        initial_states = config.setdefault("training_initial_states", {})
        initial_states["mode"] = str(self.training_state_mode_input.currentData())
        initial_states["canonical_fraction"] = (
            float(self.canonical_fraction_input.value()) / 100.0
        )
        initial_states["seed"] = int(self.seed_input.value()) + 1_000
        if initial_states["mode"] == "mixed_state_bank":
            default_initial_states = (
                (_read_json(CONFIG_PATH) or {}).get("training_initial_states", {})
            )
            for key in (
                "training_bank",
                "training_bank_manifest",
                "validation_bank",
                "validation_bank_manifest",
            ):
                if key not in initial_states and key in default_initial_states:
                    initial_states[key] = default_initial_states[key]
        action_mode = str(self.action_mode_input.currentData())
        action = config["action"]
        action["mode"] = action_mode
        if action_mode == "target_aligned_sagittal_3d":
            action.update(
                {
                    "dimensions": 3,
                    "meaning": [
                        "acceleration parallel to horizontal desired strike direction (signed forward/backward)",
                        "vertical acceleration (signed up/down)",
                        "body pitch rate",
                    ],
                    "lateral_acceleration_available": False,
                }
            )
        else:
            action.update(
                {
                    "dimensions": 6,
                    "meaning": [
                        "yaw-local acceleration x",
                        "yaw-local acceleration y",
                        "yaw-local acceleration z",
                        "body roll rate",
                        "body pitch rate",
                        "body yaw rate",
                    ],
                    "lateral_acceleration_available": True,
                }
            )
        if self.initialization_mode_input.currentData() == "fresh":
            config["initialization"] = {
                "type": "fresh_random_policy_value_and_optimizer",
                "uses_previous_policy_checkpoint": False,
            }
        elif not config.get("initialization", {}).get("policy_checkpoint"):
            raise ValueError("Continuation mode requires a selected policy checkpoint.")
        config["validation"]["every_episodes"] = validation_interval
        config["validation"]["episodes"] = int(
            self.scheduled_validation_count_input.value()
        )
        config["logging"]["training_success_rolling_window_episodes"] = int(
            self.rolling_window_input.value()
        )
        early_stopping = config.setdefault("early_stopping", {})
        early_stopping.update(
            {
                "enabled": bool(self.early_stopping_input.isChecked()),
                "minimum_episodes": int(
                    self.early_stopping_minimum_input.value()
                ),
                "validation_patience_evaluations": int(
                    self.early_stopping_patience_input.value()
                ),
            }
        )
        early_stopping.setdefault(
            "selection_metric",
            "validation_success_rate_then_lower_mean_uav_displacement",
        )
        reward = config["reward"]
        reward.update(
            {
                "progress_weight": float(self.progress_weight_input.value()),
                "progress_shaping_reference": str(
                    self.progress_reference_input.currentData()
                ),
                "progress_attachment_compensation_fraction": float(
                    self.progress_attachment_fraction_input.value()
                ),
                "directed_speed_near_target_weight": float(
                    self.speed_reward_weight_input.value()
                ),
                "directed_speed_reward_cap_m_s": float(self.speed_cap_input.value()),
                "directed_speed_shaping_reference": str(
                    self.speed_reference_input.currentData()
                ),
                "direction_near_target_weight": float(
                    self.direction_reward_weight_input.value()
                ),
                "strike_quality_improvement_weight": float(
                    self.strike_weight_input.value()
                ),
                "success_forward_return_bonus_weight": float(
                    self.forward_return_bonus_input.value()
                ),
                "success_release_bonus_weight": float(
                    self.release_bonus_input.value()
                ),
                "return_release_improvement_weight": float(
                    self.return_release_improvement_input.value()
                ),
                "success_release_at_strike": bool(
                    self.release_at_strike_input.isChecked()
                ),
                "success_bonus": float(self.success_bonus_input.value()),
                "non_tip_first_penalty": float(self.non_tip_penalty_input.value()),
                "maximum_displacement_weight": 0.0,
                "terminal_displacement_weight": float(
                    self.terminal_displacement_weight_input.value()
                ),
                "terminal_displacement_success_only": bool(
                    self.terminal_displacement_success_only_input.isChecked()
                ),
                "displacement_integral_weight": float(
                    self.displacement_integral_weight_input.value()
                ),
                "displacement_cost_scale_m": float(
                    self.displacement_scale_input.value()
                ),
                "success_compactness_bonus": 0.0,
                "time_to_success_weight_per_s": float(self.time_cost_input.value()),
                "acceleration_effort_weight": float(
                    self.acceleration_effort_input.value()
                ),
                "body_rate_effort_weight": float(self.body_rate_effort_input.value()),
                "action_smoothness_weight": float(
                    self.action_smoothness_input.value()
                ),
            }
        )
        ppo = config["ppo"]
        ppo.update(
            {
                "learning_rate": float(self.learning_rate_input.value()),
                "gamma": float(self.gamma_input.value()),
                "gae_lambda": float(self.gae_lambda_input.value()),
                "clip_ratio": float(self.clip_ratio_input.value()),
                "entropy_coefficient": float(self.entropy_input.value()),
                "value_coefficient": float(self.value_coefficient_input.value()),
                "target_kl": float(self.target_kl_input.value()),
                "maximum_gradient_norm": float(self.gradient_norm_input.value()),
                "update_epochs": int(self.update_epochs_input.value()),
                "minibatch_transitions": int(self.minibatch_input.value()),
            }
        )

    def _configuration_from_controls(self) -> dict[str, Any]:
        config = json.loads(json.dumps(self._configuration_template))
        self._apply_gui_configuration(config)
        config["gui_configuration"] = {
            "saved_or_launched_utc": datetime.now(timezone.utc).isoformat(),
            "source": "PPO Training GUI",
        }
        return config

    def _set_gui_configuration(self, config: dict[str, Any]) -> None:
        if config.get("schema") not in {
            "simple_sequential_ppo_v1",
            "simple_sequential_ppo_10s_v1",
        }:
            raise ValueError("The selected file is not a supported PPO run configuration.")
        if config.get("model_freeze") != "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI":
            raise ValueError("The selected config does not use the frozen production model.")
        initialization = config.get("initialization", {})
        mode = "continue" if initialization.get("uses_previous_policy_checkpoint") else "fresh"
        index = self.initialization_mode_input.findData(mode)
        self.initialization_mode_input.setCurrentIndex(max(index, 0))
        action_mode = str(config.get("action", {}).get("mode", "full_6d"))
        action_mode_index = self.action_mode_input.findData(action_mode)
        self.action_mode_input.setCurrentIndex(max(action_mode_index, 0))
        self.episode_budget_input.setValue(int(config.get("requested_episodes", 1_000_000)))
        self.episode_horizon_input.setValue(float(config.get("episode_duration_s", 7.0)))
        self.seed_input.setValue(int(config.get("seed", 444)))
        initial_states = config.get("training_initial_states", {})
        training_state_mode = str(initial_states.get("mode", "canonical"))
        training_state_index = self.training_state_mode_input.findData(
            training_state_mode
        )
        self.training_state_mode_input.setCurrentIndex(max(training_state_index, 0))
        self.canonical_fraction_input.setValue(
            100.0 * float(initial_states.get("canonical_fraction", 0.0))
        )
        validation = config.get("validation", {})
        self.validation_interval_input.setValue(
            int(validation.get("every_episodes", 10_240))
        )
        self.scheduled_validation_count_input.setValue(
            int(validation.get("episodes", 64))
        )
        self.rolling_window_input.setValue(
            int(
                config.get("logging", {}).get(
                    "training_success_rolling_window_episodes", 5_000
                )
            )
        )
        early_stopping = config.get("early_stopping", {})
        self.early_stopping_input.setChecked(
            bool(early_stopping.get("enabled", False))
        )
        self.early_stopping_minimum_input.setValue(
            int(early_stopping.get("minimum_episodes", 250_000))
        )
        self.early_stopping_patience_input.setValue(
            int(early_stopping.get("validation_patience_evaluations", 30))
        )
        reward = config.get("reward", {})
        progress_reference = str(
            reward.get("progress_shaping_reference", "world_tip")
        )
        progress_reference_index = self.progress_reference_input.findData(
            progress_reference
        )
        self.progress_reference_input.setCurrentIndex(
            max(progress_reference_index, 0)
        )
        self.progress_attachment_fraction_input.setValue(
            float(
                reward.get(
                    "progress_attachment_compensation_fraction", 0.5
                )
            )
        )
        speed_reference = str(
            reward.get("directed_speed_shaping_reference", "world_tip")
        )
        speed_reference_index = self.speed_reference_input.findData(speed_reference)
        self.speed_reference_input.setCurrentIndex(max(speed_reference_index, 0))
        reward_controls = (
            (self.progress_weight_input, "progress_weight"),
            (self.speed_reward_weight_input, "directed_speed_near_target_weight"),
            (self.speed_cap_input, "directed_speed_reward_cap_m_s"),
            (self.direction_reward_weight_input, "direction_near_target_weight"),
            (self.strike_weight_input, "strike_quality_improvement_weight"),
            (self.forward_return_bonus_input, "success_forward_return_bonus_weight"),
            (self.release_bonus_input, "success_release_bonus_weight"),
            (
                self.return_release_improvement_input,
                "return_release_improvement_weight",
            ),
            (self.success_bonus_input, "success_bonus"),
            (self.non_tip_penalty_input, "non_tip_first_penalty"),
            (self.terminal_displacement_weight_input, "terminal_displacement_weight"),
            (self.displacement_integral_weight_input, "displacement_integral_weight"),
            (self.displacement_scale_input, "displacement_cost_scale_m"),
            (self.time_cost_input, "time_to_success_weight_per_s"),
            (self.acceleration_effort_input, "acceleration_effort_weight"),
            (self.body_rate_effort_input, "body_rate_effort_weight"),
            (self.action_smoothness_input, "action_smoothness_weight"),
        )
        for control, key in reward_controls:
            if key in reward:
                control.setValue(float(reward[key]))
        self.release_at_strike_input.setChecked(
            bool(reward.get("success_release_at_strike", False))
        )
        self.terminal_displacement_success_only_input.setChecked(
            bool(reward.get("terminal_displacement_success_only", False))
        )
        ppo = config.get("ppo", {})
        ppo_controls = (
            (self.learning_rate_input, "learning_rate"),
            (self.gamma_input, "gamma"),
            (self.gae_lambda_input, "gae_lambda"),
            (self.clip_ratio_input, "clip_ratio"),
            (self.entropy_input, "entropy_coefficient"),
            (self.value_coefficient_input, "value_coefficient"),
            (self.target_kl_input, "target_kl"),
            (self.gradient_norm_input, "maximum_gradient_norm"),
            (self.update_epochs_input, "update_epochs"),
            (self.minibatch_input, "minibatch_transitions"),
        )
        for control, key in ppo_controls:
            if key in ppo:
                if isinstance(control, QSpinBox):
                    control.setValue(int(ppo[key]))
                else:
                    control.setValue(float(ppo[key]))
        self._configuration_template = json.loads(json.dumps(config))

    def save_configuration(self) -> None:
        try:
            config = self._configuration_from_controls()
        except (KeyError, TypeError, ValueError) as error:
            QMessageBox.warning(self, "Invalid PPO configuration", str(error))
            return
        preset_root = PROJECT_ROOT / "config" / "learning" / "ppo_presets"
        preset_root.mkdir(parents=True, exist_ok=True)
        default_path = preset_root / f"ppo_gui_{_utc_stamp()}.json"
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save PPO run configuration",
            str(default_path),
            "JSON configuration (*.json)",
        )
        if not selected:
            return
        path = Path(selected)
        if path.suffix.lower() != ".json":
            path = path.with_suffix(".json")
        _write_json(path, config)
        self._configuration_template = json.loads(json.dumps(config))
        self.loaded_config_label.setText(path.name)
        QMessageBox.information(self, "PPO configuration saved", str(path))

    def load_configuration(self) -> None:
        preset_root = PROJECT_ROOT / "config" / "learning" / "ppo_presets"
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Load PPO run configuration",
            str(preset_root if preset_root.is_dir() else CONFIG_PATH.parent),
            "JSON configuration (*.json)",
        )
        if not selected:
            return
        path = Path(selected)
        config = _read_json(path)
        if config is None:
            QMessageBox.warning(self, "PPO configuration unreadable", str(path))
            return
        try:
            self._set_gui_configuration(config)
        except (KeyError, TypeError, ValueError) as error:
            QMessageBox.warning(self, "Invalid PPO configuration", str(error))
            return
        self.loaded_config_label.setText(path.name)

    def refresh(self) -> None:
        if self.current_artifact is None:
            self.current_artifact = _latest_artifact()
        status = self._status()
        if self.current_artifact is None:
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.resume_button.setEnabled(False)
            self.export_button.setEnabled(False)
            self.folder_button.setEnabled(False)
            self.run_validation_button.setEnabled(False)
            self.curves.show_empty()
            return
        self.run_path_value.setText(str(self.current_artifact))
        self.folder_button.setEnabled(True)
        self.export_button.setEnabled((self.current_artifact / "training_log.csv").is_file())
        manual_status = _read_json(
            self.current_artifact / "manual_validation_status.json"
        )
        manual_state = (
            "READY" if manual_status is None else str(manual_status.get("status", "READY"))
        )
        manual_active = manual_state in {"LAUNCHING", "QUEUED", "RUNNING"}
        self.run_validation_button.setEnabled(not manual_active)
        self.validation_count_input.setEnabled(not manual_active)
        if manual_active:
            set_status_badge(self.manual_validation_status, manual_state, "info")
            count = int(manual_status.get("validation_episodes", 0))
            self.manual_validation_detail.setText(
                f"{count} deterministic state-bank trials • current policy"
            )
        elif manual_state == "COMPLETE":
            successes = int(manual_status.get("validation_successes", 0))
            count = int(manual_status.get("validation_episodes", 0))
            checkpoint = int(manual_status.get("checkpoint_episodes", 0))
            set_status_badge(
                self.manual_validation_status,
                f"{successes}/{count}",
                "success" if successes == count else "info",
            )
            self.manual_validation_detail.setText(
                f"Latest check at {checkpoint:,} episodes • "
                f"mean UAV excursion {float(manual_status.get('mean_maximum_uav_displacement_m', 0.0)):.3f} m"
            )
        elif manual_state == "FAILED":
            set_status_badge(self.manual_validation_status, "FAILED", "danger")
            self.manual_validation_detail.setText(str(manual_status.get("error", "Validation failed")))
        else:
            set_status_badge(self.manual_validation_status, "READY", "neutral")
        if status is None:
            state = "LAUNCHING"
            episodes = 0
            requested = 2_000_000
            success_rate = 0.0
            speed = 0.0
            batch_success_rate = 0.0
        else:
            state = str(status.get("status", "UNKNOWN"))
            episodes = int(status.get("episodes", 0))
            requested = int(status.get("requested_episodes", 2_000_000))
            success_rate = float(status.get("success_rate", 0.0))
            batch_success_rate = float(status.get("batch_success_rate", success_rate))
            speed = float(status.get("episodes_per_second", 0.0))
        configured_rolling_window = int(
            self.rolling_window_input.value()
            if status is None
            else status.get(
                "rolling_window_episodes", self.rolling_window_input.value()
            )
        )
        displayed_rolling_success = (
            batch_success_rate
            if status is None
            else float(
                status.get(
                    "rolling_success_rate",
                    status.get("rolling_5000_success_rate", batch_success_rate),
                )
            )
        )
        active = state in ACTIVE_STATUSES or state == "LAUNCHING"
        for control in self.config_inputs:
            control.setEnabled(not active)
        self.load_config_button.setEnabled(not active)
        tone = (
            "info"
            if active
            else "success"
            if state in {"COMPLETE", "EARLY_STOPPED_VALIDATION"}
            else "danger"
            if state == "FAILED"
            else "neutral"
        )
        set_status_badge(self.status_value, state, tone)
        self.episode_progress.setValue(
            0 if requested <= 0 else min(10_000, int(round(10_000 * episodes / requested)))
        )
        self.progress_card.set_metric(
            f"{episodes / 1000.0:.0f}k",
            f"{100.0 * episodes / max(requested, 1):.1f}% of {requested / 1_000_000.0:.1f}M",
        )
        endpoint_rate = 0.0 if status is None else float(
            status.get("endpoint_success_rate", success_rate)
        )
        self.training_card.set_metric(
            f"{100.0 * displayed_rolling_success:.1f}%",
            f"rolling {configured_rolling_window:,}  •  {100.0 * success_rate:.1f}% cumulative",
        )
        remaining_s = 0.0 if speed <= 0.0 else max(0, requested - episodes) / speed
        eta = (
            "estimating"
            if speed <= 0.0
            else f"~{remaining_s / 60.0:.0f} min remaining"
            if remaining_s < 3600.0
            else f"~{remaining_s / 3600.0:.1f} h remaining"
        )
        self.throughput_card.set_metric(
            f"{speed:.0f} ep/s" if speed else "—", eta
        )
        validation = _read_json(self.current_artifact / "validation_latest.json")
        if validation is None:
            self.validation_card.set_metric("—", "not run yet")
            self.displacement_card.set_metric("—", "not measured yet")
        else:
            self.validation_card.set_metric(
                f"{int(validation['validation_successes'])}/"
                f"{int(validation['validation_episodes'])}",
                f"at {int(validation['checkpoint_episodes']):,} episodes",
            )
            displacement = float(validation.get("mean_maximum_uav_displacement_m", 0.0))
            terminal_displacement = validation.get(
                "mean_terminal_uav_displacement_m"
            )
            displacement_detail = (
                "mean maximum over "
                f"{int(validation.get('validation_episodes', 0)):,} states"
                if terminal_displacement is None
                else (
                    f"terminal mean {float(terminal_displacement):.3f} m • "
                    f"{int(validation.get('validation_episodes', 0)):,} states"
                )
            )
            self.displacement_card.set_metric(
                f"{displacement:.3f} m",
                displacement_detail,
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
        stamp = _utc_stamp()
        artifact = ACTIVE_RUN_ROOT / stamp
        artifact.parent.mkdir(parents=True, exist_ok=True)
        try:
            launch_config = self._configuration_from_controls()
        except (KeyError, TypeError, ValueError) as error:
            QMessageBox.warning(self, "Invalid PPO configuration", str(error))
            return
        launch_config_path = ACTIVE_RUN_ROOT / f"{stamp}.launch_config.json"
        _write_json(launch_config_path, launch_config)
        self._launch(
            [
                str(PROJECT_ROOT / "run_simple_ppo.py"),
                "--train",
                "--config",
                str(launch_config_path),
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

    def export_figures(self) -> None:
        if self.current_artifact is None:
            return
        try:
            from learning.ppo_publication import export_ppo_publication_figures

            manifest = export_ppo_publication_figures(self.current_artifact)
        except (FileNotFoundError, ValueError, OSError) as error:
            QMessageBox.warning(self, "Figure export failed", str(error))
            return
        output = self.current_artifact / "publication_figures"
        QMessageBox.information(
            self,
            "Publication figures ready",
            f"Exported {len(manifest['figures'])} figures as PNG, PDF, and SVG.\n\n"
            f"{output}",
        )

    def run_manual_validation(self) -> None:
        if self.current_artifact is None:
            return
        existing = _read_json(self.current_artifact / "manual_validation_status.json")
        if existing is not None and str(existing.get("status")) in {"QUEUED", "RUNNING"}:
            return
        count = int(self.validation_count_input.value())
        request_id = _utc_stamp()
        run_status = self._status() or {}
        active_training = str(run_status.get("status", "")) in ACTIVE_STATUSES
        status = {
            "status": "QUEUED" if active_training else "LAUNCHING",
            "request_id": request_id,
            "validation_episodes": count,
            "requested_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(self.current_artifact / "manual_validation_status.json", status)
        if active_training:
            _write_json(
                self.current_artifact / "manual_validation_request.json",
                status,
            )
        else:
            checkpoint = self.current_artifact / "checkpoints" / "latest.pt"
            if not checkpoint.is_file():
                _write_json(
                    self.current_artifact / "manual_validation_status.json",
                    {
                        "status": "FAILED",
                        "request_id": request_id,
                        "error": f"No durable latest checkpoint exists at: {checkpoint}",
                    },
                )
                QMessageBox.warning(
                    self,
                    "Checkpoint unavailable",
                    f"No durable latest checkpoint exists at:\n{checkpoint}",
                )
                return
            self._manual_validation_process = subprocess.Popen(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "run_ppo_manual_validation.py"),
                    "--artifact-directory",
                    str(self.current_artifact),
                    "--count",
                    str(count),
                    "--request-id",
                    request_id,
                ],
                cwd=PROJECT_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                    if os.name == "nt"
                    else 0
                ),
                close_fds=True,
            )
        self.refresh()
