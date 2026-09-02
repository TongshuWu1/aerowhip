"""Read-only planning-result and authoritative replay page."""

from __future__ import annotations

import json
from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fitting.production_status import get_active_model_freeze
from planning.results import PlanningResult, latest_planning_result
from planning.video import render_replay_video
from simulator.production import PROJECT_ROOT


TASK_ITEMS = (
    ("Single Target Whip", "canonical_whip_v1"),
    ("Figure-8 Endpoint Whip", "figure8_endpoint_whip_v1"),
    ("Variable-Duration Whip (CEM)", "canonical_whip_variable_duration_v1"),
)


class PlanningPage(QWidget):
    replay_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("planningPage")
        self.setStyleSheet("#planningPage { background: #f8fafc; }")
        self.current_result: PlanningResult | None = None
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 18)
        layout.setSpacing(10)
        overview = QHBoxLayout()
        overview.setSpacing(10)
        setup = QGroupBox("Planning task")
        form = QFormLayout(setup)
        self.task_combo = QComboBox()
        for label, task_id in TASK_ITEMS:
            self.task_combo.addItem(label, task_id)
        self.task_combo.currentIndexChanged.connect(self.refresh)
        self.target_label = QLabel()
        self.direction_label = QLabel()
        self.optimizer_label = QLabel()
        self.model_label = QLabel()
        self.model_label.setObjectName("planning_model_status")
        form.addRow("Task", self.task_combo)
        form.addRow("Target", self.target_label)
        form.addRow("Desired direction", self.direction_label)
        form.addRow("Optimizer", self.optimizer_label)
        form.addRow("Model", self.model_label)
        overview.addWidget(setup, 1)
        self.run_button = QPushButton("OFFLINE RESULTS ONLY")
        self.run_button.setObjectName("run_mppi_button")
        self.run_button.setMinimumHeight(54)
        self.run_button.setStyleSheet(
            "QPushButton { font-size: 18px; font-weight: 800; color: white; "
            "background: #2563eb; border-radius: 7px; }"
            "QPushButton:disabled { background: #94a3b8; }"
        )
        self.run_button.setEnabled(False)
        self.run_button.setVisible(False)
        progress = QGroupBox("Planner progress")
        progress_form = QFormLayout(progress)
        self.iteration_value = QLabel("—")
        self.tip_error_value = QLabel("—")
        self.directed_speed_value = QLabel("—")
        self.uav_displacement_value = QLabel("—")
        self.feasible_value = QLabel("—")
        self.ess_value = QLabel("—")
        self.runtime_value = QLabel("—")
        progress_form.addRow("Iteration", self.iteration_value)
        progress_form.addRow("Best tip error", self.tip_error_value)
        progress_form.addRow("Directed tip speed", self.directed_speed_value)
        progress_form.addRow("UAV displacement", self.uav_displacement_value)
        progress_form.addRow("Feasible candidates", self.feasible_value)
        progress_form.addRow("ESS", self.ess_value)
        progress_form.addRow("Runtime", self.runtime_value)
        overview.addWidget(progress, 1)
        layout.addLayout(overview)
        result_group = QGroupBox("Final deterministic replay")
        result_layout = QVBoxLayout(result_group)
        self.result_status = QLabel("NO RESULT")
        self.result_status.setObjectName("planning_result_status")
        self.result_status.setStyleSheet("font-size: 28px; font-weight: 900; color: #64748b;")
        result_layout.addWidget(self.result_status)
        self.result_metrics = QLabel("Select a completed task result.")
        self.result_metrics.setWordWrap(True)
        self.result_metrics.setStyleSheet("font-size: 13px;")
        result_layout.addWidget(self.result_metrics)
        buttons = QHBoxLayout()
        self.replay_button = QPushButton("REPLAY")
        self.video_button = QPushButton("SAVE VIDEO")
        self.folder_button = QPushButton("OPEN RESULT FOLDER")
        self.replay_button.clicked.connect(self.replay)
        self.video_button.clicked.connect(self.save_video)
        self.folder_button.clicked.connect(self.open_folder)
        buttons.addWidget(self.replay_button)
        buttons.addWidget(self.video_button)
        buttons.addWidget(self.folder_button)
        result_layout.addLayout(buttons)
        layout.addWidget(result_group)
        advanced = QGroupBox("Advanced Planner Settings")
        advanced.setCheckable(True)
        advanced.setChecked(False)
        advanced_layout = QVBoxLayout(advanced)
        self.advanced_label = QLabel()
        self.advanced_label.setWordWrap(True)
        advanced_layout.addWidget(self.advanced_label)
        advanced.toggled.connect(self.advanced_label.setVisible)
        self.advanced_label.setVisible(False)
        layout.addWidget(advanced)
        layout.addStretch(1)

    def selected_task_id(self) -> str:
        return str(self.task_combo.currentData())

    def refresh(self) -> None:
        task_id = self.selected_task_id()
        freeze = get_active_model_freeze()
        self.model_label.setText(
            f"{'Verified / Frozen' if freeze['ready_for_mppi'] else 'Not ready'} — {freeze['name']}"
        )
        result = latest_planning_result(task_id)
        self.current_result = result
        task_config = result.task_config if result is not None else self._task_config_if_available(task_id)
        if task_config is not None:
            target = task_config["target"]
            self.target_label.setText("[" + ", ".join(f"{v:.3f}" for v in target["position_m"]) + "] m")
            self.direction_label.setText("[" + ", ".join(f"{v:.2f}" for v in target["desired_impact_direction"]) + "]")
            if "cem" in task_config:
                planning = task_config["cem"]
                self.optimizer_label.setText("Variable-Duration CEM")
                self.advanced_label.setText(
                    f"Duration: {planning['duration_min_s']:.2f}–{planning['duration_max_initial_s']:.2f} s\n"
                    f"Population: {planning['population']}\n"
                    f"Maximum iterations: {planning['maximum_iterations']}\n"
                    f"Acceleration knots: {planning['acceleration_knots']}\n"
                    f"Elite fraction: {100.0 * planning['elite_fraction']:.1f}%\n"
                    f"Covariance: {planning['covariance']}"
                )
            else:
                planning = task_config["planning"]
                self.optimizer_label.setText("MPPI")
                self.advanced_label.setText(
                    f"Horizon: {planning['horizon_s']:.2f} s\n"
                    f"Samples: {planning['samples']}\n"
                    f"Maximum iterations: {planning['maximum_iterations']}\n"
                    f"Acceleration knots: {planning['acceleration_knots']}\n"
                    f"Sigma: {planning['perturbation_sigma_m_s2']:.2f} m/s²\n"
                    f"Adaptive ESS target: {100.0 * planning.get('adaptive_ess_target_fraction', 0.02):.1f}%\n"
                    f"Seed: {planning['random_seed']}"
                )
        else:
            self.target_label.setText("Not frozen")
            self.direction_label.setText("Not frozen")
            self.optimizer_label.setText("Not defined")
            self.advanced_label.setText("No authorized task configuration exists.")
        self.run_button.setEnabled(False)
        self.run_button.setText("OFFLINE RESULTS ONLY")
        if result is None:
            self._show_no_result(task_id)
        else:
            self._show_result(result)

    @staticmethod
    def _task_config_if_available(task_id: str) -> dict | None:
        path = PROJECT_ROOT / "config" / "tasks" / f"{task_id}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def _show_no_result(self, task_id: str) -> None:
        self.result_status.setText("NO RESULT")
        self.result_status.setStyleSheet("font-size: 28px; font-weight: 900; color: #64748b;")
        self.result_metrics.setText(
            "Figure-8 endpoint planning has not been scientifically defined or run. "
            "No trajectory or video is being invented."
            if task_id == "figure8_endpoint_whip_v1"
            else "No completed deterministic replay is available."
        )
        for button in (self.replay_button, self.video_button, self.folder_button):
            button.setEnabled(False)
        for label in (
            self.iteration_value,
            self.tip_error_value,
            self.directed_speed_value,
            self.uav_displacement_value,
            self.feasible_value,
            self.ess_value,
            self.runtime_value,
        ):
            label.setText("—")

    def _show_result(self, result: PlanningResult) -> None:
        metrics = result.metrics
        history = result.iteration_history
        last = history[-1] if history else {}
        is_cem = "cem" in result.task_config
        planning = result.task_config["cem" if is_cem else "planning"]
        self.iteration_value.setText(f"{len(history)} / {int(planning['maximum_iterations'])}")
        self.tip_error_value.setText(f"{1000.0 * metrics['minimum_tip_target_distance_m']:.2f} mm")
        self.directed_speed_value.setText(f"{metrics['reported_event_directed_tip_speed_m_s']:.3f} m/s")
        self.uav_displacement_value.setText(f"{metrics['maximum_uav_displacement_m']:.3f} m")
        population = int(planning["population"] if is_cem else planning["samples"])
        feasible = int(last.get("feasible_candidate_count" if is_cem else "feasible_count", 0))
        self.feasible_value.setText(f"{feasible} / {population}")
        self.ess_value.setText("N/A (CEM elites)" if is_cem else f"{float(last.get('effective_sample_size', 0.0)):.2f}")
        self.runtime_value.setText(f"{float(last.get('cumulative_runtime_s', 0.0)):.3f} s")
        self.result_status.setText(result.status)
        self.result_status.setStyleSheet(
            "font-size: 28px; font-weight: 900; color: "
            + ("#15803d;" if result.success else "#b91c1c;")
        )
        hit = metrics.get("hit_time_s")
        hit_text = f"{float(hit):.3f} s" if hit is not None else "No valid hit"
        first_marker = metrics.get("first_target_entry_marker_label") or "None"
        optimizer = metrics.get("optimizer", "MPPI")
        duration = metrics.get("optimized_duration_s")
        duration_text = "" if duration is None else f"    •    Optimized duration: {float(duration):.3f} s"
        self.result_metrics.setText(
            f"Optimizer: {optimizer}{duration_text}\n"
            f"Tip error: {1000.0 * metrics['reported_event_tip_position_error_m']:.2f} mm    •    "
            f"Tip speed: {metrics['reported_event_tip_total_speed_m_s']:.3f} m/s    •    "
            f"Directed speed: {metrics['reported_event_directed_tip_speed_m_s']:.3f} m/s\n"
            f"Direction error: {metrics['reported_event_direction_error_deg']:.2f} deg    •    "
            f"UAV displacement: {metrics['maximum_uav_displacement_m']:.3f} m    •    "
            f"UAV max speed: {metrics['maximum_uav_speed_m_s']:.3f} m/s\n"
            f"Hit time: {hit_text}    •    First target-entry marker: {first_marker}"
        )
        for button in (self.replay_button, self.video_button, self.folder_button):
            button.setEnabled(True)

    def replay(self) -> None:
        if self.current_result is not None:
            self.replay_requested.emit(str(self.current_result.directory))

    def save_video(self) -> None:
        if self.current_result is None:
            return
        try:
            path, _ = render_replay_video(self.current_result)
        except Exception as error:
            QMessageBox.critical(self, "Video generation failed", str(error))
            return
        QMessageBox.information(self, "Phone-compatible video ready", f"H.264 MP4 saved at:\n{path}")

    def open_folder(self) -> None:
        if self.current_result is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_result.directory)))
