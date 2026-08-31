"""Simple deterministic planning-replay page."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from fitting.production_status import get_active_model_freeze
from planning.results import PlanningResult, latest_planning_result, load_planning_result, load_replay_arrays


class ReplayCanvas(FigureCanvasQTAgg):
    def __init__(self, parent: QWidget | None = None) -> None:
        self.figure = Figure(figsize=(9, 6), tight_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.axis = self.figure.add_subplot(111, projection="3d")
        self.arrays: dict[str, np.ndarray] | None = None
        self.result: PlanningResult | None = None
        self.show_placeholder()

    def show_placeholder(self) -> None:
        self.axis.clear()
        self.axis.text2D(
            0.5,
            0.5,
            "Load a saved deterministic MPPI replay",
            transform=self.axis.transAxes,
            ha="center",
            va="center",
            color="#64748b",
            fontsize=14,
        )
        self.axis.set_axis_off()
        self.draw_idle()

    def set_replay(self, result: PlanningResult, arrays: dict[str, np.ndarray]) -> None:
        self.result = result
        self.arrays = arrays
        self.show_frame(0)

    def show_frame(self, index: int) -> None:
        if self.arrays is None or self.result is None:
            return
        arrays = self.arrays
        index = max(0, min(index, len(arrays["time_s"]) - 1))
        cable = arrays["cable_position_m"][index]
        tip_trail = arrays["cable_position_m"][: index + 1, -1]
        uav_trail = arrays["uav_position_m"][: index + 1]
        uav = arrays["uav_position_m"][index]
        target = np.asarray(self.result.task_config["target"]["position_m"], dtype=float)
        direction = np.asarray(
            self.result.task_config["target"]["desired_impact_direction"], dtype=float
        )

        all_points = np.concatenate(
            (arrays["uav_position_m"], arrays["cable_position_m"].reshape(-1, 3), target[None]),
            axis=0,
        )
        center = 0.5 * (all_points.min(axis=0) + all_points.max(axis=0))
        half = 0.58 * max(float(np.ptp(all_points, axis=0).max()), 1.2)
        self.axis.clear()
        self.axis.set_axis_on()
        self.axis.set_xlim(center[0] - half, center[0] + half)
        self.axis.set_ylim(center[1] - half, center[1] + half)
        self.axis.set_zlim(center[2] - half, center[2] + half)
        self.axis.set_box_aspect((1, 1, 1))
        self.axis.view_init(elev=19, azim=-62)
        self.axis.plot(*cable.T, color="#22c7d8", linewidth=3)
        self.axis.scatter(*cable[2:].T, color="#38d5e5", s=20)
        self.axis.scatter(*cable[-1], color="#f97316", s=85, label="c10 tip")
        self.axis.scatter(*uav, color="#f59e0b", marker="D", s=125, label="UAV")
        self.axis.scatter(*target, color="#22c55e", s=130, alpha=0.55, label="Target")
        self.axis.quiver(*target, *(0.18 * direction), color="#16a34a", linewidth=2)
        self.axis.plot(*uav_trail.T, color="#f59e0b", linewidth=1.3, alpha=0.7)
        self.axis.plot(*tip_trail.T, color="#f97316", linewidth=1.3, alpha=0.75)
        self.axis.set_xlabel("X [m]")
        self.axis.set_ylabel("Y [m]")
        self.axis.set_zlabel("Z [m]")
        self.axis.set_title(
            f"{self.result.task_label} — t = {float(arrays['time_s'][index]):.2f} s"
        )
        self.axis.legend(loc="lower left", fontsize=8)
        self.draw_idle()


class SimulatorReplayPage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("simulatorReplayPage")
        self.setStyleSheet("#simulatorReplayPage { background: #f8fafc; }")
        self.current_result: PlanningResult | None = None
        self.current_arrays: dict[str, np.ndarray] | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._advance)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 16)
        status = get_active_model_freeze()
        header = QHBoxLayout()
        title = QLabel("Simulator & Replay")
        title.setObjectName("pageTitle")
        title.setStyleSheet("font-size: 22px; font-weight: 700; color: white;")
        header.addWidget(title)
        header.addStretch(1)
        self.model_status = QLabel(
            "Active model: PR + 12-node DDER    •    "
            f"Model integrity: {status['model_integrity']}    •    "
            f"Ready for MPPI: {'Yes' if status['ready_for_mppi'] else 'No'}"
        )
        self.model_status.setObjectName("headerStatus")
        self.model_status.setStyleSheet("color: #cbd5e1; font-weight: 600;")
        header.addWidget(self.model_status)
        layout.addLayout(header)

        self.canvas = ReplayCanvas(self)
        layout.addWidget(self.canvas, 1)
        controls = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        reset = QPushButton("Reset")
        reset.clicked.connect(lambda: self.timeline.setValue(0))
        load = QPushButton("Load latest plan")
        load.clicked.connect(self.load_latest_plan)
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 0)
        self.timeline.valueChanged.connect(self._frame_changed)
        self.speed = QComboBox()
        self.speed.addItems(("0.25×", "0.5×", "1×", "2×"))
        self.speed.setCurrentText("1×")
        controls.addWidget(self.play_button)
        controls.addWidget(reset)
        controls.addWidget(load)
        controls.addWidget(self.timeline, 1)
        controls.addWidget(QLabel("Playback"))
        controls.addWidget(self.speed)
        layout.addLayout(controls)
        self.loaded_label = QLabel("No replay loaded")
        self.loaded_label.setStyleSheet("color: #64748b;")
        layout.addWidget(self.loaded_label)

    def load_latest_plan(self) -> None:
        result = latest_planning_result("canonical_whip_v1")
        if result is None:
            QMessageBox.information(self, "No plan", "No completed planning result is available.")
            return
        self.load_result(result)

    def load_result(self, result_or_directory: PlanningResult | str | Path) -> None:
        result = (
            result_or_directory
            if isinstance(result_or_directory, PlanningResult)
            else load_planning_result(result_or_directory)
        )
        arrays = load_replay_arrays(result)
        self._timer.stop()
        self.play_button.setText("Play")
        self.current_result = result
        self.current_arrays = arrays
        self.timeline.setRange(0, len(arrays["time_s"]) - 1)
        self.timeline.setValue(0)
        self.canvas.set_replay(result, arrays)
        self.loaded_label.setText(
            f"{result.task_label} • {result.status} • deterministic replay • {result.directory.name}"
        )

    def toggle_play(self) -> None:
        if self.current_arrays is None:
            self.load_latest_plan()
        if self.current_arrays is None:
            return
        if self._timer.isActive():
            self._timer.stop()
            self.play_button.setText("Play")
        else:
            if self.timeline.value() >= self.timeline.maximum():
                self.timeline.setValue(0)
            self._timer.start()
            self.play_button.setText("Pause")

    def _advance(self) -> None:
        speed = {"0.25×": 0.25, "0.5×": 0.5, "1×": 1.0, "2×": 2.0}[self.speed.currentText()]
        increment = max(1, int(round(speed)))
        value = self.timeline.value() + increment
        if value > self.timeline.maximum():
            self._timer.stop()
            self.play_button.setText("Play")
            value = self.timeline.maximum()
        self.timeline.setValue(value)

    def _frame_changed(self, value: int) -> None:
        self.canvas.show_frame(value)

    def stop(self) -> None:
        self._timer.stop()
