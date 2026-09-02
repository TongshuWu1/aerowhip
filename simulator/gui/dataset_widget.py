"""Production Dataset tab backed exclusively by processed-take services."""

from __future__ import annotations

from pathlib import Path
import os
import sys

import numpy as np
from PySide6.QtCore import QProcess, Qt, QTimer
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QPushButton, QSlider, QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from experimental_data.io import atomic_json
from fitting.config import load_fit_configuration
from fitting.dataset import DEFAULT_MANIFEST, Dataset, ProcessedTake, load_dataset, load_manifest
from fitting.episodes import build_physical_episodes
from fitting.segments import manual_use_mask
from ..parameters import SimulatorSettings
from .viewer_3d import CableViewer3D, pose_transform_matrix_xyzw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROLE_LABELS = {
    "training": "Training",
    "validation": "Provisional Validation",
    "untouched_test": "Untouched Test",
    "ignore": "Ignore",
}


class _OffscreenViewer(QWidget):
    """Avoid Win32 OpenGL creation under the Qt offscreen test platform."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel("PyVista measured-take replay is enabled in the desktop UI.")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: #64748b; background: #eef2f7;")
        layout.addWidget(label)

    def set_show_commanded_pose(self, _enabled: bool) -> None:
        pass

    def update_measured_sites(self, *args: object, **kwargs: object) -> float:
        return 0.0

    def close(self) -> None:
        super().close()


class TakeTimeline(QWidget):
    """One coherent Motive-time view of commands, observations and windows."""

    TRACKS = (
        ("Propagate", "command_valid", "#2a9d8f"),
        ("UAV obs", "uav_valid", "#457b9d"),
        ("Cable obs", "cable", "#6a4c93"),
        ("Include", "use", "#90be6d"),
        ("Episodes", "episodes", "#f4a261"),
    )

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(178)
        self.take: ProcessedTake | None = None
        self.window_ranges: tuple[tuple[float, float], ...] = ()
        self.cursor_s = 0.0

    def set_take(
        self, take: ProcessedTake | None, windows: tuple[tuple[float, float], ...] = ()
    ) -> None:
        self.take, self.window_ranges, self.cursor_s = take, windows, 0.0
        self.update()

    def set_cursor(self, time_s: float) -> None:
        self.cursor_s = float(time_s)
        self.update()

    def _draw_mask(self, painter: QPainter, mask: np.ndarray, y: int, color: str) -> None:
        assert self.take is not None
        left, width = 90, max(1, self.width() - 112)
        for index in np.flatnonzero(mask):
            x = left + int(width * float(self.take.arrays["time_s"][index]) / self.take.duration_s)
            painter.fillRect(x, y, max(1, int(width / len(mask)) + 1), 16, QColor(color))

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#f7f9fb"))
        if self.take is None or self.take.duration_s <= 0.0:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Select a processed take")
            return
        left, width = 90, max(1, self.width() - 112)
        time = self.take.arrays["time_s"]
        use = manual_use_mask(self.take)
        masks = {
            "command_valid": self.take.arrays["command_valid"].astype(bool),
            "uav_valid": (
                use
                & self.take.arrays["uav_valid"].astype(bool)
                & ~self.take.arrays.get(
                    "quality_uav_jump", np.zeros(len(time), dtype=bool)
                ).astype(bool)
            ),
            "cable": (
                use
                & np.all(self.take.arrays["cable_marker_valid"], axis=1)
                & ~self.take.arrays.get(
                    "quality_marker_jump", np.zeros(len(time), dtype=bool)
                ).astype(bool)
                & ~self.take.arrays.get(
                    "quality_geometry_invalid", np.zeros(len(time), dtype=bool)
                ).astype(bool)
            ),
            "use": use,
        }
        painter.setPen(QColor("#344054"))
        painter.drawText(left, 16, "Motive manual trim — scientific time")
        for row, (label, key, color) in enumerate(self.TRACKS):
            y = 28 + 26 * row
            painter.drawText(8, y + 13, label)
            painter.fillRect(left, y, width, 16, QColor("#e5e7eb"))
            if key == "episodes":
                for start, end in self.window_ranges:
                    x0 = left + int(width * start / self.take.duration_s)
                    x1 = left + int(width * end / self.take.duration_s)
                    painter.fillRect(x0, y, max(1, x1 - x0), 16, QColor(color))
            else:
                self._draw_mask(painter, masks[key], y, color)
        cursor_x = left + int(width * min(max(self.cursor_s, 0.0), self.take.duration_s) / self.take.duration_s)
        painter.setPen(QColor("#111827"))
        for value in self.take.episode_breaks_s:
            break_x = left + int(width * float(value) / self.take.duration_s)
            painter.drawLine(break_x, 24, break_x, 158)
        painter.setPen(QColor("#d62828")); painter.drawLine(cursor_x, 24, cursor_x, 158)
        painter.setPen(QColor("#667085")); painter.drawText(left, 173, "0.00 s")
        painter.drawText(left + width - 68, 173, f"{self.take.duration_s:.2f} s")


class TakesDatasetWidget(QWidget):
    """Master/detail research interface; never parses raw CSV in the GUI."""

    def __init__(self, settings: SimulatorSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.dataset = Dataset((), {})
        self.selected: ProcessedTake | None = None
        self.frame_index = 0
        self.process: QProcess | None = None
        self.config = load_fit_configuration()
        self._timer = QTimer(self); self._timer.timeout.connect(self._play_tick)
        self._build_ui(); self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self); layout.setContentsMargins(14, 12, 14, 12)
        header = QHBoxLayout(); title_box = QVBoxLayout()
        title = QLabel("Experimental dataset"); title.setStyleSheet("font-size: 19px; font-weight: 650;")
        subtitle = QLabel("Motive manual trim defines every scientific take boundary"); subtitle.setStyleSheet("color: #667085;")
        title_box.addWidget(title); title_box.addWidget(subtitle); header.addLayout(title_box); header.addStretch(1)
        for text, callback in (("Refresh / Scan Takes", self.refresh), ("Process Selected", self._process_selected), ("Process All Changed", self._process_all), ("Save Roles / Segments", self._save_confirmation)):
            button = QPushButton(text); button.clicked.connect(callback); header.addWidget(button)
        layout.addLayout(header)
        self.message = QLabel(""); self.message.setWordWrap(True); self.message.setStyleSheet("padding: 7px; background: #eef4ff; color: #344054;")
        layout.addWidget(self.message)

        split = QSplitter(Qt.Orientation.Vertical); layout.addWidget(split, 1)
        master = QWidget(); master_layout = QVBoxLayout(master); master_layout.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(("Take", "Role", "Motive Duration", "Command Coverage", "UAV Valid", "Cable Valid", "Physical Episodes", "Status"))
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows); self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._selection_changed); self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents); self.table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        master_layout.addWidget(self.table); split.addWidget(master)

        detail = QWidget(); detail_layout = QVBoxLayout(detail); detail_layout.setContentsMargins(0, 8, 0, 0)
        self.take_summary = QLabel("Select a take"); self.take_summary.setWordWrap(True); self.take_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.take_summary.setStyleSheet("padding: 9px; border: 1px solid #d0d5dd; border-radius: 4px;"); detail_layout.addWidget(self.take_summary)
        detail_tabs = QTabWidget(); detail_tabs.setDocumentMode(True); detail_layout.addWidget(detail_tabs, 1)
        trim_tab = QWidget(); trim_layout = QVBoxLayout(trim_tab); trim_layout.setContentsMargins(8, 8, 8, 8)
        self.timeline = TakeTimeline(); trim_layout.addWidget(self.timeline)
        annotation = QGroupBox("Secondary Use / Exclude mask inside the Motive trim"); annotation_layout = QHBoxLayout(annotation)
        self.start_spin = QDoubleSpinBox(); self.start_spin.setDecimals(3); self.end_spin = QDoubleSpinBox(); self.end_spin.setDecimals(3)
        self.segment_mode = QComboBox(); self.segment_mode.addItems(("Use", "Exclude"))
        self.add_interval_button = QPushButton("Add interval"); self.add_interval_button.clicked.connect(self._add_interval)
        self.clear_intervals_button = QPushButton("Clear intervals"); self.clear_intervals_button.clicked.connect(self._clear_intervals)
        self.start_from_cursor_button = QPushButton("Start ← cursor"); self.start_from_cursor_button.clicked.connect(self._set_start_from_cursor)
        self.end_from_cursor_button = QPushButton("End ← cursor"); self.end_from_cursor_button.clicked.connect(self._set_end_from_cursor)
        for label, widget in (("Start [s]", self.start_spin), ("End [s]", self.end_spin), ("Mode", self.segment_mode)):
            annotation_layout.addWidget(QLabel(label)); annotation_layout.addWidget(widget)
        annotation_layout.addWidget(self.start_from_cursor_button); annotation_layout.addWidget(self.end_from_cursor_button)
        annotation_layout.addWidget(self.add_interval_button); annotation_layout.addWidget(self.clear_intervals_button); annotation_layout.addStretch(1); trim_layout.addWidget(annotation)
        episode_breaks = QGroupBox("Physical discontinuities (Episode Break only)")
        episode_break_layout = QHBoxLayout(episode_breaks)
        self.break_spin = QDoubleSpinBox(); self.break_spin.setDecimals(3)
        self.add_break_button = QPushButton("Add Episode Break"); self.add_break_button.clicked.connect(self._add_episode_break)
        self.clear_breaks_button = QPushButton("Clear Episode Breaks"); self.clear_breaks_button.clicked.connect(self._clear_episode_breaks)
        episode_break_layout.addWidget(QLabel("Break time [s]")); episode_break_layout.addWidget(self.break_spin)
        episode_break_layout.addWidget(self.add_break_button); episode_break_layout.addWidget(self.clear_breaks_button); episode_break_layout.addStretch(1)
        trim_layout.addWidget(episode_breaks); trim_layout.addStretch(1)
        detail_tabs.addTab(trim_tab, "Trim / masks / episodes")

        replay_tab = QWidget(); replay_layout = QVBoxLayout(replay_tab); replay_layout.setContentsMargins(8, 8, 8, 8)
        controls = QVBoxLayout(); control_buttons = QHBoxLayout()
        self.play_button = QPushButton("Play"); self.play_button.clicked.connect(self._toggle_play); reset = QPushButton("Reset"); reset.clicked.connect(self._reset_playback)
        self.speed_combo = QComboBox(); self.speed_combo.addItems(("0.25×", "0.5×", "1×", "2×")); self.speed_combo.setCurrentText("1×")
        backend = QLabel("PYVISTA / VTK · measured state + commanded UAV ghost"); backend.setStyleSheet("color: #2563eb; font-size: 8pt; font-weight: 800;")
        control_buttons.addWidget(backend)
        control_buttons.addWidget(self.play_button); control_buttons.addWidget(reset); control_buttons.addWidget(QLabel("Speed")); control_buttons.addWidget(self.speed_combo); control_buttons.addStretch(1)
        controls.addLayout(control_buttons); self.slider = QSlider(Qt.Orientation.Horizontal); self.slider.valueChanged.connect(self._slider_changed); controls.addWidget(self.slider)
        self.playback_status = QLabel(""); self.playback_status.setWordWrap(True); controls.addWidget(self.playback_status); replay_layout.addLayout(controls)
        if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
            self.viewer = _OffscreenViewer(self)
        else:
            self.viewer = CableViewer3D(self, marker_node_indices=tuple(range(1, 11)), node_count=11, cable_length_m=self.settings.cable_configuration.length_m, cable_diameter_m=self.settings.cable_configuration.diameter_m, initial_uav_position_m=self.settings.initial_uav_position_m)
        self.viewer.set_show_commanded_pose(True); replay_layout.addWidget(self.viewer, 1)
        detail_tabs.addTab(replay_tab, "Measured 3D replay")
        split.addWidget(detail); split.setSizes((230, 500))

    def _physical_episodes(self, take: ProcessedTake) -> tuple[object, ...]:
        episodes, _ = build_physical_episodes(take, self.config)
        return tuple(episodes)

    def refresh(self) -> None:
        current = None if self.selected is None else self.selected.take_id
        self.dataset = load_dataset(include_untouched_test=False); self.table.setRowCount(len(self.dataset.takes))
        self.message.setText("Processed non-protected data only. External command = position / velocity / acceleration / yaw. The Untouched Test is listed in the inventory but is not loaded into this workspace.")
        for row, take in enumerate(self.dataset.takes):
            quality, command, episodes = take.sync_report["quality"], take.sync_report["commands"], self._physical_episodes(take)
            values = (take.take_id, f"{take.duration_s:.2f} s", f"{100*float(command['command_coverage_fraction']):.1f}%", f"{100*float(quality['uav_valid_fraction']):.1f}%", f"{100*float(quality['cable_valid_fraction']):.1f}%", str(len(episodes)), str(take.metadata["quality_status"]))
            self.table.setItem(row, 0, QTableWidgetItem(values[0]))
            role = QComboBox(); role.addItems(tuple(ROLE_LABELS.values())); role.setCurrentText(ROLE_LABELS[take.role])
            if take.role == "untouched_test":
                role.setStyleSheet("font-weight: 650; color: #b42318;")
                role.setEnabled(False)
                role.setToolTip("Protected ownership is immutable in the GUI.")
            role.currentTextChanged.connect(lambda text, take_id=take.take_id: self._set_role(take_id, text)); self.table.setCellWidget(row, 1, role)
            for column, value in enumerate(values[1:], start=2): self.table.setItem(row, column, QTableWidgetItem(value))
            if take.take_id == current: self.table.selectRow(row)
        if current is None and self.dataset.takes: self.table.selectRow(0)

    def _set_role(self, take_id: str, label: str) -> None:
        existing = next((take for take in self.dataset.takes if take.take_id == take_id), None)
        if existing is not None and existing.role == "untouched_test":
            self.message.setText(f"{take_id} is protected and cannot be reassigned.")
            return
        role = next(key for key, value in ROLE_LABELS.items() if value == label)
        manifest = load_manifest(); decision = manifest.setdefault("takes", {}).setdefault(take_id, {})
        decision.update({"role": role, "enabled": role != "ignore"}); decision.setdefault("note", ""); decision.setdefault("segments", []); decision.setdefault("episode_breaks_s", [])
        atomic_json(DEFAULT_MANIFEST, manifest)
        self.message.setText(f"Saved {take_id} as {label}. " + ("This role is excluded from all fitting/normalization." if role == "untouched_test" else "")); self.refresh()

    def _selection_changed(self) -> None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self.dataset.takes): return
        self.selected = self.dataset.takes[row]; episodes = self._physical_episodes(self.selected)
        self.timeline.set_take(self.selected, tuple((float(item.time_s[0]), float(item.time_s[-1])) for item in episodes))
        self.start_spin.setRange(0.0, self.selected.duration_s); self.end_spin.setRange(0.0, self.selected.duration_s); self.end_spin.setValue(self.selected.duration_s)
        self.break_spin.setRange(0.0, self.selected.duration_s)
        editable = self.selected.role != "untouched_test"
        for widget in (
            self.start_spin,
            self.end_spin,
            self.segment_mode,
            self.add_interval_button,
            self.clear_intervals_button,
            self.start_from_cursor_button,
            self.end_from_cursor_button,
            self.break_spin,
            self.add_break_button,
            self.clear_breaks_button,
        ):
            widget.setEnabled(editable)
        self.slider.setRange(0, self.selected.frame_count - 1); self.frame_index = 0; self.slider.setValue(0)
        metadata, sync, quality = self.selected.metadata, self.selected.sync_report, self.selected.sync_report["quality"]
        self.take_summary.setText(
            f"<b>{self.selected.take_id}</b> · Role: <b>{ROLE_LABELS[self.selected.role]}</b><br>"
            f"Scientific duration: <b>{self.selected.duration_s:.3f} s (Motive manual trim)</b> · Motive: {metadata['source_files']['motive']} · Logger: {metadata['source_files']['logger']}<br>"
            f"Command coverage: {100*float(sync['commands']['command_coverage_fraction']):.2f}% · UAV valid: {100*float(quality['uav_valid_fraction']):.2f}% · Cable valid: {100*float(quality['cable_valid_fraction']):.2f}% · Physical episodes: {len(episodes)} ({sum(item.duration_s for item in episodes):.2f} s) · Sync: {1000*float(sync['ros_to_logger_motive']['rms_residual_s']):.3f} ms<br>"
            f"Episode = uninterrupted propagation; 1.0-s shooting intervals are numerical optimizer boundaries only. Episode breaks: {list(self.selected.episode_breaks_s)}<br>"
            f"Provenance: Mellinger / Kalman / bolt_3in_2s · external command p / v / a / yaw · causal logger pre-history available: {float(metadata.get('logger_pre_history_available_s', 0.0)):.2f} s"
        ); self._render_selected()

    def _write_segments(self, segments: list[dict[str, object]]) -> None:
        if self.selected is None: return
        if self.selected.role == "untouched_test":
            self.message.setText("Protected take: trim/mask metadata was not changed.")
            return
        manifest = load_manifest(); decision = manifest.setdefault("takes", {}).setdefault(self.selected.take_id, {})
        decision.setdefault("role", "ignore"); decision.setdefault("enabled", False); decision.setdefault("note", ""); decision.setdefault("episode_breaks_s", []); decision["segments"] = segments
        atomic_json(DEFAULT_MANIFEST, manifest); self.refresh()

    def _add_interval(self) -> None:
        if self.selected is None: return
        start, end = self.start_spin.value(), self.end_spin.value()
        if end <= start: self.message.setText("Segment end must be later than start."); return
        segments = list(self.selected.segments); segments.append({"start_s": start, "end_s": end, "use": self.segment_mode.currentText() == "Use", "note": ""}); self._write_segments(segments)

    def _set_start_from_cursor(self) -> None:
        if self.selected is not None:
            self.start_spin.setValue(float(self.selected.arrays["time_s"][self.frame_index]))

    def _set_end_from_cursor(self) -> None:
        if self.selected is not None:
            self.end_spin.setValue(float(self.selected.arrays["time_s"][self.frame_index]))

    def _clear_intervals(self) -> None: self._write_segments([])

    def _write_episode_breaks(self, values: list[float]) -> None:
        if self.selected is None: return
        if self.selected.role == "untouched_test":
            self.message.setText("Protected take: episode metadata was not changed.")
            return
        manifest = load_manifest(); decision = manifest.setdefault("takes", {}).setdefault(self.selected.take_id, {})
        decision.setdefault("role", "ignore"); decision.setdefault("enabled", False); decision.setdefault("note", ""); decision.setdefault("segments", [])
        decision["episode_breaks_s"] = sorted(set(float(value) for value in values))
        atomic_json(DEFAULT_MANIFEST, manifest); self.refresh()

    def _add_episode_break(self) -> None:
        if self.selected is None: return
        self._write_episode_breaks(list(self.selected.episode_breaks_s) + [self.break_spin.value()])

    def _clear_episode_breaks(self) -> None: self._write_episode_breaks([])
    def _save_confirmation(self) -> None: self.message.setText(f"Dataset manifest saved: {DEFAULT_MANIFEST}")

    def _start_process(self, take_id: str | None) -> None:
        if self.process is not None: return
        self.process = QProcess(self); self.process.setWorkingDirectory(str(PROJECT_ROOT)); self.process.setProgram(sys.executable)
        args = [str(PROJECT_ROOT / "process_all_takes.py")]
        if take_id: args.extend(("--take", take_id))
        self.process.setArguments(args); self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_process_output); self.process.finished.connect(self._process_finished); self.message.setText("Processing through the deterministic backend…"); self.process.start()

    def _process_selected(self) -> None: self._start_process(None if self.selected is None else self.selected.take_id)
    def _process_all(self) -> None: self._start_process(None)

    def _read_process_output(self) -> None:
        if self.process is not None:
            text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace").strip()
            if text: self.message.setText(text[-1500:])

    def _process_finished(self, exit_code: int, _status: object) -> None:
        self.process = None; self.refresh(); self.message.setText("Processing complete." if exit_code == 0 else f"Processing failed (exit {exit_code}).")

    def _toggle_play(self) -> None:
        if self._timer.isActive(): self._timer.stop(); self.play_button.setText("Play")
        elif self.selected is not None:
            rate = float(self.selected.metadata["motive"]["export_frame_rate_hz"]); speed = float(self.speed_combo.currentText().replace("×", ""))
            self._timer.start(max(1, int(round(1000.0 / (rate * speed))))); self.play_button.setText("Pause")

    def _reset_playback(self) -> None: self._timer.stop(); self.play_button.setText("Play"); self.slider.setValue(0)
    def _play_tick(self) -> None:
        if self.selected is None or self.frame_index + 1 >= self.selected.frame_count: self._timer.stop(); self.play_button.setText("Play"); return
        self.slider.setValue(self.frame_index + 1)
    def _slider_changed(self, value: int) -> None: self.frame_index = int(value); self._render_selected()

    def _render_selected(self) -> None:
        take = self.selected
        if take is None: return
        i, a = min(self.frame_index, take.frame_count - 1), take.arrays
        transform = pose_transform_matrix_xyzw(a["uav_position_m"][i], a["uav_orientation_xyzw"][i])
        connector = transform[:3, :3] @ np.asarray(self.settings.attachment_offset_body_m) + a["uav_position_m"][i]
        sites = np.concatenate((connector[None], a["cable_marker_positions_m"][i]), axis=0); valid = np.concatenate(([bool(a["uav_valid"][i])], a["cable_marker_valid"][i].astype(bool)))
        command_valid = bool(a["command_valid"][i])
        self.viewer.update_measured_sites(sites, valid, uav_position_m=a["uav_position_m"][i], uav_orientation_xyzw=a["uav_orientation_xyzw"][i], attachment_position_m=connector, commanded_uav_position_m=a["command_position_m"][i] if command_valid else None, commanded_uav_orientation_xyzw=a["command_orientation_xyzw"][i] if command_valid else None)
        self.timeline.set_cursor(float(a["time_s"][i])); self.playback_status.setText(f"t = {a['time_s'][i]:.3f} / {take.duration_s:.3f} s · Motive frame {int(a['motive_frame'][i])} · command {'available' if command_valid else 'unavailable'} · cable markers {int(np.count_nonzero(valid[1:]))}/10")

    def close(self) -> None:
        self._timer.stop()
        if self.process is not None: self.process.kill()
        self.viewer.close()
