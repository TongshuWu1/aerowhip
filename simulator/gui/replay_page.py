"""Simple deterministic planning-replay page."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
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
from simulator.production import PROJECT_ROOT
from simulator.parameters import SimulatorSettings
from .theme import MetricCard, set_status_badge
from .viewer_3d import CableViewer3D


PPO_SIMULATION_OUTPUT = PROJECT_ROOT / "data" / "ppo_simulation" / "current"
PPO_CHECKPOINT = PROJECT_ROOT / "results" / "ppo" / "checkpoints" / "terminal.pt"
PPO_CURATED_ARTIFACT = PROJECT_ROOT / "results" / "ppo" / "data"
PPO_TRAINING_ROOT = PROJECT_ROOT / "data" / "policy_training"
PPO_SELECTED_ARTIFACT = (
    PROJECT_ROOT
    / "results"
    / "ppo"
    / "policies"
    / "PPO_WHIP_FORWARD_REVERSE_RELEASE_D50_V1"
)
CEM_REFERENCE_TASK_ID = "canonical_whip_variable_duration_tuned_reward_v1"


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def latest_ppo_policy_source() -> tuple[Path, Path, Path, int] | None:
    """Return the selected validated policy, falling back to the newest durable PPO."""

    selected_checkpoint = PPO_SELECTED_ARTIFACT / "checkpoints" / "terminal.pt"
    selected_config = PPO_SELECTED_ARTIFACT / "config.json"
    if selected_checkpoint.is_file() and selected_config.is_file():
        selected_status = _read_json(PPO_SELECTED_ARTIFACT / "status.json") or {}
        selected_episodes = int(
            selected_status.get(
                "latest_durable_checkpoint_episodes",
                selected_status.get("episodes", 0),
            )
        )
        return (
            PPO_SELECTED_ARTIFACT,
            selected_config,
            selected_checkpoint,
            selected_episodes,
        )

    candidates: list[tuple[float, Path, Path, Path, int]] = []
    if PPO_TRAINING_ROOT.is_dir():
        for checkpoint in PPO_TRAINING_ROOT.glob("whip_ppo*/**/checkpoints/latest.pt"):
            artifact = checkpoint.parent.parent
            config = artifact / "config.json"
            snapshot = _read_json(config)
            if (
                snapshot is None
                or snapshot.get("comparison_environment") != "task_whip_once_v1"
            ):
                continue
            status = _read_json(artifact / "status.json") or {}
            episodes = int(
                status.get(
                    "latest_durable_checkpoint_episodes",
                    status.get("episodes", 0),
                )
            )
            candidates.append(
                (checkpoint.stat().st_mtime, artifact, config, checkpoint, episodes)
            )
    curated_checkpoint = PPO_CURATED_ARTIFACT / "terminal.pt"
    curated_config = PPO_CURATED_ARTIFACT / "config.json"
    if curated_checkpoint.is_file() and curated_config.is_file():
        status = _read_json(PPO_CURATED_ARTIFACT / "status.json") or {}
        candidates.append(
            (
                curated_checkpoint.stat().st_mtime,
                PPO_CURATED_ARTIFACT,
                curated_config,
                curated_checkpoint,
                int(status.get("episodes", 0)),
            )
        )
    if not candidates:
        return None
    _, artifact, config, checkpoint, episodes = max(candidates, key=lambda row: row[0])
    return artifact, config, checkpoint, episodes


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
            "Run the frozen PPO controller",
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


class ProductionReplayView(QWidget):
    """PyVista replay viewport with a lightweight offscreen test fallback."""

    def __init__(
        self,
        settings: SimulatorSettings,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.arrays: dict[str, np.ndarray] | None = None
        self.result: PlanningResult | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # VTK cannot create a Win32 OpenGL surface under Qt's offscreen test
        # plugin. The real desktop UI always uses the PyVista renderer.
        self.uses_pyvista = os.environ.get("QT_QPA_PLATFORM", "").lower() != "offscreen"
        if self.uses_pyvista:
            cable = settings.cable_configuration
            self.viewer: CableViewer3D | ReplayCanvas = CableViewer3D(
                self,
                marker_node_indices=cable.marker_node_indices,
                node_count=cable.node_count,
                cable_length_m=cable.length_m,
                cable_diameter_m=cable.diameter_m,
                initial_uav_position_m=settings.initial_uav_position_m,
            )
            self.viewer.set_show_commanded_pose(True)
        else:
            self.viewer = ReplayCanvas(self)
        layout.addWidget(self.viewer, 1)

    @property
    def backend_name(self) -> str:
        return "PYVISTA / VTK" if self.uses_pyvista else "MATPLOTLIB TEST FALLBACK"

    @staticmethod
    def _value_at(
        arrays: dict[str, np.ndarray],
        names: tuple[str, ...],
        index: int,
    ) -> np.ndarray | None:
        for name in names:
            if name in arrays:
                return np.asarray(arrays[name][index])
        return None

    @staticmethod
    def _orientation_at(arrays: dict[str, np.ndarray], index: int) -> np.ndarray:
        value = ProductionReplayView._value_at(
            arrays, ("uav_orientation_xyzw",), index
        )
        return (
            np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
            if value is None
            else np.asarray(value, dtype=np.float64)
        )

    @staticmethod
    def _command_orientation_at(
        arrays: dict[str, np.ndarray], index: int
    ) -> np.ndarray:
        value = ProductionReplayView._value_at(
            arrays,
            ("q_cmd_xyzw", "command_orientation_xyzw"),
            index,
        )
        if value is not None:
            return np.asarray(value, dtype=np.float64)
        yaw = ProductionReplayView._value_at(
            arrays, ("yaw_cmd_rad", "command_yaw_rad"), index
        )
        yaw_value = 0.0 if yaw is None else float(np.asarray(yaw).reshape(()))
        return np.asarray(
            (0.0, 0.0, np.sin(0.5 * yaw_value), np.cos(0.5 * yaw_value)),
            dtype=np.float64,
        )

    def set_replay(self, result: PlanningResult, arrays: dict[str, np.ndarray]) -> None:
        self.result = result
        self.arrays = arrays
        if not self.uses_pyvista:
            assert isinstance(self.viewer, ReplayCanvas)
            self.viewer.set_replay(result, arrays)
            return
        assert isinstance(self.viewer, CableViewer3D)
        target = np.asarray(result.task_config["target"]["position_m"], dtype=float)
        direction = np.asarray(
            result.task_config["target"]["desired_impact_direction"], dtype=float
        )
        command_path = self._value_series(arrays, ("p_cmd_m", "command_position_m"))
        self.viewer.set_replay_overlays(
            target_position_m=target,
            desired_direction=direction,
            uav_path_m=np.asarray(arrays["uav_position_m"]),
            command_path_m=command_path,
            tip_path_m=np.asarray(arrays["cable_position_m"])[:, -1],
        )
        self.show_frame(0)

    @staticmethod
    def _value_series(
        arrays: dict[str, np.ndarray], names: tuple[str, ...]
    ) -> np.ndarray | None:
        for name in names:
            if name in arrays:
                return np.asarray(arrays[name])
        return None

    def show_frame(self, index: int) -> None:
        if self.arrays is None or self.result is None:
            return
        if not self.uses_pyvista:
            assert isinstance(self.viewer, ReplayCanvas)
            self.viewer.show_frame(index)
            return
        assert isinstance(self.viewer, CableViewer3D)
        arrays = self.arrays
        index = max(0, min(index, len(arrays["time_s"]) - 1))
        cable = np.asarray(arrays["cable_position_m"][index])
        uav_position = np.asarray(arrays["uav_position_m"][index])
        command_position = self._value_at(
            arrays, ("p_cmd_m", "command_position_m"), index
        )
        self.viewer.update_replay_vectors(
            uav_position_m=uav_position,
            actual_velocity_m_s=self._value_at(
                arrays, ("uav_velocity_m_s",), index
            ),
            command_position_m=command_position,
            command_velocity_m_s=self._value_at(
                arrays, ("v_cmd_m_s", "command_velocity_m_s"), index
            ),
            command_acceleration_m_s2=self._value_at(
                arrays, ("a_cmd_m_s2", "command_acceleration_m_s2"), index
            ),
            render=False,
        )
        self.viewer.update_state(
            cable,
            uav_position_m=uav_position,
            uav_orientation_xyzw=self._orientation_at(arrays, index),
            attachment_position_m=cable[0],
            commanded_uav_position_m=command_position,
            commanded_uav_orientation_xyzw=(
                None
                if command_position is None
                else self._command_orientation_at(arrays, index)
            ),
        )

    def set_camera_preset(self, name: str) -> None:
        if self.uses_pyvista:
            assert isinstance(self.viewer, CableViewer3D)
            self.viewer.set_camera_preset(name)

    def set_show_commanded_pose(self, enabled: bool) -> None:
        if self.uses_pyvista:
            assert isinstance(self.viewer, CableViewer3D)
            self.viewer.set_show_commanded_pose(enabled)

    def set_show_body_frame(self, enabled: bool) -> None:
        if self.uses_pyvista:
            assert isinstance(self.viewer, CableViewer3D)
            self.viewer.set_show_body_frame(enabled)

    def set_show_trajectories(self, enabled: bool) -> None:
        if self.uses_pyvista:
            assert isinstance(self.viewer, CableViewer3D)
            self.viewer.set_show_trajectories(enabled)

    def set_show_command_vectors(self, enabled: bool) -> None:
        if self.uses_pyvista:
            assert isinstance(self.viewer, CableViewer3D)
            self.viewer.set_show_command_vectors(enabled)

    def close(self) -> None:
        if self.uses_pyvista:
            assert isinstance(self.viewer, CableViewer3D)
            self.viewer.close()
        super().close()


class SimulatorReplayPage(QWidget):
    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        settings: SimulatorSettings | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings or SimulatorSettings.load(
            PROJECT_ROOT / "config" / "default.json"
        )
        self.setObjectName("simulatorReplayPage")
        self.setStyleSheet("#simulatorReplayPage { background: #f8fafc; }")
        self.current_result: PlanningResult | None = None
        self.current_arrays: dict[str, np.ndarray] | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._advance)
        self._ppo_process: subprocess.Popen[bytes] | None = None
        self._ppo_timer = QTimer(self)
        self._ppo_timer.setInterval(500)
        self._ppo_timer.timeout.connect(self._poll_ppo_simulation)
        self._build_ui()
        self._refresh_policy_source()
        self._load_existing_ppo_result()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 18)
        layout.setSpacing(10)
        status = get_active_model_freeze()
        controller = QFrame()
        controller.setObjectName("toolbarCard")
        controller_layout = QHBoxLayout(controller)
        controller_layout.setContentsMargins(14, 10, 12, 10)
        controller_text = QVBoxLayout()
        controller_text.setSpacing(1)
        controller_label = QLabel("LATEST LEARNED POLICY")
        controller_label.setStyleSheet(
            "color: #64748b; font-size: 8pt; font-weight: 800; letter-spacing: 0.8px;"
        )
        self.policy_description = QLabel(
            "Locating newest durable PPO checkpoint  •  deterministic 10 Hz feedback"
        )
        self.policy_description.setStyleSheet("font-weight: 650; color: #334155;")
        self.model_status = QLabel(
            f"Model integrity {status['model_integrity']}  •  production UAV + residual + 12-node DDER"
        )
        self.model_status.setStyleSheet("color: #64748b; font-size: 8.5pt;")
        controller_text.addWidget(controller_label)
        controller_text.addWidget(self.policy_description)
        controller_text.addWidget(self.model_status)
        controller_layout.addLayout(controller_text)
        controller_layout.addStretch(1)
        self.ppo_status = QLabel("READY")
        set_status_badge(self.ppo_status, "READY", "neutral")
        controller_layout.addWidget(self.ppo_status)
        self.run_ppo_button = QPushButton("RUN SELECTED PPO")
        self.run_ppo_button.setObjectName("runPpoSimulationButton")
        self.run_ppo_button.setProperty("role", "primary")
        self.run_ppo_button.setStyleSheet(
            "background: #2563eb; color: white; border: none; font-weight: 800;"
        )
        self.run_ppo_button.clicked.connect(self.run_ppo_simulation)
        controller_layout.addWidget(self.run_ppo_button)
        layout.addWidget(controller)

        metrics = QHBoxLayout()
        metrics.setSpacing(9)
        self.result_card = MetricCard("Result", "—", "No replay loaded")
        self.tip_error_card = MetricCard("Tip error", "—", "target entry")
        self.directed_speed_card = MetricCard("Directed speed", "—", "along target vector")
        self.direction_card = MetricCard("Direction", "—", "impact error")
        self.displacement_card = MetricCard("UAV excursion", "—", "maximum from start")
        self.hit_time_card = MetricCard("Strike time", "—", "first tip entry")
        for card in (
            self.result_card,
            self.tip_error_card,
            self.directed_speed_card,
            self.direction_card,
            self.displacement_card,
            self.hit_time_card,
        ):
            metrics.addWidget(card, 1)
        layout.addLayout(metrics)

        view_toolbar = QFrame()
        view_toolbar.setObjectName("toolbarCard")
        view_controls = QHBoxLayout(view_toolbar)
        view_controls.setContentsMargins(11, 6, 11, 6)
        self.render_backend_label = QLabel("PYVISTA / VTK")
        self.render_backend_label.setStyleSheet(
            "color: #2563eb; font-size: 8pt; font-weight: 800;"
        )
        view_controls.addWidget(self.render_backend_label)
        view_controls.addWidget(QLabel("Camera"))
        self.camera = QComboBox()
        self.camera.addItems(("Perspective", "Front", "Side", "Top", "Follow UAV"))
        view_controls.addWidget(self.camera)
        self.show_command = QCheckBox("Command ghost")
        self.show_command.setChecked(True)
        self.show_paths = QCheckBox("Trajectories")
        self.show_paths.setChecked(True)
        self.show_vectors = QCheckBox("Velocity / acceleration")
        self.show_vectors.setChecked(True)
        self.show_body_axes = QCheckBox("Body axes")
        for control in (
            self.show_command,
            self.show_paths,
            self.show_vectors,
            self.show_body_axes,
        ):
            view_controls.addWidget(control)
        view_controls.addStretch(1)
        legend = QLabel(
            "● actual UAV   ● command   ● cable tip   ● target   "
            "— velocity   — acceleration"
        )
        legend.setStyleSheet("color: #64748b; font-size: 8pt;")
        view_controls.addWidget(legend)
        layout.addWidget(view_toolbar)

        self.canvas = ProductionReplayView(self.settings, self)
        self.render_backend_label.setText(self.canvas.backend_name)
        self.camera.currentTextChanged.connect(self.canvas.set_camera_preset)
        self.show_command.toggled.connect(self.canvas.set_show_commanded_pose)
        self.show_paths.toggled.connect(self.canvas.set_show_trajectories)
        self.show_vectors.toggled.connect(self.canvas.set_show_command_vectors)
        self.show_body_axes.toggled.connect(self.canvas.set_show_body_frame)
        layout.addWidget(self.canvas, 1)
        control_frame = QFrame()
        control_frame.setObjectName("toolbarCard")
        controls = QHBoxLayout(control_frame)
        controls.setContentsMargins(10, 8, 10, 8)
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        reset = QPushButton("Reset")
        reset.clicked.connect(lambda: self.timeline.setValue(0))
        load = QPushButton("Load CEM reference")
        load.clicked.connect(self.load_cem_reference)
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
        layout.addWidget(control_frame)
        self.loaded_label = QLabel("No replay loaded")
        self.loaded_label.setStyleSheet("color: #64748b;")
        layout.addWidget(self.loaded_label)
        self.frame_status = QLabel("State and FullState command telemetry will appear here.")
        self.frame_status.setStyleSheet(
            "color: #475569; background: white; border: 1px solid #e2e8f0; "
            "border-radius: 6px; padding: 5px 9px; font-family: Consolas; font-size: 8.5pt;"
        )
        layout.addWidget(self.frame_status)

    def _refresh_policy_source(self) -> tuple[Path, Path, Path, int] | None:
        source = latest_ppo_policy_source()
        if source is None:
            self.policy_description.setText("No durable PPO checkpoint found")
            self.run_ppo_button.setEnabled(False)
            return None
        artifact, _, checkpoint, episodes = source
        episode_label = f"{episodes:,} episodes" if episodes > 0 else "episode count unavailable"
        self.policy_description.setText(
            f"Selected validated checkpoint  •  {episode_label}  •  deterministic 10 Hz feedback  •  7 s"
        )
        self.policy_description.setToolTip(
            f"Artifact: {artifact}\nCheckpoint: {checkpoint}"
        )
        self.run_ppo_button.setEnabled(True)
        return source

    def _load_existing_ppo_result(self) -> None:
        status = _read_json(PPO_SIMULATION_OUTPUT / "status.json")
        if status is not None and status.get("status") == "COMPLETE":
            try:
                self.load_result(PPO_SIMULATION_OUTPUT)
                success = bool(status.get("success"))
                set_status_badge(
                    self.ppo_status, "PASS" if success else "FAIL", "success" if success else "danger"
                )
            except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
                pass

    def load_cem_reference(self) -> None:
        # The production reference is the Milestone-6A variable-duration CEM
        # replay.  ``canonical_whip_v1`` is a retained 0.70-s legacy MPPI
        # artifact and must not be presented as the CEM reference.
        result = latest_planning_result(CEM_REFERENCE_TASK_ID)
        if result is None:
            QMessageBox.information(
                self,
                "CEM reference unavailable",
                "The saved 2.40-s production CEM reference replay is unavailable.",
            )
            return
        self.load_result(result)

    def load_latest_plan(self) -> None:
        """Compatibility alias for the retained planning-replay tests."""

        self.load_cem_reference()

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
        self._frame_changed(0)
        controller = result.metrics.get(
            "controller", result.metrics.get("optimizer", "saved replay")
        )
        summary = f"{result.task_label} • {result.status} • {controller}"
        if controller == "PPO_10_HZ_CLOSED_LOOP":
            summary += (
                f" • tip {1000.0 * float(result.metrics['first_entry_tip_distance_m']):.1f} mm"
                f" • directed {float(result.metrics['first_entry_directed_speed_m_s']):.2f} m/s"
                f" • direction {float(result.metrics['first_entry_direction_error_deg']):.1f} deg"
                f" • UAV displacement {float(result.metrics['maximum_uav_displacement_m']):.2f} m"
            )
        self.loaded_label.setText(summary)
        success = bool(result.success)
        self.result_card.set_metric("PASS" if success else "FAIL", str(controller))
        self.result_card.value_label.setStyleSheet(
            "color: " + ("#15803d;" if success else "#b91c1c;")
            + " font-size: 17pt; font-weight: 800;"
        )
        tip_error = result.metrics.get(
            "first_entry_tip_distance_m",
            result.metrics.get("reported_event_tip_position_error_m"),
        )
        directed = result.metrics.get(
            "first_entry_directed_speed_m_s",
            result.metrics.get("reported_event_directed_tip_speed_m_s"),
        )
        direction_error = result.metrics.get(
            "first_entry_direction_error_deg",
            result.metrics.get("reported_event_direction_error_deg"),
        )
        displacement = result.metrics.get("maximum_uav_displacement_m")
        hit_time = result.metrics.get("hit_time_s", result.metrics.get("first_entry_time_s"))
        self.tip_error_card.set_metric(
            "—" if tip_error is None else f"{1000.0 * float(tip_error):.1f} mm",
            "≤ 50 mm task threshold",
        )
        self.directed_speed_card.set_metric(
            "—" if directed is None else f"{float(directed):.2f} m/s",
            "≥ 4 m/s task threshold",
        )
        self.direction_card.set_metric(
            "—" if direction_error is None else f"{float(direction_error):.1f}°",
            "≤ 30° task threshold",
        )
        self.displacement_card.set_metric(
            "—" if displacement is None else f"{float(displacement):.2f} m",
            "continuous cost; no hard gate",
        )
        self.hit_time_card.set_metric(
            "—" if hit_time is None else f"{float(hit_time):.2f} s",
            "single attempt",
        )

    def run_ppo_simulation(self) -> None:
        if self._ppo_process is not None and self._ppo_process.poll() is None:
            return
        source = self._refresh_policy_source()
        if source is None:
            QMessageBox.warning(
                self,
                "PPO checkpoint missing",
                "No durable learned PPO checkpoint was found.",
            )
            return
        _, config_path, checkpoint_path, checkpoint_episodes = source
        PPO_SIMULATION_OUTPUT.mkdir(parents=True, exist_ok=True)
        (PPO_SIMULATION_OUTPUT / "status.json").write_text(
            json.dumps(
                {
                    "status": "LAUNCHING",
                    "controller": "PPO_10_HZ_CLOSED_LOOP",
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_episodes": checkpoint_episodes,
                    "authorization": "SIMULATION_ONLY",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        self._ppo_process = subprocess.Popen(
            [
                sys.executable,
                str(PROJECT_ROOT / "run_ppo_simulation.py"),
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoint_path),
                "--output",
                str(PPO_SIMULATION_OUTPUT),
            ],
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
            close_fds=True,
        )
        self.run_ppo_button.setEnabled(False)
        set_status_badge(self.ppo_status, "RUNNING", "info")
        self._ppo_timer.start()

    def _poll_ppo_simulation(self) -> None:
        status = _read_json(PPO_SIMULATION_OUTPUT / "status.json")
        state = "RUNNING" if status is None else str(status.get("status", "RUNNING"))
        if state in {"LAUNCHING", "RUNNING"}:
            set_status_badge(self.ppo_status, state, "info")
            return
        self._ppo_timer.stop()
        self.run_ppo_button.setEnabled(True)
        if state == "COMPLETE":
            success = bool(status and status.get("success"))
            set_status_badge(
                self.ppo_status, "PASS" if success else "FAIL", "success" if success else "danger"
            )
            try:
                self.load_result(PPO_SIMULATION_OUTPUT)
                self.timeline.setValue(0)
                self.toggle_play()
            except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as error:
                QMessageBox.warning(self, "PPO replay error", str(error))
            return
        set_status_badge(self.ppo_status, "ERROR", "danger")
        message = "PPO simulation failed."
        if status is not None and status.get("error"):
            message += f"\n\n{status['error']}"
        QMessageBox.warning(self, "PPO simulation failed", message)

    def toggle_play(self) -> None:
        if self.current_arrays is None:
            self._load_existing_ppo_result()
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
        assert self.current_arrays is not None
        times = self.current_arrays["time_s"]
        dt_s = float(times[1] - times[0]) if len(times) > 1 else 0.01
        real_time_increment = (self._timer.interval() / 1000.0) / max(dt_s, 1.0e-6)
        increment = max(1, int(round(speed * real_time_increment)))
        value = self.timeline.value() + increment
        if value > self.timeline.maximum():
            self._timer.stop()
            self.play_button.setText("Play")
            value = self.timeline.maximum()
        self.timeline.setValue(value)

    def _frame_changed(self, value: int) -> None:
        self.canvas.show_frame(value)
        if self.current_arrays is None:
            return
        arrays = self.current_arrays
        index = max(0, min(int(value), len(arrays["time_s"]) - 1))

        def vector(names: tuple[str, ...]) -> np.ndarray | None:
            return ProductionReplayView._value_at(arrays, names, index)

        def rendered(value_: np.ndarray | None, unit: str) -> str:
            if value_ is None:
                return "—"
            row = np.asarray(value_, dtype=float).reshape(-1)
            return "[" + ", ".join(f"{entry:+.2f}" for entry in row) + f"] {unit}"

        self.frame_status.setText(
            f"t={float(arrays['time_s'][index]):.3f} s   "
            f"UAV p={rendered(vector(('uav_position_m',)), 'm')}   "
            f"p_cmd={rendered(vector(('p_cmd_m', 'command_position_m')), 'm')}   "
            f"v_cmd={rendered(vector(('v_cmd_m_s', 'command_velocity_m_s')), 'm/s')}   "
            f"a_cmd={rendered(vector(('a_cmd_m_s2', 'command_acceleration_m_s2')), 'm/s²')}"
        )

    def stop(self) -> None:
        self._timer.stop()
        self._ppo_timer.stop()

    def close(self) -> None:
        self.stop()
        self.canvas.close()
        super().close()
