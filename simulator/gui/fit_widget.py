"""Clean read-only Identification and production-model status page."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fitting.production_status import (
    get_active_model_freeze,
    get_active_model_summary,
    get_dataset_role_summary,
    get_latest_fit_summary,
    get_validation_summary,
)
from ..parameters import SimulatorSettings


def _selectable_label(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setMinimumWidth(0)
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def _card(title: str, object_name: str) -> tuple[QGroupBox, QLabel]:
    group = QGroupBox(title)
    group.setObjectName(object_name)
    layout = QVBoxLayout(group)
    label = _selectable_label()
    layout.addWidget(label)
    return group, label


def _fmt_distance(value: object) -> str:
    return f"{1000.0 * float(value):.1f} mm"


class FitValidateWidget(QWidget):
    """GUI facade; physics, data processing, and metric logic stay in backends."""

    def __init__(self, settings: SimulatorSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.fit_summary: dict[str, object] = {}
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        header = QHBoxLayout()
        heading = QVBoxLayout()
        title = QLabel("Identification & Model Status")
        title.setStyleSheet("font-size: 20px; font-weight: 650;")
        subtitle = QLabel(
            "What model is active, what data identifies it, what has been fitted, and how well does it predict?"
        )
        subtitle.setStyleSheet("color: #667085;")
        subtitle.setWordWrap(True)
        heading.addWidget(title)
        heading.addWidget(subtitle)
        header.addLayout(heading)
        header.addStretch(1)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        header.addWidget(refresh)
        root.addLayout(header)

        self.production_status = _selectable_label()
        self.production_status.setObjectName("production_status_banner")
        self.production_status.setStyleSheet(
            "padding: 10px; background: #7f1d1d; color: white; font-weight: 650;"
        )
        root.addWidget(self.production_status)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(11)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        active, self.active_model_label = _card("1. Active Model", "active_model_section")
        layout.addWidget(active)
        details, self.model_details_label = _card("Model details", "model_details_section")
        details.setCheckable(True)
        details.setChecked(False)
        details.toggled.connect(self.model_details_label.setVisible)
        self.model_details_label.setVisible(False)
        layout.addWidget(details)

        data = QGroupBox("2. Dataset / Identification Data")
        data.setObjectName("dataset_identification_section")
        data_layout = QVBoxLayout(data)
        data_layout.addWidget(
            _selectable_label(
                "Identification uses PhysicalEpisodes with one initialization and continuous propagation. "
                "The UAV residual uses causally eligible episode suffixes."
            )
        )
        self.dataset_table = QTableWidget(0, 7)
        self.dataset_table.setObjectName("identification_dataset_table")
        self.dataset_table.setHorizontalHeaderLabels(
            (
                "Take",
                "Role",
                "PhysicalEpisodes",
                "Physical duration",
                "Residual suffixes",
                "Residual duration",
                "Cable status",
            )
        )
        self.dataset_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.dataset_table.verticalHeader().setVisible(False)
        self.dataset_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.dataset_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        data_layout.addWidget(self.dataset_table)
        layout.addWidget(data)

        parameters = QGroupBox("3. Parameter Identification")
        parameters.setObjectName("parameter_identification_section")
        parameter_layout = QHBoxLayout(parameters)
        uav, self.uav_label = _card("UAV Physics", "uav_physics_block")
        residual, self.residual_label = _card("UAV Residual", "uav_residual_block")
        cable, self.cable_label = _card("Cable DDER — EI / Cb", "cable_identification_block")
        parameter_layout.addWidget(uav, 1)
        parameter_layout.addWidget(residual, 1)
        cable_layout = cable.layout()
        assert isinstance(cable_layout, QVBoxLayout)
        buttons = QHBoxLayout()
        self.landscape_button = QPushButton("Open loss landscape")
        self.profile_button = QPushButton("Open EI/Cb profiles")
        self.landscape_button.clicked.connect(lambda: self._open_artifact("loss_landscape"))
        self.profile_button.clicked.connect(lambda: self._open_artifact("profile_curves"))
        buttons.addWidget(self.landscape_button)
        buttons.addWidget(self.profile_button)
        cable_layout.addLayout(buttons)
        parameter_layout.addWidget(cable, 1)
        layout.addWidget(parameters)

        workflow = QGroupBox("Production fitting workflow")
        workflow.setObjectName("production_fitting_workflow")
        workflow_layout = QFormLayout(workflow)
        for name, description in (
            ("Fit UAV Physics", "Fit the effective command-to-UAV model on Training PhysicalEpisodes."),
            ("Fit Cable EI/Cb", "Fit DDER stiffness and damping from the measured UAV boundary and c1...c10."),
            ("Train UAV Residual", "Train the causal acceleration correction after UAV physics is frozen."),
            ("Run Provisional Validation", "Evaluate the frozen PR + DDER model on provisional validation data."),
        ):
            workflow_layout.addRow(name, _selectable_label(description))
        workflow_layout.addRow(
            "Execution",
            _selectable_label(
                "Read-only in this checkpoint. No GUI button can accidentally start an expensive fit or protected evaluation."
            ),
        )
        layout.addWidget(workflow)

        validation_group = QGroupBox("4. Validation")
        validation_group.setObjectName("validation_section")
        validation_layout = QHBoxLayout(validation_group)
        conditional, self.conditional_label = _card(
            "Conditional Cable Validation — Measured UAV Boundary → DDER",
            "conditional_validation_block",
        )
        end, self.end_to_end_label = _card(
            "End-to-End Validation — Command → UAV Physics + Residual → DDER",
            "end_to_end_validation_block",
        )
        validation_layout.addWidget(conditional, 1)
        validation_layout.addWidget(end, 1)
        layout.addWidget(validation_group)

        lead_group = QGroupBox("Task-horizon metrics")
        lead_group.setObjectName("task_horizon_metrics")
        lead_layout = QVBoxLayout(lead_group)
        self.lead_table = QTableWidget(0, 7)
        self.lead_table.setHorizontalHeaderLabels(
            (
                "Lead",
                "Conditional markers",
                "Conditional tip",
                "UAV position",
                "UAV orientation",
                "End-to-end markers",
                "End-to-end tip",
            )
        )
        self.lead_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.lead_table.verticalHeader().setVisible(False)
        self.lead_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        lead_layout.addWidget(self.lead_table)
        layout.addWidget(lead_group)

        freeze, self.freeze_label = _card(
            "5. Production Freeze / Production Status", "production_freeze_section"
        )
        layout.addWidget(freeze)
        layout.addStretch(1)

    def _open_artifact(self, key: str) -> None:
        cable = self.fit_summary.get("cable", {})
        if not isinstance(cable, dict):
            return
        path = Path(str(cable.get(key, "")))
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def refresh(self) -> None:
        active = get_active_model_summary(self.settings)
        self.fit_summary = get_latest_fit_summary(self.settings)
        validation = get_validation_summary()
        freeze = get_active_model_freeze()
        rows = get_dataset_role_summary()

        self.production_status.setText(
            f"{freeze['status']}  ·  Active geometry: {freeze['geometry_version']}  ·  "
            "Protected test: SEALED / NOT EVALUATED"
        )
        self.active_model_label.setText(
            f"UAV model: {active['uav_model']}\n"
            f"UAV correction: {active['uav_correction']} — ACTIVE\n"
            f"Cable: {active['cable_model']}\n"
            f"Coupling: {active['coupling']}\n"
            f"Attachment: {active['attachment']}\n"
            f"Cable observations: {active['observations']}\n"
            f"Cable length: {active['cable_length_m']:.4f} m\n"
            f"UAV reference → connector: {1000*active['attachment_offset_m']:.1f} mm downward\n"
            f"Production backend: {active['backend']}"
        )
        self.model_details_label.setText(
            f"Nodes / edges: {active['node_count']} / {active['edge_count']}\n"
            f"Rest lengths [m]: {active['rest_lengths_m']}\n"
            f"c1...c10 nodes: {active['marker_node_mapping']}\n"
            f"Substeps / position projections: {active['substeps']} / {active['position_projections']}\n"
            f"Total modeled cable mass: {active['total_modeled_mass_kg']:.8f} kg\n"
            f"Residual: {active['residual_architecture']} · {active['residual_parameter_count']} parameters\n"
            f"Model integrity: {active['model_integrity']}"
        )

        self.dataset_table.setRowCount(len(rows))
        for row, item in enumerate(rows):
            values = (
                item["take_id"], item["role"], item["physical_episode_count"],
                f"{item['physical_duration_s']:.2f} s",
                item["residual_eligible_episode_count"],
                f"{item['residual_eligible_duration_s']:.2f} s", item["cable_status"],
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(str(value))
                if item["role"] == "Protected Test":
                    cell.setForeground(Qt.GlobalColor.red)
                self.dataset_table.setItem(row, column, cell)

        uav = self.fit_summary["uav"]
        p = uav["parameters"]
        self.uav_label.setText(
            f"K_p = {p['K_p']:.8g}\nK_v = {p['K_v']:.8g}\nk_a = {p['k_a']:.8g}\n"
            f"K_R = {p['K_R']:.8g}\nK_omega = {p['K_omega']:.8g}\n\n"
            f"Training objective: {uav['training_objective']:.6g}\n"
            f"Training position RMSE: {_fmt_distance(uav['training_position_rmse_m'])}\n"
            f"Training orientation RMSE: {uav['training_orientation_rmse_deg']:.2f}°\n"
            f"Status: {uav['source_status']}\nFit date: {uav['created_utc']}\n"
            f"Artifact ID: {Path(str(uav['artifact'])).name}"
        )
        self.uav_label.setToolTip(str(uav["artifact"]))
        residual = self.fit_summary["residual"]
        improvement = 100.0 * (
            float(residual["validation_position_rmse_physics_m"])
            - float(residual["validation_position_rmse_pr_m"])
        ) / float(residual["validation_position_rmse_physics_m"])
        self.residual_label.setText(
            f"Active: YES\nType: {residual['type']}\nHistory: {residual['history_ms']} ms causal\n"
            f"Architecture: {residual['architecture']}\nParameters: {residual['parameter_count']}\n"
            f"Same-suffix validation position improvement: {improvement:.1f}%\n"
            f"Model hash: {residual['hash']}\n"
            f"Normalization: {Path(str(residual['normalization'])).name}"
        )
        self.residual_label.setToolTip(str(residual["normalization"]))
        cable = self.fit_summary["cable"]
        self.cable_label.setText(
            f"EI = {cable['EI']:.10g} N m²    [{cable['EI_status']}]\n"
            f"Cb = {cable['Cb']:.10g} N m² s    [{cable['Cb_status']}]\n"
            f"Training objective: {cable['training_objective']:.8g}\n"
            f"Method: {cable['method']}\nTopology: {cable['topology']}\n"
            f"Fit geometry: {cable['fit_geometry_version']}\n"
            f"Active geometry: {cable['active_geometry_version']}\n"
            f"Status: {cable['production_status']}\n"
            f"Fit date: {cable['created_utc']}\n"
            f"Artifact ID: {Path(str(cable['artifact'])).name}"
        )
        self.cable_label.setToolTip(str(cable["artifact"]))

        conditional = validation["conditional"]
        self.conditional_label.setText(
            "Tests cable prediction independently of UAV prediction.\n"
            f"Distributed c1...c10 RMSE: {_fmt_distance(conditional['distributed_marker_rmse_m'])}\n"
            f"c10 tip RMSE: {_fmt_distance(conditional['tip_rmse_m'])}\n"
            f"Terminal c10 RMSE: {_fmt_distance(conditional['terminal_tip_rmse_m'])}\n"
            f"Geometry matches active model: {'YES' if validation['geometry_matches_active'] else 'NO'}"
        )
        end = validation["end_to_end"]
        horizon = end["lead_time"]["0.7"]
        self.end_to_end_label.setText(
            "Current production predictor form: PR + 12-node DDER.\n"
            f"Saved 0.70-s UAV position: {_fmt_distance(horizon['uav_position_rmse_m'])}\n"
            f"Saved 0.70-s UAV orientation: {horizon['uav_orientation_rmse_deg']:.2f}°\n"
            f"Saved 0.70-s distributed cable: {_fmt_distance(horizon['distributed_marker_rmse_m'])}\n"
            f"Saved 0.70-s tip: {_fmt_distance(horizon['tip_rmse_m'])}\n"
            f"Saved-artifact criterion: {'PASS' if validation['saved_artifact_pass'] else 'FAIL'}\n"
            f"Active-model criterion: {'PASS / READY FOR MPPI' if validation['active_model_pass'] else 'NOT READY FOR MPPI'}"
        )

        leads = ("0.1", "0.25", "0.5", "0.7", "1.0")
        self.lead_table.setRowCount(len(leads))
        for row, lead in enumerate(leads):
            c = conditional["lead_time"].get(lead, {})
            e = end["lead_time"].get(lead, {})
            values = (
                f"{lead} s" + ("  ← planning horizon" if lead == "0.7" else ""),
                _fmt_distance(c["distributed_marker_rmse_m"]) if c else "n/a",
                _fmt_distance(c["tip_rmse_m"]) if c else "n/a",
                _fmt_distance(e["uav_position_rmse_m"]) if e else "n/a",
                f"{e['uav_orientation_rmse_deg']:.2f}°" if e else "n/a",
                _fmt_distance(e["distributed_marker_rmse_m"]) if e else "n/a",
                _fmt_distance(e["tip_rmse_m"]) if e else "n/a",
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(str(value))
                if lead == "0.7":
                    font = cell.font()
                    font.setBold(True)
                    cell.setFont(font)
                    cell.setBackground(Qt.GlobalColor.yellow)
                self.lead_table.setItem(row, column, cell)

        self.freeze_label.setText(
            f"Freeze name: {freeze['name']}\nStatus: {freeze['status']}\n"
            f"Reason: {freeze['reason_not_ready']}\nGeometry: {freeze['geometry_version']}\n"
            f"UAV parameter source: {freeze['uav_parameter_source']}\n"
            f"Model integrity: {freeze['model_integrity']}\n"
            f"EI / Cb: {freeze['EI']:.10g} / {freeze['Cb']:.10g}\n"
            f"Topology: {freeze['topology']}\n"
            f"Dataset-role snapshot: {freeze['dataset_role_snapshot']}\n"
            f"Component freeze created: {freeze['component_created_utc']}\n"
            f"Protected test: {freeze['protected_test_status']}\n"
            f"Manifest: {Path(str(freeze['manifest_path'])).name}"
        )
        self.freeze_label.setToolTip(str(freeze["manifest_path"]))
