"""Compact production-model readiness page."""

from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QGroupBox, QLabel, QScrollArea, QVBoxLayout, QWidget

from fitting.production_status import (
    get_active_model_freeze,
    get_active_model_summary,
    get_latest_fit_summary,
    get_validation_summary,
)
from simulator.parameters import SimulatorSettings


def _card(title: str, body: str, object_name: str = "") -> QGroupBox:
    group = QGroupBox(title)
    group.setStyleSheet("QGroupBox { font-weight: 700; font-size: 14px; }")
    layout = QVBoxLayout(group)
    label = QLabel(body)
    label.setWordWrap(True)
    label.setTextInteractionFlags(label.textInteractionFlags())
    if object_name:
        label.setObjectName(object_name)
    label.setStyleSheet("font-size: 13px; line-height: 1.35;")
    layout.addWidget(label)
    return group


class ModelPage(QWidget):
    def __init__(self, settings: SimulatorSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("modelPage")
        self.setStyleSheet("#modelPage { background: #f8fafc; }")
        self.settings = settings
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 16, 20, 18)
        self.ready_label = QLabel()
        self.ready_label.setObjectName("model_ready_status")
        self.ready_label.setStyleSheet("font-size: 25px; font-weight: 800;")
        outer.addWidget(self.ready_label)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { background: #f8fafc; border: none; }"
            "QScrollArea > QWidget > QWidget { background: #f8fafc; }"
        )
        content = QWidget()
        self.content_layout = QVBoxLayout(content)
        self.card_grid = QGridLayout()
        self.content_layout.addLayout(self.card_grid)
        self.advanced = QGroupBox("Advanced / Reproducibility")
        self.advanced.setCheckable(True)
        self.advanced.setChecked(False)
        advanced_layout = QVBoxLayout(self.advanced)
        self.advanced_label = QLabel()
        self.advanced_label.setWordWrap(True)
        self.advanced_label.setTextInteractionFlags(self.advanced_label.textInteractionFlags())
        advanced_layout.addWidget(self.advanced_label)
        self.advanced.toggled.connect(self.advanced_label.setVisible)
        self.advanced_label.setVisible(False)
        self.content_layout.addWidget(self.advanced)
        self.content_layout.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)
        self.refresh()

    def refresh(self) -> None:
        while self.card_grid.count():
            item = self.card_grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        model = get_active_model_summary(self.settings)
        fit = get_latest_fit_summary(self.settings)
        validation = get_validation_summary()
        freeze = get_active_model_freeze()
        ready = bool(freeze["ready_for_mppi"] and validation["active_model_pass"])
        self.ready_label.setText("MODEL READY" if ready else "MODEL NOT READY")
        self.ready_label.setStyleSheet(
            "font-size: 25px; font-weight: 800; color: " + ("#15803d;" if ready else "#b91c1c;")
        )
        lead = validation["end_to_end"]["lead_time"]["0.7"]
        conditional = validation["conditional"]["lead_time"]["0.7"]
        cards = (
            _card(
                "UAV",
                "Physics + Residual\n\n"
                "Validation @ 0.7 s\n"
                f"Position error: {1000.0 * lead['uav_position_rmse_m']:.2f} mm\n"
                f"Orientation: {lead['uav_orientation_rmse_deg']:.2f} deg",
                "uav_model_card",
            ),
            _card(
                "CABLE",
                "12-node DDER\n\n"
                f"EI: {fit['cable']['EI']:.6g} N m² ({fit['cable']['EI_status']})\n"
                f"Cb: {fit['cable']['Cb']:.6g} N m² s ({fit['cable']['Cb_status']})\n\n"
                "Validation @ 0.7 s\n"
                f"Distributed error: {1000.0 * conditional['distributed_marker_rmse_m']:.2f} mm\n"
                f"Tip error: {1000.0 * conditional['tip_rmse_m']:.2f} mm",
                "cable_model_card",
            ),
            _card(
                "END-TO-END",
                "Validation @ 0.7 s\n\n"
                f"UAV position: {1000.0 * lead['uav_position_rmse_m']:.2f} mm\n"
                f"UAV orientation: {lead['uav_orientation_rmse_deg']:.2f} deg\n"
                f"Cable: {1000.0 * lead['distributed_marker_rmse_m']:.2f} mm\n"
                f"Tip: {1000.0 * lead['tip_rmse_m']:.2f} mm",
                "end_to_end_card",
            ),
            _card(
                "GEOMETRY",
                f"Cable length: {model['cable_length_m']:.4f} m\n"
                f"UAV → connector: {1000.0 * model['attachment_offset_m']:.0f} mm downward\n"
                "Cable observations: c1...c10",
                "geometry_card",
            ),
        )
        for index, card in enumerate(cards):
            self.card_grid.addWidget(card, index // 2, index % 2)
        parameters = fit["uav"]["parameters"]
        self.advanced_label.setText(
            "Exact UAV parameters\n"
            + "\n".join(f"  {name} = {value:.12g}" for name, value in parameters.items())
            + "\n\nRest lengths [m]\n  "
            + ", ".join(f"{value:.4f}" for value in model["rest_lengths_m"])
            + f"\n\nBackend\n  {model['backend']}"
            + f"\n\nProduction freeze\n  {freeze['production_freeze']}"
            + f"\n\nResidual SHA-256\n  {freeze['residual_hash']}"
            + f"\n\nSource SHA-256\n  {freeze['source_hash']}"
            + f"\n\nProtected test\n  {freeze['protected_test_status']}"
        )
