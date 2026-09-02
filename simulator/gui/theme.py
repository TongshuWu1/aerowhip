"""Shared visual system for the production research console."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget


APP_STYLE = """
QMainWindow, QWidget#applicationShell, QTabWidget::pane {
    background: #f5f7fb;
}
QWidget {
    color: #172033;
    font-family: "Segoe UI";
    font-size: 10pt;
}
QFrame#sideBar {
    background: #101827;
    border: none;
}
QLabel#brandTitle {
    color: #f8fafc;
    font-size: 15pt;
    font-weight: 800;
    letter-spacing: 0.6px;
}
QLabel#brandSubtitle, QLabel#sideSection, QLabel#sideFootnote {
    color: #94a3b8;
}
QLabel#sideSection {
    font-size: 8pt;
    font-weight: 800;
    letter-spacing: 1.3px;
}
QPushButton#navButton {
    background: transparent;
    color: #cbd5e1;
    border: none;
    border-radius: 7px;
    min-height: 40px;
    padding: 0 13px;
    text-align: left;
    font-weight: 650;
}
QPushButton#navButton:hover {
    background: #1e293b;
    color: white;
}
QPushButton#navButton:checked {
    background: #2563eb;
    color: white;
    font-weight: 800;
}
QFrame#topBar {
    background: white;
    border-bottom: 1px solid #e2e8f0;
}
QLabel#shellPageTitle {
    color: #0f172a;
    font-size: 20pt;
    font-weight: 800;
}
QLabel#shellPageSubtitle {
    color: #64748b;
    font-size: 10pt;
}
QGroupBox {
    background: white;
    border: 1px solid #dce3ed;
    border-radius: 9px;
    margin-top: 12px;
    padding: 12px;
    font-weight: 700;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 5px;
    color: #334155;
}
QFrame#metricCard, QFrame#toolbarCard {
    background: white;
    border: 1px solid #dce3ed;
    border-radius: 9px;
}
QLabel#metricCaption {
    color: #64748b;
    font-size: 8pt;
    font-weight: 800;
    letter-spacing: 0.7px;
}
QLabel#metricValue {
    color: #0f172a;
    font-size: 17pt;
    font-weight: 800;
}
QLabel#metricDetail {
    color: #64748b;
    font-size: 8.5pt;
}
QPushButton, QComboBox, QDoubleSpinBox {
    background: white;
    color: #172033;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    min-height: 30px;
    padding: 3px 11px;
}
QPushButton:hover, QComboBox:hover, QDoubleSpinBox:hover {
    border-color: #2563eb;
}
QPushButton:disabled {
    color: #94a3b8;
    background: #eef2f7;
    border-color: #e2e8f0;
}
QPushButton#primaryButton {
    background: #2563eb;
    color: white;
    border: none;
    font-weight: 800;
}
QPushButton#dangerButton {
    background: #fff7f7;
    color: #b42318;
    border-color: #fecaca;
    font-weight: 750;
}
QTableWidget {
    background: white;
    alternate-background-color: #f8fafc;
    color: #172033;
    gridline-color: #e8edf4;
    border: 1px solid #dce3ed;
    border-radius: 8px;
}
QHeaderView::section {
    background: #f1f5f9;
    color: #475569;
    padding: 7px;
    border: none;
    border-bottom: 1px solid #dce3ed;
    font-weight: 800;
}
QProgressBar {
    background: #e8edf4;
    border: none;
    border-radius: 4px;
    min-height: 8px;
    max-height: 8px;
    text-align: center;
}
QProgressBar::chunk {
    background: #2563eb;
    border-radius: 4px;
}
QScrollArea {
    background: transparent;
    border: none;
}
QScrollArea > QWidget > QWidget {
    background: #f5f7fb;
}
QSlider::groove:horizontal {
    height: 5px;
    background: #dbe3ec;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
    background: #2563eb;
}
"""


class MetricCard(QFrame):
    """Compact KPI card with stable labels that pages can update in place."""

    def __init__(
        self,
        caption: str,
        value: str = "—",
        detail: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("metricCard")
        self.setMinimumWidth(132)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(13, 10, 13, 10)
        layout.setSpacing(2)
        self.caption_label = QLabel(caption.upper())
        self.caption_label.setObjectName("metricCaption")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("metricValue")
        self.value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.detail_label = QLabel(detail)
        self.detail_label.setObjectName("metricDetail")
        self.detail_label.setWordWrap(True)
        layout.addWidget(self.caption_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.detail_label)

    def set_metric(self, value: str, detail: str = "") -> None:
        self.value_label.setText(value)
        self.detail_label.setText(detail)


def set_status_badge(label: QLabel, text: str, tone: str = "neutral") -> None:
    colors = {
        "neutral": ("#eef2f7", "#475569"),
        "info": ("#dbeafe", "#1d4ed8"),
        "success": ("#dcfce7", "#15803d"),
        "warning": ("#fef3c7", "#a16207"),
        "danger": ("#fee2e2", "#b91c1c"),
        "dark": ("#1e293b", "#e2e8f0"),
    }
    background, foreground = colors.get(tone, colors["neutral"])
    label.setText(text)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setStyleSheet(
        f"background: {background}; color: {foreground}; border-radius: 11px; "
        "padding: 4px 10px; font-size: 8pt; font-weight: 800;"
    )
