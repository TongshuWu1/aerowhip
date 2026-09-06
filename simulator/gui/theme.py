"""Shared visual style for the simulator desktop interface."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
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
    font-size: 9pt;
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
QFrame#metricCard, QFrame#toolbarCard, QFrame#contentCard {
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
QPushButton, QComboBox, QDoubleSpinBox, QSpinBox {
    background: white;
    color: #172033;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    min-height: 30px;
    padding: 3px 11px;
}
QPushButton:hover, QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover {
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
QPushButton#secondaryButton {
    background: #eff6ff;
    color: #1d4ed8;
    border-color: #bfdbfe;
    font-weight: 750;
}
QPushButton#secondaryButton:disabled, QPushButton#primaryButton:disabled {
    color: #94a3b8;
    background: #eef2f7;
    border: 1px solid #e2e8f0;
}
QLabel#sectionLead {
    color: #0f172a;
    font-size: 16pt;
    font-weight: 800;
}
QLabel#mutedText {
    color: #64748b;
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
QScrollArea {
    background: transparent;
    border: none;
}
QScrollArea > QWidget > QWidget {
    background: #f5f7fb;
}
QTableWidget, QPlainTextEdit, QComboBox QAbstractItemView {
    background: white;
    color: #172033;
    border: 1px solid #dce3ed;
    border-radius: 6px;
    selection-background-color: #e7effc;
    selection-color: #172033;
}
QTableWidget { gridline-color: #eef2f7; }
QHeaderView::section {
    background: #f0f4f9;
    color: #475569;
    border: none;
    padding: 8px;
    font-size: 9pt;
    font-weight: 600;
}
QProgressBar {
    background: #e8eef7;
    border: none;
    border-radius: 5px;
    min-height: 27px;
    color: #172033;
    text-align: center;
}
QProgressBar::chunk { background: #b9d2fc; border-radius: 5px; }
QSplitter::handle { background: transparent; width: 10px; }
QTabWidget::pane { border: none; }
QTabBar::tab {
    background: #e9eef7;
    color: #52627c;
    border: none;
    padding: 9px 16px;
    margin-right: 4px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
}
QTabBar::tab:selected { background: #ffffff; color: #1d4ed8; }
QToolTip { background: #ffffff; color: #172033; border: 1px solid #cbd5e1; padding: 5px; }
QScrollBar:vertical { background: transparent; width: 9px; margin: 0; }
QScrollBar::handle:vertical { background: #cbd5e1; min-height: 24px; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border; subcontrol-position: top right; width: 20px; height: 17px;
    background: transparent; border: none;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border; subcontrol-position: bottom right; width: 20px; height: 17px;
    background: transparent; border: none;
}
QSpinBox, QDoubleSpinBox { padding-right: 23px; }
"""

_ASSETS = Path(__file__).resolve().parent / 'assets'
APP_STYLE += f"""
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url('{(_ASSETS / 'chevron_up.svg').as_posix()}'); width: 10px; height: 7px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url('{(_ASSETS / 'chevron_down.svg').as_posix()}'); width: 10px; height: 7px;
}}
"""


def load_application_font() -> None:
    """Load a Windows UI font when Qt's isolated runtime finds none."""

    if QFontDatabase.families():
        return
    for path in (
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ):
        if path.is_file() and QFontDatabase.addApplicationFont(str(path)) >= 0:
            return


class MetricCard(QFrame):
    def __init__(
        self,
        caption: str,
        value: str = "-",
        detail: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("metricCard")
        self.setMinimumWidth(126)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(13, 10, 13, 10)
        layout.setSpacing(2)
        self.caption_label = QLabel(caption.upper())
        self.caption_label.setObjectName("metricCaption")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("metricValue")
        self.value_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
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
