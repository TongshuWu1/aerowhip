"""Streamlined production interface for data, model, planning, and replay."""

from __future__ import annotations

from PySide6.QtGui import QColor, QCloseEvent, QPalette
from PySide6.QtWidgets import QMainWindow, QTabWidget

from ..parameters import SimulatorSettings
from .data_page import DataPage
from .model_page import ModelPage
from .planning_page import PlanningPage
from .replay_page import SimulatorReplayPage
from .training_page import TrainingPage


class SimulatorMainWindow(QMainWindow):
    """Four-page production shell with scientific logic outside Qt widgets."""

    def __init__(self, settings: SimulatorSettings) -> None:
        super().__init__()
        self.settings = settings
        self.setWindowTitle("Aerial Cable Research — Production")
        self.resize(1320, 860)
        self.setMinimumSize(1050, 700)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#f8fafc"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#0f172a"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#f1f5f9"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#0f172a"))
        palette.setColor(QPalette.ColorRole.Button, QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("#0f172a"))
        palette.setColor(QPalette.ColorRole.Highlight, QColor("#2563eb"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
        self.setPalette(palette)
        self.setStyleSheet(
            "QMainWindow, QTabWidget::pane { background: #f8fafc; }"
            "QWidget { color: #0f172a; }"
            "QLabel#pageTitle { color: white; }"
            "QLabel#headerStatus { color: #cbd5e1; }"
            "QGroupBox { background: white; border: 1px solid #dbe3ec; "
            "border-radius: 7px; margin-top: 10px; padding: 9px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
            "QPushButton, QComboBox { background: white; color: #0f172a; border: 1px solid #cbd5e1; "
            "border-radius: 5px; min-height: 30px; padding: 3px 11px; }"
            "QPushButton:disabled { color: #64748b; background: #e2e8f0; }"
            "QTableWidget { background: white; color: #0f172a; gridline-color: #e2e8f0; }"
            "QHeaderView::section { background: #f1f5f9; color: #0f172a; padding: 5px; }"
            "QTabBar::tab { background: #e2e8f0; color: #334155; min-width: 120px; "
            "min-height: 34px; font-weight: 650; border: 1px solid #cbd5e1; }"
            "QTabBar::tab:selected { background: white; color: #0f172a; }"
        )
        self.main_tabs = QTabWidget(self)
        self.setCentralWidget(self.main_tabs)
        self.simulator_page = SimulatorReplayPage(self)
        self.data_page = DataPage(self)
        self.model_page = ModelPage(settings, self)
        self.planning_page = PlanningPage(self)
        self.training_page = TrainingPage(self)
        self.main_tabs.addTab(self.simulator_page, "Simulator")
        self.main_tabs.addTab(self.data_page, "Data")
        self.main_tabs.addTab(self.model_page, "Model")
        self.main_tabs.addTab(self.planning_page, "Planning")
        self.main_tabs.addTab(self.training_page, "Training")
        self.main_tabs.currentChanged.connect(self._page_changed)
        self.planning_page.replay_requested.connect(self._show_replay)

    def _page_changed(self, index: int) -> None:
        page = self.main_tabs.widget(index)
        if page is self.data_page:
            self.data_page.refresh()
        elif page is self.model_page:
            self.model_page.refresh()
        elif page is self.planning_page:
            self.planning_page.refresh()
        elif page is self.training_page:
            self.training_page.refresh()

    def _show_replay(self, result_directory: str) -> None:
        self.simulator_page.load_result(result_directory)
        self.main_tabs.setCurrentWidget(self.simulator_page)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.simulator_page.stop()
        super().closeEvent(event)
