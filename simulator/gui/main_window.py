"""Publication-quality shell for the production research workflow."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QCloseEvent, QPalette
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..parameters import SimulatorSettings
from .data_page import DataPage
from .model_page import ModelPage
from .planning_page import PlanningPage
from .replay_page import SimulatorReplayPage
from .training_page import TrainingPage
from .theme import APP_STYLE, set_status_badge


PAGE_DEFINITIONS = (
    ("Run & Replay", "Execute the frozen controller and inspect the complete maneuver"),
    ("PPO Training", "Live learning, state-bank validation, and compactness progression"),
    ("Production Model", "Frozen UAV, residual, cable, geometry, and validation evidence"),
    ("Experimental Data", "Accepted physical takes and immutable scientific ownership"),
    ("Planning Archive", "Authoritative CEM and historical planning references"),
)


class SimulatorMainWindow(QMainWindow):
    """Workflow-first shell with scientific logic outside Qt widgets."""

    def __init__(self, settings: SimulatorSettings) -> None:
        super().__init__()
        self.settings = settings
        self.setWindowTitle("Aerial Cable Research — Production")
        self.resize(1440, 900)
        self.setMinimumSize(1120, 720)
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
        self.setStyleSheet(APP_STYLE)
        shell = QWidget(self)
        shell.setObjectName("applicationShell")
        shell_layout = QHBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        self.setCentralWidget(shell)

        sidebar = QFrame(shell)
        sidebar.setObjectName("sideBar")
        sidebar.setFixedWidth(224)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(18, 22, 18, 18)
        sidebar_layout.setSpacing(8)
        brand = QLabel("AERIAL CABLE\nRESEARCH")
        brand.setObjectName("brandTitle")
        brand_subtitle = QLabel("PRODUCTION CONSOLE")
        brand_subtitle.setObjectName("brandSubtitle")
        sidebar_layout.addWidget(brand)
        sidebar_layout.addWidget(brand_subtitle)
        sidebar_layout.addSpacing(26)
        section = QLabel("WORKFLOW")
        section.setObjectName("sideSection")
        sidebar_layout.addWidget(section)
        self.navigation_group = QButtonGroup(self)
        self.navigation_group.setExclusive(True)
        self.navigation_buttons: list[QPushButton] = []
        for index, (title, _) in enumerate(PAGE_DEFINITIONS):
            button = QPushButton(f"{index + 1:02d}   {title}")
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.clicked.connect(
                lambda checked=False, page_index=index: self.main_tabs.setCurrentIndex(
                    page_index
                )
            )
            self.navigation_group.addButton(button, index)
            self.navigation_buttons.append(button)
            sidebar_layout.addWidget(button)
        sidebar_layout.addStretch(1)
        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet("background: #334155; border: none;")
        sidebar_layout.addWidget(divider)
        freeze_label = QLabel("MODEL FREEZE\nREMEASURED GEOMETRY")
        freeze_label.setObjectName("sideFootnote")
        freeze_label.setStyleSheet("font-size: 8pt; font-weight: 700;")
        sidebar_layout.addWidget(freeze_label)
        simulation_label = QLabel("SIMULATION ONLY")
        set_status_badge(simulation_label, "SIMULATION ONLY", "dark")
        sidebar_layout.addWidget(simulation_label)
        shell_layout.addWidget(sidebar)

        content = QWidget(shell)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        top_bar = QFrame(content)
        top_bar.setObjectName("topBar")
        top_bar.setFixedHeight(78)
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(24, 14, 24, 14)
        heading = QVBoxLayout()
        heading.setSpacing(1)
        self.shell_page_title = QLabel()
        self.shell_page_title.setObjectName("shellPageTitle")
        self.shell_page_subtitle = QLabel()
        self.shell_page_subtitle.setObjectName("shellPageSubtitle")
        heading.addWidget(self.shell_page_title)
        heading.addWidget(self.shell_page_subtitle)
        top_bar_layout.addLayout(heading)
        top_bar_layout.addStretch(1)
        model_badge = QLabel()
        set_status_badge(model_badge, "MODEL FROZEN", "success")
        top_bar_layout.addWidget(model_badge)
        physics_badge = QLabel()
        set_status_badge(physics_badge, "FULL UAV + DDER", "info")
        top_bar_layout.addWidget(physics_badge)
        content_layout.addWidget(top_bar)

        self.main_tabs = QTabWidget(content)
        self.main_tabs.tabBar().hide()
        self.main_tabs.setDocumentMode(True)
        content_layout.addWidget(self.main_tabs, 1)
        shell_layout.addWidget(content, 1)
        self.simulator_page = SimulatorReplayPage(self)
        self.data_page = DataPage(self)
        self.model_page = ModelPage(settings, self)
        self.planning_page = PlanningPage(self)
        self.training_page = TrainingPage(self)
        self.main_tabs.addTab(self.simulator_page, "Run & Replay")
        self.main_tabs.addTab(self.training_page, "PPO Training")
        self.main_tabs.addTab(self.model_page, "Production Model")
        self.main_tabs.addTab(self.data_page, "Experimental Data")
        self.main_tabs.addTab(self.planning_page, "Planning Archive")
        self.main_tabs.currentChanged.connect(self._page_changed)
        self.planning_page.replay_requested.connect(self._show_replay)
        self.navigation_buttons[0].setChecked(True)
        self._page_changed(0)

    def _page_changed(self, index: int) -> None:
        page = self.main_tabs.widget(index)
        title, subtitle = PAGE_DEFINITIONS[index]
        self.shell_page_title.setText(title)
        self.shell_page_subtitle.setText(subtitle)
        self.navigation_buttons[index].setChecked(True)
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
