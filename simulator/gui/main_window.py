"""Five-stage interface for preliminary calibration and open-loop policy research."""
from pathlib import Path
from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame,
                              QLabel, QPushButton, QTabWidget, QButtonGroup)
from .calibration_page import BaselinePage
from .reward_page import RewardSettingsPage
from .training_workspace import AlgorithmTrainingPage
from .adaptation_page import AdaptationPage
from .research_widgets import note
from .theme import APP_STYLE, load_application_font

PAGE_DEFINITIONS = (
    ('Data & Calibration', 'Recordings → fit setup → review and apply a physical baseline'),
    ('Task & Rewards', 'Shared strike conditions for PPO and SAC'),
    ('PPO', 'Train, inspect validation, and fly the latest policy'),
    ('SAC', 'Train, inspect validation, and fly the latest policy'),
    ('Real-world Updates', 'Recorded-flight replay, physical candidates and force-sequence refinement'),
)


class SimulatorMainWindow(QMainWindow):
    def __init__(self, project_root, model_config, task_config, ppo_config):
        super().__init__()
        self.project_root=Path(project_root)
        load_application_font()
        self.setWindowTitle('Aerial Cable Research')
        self.resize(1440, 900)
        self.setMinimumSize(1120, 720)
        palette = self.palette()
        for role, color in ((QPalette.ColorRole.Window, '#f5f7fb'),
                (QPalette.ColorRole.WindowText, '#172033'), (QPalette.ColorRole.Base, '#ffffff'),
                (QPalette.ColorRole.AlternateBase, '#f5f7fb'), (QPalette.ColorRole.Text, '#172033'),
                (QPalette.ColorRole.Button, '#ffffff'), (QPalette.ColorRole.ButtonText, '#172033'),
                (QPalette.ColorRole.Highlight, '#2563eb'), (QPalette.ColorRole.HighlightedText, '#ffffff')):
            palette.setColor(role, QColor(color))
        self.setPalette(palette)
        self.setStyleSheet(APP_STYLE)
        shell = QWidget()
        shell.setObjectName('applicationShell')
        self.setCentralWidget(shell)
        layout = QHBoxLayout(shell)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        sidebar = QFrame()
        sidebar.setObjectName('sideBar')
        sidebar.setFixedWidth(218)
        navigation = QVBoxLayout(sidebar)
        navigation.setContentsMargins(16, 24, 16, 20)
        navigation.setSpacing(10)
        brand = QLabel('AERIAL CABLE\nRESEARCH')
        brand.setObjectName('brandTitle')
        navigation.addWidget(brand)
        subtitle = QLabel('SIM → REAL → SIM')
        subtitle.setObjectName('brandSubtitle')
        navigation.addWidget(subtitle)
        navigation.addSpacing(30)
        self.navigation_group = QButtonGroup(self)
        self.navigation_buttons = []
        self.main_tabs = QTabWidget()
        self.main_tabs.tabBar().hide()
        self.main_tabs.setDocumentMode(True)
        for index, (title, _) in enumerate(PAGE_DEFINITIONS):
            button = QPushButton(f'{index+1:02d}   {title.replace("&", "&&")}')
            button.setObjectName('navButton')
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, index=index: self.main_tabs.setCurrentIndex(index))
            self.navigation_group.addButton(button, index)
            self.navigation_buttons.append(button)
            navigation.addWidget(button)
        navigation.addStretch()
        footnote = QLabel('INITIAL STATE → ONE STRIKE\nPoint force + DDER cable')
        footnote.setObjectName('sideFootnote')
        footnote.setStyleSheet('font-size: 9pt;')
        navigation.addWidget(footnote)
        layout.addWidget(sidebar)
        content = QVBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)
        header = QFrame()
        header.setObjectName('topBar')
        heading = QVBoxLayout(header)
        heading.setContentsMargins(24, 16, 24, 16)
        heading.setSpacing(3)
        self.shell_page_title = QLabel()
        self.shell_page_title.setObjectName('shellPageTitle')
        self.shell_page_subtitle = QLabel()
        self.shell_page_subtitle.setObjectName('shellPageSubtitle')
        heading.addWidget(self.shell_page_title)
        heading.addWidget(self.shell_page_subtitle)
        self.model_status=note('');heading.addWidget(self.model_status)
        content.addWidget(header)
        content.addWidget(self.main_tabs, 1)
        layout.addLayout(content, 1)
        self.baseline_page = BaselinePage(project_root)
        self.baseline_page.baseline_applied.connect(self.refresh_model_status)
        self.reward_page = RewardSettingsPage(project_root, ppo_config)
        self.training_page = AlgorithmTrainingPage(project_root, 'PPO')
        self.sac_page = AlgorithmTrainingPage(project_root, 'SAC')
        self.adaptation_page = AdaptationPage(project_root)
        for page, (title, _) in zip((self.baseline_page, self.reward_page, self.training_page,
                                    self.sac_page, self.adaptation_page), PAGE_DEFINITIONS):
            self.main_tabs.addTab(page, title)
        self.closing = False
        for page in (self.training_page, self.sac_page):
            page.viewport.flight_finished.connect(self.retry_close)
        self.main_tabs.currentChanged.connect(self._page_changed)
        self._page_changed(0)
        self.refresh_model_status()

    def refresh_model_status(self):
        from simulator.workflow import read_json
        model=read_json(self.project_root/'config/model.json');baseline=read_json(self.project_root/'config/baseline.json',{})
        self.model_status.setText(f'Active model: {baseline.get("version","initial")}  ·  Cable drag {model["cable"].get("external_drag_s_inv",0):g}/s  ·  Applied to new runs')
        for page in (self.training_page,self.sac_page):page.refresh()

    def _page_changed(self, index):
        title, subtitle = PAGE_DEFINITIONS[index]
        self.shell_page_title.setText(title)
        self.shell_page_subtitle.setText(subtitle)
        self.navigation_buttons[index].setChecked(True)
        self.training_page.set_page_active(index == 2)
        self.sac_page.set_page_active(index == 3)
        if index == 1:
            self.reward_page.page_activated()

    def retry_close(self):
        if self.closing:
            QTimer.singleShot(0, self.close)

    def closeEvent(self, event):
        self.closing = True
        ready = [page.shutdown() for page in (self.training_page, self.sac_page)]
        if not all(ready):
            event.ignore()
            return
        super().closeEvent(event)
