"""Calibration, PPO and offline CEM trajectory research interface."""
from pathlib import Path
from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame,
                              QLabel, QPushButton, QTabWidget, QButtonGroup)
from .model_workspace import ModelWorkspace,RecordingsWorkspace,DiagnosticsPage
from .reward_page import RewardSettingsPage
from .training_workspace import AlgorithmTrainingPage
from .rehearsal_workspace import RehearsalWorkspace
from .research_widgets import note
from .theme import APP_STYLE, load_application_font

PAGE_DEFINITIONS = (
    ('Model', 'The selected simulator, fitted components and their measured limitations'),
    ('Recordings', 'Preserve, align and review each round of real experiments'),
    ('PPO', 'Train a named policy and compare its validation results'),
    ('Diagnostics', 'Measured and predicted motion, from the attachment to the cable tip'),
    ('Rehearsal & Export', 'Inspect the frozen whip and complete 30 Hz FullState trajectory'),
    ('CEM Planner', 'Optimize position splines through the fitted drone and cable models'),
    ('Adaptation Check', 'Replay measured flights beside their saved predicted drone and cable motion'),
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
        footnote = QLabel('BOOTSTRAP M0\n30 Hz force → 30 Hz FullState\nDrone + cable · both NNs')
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
        from simulator.research_config import prepare_workspace,workspace_configs
        config_folder=prepare_workspace(project_root)
        model_config,task_config,ppo_config=workspace_configs(project_root)
        self.model_page = ModelWorkspace(project_root)
        self.recordings_page = RecordingsWorkspace(project_root)
        self.reward_page = RewardSettingsPage(project_root, ppo_config,config_directory=config_folder)
        self.training_page = AlgorithmTrainingPage(project_root, 'PPO')
        self.training_page.tabs.addTab(self.reward_page,'Task & rewards')
        self.training_page.tabs.currentChanged.connect(lambda i:self.reward_page.page_activated() if i==2 else None)
        from .multidrone_page import MultiDronePage
        self.multidrone_page = MultiDronePage(project_root)
        self.training_page.tabs.addTab(self.multidrone_page, 'Multi-drone scene')
        self.training_page.live_scene_requested.connect(self.multidrone_page.follow_training)
        self.diagnostics_page = DiagnosticsPage(project_root)
        self.fullstate_page = RehearsalWorkspace(project_root)
        from .cem_page import CEMPage
        self.cem_page = CEMPage(project_root)
        from .adaptation_check_page import AdaptationCheckPage
        self.adaptation_check_page = AdaptationCheckPage(project_root)
        for page, (title, _) in zip((self.model_page, self.recordings_page, self.training_page,
                                    self.diagnostics_page, self.fullstate_page, self.cem_page, self.adaptation_check_page), PAGE_DEFINITIONS):
            self.main_tabs.addTab(page, title)
        self.training_page.checkpoint_requested.connect(self.use_fullstate_checkpoint)
        self.training_page.library.changed.connect(self.fullstate_page.refresh_checkpoints)
        self.training_page.library.protected_paths = lambda: [
            self.fullstate_page.checkpoints.currentData(), self.training_page.resume_checkpoint]
        self.closing = False
        for page in (self.training_page,):
            page.viewport.flight_finished.connect(self.retry_close)
        self.fullstate_page.flight_finished.connect(self.retry_close)
        self.main_tabs.currentChanged.connect(self._page_changed)
        self._page_changed(0)
        self.refresh_model_status()

    def refresh_model_status(self):
        from simulator.workflow import read_json
        self.model_status.setText('M0 · both residuals enabled    /    30 Hz planning and export    /    Historical data: development baseline')
        self.training_page.refresh()

    def use_fullstate_checkpoint(self, checkpoint):
        if self.fullstate_page.job.running or (self.fullstate_page.legacy is not None and self.fullstate_page.legacy.thread is not None):
            self.training_page.library.note.setText('Finish or stop the current rehearsal before changing its checkpoint.')
            return
        self.fullstate_page.refresh_checkpoints()
        index = self.fullstate_page.checkpoints.findData(checkpoint)
        if index < 0:
            from simulator.workflow import read_json
            model=read_json(Path(checkpoint).parent.parent/'model.json',{})
            if model and model.get('fullstate_execution',{}).get('schema')!='tracked_pose_execution_v1':
                self.fullstate_page.legacy_open.click();self.fullstate_page.tabs.setCurrentIndex(1)
                old=self.fullstate_page.legacy;old.refresh_checkpoints();old.checkpoints.setCurrentIndex(old.checkpoints.findData(checkpoint))
                self.main_tabs.setCurrentWidget(self.fullstate_page);return
            self.training_page.library.note.setText('This checkpoint is missing its saved model, task or PPO configuration.');return
        self.fullstate_page.checkpoints.setCurrentIndex(index)
        self.fullstate_page.tabs.setCurrentIndex(0)
        self.main_tabs.setCurrentWidget(self.fullstate_page)

    def _page_changed(self, index):
        if not self.main_tabs.isTabEnabled(index):
            self.main_tabs.setCurrentIndex(2)
            return
        title, subtitle = PAGE_DEFINITIONS[index]
        self.shell_page_title.setText(title)
        self.shell_page_subtitle.setText(subtitle)
        self.navigation_buttons[index].setChecked(True)
        self.training_page.set_page_active(index == 2)
        self.fullstate_page.set_page_active(index == 4)
        self.cem_page.set_page_active(index == 5)
        self.adaptation_check_page.set_page_active(index == 6)

    def retry_close(self):
        if self.closing:
            QTimer.singleShot(0, self.close)

    def closeEvent(self, event):
        self.closing = True
        ready = [page.shutdown() for page in (self.training_page, self.fullstate_page, self.cem_page, self.adaptation_check_page)]
        if not all(ready):
            event.ignore()
            QTimer.singleShot(250, self.retry_close)
            return
        super().closeEvent(event)
