"""Current PVA research workflow; historical flight artifacts remain readable."""
from pathlib import Path
from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor,QPalette
from PySide6.QtWidgets import QMainWindow,QWidget,QHBoxLayout,QVBoxLayout,QFrame,QLabel,QPushButton,QButtonGroup,QTabWidget,QComboBox,QFileDialog
from .theme import APP_STYLE,load_application_font
from .research_widgets import note
from .pva_model_page import PVAModelPage
from .pva_workspace import PVAPlannerPage
from .rehearsal_workspace import RehearsalWorkspace
from simulator.workflow import read_json

PAGES=[('Models & fitting','Choose the model, monitor fitting, and inspect the evidence'),
       ('Recordings','Organize measured OptiTrack data and the commands sent to the drone'),
       ('PPO','Learn a direct P/V/A trajectory generator in the fitted model'),
       ('MPPI','Optimize a direct P/V/A command sequence independently of PPO'),
       ('Rehearsals','Inspect saved commands, predicted motion, and complete CSV exports'),
       ('Flight comparison','Compare normalized measured flights with their exact saved forecasts')]


class PVAResearchWindow(QMainWindow):
    def __init__(self,project_root,*legacy_configs):
        super().__init__();self.root=Path(project_root);load_application_font();self.setWindowTitle('Aerial Cable Research · PVA')
        self.resize(1520,960);self.setMinimumSize(1180,760);self.setStyleSheet(APP_STYLE)
        palette=self.palette()
        for role,color in ((QPalette.ColorRole.Window,'#f5f7fb'),(QPalette.ColorRole.WindowText,'#172033'),
            (QPalette.ColorRole.Base,'#ffffff'),(QPalette.ColorRole.Text,'#172033'),(QPalette.ColorRole.Button,'#ffffff'),
            (QPalette.ColorRole.ButtonText,'#172033'),(QPalette.ColorRole.Highlight,'#2563eb'),(QPalette.ColorRole.HighlightedText,'#ffffff')):
            palette.setColor(role,QColor(color))
        self.setPalette(palette)
        shell=QWidget();shell.setObjectName('applicationShell');self.setCentralWidget(shell);layout=QHBoxLayout(shell);layout.setContentsMargins(0,0,0,0);layout.setSpacing(0)
        sidebar=QFrame();sidebar.setObjectName('sideBar');sidebar.setFixedWidth(216);nav=QVBoxLayout(sidebar);nav.setContentsMargins(16,26,16,20)
        brand=QLabel('AERIAL CABLE\nRESEARCH');brand.setObjectName('brandTitle');nav.addWidget(brand)
        subtitle=QLabel('PLAN · FLY · ADAPT');subtitle.setObjectName('brandSubtitle');nav.addWidget(subtitle);nav.addSpacing(28)
        self.navigation=QButtonGroup(self);self.buttons=[];self.main_tabs=QTabWidget();self.main_tabs.tabBar().hide();self.main_tabs.setDocumentMode(True)
        for i,(title,_) in enumerate(PAGES):
            button=QPushButton(f'{i+1:02d}   '+title.replace('&','&&'));button.setObjectName('navButton');button.setCheckable(True)
            button.clicked.connect(lambda _,index=i:self.main_tabs.setCurrentIndex(index));self.navigation.addButton(button,i);nav.addWidget(button);self.buttons.append(button)
        nav.addStretch();foot=QLabel('30 Hz desired P/V/A\nOne frozen model per run\nOffline trajectory generation');foot.setObjectName('sideFootnote');nav.addWidget(foot);layout.addWidget(sidebar)
        body=QVBoxLayout();body.setSpacing(0);body.setContentsMargins(0,0,0,0);layout.addLayout(body,1)
        header=QFrame();header.setObjectName('topBar');h=QVBoxLayout(header);h.setContentsMargins(22,14,22,14)
        self.title=QLabel();self.title.setObjectName('shellPageTitle');h.addWidget(self.title);self.subtitle=note('');h.addWidget(self.subtitle);body.addWidget(header);body.addWidget(self.main_tabs,1)
        self.model_page=PVAModelPage(self.root);self.model_page.model_requested.connect(self.select_model)
        from .model_workspace import RecordingsWorkspace
        self.recordings_page=RecordingsWorkspace(self.root)
        self.ppo_page=PVAPlannerPage(self.root,'ppo');self.training_page=self.ppo_page
        self.mppi_page=PVAPlannerPage(self.root,'mppi')
        self.rehearsal_page=QWidget();re=QVBoxLayout(self.rehearsal_page);self.rehearsal_tabs=QTabWidget();re.addWidget(self.rehearsal_tabs)
        current=QWidget();c=QVBoxLayout(current);self.rehearsal_tabs.addTab(current,'Direct PVA rehearsals')
        row=QHBoxLayout();self.rehearsals=QComboBox();row.addWidget(self.rehearsals,1);refresh=QPushButton('Refresh');refresh.clicked.connect(self.refresh_rehearsals);row.addWidget(refresh)
        open_button=QPushButton('Open selected');open_button.clicked.connect(self.open_rehearsal);row.addWidget(open_button);c.addLayout(row)
        self.rehearsal_status=note('Generate a rehearsal in PPO or MPPI, then inspect its saved command and predicted drone/cable here.');c.addWidget(self.rehearsal_status)
        self.inspector=RehearsalWorkspace(self.root,inspection_only=True);c.addWidget(self.inspector,1)
        historical=QWidget();hist=QVBoxLayout(historical);self.rehearsal_tabs.addTab(historical,'Historical force-policy flights')
        hist.addWidget(note('Original force PPO checkpoints and saved flight rehearsals retain their original models, commands and coordinates. They are distinct from new jerk/PVA policies.'))
        self.history_load=QPushButton('Open historical rehearsal workspace');hist.addWidget(self.history_load);self.historical=None
        def load_history():
            if self.historical is None:self.historical=RehearsalWorkspace(self.root);hist.addWidget(self.historical,1);self.history_load.hide()
            self.page_changed(self.main_tabs.currentIndex())
        self.history_load.clicked.connect(load_history)
        from .adaptation_check_page import AdaptationCheckPage
        self.adaptation_check_page=AdaptationCheckPage(self.root)
        self.recordings_page.current.comparison_requested.connect(self.open_flight_comparison)
        self.adaptation_check_page.progress_page.model_requested.connect(lambda p:self.select_model(p,'ppo'))
        for page,(title,_) in zip([self.model_page,self.recordings_page,self.ppo_page,self.mppi_page,self.rehearsal_page,self.adaptation_check_page],PAGES):self.main_tabs.addTab(page,title)
        self.main_tabs.currentChanged.connect(self.page_changed);self.rehearsal_tabs.currentChanged.connect(lambda _:self.page_changed(self.main_tabs.currentIndex()))
        self.ppo_page.changed.connect(self.refresh_rehearsals);self.mppi_page.changed.connect(self.refresh_rehearsals)
        self.refresh_rehearsals();self.page_changed(0)

    def select_model(self,path,method='ppo'):
        page=self.ppo_page if method=='ppo' else self.mppi_page;page.select_model(path);self.main_tabs.setCurrentWidget(page)

    def refresh_rehearsals(self):
        selected=self.rehearsals.currentData();self.rehearsals.clear()
        for p in sorted((self.root/'runs/rehearsals_pva').glob('*/rehearsal.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True):
            m=read_json(p,{});self.rehearsals.addItem(f'{m.get("planner","PVA")} · {p.parent.name}',str(p.parent))
        self.rehearsals.setCurrentIndex(max(0,self.rehearsals.findData(selected)))

    def open_rehearsal(self):
        if self.rehearsals.currentData():
            try:self.inspector.clear_result();self.inspector.load_result(self.rehearsals.currentData())
            except (OSError,ValueError,KeyError) as e:self.rehearsal_status.setText(str(e))

    def open_flight_comparison(self,batch,take):
        page=self.adaptation_check_page
        if page.worker is not None and page.worker.isRunning():return
        i=page.batches.findData(batch)
        if i<0:page.batches.addItem(Path(batch).name,batch);i=page.batches.count()-1
        page.batches.setCurrentIndex(i);page.takes.setCurrentIndex(page.takes.findText(take));page.views.setCurrentIndex(0);self.main_tabs.setCurrentIndex(5)

    def page_changed(self,index):
        self.title.setText(PAGES[index][0]);self.subtitle.setText(PAGES[index][1]);self.buttons[index].setChecked(True)
        self.ppo_page.set_page_active(index==2);self.mppi_page.set_page_active(index==3)
        self.inspector.set_page_active(index==4 and self.rehearsal_tabs.currentIndex()==0)
        if self.historical:self.historical.set_page_active(index==4 and self.rehearsal_tabs.currentIndex()==1)
        self.adaptation_check_page.set_page_active(index==5)

    def closeEvent(self,event):
        ready=[p.shutdown() for p in (self.model_page,self.ppo_page,self.mppi_page,self.inspector,self.adaptation_check_page)]
        if self.historical:ready.append(self.historical.shutdown())
        if not all(ready):event.ignore();QTimer.singleShot(250,self.close);return
        super().closeEvent(event)
