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
       ('MPPI','Refine the selected M0 motion using position B-splines'),
       ('Rehearsals','Inspect saved commands, predicted motion, and complete CSV exports'),
       ('Flight comparison','Track model generations, compare predictions, and inspect real flights')]


class PVAResearchWindow(QMainWindow):
    def __init__(self,project_root,*legacy_configs):
        super().__init__();self.root=Path(project_root);load_application_font();self.setWindowTitle('AeroWhip Research · PVA')
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
        from .recordings_workspace import RecordingsWorkspace
        self.recordings_page=RecordingsWorkspace(self.root)
        self.mppi_page=PVAPlannerPage(self.root,'mppi')
        self.rehearsal_page=QWidget();re=QVBoxLayout(self.rehearsal_page);self.rehearsal_tabs=QTabWidget();re.addWidget(self.rehearsal_tabs)
        current=QWidget();c=QVBoxLayout(current);self.rehearsal_tabs.addTab(current,'Direct PVA rehearsals')
        row=QHBoxLayout();self.rehearsals=QComboBox();row.addWidget(self.rehearsals,1);refresh=QPushButton('Refresh');refresh.clicked.connect(self.refresh_rehearsals);row.addWidget(refresh)
        open_button=QPushButton('Open selected');open_button.clicked.connect(self.open_rehearsal);row.addWidget(open_button);c.addLayout(row)
        self.export_flights_button=QPushButton('Export flights…');self.export_flights_button.clicked.connect(self.export_flights);row.addWidget(self.export_flights_button)
        self.rehearsal_status=note('Generate a rehearsal in MPPI, then inspect its saved command and predicted drone/cable here.');c.addWidget(self.rehearsal_status)
        self.inspector=RehearsalWorkspace(self.root,inspection_only=True);c.addWidget(self.inspector,1)
        from .recorded_takes_replay import RecordedTakesReplay
        self.recorded_replay=RecordedTakesReplay(self.root);self.rehearsal_tabs.addTab(self.recorded_replay,'Recorded takes')
        if hasattr(self.recordings_page,'preliminary'):self.recordings_page.preliminary.replay_requested.connect(self.open_recorded_take)
        from .adaptation_check_page import AdaptationCheckPage
        self.adaptation_check_page=AdaptationCheckPage(self.root)
        from .model_evolution_page import ModelEvolutionPage
        self.evolution_page=ModelEvolutionPage(self.root)
        from .system_comparison_page import SystemComparisonPage
        self.system_comparison_page=SystemComparisonPage(self.root)
        self.flight_workspace=QTabWidget()
        self.flight_workspace.addTab(self.system_comparison_page,'Study overview')
        self.flight_workspace.addTab(self.adaptation_check_page,'Flights by model')
        self.flight_workspace.addTab(self.evolution_page,'Compare generations')
        self.system_comparison_page.flight_requested.connect(self.open_flight_comparison)
        self.system_comparison_page.details_requested.connect(lambda:self.flight_workspace.setCurrentWidget(self.evolution_page))
        self.evolution_page.flight_requested.connect(self.open_flight_comparison)
        self.flight_workspace.currentChanged.connect(lambda _:self.page_changed(self.main_tabs.currentIndex()))
        self.recordings_page.current.comparison_requested.connect(self.open_flight_comparison)
        for page,(title,_) in zip([self.model_page,self.recordings_page,self.mppi_page,self.rehearsal_page,self.flight_workspace],PAGES):self.main_tabs.addTab(page,title)
        self.main_tabs.currentChanged.connect(self.page_changed);self.rehearsal_tabs.currentChanged.connect(lambda _:self.page_changed(self.main_tabs.currentIndex()))
        self.mppi_page.changed.connect(self.refresh_rehearsals)
        self.refresh_rehearsals();self.page_changed(0)
        if (self.root/'config/pva/flight_replay.json').is_file():QTimer.singleShot(200,self.restore_flight_replay)
        elif (self.root/'config/pva/replay.json').is_file():QTimer.singleShot(200,self.restore_replay)

    def restore_flight_replay(self):
        selection=read_json(self.root/'config/pva/flight_replay.json',{})
        batch=Path(selection.get('batch',''))
        batch=(batch if batch.is_absolute() else self.root/batch).resolve()
        take=selection.get('take','')
        if not (batch/'flight_take'/f'{take}.csv').is_file():
            self.restore_replay();return
        page=self.adaptation_check_page
        page.camera.setCurrentText(selection.get('camera','Side XZ'))
        page.speed.setCurrentText(selection.get('speed','0.25×'))
        page.whip.setChecked(selection.get('whip_only',True))
        self.open_flight_comparison(str(batch),take)

    def restore_replay(self):
        """Open the explicitly saved replay selection without generating a job."""
        selection=read_json(self.root/'config/pva/replay.json',{})
        path=Path(selection.get('rehearsal',''))
        path=(path if path.is_absolute() else self.root/path).resolve()
        index=self.rehearsals.findData(str(path))
        if index<0:return  # The selected artifact may be absent in a source-only checkout.
        self.rehearsals.setCurrentIndex(index);self.open_rehearsal();self.main_tabs.setCurrentWidget(self.rehearsal_page)
        if self.inspector.arrays is None:return
        self.inspector.camera.setCurrentText(selection.get('camera','Side XZ'))
        self.inspector.speed.setCurrentText(selection.get('speed','0.25×'))
        import numpy as np
        times=self.inspector.arrays['prediction_time_s']
        frame=int(np.argmin(np.abs(times-float(selection.get('time_s',0.)))))
        self.inspector.timeline.setValue(frame)
        QTimer.singleShot(100,lambda:self.inspector.draw_frame(frame) if self.inspector.arrays is not None else None)
        self.rehearsal_status.setText('Saved replay loaded. Press Play to inspect the original predicted whip and recovery.')

    def select_model(self,path,method='mppi'):
        page=self.mppi_page;page.select_model(path);self.main_tabs.setCurrentWidget(page)

    def refresh_rehearsals(self):
        selected=self.rehearsals.currentData();self.rehearsals.clear()
        for p in sorted((self.root/'runs/rehearsals_pva').glob('*/rehearsal.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True):
            if (p.parent/'ARCHIVED').exists():continue
            m=read_json(p,{});self.rehearsals.addItem(f'{m.get("planner","PVA")} · {p.parent.name}',str(p.parent))
        self.rehearsals.setCurrentIndex(max(0,self.rehearsals.findData(selected)))

    def open_rehearsal(self):
        if self.rehearsals.currentData():
            try:self.inspector.clear_result();self.inspector.load_result(self.rehearsals.currentData())
            except (OSError,ValueError,KeyError) as e:self.rehearsal_status.setText(str(e))

    def export_flights(self):
        from .flight_export_dialog import FlightExportDialog
        dialog=FlightExportDialog(self.root,self.rehearsals.currentData(),self)
        dialog.exec()
        if dialog.exported:self.rehearsal_status.setText('Flight commands exported to '+str(dialog.exported))

    def open_flight_comparison(self,batch,take):
        page=self.adaptation_check_page
        if page.worker is not None and page.worker.isRunning():return
        if not page.select_flight(batch,take):return
        page.views.setCurrentIndex(0)
        self.flight_workspace.setCurrentWidget(page);self.main_tabs.setCurrentWidget(self.flight_workspace)
        page.load_flight()

    def open_recorded_take(self,name):
        self.recorded_replay.open_take(name)
        self.rehearsal_tabs.setCurrentWidget(self.recorded_replay);self.main_tabs.setCurrentWidget(self.rehearsal_page)

    def page_changed(self,index):
        self.title.setText(PAGES[index][0]);self.subtitle.setText(PAGES[index][1]);self.buttons[index].setChecked(True)
        self.mppi_page.set_page_active(index==2)
        self.inspector.set_page_active(index==3 and self.rehearsal_tabs.currentIndex()==0)
        self.adaptation_check_page.set_page_active(index==4 and self.flight_workspace.currentWidget() is self.adaptation_check_page)
        self.recorded_replay.set_page_active(index==3 and self.rehearsal_tabs.currentWidget() is self.recorded_replay)

    def closeEvent(self,event):
        ready=[p.shutdown() for p in (self.model_page,self.mppi_page,self.inspector,self.adaptation_check_page,self.recorded_replay,self.evolution_page)]
        if not all(ready):event.ignore();QTimer.singleShot(250,self.close);return
        super().closeEvent(event)
