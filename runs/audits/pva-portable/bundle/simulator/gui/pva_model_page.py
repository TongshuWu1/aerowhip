"""Model lineage, normalized bootstrap fitting and essential evidence."""
from pathlib import Path
from PySide6.QtCore import Signal,QTimer,QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QComboBox,QTabWidget,
    QTableWidget,QTableWidgetItem,QHeaderView,QPlainTextEdit,QGroupBox)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from simulator.workflow import read_json,stamp
from .pva_workspace import model_paths,model_label
from .research_widgets import note,BackgroundJob


class PVAModelPage(QWidget):
    model_requested=Signal(str,str)
    def __init__(self,root):
        super().__init__();self.root=Path(root);self.last_progress=None
        outer=QVBoxLayout(self);outer.setContentsMargins(18,12,18,14);self.tabs=QTabWidget();outer.addWidget(self.tabs)
        library=QWidget();body=QVBoxLayout(library);self.tabs.addTab(library,'Model library')
        banner=note('Select a fitted model for the next plan. Each PPO run, MPPI plan and exported trajectory keeps its own frozen model.');banner.setObjectName('pipelineBanner');body.addWidget(banner)
        row=QHBoxLayout();self.models=QComboBox();row.addWidget(self.models,1);refresh=QPushButton('Refresh');refresh.clicked.connect(self.refresh);row.addWidget(refresh);body.addLayout(row)
        self.identity=note('');body.addWidget(self.identity)
        self.table=QTableWidget(0,3);self.table.setHorizontalHeaderLabels(['Component','Value / state','Interpretation']);self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch);self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers);body.addWidget(self.table,1)
        row=QHBoxLayout()
        for method in ('ppo','mppi'):
            button=QPushButton('Use for '+method.upper()+' setup');button.setObjectName('primaryButton');button.clicked.connect(lambda _,m=method:self.choose(m));row.addWidget(button)
        open_files=QPushButton('Open model files');open_files.clicked.connect(self.open_model);row.addWidget(open_files);body.addLayout(row)
        self.models.currentIndexChanged.connect(self.inspect)

        fitting=QWidget();body=QVBoxLayout(fitting);self.tabs.addTab(fitting,'Fit model')
        body.addWidget(note('Current bootstrap: normalized cf7/adp0 flights only. Fresh nominal estimates and fresh residuals; old preliminary recordings and old learned models are excluded.'))
        row=QHBoxLayout();self.fit_start=QPushButton('Fit new preliminary M0');self.fit_start.setObjectName('primaryButton');self.fit_start.clicked.connect(self.fit);row.addWidget(self.fit_start)
        self.fit_stop=QPushButton('Stop at update boundary');self.fit_stop.clicked.connect(self.stop);row.addWidget(self.fit_stop);body.addLayout(row)
        row=QHBoxLayout();row.addWidget(QLabel('Fit job'));self.jobs=QComboBox();row.addWidget(self.jobs,1);body.addLayout(row)
        self.fit_status=note('');self.fit_status.setObjectName('pipelineBanner');body.addWidget(self.fit_status)
        self.figure=Figure(layout='constrained',facecolor='white');self.canvas=FigureCanvasQTAgg(self.figure);body.addWidget(self.canvas,1)
        body.addWidget(note('Stages: nominal drone response → drone residual → cable physics → cable residual → coupled whip replay. Plateau stopping retains the best weights. Curves are training objectives, not measured adaptation improvement.'))
        self.log=QPlainTextEdit();self.log.setReadOnly(True);self.log.setMaximumHeight(150);self.log.hide();body.addWidget(self.log)
        show=QPushButton('Show fit log');show.setCheckable(True);show.toggled.connect(self.log.setVisible);body.addWidget(show)
        self.worker=BackgroundJob(root);body.addWidget(self.worker);self.worker.finished.connect(lambda _:self.refresh())
        self.jobs.currentIndexChanged.connect(lambda _:self.poll(force=True))

        evidence=QWidget();body=QVBoxLayout(evidence);self.tabs.addTab(evidence,'Fit diagnostics')
        body.addWidget(note('These RMS values replay training commands from normalized measured initialization. New real flights must establish whether the next trajectory performs better.'))
        self.diagnostics=QTableWidget(0,3);self.diagnostics.setHorizontalHeaderLabels(['Take','Drone RMS [cm]','Cable tip RMS [cm]'])
        self.diagnostics.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch);self.diagnostics.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers);body.addWidget(self.diagnostics,1)
        self.evidence_note=note('Select a completed bootstrap model from the library.');body.addWidget(self.evidence_note)
        history=QWidget();hb=QVBoxLayout(history);self.tabs.addTab(history,'Historical studies')
        hb.addWidget(note('Preserved model fits and diagnostics use their original data and conventions. They do not define the new PVA bootstrap.'))
        load=QPushButton('Open historical model and fit workspace');hb.addWidget(load);self.historical=None
        def open_history():
            if self.historical is None:
                from .model_hub import ModelHub
                self.historical=ModelHub(self.root)
                # Old fixed-budget jobs remain readable; new fitting uses the current path.
                self.historical.tabs.setTabEnabled(1,False);self.historical.use.hide();hb.addWidget(self.historical,1);load.hide()
        load.clicked.connect(open_history);hb.addStretch()
        self.timer=QTimer(self);self.timer.setInterval(2000);self.timer.timeout.connect(self.poll);self.timer.start();self.refresh()

    def refresh(self):
        selected=self.models.currentData();self.models.blockSignals(True);self.models.clear()
        for p in model_paths(self.root):self.models.addItem(model_label(p),str(p.resolve()))
        self.models.setCurrentIndex(max(0,self.models.findData(selected)));self.models.blockSignals(False);self.inspect()
        selected=self.jobs.currentData();self.jobs.blockSignals(True);self.jobs.clear()
        for p in sorted((self.root/'runs/adaptation').glob('*/protocol.json'),reverse=True):
            if read_json(p,{}).get('schema')=='normalized_adp0_cold_pva_bootstrap_v1':self.jobs.addItem(p.parent.name,str(p.parent))
        self.jobs.setCurrentIndex(max(0,self.jobs.findData(selected)));self.jobs.blockSignals(False);self.poll(force=True)

    def inspect(self):
        path=self.models.currentData();m=read_json(path,{}) if path else {};self.table.setRowCount(0)
        if not m:self.identity.setText('No completed model yet. The fit monitor shows progress.');return
        prov=m.get('provenance',{});mass=m.get('mass_measurement',{});c=m['cable'];offset=m.get('recorded_data',{}).get('optitrack_to_attachment_offset_body_m')
        self.identity.setText(self.models.currentText()+'\n'+str(path))
        rows=[('Drone response','Enabled' if m.get('fullstate_execution',{}).get('enabled') else 'Disabled','Effective cmdFullState tracking at OptiTrack origin'),
            ('Drone residual','Bounded acceleration','Fresh bootstrap uses a smooth zero-at-rest gate'),
            ('Cable physics',f'EI {c["EI_n_m2"]:.4g} · Cb {c["Cb_n_m2_s"]:.4g}','DDER with a moving, freely pivoting attachment'),
            ('Cable residual',m.get('motion_residual',{}).get('specification',{}).get('mode','unspecified'),'Learned damping and bounded acceleration correction'),
            ('Separate fixed drag',str(c.get('external_drag_s_inv',0))+' s⁻¹','Zero for active residual-based cable model'),
            ('Tracking origin → attachment',str(offset),'Body-frame metres, rotated using predicted/measured attitude'),
            ('Data-source mass',f'{mass.get("drone_mass_kg",0)*1000:g} g drone + {mass.get("cable_assembly_mass_kg",0)*1000:g} g cable','Model provenance; not an automatic hardware-mass correction'),
            ('Measured state','Normalized OptiTrack pose + cable','Controller logs supply commands and timing only'),
            ('Evidence', 'Training replay' if prov.get('fit_complete') else 'Historical model','Prospective flight performance is shown in Flight comparison')]
        self.table.setRowCount(len(rows))
        for i,row in enumerate(rows):
            for j,value in enumerate(row):self.table.setItem(i,j,QTableWidgetItem(value))
        self.table.resizeRowsToContents()
        diag=read_json(Path(path).parent/'training_diagnostics.json',{});takes=diag.get('takes',{});self.diagnostics.setRowCount(len(takes))
        for i,(name,result) in enumerate(takes.items()):
            for j,value in enumerate([name,*[f'{result[k]["rmse_m"]*100:.2f}' if result[k].get('rmse_m') is not None else 'Unavailable' for k in ('drone','tip')]]):self.diagnostics.setItem(i,j,QTableWidgetItem(value))
        self.evidence_note.setText(diag.get('evidence','This historical model keeps its original diagnostics in Historical studies.'))

    def choose(self,method):
        if self.models.currentData():self.model_requested.emit(self.models.currentData(),method)

    def open_model(self):
        if self.models.currentData():QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.models.currentData()).parent)))

    def fit(self):
        if self.worker.running:return
        if any(read_json(Path(self.jobs.itemData(i))/'status.json',{}).get('status')=='running' for i in range(self.jobs.count())):
            self.fit_status.setText('An existing bootstrap fit is running. Follow it here; no duplicate was started.');return
        job=self.root/'runs/adaptation'/(stamp()+'-pva-M0-bootstrap')
        command=[str(self.root/'.venv/Scripts/python.exe'),'-u',str(self.root/'tools/fit_pva_bootstrap.py'),'--job',str(job),'--prepare','--run']
        self.worker.start(self.root/'runs/model_jobs'/stamp(),command);self.fit_start.setEnabled(False)

    def stop(self):
        if self.jobs.currentData() and read_json(Path(self.jobs.currentData())/'status.json',{}).get('status')=='running':(Path(self.jobs.currentData())/'STOP').touch()

    def poll(self,force=False):
        path=self.jobs.currentData()
        if not path:return
        path=Path(path);status=read_json(path/'status.json',{});progress=read_json(path/'progress.json',{})
        self.fit_status.setText(status.get('status','unknown').upper()+' · '+progress.get('stage','')+'\n'+
            (f'Update {progress.get("update")} of safety ceiling {progress.get("ceiling")}' if 'update' in progress else status.get('error','Best weights are retained; stopping uses practical plateau checks.')))
        self.fit_stop.setEnabled(status.get('status')=='running');self.fit_start.setEnabled(status.get('status')!='running' and not self.worker.running)
        histories=[path/'drone/history.json',path/'cable/physics/history.json',path/'cable/residual/history.json']
        fingerprint=tuple(p.stat().st_mtime_ns if p.exists() else 0 for p in histories)
        if force or fingerprint!=self.last_progress:
            self.last_progress=fingerprint;self.figure.clear();axes=self.figure.subplots(1,3)
            for ax,p,title in zip(axes,histories,['Drone residual','Cable physics','Cable residual']):
                rows=read_json(p,[]);ax.set_title(title,loc='left',fontsize=10);ax.set_xlabel('Updates');ax.set_ylabel('Training loss');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
                if rows:ax.plot([r['update'] for r in rows],[r['loss'] for r in rows],color='#2563eb')
                chosen=[r for r in rows if 'selection_loss' in r]
                if chosen:ax.plot([r['update'] for r in chosen],[r['selection_loss'] for r in chosen],color='#ea580c',label='Rollout selection');ax.legend(fontsize=7)
            self.canvas.draw_idle()
        if self.log.isVisible():
            text=[]
            for name in ('worker.stdout.log','worker.stderr.log'):
                p=path/name
                if p.exists():
                    with p.open('rb') as stream:stream.seek(max(0,p.stat().st_size-12000));text.append(stream.read().decode('utf-8',errors='replace'))
            self.log.setPlainText('\n'.join(text))

    def shutdown(self):
        self.timer.stop();return True
