"""Model lineage, reviewed preliminary fitting and essential evidence."""
from pathlib import Path
import sys
import time
from PySide6.QtCore import Signal,QTimer,QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QComboBox,QTabWidget,
    QTableWidget,QTableWidgetItem,QHeaderView,QPlainTextEdit,QGroupBox,QProgressBar,QCheckBox)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from simulator.workflow import read_json,stamp
from .pva_workspace import model_paths,model_label
from .research_widgets import note,BackgroundJob
from . import fit_monitor
from .command_correction_page import CommandCorrectionPage


class PVAModelPage(QWidget):
    model_requested=Signal(str,str)
    def __init__(self,root):
        super().__init__();self.root=Path(root);self.last_progress=None;self.last_job_scan=0;self.job_entries=[]
        outer=QVBoxLayout(self);outer.setContentsMargins(18,12,18,14);self.tabs=QTabWidget();outer.addWidget(self.tabs)
        library=QWidget();body=QVBoxLayout(library);self.tabs.addTab(library,'Models')
        banner=note('Select a fitted model for the next plan. Each MPPI plan and exported trajectory keeps its own frozen model.');banner.setObjectName('pipelineBanner');body.addWidget(banner)
        row=QHBoxLayout();self.models=QComboBox();row.addWidget(self.models,1);refresh=QPushButton('Refresh');refresh.clicked.connect(self.refresh);row.addWidget(refresh);body.addLayout(row)
        self.identity=note('');body.addWidget(self.identity)
        self.table=QTableWidget(0,3);self.table.setHorizontalHeaderLabels(['Component','Value / state','Interpretation']);self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch);self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers);body.addWidget(self.table,1)
        row=QHBoxLayout()
        for method in ('mppi',):
            button=QPushButton('Use for '+method.upper()+' setup');button.setObjectName('primaryButton');button.clicked.connect(lambda _,m=method:self.choose(m));row.addWidget(button)
        open_files=QPushButton('Open model files');open_files.clicked.connect(self.open_model);row.addWidget(open_files);body.addLayout(row)
        view_fit=QPushButton('View fitting history');view_fit.clicked.connect(self.view_model_fit);row.addWidget(view_fit)
        self.models.currentIndexChanged.connect(self.inspect)

        fitting=QWidget();body=QVBoxLayout(fitting);self.tabs.addTab(fitting,'Live fitting')
        self.experiment=read_json(self.root/'config/experiment.json',{})
        fresh=self.experiment.get('schema')=='unseen_system_experiment_v1'
        row=QHBoxLayout();row.addWidget(QLabel('Fit run'));self.jobs=QComboBox();row.addWidget(self.jobs,1)
        self.follow_fit=QCheckBox('Follow active fit');self.follow_fit.setChecked(True);row.addWidget(self.follow_fit)
        refresh_fit=QPushButton('Refresh');refresh_fit.clicked.connect(lambda:self.refresh_jobs(force=True));row.addWidget(refresh_fit)
        open_fit=QPushButton('Open run folder');open_fit.clicked.connect(self.open_fit);row.addWidget(open_fit);body.addLayout(row)
        self.fit_status=note('');self.fit_status.setObjectName('pipelineBanner');body.addWidget(self.fit_status)
        self.fit_detail=note('');body.addWidget(self.fit_detail)
        self.fit_progress=QProgressBar();self.fit_progress.setRange(0,100);self.fit_progress.setValue(0);body.addWidget(self.fit_progress)
        row=QHBoxLayout();self.stage_cards=[]
        for title in ('Vehicle parameters','Vehicle residual','Cable parameters','Cable residual'):
            card=QGroupBox(title);layout=QVBoxLayout(card);value=QLabel('—');value.setStyleSheet('font-size: 20px; font-weight: 600; color: #172033;')
            detail=note('Waiting');layout.addWidget(value);layout.addWidget(detail);row.addWidget(card,1);self.stage_cards.append((card,value,detail))
        body.addLayout(row)
        self.figure=Figure(layout='constrained',facecolor='white');self.canvas=FigureCanvasQTAgg(self.figure);self.canvas.setMinimumHeight(260);body.addWidget(self.canvas,1)
        row=QHBoxLayout();row.addWidget(note('Evaluated loss and best loss · lower is better · each stage has its own objective · refreshes every 2 s'),1)
        self.log_scale=QCheckBox('Log scale');row.addWidget(self.log_scale);body.addLayout(row)
        self.stopping_note=note('');body.addWidget(self.stopping_note)
        self.log=QPlainTextEdit();self.log.setReadOnly(True);self.log.setMaximumHeight(150);self.log.hide();body.addWidget(self.log)
        row=QHBoxLayout();show=QPushButton('Show fit log');show.setCheckable(True);show.toggled.connect(self.log.setVisible);row.addWidget(show);row.addStretch()
        self.fit_start=QPushButton('Start reviewed preliminary M0');self.fit_start.clicked.connect(self.fit);row.addWidget(self.fit_start)
        self.fit_stop=QPushButton('Stop at update boundary');self.fit_stop.clicked.connect(self.stop);row.addWidget(self.fit_stop);body.addLayout(row)
        if fresh and not self.experiment.get('preliminary_batch'):self.fit_start.setEnabled(False)
        self.worker=BackgroundJob(root);body.addWidget(self.worker);self.worker.hide();self.worker.finished.connect(lambda _:self.refresh())
        self.jobs.activated.connect(self.select_job)
        self.follow_fit.toggled.connect(lambda _:self.poll(force=True))
        self.log_scale.toggled.connect(lambda _:self.poll(force=True))

        self.correction=CommandCorrectionPage(self.root);self.tabs.addTab(self.correction,'Command correction')
        evidence=QWidget();body=QVBoxLayout(evidence);self.tabs.addTab(evidence,'Saved diagnostics')
        body.addWidget(note('Retrospective prediction errors from measured initialization. Training and validation roles are shown for preliminary fits. New real flights must establish whether the next trajectory performs better.'))
        self.diagnostics=QTableWidget(0,3);self.diagnostics.setHorizontalHeaderLabels(['Take','Drone RMS [cm]','Cable tip RMS [cm]'])
        self.diagnostics.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch);self.diagnostics.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers);body.addWidget(self.diagnostics,1)
        self.evidence_note=note('Select a completed bootstrap model from the library.');body.addWidget(self.evidence_note)
        self.timer=QTimer(self);self.timer.setInterval(2000);self.timer.timeout.connect(self.poll);self.timer.start();self.refresh()
        if self.jobs.currentData() and fit_monitor.read(Path(self.jobs.currentData())/'status.json',{}).get('status')=='running':self.tabs.setCurrentIndex(1)

    def refresh(self):
        selected=self.models.currentData();self.models.blockSignals(True);self.models.clear()
        for p in model_paths(self.root):self.models.addItem(model_label(p),str(p.resolve()))
        self.models.setCurrentIndex(max(0,self.models.findData(selected)));self.models.blockSignals(False);self.inspect()
        self.refresh_jobs(force=True);self.poll(force=True)

    def refresh_jobs(self,force=False):
        if not force and time.monotonic()-self.last_job_scan<4:return
        self.last_job_scan=time.monotonic();self.job_entries=fit_monitor.discover(self.root)
        selected=self.jobs.currentData();current=next((j for j in self.job_entries if str(j['path'])==selected),None)
        active=next((j for j in self.job_entries if j['status'].get('status')=='running'),None)
        if self.follow_fit.isChecked() and active and (current is None or active['modified']>=current['modified']):selected=str(active['path'])
        self.jobs.blockSignals(True);self.jobs.clear()
        for j in self.job_entries:self.jobs.addItem(j['label'],str(j['path']))
        self.jobs.setCurrentIndex(max(0,self.jobs.findData(selected)));self.jobs.blockSignals(False)

    def select_job(self,index):
        self.follow_fit.setChecked(False);self.poll(force=True)

    def open_fit(self):
        if self.jobs.currentData():QDesktopServices.openUrl(QUrl.fromLocalFile(self.jobs.currentData()))

    def view_model_fit(self):
        value=self.models.currentData()
        if not value:return
        path=Path(value);model=fit_monitor.read(path,{})
        source=model.get('provenance',{}).get('source_job')
        job=Path(source) if source else path.parent.parent
        if not job.is_absolute():job=self.root/job
        self.follow_fit.setChecked(False);self.refresh_jobs(force=True)
        index=self.jobs.findData(str(job))
        if index>=0:self.jobs.setCurrentIndex(index);self.tabs.setCurrentIndex(1);self.poll(force=True)

    def inspect(self):
        path=self.models.currentData();m=read_json(path,{}) if path else {};self.table.setRowCount(0)
        if not m:self.identity.setText('No completed model yet. The fit monitor shows progress.');return
        prov=m.get('provenance',{});mass=m.get('mass_measurement',{});c=m['cable'];offset=m.get('recorded_data',{}).get('optitrack_to_attachment_offset_body_m')
        self.identity.setText(self.models.currentText()+'\n'+str(path))
        rows=[('Drone response','Enabled' if m.get('fullstate_execution',{}).get('enabled') else 'Disabled','Effective cmdFullState tracking at OptiTrack origin'),
            ('Drone residual','Disabled' if prov.get('drone_residual_enabled') is False else 'Bounded acceleration',
                'Nominal quadrotor response only' if prov.get('drone_residual_enabled') is False else 'Fresh bootstrap uses a smooth zero-at-rest gate'),
            ('Cable physics',f'EI {c["EI_n_m2"]:.4g} · Cb {c["Cb_n_m2_s"]:.4g}','DDER with a moving, freely pivoting attachment'),
            ('Cable residual',m.get('motion_residual',{}).get('specification',{}).get('mode','enabled') if m.get('motion_residual',{}).get('enabled') else 'Disabled','Enabled only when the selected model contains a learned correction'),
            ('Cable velocity damping',str(c.get('external_drag_s_inv',0))+' s⁻¹','Effective coefficient saved in this model'),
            ('Tracking origin → attachment',str(offset),'Body-frame metres, rotated using predicted/measured attitude'),
            ('Data-source mass',f'{mass.get("drone_mass_kg",0)*1000:g} g drone + {mass.get("cable_assembly_mass_kg",0)*1000:g} g cable','Model provenance; not an automatic hardware-mass correction'),
            ('Measured state',prov.get('fit_state','See saved source convention'),'Controller XYZ is an OptiTrack-derived timing check, not an independent sensor'),
            ('Evidence', 'Retrospective fit diagnostics' if prov.get('fit_complete') else 'Development candidate','Prospective flight performance is shown in Flight comparison')]
        self.table.setRowCount(len(rows))
        for i,row in enumerate(rows):
            for j,value in enumerate(row):self.table.setItem(i,j,QTableWidgetItem(value))
        self.table.resizeRowsToContents()
        diag=read_json(Path(path).parent/'window_diagnostics.json',{})
        if diag:
            takes=diag.get('takes',{});self.diagnostics.setColumnCount(4);self.diagnostics.setRowCount(len(takes))
            self.diagnostics.setHorizontalHeaderLabels(['Take / role','Drone 2 s RMS [cm]','Tip 1 s / measured root [cm]','Coupled 2 s tip RMS [cm]'])
            coupled=read_json(Path(path).parent/'coupled_diagnostics.json',{}).get('takes',{})
            for i,(name,result) in enumerate(takes.items()):
                values=[name+' · '+result['role'],*[f'{result[k]["fitted"]["rmse_m"]*100:.2f}' for k in ('drone','cable_tip')]]
                error=coupled.get(name,{}).get('fitted',{}).get('tip',{}).get('rmse_m')
                values.append(f'{error*100:.2f}' if error is not None else 'Unavailable')
                for j,value in enumerate(values):self.diagnostics.setItem(i,j,QTableWidgetItem(value))
            self.evidence_note.setText(prov.get('quality_note','')+'\n'+diag['evidence']+' Coupled columns show one representative window per take, using predicted attachment.')
            return
        diag=read_json(Path(path).parent/'training_diagnostics.json',{});takes=diag.get('takes',{});self.diagnostics.setRowCount(len(takes))
        self.diagnostics.setColumnCount(3)
        self.diagnostics.setHorizontalHeaderLabels(['Take','Drone RMS [cm]','Cable tip RMS [cm]'])
        for i,(name,result) in enumerate(takes.items()):
            for j,value in enumerate([name,*[f'{result[k]["rmse_m"]*100:.2f}' if result[k].get('rmse_m') is not None else 'Unavailable' for k in ('drone','tip')]]):self.diagnostics.setItem(i,j,QTableWidgetItem(value))
        self.evidence_note.setText(diag.get('evidence','Open the model files for its saved diagnostics. No prospective validation is implied.'))

    def choose(self,method):
        if self.models.currentData():self.model_requested.emit(self.models.currentData(),method)

    def open_model(self):
        if self.models.currentData():QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.models.currentData()).parent)))

    def fit(self):
        if self.worker.running:return
        experiment=read_json(self.root/'config/experiment.json',{})
        if experiment.get('schema')=='unseen_system_experiment_v1' and not experiment.get('preliminary_batch'):
            self.fit_status.setText('Collect and review preliminary recordings first. No historical batch will be used.');return
        if any(read_json(Path(self.jobs.itemData(i))/'status.json',{}).get('status')=='running' for i in range(self.jobs.count())):
            self.fit_status.setText('An existing fit is running. Follow it here; no duplicate was started.');return
        if experiment.get('schema')=='unseen_system_experiment_v1':
            value=experiment.get('fit_job');job=self.root/value if value else None
            if job is None or read_json(job/'status.json',{}).get('status')!='prepared':
                self.fit_status.setText('No unstarted reviewed preliminary job. Existing results remain available; prepare a new job before another fit.');return
            script=job/'code_snapshot/tools/fit_preliminary.py'
            if not script.exists():
                self.fit_status.setText('Reviewed fit source snapshot is missing. No job was started.');return
            command=[sys.executable,'-u',str(script),'--job',str(job)]
            self.active_fit=job;self.worker.show();self.worker.start(self.root/'runs/model_jobs'/stamp(),command);self.fit_start.setEnabled(False);return
        self.fit_status.setText('Prepare a reviewed preliminary job first. The retired normalized bootstrap cannot be launched here.')

    def stop(self):
        path=getattr(self,'active_fit',None) if self.worker.running else self.jobs.currentData()
        if path and Path(path).exists() and fit_monitor.snapshot(path)['stop_supported']:(Path(path)/'STOP').touch()

    def poll(self,force=False):
        self.refresh_jobs(force=force)
        path=self.jobs.currentData()
        if not path:
            self.fit_status.setText('No fitting runs yet')
            self.fit_detail.setText('Reviewed preliminary and model-adaptation runs will appear here automatically.')
            self.fit_progress.hide();self.fit_stop.hide();self.stopping_note.clear()
            self.fit_start.setEnabled(False)
            if force or self.last_progress!='empty':
                self.last_progress='empty';self.figure.clear()
                axis=self.figure.subplots();axis.axis('off');axis.text(.5,.5,'Loss curves appear when fitting starts',ha='center',va='center',color='#64748b')
                for _,value,detail in self.stage_cards:value.setText('—');detail.setText('Waiting')
                self.canvas.draw_idle()
            return
        view=fit_monitor.snapshot(path);status=view['status'];progress=view['progress'];full=view['protocol'].get('full_update',{})
        stage=progress.get('stage','Preparing').replace('_',' ')
        state=status.get('status','unavailable')
        lineage=f"{full.get('parent_id','?')} → {full.get('candidate_id','?')}" if full else 'Preliminary M0'
        self.fit_status.setText(f'{lineage}  ·  {state.upper()}  ·  {stage}')
        update=progress.get('update');parts=[]
        if update is not None:parts.append(f'Update {update}')
        if progress.get('best_loss') is not None:parts.append(f"Best stage loss {progress['best_loss']:.6g}")
        if status.get('error'):parts.append(status['error'])
        if any(v['state']=='Finalizing' for v in view['stages']):parts.append('Finalizing the selected checkpoint')
        takes=view['protocol'].get('takes',{})
        if takes:parts.append(f'{len(takes)} whips')
        parts.append(Path(path).name)
        self.fit_detail.setText(' · '.join(parts));self.fit_detail.setToolTip(str(path))
        self.stopping_note.setText(view['stopping'])
        self.fit_stop.setVisible(view['stop_supported']);self.fit_stop.setEnabled(view['stop_supported'] and state=='running')
        fresh=self.experiment.get('schema')=='unseen_system_experiment_v1'
        self.fit_start.setEnabled(fresh and not full and state=='prepared' and not self.worker.running)
        ceiling=progress.get('ceiling') or progress.get('budget',{}).get('maximum_updates')
        self.fit_progress.setVisible(state=='running' and isinstance(ceiling,(int,float)) and ceiling>0)
        if isinstance(ceiling,(int,float)) and ceiling>0:
            self.fit_progress.setRange(0,int(ceiling));self.fit_progress.setValue(int(update or 0));self.fit_progress.setFormat('Update %v / %m cap')
        for (card,value,detail),v in zip(self.stage_cards,view['stages']):
            chosen=v['result'].get('numerically_verified') is True
            value.setText(('Selected ' if chosen else 'Best ')+f"{v['best']:.5g}" if v['best'] is not None else '—')
            text=v['state']
            if v['update'] is not None:text+=f" · {int(v['update'])} updates"
            if v['reduction'] is not None:text+=f"\n{v['reduction']:.1f}% lower than stage start"
            detail.setText(text)
            card.setStyleSheet('QGroupBox {background: #eff6ff; border: 1px solid #93c5fd; border-radius: 6px; margin-top: 10px; padding-top: 8px;} QGroupBox::title {subcontrol-origin: margin; left: 10px;}' if v['active'] else '')
        if force or view['fingerprint']!=self.last_progress:
            self.last_progress=view['fingerprint'];self.figure.clear();axes=self.figure.subplots(2,2).flat
            for ax,v in zip(axes,view['stages']):
                ax.set_title(v['title'],loc='left',fontsize=10,fontweight='bold');ax.set_xlabel('Update',fontsize=9);ax.set_ylabel('Loss',fontsize=9)
                ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=8)
                points=v['points']
                if points:
                    x=[r[0] for r in points];y=[r[1] for r in points];best=[];minimum=float('inf')
                    for _,loss,saved in points:
                        minimum=min(minimum,loss if saved is None else saved);best.append(minimum)
                    ax.plot(x,y,color='#2563eb',lw=1.5,label='Evaluated loss')
                    ax.plot(x,best,color='#d97706',lw=1.5,ls='--',label='Best loss')
                    if v['selected_update'] is not None and v['best'] is not None:
                        ax.plot(v['selected_update'],v['best'],'*',color='#15803d',ms=10,label='Selected')
                    if self.log_scale.isChecked() and min(y+best)>0:ax.set_yscale('log')
                    ax.legend(frameon=False,fontsize=8,loc='best')
                    ax.margins(x=.03)
                else:
                    text='Waiting for the first saved loss' if v['active'] else 'No saved history yet'
                    ax.text(.5,.5,text,ha='center',va='center',transform=ax.transAxes,color='#64748b',fontsize=9)
            self.canvas.draw_idle()
        if self.log.isVisible():
            text=[]
            logs=sorted(Path(path).glob('*.log'),key=fit_monitor.modified,reverse=True)[:3]
            for p in logs:
                try:
                    with p.open('rb') as stream:
                        stream.seek(max(0,p.stat().st_size-12000));text.append(p.name+'\n'+stream.read().decode('utf-8',errors='replace'))
                except OSError:continue
            self.log.setPlainText('\n'.join(text))

    def shutdown(self):
        self.timer.stop();self.correction.shutdown();return True
