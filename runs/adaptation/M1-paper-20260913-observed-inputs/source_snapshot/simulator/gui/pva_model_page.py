"""Model lineage, reviewed preliminary fitting and essential evidence."""
from pathlib import Path
import sys
from PySide6.QtCore import Signal,QTimer,QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QComboBox,QTabWidget,
    QTableWidget,QTableWidgetItem,QHeaderView,QPlainTextEdit,QGroupBox,QProgressBar)
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
        banner=note('Select a fitted model for the next plan. Each MPPI plan and exported trajectory keeps its own frozen model.');banner.setObjectName('pipelineBanner');body.addWidget(banner)
        row=QHBoxLayout();self.models=QComboBox();row.addWidget(self.models,1);refresh=QPushButton('Refresh');refresh.clicked.connect(self.refresh);row.addWidget(refresh);body.addLayout(row)
        self.identity=note('');body.addWidget(self.identity)
        self.table=QTableWidget(0,3);self.table.setHorizontalHeaderLabels(['Component','Value / state','Interpretation']);self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch);self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers);body.addWidget(self.table,1)
        row=QHBoxLayout()
        for method in ('mppi',):
            button=QPushButton('Use for '+method.upper()+' setup');button.setObjectName('primaryButton');button.clicked.connect(lambda _,m=method:self.choose(m));row.addWidget(button)
        open_files=QPushButton('Open model files');open_files.clicked.connect(self.open_model);row.addWidget(open_files);body.addLayout(row)
        self.models.currentIndexChanged.connect(self.inspect)

        fitting=QWidget();body=QVBoxLayout(fitting);self.tabs.addTab(fitting,'Fit model')
        self.experiment=read_json(self.root/'config/experiment.json',{})
        fresh=self.experiment.get('schema')=='unseen_system_experiment_v1'
        body.addWidget(note('Review preliminary takes, commands, timing and data roles before preparing an M0 fit. For M0 → M1 adaptation and comparisons, use Flight comparison and the reviewed raw-data workflow.'))
        row=QHBoxLayout();self.fit_start=QPushButton('Fit new preliminary M0');self.fit_start.setObjectName('primaryButton');self.fit_start.clicked.connect(self.fit);row.addWidget(self.fit_start)
        self.fit_stop=QPushButton('Stop at update boundary');self.fit_stop.clicked.connect(self.stop);row.addWidget(self.fit_stop);body.addLayout(row)
        row=QHBoxLayout();row.addWidget(QLabel('Fit job'));self.jobs=QComboBox();row.addWidget(self.jobs,1);body.addLayout(row)
        self.fit_status=note('');self.fit_status.setObjectName('pipelineBanner');body.addWidget(self.fit_status)
        self.fit_progress=QProgressBar();self.fit_progress.setRange(0,100);self.fit_progress.setValue(0);body.addWidget(self.fit_progress)
        if fresh and not self.experiment.get('preliminary_batch'):
            self.fit_start.setEnabled(False);self.fit_stop.setEnabled(False)
            self.fit_status.setText('Waiting for new preliminary recordings and reviewed fit inputs. No fit has started.')
        self.figure=Figure(layout='constrained',facecolor='white');self.canvas=FigureCanvasQTAgg(self.figure);body.addWidget(self.canvas,1)
        body.addWidget(note('Stages: drone response → drone residual → cable physics → cable residual → coupled replay checks. Plateau stopping retains the best weights. Update counts show the safety ceiling; completion can occur earlier.'))
        self.log=QPlainTextEdit();self.log.setReadOnly(True);self.log.setMaximumHeight(150);self.log.hide();body.addWidget(self.log)
        show=QPushButton('Show fit log');show.setCheckable(True);show.toggled.connect(self.log.setVisible);body.addWidget(show)
        self.worker=BackgroundJob(root);body.addWidget(self.worker);self.worker.finished.connect(lambda _:self.refresh())
        self.jobs.currentIndexChanged.connect(lambda _:self.poll(force=True))

        evidence=QWidget();body=QVBoxLayout(evidence);self.tabs.addTab(evidence,'Fit diagnostics')
        body.addWidget(note('Retrospective prediction errors from measured initialization. Training and validation roles are shown for preliminary fits. New real flights must establish whether the next trajectory performs better.'))
        self.diagnostics=QTableWidget(0,3);self.diagnostics.setHorizontalHeaderLabels(['Take','Drone RMS [cm]','Cable tip RMS [cm]'])
        self.diagnostics.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch);self.diagnostics.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers);body.addWidget(self.diagnostics,1)
        self.evidence_note=note('Select a completed bootstrap model from the library.');body.addWidget(self.evidence_note)
        self.timer=QTimer(self);self.timer.setInterval(2000);self.timer.timeout.connect(self.poll);self.timer.start();self.refresh()

    def refresh(self):
        selected=self.models.currentData();self.models.blockSignals(True);self.models.clear()
        for p in model_paths(self.root):self.models.addItem(model_label(p),str(p.resolve()))
        self.models.setCurrentIndex(max(0,self.models.findData(selected)));self.models.blockSignals(False);self.inspect()
        selected=self.jobs.currentData();self.jobs.blockSignals(True);self.jobs.clear()
        for p in sorted((self.root/'runs/adaptation').glob('*/protocol.json'),reverse=True):
            if read_json(p,{}).get('schema') in ('normalized_adp0_cold_pva_bootstrap_v1','preliminary_pva_bootstrap_v1'):self.jobs.addItem(p.parent.name,str(p.parent))
        self.jobs.setCurrentIndex(max(0,self.jobs.findData(selected)));self.jobs.blockSignals(False);self.poll(force=True)

    def inspect(self):
        path=self.models.currentData();m=read_json(path,{}) if path else {};self.table.setRowCount(0)
        if not m:self.identity.setText('No completed model yet. The fit monitor shows progress.');return
        prov=m.get('provenance',{});mass=m.get('mass_measurement',{});c=m['cable'];offset=m.get('recorded_data',{}).get('optitrack_to_attachment_offset_body_m')
        self.identity.setText(self.models.currentText()+'\n'+str(path))
        rows=[('Drone response','Enabled' if m.get('fullstate_execution',{}).get('enabled') else 'Disabled','Effective cmdFullState tracking at OptiTrack origin'),
            ('Drone residual','Bounded acceleration','Fresh bootstrap uses a smooth zero-at-rest gate'),
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
            self.fit_status.setText('An existing bootstrap fit is running. Follow it here; no duplicate was started.');return
        if experiment.get('schema')=='unseen_system_experiment_v1':
            value=experiment.get('fit_job');job=self.root/value if value else None
            if job is None or read_json(job/'status.json',{}).get('status')!='prepared':
                self.fit_status.setText('No unstarted reviewed preliminary job. Existing results remain available; prepare a new job before another fit.');return
            script=job/'code_snapshot/tools/fit_preliminary.py'
            if not script.exists():
                self.fit_status.setText('Reviewed fit source snapshot is missing. No job was started.');return
            command=[sys.executable,'-u',str(script),'--job',str(job)]
            self.active_fit=job;self.worker.start(self.root/'runs/model_jobs'/stamp(),command);self.fit_start.setEnabled(False);return
        self.fit_status.setText('Prepare a reviewed preliminary job first. The retired normalized bootstrap cannot be launched here.')

    def stop(self):
        path=getattr(self,'active_fit',None) if self.worker.running else self.jobs.currentData()
        if path and Path(path).exists():(Path(path)/'STOP').touch()

    def poll(self,force=False):
        active=getattr(self,'active_fit',None)
        if self.worker.running and active is not None and (active/'protocol.json').exists() and self.jobs.findData(str(active))<0:
            self.jobs.blockSignals(True);self.jobs.addItem(active.name,str(active));self.jobs.setCurrentIndex(self.jobs.count()-1);self.jobs.blockSignals(False)
        path=self.jobs.currentData()
        if not path:return
        path=Path(path);status=read_json(path/'status.json',{});progress=read_json(path/'progress.json',{})
        self.fit_status.setText(status.get('status','unknown').upper()+' · '+progress.get('stage','')+'\n'+
            (f'Update {progress.get("update")} of safety ceiling {progress.get("ceiling")}' if 'update' in progress else status.get('error','Best weights are retained; stopping uses practical plateau checks.')))
        self.fit_stop.setEnabled(status.get('status')=='running')
        fresh=read_json(self.root/'config/experiment.json',{}).get('schema')=='unseen_system_experiment_v1'
        self.fit_start.setEnabled(fresh and status.get('status')=='prepared' and not self.worker.running)
        if status.get('status')=='running' and 'update' not in progress:self.fit_progress.setRange(0,0)
        else:
            self.fit_progress.setRange(0,int(progress.get('ceiling',100)));self.fit_progress.setValue(int(progress.get('update',0)) if status.get('status')!='completed' else self.fit_progress.maximum())
        self.fit_progress.setFormat('Completed' if status.get('status')=='completed' else ('Update %v / %m ceiling' if 'update' in progress else status.get('status','Waiting')))
        residual=path/'cable/residual_full_whip/history.json'
        if not residual.exists():residual=path/'cable/residual/history.json'
        histories=[path/'drone/history.json',path/'cable/physics/history.json',residual]
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
            logs=sorted(path.glob('*.log'),key=lambda p:p.stat().st_mtime_ns,reverse=True)[:3]
            for p in logs:
                with p.open('rb') as stream:
                    stream.seek(max(0,p.stat().st_size-12000));text.append(p.name+'\n'+stream.read().decode('utf-8',errors='replace'))
            self.log.setPlainText('\n'.join(text))

    def shutdown(self):
        self.timer.stop();return True
