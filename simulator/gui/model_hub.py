"""Model selection and explicit adaptation jobs; opening the page never fits."""
from pathlib import Path
import sys
from PySide6.QtCore import Signal, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,
    QComboBox,QTabWidget,QTableWidget,QTableWidgetItem,QHeaderView,QCheckBox,QScrollArea)
from simulator.workflow import read_json,stamp
from .research_widgets import note,BackgroundJob


class ModelHub(QWidget):
    model_requested=Signal(str)
    page_requested=Signal(int)

    def __init__(self,root):
        super().__init__();self.root=Path(root)
        outer=QVBoxLayout(self);outer.setContentsMargins(18,12,18,14)
        self.tabs=QTabWidget();outer.addWidget(self.tabs)
        library=QWidget();body=QVBoxLayout(library);self.tabs.addTab(library,'Model library')
        row=QHBoxLayout();row.addWidget(QLabel('Inspect model'))
        self.models=QComboBox();row.addWidget(self.models,1)
        refresh=QPushButton('Refresh');refresh.clicked.connect(self.refresh);row.addWidget(refresh);body.addLayout(row)
        self.identity=note('');self.identity.setObjectName('pipelineBanner');body.addWidget(self.identity)
        self.components=QTableWidget(0,3);self.components.setHorizontalHeaderLabels(['Component','Configuration','Source / meaning'])
        self.components.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.components.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.components.verticalHeader().hide();body.addWidget(self.components,1)
        body.addWidget(note('Selection here is for inspection. “Use for new PPO setup” stages a future run; saved policies, exports and flight ghosts keep their frozen models.'))
        row=QHBoxLayout();self.use=QPushButton('Use for new PPO setup');self.use.setObjectName('primaryButton')
        self.use.clicked.connect(lambda:self.model_requested.emit(self.models.currentData()))
        row.addWidget(self.use);open_model=QPushButton('Open model files');open_model.clicked.connect(self.open_model);row.addWidget(open_model)
        fit_shortcut=QPushButton('Fit model from flight data');fit_shortcut.setObjectName('primaryButton')
        fit_shortcut.clicked.connect(lambda:self.tabs.setCurrentIndex(1));row.addWidget(fit_shortcut)
        row.addStretch();body.addLayout(row)
        self.models.currentIndexChanged.connect(self.inspect_model)

        jobs=QWidget();body=QVBoxLayout(jobs);self.tabs.addTab(jobs,'Fit model')
        body.addWidget(note('This workspace starts with no collected data and no fitted M0. The next step is preliminary recording in the independent Isaac Lab simulator. The inherited adaptation runner requires a reviewed, compatible command/log batch; its five-take workflow must be adapted for preliminary M0 fitting.'))
        row=QHBoxLayout();row.addWidget(QLabel('Flight batch'));self.batches=QComboBox();row.addWidget(self.batches,1);body.addLayout(row)
        self.readiness=note('');body.addWidget(self.readiness)
        self.prepare=QPushButton('Fit drone + cable model');self.prepare.setObjectName('primaryButton');self.prepare.clicked.connect(self.prepare_job);body.addWidget(self.prepare)
        body.addWidget(note('Fits drone physics/residual, cable physics/residual, then validates the combined model. Results are saved separately. No PPO training starts.'))
        self.advanced_toggle=QPushButton('Show individual fit stages');self.advanced_toggle.setCheckable(True);body.addWidget(self.advanced_toggle)
        advanced=QWidget();advanced_body=QVBoxLayout(advanced);body.addWidget(advanced);advanced.hide();self.advanced_toggle.toggled.connect(advanced.setVisible)
        main_body=body;body=advanced_body
        row=QHBoxLayout();row.addWidget(QLabel('Prepared job'));self.jobs=QComboBox();row.addWidget(self.jobs,1);body.addLayout(row)
        self.job_info=note('');body.addWidget(self.job_info)
        row=QHBoxLayout();self.stage=QComboBox()
        for label,key in [('1 · Fit drone physics and residual','drone'),('2 · Fit cable physics and residual','cable_batched'),
                          ('3 · Refine attitude response','attitude_refine'),('4 · Evaluate baseline','baseline'),('5 · Validate combined model','validate')]:
            self.stage.addItem(label,key)
        row.addWidget(self.stage,1);self.run=QPushButton('Run selected stage');self.run.clicked.connect(self.run_stage);row.addWidget(self.run);body.addLayout(row)
        body=main_body
        self.job=BackgroundJob(root);body.addWidget(self.job);self.job.finished.connect(self.job_finished)
        self.stop=QPushButton('Stop after current fit update');self.stop.clicked.connect(self.stop_job);body.addWidget(self.stop)
        self.result_note=note('New candidates require review and explicit selection. Fit diagnostics are not real-flight performance.');body.addWidget(self.result_note)
        body.addStretch()
        self.jobs.currentIndexChanged.connect(self.update_job_controls)
        self.batches.currentIndexChanged.connect(self.update_job_controls)

        evidence=QWidget();self.evidence_layout=QVBoxLayout(evidence);self.tabs.addTab(evidence,'Fit evidence')
        self.evidence_layout.addWidget(note('Model fitting diagnostics only. Real-flight progress remains in Adaptation Check.'))
        self.evidence_button=QPushButton('Load historical physics and residual diagnostics');self.evidence_layout.addWidget(self.evidence_button)
        self.evidence_button.clicked.connect(self.load_evidence);self.evidence=None
        self.adapted_evidence=None;adapted=QPushButton('Review saved adaptation fits');adapted.clicked.connect(self.load_adapted_evidence);self.evidence_layout.addWidget(adapted)
        workflow=QWidget();body=QVBoxLayout(workflow);self.tabs.addTab(workflow,'Workflow')
        for title,description,index in [
            ('1. Review recordings','Preserve raw OptiTrack and command logs; establish clock alignment and exclusions.',1),
            ('2. Compare real execution','Score the take against its exact preflight saved prediction before fitting.',6),
            ('3. Fit and review','Use Adaptation jobs. Compare model-fit diagnostics separately from real-flight outcomes.',0),
            ('4. Optimize a policy','Select the reviewed model, then explicitly start a named PPO run.',2),
            ('5. Rehearse and export','Inspect the complete command trajectory and its saved model prediction.',4)]:
            b=QPushButton(title);b.clicked.connect(lambda _,i=index:self.page_requested.emit(i));body.addWidget(b);body.addWidget(note(description))
        body.addStretch();self.refresh()

    def refresh(self):
        previous=self.models.currentData();self.models.blockSignals(True);self.models.clear()
        workspace=read_json(self.root/'config/research_workspace.json',{})
        self.models.addItem(workspace.get('model_label','Current research workspace'),str((self.root/'config/research_30hz/model.json').resolve()))
        for p in sorted((self.root/'data/model_candidates').glob('*/model.json')):
            if read_json(p,{}).get('adaptation',{}).get('training'):
                self.models.addItem(p.parent.name,str(p.resolve()))
        self.models.setCurrentIndex(max(0,self.models.findData(previous)));self.models.blockSignals(False);self.inspect_model()
        from experimental_data.adaptation_check import discover_batches
        previous=self.batches.currentData();self.batches.clear()
        for p in discover_batches(self.root):self.batches.addItem(f'{p.parent.name} / {p.name}',str(p))
        self.batches.setCurrentIndex(max(0,self.batches.findData(previous)))
        previous=self.jobs.currentData();self.jobs.blockSignals(True);self.jobs.clear()
        for p in sorted((self.root/'runs/adaptation').glob('*/protocol.json'),reverse=True):
            self.jobs.addItem(p.parent.name,str(p.parent))
        self.jobs.setCurrentIndex(max(0,self.jobs.findData(previous)));self.jobs.blockSignals(False);self.update_job_controls()

    def inspect_model(self):
        path=self.models.currentData();model=read_json(path,{}) if path else {}
        self.use.setEnabled(bool(model));self.components.setRowCount(0)
        if not model:self.identity.setText('Model unavailable.');return
        self.identity.setText(self.models.currentText()+'\n'+str(path))
        rows=[]
        for key,title in [('fullstate_execution','Drone response'),('motion_residual','Cable residual')]:
            item=model.get(key)
            if item is not None:rows.append((title,'Enabled' if item.get('enabled') else 'Disabled',Path(item['checkpoint']).name if item.get('checkpoint') else str(item)))
        cable=model.get('cable',{})
        for key,label,unit in [('EI_n_m2','Cable bending stiffness','N·m²'),('Cb_n_m2_s','Cable internal damping','N·m²·s'),('external_drag_s_inv','Fixed external drag','s⁻¹')]:
            rows.append((label,str(cable.get(key,'not specified')),unit))
        offset=model.get('recorded_data',{}).get('optitrack_to_attachment_offset_body_m')
        rows.append(('Tracked origin → attachment',', '.join(f'{v:.6f}' for v in offset) if offset else 'Not specified','Body-frame metres; rotated by measured orientation'))
        mass=model.get('mass_measurement',{}) or {'drone_mass_kg':model.get('point_mass',{}).get('mass_kg',0), 'cable_assembly_mass_kg':cable.get('bare_cable_mass_kg',0)+sum(cable.get('moving_marker_masses_kg',[]))}
        rows.append(('Drone / cable mass',f'{1000*mass.get("drone_mass_kg",0):g} / {1000*mass.get("cable_assembly_mass_kg",0):g}','grams, including measured cable assembly'))
        current=read_json(self.root/'config/current_vehicle.json',{})
        if current:
            rows.append(('Current hardware mass',f'{1000*current["drone_mass_kg"]:g} g drone + {1000*current["cable_assembly_mass_kg"]:g} g cable',
                         'Hardware observation; saved model and original CSV retain their own mass'))
        rows.append(('Measured state','Raw OptiTrack only','Controller log supplies commands and timing'))
        rows.append(('Assessment','Simulation / fit evidence','Real-flight improvement requires new recorded flights'))
        self.components.setRowCount(len(rows))
        for i,row in enumerate(rows):
            for j,value in enumerate(row):self.components.setItem(i,j,QTableWidgetItem(value))
        self.components.resizeRowsToContents()

    def open_model(self):
        if self.models.currentData():QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.models.currentData()).parent)))

    def load_evidence(self):
        if self.evidence is None:
            from .model_workspace import ModelWorkspace
            self.evidence=ModelWorkspace(self.root);self.evidence_layout.addWidget(self.evidence,1)
        self.evidence.show()
        if self.adapted_evidence:self.adapted_evidence.hide()

    def load_adapted_evidence(self):
        if self.adapted_evidence is None:
            from .adaptation_progress_page import AdaptationProgressPage
            self.adapted_evidence=AdaptationProgressPage(self.root)
            self.adapted_evidence.tabs.setTabVisible(0,False);self.adapted_evidence.tabs.setCurrentIndex(1)
            self.adapted_evidence.model_requested.connect(self.model_requested.emit)
            self.evidence_layout.addWidget(self.adapted_evidence,1)
        self.adapted_evidence.refresh();self.adapted_evidence.show()
        if self.evidence:self.evidence.hide()

    def update_job_controls(self):
        from experimental_data.adaptation_check import flight_names
        running=self.job.running;enabled=not running
        count=len(flight_names(self.batches.currentData())) if self.batches.currentData() else 0
        self.readiness.setText(f'{count} / 5 paired takes. '+('Ready to check alignment and fit the reviewed batch.' if count==5 else 'Collect five paired takes for the current fitting workflow.'))
        self.prepare.setEnabled(enabled and count==5)
        self.run.setEnabled(enabled and bool(self.jobs.currentData()))
        self.stage.setEnabled(enabled);self.jobs.setEnabled(not running);self.batches.setEnabled(not running)
        self.stop.setEnabled(running)
        path=self.jobs.currentData()
        protocol=read_json(Path(path)/'protocol.json',{}) if path else {}
        self.job_info.setText(('Frozen source: '+str(protocol.get('source_model'))+'\nBatch: '+str(protocol.get('batch'))) if protocol else 'No prepared adaptation jobs. Retired M1 remains outside the active library.')

    def launch(self,args):
        logdir=self.root/'runs/model_jobs'/stamp()
        try:self.job.start(logdir,[str(self.root/'.venv/Scripts/python.exe'),'-u',str(self.root/'tools/model_job.py'),*args])
        except (ValueError,OSError) as error:self.job.status.setText(str(error))
        self.update_job_controls()

    def prepare_job(self):
        if not self.prepare.isEnabled():return
        self.active_fit=self.root/'runs/adaptation'/(stamp()+'-M0-adaptation')
        self.launch(['all','--job',str(self.active_fit),'--batch',self.batches.currentData()])

    def run_stage(self):
        if not self.run.isEnabled():return
        self.active_fit=Path(self.jobs.currentData())
        self.launch([self.stage.currentData(),'--job',str(self.active_fit)])

    def stop_job(self):
        if self.job.running and hasattr(self,'active_fit'):
            self.active_fit.mkdir(parents=True,exist_ok=True)
            (self.active_fit/'STOP').touch();self.job.status.setText('Stop requested; waiting for the next stage/update boundary.')

    def job_finished(self,code):
        self.refresh();self.result_note.setText('Job completed. Review its saved evidence before selecting a model.' if code==0 else 'Job stopped or failed. Inspect the log; no model was selected automatically.')

    def shutdown(self):
        if self.job.running:
            self.stop_job();return False
        return True
