"""Recorded flights, physical candidates, and local command correction."""
from pathlib import Path
import shutil
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QPushButton,QLabel,QTableWidget,
    QTableWidgetItem,QFileDialog,QMessageBox,QComboBox,QHeaderView,QSplitter)
from simulator.workflow import read_json,stamp
from .research_widgets import note,BackgroundJob


class AdaptationPage(QWidget):
    def __init__(self,root,parent=None):
        super().__init__(parent);self.root=Path(root);self.last_output=None
        layout=QVBoxLayout(self);layout.setContentsMargins(24,20,24,20)
        layout.addWidget(note('Between flights: import an attempt → compare replay → fit a candidate → refine the force sequence. '
                             'Both successful and failed attempts are useful. PPO/SAC weights stay unchanged.'))
        actions=QHBoxLayout()
        for text,slot in [('Create flight-log template',self.create_template),('Import recorded flight',self.import_recording),('Refresh',self.refresh)]:
            button=QPushButton(text);button.clicked.connect(slot);actions.addWidget(button)
        actions.addStretch();layout.addLayout(actions)
        self.table=QTableWidget(0,4);self.table.setHorizontalHeaderLabels(['Use','Flight','Role','Pre-contact duration'])
        self.table.horizontalHeader().setSectionResizeMode(1,QHeaderView.ResizeMode.Stretch)
        for column in (0,2,3):self.table.horizontalHeader().setSectionResizeMode(column,QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setMaximumHeight(220);layout.addWidget(self.table)
        layout.addWidget(note('Roles come from trial.json: adaptation fits parameters; validation checks complete held-out flights. '
                             'Original tracking, sent commands and controller/IMU logs are preserved.'))
        model_line=QHBoxLayout();model_line.addWidget(QLabel('Model to use'));self.models=QComboBox();model_line.addWidget(self.models,1)
        layout.addLayout(model_line)
        operations=QHBoxLayout()
        for text,mode in [('Compare selected flight','replay'),('Fit cable drag','fit'),('Refine selected force sequence','refine')]:
            button=QPushButton(text);button.clicked.connect(lambda checked=False,m=mode:self.launch(m));operations.addWidget(button)
        layout.addLayout(operations)
        layout.addWidget(note('Fit: fixed geometry, mass, EI and internal damping; bounded drag update. '
                             'Refine: four force-correction knots, fixed cutoff, then a nominal first-contact and recovery check. '
                             'These are simulation candidates; controller verification and robustness checks are still pending.'))
        self.summary=note('No flight data imported yet. Create a template for tomorrow’s logger.');layout.addWidget(self.summary)
        self.preview=QLabel();self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter);self.preview.setMinimumHeight(180)
        layout.addWidget(self.preview,1)
        footer=QHBoxLayout();self.open_button=QPushButton('Open result folder');self.open_button.clicked.connect(self.open_result)
        footer.addWidget(self.open_button);footer.addStretch();layout.addLayout(footer)
        self.job=BackgroundJob(root);self.job.finished.connect(self.job_finished);layout.addWidget(self.job)
        self.refresh()

    def failure(self,error):QMessageBox.warning(self,'Flight adaptation',str(error))

    def refresh(self):
        self.trials=sorted((self.root/'data/flight_trials').glob('*/import.json'))
        self.table.setRowCount(len(self.trials))
        for row,path in enumerate(self.trials):
            meta=read_json(path.parent/'trial.json');diag=read_json(path.parent/'diagnostics.json')
            item=QTableWidgetItem();item.setFlags(Qt.ItemFlag.ItemIsEnabled|Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked);self.table.setItem(row,0,item)
            for col,value in enumerate([meta['trial_id'],meta['role'],f'{diag["duration_s"]:.2f} s'],1):
                self.table.setItem(row,col,QTableWidgetItem(value))
        if self.trials:self.table.selectRow(0)
        previous=self.models.currentData();self.models.clear();self.models.addItem('Active physical baseline',str(self.root/'config/model.json'))
        for path in sorted((self.root/'data/adaptation_jobs').glob('*/result/model.json'),reverse=True):
            self.models.addItem(f'Candidate · {path.parent.parent.name}',str(path))
        index=self.models.findData(previous)
        if index>=0:self.models.setCurrentIndex(index)

    def create_template(self):
        parent=QFileDialog.getExistingDirectory(self,'Choose where to create the flight template')
        if not parent:return
        try:
            from experimental_data.flight_trials import template
            path=Path(parent)/f'flight_template_{stamp()}';template(path)
            for name in ('model','task'):shutil.copy2(self.root/f'config/{name}.json',path/f'{name}.json')
            self.last_output=path;self.summary.setText('Template created. Fill timestamps and coordinate conventions, and replace model/task snapshots if the flight used a different version.');self.open_result()
        except Exception as error:self.failure(error)

    def import_recording(self):
        source=QFileDialog.getExistingDirectory(self,'Select the normalized flight-log folder')
        if not source:return
        try:
            from experimental_data.flight_trials import import_trial
            destination=import_trial(self.root,source);self.refresh()
            self.summary.setText(f'Imported {destination.name}. Raw logs are preserved; launch-state and timing checks passed.')
        except Exception as error:self.failure(error)

    def launch(self,mode):
        if self.job.running:return self.failure('A flight adaptation job is already running.')
        selected=[p.parent for i,p in enumerate(self.trials) if self.table.item(i,0).checkState()==Qt.CheckState.Checked]
        if not selected:return self.failure('Import and select a flight first.')
        row=self.table.currentRow()
        trial=self.trials[row].parent if row>=0 else selected[0]
        directory=self.root/'data/adaptation_jobs'/f'{stamp()}-{mode}'
        self.last_output=directory/'result'
        command=[str(self.root/'.venv/Scripts/python.exe'),'-u','tools/adapt_flight.py',mode,
                 '--model',self.models.currentData(),'--output',str(self.last_output)]
        command+=['--trials',*map(str,selected)] if mode=='fit' else ['--source',str(trial)]
        try:self.job.start(directory,command)
        except Exception as error:self.failure(error)

    def job_finished(self,code):
        if code:return
        report=read_json(self.last_output/'result.json',{})
        metrics=read_json(self.last_output/'metrics.json',{})
        if metrics:
            self.summary.setText(f'Measured-attachment replay: {metrics["boundary"]["marker_rmse_m"]*1000:.1f} mm marker RMSE. '
                f'Force-driven replay: {metrics["coupled"]["marker_rmse_m"]*1000:.1f} mm. '
                f'Drone attachment error: {metrics["root_rmse_m"]*1000:.1f} mm. Large coupled error can include controller mismatch.')
        elif 'validation_improved' in report:
            self.summary.setText(f'Candidate cable drag: {report["parameter_values"]["external_drag_s_inv"]:.4g}/s. '
                +('Held-out flight checks improved.' if report['validation_improved'] else 'Held-out improvement has not been established.')
                +' Active baseline unchanged; review the saved plots and report.')
        else:
            candidate=report.get('candidate',{})
            self.summary.setText(f'Force candidate: {report.get("status","complete")}. '
                f'Nominal valid hit: {candidate.get("valid_hit")}; recovery: {candidate.get("recovered")}. '
                'Saved for review; not released for flight.')
        images=list(self.last_output.glob('tip_replay.png'))+list(self.last_output.glob('*-candidate/tip_replay.png'))+list(self.last_output.glob('force_comparison.png'))
        if images:self.preview.setPixmap(QPixmap(str(images[0])).scaled(950,290,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))
        self.refresh()

    def open_result(self):
        if self.last_output:QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_output)))
