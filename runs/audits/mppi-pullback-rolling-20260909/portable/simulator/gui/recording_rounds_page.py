"""Import, align, inspect and review immutable recording rounds."""
import json
from pathlib import Path
import sys

from PySide6.QtCore import Qt,QUrl
from PySide6.QtGui import QDesktopServices,QPixmap
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,
    QComboBox,QSpinBox,QLineEdit,QTableWidget,QTableWidgetItem,QFileDialog,QMessageBox,
    QCheckBox,QHeaderView,QScrollArea)

from simulator.workflow import stamp,read_json
from experimental_data.adaptation_rounds import save_review,write_json,preparation_enabled
from .research_widgets import BackgroundJob,note


class RecordingRoundsPage(QWidget):
    def __init__(self,root):
        super().__init__();self.root=Path(root);self.directory=None;self.output=None;self.reports=[]
        layout=QVBoxLayout(self)
        layout.addWidget(note('Recording rounds: adaptation0 is the unadapted baseline. Import original CSV pairs → align → review → save a processed version. No model fitting or PPO training runs here.'))
        row=QHBoxLayout();self.rounds=QComboBox();self.versions=QComboBox()
        row.addWidget(QLabel('Round'));row.addWidget(self.rounds,1);row.addWidget(QLabel('Processed version'));row.addWidget(self.versions,1)
        refresh=QPushButton('Refresh');refresh.clicked.connect(self.refresh);row.addWidget(refresh);layout.addLayout(row)
        row=QHBoxLayout();self.number=QSpinBox();self.number.setRange(0,9999)
        self.parent_round=QComboBox();self.import_notes=QLineEdit();self.import_notes.setPlaceholderText('New round: collection date, controller version, changes since parent…')
        row.addWidget(QLabel('New round number'));row.addWidget(self.number);row.addWidget(QLabel('Parent'));row.addWidget(self.parent_round);row.addWidget(self.import_notes,1)
        self.import_button=QPushButton('Import CSV pairs…');self.import_button.clicked.connect(self.import_pairs);row.addWidget(self.import_button);layout.addLayout(row)
        row=QHBoxLayout();self.reference=QLineEdit();self.reference.setPlaceholderText('Optional exact full-state reference CSV used for this round')
        row.addWidget(self.reference,1);browse=QPushButton('Choose reference…');browse.clicked.connect(self.choose_reference);row.addWidget(browse);layout.addLayout(row)
        self.table=QTableWidget(0,5);self.table.setHorizontalHeaderLabels(['Trial','Status','Offset (s)','Alignment RMS (cm)','Missing cable samples'])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows);self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers);self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setMaximumHeight(170);layout.addWidget(self.table)
        row=QHBoxLayout();self.offset=QLineEdit();self.offset.setPlaceholderText('Optional offset for selected trial (seconds)')
        row.addWidget(self.offset,1);self.process=QPushButton('Process round into new version');self.process.clicked.connect(self.process_selected);row.addWidget(self.process)
        open_button=QPushButton('Open round folder');open_button.clicked.connect(self.open_folder);row.addWidget(open_button);layout.addLayout(row)
        row=QHBoxLayout();self.role=QComboBox();self.role.addItems(['unassigned','adaptation','validation'])
        self.outcome=QComboBox();self.outcome.addItems(['unreviewed','success','failure','aborted'])
        self.checked=QCheckBox('Alignment reviewed');self.start_time=QLineEdit();self.end_time=QLineEdit()
        self.start_time.setPlaceholderText('Pre-contact start (s)');self.end_time.setPlaceholderText('End (s)')
        for w in (QLabel('Trial role'),self.role,QLabel('Outcome'),self.outcome,self.checked,self.start_time,self.end_time):row.addWidget(w)
        layout.addLayout(row)
        row=QHBoxLayout();self.notes=QLineEdit();self.notes.setPlaceholderText('Review notes: contact, intervention, marker direction, known failure…')
        row.addWidget(self.notes,1);self.save=QPushButton('Save review revision');self.save.clicked.connect(self.save_selected_review);row.addWidget(self.save);layout.addLayout(row)
        self.summary=note('');layout.addWidget(self.summary)
        scroll=QScrollArea();scroll.setWidgetResizable(True);self.preview=QLabel();self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter);scroll.setWidget(self.preview);layout.addWidget(scroll,1)
        self.job=BackgroundJob(root);self.job.finished.connect(self.job_done);layout.addWidget(self.job)
        self.rounds.currentIndexChanged.connect(self.select_round);self.versions.currentIndexChanged.connect(self.select_version)
        self.table.currentCellChanged.connect(self.select_trial)
        self.refresh()
        self.organize_workspace()

    def organize_workspace(self):
        """Keep inspection prominent; importing and editing are separate tasks."""
        from PySide6.QtWidgets import QTabWidget
        outer=self.layout();items=[]
        while outer.count():items.append(outer.takeAt(0))
        items[0].widget().setText('Preserved originals → aligned version → reviewed take. Processing creates a new version; it does not fit or train a model.')
        outer.addItem(items[0]);outer.addItem(items[1]);outer.addItem(items[4])
        self.table.verticalHeader().hide();self.table.setMinimumHeight(135);self.table.setMaximumHeight(180)
        tabs=QTabWidget();outer.addWidget(tabs,1)
        inspection=QWidget();view=QVBoxLayout(inspection);tabs.addTab(inspection,'Inspect recording')
        view.addItem(items[9])
        details=QPushButton('Alignment and phase details');details.setCheckable(True);view.addWidget(details)
        view.addItem(items[8]);self.summary.hide();details.toggled.connect(self.summary.setVisible)
        importing=QWidget();form=QVBoxLayout(importing);tabs.addTab(importing,'Import next round')
        form.addWidget(note('Choose the original OptiTrack and controller CSV pairs. Keep the exact FullState reference with the recordings. '
            'The current importer associates one reference with a round; use separate rounds for different references until per-take association is added.'))
        form.addItem(items[2]);form.addItem(items[3]);form.addStretch()
        reviewing=QWidget();review=QVBoxLayout(reviewing);tabs.addTab(reviewing,'Process and review')
        review.addWidget(note('Select a take in the table. Reprocessing creates an immutable aligned version; review annotations are saved separately. '
            'Saved reviews do not, by themselves, make a new take ready for model fitting.'))
        for i in [5,6,7]:review.addItem(items[i])
        review.addStretch();outer.addItem(items[10])

    def error(self,error):QMessageBox.warning(self,'Recording preparation',str(error))

    def refresh(self):
        previous=self.rounds.currentData();self.rounds.blockSignals(True);self.rounds.clear()
        parent=self.parent_round.currentData();self.parent_round.clear();self.parent_round.addItem('None',None)
        paths=sorted((self.root/'data/adaptation_rounds').glob('adaptation*/round.json'),key=lambda p:int(p.parent.name.removeprefix('adaptation')))
        all_paths=paths
        paths=[p for p in paths if preparation_enabled(p.parent)]
        for path in paths:
            self.rounds.addItem(path.parent.name,str(path.parent));self.parent_round.addItem(path.parent.name,path.parent.name)
        index=self.rounds.findData(previous);self.rounds.setCurrentIndex(index if index>=0 else len(paths)-1);self.rounds.blockSignals(False)
        self.parent_round.setCurrentIndex(max(0,self.parent_round.findData(parent)))
        if all_paths:self.number.setValue(max(int(p.parent.name.removeprefix('adaptation')) for p in all_paths)+1)
        self.select_round()

    def select_round(self,*_):
        path=self.rounds.currentData();self.directory=Path(path) if path else None
        self.versions.blockSignals(True);self.versions.clear()
        if self.directory:
            for p in sorted((self.directory/'processed').glob('*/processing.json'),reverse=True):self.versions.addItem(p.parent.name,str(p.parent))
        self.versions.blockSignals(False);self.select_version()

    def select_version(self,*_):
        path=self.versions.currentData();self.output=Path(path) if path else None
        self.reports=read_json(self.output/'processing.json')['reports'] if self.output else []
        self.table.setRowCount(len(self.reports));self.preview.clear()
        for i,r in enumerate(self.reports):
            a=r.get('alignment',{});values=[r['trial_id'],r['status'],f'{a["offset_s"]:.6f}' if a else '—',f'{a["rms_m"]*100:.2f}' if a else '—',str(sum(r.get('missing_cable_samples_per_marker',[])))]
            for j,v in enumerate(values):self.table.setItem(i,j,QTableWidgetItem(v))
        if self.reports:self.table.selectRow(0);self.select_trial()
        else:self.summary.setText('Import a round or process its preserved originals. Review status never implies training readiness.')

    def select_trial(self,*_):
        row=self.table.currentRow()
        if not 0<=row<len(self.reports):return
        r=self.reports[row];self.notes.clear();self.offset.clear();self.start_time.clear();self.end_time.clear();self.checked.setChecked(False)
        self.role.setCurrentIndex(0);self.outcome.setCurrentIndex(0)
        if r['status']=='failed':self.summary.setText(r['error']);self.preview.clear();return
        match=r.get('reference_comparison') or {}
        intervals=', '.join(f'{s["start_s"]:.3f}–{s["end_s"]:.3f} s' for s in r['candidate_motion_intervals']) or 'none detected'
        if match.get('prefix_matches'):
            reference=f'{match["recorded_distinct_samples"]} of {match["reference_samples"]} reference samples match in order'
            if match.get('absent_tail_times_s'):reference+='; final samples were not recorded'
        else:
            reference='reference does not match' if match else 'no reference supplied'
        phases=r.get('execution_phases')
        phase_text=''
        if phases:
            phase_text=(f'Legacy CSV-only maneuver: {phases["csv_start_s"]:.3f}–{phases["csv_end_s"]:.3f} s. '
                'Plot shading: blue = pre-hold, orange = CSV, green = post-hold, gray = command unknown. '
                'Post-hold targets the measured end position; it is not the exported CSV recovery. '
                'Commands stay at the tracked drone origin; the cable attachment is reconstructed separately. ')
        self.summary.setText(f'cf_7 + cable1:c1–c10 only. Logged position updates: {r["logged_position_update_hz"] or 0:.1f} Hz. '
            f'{phase_text}Motion intervals (controller clock): {intervals}. '
            f'{reference}. Missing samples are preserved. Review marker direction, contact window and attachment offset before fitting.')
        image=self.output/r['trial_id']/'review.png'
        if image.is_file():self.preview.setPixmap(QPixmap(str(image)).scaledToWidth(950,Qt.TransformationMode.SmoothTransformation))
        for path in sorted((self.directory/'reviews').glob('*.json'),reverse=True):
            review=read_json(path)
            if review['processed_version']==self.output.name and review['trial_id']==r['trial_id']:
                self.role.setCurrentText(review['role']);self.outcome.setCurrentText(review['outcome']);self.notes.setText(review['notes'])
                self.checked.setChecked(review['alignment_reviewed'])
                for edit,key in ((self.start_time,'precontact_start_s'),(self.end_time,'precontact_end_s')):
                    if review[key] is not None:edit.setText(str(review[key]))
                break

    def choose_reference(self):
        path,_=QFileDialog.getOpenFileName(self,'Exact full-state reference CSV',str(self.root),'CSV (*.csv)')
        if path:self.reference.setText(path)

    def begin_job(self,args):
        if self.job.running:raise ValueError('A processing job is already running')
        folder=self.root/'runs/recording_preparation'/stamp();folder.mkdir(parents=True)
        self.receipt=folder/'receipt.json'
        self.job.start(folder,[sys.executable,'-u','tools/process_recording_round.py','--root',str(self.root),'--receipt',str(self.receipt),*args])
        self.import_button.setEnabled(False);self.process.setEnabled(False)

    def import_pairs(self):
        source=QFileDialog.getExistingDirectory(self,'Folder containing experiment_<trial>.csv and <trial>.csv pairs')
        if not source:return
        args=['--source',source,'--number',str(self.number.value()),'--notes',self.import_notes.text()]
        if self.parent_round.currentData():args+=['--parent',self.parent_round.currentData()]
        if self.reference.text().strip():args+=['--reference',self.reference.text().strip()]
        try:self.begin_job(args)
        except Exception as error:self.error(error)

    def process_selected(self):
        if self.directory is None:return
        args=['--round',str(self.directory)]
        try:
            if self.offset.text().strip():
                row=self.table.currentRow()
                if row<0:raise ValueError('Select the trial to override')
                folder=self.root/'runs/recording_preparation'/stamp();folder.mkdir(parents=True)
                path=folder/'overrides.json';write_json(path,{self.reports[row]['trial_id']:float(self.offset.text())});args+=['--overrides',str(path)]
            self.begin_job(args)
        except Exception as error:self.error(error)

    def job_done(self,code):
        self.import_button.setEnabled(True);self.process.setEnabled(True);self.refresh()
        if self.receipt.is_file():
            receipt=read_json(self.receipt);self.rounds.setCurrentIndex(self.rounds.findData(receipt['round']))
        if code:self.summary.setText('Some files could not be processed. Inspect the failed rows and job log; originals remain preserved.')

    def save_selected_review(self):
        row=self.table.currentRow()
        if self.output is None or row<0:return
        try:
            save_review(self.directory,self.output.name,self.reports[row]['trial_id'],role=self.role.currentText(),
                outcome=self.outcome.currentText(),notes=self.notes.text(),offset_verified=self.checked.isChecked(),
                precontact_start_s=float(self.start_time.text()) if self.start_time.text().strip() else None,
                precontact_end_s=float(self.end_time.text()) if self.end_time.text().strip() else None)
            self.summary.setText('Review revision saved. Dataset is prepared for later work; no fitting or training has run.')
        except Exception as error:self.error(error)

    def open_folder(self):
        if self.directory:QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.directory)))
