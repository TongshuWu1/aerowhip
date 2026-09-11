"""Guided, local lab workflow; heavy work runs through the public CLI.

Opening the application only reads the experiment state. Model fitting,
planning, data import and exports each require an explicit operator action.
"""
from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import sys

from PySide6.QtCore import Qt, QProcess, QProcessEnvironment, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QPalette, QColor
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
    QFileDialog, QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow, QPlainTextEdit, QProgressBar,
    QPushButton, QScrollArea, QSizePolicy, QSplitter, QStackedWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from simulator.gui.theme import APP_STYLE, load_application_font


PAGES = (
    ('Overview', 'One study. A preserved baseline. Twenty new flights.'),
    ('Plan & export', 'Freeze a complete command sequence before collecting its flights.'),
    ('Record & review', 'Import paired native logs and record the review for each assigned take.'),
    ('Update model', 'Build M1 and M2 from the reviewed adaptation takes.'),
    ('Results', 'Compare physical target error and prediction error on the final recordings.'),
)
STAGE_LABELS = {'M0': '1 · M0 flights → build M1', 'M1': '2 · M1 flights → build M2',
                'final': '3 · Final M0 / M2 comparison'}
ROLE_LABELS = {'adaptation': 'Training', 'validation': 'Validation', 'final': 'Final test'}


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def note(text=''):
    widget = QLabel(text)
    widget.setWordWrap(True)
    widget.setObjectName('mutedText')
    return widget


def button(label, callback, primary=False):
    widget = QPushButton(label)
    if primary:
        widget.setObjectName('primaryButton')
    widget.clicked.connect(callback)
    return widget


def table(headers):
    widget = QTableWidget(0, len(headers))
    widget.setHorizontalHeaderLabels(headers)
    widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    widget.verticalHeader().hide()
    widget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    widget.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    widget.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
    widget.setAlternatingRowColors(True)
    return widget


def fill(widget, rows):
    widget.setRowCount(len(rows))
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            item = QTableWidgetItem(str(value))
            item.setToolTip(str(value))
            widget.setItem(i, j, item)
    widget.resizeRowsToContents()


class InspectionDialog(QDialog):
    """Allow a reader thread to finish before its native viewer is destroyed."""

    inspector = None

    def closeEvent(self, event):
        if self.inspector is not None and not self.inspector.shutdown():
            event.ignore()
            QTimer.singleShot(150, self.close)
            return
        super().closeEvent(event)


class LabWindow(QMainWindow):
    """The colleague-facing application; backend injection supports isolated UI tests."""

    def __init__(self, root, workspace_factory=None):
        super().__init__()
        self.root = Path(root).resolve()
        if workspace_factory is None:
            from deployment.lab_workflow import LabWorkspace
            workspace_factory = LabWorkspace
        self.workspace_factory = workspace_factory
        self.snapshot = {}
        self.slot_rows = []
        self.process = None
        self.process_action = None
        self.log_path = None
        self.replay_dialog = None
        load_application_font()
        self.setWindowTitle('AeroWhip · Lab')
        self.resize(1450, 930)
        self.setMinimumSize(1120, 680)
        screen = QApplication.primaryScreen()
        if screen and screen.availableGeometry().width() >= 1152:
            area = screen.availableGeometry()
            self.resize(min(1450, area.width() - 32), min(930, area.height() - 48))
        self.setStyleSheet(APP_STYLE + '''
            QWidget { font-family: "Segoe UI", "DejaVu Sans", sans-serif; }
            QLabel#labHeading { font-size: 25pt; font-weight: 700; color: #122039; }
            QLabel#labCardValue { font-size: 21pt; font-weight: 700; color: #122039; }
            QLabel#labEyebrow { color: #526781; font-size: 9pt; font-weight: 700; }
            QTableWidget { alternate-background-color: #f8fafc; }
            QFrame#jobBar { background: #ffffff; border-top: 1px solid #dce3ed; }
        ''')
        palette = self.palette()
        for role, color in ((QPalette.ColorRole.Window, '#f5f7fb'),
                            (QPalette.ColorRole.WindowText, '#172033'),
                            (QPalette.ColorRole.Base, '#ffffff'),
                            (QPalette.ColorRole.Text, '#172033')):
            palette.setColor(role, QColor(color))
        self.setPalette(palette)
        shell = QWidget()
        shell.setObjectName('applicationShell')
        self.setCentralWidget(shell)
        layout = QHBoxLayout(shell)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        sidebar = QFrame()
        sidebar.setObjectName('sideBar')
        sidebar.setFixedWidth(215)
        nav = QVBoxLayout(sidebar)
        nav.setContentsMargins(17, 29, 17, 22)
        brand = QLabel('AeroWhip\nLAB')
        brand.setObjectName('brandTitle')
        nav.addWidget(brand)
        sub = QLabel('PLAN  /  RECORD  /  REFINE')
        sub.setObjectName('brandSubtitle')
        nav.addWidget(sub)
        nav.addSpacing(30)
        self.navigation = QButtonGroup(self)
        self.nav_buttons = []
        for index, (title, _) in enumerate(PAGES):
            item = button(f'{index + 1:02d}   {title.replace("&", "&&")}',
                          lambda checked=False, i=index: self.select_page(i))
            item.setObjectName('navButton')
            item.setCheckable(True)
            self.navigation.addButton(item, index)
            self.nav_buttons.append(item)
            nav.addWidget(item)
        nav.addStretch()
        guide = button('Operator guide', self.open_guide)
        guide.setObjectName('navButton')
        nav.addWidget(guide)
        footer = QLabel('30 Hz P/V/A commands\nOffline planning\nAll study files stay in this repo')
        footer.setWordWrap(True)
        footer.setObjectName('sideFootnote')
        nav.addWidget(footer)
        layout.addWidget(sidebar)
        body = QVBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        layout.addLayout(body, 1)
        header = QFrame()
        header.setObjectName('topBar')
        heading = QVBoxLayout(header)
        heading.setContentsMargins(28, 20, 28, 16)
        top = QHBoxLayout()
        self.title = QLabel()
        self.title.setObjectName('shellPageTitle')
        top.addWidget(self.title)
        top.addStretch()
        top.addWidget(QLabel('Study'))
        self.study = QComboBox()
        self.study.setMinimumWidth(205)
        self.study.currentIndexChanged.connect(self.refresh)
        top.addWidget(self.study)
        self.refresh_button = button('Refresh', self.refresh)
        top.addWidget(self.refresh_button)
        heading.addLayout(top)
        self.subtitle = note()
        heading.addWidget(self.subtitle)
        body.addWidget(header)
        self.pages = QStackedWidget()
        body.addWidget(self.pages, 1)
        self._build_overview()
        self._build_plan()
        self._build_recordings()
        self._build_update()
        self._build_results()
        self._build_job_bar(body)
        self.device.currentIndexChanged.connect(self.refresh_update)
        self.device.currentIndexChanged.connect(self.refresh_results)
        self.select_page(0)
        self.refresh()
        self.timer = QTimer(self)
        self.timer.setInterval(3000)
        self.timer.timeout.connect(self.poll)
        self.timer.start()

    def page(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(28, 22, 28, 24)
        layout.setSpacing(17)
        scroll.setWidget(content)
        self.pages.addWidget(scroll)
        return layout

    def card(self, caption, value, detail):
        frame = QFrame()
        frame.setObjectName('metricCard')
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(19, 15, 19, 17)
        title = QLabel(caption.upper())
        title.setObjectName('labEyebrow')
        layout.addWidget(title)
        main = QLabel(value)
        main.setObjectName('labCardValue')
        layout.addWidget(main)
        layout.addWidget(note(detail))
        return frame, main

    def _build_overview(self):
        layout = self.page()
        heading = QLabel('Your whip experiment')
        heading.setObjectName('labHeading')
        layout.addWidget(heading)
        layout.addWidget(note('Keep the existing M0 and preliminary data. Collect new whip flights, '
                              'update twice, then compare M0 and M2 on held-out final flights.'))
        cards = QHBoxLayout()
        for caption, value, detail in (
            ('Baseline', 'M0 retained', 'No preliminary recollection'),
            ('Collection', '5 + 5 + 10', 'Two update batches and five final pairs'),
            ('Primary metric', '3D error · cm', 'Closest observed tip-to-target distance'),
        ):
            frame, _ = self.card(caption, value, detail)
            cards.addWidget(frame, 1)
        layout.addLayout(cards)
        self.baseline_status = note()
        self.baseline_status.setObjectName('pipelineBanner')
        layout.addWidget(self.baseline_status)
        create = QGroupBox('Study workspace')
        row = QHBoxLayout(create)
        self.study_name = QLineEdit()
        self.study_name.setPlaceholderText('New study name, e.g. lab-20260912')
        row.addWidget(self.study_name, 1)
        self.create_button = button('Create study', self.create_study, True)
        row.addWidget(self.create_button)
        row.addWidget(button('Open study files', self.open_study))
        layout.addWidget(create)
        self.overview_status = note()
        layout.addWidget(self.overview_status)
        self.sequence = table(['Step', 'Action', 'Current state'])
        self.sequence.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.sequence.setMinimumHeight(228)
        layout.addWidget(self.sequence)
        row = QHBoxLayout()
        row.addWidget(button('Plan / export commands', lambda: self.select_page(1), True))
        row.addWidget(button('Import next flight', lambda: self.select_page(2)))
        row.addStretch()
        layout.addLayout(row)
        layout.addWidget(note('Training takes are 001, 002 and 004 in each update batch. '
                              'Validation takes are 003 and 005. Every final comparison take is excluded from fitting.'))
        layout.addStretch()

    def _build_plan(self):
        layout = self.page()
        row = QHBoxLayout()
        row.addWidget(QLabel('Model generation'))
        self.plan_generation = QComboBox()
        self.plan_generation.addItems(['M0', 'M1', 'M2'])
        self.plan_generation.currentIndexChanged.connect(self.refresh_plan)
        row.addWidget(self.plan_generation)
        row.addStretch()
        row.addWidget(QLabel('Compute device'))
        self.device = QComboBox()
        self.device.addItems(['cuda', 'cpu'])
        self.device.setToolTip('Use cuda on the lab NVIDIA GPU. CPU is available for diagnostic use.')
        row.addWidget(self.device)
        layout.addLayout(row)
        self.plan_status = note()
        self.plan_status.setObjectName('pipelineBanner')
        layout.addWidget(self.plan_status)
        self.plan_details = table(['Item', 'Frozen value / state'])
        self.plan_details.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.plan_details.setMinimumHeight(295)
        layout.addWidget(self.plan_details)
        row = QHBoxLayout()
        self.plan_button = button('1  Generate MPPI plan', self.start_plan, True)
        self.export_button = button('2  Rehearse + export CSV', self.export_plan, True)
        self.preview_button = button('Preview saved motion', self.preview_plan)
        row.addWidget(self.plan_button)
        row.addWidget(self.export_button)
        row.addWidget(self.preview_button)
        layout.addLayout(row)
        layout.addWidget(note('Planning uses the frozen study setup. Export checks the complete whip and recovery '
                              'and saves the command CSV under exports/<study>/<generation>/. '
                              'Replay the saved motion before using the CSV with your lab sender.'))
        self.export_path = QLineEdit()
        self.export_path.setReadOnly(True)
        self.export_path.setPlaceholderText('The exported CSV path appears here')
        layout.addWidget(self.export_path)
        row = QHBoxLayout()
        row.addWidget(button('Open CSV export folder', self.open_exports))
        row.addStretch()
        layout.addLayout(row)
        layout.addWidget(note('This application prepares commands and analyzes recordings. '
                              'Takeoff, the verified sender, recovery monitoring and landing are operated separately.'))
        layout.addStretch()

    def _build_recordings(self):
        layout = self.page()
        row = QHBoxLayout()
        row.addWidget(QLabel('Collection stage'))
        self.capture_stage = QComboBox()
        for key, label in STAGE_LABELS.items():
            self.capture_stage.addItem(label, key)
        self.capture_stage.currentIndexChanged.connect(self.refresh_slots)
        row.addWidget(self.capture_stage, 1)
        row.addWidget(button('Open raw recording folder', self.open_study))
        layout.addLayout(row)
        self.capture_status = note()
        layout.addWidget(self.capture_status)
        self.slots = table(['Assigned take', 'Model', 'Data role', 'State'])
        self.slots.setMinimumHeight(235)
        self.slots.setMaximumHeight(320)
        self.slots.itemSelectionChanged.connect(self.selected_slot_changed)
        layout.addWidget(self.slots)
        groups = QHBoxLayout()
        intake = QGroupBox('1  Import the native file pair')
        form = QFormLayout(intake)
        self.tracking_file = self.path_field(form, 'OptiTrack CSV', 'Select the original tracking CSV')
        self.controller_file = self.path_field(form, 'Controller CSV', 'Select its matching controller log')
        self.drone_label = QLineEdit()
        self.drone_label.setPlaceholderText('Optional exact rigid-body name')
        form.addRow('Drone rigid body', self.drone_label)
        self.offset = QDoubleSpinBox()
        self.offset.setDecimals(6)
        self.offset.setRange(-1e8, 1e8)
        self.offset.setSingleStep(.001)
        form.addRow('Clock offset [s]', self.offset)
        self.clock_source = QLineEdit()
        self.clock_source.setPlaceholderText('Shared timestamp or observed synchronization event')
        form.addRow('Timing evidence', self.clock_source)
        self.clock_verified = QCheckBox('Clocks independently synchronized')
        form.addRow(self.clock_verified)
        self.estimate_button = button('Estimate timing from recorded motion', self.estimate_timing)
        form.addRow(self.estimate_button)
        self.timing_note = note()
        self.timing_note.hide()
        form.addRow(self.timing_note)
        form.addRow(note('Controller time = OptiTrack time + offset. '
                         'Use timing evidence; do not align measured motion to the desired trajectory.'))
        self.import_button = button('Import pair into selected slot', self.import_pair, True)
        form.addRow(self.import_button)
        self.alignment_button = button('Save corrected timing', self.save_alignment)
        form.addRow(self.alignment_button)
        self.flight_preview_button = button('Inspect flight + saved prediction', self.preview_flight)
        form.addRow(self.flight_preview_button)
        groups.addWidget(intake, 1)
        review = QGroupBox('2  Review the selected take')
        form = QFormLayout(review)
        self.reviewer = QLineEdit()
        self.reviewer.setPlaceholderText('Reviewer name')
        form.addRow('Reviewed by', self.reviewer)
        self.free_end = QDoubleSpinBox()
        self.free_end.setRange(.001, 30.)
        self.free_end.setDecimals(4)
        self.free_end.setValue(1.5)
        self.free_end.setToolTip('Seconds after the first CSV command. Stop before physical contact or intervention. Final comparison requires the full 0–1.5 s interval or an explicit exclusion.')
        form.addRow('Free motion ends [s]', self.free_end)
        self.contact = QComboBox()
        self.contact.addItem('No physical contact in reviewed interval', 'none')
        self.contact.addItem('Contact at or after interval end', 'at_or_after_end')
        form.addRow('Contact observation', self.contact)
        self.review_clock = QCheckBox('Timing alignment reviewed')
        self.review_hardware = QCheckBox('Same hardware and controller')
        self.review_intervention = QCheckBox('No intervention in reviewed interval')
        for check in (self.review_clock, self.review_hardware, self.review_intervention):
            form.addRow(check)
        self.review_notes = QPlainTextEdit()
        self.review_notes.setPlaceholderText('Battery, marker gaps, intervention or exclusion notes')
        self.review_notes.setMaximumHeight(75)
        form.addRow('Notes', self.review_notes)
        self.exclude = QCheckBox('Exclude this take; retain its original files')
        form.addRow(self.exclude)
        self.review_button = button('Save take review', self.save_review, True)
        form.addRow(self.review_button)
        groups.addWidget(review, 1)
        layout.addLayout(groups)
        self.take_status = note('Select an assigned take. Keep every attempt, including failed or excluded takes.')
        layout.addWidget(self.take_status)

    def path_field(self, form, label, placeholder):
        row = QHBoxLayout()
        field = QLineEdit()
        field.setPlaceholderText(placeholder)
        row.addWidget(field, 1)
        row.addWidget(button('Browse…', lambda: self.choose_file(field)))
        form.addRow(label, row)
        return field

    def choose_file(self, field):
        path, _ = QFileDialog.getOpenFileName(self, 'Select original CSV', str(self.root), 'CSV files (*.csv);;All files (*)')
        if path:
            field.setText(path)

    def _build_update(self):
        layout = self.page()
        row = QHBoxLayout()
        row.addWidget(QLabel('Next model'))
        self.update_generation = QComboBox()
        self.update_generation.addItems(['M1', 'M2'])
        self.update_generation.currentIndexChanged.connect(self.refresh_update)
        row.addWidget(self.update_generation)
        row.addStretch()
        layout.addLayout(row)
        self.update_status = note()
        self.update_status.setObjectName('pipelineBanner')
        layout.addWidget(self.update_status)
        self.fit_stages = table(['Stage', 'Purpose', 'Current state'])
        self.fit_stages.setMinimumHeight(290)
        layout.addWidget(self.fit_stages)
        row = QHBoxLayout()
        self.prepare_button = button('1  Freeze reviewed fit inputs', self.prepare_update, True)
        self.fit_button = button('2  Start full model update', self.fit_update, True)
        row.addWidget(self.prepare_button)
        row.addWidget(self.fit_button)
        row.addWidget(button('Review batch first', lambda: self.select_page(2)))
        row.addStretch()
        layout.addLayout(row)
        layout.addWidget(note('Only the preassigned training takes update the model. The retained preliminary '
                              'training data and earlier training takes provide replay. Validation and final '
                              'takes never choose weights. A completed model becomes available on Plan & export.'))
        self.update_details = note()
        layout.addWidget(self.update_details)
        layout.addStretch()

    def _build_results(self):
        layout = self.page()
        self.result_status = note('Final results appear after the held-out comparison flights are imported and reviewed.')
        self.result_status.setObjectName('pipelineBanner')
        layout.addWidget(self.result_status)
        row = QHBoxLayout()
        self.evaluate_button = button('Evaluate final recordings', self.evaluate, True)
        row.addWidget(self.evaluate_button)
        row.addWidget(button('Open result files', self.open_results))
        row.addStretch()
        layout.addLayout(row)
        self.physical_results = table(['Final take', 'Model', 'Closest tip–target [cm]', 'Observed coverage', 'Review'])
        self.physical_results.setMinimumHeight(270)
        layout.addWidget(self.physical_results)
        self.result_summary = note()
        layout.addWidget(self.result_summary)
        self.prediction_results = table(['Model', 'Tip prediction RMS [cm]', 'Drone prediction RMS [cm]', 'Takes'])
        self.prediction_results.setMinimumHeight(140)
        layout.addWidget(self.prediction_results)
        self.result_note = note('Physical error compares the executed M0 and M2 commands. Prediction error compares '
                                'frozen M0, M1 and M2 on the same recorded commands. Lower error is better; '
                                'there is no 5 cm success threshold. Missing observations remain visible.')
        layout.addWidget(self.result_note)
        layout.addStretch()

    def _build_job_bar(self, parent):
        frame = QFrame()
        frame.setObjectName('jobBar')
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(26, 11, 26, 13)
        row = QHBoxLayout()
        self.job_status = note('Ready. Opening the app starts no fitting or planning.')
        row.addWidget(self.job_status, 1)
        self.job_progress = QProgressBar()
        self.job_progress.setMaximumWidth(200)
        self.job_progress.setRange(0, 1)
        self.job_progress.setValue(0)
        self.job_progress.hide()
        row.addWidget(self.job_progress)
        self.stop_button = button('Stop after current update', self.stop_job)
        self.stop_button.hide()
        row.addWidget(self.stop_button)
        self.log_toggle = QPushButton('Show job log')
        self.log_toggle.setCheckable(True)
        row.addWidget(self.log_toggle)
        layout.addLayout(row)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        self.log.setMaximumHeight(145)
        self.log.hide()
        self.log_toggle.toggled.connect(self.log.setVisible)
        layout.addWidget(self.log)
        parent.addWidget(frame)

    @property
    def busy(self):
        return self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning

    def selected_study(self):
        return self.study.currentData()

    def resolve(self, path):
        if not path:
            return None
        value = Path(path)
        return value if value.is_absolute() else self.root / value

    def select_page(self, index):
        self.pages.setCurrentIndex(index)
        self.nav_buttons[index].setChecked(True)
        self.title.setText(PAGES[index][0])
        self.subtitle.setText(PAGES[index][1])

    def refresh(self, *_):
        requested = self.selected_study()
        try:
            data = self.workspace_factory(self.root, study=requested).overview()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.error(str(exc))
            return
        self.snapshot = data
        state = data.get('study') or {}
        names = data.get('studies', [])
        selected = requested or state.get('name')
        self.study.blockSignals(True)
        self.study.clear()
        for name in names:
            self.study.addItem(str(name), str(name))
        if selected and self.study.findData(selected) < 0:
            self.study.addItem(selected, selected)
        if not self.study.count():
            self.study.addItem('No study yet', None)
        self.study.setCurrentIndex(max(0, self.study.findData(selected)))
        self.study.blockSignals(False)
        self.study.setEnabled(not self.busy)
        baseline = data.get('baseline')
        self.baseline_status.setText('Retained M0 and preliminary inputs are available. Create or choose a study to begin.'
                                     if baseline else 'Retained M0 package is unavailable. Open the operator guide and verify the repository installation.')
        slots = data.get('slots', [])
        reviewed = sum(s.get('status') == 'reviewed' for s in slots)
        imported = sum(s.get('status') != 'pending' for s in slots)
        self.overview_status.setText(f'{imported} / {len(slots)} file pairs imported · {reviewed} / {len(slots)} reviews saved'
                                     if slots else 'Create a study to reserve the complete flight order and fixed data roles.')
        counts = lambda stage: (f'{sum(s.get("status") == "reviewed" for s in slots if s["stage"] == stage)} / {sum(s["stage"] == stage for s in slots)} reviewed'
                                if slots else 'Create study')
        models = data.get('models', {})
        fill(self.sequence, [
            ['1', 'Fly five M0 takes; review and build M1', counts('M0')],
            ['2', 'Fly five M1 takes; review and build M2', counts('M1')],
            ['3', 'Fly five interleaved M0 / M2 pairs', counts('final')],
            ['4', 'Evaluate final distance and matched prediction error', 'Results available' if state.get('results') else 'Pending'],
        ])
        self.create_button.setEnabled(bool(baseline) and not self.busy)
        self.refresh_plan()
        self.refresh_slots()
        self.refresh_update()
        self.refresh_results()

    def generation_model(self):
        return self.snapshot.get('models', {}).get(self.plan_generation.currentText(), {})

    def refresh_plan(self, *_):
        generation = self.plan_generation.currentText()
        model = self.generation_model()
        path = self.resolve(model.get('model'))
        model_available = bool(path and path.is_file())
        plan = self.resolve(model.get('plan'))
        exported = self.resolve(model.get('export'))
        self.plan_status.setText(f'{generation} · '+(model.get('status', 'Frozen model available') if model_available else 'Model not available yet')+
                                (' · complete command exported' if exported and exported.exists() else ''))
        baseline = self.snapshot.get('baseline') or {}
        settings_file = plan / 'settings.json' if plan else self.resolve(baseline.get('mppi_settings'))
        setup = read_json(settings_file, {}) if settings_file else {}
        launch = setup.get('launch', {})
        fill(self.plan_details, [
            ['Model', model.get('model', 'Awaiting retained baseline / completed update')],
            ['Plan', model.get('plan', 'Not generated')],
            ['Command rate', '30 Hz · full sequence generated before flight'],
            ['Launch tracked origin [m]', str(launch.get('origin_m', 'See the retained setup'))],
            ['Target [m]', str(launch.get('target_m', 'See the retained setup'))],
            ['CSV', model.get('export', 'Not exported')],
        ])
        retained = generation == 'M0' and self.rehearsal_path() and not plan
        self.plan_button.setText('Retained M0 plan' if retained else '1  Generate MPPI plan')
        self.export_button.setText('Export retained M0 CSV' if retained else '2  Rehearse + export CSV')
        self.plan_button.setEnabled(model_available and not self.busy and not plan and not exported and not retained)
        self.export_button.setEnabled(bool(plan or self.rehearsal_path()) and not exported and not self.busy)
        self.preview_button.setEnabled(bool(self.rehearsal_path()))
        self.export_path.setText(str(exported) if exported else '')

    def refresh_slots(self, *_):
        selected = self.selected_slot()
        selected_name = selected.get('take') if selected else None
        stage = self.capture_stage.currentData()
        self.slot_rows = [s for s in self.snapshot.get('slots', []) if s['stage'] == stage]
        self.slots.blockSignals(True)
        fill(self.slots, [[s['take'], s['generation'], ROLE_LABELS.get(s['role'], s['role']),
                           'Excluded' if s.get('excluded') else s.get('status', 'pending').capitalize()]
                          for s in self.slot_rows])
        self.slots.blockSignals(False)
        self.capture_status.setText('Follow the listed interleaved order. All ten final takes remain excluded from fitting.'
                                    if stage == 'final' else 'Use the same frozen command for all five takes. Training: 001, 002, 004. Validation: 003, 005.')
        if self.slot_rows:
            index = next((i for i, s in enumerate(self.slot_rows) if s['take'] == selected_name), 0)
            self.slots.selectRow(index)
        else:
            self.selected_slot_changed()

    def selected_slot(self):
        index = self.slots.currentRow()
        return self.slot_rows[index] if 0 <= index < len(self.slot_rows) else None

    def selected_slot_changed(self):
        slot = self.selected_slot()
        available = slot is not None and bool(self.selected_study())
        state = self.snapshot.get('study') or {}
        saved_results = state.get('results') or {}
        frozen = bool(slot) and (any(u.get('source_stage') == slot['stage'] for u in state.get('updates', {}).values())
                                  or (isinstance(saved_results, dict) and saved_results.get('predictions_complete', False)))
        self.import_button.setEnabled(available and slot.get('status') == 'pending' and not self.busy if slot else False)
        self.review_button.setEnabled(available and slot.get('status') != 'pending' and not self.busy and not frozen if slot else False)
        self.flight_preview_button.setEnabled(available and slot.get('status') != 'pending' and not self.busy if slot else False)
        self.alignment_button.setEnabled(available and slot.get('status') != 'pending' and not self.busy and not frozen if slot else False)
        self.estimate_button.setEnabled(available and not self.busy and not frozen)
        if not slot:
            self.take_status.setText('Create or select a study to see its reserved flight slots.')
            return
        self.timing_note.hide()
        review = slot.get('review') or {}
        self.tracking_file.setText(slot.get('tracking', ''))
        self.controller_file.setText(slot.get('controller', ''))
        self.offset.setValue(slot.get('offset_s', slot.get('alignment', {}).get('offset_s', 0.)))
        self.clock_source.setText(slot.get('clock_source', slot.get('alignment', {}).get('source', '')))
        self.clock_verified.setChecked(slot.get('clock_verified', False))
        self.drone_label.setText(slot.get('drone_label', ''))
        self.reviewer.setText(review.get('reviewed_by', review.get('reviewer', '')))
        self.free_end.setValue(review.get('free_motion_end_s') or 1.5)
        self.contact.setCurrentIndex(max(0, self.contact.findData(review.get('physical_contact', 'none'))))
        self.review_clock.setChecked(review.get('clock_reviewed', False))
        self.review_hardware.setChecked(review.get('same_controller_and_hardware', False))
        self.review_intervention.setChecked(review.get('no_intervention', False))
        self.exclude.setChecked(slot.get('excluded', False))
        self.review_notes.setPlainText(review.get('notes', ''))
        self.take_status.setText(f'{slot["take"]} · {slot["generation"]} · {ROLE_LABELS.get(slot["role"], slot["role"])} · ' +
                                 ('Review frozen in fit/evaluation evidence. Original files remain available for inspection.' if frozen else
                                  'Original files are copied into this study and retained unchanged.') +
                                 ('\nComparison needs attention: ' + slot['import_error'] if slot.get('import_error') else ''))

    def update_job(self):
        state = self.snapshot.get('study') or {}
        generation = self.update_generation.currentText()
        entry = state.get('updates', {}).get(generation, {})
        if isinstance(entry, str):
            return self.resolve(entry)
        return self.resolve(entry.get('job') or entry.get('path'))

    def refresh_update(self, *_):
        generation = self.update_generation.currentText()
        source = 'M0' if generation == 'M1' else 'M1'
        slots = [s for s in self.snapshot.get('slots', []) if s['stage'] == source]
        count = sum(s.get('status') == 'reviewed' for s in slots)
        job = self.update_job()
        status = read_json(job / 'status.json', {}) if job else {}
        progress = read_json(job / 'progress.json', {}) if job else {}
        ready = bool(slots) and count == len(slots)
        self.update_status.setText(f'{source} → {generation} · {count} / {len(slots)} takes reviewed · ' +
                                   status.get('status', 'Inputs not prepared'))
        stages = [('drone_nominal', 'Loaded drone response'), ('drone_residual', 'Bounded drone correction'),
                  ('attitude_refinement', 'Attachment orientation'), ('cable_physics', 'Cable stiffness and damping'),
                  ('cable_residual', 'Bounded cable correction'), ('combined_validation', 'Frozen-model checks')]
        rows = []
        for name, purpose in stages:
            result = read_json(job / name / 'result.json', {}) if job else {}
            label = 'Completed' if status.get('status') == 'completed' else result.get('status', 'Pending')
            if status.get('stage') == name:
                label = status.get('status', 'Running')
            rows.append([name.replace('_', ' ').capitalize(), purpose, label])
        fill(self.fit_stages, rows)
        state = self.snapshot.get('study') or {}
        final_started = any(s.get('status') != 'pending' for s in self.snapshot.get('slots', []) if s['stage'] == 'final')
        self.retry_preparation = bool(job) and status.get('status') in ('failed', 'stopped') and generation not in self.snapshot.get('models', {}) and not final_started
        self.prepare_button.setText('Prepare a new attempt' if self.retry_preparation else '1  Freeze reviewed fit inputs')
        self.prepare_button.setToolTip('Create a new job from the same frozen inputs. Retain the previous failed/stopped attempt; fitting starts only with the next button.' if self.retry_preparation else '')
        self.prepare_button.setEnabled(ready and (not job or self.retry_preparation) and not final_started and not self.busy)
        self.fit_button.setEnabled(bool(job) and status.get('status') == 'prepared' and not self.busy and self.device.currentText() == 'cuda')
        self.fit_button.setToolTip('The full model update requires CUDA. Select cuda on Plan & export.')
        self.update_details.setText(((str(job) + '\n') if job else '') +
                                   ('Progress: '+json.dumps(progress, ensure_ascii=False) if progress else
                                    'Complete the source batch reviews before freezing inputs.') +
                                   (f'\n{len(state["update_history"][generation])} previous attempts retained.'
                                    if state.get('update_history', {}).get(generation) else ''))

    def refresh_results(self, *_):
        state = self.snapshot.get('study') or {}
        value = state.get('results')
        path = self.resolve(value.get('report') if isinstance(value, dict) else value)
        if path and path.is_dir():
            path = path / 'report.json'
        report = read_json(path, {}) if path else {}
        final = [s for s in self.snapshot.get('slots', []) if s['stage'] == 'final']
        ready = bool(final) and all(s.get('status') == 'reviewed' for s in final)
        self.evaluate_button.setEnabled(ready and not self.busy and self.device.currentText() == 'cuda')
        self.evaluate_button.setToolTip('The matched model predictions require CUDA. Select cuda on Plan & export.')
        self.result_status.setText('Saved final comparison · continuous error metrics · original recordings retained'
                                   if report else f'{sum(s.get("status") == "reviewed" for s in final)} / {len(final)} final flight reviews saved. '
                                   'Evaluate after completing the final comparison.')
        physical = report.get('physical', report.get('flights', []))
        if isinstance(physical, dict):
            physical = [dict(value, take=key) for key, value in physical.items()]
        rows = []
        for item in physical:
            if item.get('stage') not in (None, 'final'):
                continue
            distance = item.get('minimum_tip_target_m', item.get('minimum_tip_target_distance_m', item.get('distance_m')))
            coverage = item.get('sample_coverage', item.get('coverage'))
            rows.append([item.get('take', ''), item.get('generation', item.get('model', '')),
                         f'{distance * 100:.2f}' if distance is not None else 'Unavailable',
                         ((f'{coverage * 100:.1f}%' if coverage is not None else 'See report') +
                          (' · partial window' if item.get('window_complete') is False else
                           ' · gaps' if item.get('fully_observed') is False else '')),
                         'Excluded' if item.get('excluded') else 'Not reviewed' if item.get('reviewed') is False else item.get('status', 'Reviewed')])
        fill(self.physical_results, rows)
        summary = report.get('summary', {})
        self.result_summary.setText(('Mean observed closest distance: M0 ' + self.cm(summary.get('M0_mean_m')) +
                                     ' cm · M2 ' + self.cm(summary.get('M2_mean_m')) +
                                     ' cm. Paired improvement (M0 − M2): ' + self.cm(summary.get('paired_mean_improvement_m')) + ' cm.' +
                                     (f' {summary["fully_observed_pair_count"]} / {summary.get("paired_count", 5)} pairs fully observed.'
                                      if 'fully_observed_pair_count' in summary else ''))
                                    if summary else '')
        self.result_note.setText('Physical error compares the executed M0 and M2 commands. Prediction error compares '
                                'frozen M0, M1 and M2 on the same recorded commands. Lower error is better; '
                                'there is no 5 cm success threshold. Missing observations may hide a closer encounter. ' +
                                report.get('qualification', ''))
        prediction = report.get('prediction_summary', {})
        if isinstance(prediction, list):
            prediction = {item['model']: item for item in prediction}
        fill(self.prediction_results, [[generation,
            self.cm(item.get('tip_rmse_m')), self.cm(item.get('drone_rmse_m')), item.get('takes', '')]
            for generation, item in prediction.items()])

    @staticmethod
    def cm(value):
        return 'Unavailable' if value is None else f'{100 * value:.2f}'

    def error(self, text):
        self.job_status.setText(text)
        self.job_status.setStyleSheet('color: #a23b32;')

    def create_study(self):
        name = self.study_name.text().strip()
        if not name:
            self.error('Enter a name for the new study.')
            return
        self.run_action('create', study=name)

    def start_plan(self):
        self.run_action('plan', ['--generation', self.plan_generation.currentText(), '--device', self.device.currentText()])

    def export_plan(self):
        self.run_action('export', ['--generation', self.plan_generation.currentText(), '--device', self.device.currentText()])

    def import_pair(self):
        slot = self.selected_slot()
        if not slot:
            return
        if not self.tracking_file.text().strip() or not self.controller_file.text().strip() or not self.clock_source.text().strip():
            self.error('Choose both CSV files and describe the timing evidence before importing.')
            return
        args = ['--stage', slot['stage'], '--take', slot['take'], '--tracking', self.tracking_file.text().strip(),
                '--controller', self.controller_file.text().strip(), '--offset', str(self.offset.value()),
                '--clock-source', self.clock_source.text().strip()]
        if self.drone_label.text().strip():
            args += ['--drone-label', self.drone_label.text().strip()]
        if self.clock_verified.isChecked():
            args += ['--clock-verified']
        self.run_action('import', args)

    def save_review(self):
        slot = self.selected_slot()
        if not slot:
            return
        if not self.reviewer.text().strip():
            self.error('Enter the reviewer name before saving.')
            return
        args = ['--stage', slot['stage'], '--take', slot['take'], '--reviewer', self.reviewer.text().strip(),
                '--free-motion-end', str(self.free_end.value()), '--contact', self.contact.currentData(),
                '--notes', self.review_notes.toPlainText().strip()]
        for widget, flag in ((self.review_clock, '--clock-reviewed'), (self.review_hardware, '--same-hardware'),
                             (self.review_intervention, '--no-intervention'), (self.exclude, '--exclude')):
            if widget.isChecked():
                args.append(flag)
        self.run_action('review', args)

    def save_alignment(self):
        slot = self.selected_slot()
        if not slot:
            return
        if not self.clock_source.text().strip():
            self.error('Describe the timing evidence before saving a corrected offset.')
            return
        args = ['--stage', slot['stage'], '--take', slot['take'], '--offset', str(self.offset.value()),
                '--clock-source', self.clock_source.text().strip()]
        if self.clock_verified.isChecked():
            args += ['--clock-verified']
        self.run_action('align', args)

    def estimate_timing(self):
        tracking = self.tracking_file.text().strip()
        controller = self.controller_file.text().strip()
        if not tracking or not controller:
            self.error('Choose the tracking CSV and its matching controller log before estimating timing.')
            return
        self.timing_context = dict(tracking=tracking, controller=controller, drone=self.drone_label.text().strip(),
                                   offset=self.offset.value(), source=self.clock_source.text(),
                                   verified=self.clock_verified.isChecked())
        args = ['--tracking', tracking, '--controller', controller]
        if self.timing_context['drone']:
            args += ['--drone-label', self.timing_context['drone']]
        self.run_action('estimate-timing', args)

    def restore_timing_fields(self):
        context = getattr(self, 'timing_context', None)
        if not context:
            return
        self.tracking_file.setText(context['tracking'])
        self.controller_file.setText(context['controller'])
        self.drone_label.setText(context['drone'])
        self.offset.setValue(context['offset'])
        self.clock_source.setText(context['source'])
        self.clock_verified.setChecked(context['verified'])

    def apply_timing_estimate(self, result):
        offset = result.get('offset_s')
        if not isinstance(offset, (int, float)) or not math.isfinite(offset):
            raise ValueError('The timing estimator did not return a finite offset.')
        error = result.get('rmse_m')
        spread = result.get('chunk_spread_s')
        details = []
        if isinstance(error, (int, float)) and math.isfinite(error):
            details.append(f'position RMS {100 * error:.2f} cm')
        if isinstance(spread, (int, float)) and math.isfinite(spread):
            details.append(f'chunk offset spread {1000 * spread:.1f} ms')
        description = result.get('method', 'Timing estimated from common recorded motion')
        self.offset.setValue(offset)
        self.clock_source.setText(description + ('; ' + '; '.join(details) if details else ''))
        self.clock_verified.setChecked(False)
        self.review_clock.setChecked(False)
        self.timing_note.setText(f'Estimated offset {offset:+.6f} s · ' + ' · '.join(details) + '\n' +
                                 result.get('limitation', 'Estimated from recorded motion; this does not establish synchronized clocks.') +
                                 '\nReview this candidate before importing or saving the alignment. No recordings or reviews were changed.')
        self.timing_note.show()
        self.job_status.setStyleSheet('color: #14695d;')
        self.job_status.setText('Timing estimate is ready for review. No recording was imported or review approved.')

    def prepare_update(self):
        args = ['--generation', self.update_generation.currentText()]
        if self.retry_preparation:
            args += ['--retry']
        self.run_action('prepare', args)

    def fit_update(self):
        self.run_action('fit', ['--generation', self.update_generation.currentText(), '--device', self.device.currentText()])

    def evaluate(self):
        self.run_action('evaluate', ['--device', self.device.currentText()])

    def run_action(self, action, arguments=None, *, study=None):
        """Pass arguments as a list to the active interpreter; no shell interpolation."""
        if self.busy:
            self.error('A job is already running. Follow its progress below.')
            return
        study = study or self.selected_study()
        if not study:
            self.error('Create or select a study first.')
            return
        cli = self.root / 'tools/lab.py'
        if not cli.is_file():
            self.error('The lab command-line entry point is missing. Check the repository installation.')
            return
        log_dir = self.root / 'runs/lab_ui_logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / (datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '-' + action + '.log')
        self.log.clear()
        self.process = QProcess(self)
        self.process_action = action
        self.process.setWorkingDirectory(str(self.root))
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert('PYTHONUNBUFFERED', '1')
        self.process.setProcessEnvironment(environment)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.finished.connect(self.job_finished)
        self.process.errorOccurred.connect(self.process_error)
        self.job_status.setStyleSheet('')
        self.job_status.setText(action.capitalize() + ' · running. See the job log for details.')
        self.job_progress.setRange(0, 0)
        self.job_progress.show()
        self.stop_button.setVisible(action in ('plan', 'fit'))
        self.stop_button.setEnabled(True)
        self.process.start(sys.executable, ['-u', str(cli), '--study', study, action] + list(arguments or []))
        self.pending_study = study
        self.refresh()
        if action == 'estimate-timing':
            self.restore_timing_fields()

    def stop_job(self):
        if not self.busy or self.process_action not in ('plan', 'fit'):
            return
        self.stop_process = QProcess(self)
        self.stop_process.setWorkingDirectory(str(self.root))
        self.stop_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.stop_process.finished.connect(self.stop_finished)
        self.stop_process.start(sys.executable, ['-u', str(self.root / 'tools/lab.py'),
                                               '--study', self.selected_study(), 'stop'])
        self.stop_button.setEnabled(False)
        self.job_status.setText('Requesting a stop at the next supported update boundary…')

    def stop_finished(self, exit_code, exit_status):
        output = bytes(self.stop_process.readAllStandardOutput()).decode('utf-8', errors='replace')
        self.log.appendPlainText(output)
        if exit_code:
            self.error('The stop request failed. The job is still running; see the log.')
            self.stop_button.setEnabled(True)
        else:
            self.job_status.setText('Stop requested. Waiting for the current update to finish and save its state.')

    def read_output(self):
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode('utf-8', errors='replace')
        if not text:
            return
        cursor = self.log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()
        if self.log_path:
            with self.log_path.open('a', encoding='utf-8') as stream:
                stream.write(text)

    def process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.error('Could not start the current Python interpreter: ' + self.process.errorString())
            self.job_progress.hide()
            self.stop_button.hide()
            self.refresh()

    def job_finished(self, exit_code, exit_status):
        self.read_output()
        action = self.process_action
        self.job_progress.setRange(0, 1)
        self.job_progress.setValue(1 if exit_code == 0 else 0)
        self.job_progress.hide()
        self.stop_button.hide()
        if exit_code == 0:
            self.job_status.setStyleSheet('color: #14695d;')
            self.job_status.setText(action.capitalize() + ' completed. Saved files remain in this repository.')
            if action == 'create':
                self.study.addItem(self.pending_study, self.pending_study)
                self.study.setCurrentIndex(self.study.findData(self.pending_study))
                self.study_name.clear()
        else:
            self.error(action.capitalize() + ' failed. Open the job log below; existing evidence is retained.')
            self.log_toggle.setChecked(True)
        self.refresh()
        if action == 'estimate-timing':
            self.restore_timing_fields()
            if exit_code == 0:
                try:
                    self.apply_timing_estimate(json.loads(self.log.toPlainText()))
                except (ValueError, KeyError, TypeError) as exc:
                    self.error('Could not read the timing estimate: ' + str(exc))
                    self.log_toggle.setChecked(True)

    def poll(self):
        if self.busy:
            # Refresh immutable/status metadata without resetting an operator's unsaved review fields.
            try:
                self.snapshot = self.workspace_factory(self.root, study=self.selected_study()).overview()
                self.refresh_plan()
                self.refresh_update()
            except (OSError, ValueError, KeyError):
                pass

    def rehearsal_path(self):
        model = self.generation_model()
        for key in ('rehearsal', 'export'):
            path = self.resolve(model.get(key))
            if path:
                path = path.parent if path.is_file() else path
                if (path / 'rehearsal.json').is_file():
                    return path
                for candidate in (path / 'rehearsal', path / 'forecast'):
                    if (candidate / 'rehearsal.json').is_file():
                        return candidate
        return None

    def preview_plan(self):
        path = self.rehearsal_path()
        if not path:
            self.error('Complete the rehearsal/export first to preview the frozen motion.')
            return
        from simulator.gui.rehearsal_workspace import RehearsalWorkspace
        dialog = InspectionDialog(self)
        dialog.setWindowTitle('Saved command and predicted recovery')
        dialog.resize(1260, 850)
        layout = QVBoxLayout(dialog)
        viewer = RehearsalWorkspace(self.root, inspection_only=True)
        dialog.inspector = viewer
        # The lab export action is the only export route; this dialog only inspects.
        viewer.save.hide()
        viewer.package.hide()
        viewer.open.hide()
        try:
            viewer.load_result(path)
        except (OSError, ValueError, KeyError) as exc:
            self.error('Saved motion could not be opened: ' + str(exc))
            viewer.shutdown()
            dialog.deleteLater()
            return
        layout.addWidget(viewer)
        viewer.set_page_active(True)
        self.replay_dialog = dialog
        dialog.show()

    def preview_flight(self):
        slot = self.selected_slot()
        if not slot:
            return
        batch = self.resolve(slot.get('batch'))
        if not batch:
            self.error('The imported recording has no batch location. Refresh after the import completes.')
            return
        from simulator.gui.adaptation_check_page import AdaptationCheckPage
        dialog = InspectionDialog(self)
        dialog.setWindowTitle('Recorded flight and original forecast · ' + slot['take'])
        dialog.resize(1260, 850)
        layout = QVBoxLayout(dialog)
        page = AdaptationCheckPage(self.root)
        dialog.inspector = page
        if QApplication.platformName() == 'offscreen':
            # Native VTK needs an interactive OpenGL surface; error plots remain
            # useful for read-only verification in headless test environments.
            page.views.setCurrentIndex(1)
            page.views.setTabVisible(0, False)
        if not page.select_flight(str(batch), slot['take']):
            self.error('This recording is not available in the flight index yet.')
            page.shutdown()
            dialog.deleteLater()
            return
        layout.addWidget(page)
        page.set_page_active(True)
        page.load_flight()
        self.replay_dialog = dialog
        dialog.show()

    def open_path(self, path):
        path = self.resolve(path)
        if path and path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        else:
            self.error('This folder is not available yet. Complete its workflow step first.')

    def open_guide(self):
        for name in ('docs/lab/LAB_RUNBOOK.md', 'docs/LAB_RUNBOOK.md', 'README.md'):
            if (self.root / name).is_file():
                self.open_path(name)
                return

    def open_study(self):
        name = self.selected_study()
        self.open_path(self.root / 'experiments' / name if name else None)

    def open_exports(self):
        name = self.selected_study()
        self.open_path(self.root / 'exports' / name / self.plan_generation.currentText() if name else None)

    def open_results(self):
        name = self.selected_study()
        self.open_path(self.root / 'experiments' / name / 'results' if name else None)

    def closeEvent(self, event):
        if self.busy:
            self.error('A job is still running. Keep this window open until its current operation completes.')
            event.ignore()
            return
        self.timer.stop()
        if self.replay_dialog:
            if self.replay_dialog.inspector is not None and not self.replay_dialog.inspector.shutdown():
                event.ignore()
                QTimer.singleShot(150, self.close)
                return
            self.replay_dialog.close()
        super().closeEvent(event)


def main(root=None):
    root = Path(root or Path(__file__).resolve().parents[1])
    application = QApplication.instance()
    owns_application = application is None
    if application is None:
        application = QApplication([sys.argv[0]])
    application.setApplicationName('AeroWhip')
    application.setOrganizationName('AeroWhip')
    application.setStyle('Fusion')
    window = LabWindow(root)
    window.show()
    return application.exec() if owns_application else 0
