"""Train and review the cable-only residual without changing the active baseline."""
from pathlib import Path
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView)
from simulator.workflow import read_json
from experimental_data.cable_residual_fit import prepare_residual_job, apply_candidate
from .research_widgets import BackgroundJob, note


class CableResidualPage(QWidget):
    baseline_applied = Signal()

    def __init__(self, root):
        super().__init__()
        self.root = Path(root)
        layout = QVBoxLayout(self)
        layout.addWidget(note('Cable residual · preliminary recordings · measured attachment motion. '
            'Learn cable effects with an NN only. Masses, geometry, EI and internal damping stay fixed.'))
        row = QHBoxLayout()
        self.updates = QSpinBox()
        self.updates.setRange(1, 10000)
        self.updates.setValue(read_json(self.root/'config/cable_residual.json', {'updates':24})['updates'])
        row.addWidget(QLabel('Optimizer updates'))
        row.addWidget(self.updates)
        self.train = QPushButton('Train cable residual')
        self.train.setObjectName('primaryButton')
        self.train.clicked.connect(self.start)
        row.addWidget(self.train)
        self.stop = QPushButton('Stop training')
        self.stop.setEnabled(False)
        self.stop.clicked.connect(self.request_stop)
        row.addWidget(self.stop)
        row.addStretch()
        layout.addLayout(row)
        layout.addWidget(note('No separate drag coefficient or calibrated-drag initialization. The NN starts with zero correction. '
            'A small bounded NN adds cable-node acceleration, with zero direct drone correction. '
            'Selection uses training takes; separate preliminary validation takes check 2 s and 5 s predictions.'))
        row = QHBoxLayout()
        row.addWidget(QLabel('Candidate run'))
        self.runs = QComboBox()
        self.runs.currentIndexChanged.connect(self.select)
        row.addWidget(self.runs, 1)
        refresh = QPushButton('Refresh')
        refresh.clicked.connect(self.refresh)
        row.addWidget(refresh)
        layout.addLayout(row)
        self.summary = note('No cable residual trained yet.')
        layout.addWidget(self.summary)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(['Horizon / split','Physical marker (cm)','Residual marker (cm)',
                                             'Physical tip (cm)','Residual tip (cm)'])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)
        self.apply = QPushButton('Apply validated cable residual')
        self.apply.clicked.connect(self.apply_selected)
        layout.addWidget(self.apply)
        layout.addWidget(note('Applying creates a new baseline for future training. Existing PPO runs retain their saved model. '
            'Preliminary validation is not evidence of improved real whipping.'))
        self.job = BackgroundJob(root)
        self.job.finished.connect(self.finished)
        layout.addWidget(self.job)
        self.refresh()

    def start(self):
        if self.job.running:
            return
        try:
            directory, command = prepare_residual_job(self.root, updates=self.updates.value())
            self.job.start(directory, command)
            self.train.setEnabled(False)
            self.apply.setEnabled(False)
            self.stop.setEnabled(True)
            self.refresh()
        except (OSError, ValueError) as error:
            self.job.status.setText(str(error))

    def request_stop(self):
        if self.job.running:
            (self.job.directory/'STOP_REQUESTED').touch()
            self.stop.setEnabled(False)
            self.job.status.setText('Stopping at the next training boundary; active calibration stays unchanged.')

    def finished(self, code):
        self.train.setEnabled(True)
        self.stop.setEnabled(False)
        self.refresh()
        index = self.runs.findData(str(self.job.directory))
        if index >= 0:
            self.runs.setCurrentIndex(index)

    def refresh(self):
        selected = self.runs.currentData()
        self.runs.blockSignals(True)
        self.runs.clear()
        for path in sorted((self.root/'data/cable_residual_runs').glob('*/settings.json'), reverse=True):
            self.runs.addItem(path.parent.name, str(path.parent))
        index = self.runs.findData(selected)
        if index >= 0:
            self.runs.setCurrentIndex(index)
        self.runs.blockSignals(False)
        self.select()

    def select(self):
        path = self.runs.currentData()
        self.table.setRowCount(0)
        self.apply.setEnabled(False)
        if not path:
            self.summary.setText('No cable residual trained yet.')
            return
        directory = Path(path)
        review = read_json(directory/'review.json', {})
        status = read_json(directory/'status.json', {})
        if not review:
            self.summary.setText(status.get('status', 'Prepared')+': '+status.get('error',
                read_json(directory/'progress.json', {}).get('label', 'Waiting for training')))
            return
        damping = (f'learned drag {review["initial_drag_s_inv"]:.6g} → {review["fitted_drag_s_inv"]:.6g} s⁻¹'
                   if review.get('learned_drag') else 'historical fit: external drag held fixed')
        if review.get('drag_mode') == 'nn_only':
            damping = 'NN only · zero initial correction · no separate drag coefficient'
        self.summary.setText(('Passed preliminary validation' if review['accepted'] else 'Not accepted — keep the physical baseline')+
            f' · selected update {review["selected_update"]} · {damping}')
        self.apply.setEnabled(review['accepted'] and not self.job.running)
        rows = []
        for horizon, pair in read_json(directory/'evaluation.json').items():
            for role in ('training', 'validation'):
                p, n = pair['physics'][role], pair['residual'][role]
                rows.append([f'{horizon} s / {role}',
                    *[f'{100*value:.3f}' for value in (p['equal_take_marker_rmse_m'], n['equal_take_marker_rmse_m'],
                                                      p['equal_take_tip_rmse_m'], n['equal_take_tip_rmse_m'])]])
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                self.table.setItem(i, j, QTableWidgetItem(value))

    def apply_selected(self):
        if self.job.running or not self.runs.currentData():
            return
        try:
            version = apply_candidate(self.root, self.runs.currentData())
            self.job.status.setText(f'Applied cable residual baseline {version}; existing PPO runs unchanged.')
            self.apply.setEnabled(False)
            self.baseline_applied.emit()
        except (OSError, ValueError) as error:
            self.job.status.setText(str(error))
