"""PPO checkpoint selection; all training remains an explicit user action."""
from pathlib import Path
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QCheckBox, QLabel, QHeaderView)
from simulator.policy_library import list_policies, set_deleted
from simulator.workflow import read_json
from .process_status import process_is_running


class PolicyLibraryPage(QWidget):
    use_requested = Signal(str)
    continue_requested = Signal(str)
    changed = Signal()

    def __init__(self, root):
        super().__init__()
        self.root = Path(root)
        self.protected_paths = lambda: []
        self.rows = []
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(['Run name', 'Checkpoint', 'Model / rate', 'Validation hit', 'Run status', 'Library'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3,4,5):
            self.table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self.update_actions)
        layout.addWidget(self.table)
        controls = QHBoxLayout()
        self.use = QPushButton('Use in rehearsal')
        self.resume = QPushButton('Continue training…')
        self.delete = QPushButton('Delete from library')
        self.restore = QPushButton('Restore')
        self.show_deleted = QCheckBox('Show deleted')
        refresh = QPushButton('Refresh')
        for widget in (self.use, self.resume, self.delete, self.restore, self.show_deleted, refresh):
            controls.addWidget(widget)
        layout.addLayout(controls)
        self.note = QLabel('Delete removes a checkpoint from the library. Saved files remain available for restoration.')
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.use.clicked.connect(lambda: self.request(self.use_requested,allow_busy=True))
        self.resume.clicked.connect(lambda: self.request(self.continue_requested))
        self.delete.clicked.connect(lambda: self.remove(True))
        self.restore.clicked.connect(lambda: self.remove(False))
        self.show_deleted.toggled.connect(self.refresh)
        refresh.clicked.connect(self.refresh)
        self.refresh()

    def selected(self):
        index = self.table.currentRow()
        return self.rows[index] if 0 <= index < len(self.rows) else None

    def refresh(self):
        selected = self.selected()
        path = selected['path'] if selected else None
        rows = list_policies(self.root, self.show_deleted.isChecked())
        if rows != self.rows:
            self.table.blockSignals(True)
            self.rows = rows
            self.table.setRowCount(len(rows))
            for index, row in enumerate(rows):
                run=Path(row['run']);model=read_json(run/'model.json',{})
                native=model.get('fullstate_execution',{}).get('schema')=='tracked_pose_execution_v1'
                validation=read_json(run/('best_validation.json' if row['checkpoint']=='best_validation.pt' else 'validation_latest.json'),{})
                hit=(f'{100*validation["success_rate"]:.1f}%' if 'success_rate' in validation and row['checkpoint'] in ('latest.pt','best_validation.pt') else '—')
                for column, text in enumerate((row['name'], row['checkpoint'], 'M0 · 30 Hz · both NNs' if native else 'Legacy · saved rate',hit,row['status'],
                                               'Deleted' if row['deleted'] else 'Available')):
                    item = QTableWidgetItem(text)
                    item.setToolTip(row['path'])
                    self.table.setItem(index, column, item)
            target = next((i for i, row in enumerate(rows) if row['path'] == path), 0)
            if rows:
                self.table.selectRow(target)
            self.table.resizeColumnsToContents()
            self.table.blockSignals(False)
        self.update_actions()

    def is_busy(self, row):
        status = read_json(Path(row['run'])/'status.json', {})
        return (status.get('status') in ('STARTING', 'RUNNING', 'STOPPING')
                and process_is_running(int(status.get('pid', 0))))

    def update_actions(self):
        row = self.selected()
        available = bool(row and not row['deleted'])
        self.use.setEnabled(available)
        self.resume.setEnabled(available and not self.is_busy(row))
        protected = {str(Path(p).resolve()) for p in self.protected_paths() if p}
        self.delete.setEnabled(available and not self.is_busy(row) and row['path'] not in protected)
        self.restore.setEnabled(bool(row and row['deleted']))

    def request(self, signal,allow_busy=False):
        row = self.selected()
        if row and not row['deleted'] and (allow_busy or not self.is_busy(row)):
            signal.emit(row['path'])

    def remove(self, deleted):
        row = self.selected()
        if not row:
            return
        try:
            protected = {str(Path(p).resolve()) for p in self.protected_paths() if p}
            if deleted and (self.is_busy(row) or row['path'] in protected):
                raise ValueError('This checkpoint is selected for use or belongs to an active training run.')
            set_deleted(self.root, row['path'], deleted)
            self.refresh()
            self.changed.emit()
        except (OSError, ValueError) as error:
            self.note.setText(str(error))
