"""Small batch flight-export dialog for saved rehearsals."""
from datetime import datetime
from pathlib import Path
from PySide6.QtCore import Qt,QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,
    QLineEdit,QFileDialog,QTableWidget,QTableWidgetItem,QHeaderView,QAbstractItemView)
from simulator.workflow import read_json
from deployment.flight_export import export_flights


class FlightExportDialog(QDialog):
    def __init__(self, root, selected=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Export flight commands'); self.resize(940,520)
        self.root = Path(root); self.exported = None
        layout = QVBoxLayout(self)
        text = QLabel('Select one or more rehearsals. Each gets its CSV and recording folder; one ZIP contains all selected exports.')
        text.setWordWrap(True); layout.addWidget(text)
        self.table = QTableWidget(0,3)
        self.table.setHorizontalHeaderLabels(['Rehearsal','Start height (m)','Duration (s)'])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.Stretch)
        for column in (1,2): self.table.horizontalHeader().setSectionResizeMode(column,QHeaderView.ResizeMode.ResizeToContents)
        for path in sorted((self.root/'runs/rehearsals_pva').glob('*/rehearsal.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True):
            if (path.parent/'ARCHIVED').exists(): continue
            m = read_json(path,{})
            if m.get('schema') != 'pva_fullstate_30hz_v1' or not m.get('recovery_prediction_complete') or m.get('preview_only'): continue
            row = self.table.rowCount(); self.table.insertRow(row)
            item = QTableWidgetItem(path.parent.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setData(Qt.ItemDataRole.UserRole,str(path.parent.resolve()))
            item.setCheckState(Qt.CheckState.Checked if selected and path.parent.resolve()==Path(selected).resolve() else Qt.CheckState.Unchecked)
            self.table.setItem(row,0,item)
            self.table.setItem(row,1,QTableWidgetItem(f'{m["initial_tracking_origin_m"][2]:.3f}'))
            self.table.setItem(row,2,QTableWidgetItem(f'{m["total_duration_s"]:.2f}'))
        layout.addWidget(self.table)
        destination = QHBoxLayout(); destination.addWidget(QLabel('New export folder'))
        self.destination = QLineEdit(str(self.root/'exports'/('Flight_exports_'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))))
        destination.addWidget(self.destination,1)
        browse = QPushButton('Browse…'); browse.clicked.connect(self.browse); destination.addWidget(browse); layout.addLayout(destination)
        self.status = QLabel(''); self.status.setWordWrap(True); layout.addWidget(self.status)
        buttons = QHBoxLayout(); buttons.addStretch()
        self.open_button = QPushButton('Open export folder'); self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self.open_folder); buttons.addWidget(self.open_button)
        self.export_button = QPushButton('Export selected'); self.export_button.clicked.connect(self.export_selected); buttons.addWidget(self.export_button)
        close = QPushButton('Close'); close.clicked.connect(self.accept); buttons.addWidget(close); layout.addLayout(buttons)
        self.table.itemChanged.connect(self.update_selection); self.update_selection()

    def selected_paths(self):
        return [Path(self.table.item(row,0).data(Qt.ItemDataRole.UserRole)) for row in range(self.table.rowCount())
                if self.table.item(row,0).checkState()==Qt.CheckState.Checked]

    def update_selection(self):
        count = len(self.selected_paths()); self.export_button.setEnabled(count>0)
        self.export_button.setText(f'Export {count} selected' if count else 'Select a rehearsal')

    def browse(self):
        parent = QFileDialog.getExistingDirectory(self,'Choose export location',str(self.root/'exports'))
        if parent: self.destination.setText(str(Path(parent)/Path(self.destination.text()).name))

    def export_selected(self):
        if not self.destination.text().strip():
            self.status.setText('Choose a new export folder.'); return
        try:
            self.exported = export_flights(self.selected_paths(),self.destination.text().strip())
        except (OSError,ValueError,KeyError,TypeError) as exc:
            self.status.setText('Export failed: '+str(exc)); return
        self.status.setText(f'Exported {len(self.selected_paths())} rehearsals to {self.exported}\nCSV files, recording folders and flight_commands.zip are ready.')
        self.open_button.setEnabled(True)

    def open_folder(self):
        if self.exported: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.exported)))
