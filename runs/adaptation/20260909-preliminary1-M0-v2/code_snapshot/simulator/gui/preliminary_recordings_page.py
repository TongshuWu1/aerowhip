"""Collection inbox for a fresh system, without historical fitting shortcuts."""
from pathlib import Path
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QPushButton,QTableWidget,QTableWidgetItem,QHeaderView,QFileDialog
from simulator.workflow import read_json,import_take
from .research_widgets import note


class PreliminaryRecordingsPage(QWidget):
    def __init__(self,root):
        super().__init__();self.root=Path(root);layout=QVBoxLayout(self)
        h=read_json(self.root/'config/current_vehicle.json',{})
        layout.addWidget(note(f'New system · drone {1000*h.get("drone_mass_kg",0):g} g · cable {1000*h.get("cable_assembly_mass_kg",0):g} g · total {1000*h.get("total_mass_kg",0):g} g'))
        layout.addWidget(note('Import one folder per preliminary take, containing the original OptiTrack CSV and controller logger CSV. Keep the native recording, actual P/V/A commands, timing evidence and intervention notes. Geometry is unchanged.'))
        row=QHBoxLayout()
        for label,callback in [('Import preliminary take',self.import_recording),('Refresh',self.refresh),('Open recordings folder',self.open_folder)]:
            button=QPushButton(label);button.clicked.connect(callback);row.addWidget(button)
        row.addStretch();layout.addLayout(row)
        self.table=QTableWidget(0,3);self.table.setHorizontalHeaderLabels(['Take','CSV files','Review state'])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers);self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch);layout.addWidget(self.table,1)
        self.status=note('');layout.addWidget(self.status)
        layout.addWidget(note('Next: review preliminary data and fit fresh M0 → MPPI rehearsal → new flight against its saved forecast → fit M1 → new MPPI and prospective check. Importing does not start fitting or flight.'))
        self.refresh()

    def refresh(self):
        folders=sorted(p for p in (self.root/'data/raw_takes').glob('*') if p.is_dir())
        manifest=read_json(self.root/'data/dataset_manifest.json',{}).get('takes',{})
        self.table.setRowCount(len(folders))
        for i,p in enumerate(folders):
            for j,value in enumerate([p.name,str(len(list(p.glob('*.csv')))),'Imported · review pending' if p.name in manifest else 'Files present · import / review pending']):
                self.table.setItem(i,j,QTableWidgetItem(value))
        self.status.setText(f'{len(folders)} preliminary takes. No fitted model yet.' if not folders else f'{len(folders)} preliminary takes. Review data roles before fitting.')

    def import_recording(self):
        source=QFileDialog.getExistingDirectory(self,'Import a preliminary take')
        if source:
            try:
                name=import_take(self.root,source);self.refresh();self.status.setText('Imported '+name+' with the current hardware identity. Original source files preserved.')
            except (OSError,ValueError,KeyError) as error:self.status.setText(str(error))

    def open_folder(self):
        folder=self.root/'data/raw_takes';folder.mkdir(parents=True,exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
