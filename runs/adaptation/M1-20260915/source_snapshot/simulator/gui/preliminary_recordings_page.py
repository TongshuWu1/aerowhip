"""Collection inbox for a fresh system, without historical fitting shortcuts."""
from pathlib import Path
from PySide6.QtCore import QUrl,Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QPushButton,QTableWidget,QTableWidgetItem,QHeaderView,QFileDialog
from simulator.workflow import read_json,import_take
from .research_widgets import note


class PreliminaryRecordingsPage(QWidget):
    replay_requested=Signal(str)
    def __init__(self,root):
        super().__init__();self.root=Path(root);layout=QVBoxLayout(self)
        h=read_json(self.root/'config/current_vehicle.json',{})
        layout.addWidget(note(f'New system · drone {1000*h.get("drone_mass_kg",0):g} g · cable {1000*h.get("cable_assembly_mass_kg",0):g} g · total {1000*h.get("total_mass_kg",0):g} g'))
        layout.addWidget(note('Import one folder per preliminary take, containing the original OptiTrack CSV and controller logger CSV. Keep the native recording, actual P/V/A commands, timing evidence and intervention notes. Geometry is unchanged.'))
        row=QHBoxLayout()
        for label,callback in [('Import preliminary take',self.import_recording),('Refresh',self.refresh),('Open recordings folder',self.open_folder)]:
            button=QPushButton(label);button.clicked.connect(callback);row.addWidget(button)
        row.addStretch();layout.addLayout(row)
        self.replay=QPushButton('Replay selected take');self.replay.setObjectName('primaryButton');self.replay.clicked.connect(self.replay_selected);row.insertWidget(0,self.replay)
        self.table=QTableWidget(0,3);self.table.setHorizontalHeaderLabels(['Take','CSV files','Review state'])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers);self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows);self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.doubleClicked.connect(self.replay_selected)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch);layout.addWidget(self.table,1)
        self.status=note('');layout.addWidget(self.status)
        layout.addWidget(note('Next: review preliminary data and fit fresh M0 → MPPI rehearsal → new flight against its saved forecast → fit M1 → new MPPI and prospective check. Importing does not start fitting or flight.'))
        self.refresh()

    def refresh(self):
        folders=sorted(p for p in (self.root/'data/raw_takes').glob('*') if p.is_dir())
        manifest=read_json(self.root/'data/dataset_manifest.json',{}).get('takes',{})
        self.table.setRowCount(len(folders))
        for i,p in enumerate(folders):
            meta=read_json(p/'experiment.json',{});state=('Reviewed · '+meta.get('fit_role','role unassigned')) if meta.get('reviewed_for_fitting') else 'Imported · review pending' if p.name in manifest else 'Files present · import / review pending'
            for j,value in enumerate([p.name,str(len(list(p.glob('*.csv')))),state]):
                self.table.setItem(i,j,QTableWidgetItem(value))
        self.replay.setEnabled(bool(folders))
        if folders:self.table.selectRow(0)
        self.status.setText(f'{len(folders)} preliminary takes. Select a row to replay the original measurements.')

    def replay_selected(self,*_):
        item=self.table.item(self.table.currentRow(),0)
        if item is not None:self.replay_requested.emit(item.text())

    def import_recording(self):
        source=QFileDialog.getExistingDirectory(self,'Import a preliminary take')
        if source:
            try:
                name=import_take(self.root,source);self.refresh();self.status.setText('Imported '+name+' with the current hardware identity. Original source files preserved.')
            except (OSError,ValueError,KeyError) as error:self.status.setText(str(error))

    def open_folder(self):
        folder=self.root/'data/raw_takes';folder.mkdir(parents=True,exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
