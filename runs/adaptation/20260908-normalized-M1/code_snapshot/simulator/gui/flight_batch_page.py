"""Current paired recordings and explicit source-bound clock alignment."""
from pathlib import Path
from PySide6.QtCore import Signal,QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QLabel,QComboBox,QPushButton,
    QTableWidget,QTableWidgetItem,QHeaderView,QDoubleSpinBox,QLineEdit,QCheckBox)
from simulator.workflow import read_json
from experimental_data.adaptation_rounds import write_json
from experimental_data.adaptation_check import discover_batches,flight_names,recorded_alignment,sha256
from .research_widgets import note


class FlightBatchPage(QWidget):
    comparison_requested=Signal(str,str)

    def __init__(self,root):
        super().__init__();self.root=Path(root);layout=QVBoxLayout(self)
        layout.addWidget(note('Current flight batches · measured state from OptiTrack; commands from the controller log. Raw files remain unchanged.'))
        row=QHBoxLayout();self.batches=QComboBox();row.addWidget(self.batches,1)
        refresh=QPushButton('Refresh batches');refresh.clicked.connect(self.refresh);row.addWidget(refresh)
        folder=QPushButton('Open batch folder');folder.clicked.connect(self.open_folder);row.addWidget(folder);layout.addLayout(row)
        self.table=QTableWidget(0,3);self.table.setHorizontalHeaderLabels(['Flight pair','Clock offset [s]','Timing source / action needed'])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.verticalHeader().hide();layout.addWidget(self.table,1)
        row=QHBoxLayout();row.addWidget(QLabel('Drone rigid body'))
        self.drone=QLineEdit();self.drone.setPlaceholderText('Auto-detect one cf pose, or enter cf_3 / cf3 exactly');row.addWidget(self.drone,1)
        choose=QPushButton('Save drone selection');choose.clicked.connect(self.save_drone);row.addWidget(choose)
        self.drone_controls=QWidget();self.drone_controls.setLayout(row);layout.addWidget(self.drone_controls);self.drone_controls.hide()
        layout.addWidget(note('Clock convention: controller time = OptiTrack time + offset. Use shared timestamps or an identified shared event; do not match measured motion to the desired trajectory.'))
        row=QHBoxLayout();row.addWidget(QLabel('Offset [s]'));self.offset=QDoubleSpinBox();self.offset.setRange(-1e8,1e8);self.offset.setDecimals(6);row.addWidget(self.offset)
        self.source=QLineEdit();self.source.setPlaceholderText('Timestamp / shared event used to establish this offset');row.addWidget(self.source,1);layout.addLayout(row)
        self.reviewed=QCheckBox('I reviewed this clock offset for the selected pair');layout.addWidget(self.reviewed)
        row=QHBoxLayout();self.save=QPushButton('Save reviewed alignment');self.save.clicked.connect(self.save_alignment);row.addWidget(self.save)
        self.compare=QPushButton('Compare flight in 3D');self.compare.setObjectName('primaryButton');self.compare.clicked.connect(self.open_comparison);row.addWidget(self.compare);layout.addLayout(row)
        self.status=note('');layout.addWidget(self.status)
        self.batches.currentIndexChanged.connect(self.load_batch);self.table.itemSelectionChanged.connect(self.selection_changed)
        self.refresh()

    def refresh(self):
        previous=self.batches.currentData();self.batches.blockSignals(True);self.batches.clear()
        for p in discover_batches(self.root):self.batches.addItem(f'{p.parent.name} / {p.name}',str(p))
        self.batches.setCurrentIndex(max(0,self.batches.findData(previous)));self.batches.blockSignals(False);self.load_batch()

    def selected(self):
        i=self.table.currentRow();item=self.table.item(i,0) if i>=0 else None
        return item.text() if item is not None else None

    def load_batch(self):
        self.table.setRowCount(0);self.reviewed.setChecked(False)
        p=self.batches.currentData()
        if not p:return
        p=Path(p);names=flight_names(p);self.table.setRowCount(len(names))
        for i,name in enumerate(names):
            try:
                a=recorded_alignment(self.root,p,name,p/'flight_take'/f'{name}.csv',p/'flight_take'/f'experiment_{name}.csv')
                values=[name,f'{a["offset_s"]:.6f}',a['method']]
            except (ValueError,OSError,KeyError) as e:values=[name,'Not set',str(e)]
            for j,v in enumerate(values):self.table.setItem(i,j,QTableWidgetItem(v))
        self.table.resizeRowsToContents();self.status.setText(f'{len(names)} paired flights. Keep unpaired raw files; they are not ready for comparison.')
        if names:self.table.selectRow(0)

    def selection_changed(self):
        self.reviewed.setChecked(False);self.source.clear();self.offset.setValue(0);self.drone.clear();self.drone_controls.hide()
        name=self.selected()
        if name:
            batch=Path(self.batches.currentData())
            identity=read_json((batch/'flight_take'/f'{name}.csv').with_suffix('.tracking.json'),{})
            self.drone.setText(identity.get('drone',''))
            from experimental_data.adaptation_rounds import read_optitrack
            try:read_optitrack(batch/'flight_take'/f'{name}.csv')
            except (ValueError,OSError,KeyError,IndexError) as error:
                if 'Select the drone rigid body explicitly' in str(error):self.drone_controls.show()
            data=read_json(batch/'time_alignment.json',{}).get(name,{})
            if not data:
                try:
                    saved=recorded_alignment(self.root,batch,name,batch/'flight_take'/f'{name}.csv',batch/'flight_take'/f'experiment_{name}.csv')
                    data=dict(offset_s=saved['offset_s'],source=saved['method'])
                except (ValueError,OSError,KeyError):pass
            self.offset.setValue(data.get('offset_s',0));self.source.setText(data.get('source',''))

    def save_alignment(self):
        name=self.selected()
        if not name or not self.reviewed.isChecked() or not self.source.text().strip():
            self.status.setText('Select a flight, describe the timing evidence and mark the offset reviewed.');return
        p=Path(self.batches.currentData());path=p/'time_alignment.json';entries=read_json(path,{})
        entries[name]=dict(offset_s=self.offset.value(),source=self.source.text().strip(),clock_verified=False,
            optitrack_sha256=sha256(p/'flight_take'/f'{name}.csv'),controller_sha256=sha256(p/'flight_take'/f'experiment_{name}.csv'))
        from experimental_data.io import atomic_json
        atomic_json(path,entries);self.load_batch();self.status.setText('Reviewed offset saved for these exact source files. Raw measurements and flight prediction are unchanged.')

    def save_drone(self):
        name=self.selected();label=self.drone.text().strip()
        if not name or not label:
            self.status.setText('Select a flight and enter its exact OptiTrack rigid-body name.');return
        path=Path(self.batches.currentData())/'flight_take'/f'{name}.csv'
        try:
            from experimental_data.adaptation_rounds import read_optitrack
            from experimental_data.io import atomic_json
            read_optitrack(path,drone_label=label)
            atomic_json(path.with_suffix('.tracking.json'),dict(drone=label,optitrack_sha256=sha256(path)))
            self.status.setText('Saved drone '+label+' for this exact OptiTrack file. Pose coordinates and attachment geometry are unchanged.')
        except (ValueError,OSError,KeyError) as e:self.status.setText(str(e))

    def open_folder(self):
        if self.batches.currentData():QDesktopServices.openUrl(QUrl.fromLocalFile(self.batches.currentData()))

    def open_comparison(self):
        if self.selected():self.comparison_requested.emit(self.batches.currentData(),self.selected())
