"""Inspect the provisional tracking surrogate; never applies it or sends commands."""
from pathlib import Path
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QPushButton,QComboBox,
    QLabel,QTableWidget,QTableWidgetItem,QHeaderView)
from simulator.workflow import read_json
from experimental_data.drone_residual_fit import prepare_job
from .research_widgets import BackgroundJob,note


class DroneResidualPage(QWidget):
    def __init__(self,root):
        super().__init__();self.root=Path(root)
        layout=QVBoxLayout(self)
        layout.addWidget(note('Initial drone tracking model + small NN residual. Uses yesterday’s logged full-state commands '
            '(including early hold) and OptiTrack cf_7 position. Whip trials 1–2 train; trial 3 checks prediction.'))
        row=QHBoxLayout()
        self.train=QPushButton('Fit initial drone residual')
        self.train.clicked.connect(self.start);row.addWidget(self.train)
        self.runs=QComboBox();self.runs.currentIndexChanged.connect(self.select)
        row.addWidget(self.runs,1)
        refresh=QPushButton('Refresh');refresh.clicked.connect(self.refresh);row.addWidget(refresh)
        layout.addLayout(row)
        self.summary=note('');layout.addWidget(self.summary)
        self.table=QTableWidget(0,4)
        self.table.setHorizontalHeaderLabels(['Trial / role','Tracking model RMSE (cm)','With NN RMSE (cm)','With NN max error (cm)'])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table,1)
        self.open=QPushButton('Open trajectory comparison')
        self.open.clicked.connect(self.open_plot);layout.addWidget(self.open)
        layout.addWidget(note('Provisional model for one repeated maneuver and fixed cable setup. '
            'Fitted delay includes logging uncertainty; fitted gains are not firmware parameters. '
            'This page does not change the active simulation, cable calibration, PPO, controller or logger.'))
        self.job=BackgroundJob(root);self.job.finished.connect(self.finished);layout.addWidget(self.job)
        self.refresh()

    def start(self):
        if self.job.running:return
        try:
            job,command=prepare_job(self.root)
            self.job.start(job,command);self.train.setEnabled(False)
            self.refresh()
        except (OSError,ValueError,KeyError) as error:self.job.status.setText(str(error))

    def finished(self,code):
        self.train.setEnabled(True);self.refresh()
        index=self.runs.findData(str(self.job.directory))
        if index>=0:self.runs.setCurrentIndex(index)

    def refresh(self):
        selected=self.runs.currentData();self.runs.blockSignals(True);self.runs.clear()
        for p in sorted((self.root/'data/drone_residual_runs').glob('*/settings.json'),reverse=True):
            self.runs.addItem(p.parent.name,str(p.parent))
        index=self.runs.findData(selected)
        if index>=0:self.runs.setCurrentIndex(index)
        self.runs.blockSignals(False);self.select()

    def select(self):
        path=self.runs.currentData();self.table.setRowCount(0);self.open.setEnabled(False)
        if not path:self.summary.setText('No initial drone residual fitted yet.');return
        folder=Path(path);review=read_json(folder/'review.json',{})
        if not review:
            status=read_json(folder/'status.json',{})
            self.summary.setText(status.get('error',read_json(folder/'progress.json',{}).get('label','Prepared')))
            return
        self.summary.setText(('Prediction improved on trial 3' if review['validation_improved'] else 'No accepted prediction improvement on trial 3')+
            f' · selected update {review["selected_update"]} · not applied to simulation')
        results=read_json(folder/'evaluation.json')
        self.table.setRowCount(len(results))
        for i,(name,row) in enumerate(results.items()):
            values=[f'{name} / {row["role"]}',f'{row["nominal"]["position_rmse_m"]*100:.2f}',
                f'{row["residual"]["position_rmse_m"]*100:.2f}',f'{row["residual"]["maximum_position_error_m"]*100:.2f}']
            for j,value in enumerate(values):self.table.setItem(i,j,QTableWidgetItem(value))
        self.open.setEnabled((folder/'tracking_comparison.png').is_file())

    def open_plot(self):
        if self.runs.currentData():
            path=Path(self.runs.currentData())/'tracking_comparison.png'
            if path.is_file():QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))
