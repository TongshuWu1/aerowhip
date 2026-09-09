"""Actual recorded-flight outcomes; missing adaptation rounds have no score."""
import numpy as np
from PySide6.QtCore import QThread,Signal
from PySide6.QtWidgets import QWidget,QVBoxLayout,QPushButton,QTableWidget,QTableWidgetItem,QHeaderView
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from experimental_data.adaptation_progress import recorded_flight_progress,FLIGHT_METRICS
from .research_widgets import note


class FlightProgressLoader(QThread):
    loaded=Signal(object)
    failed=Signal(str)
    def __init__(self,root,parent=None):super().__init__(parent);self.root=root
    def run(self):
        try:self.loaded.emit(recorded_flight_progress(self.root))
        except Exception as error:self.failed.emit(str(error))


class RealFlightProgressPage(QWidget):
    def __init__(self,root):
        super().__init__();self.root=root;self.worker=None;self.data=None
        layout=QVBoxLayout(self)
        self.status=note('M1: awaiting adp1 flights. No adapted-flight RMS is available yet.');layout.addWidget(self.status)
        self.refresh_button=QPushButton('Refresh normalized flight results');self.refresh_button.clicked.connect(self.refresh);layout.addWidget(self.refresh_button)
        self.figure=Figure(figsize=(9,4),layout='constrained');self.canvas=FigureCanvasQTAgg(self.figure);layout.addWidget(self.canvas,1)
        self.table=QTableWidget();self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers);self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch);layout.addWidget(self.table)
        layout.addWidget(note('All scores use hover-normalized drone/cable Z versus the exact saved forecast and reference target. These are normalized comparison metrics, not uncorrected physical hitting distances. Batch calibration uses pre/post hover. Means weight complete flights equally; missing data stays excluded. Model-fit cross-validation never supplies flight scores.'))
        self.draw(dict(rows=[dict(round='adp1',model='M1',actual_flights=0,complete_flights=0,flights=[],means={k:None for k,_ in FLIGHT_METRICS})],errors=[]))
        self.data=None

    def refresh(self):
        if self.worker is not None:return
        self.refresh_button.setEnabled(False);self.status.setText('Reading real flight pairs and matching their original saved CSV predictions...')
        self.worker=FlightProgressLoader(self.root,self);self.worker.loaded.connect(self.draw)
        self.worker.failed.connect(lambda message:self.status.setText('Cannot load recorded results: '+message))
        self.worker.finished.connect(self.finished);self.worker.start()

    def finished(self):
        self.worker.deleteLater();self.worker=None;self.refresh_button.setEnabled(True)

    def draw(self,data):
        self.data=data;rows=data['rows'];self.figure.clear();axes=self.figure.subplots(1,4)
        labels=[r['model']+' / '+r['round'] for r in rows]
        for axis,(key,label) in zip(axes,FLIGHT_METRICS):
            for i,row in enumerate(rows):
                value=row['means'].get(key)
                if value is None:axis.text(i,.5,'Awaiting flights' if not row['actual_flights'] else 'No complete data',ha='center',rotation=90,transform=axis.get_xaxis_transform(),color='#64748b')
                else:
                    axis.bar(i,value*100,color='#2563eb' if row['model']=='M0' else '#0d9488',alpha=.7)
                    samples=[f[key]*100 for f in row['flights'] if f['complete_whip'] and f[key] is not None]
                    axis.scatter(i+np.linspace(-.08,.08,len(samples)),samples,color='#172033',s=15,zorder=3)
            axis.set(xticks=np.arange(len(rows)),xticklabels=labels,title=label.replace(' at planned strike','\nat planned strike'),ylabel='Normalized error [cm]',xlim=(-.6,len(rows)-.4));axis.tick_params(axis='x',labelrotation=15);axis.grid(axis='y',alpha=.2)
        self.canvas.draw_idle()
        details=[]
        for r in rows:
            details.append((r['model']+' / '+r['round']+' mean',f'{r["complete_flights"]}/{r["actual_flights"]} complete',r['means']))
            details.extend(('  '+f['take'],'complete' if f['complete_whip'] else 'partial - excluded from mean',f) for f in r['flights'])
        self.table.setColumnCount(6);self.table.setHorizontalHeaderLabels(['Round / flight','Coverage','Drone RMS [cm]','Tip RMS [cm]','Target @ strike [cm]','Closest target [cm]']);self.table.setRowCount(len(details))
        for i,(name,coverage,values) in enumerate(details):
            texts=[name,coverage]+['—' if values.get(key) is None else f'{100*values[key]:.2f}' for key,_ in FLIGHT_METRICS]
            for j,text in enumerate(texts):self.table.setItem(i,j,QTableWidgetItem(text))
        pending=any(r['model']=='M1' and not r['actual_flights'] for r in rows)
        self.status.setText(('M1: Awaiting adp1 flights. No measured M1 performance claim.\n' if pending else '')+
            f'{sum(r["actual_flights"] for r in rows)} hover-normalized flight pairs loaded. '+('\nIssues: '+'; '.join(data['errors']) if data['errors'] else ''))

    def shutdown(self):
        return self.worker is None or not self.worker.isRunning()
