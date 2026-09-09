"""Saved adaptation evidence, separate from the exact flown-CSV ghost."""
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt,Signal
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QComboBox,QPushButton,
    QLabel,QTabWidget,QTableWidget,QTableWidgetItem,QSlider,QFileDialog,QHeaderView)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from experimental_data.adaptation_progress import discover_studies,load_study,comparison_arrays,policies_for_model,METRICS
from .research_widgets import note


class AdaptationProgressPage(QWidget):
    model_requested=Signal(str)

    def __init__(self,root):
        super().__init__();self.root=Path(root);self.study=None;self.arrays=None
        layout=QVBoxLayout(self);row=QHBoxLayout();layout.addLayout(row)
        row.addWidget(QLabel('Adaptation study'));self.studies=QComboBox();row.addWidget(self.studies,1)
        refresh=QPushButton('Refresh studies');refresh.clicked.connect(self.refresh);row.addWidget(refresh)
        self.status=note('');layout.addWidget(self.status)
        self.summary=note('');layout.addWidget(self.summary)
        self.tabs=QTabWidget();layout.addWidget(self.tabs,1)
        overview=QWidget();body=QVBoxLayout(overview);self.tabs.addTab(overview,'Model improvement')
        self.figure=Figure(figsize=(9,4),layout='constrained');self.canvas=FigureCanvasQTAgg(self.figure);body.addWidget(self.canvas,1)
        self.table=QTableWidget();self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch);body.addWidget(self.table)
        body.addWidget(note('Bars: equal-weight means of five separate held-out flights. Dots: individual flights. Lower is better. Strike error means prediction vs measurement at the saved strike time; it is not distance to the target.'))
        trajectory=QWidget();body=QVBoxLayout(trajectory);self.tabs.addTab(trajectory,'Compare model trajectories')
        row=QHBoxLayout();body.addLayout(row);self.take=QComboBox();row.addWidget(self.take)
        self.mode=QComboBox();self.mode.addItem('Held-out fitted model (flight excluded)','heldout');self.mode.addItem('Published all-data model (in-sample)','all_five');row.addWidget(self.mode,1)
        self.trace_note=note('');body.addWidget(self.trace_note)
        self.trace_figure=Figure(figsize=(9,5),layout='constrained');self.trace_canvas=FigureCanvasQTAgg(self.trace_figure);body.addWidget(self.trace_canvas,1)
        row=QHBoxLayout();body.addLayout(row);row.addWidget(QLabel('Whip frame'));self.timeline=QSlider(Qt.Orientation.Horizontal);row.addWidget(self.timeline,1)
        self.time_label=QLabel('');row.addWidget(self.time_label)
        self.next_step=note('');layout.addWidget(self.next_step)
        row=QHBoxLayout();layout.addLayout(row)
        self.use_model=QPushButton('Select adapted model for a new PPO run');self.use_model.clicked.connect(self.select_model);row.addWidget(self.use_model)
        export=QPushButton('Export comparison figure…');export.clicked.connect(self.export_figure);row.addWidget(export)
        self.studies.currentIndexChanged.connect(self.load_selected)
        self.take.currentIndexChanged.connect(self.load_trace);self.mode.currentIndexChanged.connect(self.load_trace)
        self.timeline.valueChanged.connect(self.draw_trace);self.refresh()

    def refresh(self):
        previous=self.studies.currentData();self.studies.blockSignals(True);self.studies.clear()
        for path in discover_studies(self.root):self.studies.addItem(path.name,str(path))
        self.studies.setCurrentIndex(max(0,self.studies.findData(previous)));self.studies.blockSignals(False);self.load_selected()

    def load_selected(self,*_):
        self.study=None;self.arrays=None;self.use_model.setEnabled(False)
        self.figure.clear();self.trace_figure.clear();self.table.setRowCount(0);self.summary.clear();self.next_step.clear()
        path=self.studies.currentData()
        if not path:
            self.status.setText('No completed adaptation study yet. Flight replay remains available on the other tabs.');return
        try:
            self.study=load_study(self.root,path);s=self.study
            self.status.setText(f'{Path(path).name} | {len(s["rows"])} whole-flight held-out comparisons | saved results, no fitting during display')
            b=s['means']['baseline_tip_whip_rmse_m'];a=s['means']['adapted_tip_whip_rmse_m']
            self.summary.setText(f'Whip tip RMS: {100*b:.2f} → {100*a:.2f} cm ({100*(b-a)/b:.1f}% reduction). '
                'These held-out fits assess the adaptation procedure. The published M1 uses all five flights and needs a new-flight test.')
            axes=self.figure.subplots(1,4)
            for axis,(key,label) in zip(axes,METRICS):
                values=[[100*r[prefix+'_'+key] for r in s['rows']] for prefix in ('baseline','adapted')]
                axis.bar([0,1],[np.mean(v) for v in values],color=['#64748b','#0d9488'],alpha=.7)
                for j,v in enumerate(values):axis.scatter(j+np.linspace(-.1,.1,len(v)),v,color='#172033',s=14,zorder=3)
                axis.set(xticks=[0,1],xticklabels=['M0','Adapted'],title=label,ylabel='Error [cm]');axis.grid(axis='y',alpha=.2)
            self.canvas.draw_idle()
            headers=['Flight']+[name+' [cm]\nM0 → adapted' for name in ('Drone whip RMS','Markers whip RMS','Tip whip RMS','Tip error at strike')]
            self.table.setColumnCount(len(headers));self.table.setHorizontalHeaderLabels(headers);self.table.setRowCount(len(s['rows']))
            for i,r in enumerate(s['rows']):
                values=[r['take']]+[f'{100*r["baseline_"+key]:.2f} → {100*r["adapted_"+key]:.2f}' for key,_ in METRICS]
                for j,v in enumerate(values):self.table.setItem(i,j,QTableWidgetItem(v))
            self.take.blockSignals(True);self.take.clear();self.take.addItems([r['take'] for r in s['rows']]);self.take.blockSignals(False)
            assessment=s['assessment'];extra=''
            if 'baseline_whip_attitude_rmse_deg' in assessment:
                extra=f'Other outcomes: whip attitude RMS {assessment["baseline_whip_attitude_rmse_deg"]:.2f} → {assessment["adapted_whip_attitude_rmse_deg"]:.2f}°; late-hold drone RMS {100*assessment["baseline_late_drone_rmse_m"]:.2f} → {100*assessment["adapted_late_drone_rmse_m"]:.2f} cm.\n'
            models=s['models'];self.use_model.setEnabled(len(models)==1)
            policies=policies_for_model(self.root,models[0]) if len(models)==1 else []
            stage=('M1 PPO: '+', '.join(run.get('display_name',p.name)+' — '+status.get('status','prepared') for p,run,status in policies)) if policies else 'M1 PPO: no linked run yet.'
            self.next_step.setText(extra+stage+'\nNext: freeze a new policy/model/CSV, collect adp1, then assess its prediction and target error before fitting M2.')
            self.load_trace()
        except (OSError,ValueError,KeyError) as error:self.status.setText('Cannot load saved study: '+str(error))

    def load_trace(self,*_):
        self.arrays=None
        if not self.study or not self.take.currentText():return
        try:
            self.arrays=comparison_arrays(self.study,self.take.currentText(),self.mode.currentData())
            t=self.arrays[0]['time'];end=self.study['protocol']['phases']['whip'][1]
            self.timeline.setRange(0,max(0,int(np.searchsorted(t,end,side='right')-1)));self.timeline.setValue(0)
            mode='separate model fitted without this flight' if self.mode.currentData()=='heldout' else 'published model fitted using this flight (in-sample)'
            self.trace_note.setText('Orange: measured | Gray: M0 | Teal: '+mode+'. Same recorded commands and initial-state protocol. These are saved model comparisons, separate from the original flight ghost.\nMeasurements here use the saved 150 Hz display grid; headline RMS uses native 100 Hz samples.')
            self.draw_trace()
        except (OSError,ValueError,KeyError) as error:self.trace_note.setText('No saved comparison: '+str(error))

    def draw_trace(self,*_):
        if self.arrays is None:return
        base,adapted=self.arrays;i=self.timeline.value();n=self.timeline.maximum()+1;t=base['time'][:n]
        self.time_label.setText(f'{base["time"][i]:.3f} s');self.trace_figure.clear()
        axis=self.trace_figure.add_subplot(121,projection='3d');errors=self.trace_figure.add_subplot(122)
        for label,color,position,cable in [('Measured','#ea580c',base['measured_position'],base['measured_sites']),
                ('M0','#64748b',base['position'],base['cable']),('Adapted','#0d9488',adapted['position'],adapted['cable'])]:
            tip=cable[:n,-1];axis.plot(*tip.T,color=color,label=label+' tip')
            axis.plot(*position[:n].T,color=color,linestyle=':',alpha=.7)
            axis.plot(*cable[i].T,color=color,marker='o',markersize=3);axis.scatter(*position[i],color=color,s=30)
        allpoints=np.concatenate([base['measured_sites'][:n].reshape(-1,3),base['cable'][:n].reshape(-1,3),adapted['cable'][:n].reshape(-1,3)])
        span=np.nanmax(allpoints,axis=0)-np.nanmin(allpoints,axis=0);axis.set_box_aspect(np.maximum(span,.1))
        axis.set(xlabel='X [m]',ylabel='Y [m]',zlabel='Z [m]',title='Tip paths and cable at selected time');axis.legend(fontsize=7)
        for label,color,data in [('M0','#64748b',base),('Adapted','#0d9488',adapted)]:
            errors.plot(t,100*np.linalg.norm(data['cable'][:n,-1]-base['measured_sites'][:n,-1],axis=1),color=color,label=label+' tip')
            errors.plot(t,100*np.linalg.norm(data['position'][:n]-base['measured_position'][:n],axis=1),color=color,linestyle=':',label=label+' drone')
        errors.axvline(t[i],color='#172033',alpha=.5);errors.set(xlabel='Time from CSV onset [s]',ylabel='Prediction error [cm]',title='Whip prediction');errors.grid(alpha=.2);errors.legend(fontsize=8)
        self.trace_canvas.draw_idle()

    def select_model(self):
        if self.study and len(self.study['models'])==1:self.model_requested.emit(str(self.study['models'][0]))

    def export_figure(self):
        path,_=QFileDialog.getSaveFileName(self,'Export current comparison figure','adaptation-comparison.png','PNG (*.png);;PDF (*.pdf)')
        if path:(self.figure if self.tabs.currentIndex()==0 else self.trace_figure).savefig(path,dpi=300)
