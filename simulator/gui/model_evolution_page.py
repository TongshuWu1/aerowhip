"""Model generations, same-flight diagnostics, and prospective flight evidence."""
from datetime import datetime
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt, QUrl, Signal, QTimer
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QComboBox,
    QTableWidget,QTableWidgetItem,QHeaderView,QTabWidget,QFileDialog,QProgressBar,QFrame,
    QDialog,QDialogButtonBox,QFormLayout,QLineEdit,QPlainTextEdit)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from .research_widgets import note, BackgroundJob
from experimental_data import model_evaluation as evaluation
from experimental_data.whip_adaptation import verify_hashes
from simulator.workflow import read_json

METRICS = [('command_driven_tip','Tip prediction'),('command_driven_markers','Cable shape prediction'),
           ('drone','Drone prediction'),('conditional_cable_tip','Cable with measured attachment')]


def table(headers):
    widget=QTableWidget(0,len(headers));widget.setHorizontalHeaderLabels(headers)
    widget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    widget.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    widget.verticalHeader().hide()
    return widget


def fill(widget,rows):
    widget.setRowCount(len(rows))
    for i,row in enumerate(rows):
        for j,value in enumerate(row):
            item=QTableWidgetItem(str(value));item.setToolTip(str(value));widget.setItem(i,j,item)


def cm(value):
    return '—' if value is None else f'{100*value:.2f}'


class ModelEvolutionPage(QWidget):
    flight_requested=Signal(str,str)

    def __init__(self,root,parent=None):
        super().__init__(parent);self.root=Path(root);self.report=None;self.report_folder=None
        self.catalog={};self.flight_rows=[]
        layout=QVBoxLayout(self);layout.setSpacing(12)
        row=QHBoxLayout();layout.addLayout(row)
        self.cards=[]
        for caption in ('MODEL LINEAGE','RECORDED TAKES','NEXT STEP'):
            card=QFrame();card.setObjectName('metricCard');body=QVBoxLayout(card)
            label=QLabel(caption);label.setObjectName('metricCaption');body.addWidget(label)
            value=QLabel('—');value.setWordWrap(True);value.setStyleSheet('font-size:16px;font-weight:600;');body.addWidget(value)
            self.cards.append(value);row.addWidget(card,1)
        row=QHBoxLayout();layout.addLayout(row)
        refresh=QPushButton('Refresh');refresh.clicked.connect(self.refresh);row.addWidget(refresh)
        register=QPushButton('Add completed fit…');register.clicked.connect(self.register_fit);row.addWidget(register)
        flight=QPushButton('Add flight comparison…');flight.clicked.connect(self.register_flight);row.addWidget(flight)
        guide=QPushButton('Evaluation guide');guide.clicked.connect(lambda:QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.root/'docs/SIM_REAL_EVALUATION.md'))));row.addWidget(guide)
        row.addStretch()
        self.status=note('');layout.addWidget(self.status)
        self.models=table(['Model','Parent','Evidence','Damping [/s]','Identity']);self.models.setMaximumHeight(155);layout.addWidget(self.models)
        self.tabs=QTabWidget();layout.addWidget(self.tabs,1)
        same=QWidget();body=QVBoxLayout(same);self.tabs.addTab(same,'Same flight · compare models')
        row=QHBoxLayout();body.addLayout(row)
        self.reports=QComboBox();row.addWidget(self.reports,1)
        browse=QPushButton('Open evaluation…');browse.clicked.connect(self.browse_report);row.addWidget(browse)
        self.run=QPushButton('Evaluate models on reviewed data…');self.run.clicked.connect(self.run_evaluation);row.addWidget(self.run)
        self.metric_select=QComboBox()
        for key,label in METRICS:self.metric_select.addItem(label,key)
        self.role=QComboBox();self.role.addItem('Validation takes','validation');self.role.addItem('Adaptation takes','adaptation');self.role.addItem('All takes · development','all')
        row=QHBoxLayout();body.addLayout(row);row.addWidget(self.metric_select);row.addWidget(self.role);row.addStretch()
        self.figure=Figure(figsize=(9,3),layout='constrained');self.canvas=FigureCanvasQTAgg(self.figure);self.canvas.setMinimumHeight(180);body.addWidget(self.canvas,1)
        self.scores=table(['Take','Model','Data use','Error [cm]','Coverage','Interval [s]']);self.scores.setMaximumHeight(110);body.addWidget(self.scores)
        self.mean_note=note('Select one reviewed dataset. All models use its same commands, initial state, time grid and observation masks.');body.addWidget(self.mean_note)
        row=QHBoxLayout();body.addLayout(row)
        self.progress=QProgressBar();self.progress.setRange(0,1);self.progress.setValue(0);self.progress.setFormat('No evaluation running');row.addWidget(self.progress,1)
        self.stop=QPushButton('Stop after model');self.stop.setEnabled(False);self.stop.clicked.connect(self.stop_evaluation);row.addWidget(self.stop)
        export=QPushButton('Export figure…');export.clicked.connect(self.export_figure);row.addWidget(export)
        self.job=BackgroundJob(self.root);body.addWidget(self.job);self.job.hide()
        self.job.progress.connect(self.on_progress);self.job.finished.connect(self.on_finished)
        traces=QWidget();body=QVBoxLayout(traces);self.tabs.addTab(traces,'Prediction traces')
        row=QHBoxLayout();body.addLayout(row);self.take=QComboBox();row.addWidget(QLabel('Recorded take'));row.addWidget(self.take,1)
        self.trace_figure=Figure(figsize=(9,4),layout='constrained');self.trace_canvas=FigureCanvasQTAgg(self.trace_figure);body.addWidget(self.trace_canvas,1)
        body.addWidget(note('Each model overlays the same measured take. Error curves preserve missing observations as gaps.'))
        real=QWidget();body=QVBoxLayout(real);self.tabs.addTab(real,'Real flights · original forecasts')
        self.flights=table(['Model / take','Command identity','Drone error [cm]','Tip error [cm]','Nearest target [cm]','Tip coverage','Outcome']);body.addWidget(self.flights,1)
        self.replay=QPushButton('Open selected flight in replay');self.replay.clicked.connect(self.replay_flight);body.addWidget(self.replay)
        review=QPushButton('Review selected flight outcome…');review.clicked.connect(self.review_flight);body.addWidget(review)
        self.outcomes=note('No assessed real flights yet.');body.addWidget(self.outcomes)
        body.addWidget(note('Each row uses the forecast saved before that flight. Nearest sampled tip distance is geometry; physical contact remains a separately reviewed outcome. Changed commands make cross-round results a combined model + planner comparison.'))
        fits=QWidget();body=QVBoxLayout(fits);self.tabs.addTab(fits,'Fitting progress')
        self.fit_jobs=QComboBox();body.addWidget(self.fit_jobs)
        self.fit_status=note('No reviewed whip fitting job yet.');body.addWidget(self.fit_status)
        self.fit_progress=QProgressBar();body.addWidget(self.fit_progress)
        self.fit_figure=Figure(figsize=(9,3),layout='constrained');self.fit_canvas=FigureCanvasQTAgg(self.fit_figure);body.addWidget(self.fit_canvas,1)
        self.fit_log=note('');body.addWidget(self.fit_log)
        body.addWidget(note('Loss is a training diagnostic. Each stage retains its best state and reports why it stopped. Validation and a prospective flight test model accuracy.'))
        self.reports.currentIndexChanged.connect(self.load_report)
        self.metric_select.currentIndexChanged.connect(self.draw);self.role.currentIndexChanged.connect(self.draw)
        self.take.currentIndexChanged.connect(self.draw_trace);self.fit_jobs.currentIndexChanged.connect(self.draw_fit)
        self.timer=QTimer(self);self.timer.setInterval(2000);self.timer.timeout.connect(lambda:self.draw_fit() if self.isVisible() else None);self.timer.start()
        self.refresh()

    def refresh(self):
        try:
            self.catalog=evaluation.load_catalog(self.root);rows=[]
            for model in self.catalog['models']:
                try:
                    verify_hashes(model['hashes']);value=read_json(model['model']);state=model['status']
                    damping=f'{value["cable"]["external_drag_s_inv"]:.4g}'
                except (OSError,ValueError,KeyError) as exc:
                    state='Evidence changed / unavailable';damping='—'
                rows.append([model['id'],model['parent'] or '—',state,damping,model['signature'][:12]])
            fill(self.models,rows)
            row_height=sum(self.models.rowHeight(i) for i in range(min(3,len(rows)))) if rows else self.models.verticalHeader().defaultSectionSize()
            self.models.setFixedHeight(self.models.horizontalHeader().sizeHint().height()+row_height+8)
            edges=[f'{m["parent"]} → {m["id"]}' for m in self.catalog['models'] if m.get('parent')]
            roots=[m['id'] for m in self.catalog['models'] if not m.get('parent')]
            self.cards[0].setText(' · '.join(edges or roots) or 'No registered models')
            self.refresh_flights()
            self.cards[1].setText(str(sum(bool(f.get('take')) for f in self.flight_rows)))
            self.cards[2].setText('Collect M0 flight' if not any(f.get('take') for f in self.flight_rows) else 'Review prediction and flight evidence')
            self.status.setText('Frozen model versions · original forecasts preserved')
            selected=self.reports.currentData();self.reports.blockSignals(True);self.reports.clear()
            for p in sorted((self.root/'runs/evaluation').glob('*/report.json'),reverse=True):
                if read_json(p,{}).get('schema')==evaluation.REPORT:self.reports.addItem(p.parent.name,str(p.parent))
            self.reports.setCurrentIndex(max(0,self.reports.findData(selected)));self.reports.blockSignals(False)
            selected=self.fit_jobs.currentData();self.fit_jobs.blockSignals(True);self.fit_jobs.clear()
            for p in sorted((self.root/'runs/adaptation').glob('*/protocol.json'),reverse=True):
                if read_json(p,{}).get('schema')=='prospective_whip_adaptation_v1':self.fit_jobs.addItem(p.parent.name,str(p.parent))
            self.fit_jobs.setCurrentIndex(max(0,self.fit_jobs.findData(selected)));self.fit_jobs.blockSignals(False)
            self.run.setEnabled(len(rows)>=2 and self.fit_jobs.count()>0 and not self.job.running)
            self.load_report();self.draw_fit()
        except (OSError,ValueError,KeyError) as exc:self.status.setText(str(exc))

    def refresh_flights(self):
        rows=[];self.flight_rows=[];outcomes=[]
        for flight in self.catalog.get('flights',[]):
            try:
                verify_hashes({flight['command_csv']:flight['command_sha256'],str(Path(flight['rehearsal'])/'rehearsal.npz'):flight['forecast_sha256']})
                verify_hashes(flight.get('hashes',{}))
                takes=read_json(Path(flight['comparison'])/'report.json')['takes'] if flight.get('comparison') else {}
                reviewed=evaluation.flight_outcomes(flight)
                for take,value in takes.items():
                    outcome=reviewed.get(take,dict(outcome='unknown',execution='unknown'))
                    outcomes.append(dict(outcome,model=flight['model']))
                    label=outcome['outcome']+' · '+outcome['execution'] if take in reviewed else 'Not reviewed'
                    rows.append([flight['model']+' / '+take,flight['command_sha256'][:12],cm(value['drone']['rmse_m']),cm(value['tip']['rmse_m']),
                        cm(value['minimum_observed_tip_target_m']),f'{100*value["tip"]["coverage"]:.1f}%',label])
                    self.flight_rows.append(dict(**flight,take=take))
                if not takes:
                    rows.append([flight['model'],flight['command_sha256'][:12],'—','—','—','—','Awaiting recorded flight']);self.flight_rows.append(flight)
            except (OSError,ValueError,KeyError):
                rows.append([flight['model'],'Evidence changed','—','—','—','—','Comparison blocked']);self.flight_rows.append({})
        fill(self.flights,rows);self.replay.setEnabled(any(f.get('take') for f in self.flight_rows))
        counts=evaluation.success_counts(outcomes)
        self.outcomes.setText(f'{counts["attempts"]} recorded takes · {counts["hits"]} reviewed feasible hits · {counts["failures"]} failures · {counts["unknown"]} unresolved. Compare rounds using the individual rows; these counts combine all commands.')
        per_model=[]
        for model in self.catalog.get('models',[]):
            c=evaluation.success_counts([r for r in outcomes if r['model']==model['id']])
            per_model.append(f'{model["id"]}: {c["hits"]} hits / {c["hits"]+c["failures"]} assessed; {c["unknown"]} unresolved')
        if outcomes:self.outcomes.setText('  |  '.join(per_model))

    def register_fit(self):
        path=QFileDialog.getExistingDirectory(self,'Choose completed reviewed fit',str(self.root/'runs/adaptation'))
        if path:
            try:
                if read_json(Path(path)/'protocol.json',{}).get('full_update'):
                    from experimental_data.whip_full_fit import register
                    register(path,read_json(Path(path)/'fit/result.json')['comparison'])
                else:evaluation.add_candidate(self.root,path)
                self.refresh()
            except (OSError,ValueError,KeyError) as exc:self.status.setText(str(exc))

    def register_flight(self):
        path=QFileDialog.getExistingDirectory(self,'Choose frozen-forecast comparison',str(self.root/'runs/data_review'))
        if path:
            try:evaluation.add_flight_report(self.root,path);self.refresh()
            except (OSError,ValueError,KeyError) as exc:self.status.setText(str(exc))

    def browse_report(self):
        path=QFileDialog.getExistingDirectory(self,'Open same-flight evaluation',str(self.root/'runs/evaluation'))
        if path:
            i=self.reports.findData(path)
            if i<0:self.reports.addItem(Path(path).name,path);i=self.reports.count()-1
            self.reports.setCurrentIndex(i)

    def load_report(self):
        self.report=None;self.report_folder=None;self.take.clear()
        path=self.reports.currentData()
        if path:
            try:
                self.report=evaluation.load_evaluation(path);self.report_folder=Path(path)
                first=next(iter(self.report['models'].values()));self.take.addItems(list(first['takes']))
            except (OSError,ValueError,KeyError) as exc:self.status.setText(str(exc))
        self.draw()

    def draw(self):
        self.figure.clear();axis=self.figure.subplots();fill(self.scores,[]);self.scores.hide()
        if not self.report:
            axis.axis('off');axis.text(.5,.55,'Compare M0, M1, M2 on the same recorded flight',ha='center',fontsize=14)
            axis.text(.5,.4,'Awaiting reviewed flight data and a completed adapted model',ha='center',color='#64748b')
            self.canvas.draw_idle();self.draw_trace();return
        key=self.metric_select.currentData();role=self.role.currentData()
        try:names,values=evaluation.paired_summary(self.report,key,role)
        except ValueError as exc:
            axis.axis('off');axis.text(.5,.5,str(exc),ha='center');self.canvas.draw_idle();self.status.setText(str(exc));return
        models=list(self.report['models']);rows=[]
        for model in models:
            for name in names:
                row=self.report['models'][model]['takes'][name];m=row['metrics'][key]
                rows.append([name,model,row['data_use'],cm(m['rmse_m']),f'{100*m["coverage"]:.1f}%',f'0–{row["end_s"]:.3f}'])
        fill(self.scores,rows)
        self.scores.setVisible(bool(rows))
        if names:
            for i,name in enumerate(names):
                v=[100*x[i] if x[i] is not None else np.nan for x in values]
                axis.plot(models,v,'o-',alpha=.7,label=name)
            means=[np.mean([100*x for x in v]) if v and all(x is not None for x in v) else np.nan for v in values]
            axis.plot(models,means,'D--',color='#111827',lw=2,label='Equal-take mean')
            axis.legend(frameon=False,fontsize=8);axis.set_ylabel('Prediction RMSE [cm]');axis.set_title(self.metric_select.currentText(),loc='left');axis.grid(axis='y',alpha=.2)
        else:axis.axis('off');axis.text(.5,.5,'No takes with this role. No validation claim is available.',ha='center')
        self.mean_note.setText(f'{len(names)} paired takes · raw XYZ · fixed observations and interval · lower error is better. Repeated inspection supports development; it does not create a new independent test.')
        self.canvas.draw_idle();self.draw_trace()

    def draw_trace(self):
        self.trace_figure.clear();axes=self.trace_figure.subplots(1,2)
        if not self.report or not self.take.currentText():
            for axis in axes:axis.axis('off');axis.text(.5,.5,'No evaluated take yet',ha='center')
        else:
            name=self.take.currentText();key=self.metric_select.currentData()
            for model,value in self.report['models'].items():
                a,score,errors=evaluation.metric_arrays(self.report_folder/model/(name+'.npz'),value['marker_ids'],value['takes'][name]['end_s'])
                error=errors[key]
                if error.ndim>1:
                    valid=np.isfinite(error);error=np.sqrt(np.divide(np.nansum(error**2,axis=1),valid.sum(1),out=np.full(len(error),np.nan),where=valid.sum(1)>0))
                axes[0].plot(a['time_s'][score],100*error[score],label=model)
                points=a['coupled_cable'][:,-1];axes[1].plot(points[score,0],points[score,2],label=model)
            measured=a['measured_sites'][:,-1].copy();measured[~a['mask'][:,-1]]=np.nan
            axes[1].plot(measured[score,0],measured[score,2],color='#ea580c',lw=2,label='Measured')
            axes[0].set(xlabel='Time from command onset [s]',ylabel='Error [cm]',title=self.metric_select.currentText())
            axes[1].set(xlabel='World X [m]',ylabel='World Z [m]',title='Tip path · command-driven');axes[1].set_aspect('equal',adjustable='datalim')
            for axis in axes:axis.legend(frameon=False,fontsize=8);axis.grid(alpha=.2)
        self.trace_canvas.draw_idle()

    def run_evaluation(self):
        if self.job.running:return
        path=QFileDialog.getExistingDirectory(self,'Choose reviewed prepared data (all registered models will be compared)',str(self.root/'runs/adaptation'))
        if not path:return
        out=self.root/'runs/evaluation'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        # Log folder differs from immutable result folder, created by the worker.
        log=out.parent/(out.name+'-process')
        command=[str(self.root/'.venv/Scripts/python.exe'),'-u','tools/evaluate_models.py','evaluate','--job',path,
                 '--models',*[m['id'] for m in self.catalog['models']],'--output',str(out),'--device','cuda']
        self.output=out;self.job.show();self.job.start(log,command);self.run.setEnabled(False);self.stop.setEnabled(True)
        self.progress.setRange(0,0);self.progress.setFormat('Preparing reviewed data…')
        self.poll_timer=QTimer(self);self.poll_timer.setInterval(1000);self.poll_timer.timeout.connect(self.poll_evaluation);self.poll_timer.start()

    def poll_evaluation(self):
        p=read_json(self.output/'progress.json',{})
        if p:self.on_progress(p)

    def on_progress(self,p):
        self.progress.setRange(0,p.get('total',1));self.progress.setValue(p.get('completed',0));self.progress.setFormat(p.get('label','Evaluating…'))

    def on_finished(self,code):
        if hasattr(self,'poll_timer'):self.poll_timer.stop()
        self.progress.setRange(0,1);self.progress.setValue(1 if code==0 else 0)
        self.progress.setFormat('Completed' if code==0 else 'Stopped or failed · see log');self.stop.setEnabled(False);self.refresh()
        if code==0:self.reports.setCurrentIndex(self.reports.findData(str(self.output)))

    def stop_evaluation(self):
        if hasattr(self,'output') and self.output.exists():(self.output/'STOP').touch();self.stop.setEnabled(False)

    def replay_flight(self):
        i=self.flights.currentRow()
        if 0<=i<len(self.flight_rows):
            f=self.flight_rows[i]
            if f.get('take'):self.flight_requested.emit(f['batch'],f['take'])

    def review_flight(self):
        i=self.flights.currentRow()
        if not 0<=i<len(self.flight_rows) or not self.flight_rows[i].get('take'):
            self.status.setText('Select a recorded flight row first.');return
        flight=self.flight_rows[i];dialog=QDialog(self);dialog.setWindowTitle('Review '+flight['model']+' / '+flight['take'])
        layout=QFormLayout(dialog)
        outcome=QComboBox();outcome.addItems(['unknown','hit','miss'])
        execution=QComboBox();execution.addItems(['unknown','feasible','infeasible'])
        reviewer=QLineEdit();evidence=QPlainTextEdit();evidence.setPlaceholderText('Observation supporting the tip-target outcome and execution assessment; record uncertainty or missing evidence.')
        layout.addRow('Tip-target outcome',outcome);layout.addRow('Execution',execution);layout.addRow('Reviewed by',reviewer);layout.addRow('Evidence / uncertainty',evidence)
        layout.addRow(note('A virtual target hit and a physical collision are different observations. Preserve unknown when the recording cannot establish the outcome.'))
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Save|QDialogButtonBox.StandardButton.Cancel);layout.addRow(buttons)
        buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject)
        if dialog.exec()==QDialog.DialogCode.Accepted:
            try:
                evaluation.review_outcome(self.root,flight['comparison'],flight['take'],outcome.currentText(),execution.currentText(),reviewer.text(),evidence.toPlainText())
                self.refresh()
            except (OSError,ValueError,KeyError) as exc:self.status.setText(str(exc))

    def draw_fit(self):
        path=self.fit_jobs.currentData();self.fit_figure.clear();axis=self.fit_figure.subplots()
        if not path:
            self.fit_progress.setRange(0,1);self.fit_progress.setValue(0);self.fit_progress.setFormat('Awaiting reviewed fitting inputs')
            axis.axis('off');axis.text(.5,.5,'No real adaptation fit yet',ha='center');self.fit_canvas.draw_idle();return
        job=Path(path);status=read_json(job/'status.json',{});history=read_json(job/'fit/history.json',[])
        if read_json(job/'protocol.json',{}).get('full_update'):
            progress=read_json(job/'progress.json',{})
            stages=['drone_nominal','drone_residual','attitude_refinement','cable_physics','cable_residual']
            complete=sum((job/s/'result.json').exists() or (job/s/'parameters.json').exists() for s in stages)
            done=status.get('status')=='completed'
            self.fit_status.setText(f'{job.name} · {status.get("status","prepared")} · {status.get("stage",progress.get("stage","Full adaptation"))}')
            self.fit_progress.setRange(0,6);self.fit_progress.setValue(6 if done else complete)
            self.fit_progress.setFormat('All stages and validation complete' if done else f'{complete}/6 stages complete')
            active=str(progress.get('stage',''))
            folder=job/active
            h=read_json(folder/'history.json',[])
            if not h:
                candidates=sorted((job/'drone_nominal').glob('delay-*/history.json'))
                if candidates and 'nominal' in active:h=read_json(candidates[-1],[])
            if h:axis.plot([r['update'] for r in h],[r['best_loss'] for r in h],label='Stage training loss');axis.legend(frameon=False)
            self.fit_log.setText(f'Update {progress.get("update","—")} · best loss {progress.get("best_loss","—")} · '+str(status.get('error','')))
            axis.set(xlabel='Stage update',ylabel='Training objective');self.fit_canvas.draw_idle();return
        self.fit_status.setText(f'{job.name} · {status.get("status","prepared")} · {status.get("stop_reason",status.get("stage","reviewed data"))}')
        self.fit_progress.setRange(0,12);self.fit_progress.setValue(len(history));self.fit_progress.setFormat(f'{len(history)} updates · ceiling 12; may plateau earlier')
        if history:
            axis.plot([r['update'] for r in history],[r['best_loss'] for r in history],'o-',label='Best training loss')
            axis.axhline(history[0]['baseline_loss'],ls='--',color='#64748b',label='Parent model');axis.legend(frameon=False)
            last=history[-1]
            parameter='Horizontal response gain' if last.get('parameter')=='feedforward_xy' else 'Cable damping [/s]'
            self.fit_log.setText(f'{parameter} {last["best"]:.4g} · elapsed {last["elapsed_s"]:.1f} s · best loss {last["best_loss"]:.5g}')
        axis.set(xlabel='Update',ylabel='Robust training loss');self.fit_canvas.draw_idle()

    def export_figure(self):
        if not self.report:return
        path,_=QFileDialog.getSaveFileName(self,'Export current comparison figure','model-comparison.png','PNG (*.png);;PDF (*.pdf);;SVG (*.svg)')
        if path:
            self.figure.savefig(path,dpi=200)
            from experimental_data.io import atomic_json,sha256_file
            atomic_json(Path(path).with_suffix('.metadata.json'),dict(report=str(self.report_folder/'report.json'),
                report_sha256=sha256_file(self.report_folder/'report.json'),metric=self.metric_select.currentData(),role=self.role.currentData()))

    def shutdown(self):
        if self.job.running:self.stop_evaluation();return False
        self.timer.stop();return True
