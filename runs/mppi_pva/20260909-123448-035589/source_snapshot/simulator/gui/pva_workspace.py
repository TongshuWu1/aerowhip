"""Independent direct-PVA planner pages with one explicit setup per method."""
from copy import deepcopy
from pathlib import Path
import json
import shutil
import sys
from PySide6.QtCore import Qt,QTimer,Signal,QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QGroupBox,QLabel,QPushButton,QComboBox,
    QLineEdit,QSpinBox,QDoubleSpinBox,QCheckBox,QTabWidget,QScrollArea,QSplitter,QTableWidget,QTableWidgetItem,
    QHeaderView,QFileDialog,QPlainTextEdit)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from simulator.workflow import read_json,stamp
from experimental_data.io import atomic_json
from planning.pva_job import load_settings,settings_path,prepare,validate_settings
from .research_widgets import note,BackgroundJob
from .rehearsal_workspace import RehearsalWorkspace


def model_paths(root):
    root=Path(root);paths=[]
    for p in sorted((root/'runs/adaptation').glob('*/candidate/model.json'),reverse=True):
        if read_json(p,{}).get('provenance',{}).get('fit_complete'):paths.append(p)
    for p in sorted((root/'data/model_candidates').glob('*/model.json'),reverse=True):paths.append(p)
    return paths


def model_label(path):
    model=read_json(path,{})
    return model.get('provenance',{}).get('label',path.parent.name)


class PVAPlannerPage(QWidget):
    changed=Signal()
    def __init__(self,root,method):
        super().__init__();self.root=Path(root);self.method=method;self.cfg=load_settings(root,method)
        self.active=False;self.current_run=None;self.resume_checkpoint=None;self.fields={};self.last_history=None;self.library_fingerprint=None
        outer=QVBoxLayout(self);outer.setContentsMargins(18,12,18,14)
        self.tabs=QTabWidget();outer.addWidget(self.tabs)
        setup=QWidget();self.tabs.addTab(setup,'Setup and run');body=QVBoxLayout(setup)
        row=QHBoxLayout();row.addWidget(QLabel('Run name'));self.run_name=QLineEdit(method.upper()+' · PVA whip');row.addWidget(self.run_name,1)
        self.run=QPushButton('Start new training' if method=='ppo' else 'Optimize new plan');self.run.setObjectName('primaryButton');self.run.clicked.connect(self.start_run);row.addWidget(self.run);body.addLayout(row)
        row=QHBoxLayout();row.addWidget(QLabel('Fitted model'));self.models=QComboBox();row.addWidget(self.models,1)
        refresh=QPushButton('Refresh models');refresh.clicked.connect(self.refresh_models);row.addWidget(refresh);body.addLayout(row)
        self.contract=note('Desired P/V/A → loaded-drone response + NN → rotated cable attachment → DDER + NN. 30 Hz commands; the complete sequence is planned before flight.')
        self.contract.setObjectName('pipelineBanner');body.addWidget(self.contract)
        split=QSplitter();body.addWidget(split,1)
        left=QWidget();left_body=QVBoxLayout(left);left_body.setContentsMargins(0,0,8,0)
        self.add_vector(left_body,'Launch · OptiTrack tracked origin','launch','origin_m',-10,10,.01,'m')
        self.add_vector(left_body,'Target position','launch','target_m',-10,10,.01,'m')
        if method=='ppo':
            group=QGroupBox('Initial-state variation');form=QFormLayout(group)
            self.number(form,'Start radius [m]',('launch','start_radius_m'),0,.2,.01)
            self.number(form,'Target radius [m]',('launch','target_radius_m'),0,.2,.01);left_body.addWidget(group)
        group=QGroupBox('Maneuver and strike');form=QFormLayout(group)
        receding=method=='mppi' and self.cfg['mppi'].get('mode')=='receding'
        self.horizon=QSpinBox();self.horizon.setRange(3,150);self.horizon.setValue(round((self.cfg['mppi']['horizon_s'] if receding else self.cfg['task']['duration_s'])*30));form.addRow('MPPI lookahead intervals (30 Hz)' if receding else '30 Hz command intervals',self.horizon)
        self.duration=note('');form.addRow('Duration',self.duration);self.horizon.valueChanged.connect(lambda n:self.duration.setText(f'{n/30:.3f} s'))
        self.duration.setText(f'{self.horizon.value()/30:.3f} s')
        if receding:self.number(form,'Maneuver safety limit [s]',('task','duration_s'),.1,30.,1.)
        self.number(form,'Hit radius [m]',('task','target_radius_m'),.005,.3,.005)
        self.number(form,'Minimum directed tip speed [m/s]',('task','minimum_directed_speed_m_s'),.1,20,.25)
        self.number(form,'Maximum strike angle [deg]',('task','maximum_angle_deg'),1,90,1)
        self.first_contact=QCheckBox('Only the first target entry can count');self.first_contact.setChecked(self.cfg['task']['first_contact_only']);form.addRow(self.first_contact)
        left_body.addWidget(group);self.add_vector(left_body,'Desired strike direction','task','strike_direction',-1,1,.1,'unit vector')
        left_body.addStretch();scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(left);split.addWidget(scroll)
        right=QWidget();rb=QVBoxLayout(right);rb.setContentsMargins(8,0,0,0)
        controls=QTabWidget();rb.addWidget(controls)
        for title,section in [('Rewards','reward'),('PVA limits','limits'),('PPO training' if method=='ppo' else 'MPPI sampling','training' if method=='ppo' else 'mppi')]:
            content=QWidget();v=QVBoxLayout(content);group=QGroupBox(title);form=QFormLayout(group);v.addWidget(group)
            labels={
                'progress':'Closest-distance progress','strike_quality':'Near-target directed-speed progress','success':'Valid hit bonus',
                'time_per_s':'Time cost [reward / s]','failure':'Infeasible / numerical failure cost','invalid_contact':'Invalid first-contact cost',
                'displacement':'Drone displacement cost [reward / m²·s]','jerk':'Normalized jerk cost [reward / s]',
                'proximity_scale_m':'Speed-shaping distance [m]',
                'minimum_origin_z_m':'Minimum drone height [m]','maximum_origin_z_m':'Maximum drone height [m]',
                'minimum_cable_z_m':'Minimum cable height [m]','maximum_specific_force_m_s2':'Maximum specific acceleration [m/s²]',
                'minimum_specific_vertical_m_s2':'Minimum upward specific acceleration [m/s²]',
                'maximum_speed_m_s':'Maximum commanded speed [m/s]','maximum_tilt_deg':'Maximum reference tilt [deg]',
                'batch_size':'Parallel environments','hidden_dim':'Policy hidden width','learning_rate':'Learning rate','epochs':'PPO epochs per rollout',
                'minibatch_size':'Optimizer minibatch','entropy_coefficient':'Exploration entropy weight','seed':'Random seed',
                'minimum_attempts':'Minimum attempts before plateau stop','plateau_attempts':'Attempts without meaningful improvement',
                'maximum_attempts':'Maximum attempts (safety ceiling)','relative_improvement':'Meaningful relative improvement',
                'evaluate_every_updates':'Evaluate every N updates','samples':'Parallel candidate trajectories','iterations':'Iteration limit',
                'temperature':'MPPI temperature','noise_std':'Latent action noise std','noise_correlation':'Temporal noise correlation',
                'minimum_iterations':'Minimum iterations','patience':'Iterations without improvement'}
            for key,value in self.cfg[section].items():
                if section=='mppi' and key in ('mode','horizon_s'):continue
                upper=10000000 if isinstance(value,int) else (0.999 if key=='noise_correlation' else 10000)
                lower=0 if key in ('seed','iterations') or isinstance(value,float) else 1
                self.number(form,labels.get(key,key.replace('_',' ').capitalize()),(section,key),lower,upper,1 if isinstance(value,int) else .01,integer=isinstance(value,int))
                if section=='mppi' and key=='iterations':self.fields[(section,key)].setSpecialValueText('No limit')
            if section=='mppi':v.addWidget(note('Each lookahead optimizes until plateau or manual stop, then advances one 30 Hz command. The maneuver continues until a hit, failure or its separate safety limit. Terminal guidance is separate from actual strike reward.'))
            if section=='limits':v.addWidget(note('Specific acceleration includes gravity only for feasibility screening. The CSV acceleration is kinematic. These limits are provisional, not measured actuator limits.'))
            if section=='reward':v.addWidget(note('Return ends at the first valid modeled hit or the time limit. Recovery is appended after planning and receives no whip reward.'))
            v.addStretch();sc=QScrollArea();sc.setWidgetResizable(True);sc.setWidget(content);controls.addTab(sc,title)
        self.add_vector(rb,'Jerk action bounds','action','jerk_limit_m_s3',1,300,5,'m/s³')
        self.model_note=note('');rb.addWidget(self.model_note)
        save=QPushButton('Save this method’s setup');save.clicked.connect(self.save_settings);rb.addWidget(save)
        split.addWidget(right);split.setSizes([460,650])
        self.setup_note=note('PPO and MPPI settings are independent. Changes affect new runs; saved runs retain their model and launch.');body.addWidget(self.setup_note)
        self.models.currentIndexChanged.connect(self.describe_model);self.refresh_models()

        progress_page=QWidget();pb=QVBoxLayout(progress_page);self.tabs.addTab(progress_page,'Training progress' if method=='ppo' else 'Optimization progress')
        self.progress_note=note('Select a saved run or start a new one.');self.progress_note.setObjectName('pipelineBanner');pb.addWidget(self.progress_note)
        self.figure=Figure(layout='constrained',facecolor='white');self.canvas=FigureCanvasQTAgg(self.figure);pb.addWidget(self.canvas,1)
        options=QHBoxLayout();self.episode_points=QCheckBox('Show individual episodes · latest 10,000');self.episode_points.setVisible(method=='ppo')
        def change_points():self.last_history=None;self.poll()
        self.episode_points.toggled.connect(change_points);options.addWidget(self.episode_points)
        save_plot=QPushButton('Save plots…');save_plot.clicked.connect(self.save_plot);options.addWidget(save_plot);options.addStretch();pb.addLayout(options)
        row=QHBoxLayout();self.stop=QPushButton('Stop after current update');self.stop.clicked.connect(self.stop_run);row.addWidget(self.stop)
        row.addWidget(note('All results here are simulation results. Real-flight progress is measured separately.'));pb.addLayout(row)
        self.job=BackgroundJob(root);pb.addWidget(self.job);self.job.finished.connect(self.job_finished)

        library=QWidget();lb=QVBoxLayout(library);self.tabs.addTab(library,'Policy library' if method=='ppo' else 'Saved plans')
        row=QHBoxLayout();refresh=QPushButton('Refresh library');refresh.clicked.connect(self.refresh_library);row.addWidget(refresh);row.addStretch();lb.addLayout(row)
        self.library=QTableWidget(0,5);self.library.setHorizontalHeaderLabels(['Name','Model','Status','Attempts / iterations','Created'])
        self.library.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows);self.library.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.library.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers);self.library.verticalHeader().hide()
        self.library.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch);lb.addWidget(self.library,1)
        row=QHBoxLayout();self.checkpoint=QComboBox();self.checkpoint.addItems(['best.pt','latest.pt'])
        if method=='ppo':row.addWidget(self.checkpoint)
        for label,callback in [('Show progress',self.show_selected),('Rehearse selected',self.rehearse_selected),('Open files',self.open_selected),('Archive from library',self.archive_selected)]:
            button=QPushButton(label);button.clicked.connect(callback);row.addWidget(button)
        lb.addLayout(row)
        if method=='ppo':
            row=QHBoxLayout();continue_button=QPushButton('Continue selected policy in a new run');continue_button.clicked.connect(self.stage_continuation);row.addWidget(continue_button)
            fresh=QPushButton('Use fresh initialization');fresh.clicked.connect(self.clear_continuation);row.addWidget(fresh);lb.addLayout(row)
        self.library_note=note('Historical force policies remain in the historical workspace. Their weights are not compatible with jerk actions.');lb.addWidget(self.library_note)

        rehearsal=QWidget();re=QVBoxLayout(rehearsal);self.tabs.addTab(rehearsal,'Rehearsal and export')
        row=QHBoxLayout();self.rehearsal_note=note('Choose a completed policy or plan from the library.');row.addWidget(self.rehearsal_note,1)
        open_saved=QPushButton('Open saved PVA rehearsal');open_saved.clicked.connect(self.open_saved);row.addWidget(open_saved);re.addLayout(row)
        self.inspector=RehearsalWorkspace(root,inspection_only=True);re.addWidget(self.inspector,1)
        self.rehearsal_job=BackgroundJob(root);re.addWidget(self.rehearsal_job);self.rehearsal_job.finished.connect(self.rehearsal_finished)
        self.timer=QTimer(self);self.timer.setInterval(2000);self.timer.timeout.connect(self.poll);self.timer.start()
        self.tabs.currentChanged.connect(lambda _:self.set_page_active(self.active));self.refresh_library()

    def number(self,form,label,key,low,high,step,integer=False):
        widget=QSpinBox() if integer else QDoubleSpinBox()
        if not integer:widget.setDecimals(6 if key[-1] in ('learning_rate','entropy_coefficient','relative_improvement') else 4)
        widget.setRange(low,high);widget.setSingleStep(step);widget.setValue(self.cfg[key[0]][key[1]])
        self.fields[key]=widget;form.addRow(label,widget)

    def add_vector(self,layout,title,section,key,low,high,step,unit):
        group=QGroupBox(title+' ['+unit+']');row=QHBoxLayout(group);fields=[]
        for i,axis in enumerate('XYZ'):
            row.addWidget(QLabel(axis));spin=QDoubleSpinBox();spin.setDecimals(2 if section=='action' else 4);spin.setRange(low,high);spin.setSingleStep(step);spin.setValue(self.cfg[section][key][i]);spin.setFixedWidth(100);row.addWidget(spin);fields.append(spin)
        self.fields[(section,key)]=fields;layout.addWidget(group)

    def collect(self):
        cfg=deepcopy(self.cfg)
        for (section,key),widget in self.fields.items():cfg[section][key]=[w.value() for w in widget] if isinstance(widget,list) else widget.value()
        cfg['task']['first_contact_only']=self.first_contact.isChecked()
        if self.method=='mppi' and cfg['mppi'].get('mode')=='receding':cfg['mppi']['horizon_s']=self.horizon.value()/30
        else:cfg['task']['duration_s']=self.horizon.value()/30
        cfg['model_path']=self.models.currentData() or self.cfg['model_path'];validate_settings(cfg);return cfg

    def save_settings(self):
        try:self.cfg=self.collect();atomic_json(settings_path(self.root,self.method),self.cfg);self.setup_note.setText('Saved '+self.method.upper()+' setup. Existing runs and the other method are unchanged.');self.changed.emit()
        except (ValueError,OSError) as e:self.setup_note.setText(str(e))

    def refresh_models(self):
        selected=self.models.currentData() or self.cfg['model_path'];p=Path(selected);selected=str(p if p.is_absolute() else self.root/p)
        self.models.blockSignals(True);self.models.clear()
        for p in model_paths(self.root):self.models.addItem(model_label(p),str(p.resolve()))
        index=self.models.findData(selected)
        if index<0:self.models.addItem('Selected bootstrap model · fitting / unavailable',selected);index=self.models.count()-1
        self.models.setCurrentIndex(index);self.models.blockSignals(False);self.describe_model()

    def describe_model(self):
        path=self.models.currentData();m=read_json(path,{}) if path else {}
        self.run.setEnabled(bool(m) and m.get('provenance',{}).get('fit_complete') is not False and not (hasattr(self,'job') and self.job.running))
        provenance=m.get('provenance',{});mass=m.get('mass_measurement',{})
        self.model_note.setText((f'{provenance.get("label",self.models.currentText())}\nDrone {mass.get("drone_mass_kg",0)*1000:g} g · cable {mass.get("cable_assembly_mass_kg",0)*1000:g} g\n'+
            ('Normalized adp0 bootstrap; next flights provide prospective evidence.' if provenance.get('fit_complete') else 'Historical model for simulation. The current drone tracking problem remains under investigation.')) if m else 'Selected model unavailable. The stopped bootstrap has no finished candidate; select an existing model for simulation.')

    def select_model(self,path):
        i=self.models.findData(path)
        if i<0:self.models.addItem(model_label(Path(path)),path);i=self.models.count()-1
        self.models.setCurrentIndex(i);self.tabs.setCurrentIndex(0)

    def start_run(self):
        if self.job.running:return
        try:
            cfg=self.collect();directory,command=prepare(self.root,cfg,self.run_name.text(),checkpoint=self.resume_checkpoint if self.method=='ppo' else None)
            command[0]=str(self.root/'.venv/Scripts/python.exe')
            self.current_run=directory;self.job.start(directory,command);self.run.setEnabled(False);self.tabs.setCurrentIndex(1)
            self.refresh_library()
        except (OSError,ValueError,KeyError) as e:self.setup_note.setText('Could not start: '+str(e))

    def refresh_library(self):
        selected=self.selected() if hasattr(self,'runs') else None
        base=self.root/'runs'/('ppo_pva' if self.method=='ppo' else 'mppi_pva')
        self.runs=[p.parent for p in sorted(base.glob('*/identity.json'),reverse=True) if not (p.parent/'ARCHIVED').exists()]
        self.library_fingerprint=self.run_fingerprint()
        self.library.setRowCount(len(self.runs))
        for i,p in enumerate(self.runs):
            identity=read_json(p/'identity.json',{});status=read_json(p/'status.json',{})
            values=[identity.get('name',p.name),model_label(p/'model.json'),status.get('status','unknown'),str(status.get('attempts',status.get('iteration',status.get('iterations','—')))),p.name]
            for j,v in enumerate(values):self.library.setItem(i,j,QTableWidgetItem(v))
        if self.runs:
            self.library.selectRow(self.runs.index(selected) if selected in self.runs else 0)
            if self.current_run is None:self.current_run=self.runs[0]

    def run_fingerprint(self):
        base=self.root/'runs'/('ppo_pva' if self.method=='ppo' else 'mppi_pva')
        return tuple((str(p),p.stat().st_mtime_ns,(p.parent/'ARCHIVED').exists()) for p in sorted(base.glob('*/status.json')))

    def selected(self):
        row=self.library.currentRow();return self.runs[row] if 0<=row<len(self.runs) else None

    def show_selected(self):
        if self.selected():self.current_run=self.selected();self.last_history=None;self.tabs.setCurrentIndex(1);self.poll()

    def open_selected(self):
        if self.selected():QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.selected())))

    def archive_selected(self):
        p=self.selected()
        if not p:return
        if read_json(p/'status.json',{}).get('status')=='running':self.library_note.setText('Stop the run before archiving it.');return
        (p/'ARCHIVED').touch();self.library_note.setText('Hidden from this library. Files remain in '+str(p));self.refresh_library()

    def stage_continuation(self):
        p=self.selected()
        if not p:return
        checkpoint=p/'checkpoints'/self.checkpoint.currentText()
        if not checkpoint.exists():self.library_note.setText('Selected checkpoint is not available.');return
        self.resume_checkpoint=checkpoint;self.run_name.setText('Continue · '+read_json(p/'identity.json')['name'])
        old=read_json(p/'settings.json');self.fields[('training','hidden_dim')].setValue(old['training']['hidden_dim'])
        self.setup_note.setText('New run will initialize from '+str(checkpoint)+'. Review the selected model and setup.');self.tabs.setCurrentIndex(0)

    def clear_continuation(self):
        self.resume_checkpoint=None;self.setup_note.setText('Fresh PPO initialization selected.');self.tabs.setCurrentIndex(0)

    def stop_run(self):
        if self.current_run and read_json(self.current_run/'status.json',{}).get('status')=='running':(self.current_run/'STOP').touch()

    def job_finished(self,code):
        self.describe_model();self.refresh_library();self.poll()

    def poll(self):
        if self.run_fingerprint()!=self.library_fingerprint:self.refresh_library()
        p=self.current_run
        if not p:return
        status=read_json(p/'status.json',{});self.stop.setEnabled(status.get('status')=='running')
        identity=read_json(p/'identity.json',{});amount=status.get('attempts',status.get('iteration',status.get('iterations','—')))
        self.progress_note.setText(f'{identity.get("name",p.name)} · {status.get("status","unknown")} · {amount} '+('attempts' if self.method=='ppo' else 'iterations')+'\n'+status.get('stage',status.get('error',status.get('stop_reason','')))+
            ('\nAutomatic stopping follows the orange deterministic evaluation reward.' if self.method=='ppo' else
             ('\nSaved best plan: '+('infeasible' if status['best_failed'] else 'modeled valid hit' if status.get('best_success') else 'no modeled valid hit')) if 'best_failed' in status else ''))
        path=p/'history.json'
        if not path.exists() or self.last_history==(str(p),path.stat().st_mtime_ns):return
        self.last_history=(str(p),path.stat().st_mtime_ns);rows=read_json(path,[])
        if not rows:return
        self.figure.clear();axes=self.figure.subplots(2,2).ravel();x=[r.get('attempts',r.get('iteration')) for r in rows]
        keys=['reward','success','minimum_tip_distance_m','failures'] if self.method=='ppo' else ['best_reward','success','best_minimum_tip_distance_m','effective_samples']
        titles=['Episode return (batch mean)','Modeled valid hits','Closest tip distance','Infeasible attempts'] if self.method=='ppo' else ['Saved best return','Modeled valid hits (sampled batch)','Saved best tip distance','Effective importance samples']
        for ax,key,title in zip(axes,keys,titles):
            factor=100 if key in ('success','failures') else 1;ax.plot(x,[r.get(key,0)*factor for r in rows],color='#2563eb',lw=1.6,label='Training batch' if self.method=='ppo' else None)
            ax.set_title(title,loc='left',fontsize=10);ax.set_xlabel('Attempts' if self.method=='ppo' else 'Iterations');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
            if key in ('success','failures'):ax.set_ylabel('%');ax.set_ylim(0,100)
            if 'distance' in key:ax.set_ylabel('m')
        if self.method=='ppo':
            for ax,key,scale in ((axes[0],'evaluation_reward',1),(axes[1],'evaluation_success',100)):
                evaluation=[r for r in rows if key in r]
                if evaluation:
                    ax.plot([r['attempts'] for r in evaluation],[r[key]*scale for r in evaluation],color='#ea580c',lw=1.7,label='Deterministic evaluation')
                    ax.legend(frameon=False,fontsize=8)
        if self.method=='ppo' and self.episode_points.isChecked() and (p/'episodes.csv').exists():
            import io,numpy as np
            file=p/'episodes.csv'
            with file.open('rb') as stream:
                start=max(0,file.stat().st_size-1500000);stream.seek(start);lines=stream.read().decode('utf-8').splitlines()
            if start:lines=lines[1:]
            elif lines:lines=lines[1:]
            if lines:
                data=np.genfromtxt(io.StringIO('\n'.join(lines[-10000:])),delimiter=',',invalid_raise=False)
                data=np.atleast_2d(data)
                if data.shape[1]==6:
                    for ax,col,scale in [(axes[0],1,1),(axes[1],2,100),(axes[2],5,1),(axes[3],3,100)]:
                        ax.scatter(data[:,0],data[:,col]*scale,s=2,alpha=.15,color='#64748b')
        self.canvas.draw_idle()

    def save_plot(self):
        path,_=QFileDialog.getSaveFileName(self,'Save learning plots',str(self.root/'runs'/(self.method+'-progress.png')),'PNG (*.png);;PDF (*.pdf)')
        if path:self.figure.savefig(path,dpi=180)

    def rehearse_selected(self):
        p=self.selected()
        if p is None or self.rehearsal_job.running:return
        if read_json(p/'status.json',{}).get('status')=='running':self.library_note.setText('Stop or finish the run before rehearsing.');return
        cfg=read_json(p/'settings.json');checkpoint=p/'checkpoints'/self.checkpoint.currentText()
        if self.method=='ppo' and not checkpoint.exists():self.library_note.setText('Checkpoint not yet available.');return
        if self.method=='mppi' and not (p/'plan.npz').exists():self.library_note.setText('Plan not yet available.');return
        self.rehearsal_output=self.root/'runs/rehearsals_pva'/(stamp()+'-'+self.method)
        tool=p/'source_snapshot/tools/rehearse_pva.py'
        command=[str(self.root/'.venv/Scripts/python.exe'),'-u',str(tool),'--job',str(p),'--output',str(self.rehearsal_output)]
        if self.method=='ppo':command.extend(['--checkpoint',str(checkpoint)])
        self.rehearsal_note.setText('Generating from the selected run’s frozen launch and model: '+read_json(p/'identity.json')['name'])
        self.rehearsal_job.start(self.root/'runs/pva_jobs'/stamp(),command);self.tabs.setCurrentIndex(3);self.inspector.clear_result()

    def rehearsal_finished(self,code):
        if code:self.rehearsal_note.setText('Generation failed. Open the log; no new command has been exported.');return
        try:self.inspector.load_result(self.rehearsal_output);self.rehearsal_note.setText('Frozen run rehearsal · complete command is ready for inspection.');self.changed.emit()
        except (OSError,ValueError,KeyError) as e:self.rehearsal_note.setText(str(e))

    def open_saved(self):
        p=QFileDialog.getExistingDirectory(self,'Open saved PVA rehearsal',str(self.root/'runs/rehearsals_pva'))
        if p:
            try:self.inspector.clear_result();self.inspector.load_result(p)
            except (OSError,ValueError,KeyError) as e:self.rehearsal_note.setText(str(e))

    def set_page_active(self,active):
        self.active=active;self.inspector.set_page_active(active and self.tabs.currentIndex()==3)

    def shutdown(self):
        self.timer.stop();return self.inspector.shutdown()
