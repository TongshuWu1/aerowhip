"""Independent direct-PVA planner pages with one explicit setup per method."""
from copy import deepcopy
from pathlib import Path
import json
import shutil
import sys
import time
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
from .ppo_training_dashboard import PPOTrainingDashboard


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
        if method=='mppi' or self.cfg.get('observation_contract') in ('pva_whip_phase_v2','pva_whip_phase_v3'):
            for key in ('vertical_excursion','vertical_tip_velocity','vertical_tip_alignment','horizontal_contact',
                        'drone_approach','lateral_excursion','near_target_reach','outward_contact'):
                self.cfg['reward'].setdefault(key,0.)
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
        from learning.pva_success import criterion,TIP_CONTACT,LEGACY
        self.strike_form=form;self.success_rule=QComboBox()
        self.success_rule.addItem('Tip reaches target',TIP_CONTACT)
        self.success_rule.addItem('Legacy speed / strategy / wave gates',LEGACY)
        self.success_rule.setCurrentIndex(self.success_rule.findData(criterion(self.cfg['task'])))
        form.addRow('Success condition',self.success_rule)
        self.success_note=note('');form.addRow(self.success_note)
        receding=method=='mppi' and self.cfg['mppi'].get('mode')=='receding'
        self.horizon=QSpinBox();self.horizon.setRange(3,150);self.horizon.setValue(round((self.cfg['mppi']['horizon_s'] if receding else self.cfg['task']['duration_s'])*30));form.addRow('MPPI lookahead intervals (30 Hz)' if receding else '30 Hz command intervals',self.horizon)
        self.duration=note('');form.addRow('Lookahead' if receding else 'Duration',self.duration);self.horizon.valueChanged.connect(lambda n:self.duration.setText(f'{n/30:.3f} s'))
        self.duration.setText(f'{self.horizon.value()/30:.3f} s')
        if receding:self.number(form,'Maneuver safety limit [s]',('task','duration_s'),.1,30.,1.)
        self.number(form,'Hit radius [m]',('task','target_radius_m'),.005,.3,.005)
        self.number(form,'Minimum directed tip speed [m/s]',('task','minimum_directed_speed_m_s'),.1,20,.25)
        self.number(form,'Maximum strike angle [deg]',('task','maximum_angle_deg'),1,90,1)
        self.first_contact=QCheckBox('Only the first target entry can count');self.first_contact.setChecked(self.cfg['task']['first_contact_only']);form.addRow(self.first_contact)
        if method=='mppi' or self.cfg.get('observation_contract') in ('pva_whip_phase_v2','pva_whip_phase_v3'):
            self.pullback=QCheckBox('Require forward pull, then backward drone motion at contact')
            self.pullback.setChecked(self.cfg['task'].get('require_pullback',False));form.addRow(self.pullback)
            from learning.whip_wave import DEFAULTS
            self.wave=QCheckBox('Require a travelling bend before tip contact')
            self.wave.setChecked(self.cfg['task'].get('require_wave',False));form.addRow(self.wave)
            for key,value in DEFAULTS.items():
                self.cfg['task'].setdefault(key,value)
                self.number(form,key.replace('wave_','Bend ').replace('_',' '),('task',key),.001,3.14,.01)
            for key,label,value in (
                ('minimum_pull_distance_m','Forward pull distance [m]',.25),
                ('minimum_pull_speed_m_s','Forward pull speed [m/s]',1.),
                ('minimum_backward_distance_m','Backward travel before contact [m]',.1),
                ('minimum_backward_speed_m_s','Backward drone speed at contact [m/s]',.5)):
                self.cfg['task'].setdefault(key,value);self.number(form,label,('task',key),.01,5.,.05)
        self.gate_controls=[widget for (section,key),widget in self.fields.items()
            if section=='task' and key not in ('target_radius_m','duration_s')]
        self.gate_controls+=[self.first_contact]+[getattr(self,key) for key in ('pullback','wave') if hasattr(self,key)]
        self.success_rule.currentIndexChanged.connect(self.update_success_controls);self.update_success_controls()
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
                'pull_phase':'Forward-pull progress reward','reverse_phase':'Backward-release progress reward',
                'wave_progress':'Travelling-bend progress reward',
                'reach_progress':'Outward release progress reward (once per episode)',
                'brake_progress':'Braking after pull progress reward',
                'joint_strike':'Joint tip position / direction / release reward',
                'command_acceleration':'Specific acceleration cost [reward / s at limit]',
                'reset_value_on_resume':'Reset value estimator and optimizers when continuing',
                'terminate_invalid_contact':'End attempt at invalid first contact',
                'success_priority':'Prefer valid hit rate for best policy and plateau stopping',
                'command_speed':'Commanded speed cost [reward / s at speed limit]',
                'reward_scale':'PPO learning reward scale',
                'initial_log_std':'Initial exploration log standard deviation',
                'gae_lambda':'Credit assignment decay (GAE lambda)',
                'vertical_excursion':'Vertical drone excursion cost [reward / m²·s]',
                'vertical_tip_velocity':'Near-strike vertical tip velocity cost',
                'vertical_tip_alignment':'Near-strike vertical cable alignment cost',
                'horizontal_contact':'Horizontal contact quality bonus',
                'drone_approach':'Forward drone approach cost [reward / m²·s]',
                'lateral_excursion':'Sideways drone excursion cost [reward / m²·s]',
                'near_target_reach':'Near-target cable reach shortfall cost',
                'outward_contact':'Outward cable reach at contact bonus',
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
                'minimum_iterations':'Minimum iterations per lookahead','patience':'Iterations without improvement',
                'initial_minimum_iterations':'First lookahead minimum iterations','initial_patience':'First lookahead plateau patience',
                'strike_exit_speed':'Strike-exit speed cost','strike_exit_climb':'Strike-exit upward speed cost',
                'strike_exit_acceleration':'Strike-exit forward acceleration cost',
                'control_prior':'Additional zero-jerk preference [0–1]',
                'terminal_distance':'Terminal tip-position guidance','terminal_velocity':'Terminal strike-velocity guidance',
                'terminal_anchor':'Terminal drone-position guidance',
                'terminal_command_speed':'Terminal command-speed cost','terminal_command_acceleration':'Terminal command-acceleration cost'}
            for key,value in self.cfg[section].items():
                if isinstance(value,bool):
                    checkbox=QCheckBox();checkbox.setChecked(value);self.fields[(section,key)]=checkbox
                    form.addRow(labels.get(key,key.replace('_',' ').capitalize()),checkbox);continue
                if section=='mppi' and key in ('mode','horizon_s','initialization'):continue
                if section=='mppi' and key=='noise_scales':
                    row=QHBoxLayout();fields=[]
                    for scale in value:
                        spin=QDoubleSpinBox();spin.setDecimals(3);spin.setRange(.001,3.);spin.setValue(scale)
                        fields.append(spin);row.addWidget(spin)
                    self.fields[(section,key)]=fields;form.addRow('Mixed exploration noise scales',row)
                    continue
                upper=10000000 if isinstance(value,int) else (0.999 if key=='noise_correlation' else 1. if key in ('control_prior','gae_lambda') else 10000)
                lower=-10 if key=='initial_log_std' else 0 if key in ('seed','iterations') or isinstance(value,float) else 1
                self.number(form,labels.get(key,key.replace('_',' ').capitalize()),(section,key),lower,upper,1 if isinstance(value,int) else .01,integer=isinstance(value,int))
                if section=='mppi' and key=='noise_std' and self.cfg['mppi'].get('noise_scales'):
                    self.fields[(section,key)].setEnabled(False)
                    self.fields[(section,key)].setToolTip('The mixed exploration scales above replace this single-scale setting.')
                if section=='mppi' and key=='iterations':self.fields[(section,key)].setSpecialValueText('No limit')
            if section=='mppi':
                self.initialization=QComboBox();self.initialization.addItem('Zero jerk','zero');self.initialization.addItem('Forward/backward jerk guesses','pullback')
                self.initialization.addItem('Varied whip pulse timing and lift','wave')
                self.initialization.setCurrentIndex(self.initialization.findData(self.cfg['mppi'].get('initialization','zero')))
                form.addRow('Initial proposal',self.initialization)
                v.addWidget(note('Each lookahead optimizes until plateau or manual stop, then advances one 30 Hz command. The maneuver continues until a hit, failure or its separate safety limit. Terminal guidance is separate from actual strike reward.'))
            if section=='limits':v.addWidget(note('Specific acceleration includes gravity only for feasibility screening. The CSV acceleration is kinematic. These limits are provisional, not measured actuator limits.'))
            if section=='reward':v.addWidget(note('Return ends at the first valid modeled hit or the time limit. Recovery is appended after planning and receives no whip reward.'))
            v.addStretch();sc=QScrollArea();sc.setWidgetResizable(True);sc.setWidget(content);controls.addTab(sc,title)
        self.add_vector(rb,'Jerk action bounds','action','jerk_limit_m_s3',1,300,5,'m/s³')
        self.model_note=note('');rb.addWidget(self.model_note)
        save=QPushButton('Save this method’s setup');save.clicked.connect(self.save_settings);rb.addWidget(save)
        split.addWidget(right);split.setSizes([460,650])
        self.setup_note=note('PPO and MPPI settings are independent. Changes affect new runs; saved runs retain their model and launch.');body.addWidget(self.setup_note)
        if method=='mppi':
            quick=QPushButton('Quick setup · 256 samples');quick.clicked.connect(self.quick_setup);rb.addWidget(quick)
            self.live_enabled=QCheckBox('Record live 3D candidate snapshots');self.live_enabled.setChecked(self.cfg.get('visualization',{}).get('live_mppi',True));rb.addWidget(self.live_enabled)
        self.models.currentIndexChanged.connect(self.describe_model);self.refresh_models()

        progress_page=QWidget();pb=QVBoxLayout(progress_page);self.tabs.addTab(progress_page,'Training progress' if method=='ppo' else 'Optimization progress')
        if method=='mppi':
            progress_scroll=QScrollArea();progress_scroll.setWidgetResizable(True);progress_scroll.setFrameShape(QScrollArea.Shape.NoFrame);pb.addWidget(progress_scroll)
            progress_content=QWidget();pb=QVBoxLayout(progress_content);progress_scroll.setWidget(progress_content)
        row=QHBoxLayout();row.addWidget(QLabel('Run'));self.progress_runs=QComboBox();self.progress_runs.setMinimumContentsLength(20)
        self.progress_runs.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.progress_runs.currentIndexChanged.connect(self.select_progress_run);row.addWidget(self.progress_runs,1)
        self.stop=QPushButton('Stop after current update');self.stop.setEnabled(False);self.stop.clicked.connect(self.stop_run);row.addWidget(self.stop);pb.addLayout(row)
        details_body=pb
        if method=='mppi':
            from .mppi_dashboard import MPPIDashboard
            self.mppi_dashboard=MPPIDashboard();pb.addWidget(self.mppi_dashboard)
        if method=='ppo':
            preview_row=QHBoxLayout();self.preview_checkpoint=QComboBox()
            self.preview_checkpoint.addItem('Latest saved policy','latest.pt');self.preview_checkpoint.addItem('Best evaluated policy','best.pt')
            preview_row.addWidget(self.preview_checkpoint)
            self.preview_complete=QCheckBox('Include recovery / export');preview_row.addWidget(self.preview_complete)
            self.preview_button=QPushButton('Rehearse policy');self.preview_button.setObjectName('primaryButton')
            self.preview_button.clicked.connect(self.rehearse_current);preview_row.addWidget(self.preview_button);preview_row.addStretch();pb.addLayout(preview_row)
            self.policy_preview_note=note('Replay a frozen checkpoint from this run. Training can continue; refresh to see newer learning.')
            pb.addWidget(self.policy_preview_note)
            self.progress_views=QTabWidget();pb.addWidget(self.progress_views,1)
            self.dashboard=PPOTrainingDashboard();dashboard_scroll=QScrollArea();dashboard_scroll.setWidgetResizable(True)
            dashboard_scroll.setFrameShape(QScrollArea.Shape.NoFrame);dashboard_scroll.setWidget(self.dashboard)
            self.progress_views.addTab(dashboard_scroll,'Overview')
            details=QWidget();details_body=QVBoxLayout(details);self.progress_views.addTab(details,'Trends and logs')
        self.progress_note=note('Select a saved run or start a new one.');self.progress_note.setObjectName('pipelineBanner');details_body.addWidget(self.progress_note)
        self.figure=Figure(layout='constrained',facecolor='white');self.canvas=FigureCanvasQTAgg(self.figure);details_body.addWidget(self.canvas,1)
        if method=='mppi':self.progress_note.hide();self.canvas.setMinimumHeight(180)
        options=QHBoxLayout();self.episode_points=QCheckBox('Show individual episodes · latest 10,000');self.episode_points.setVisible(method=='ppo')
        def change_points():self.last_history=None;self.poll()
        self.episode_points.toggled.connect(change_points);options.addWidget(self.episode_points)
        if method=='mppi':
            self.mppi_chart=QComboBox();self.mppi_chart.addItems(['Current lookahead search','Committed path trends']);self.mppi_chart.currentIndexChanged.connect(change_points);options.addWidget(self.mppi_chart)
            live_button=QPushButton('Watch live 3D');live_button.clicked.connect(lambda:self.tabs.setCurrentWidget(self.live_view));options.addWidget(live_button)
        save_plot=QPushButton('Save plots…');save_plot.clicked.connect(self.save_plot);options.addWidget(save_plot)
        files=QPushButton('Open run files');files.clicked.connect(self.open_progress_files);options.addWidget(files);options.addStretch();details_body.addLayout(options)
        self.saved_log=QPlainTextEdit();self.saved_log.setReadOnly(True);self.saved_log.setMaximumHeight(150);self.saved_log.setMaximumBlockCount(200);self.saved_log.setVisible(method=='mppi')
        if method=='mppi':self.saved_log.setMinimumHeight(100)
        self.log_toggle=QCheckBox('Show selected run log');self.log_toggle.toggled.connect(self.toggle_progress_log)
        self.follow_log=QCheckBox('Follow latest log');self.follow_log.setChecked(True);options.addWidget(self.follow_log)
        details_body.addWidget(self.log_toggle);details_body.addWidget(self.saved_log)
        self.job=BackgroundJob(root);self.job.setParent(self);self.job.hide();self.job.finished.connect(self.job_finished)

        library=QWidget();lb=QVBoxLayout(library);self.tabs.addTab(library,'Policy library' if method=='ppo' else 'Saved plans')
        row=QHBoxLayout();refresh=QPushButton('Refresh library');refresh.clicked.connect(self.refresh_library);row.addWidget(refresh);row.addStretch();lb.addLayout(row)
        headers=['Name','Model','Status','Attempts / iterations','Created']
        if method=='mppi':headers.append('Strike definition')
        self.library=QTableWidget(0,len(headers));self.library.setHorizontalHeaderLabels(headers)
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
        row=QHBoxLayout();self.rehearsal_note=note('Rehearse a saved PPO checkpoint or a completed MPPI plan.');row.addWidget(self.rehearsal_note,1)
        self.preview_context=None
        self.refresh_preview=QPushButton('Refresh policy rehearsal');self.refresh_preview.setVisible(method=='ppo');self.refresh_preview.setEnabled(False)
        self.refresh_preview.clicked.connect(self.refresh_policy_rehearsal);row.addWidget(self.refresh_preview)
        open_saved=QPushButton('Open saved PVA rehearsal');open_saved.clicked.connect(self.open_saved);row.addWidget(open_saved);re.addLayout(row)
        self.inspector=RehearsalWorkspace(root,inspection_only=True);re.addWidget(self.inspector,1)
        self.rehearsal_job=BackgroundJob(root);re.addWidget(self.rehearsal_job);self.rehearsal_job.finished.connect(self.rehearsal_finished)
        if method=='mppi':
            from .mppi_live_view import MPPILiveView
            self.live_view=MPPILiveView();self.tabs.addTab(self.live_view,'Live 3D search')
        self.timer=QTimer(self);self.timer.setInterval(2000);self.timer.timeout.connect(self.poll);self.timer.start()
        self.tabs.currentChanged.connect(lambda _:self.set_page_active(self.active));self.refresh_library()
        if method=='ppo' and self.current_run:self.tabs.setCurrentIndex(1)
        self.poll()
        if method=='mppi':self.log_toggle.setChecked(True)

    def number(self,form,label,key,low,high,step,integer=False):
        widget=QSpinBox() if integer else QDoubleSpinBox()
        if not integer:widget.setDecimals(6 if key[-1] in ('learning_rate','entropy_coefficient','relative_improvement') else 4)
        widget.setRange(low,high);widget.setSingleStep(step);widget.setValue(self.cfg[key[0]][key[1]])
        self.fields[key]=widget;form.addRow(label,widget)

    def update_success_controls(self):
        simple=self.success_rule.currentData()=='tip_contact_v1'
        self.success_note.setText('Success: the tip reaches the target. Speed, reversal and bend statistics do not gate a hit. Existing feasibility checks still apply.'
            if simple else 'Historical task: saved speed, direction, first-contact and enabled strategy gates all apply.')
        for widget in self.gate_controls:self.strike_form.setRowVisible(widget,not simple)

    def quick_setup(self):
        for key,value in dict(samples=256,initial_minimum_iterations=8,initial_patience=4,minimum_iterations=3,patience=2).items():self.fields[('mppi',key)].setValue(value)
        self.setup_note.setText('Quick setup staged for the next run: 256 samples, 8 initial / 3 later minimum iterations, shorter plateau patience. Horizon, physics and running jobs are unchanged.')

    def add_vector(self,layout,title,section,key,low,high,step,unit):
        group=QGroupBox(title+' ['+unit+']');row=QHBoxLayout(group);fields=[]
        for i,axis in enumerate('XYZ'):
            row.addWidget(QLabel(axis));spin=QDoubleSpinBox();spin.setDecimals(2 if section=='action' else 4);spin.setRange(low,high);spin.setSingleStep(step);spin.setValue(self.cfg[section][key][i]);spin.setFixedWidth(100);row.addWidget(spin);fields.append(spin)
        self.fields[(section,key)]=fields;layout.addWidget(group)

    def collect(self):
        cfg=deepcopy(self.cfg)
        cfg['task']['success_criterion']=self.success_rule.currentData()
        for (section,key),widget in self.fields.items():
            cfg[section][key]=[w.value() for w in widget] if isinstance(widget,list) else widget.isChecked() if isinstance(widget,QCheckBox) else widget.value()
        cfg['task']['first_contact_only']=self.first_contact.isChecked()
        if hasattr(self,'pullback'):
            cfg['task']['require_pullback']=self.pullback.isChecked()
            cfg['task']['require_wave']=self.wave.isChecked()
        if self.method=='mppi':
            cfg['mppi']['initialization']=self.initialization.currentData()
            cfg.setdefault('visualization',{})['live_mppi']=self.live_enabled.isChecked()
        if self.method=='mppi' and cfg['mppi'].get('mode')=='receding':cfg['mppi']['horizon_s']=self.horizon.value()/30
        else:cfg['task']['duration_s']=self.horizon.value()/30
        cfg['model_path']=self.models.currentData() or self.cfg['model_path'];validate_settings(cfg);return cfg

    def save_settings(self):
        try:self.cfg=self.collect();atomic_json(settings_path(self.root,self.method),self.cfg);self.setup_note.setText('Saved '+self.method.upper()+' setup. Existing runs and the other method are unchanged.');self.changed.emit()
        except (ValueError,OSError) as e:self.setup_note.setText(str(e))

    def refresh_models(self):
        selected=self.models.currentData() or self.cfg['model_path']
        if selected:
            p=Path(selected);selected=str(p if p.is_absolute() else self.root/p)
        self.models.blockSignals(True);self.models.clear()
        for p in model_paths(self.root):self.models.addItem(model_label(p),str(p.resolve()))
        index=self.models.findData(selected)
        if index<0:self.models.addItem('Selected bootstrap model · fitting / unavailable' if selected else 'No fitted model selected',selected);index=self.models.count()-1
        self.models.setCurrentIndex(index);self.models.blockSignals(False);self.describe_model()

    def describe_model(self):
        path=self.models.currentData();m=read_json(path,{}) if path else {}
        self.run.setEnabled(bool(m) and m.get('provenance',{}).get('fit_complete') is not False and not (hasattr(self,'job') and self.job.running))
        provenance=m.get('provenance',{});mass=m.get('mass_measurement',{})
        self.model_note.setText((f'{provenance.get("label",self.models.currentText())}\nDrone {mass.get("drone_mass_kg",0)*1000:g} g · cable {mass.get("cable_assembly_mass_kg",0)*1000:g} g\n'+
            (provenance.get('quality_note','Fitted model; next flights provide prospective evidence.') if provenance.get('fit_complete') else 'Unfinished or historical model; inspect its original provenance.')) if m else 'No fitted model available. Collect and review the new preliminary takes before fitting M0 and planning.')

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
            if self.method=='mppi':
                task=read_json(p/'settings.json',{}).get('task',{})
                values.append('Tip reaches target' if task.get('success_criterion')=='tip_contact_v1' else
                    'Legacy: travelling bend + pullback' if task.get('require_wave') else 'Legacy: forward/backward whip' if task.get('require_pullback') else 'Legacy: directed tip hit')
            for j,v in enumerate(values):self.library.setItem(i,j,QTableWidgetItem(v))
        if self.runs:
            self.library.selectRow(self.runs.index(selected) if selected in self.runs else 0)
            if self.current_run is None:self.current_run=self.runs[0]
        if hasattr(self,'progress_runs'):
            self.progress_runs.blockSignals(True);self.progress_runs.clear()
            for p in self.runs:
                identity=read_json(p/'identity.json',{})
                self.progress_runs.addItem(identity.get('name',p.name)+' · '+p.name,str(p))
            self.progress_runs.setCurrentIndex(self.progress_runs.findData(str(self.current_run)))
            self.progress_runs.blockSignals(False)

    def select_progress_run(self):
        path=self.progress_runs.currentData()
        if not path:return
        self.current_run=Path(path);self.last_history=None
        if self.current_run in self.runs:self.library.selectRow(self.runs.index(self.current_run))
        self.poll()

    def open_progress_files(self):
        if self.current_run:QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_run)))

    def toggle_progress_log(self,visible):
        self.saved_log.setVisible(visible);self.poll()

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
        if self.method=='mppi':self.live_view.set_job(p)
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
        if self.method=='ppo':self.preview_button.setEnabled(bool(p) and not self.rehearsal_job.running)
        if not p:return
        self.progress_runs.blockSignals(True);self.progress_runs.setCurrentIndex(self.progress_runs.findData(str(p)));self.progress_runs.blockSignals(False)
        status=read_json(p/'status.json',{});self.stop.setEnabled(status.get('status')=='running')
        if (p/'STOP').exists() and status.get('status')=='running':
            self.stop.setText('Stop requested');self.stop.setEnabled(False)
        else:self.stop.setText('Stop after current update')
        if self.log_toggle.isChecked():
            log=p/'console.log'
            if log.exists():
                with log.open('rb') as stream:
                    stream.seek(max(0,log.stat().st_size-14000));text=stream.read().decode('utf-8',errors='replace')
                if self.method=='mppi':
                    from .mppi_dashboard import readable_log
                    text=readable_log(text)
                if self.saved_log.toPlainText()!=text:
                    scroll=self.saved_log.verticalScrollBar();position=scroll.value();self.saved_log.setPlainText(text)
                    scroll.setValue(scroll.maximum() if self.follow_log.isChecked() else position)
            else:self.saved_log.setPlainText('No console log saved for this run.')
        receding=self.method=='mppi' and read_json(p/'settings.json',{}).get('mppi',{}).get('mode')=='receding'
        identity=read_json(p/'identity.json',{});amount=status.get('attempts',status.get('iteration',status.get('iterations','—')))
        self.progress_note.setText(f'{identity.get("name",p.name)} · {status.get("status","unknown")} · {amount} '+('attempts' if self.method=='ppo' else 'iterations')+'\n'+status.get('stage',status.get('error',status.get('stop_reason','')))+
            ('\nAutomatic stopping follows the orange deterministic evaluation reward.' if self.method=='ppo' else
             ('\nSaved best plan: '+('infeasible' if status['best_failed'] else 'modeled valid hit' if status.get('best_success') else 'no modeled valid hit')) if 'best_failed' in status else ''))
        if receding:
            clock=status.get('maneuver_time_s',status.get('time_s',status.get('maneuver_duration_s',0.)))
            self.progress_note.setText(self.progress_note.text()+f'\nSimulated maneuver: {clock:.3f} s · receding lookahead')
        if self.method=='ppo' and 'pull_fraction' in status:
            self.progress_note.setText(self.progress_note.text()+
                f'\nExploration stages: pull {100*status["pull_fraction"]:.1f}% · release {100*status["release_fraction"]:.1f}% · travelling bend {100*status["wave_complete_fraction"]:.1f}%')
        if self.method=='ppo' and read_json(p/'settings.json',{}).get('training',{}).get('success_priority'):
            self.progress_note.setText(self.progress_note.text().replace('Automatic stopping follows the orange deterministic evaluation reward.',
                'Best policy and plateau stopping prioritize valid hit rate, then evaluation reward.'))
        if self.method=='mppi':
            requirement=read_json(p/'settings.json',{}).get('task',{}).get('require_pullback',False)
            self.progress_note.setText(self.progress_note.text()+('\nTask requires forward pull, then backward drone motion at tip contact.' if requirement else '\nSaved tip-hit task: backward release was not required.'))
        path=p/('windows.json' if receding else 'history.json')
        history_key=(str(p),path.stat().st_mtime_ns if path.exists() else None)
        if getattr(self,'history_cache_key',None)!=history_key:
            self.history_cache_key=history_key;self.history_rows=read_json(path,[])
        rows=self.history_rows
        search_view=False
        if self.method=='mppi':
            history_path=p/'history.json';key=(str(p),history_path.stat().st_mtime_ns if history_path.exists() else None)
            if getattr(self,'mppi_history_key',None)!=key:self.mppi_history_key=key;self.mppi_history=read_json(history_path,[])
            modified=max((f.stat().st_mtime for f in (p/'status.json',history_path) if f.exists()),default=time.time())
            self.mppi_dashboard.update_run(status,read_json(p/'settings.json',{}),self.mppi_history,rows if receding else [],max(0,time.time()-modified))
            search_view=receding and self.mppi_chart.currentIndex()==0
            if search_view:
                step=status.get('command_step',status.get('command_steps',0))
                rows=[r for r in self.mppi_history if r.get('command_step')==step]
                if not rows and status.get('status')!='running':rows=self.mppi_history
                history_key=(key,step,'search')
        if self.method=='ppo':self.dashboard.update_run(status,read_json(p/'settings.json',{}),rows)
        if self.last_history==history_key:return
        self.last_history=history_key
        if not rows:
            self.figure.clear()
            if self.method=='mppi':
                ax=self.figure.subplots();ax.axis('off');ax.text(.5,.5,'Waiting for the first iteration result.' if search_view else 'Committed-path trends appear after the first command.',ha='center',va='center',wrap=True)
            self.canvas.draw_idle();return
        self.figure.clear();axes=self.figure.subplots(2,2).ravel() if self.method=='ppo' else self.figure.subplots(1,2).ravel();x=[r.get('attempts',r.get('iteration')) for r in rows]
        keys=['reward','success','minimum_tip_distance_m','failures'] if self.method=='ppo' else ['best_reward','success','best_minimum_tip_distance_m','effective_samples']
        titles=['Episode return (batch mean)','Modeled valid hits','Closest tip distance','Infeasible attempts'] if self.method=='ppo' else ['Saved best return','Modeled valid hits (sampled batch)','Saved best tip distance','Effective importance samples']
        if self.method=='mppi':
            keys=['best_reward','best_minimum_tip_distance_m'];titles=['Best return','Closest tip distance']
        if receding and not search_view:
            x=[r['time_s'] for r in rows]
            keys=['actual_reward','actual_minimum_tip_distance_m'];titles=['Committed return','Committed closest tip distance']
        if search_view:
            x=[r['iteration'] for r in rows];keys=['lookahead_score','predicted_distance_m'];titles=['Best lookahead score','Best proposal closest tip distance']
        for ax,key,title in zip(axes,keys,titles):
            factor=100 if key in ('success','failures') else 1;ax.plot(x,[r.get(key,0)*factor for r in rows],color='#2563eb',lw=1.6,label='Training batch' if self.method=='ppo' else None)
            ax.set_title(title,loc='left',fontsize=10);ax.set_xlabel('Simulated maneuver time [s]' if receding and not search_view else 'Attempts' if self.method=='ppo' else 'Iterations');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
            if key=='actual_success':ax.set_ylim(0,1);ax.set_yticks([0,1],['No','Yes'])
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
        if self.method=='ppo':
            self.launch_policy_rehearsal(p,self.checkpoint.currentText(),True);return
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

    def rehearse_current(self):
        if self.current_run:self.launch_policy_rehearsal(self.current_run,self.preview_checkpoint.currentData(),self.preview_complete.isChecked())

    def refresh_policy_rehearsal(self):
        if self.preview_context:self.launch_policy_rehearsal(*self.preview_context)

    def restore_preview_context(self):
        snapshot=self.inspector.metadata.get('policy_snapshot',{})
        self.preview_context=((Path(snapshot['source_run']),snapshot['checkpoint_choice'],not self.inspector.metadata.get('preview_only',False))
            if snapshot.get('source_run') and snapshot.get('checkpoint_choice') in ('latest.pt','best.pt') else None)
        self.refresh_preview.setEnabled(self.method=='ppo' and self.preview_context is not None and not self.rehearsal_job.running)

    def launch_policy_rehearsal(self,p,choice,complete=False):
        if self.rehearsal_job.running:return
        from deployment.ppo_snapshot import prepare_snapshot
        work=self.root/'runs/pva_jobs'/stamp()
        try:provenance=prepare_snapshot(self.root,p,work/'snapshot',choice)
        except (OSError,ValueError,KeyError,RuntimeError) as e:
            message='Could not snapshot policy: '+str(e)
            self.policy_preview_note.setText(message);self.library_note.setText(message);self.rehearsal_note.setText(message);return
        self.preview_context=(Path(p),choice,complete)
        self.rehearsal_output=self.root/'runs/rehearsals_pva'/(work.name+('-ppo' if complete else '-ppo-preview'))
        command=[str(self.root/'.venv/Scripts/python.exe'),'-u',str(work/'snapshot/source_snapshot/tools/rehearse_ppo_snapshot.py'),
            '--job',str(work/'snapshot'),'--output',str(self.rehearsal_output)]
        if complete:command.append('--complete')
        message=f'{provenance["source_run_name"]} · {choice} · {provenance["checkpoint_attempts"]:,} saved attempts · generating…'
        self.policy_preview_note.setText(message);self.rehearsal_note.setText(message)
        self.refresh_preview.setEnabled(False);self.preview_button.setEnabled(False)
        self.inspector.clear_result();self.rehearsal_job.start(work,command);self.tabs.setCurrentIndex(3)

    def rehearsal_finished(self,code):
        self.refresh_preview.setEnabled(self.preview_context is not None)
        if self.method=='ppo':self.preview_button.setEnabled(self.current_run is not None)
        if code:self.rehearsal_note.setText('Generation failed. Open the log; no new command has been exported.');return
        try:
            self.inspector.load_result(self.rehearsal_output);self.restore_preview_context();m=self.inspector.metadata;s=m.get('policy_snapshot')
            message=(f'{s["checkpoint_choice"]} · {s["checkpoint_attempts"]:,} saved attempts · {s["checkpoint_sha256"][:8]} · ' if s else '')
            message+=m['outcome'] if m.get('preview_only') else 'Complete command is ready for inspection.'
            self.rehearsal_note.setText(message)
            if self.method=='ppo':self.policy_preview_note.setText(message)
            self.changed.emit()
        except (OSError,ValueError,KeyError) as e:self.rehearsal_note.setText(str(e))

    def open_saved(self):
        p=QFileDialog.getExistingDirectory(self,'Open saved PVA rehearsal',str(self.root/'runs/rehearsals_pva'))
        if p:
            try:self.inspector.clear_result();self.inspector.load_result(p);self.restore_preview_context()
            except (OSError,ValueError,KeyError) as e:self.rehearsal_note.setText(str(e))

    def set_page_active(self,active):
        self.active=active;self.inspector.set_page_active(active and self.tabs.currentIndex()==3)
        if self.method=='mppi':self.live_view.set_active(active and self.tabs.currentWidget() is self.live_view)

    def shutdown(self):
        self.timer.stop()
        if self.method=='mppi':self.live_view.shutdown()
        return self.inspector.shutdown()
