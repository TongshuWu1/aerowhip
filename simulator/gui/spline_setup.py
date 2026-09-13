"""New-study MPPI setup; historical saved plans retain the common library."""
from copy import deepcopy
from pathlib import Path
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QGroupBox,
    QLabel,QPushButton,QComboBox,QLineEdit,QTabWidget,QScrollArea,QCheckBox,QDoubleSpinBox)
from .research_widgets import note
from planning.pva_job import validate_settings


def build(page):
    if page.cfg['trajectory_objective'].get('schema')=='preferred_fold_v1':return build_retained(page)
    objective=page.cfg['trajectory_objective']
    objective.setdefault('speed_proximity_scale_m',objective['distance_scale_m'])
    objective.setdefault('maximum_strike_angle_deg',90.)
    page.active=False;page.current_run=None;page.resume_checkpoint=None;page.fields={}
    page.last_history=None;page.library_fingerprint=None
    outer=QVBoxLayout(page);outer.setContentsMargins(18,12,18,14)
    page.tabs=QTabWidget();outer.addWidget(page.tabs)
    setup=QWidget();page.tabs.addTab(setup,'Spline setup');body=QVBoxLayout(setup)
    row=QHBoxLayout();row.addWidget(QLabel('Run name'));page.run_name=QLineEdit('Spline targeted strike');row.addWidget(page.run_name,1)
    page.run=QPushButton('Optimize new spline');page.run.setObjectName('primaryButton');page.run.clicked.connect(page.start_run);row.addWidget(page.run);body.addLayout(row)
    row=QHBoxLayout();row.addWidget(QLabel('Fitted model'));page.models=QComboBox();row.addWidget(page.models,1)
    refresh=QPushButton('Refresh models');refresh.clicked.connect(page.refresh_models);row.addWidget(refresh);body.addLayout(row)
    page.contract=note('12 position control points: 3 fixed at launch, 9 adjustable in XYZ. Smooth quintic spline → 30 Hz PVA → predicted quadrotor → DDER cable.')
    if page.cfg['mppi'].get('initialization')=='from_scratch':page.contract.setText(page.contract.text()+' Starts from rest; no saved strike templates.')
    page.contract.setObjectName('pipelineBanner');body.addWidget(page.contract)
    tabs=QTabWidget();body.addWidget(tabs,1)
    for title in ('Motion and target','Strike objective','Recovery','Search and limits'):
        content=QWidget();layout=QVBoxLayout(content);scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(content);tabs.addTab(scroll,title)
        if title=='Motion and target':
            page.add_vector(layout,'Launch tracked origin','launch','origin_m',-10,10,.01,'m')
            page.add_vector(layout,'Target','launch','target_m',-10,10,.01,'m')
            page.add_vector(layout,'Strike direction','task','strike_direction',-1,1,.1,'unit vector')
            group=QGroupBox('Whip interval');form=QFormLayout(group)
            page.number(form,'Duration [s, multiples of 1/30]',('task','duration_s'),.1,5.,1/30);layout.addWidget(group)
            layout.addWidget(note('The whip ends without a forced stop. Recovery starts from its final PVA. All three axes remain adjustable; lateral drift has a soft cost.'))
        elif title=='Strike objective':
            layout.addWidget(note('Optimize target distance and forward tip speed at the same event. Fold propagation is recorded as a diagnostic; it does not reject a plan. No binary hit-radius threshold.'))
            if objective.get('speed_metric')=='tip_gain_over_root':layout.addWidget(note('Speed credit counts only the tip forward speed beyond the root forward speed. Carrying them together earns zero speed bonus; backward root motion cannot inflate it.'))
            group=QGroupBox('Strike and motion costs');form=QFormLayout(group)
            labels=dict(distance_scale_m='Distance scale [m]',speed_scale_m_s='Speed scale [m/s]',speed_proximity_scale_m='Speed bonus distance scale [m]',intensity_weight='Strike intensity weight',jerk_weight='Jerk magnitude weight',lateral_weight='Lateral drift weight',lateral_scale_m='Lateral distance scale [m]')
            for key,label in labels.items():page.number(form,label,('trajectory_objective',key),.0001 if key.endswith(('_m','_m_s')) else 0,100,.01)
            if 'minimum_tip_speed_gain_m_s' in objective:page.number(form,'Required tip speed gain [m/s]',('trajectory_objective','minimum_tip_speed_gain_m_s'),.01,20.,.1)
            page.number(form,'Maximum tip-velocity deviation [deg]',('trajectory_objective','maximum_strike_angle_deg'),.1,90.,1.)
            layout.addWidget(note('The angle limits tip velocity relative to the strike direction at the scored event. 30 degrees means a 60-degree full cone; it is not a quadrotor tilt limit.'))
            page.aligned_strike=QCheckBox('Prefer straighter strikes within the angle limit')
            page.aligned_strike.setChecked(objective.get('prefer_aligned_strike',False))
            form.addRow(page.aligned_strike)
            layout.addWidget(group);layout.addWidget(note('When enabled, speed credit is strongest on-axis and falls to zero at the angle limit. Distance remains a separate penalty.'))
            layout.addWidget(note('Distance penalty and speed bonus have separate distance scales; neither is a hit threshold. Fold diagnostics use the saved geometry settings. High tip speed alone does not establish whipping; inspect deformation along the cable and confirm it in recorded experiments.'))
        elif title=='Recovery':
            layout.addWidget(note('1. Brake smoothly to rest.  2. Return to launch.  3. Hold.\nThe stopping position follows from the exit motion and braking duration. The complete predicted attitude and cable motion must also pass export checks.'))
            group=QGroupBox('Brake, return and hold');form=QFormLayout(group)
            labels=dict(minimum_brake_s='Minimum braking time [s]',maximum_brake_s='Maximum braking time [s]',minimum_return_s='Minimum return time [s]',maximum_return_s='Maximum return time [s]',hold_s='Final hold [s]',braking_horizontal_acceleration_m_s2='Braking horizontal acceleration [m/s²]',braking_downward_acceleration_m_s2='Braking downward acceleration [m/s²]',braking_tilt_deg='Preferred braking tilt limit [deg]',return_speed_m_s='Return speed limit [m/s]')
            for key,label in labels.items():page.number(form,label,('recovery',key),.01,89 if key.endswith('deg') else 120,.1)
            layout.addWidget(group);layout.addWidget(note('Existing exit acceleration is preserved during handover. The saved flight envelope always applies.'))
        else:
            group=QGroupBox('Offline search');form=QFormLayout(group)
            for key,label in [('samples','Random candidates per iteration'),('iterations','Fixed iteration budget'),('seed','Random seed')]:page.number(form,label,('mppi',key),0 if key=='seed' else 1,1000000,1,integer=True)
            if page.cfg['mppi'].get('initialization')=='from_scratch':
                page.number(form,'Fresh candidates fraction',('mppi','fresh_sample_fraction'),0.,1.,.05)
                page.exploration_scales=[]
                for i,value in enumerate(page.cfg['mppi']['position_noise_scales_m']):
                    box=QDoubleSpinBox();box.setRange(.001,5.);box.setDecimals(3);box.setSingleStep(.05);box.setValue(value)
                    form.addRow(f'Smooth position exploration scale {i+1} [m]',box);page.exploration_scales.append(box)
            layout.addWidget(group)
            page.add_vector(layout,'Continuous spline / recovery jerk bounds','action','jerk_limit_m_s3',1,300,1,'m/s³')
            group=QGroupBox('Reference and prediction limits');form=QFormLayout(group)
            for key,value in page.cfg['limits'].items():page.number(form,key.replace('_',' '),('limits',key),.001,89 if key=='maximum_tilt_deg' else 100,.1)
            layout.addWidget(group)
        layout.addStretch()
    page.live_enabled=QCheckBox('Record live candidate previews');page.live_enabled.setChecked(page.cfg.get('visualization',{}).get('live_mppi',True));body.addWidget(page.live_enabled)
    page.model_note=note('');body.addWidget(page.model_note)
    save=QPushButton('Save new-study spline setup');save.clicked.connect(page.save_settings);body.addWidget(save)
    page.setup_note=note('New runs use this setup. Saved plans keep their original representation and recovery.');body.addWidget(page.setup_note)
    page.models.currentIndexChanged.connect(page.describe_model);page.refresh_models()


def collect(page):
    cfg=deepcopy(page.cfg)
    for (section,key),widget in page.fields.items():
        cfg[section][key]=[w.value() for w in widget] if isinstance(widget,list) else widget.value()
    if hasattr(page,'aligned_strike'):cfg['trajectory_objective']['prefer_aligned_strike']=page.aligned_strike.isChecked()
    cfg['task']['duration_s']=round(cfg['task']['duration_s']*30)/30
    cfg['visualization']['live_mppi']=page.live_enabled.isChecked()
    cfg['model_path']=page.models.currentData() or cfg['model_path']
    validate_settings(cfg)
    if cfg.get('spline_seed_directory'):
        for name in ('initial_proposal.npz','proposal_baselines.npz',cfg['trajectory_objective']['reference_file']):
            if not (page.root/cfg['spline_seed_directory']/name).is_file():raise ValueError('Saved M0 initialization asset missing: '+name)
        return cfg
    if cfg['mppi'].get('initialization')=='from_scratch':
        cfg['mppi']['position_noise_scales_m']=[box.value() for box in page.exploration_scales]
        validate_settings(cfg)
    else:
        template=Path(cfg['proposal_templates_path']);template=template if template.is_absolute() else page.root/template
        if not template.is_file():raise ValueError('Command initialization templates are unavailable')
    return cfg



def build_retained(page):
    page.active=False;page.current_run=None;page.resume_checkpoint=None;page.fields={}
    page.last_history=None;page.library_fingerprint=None
    outer=QVBoxLayout(page);page.tabs=QTabWidget();outer.addWidget(page.tabs)
    setup=QWidget();page.tabs.addTab(setup,'M0 B-spline setup');body=QVBoxLayout(setup)
    row=QHBoxLayout();row.addWidget(QLabel('Run name'));page.run_name=QLineEdit('M0 B-spline refinement');row.addWidget(page.run_name,1)
    page.run=QPushButton('Optimize M0 spline');page.run.setObjectName('primaryButton');page.run.clicked.connect(page.start_run);row.addWidget(page.run);body.addLayout(row)
    row=QHBoxLayout();row.addWidget(QLabel('Fitted model'));page.models=QComboBox();row.addWidget(page.models,1);body.addLayout(row)
    page.contract=note('Selected M0 motion initializes a quintic position B-spline: 3 fixed launch points and 9 adjustable XYZ points. Analytic P/V/A commands at 30 Hz. Original M0 objective; smooth braking, return and hover.')
    page.contract.setObjectName('pipelineBanner');body.addWidget(page.contract)
    tabs=QTabWidget();body.addWidget(tabs,1)
    for title in ('Motion','Objective','Recovery','Search'):
        content=QWidget();layout=QVBoxLayout(content);scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(content);tabs.addTab(scroll,title)
        if title=='Motion':
            page.add_vector(layout,'Launch tracked origin','launch','origin_m',-10,10,.01,'m')
            page.add_vector(layout,'Target','launch','target_m',-10,10,.01,'m')
            layout.addWidget(note('Original M0 launch and target. The motion ends at the first tip contact or the 1.5 s limit, then brakes smoothly, returns to launch and holds.'))
        elif title=='Objective':
            layout.addWidget(note('Straight forward strikes receive more cast reward through cos(angle)^2. There is no strike-angle cutoff. The original cable-shape preference, contact/reversal reward, and motion costs remain active.'))
            group=QGroupBox('Original M0 objective weights');form=QFormLayout(group)
            for key,label in [('cast','Straight strike reward'),('fold','Cable-shape preference'),('contact','Contact and reversal reward'),('miss','Miss cost'),('approach','Forward drone excursion cost'),('lateral','Lateral drift cost'),('jerk','Jerk cost')]:
                page.number(form,label,('trajectory_objective',key),0.,5000.,.1)
            layout.addWidget(group)
            layout.addWidget(note('Cast reward = weight x target proximity x forward tip-speed factor x cable reach x cos(angle)^2. Angle is measured from the saved world strike direction at the same cable-target encounter.'))
        elif title=='Recovery':
            layout.addWidget(note('Brake continuously from the whip exit to rest, then return along a smooth straight path to launch and hover. No forced reset of position, velocity or acceleration.'))
            group=QGroupBox('Smooth brake and return');form=QFormLayout(group)
            for key,label in [('minimum_brake_s','Minimum braking time [s]'),('maximum_brake_s','Maximum braking time [s]'),('minimum_return_s','Minimum return time [s]'),('maximum_return_s','Maximum return time [s]'),('return_speed_m_s','Return speed [m/s]'),('hold_s','Final hover [s]')]:
                page.number(form,label,('recovery',key),.01,120.,.1)
            layout.addWidget(group)
        else:
            group=QGroupBox('MPPI refinement');form=QFormLayout(group)
            for key,label,low in [('samples','Candidates per update',2),('minimum_iterations','Minimum updates',1),('patience','Plateau patience',1),('iterations','Update ceiling (0 = plateau stopping)',0),('seed','Random seed',0)]:
                page.number(form,label,('mppi',key),low,1000000,1,integer=True)
            layout.addWidget(group)
            page.add_vector(layout,'Jerk limits','action','jerk_limit_m_s3',1,300,1,'m/s^3')
            layout.addWidget(note('Search starts around the converted selected M0 motion. It does not resume any later optimization.'))
        layout.addStretch()
    page.live_enabled=QCheckBox('Record live candidate previews');page.live_enabled.setChecked(True);body.addWidget(page.live_enabled)
    page.model_note=note('');body.addWidget(page.model_note)
    save=QPushButton('Save M0 spline setup');save.clicked.connect(page.save_settings);body.addWidget(save)
    page.setup_note=note('The selected original M0 rehearsal remains unchanged. A new spline plan must be rehearsed before export.');body.addWidget(page.setup_note)
    page.models.currentIndexChanged.connect(page.describe_model);page.refresh_models()
