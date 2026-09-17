"""MPPI setup, progress, saved plans, and command rehearsal."""
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
from planning.pva_job import load_settings,settings_path,prepare,validate_settings,configured_development_review
from .research_widgets import note,BackgroundJob
from .rehearsal_workspace import RehearsalWorkspace


def model_paths(root):
    root=Path(root).resolve();paths=[]
    def active(path):
        path=Path(path).resolve()
        if not path.is_relative_to(root):return False
        return not any((parent/'ARCHIVED').exists() for parent in path.parents if parent.is_relative_to(root))
    # Registered retained models remain available even when their original fit
    # directory is not the canonical packaged model (for example retained M0).
    for item in read_json(root/'config/evaluation/campaign.json',{}).get('models',[]):
        path=Path(item['model']);path=path if path.is_absolute() else root/path
        if path.is_file() and active(path):paths.append(path)
    for p in sorted((root/'runs/adaptation').glob('*/candidate/model.json'),reverse=True):
        if active(p) and read_json(p,{}).get('provenance',{}).get('fit_complete'):paths.append(p)
    for p in sorted((root/'data/model_candidates').glob('*/model.json'),reverse=True):
        if active(p):paths.append(p)
    return list(dict.fromkeys(p.resolve() for p in paths))


def model_label(path):
    path=Path(path).resolve();model=read_json(path,{})
    label=model.get('provenance',{}).get('label',path.parent.name)
    # Registered renaming is authoritative for display; frozen model provenance
    # remains intact so old forecasts and their hashes stay reproducible.
    for root in path.parents:
        catalog=root/'config/evaluation/campaign.json'
        if not catalog.is_file():continue
        for entry in read_json(catalog,{}).get('models',[]):
            source=Path(entry['model']);source=source if source.is_absolute() else root/source
            previous=entry.get('renaming',{}).get('previous_id')
            if source.resolve()==path and previous and (label==previous or label.startswith(previous+' ')):
                return entry['id']+label[len(previous):]
        break
    return label


class PVAPlannerPage(QWidget):
    changed = Signal()

    def __init__(self, root, method):
        if method != 'mppi':
            raise ValueError('Only the MPPI planner is supported')
        super().__init__()
        self.root = Path(root)
        self.method = 'mppi'
        self.cfg = load_settings(root, 'mppi')
        staged = read_json(self.root / 'config/pva/systematic_strike.json', {})
        if staged.get('command_contract') == 'position_spline_pva_30hz_v1':
            self.cfg = staged
        self.spline_setup = self.cfg.get('command_contract') == 'position_spline_pva_30hz_v1'
        if self.spline_setup:
            from .spline_setup import build
            build(self)
        else:
            for key in ('vertical_excursion', 'vertical_tip_velocity', 'vertical_tip_alignment', 'horizontal_contact', 'drone_approach', 'lateral_excursion', 'near_target_reach', 'outward_contact'):
                self.cfg['reward'].setdefault(key, 0.0)
            self.active = False
            self.current_run = None
            self.fields = {}
            self.last_history = None
            self.library_fingerprint = None
            outer = QVBoxLayout(self)
            outer.setContentsMargins(18, 12, 18, 14)
            self.tabs = QTabWidget()
            outer.addWidget(self.tabs)
            setup = QWidget()
            self.tabs.addTab(setup, 'Setup and run')
            body = QVBoxLayout(setup)
            row = QHBoxLayout()
            row.addWidget(QLabel('Run name'))
            self.run_name = QLineEdit('mppi'.upper() + ' · PVA whip')
            row.addWidget(self.run_name, 1)
            self.run = QPushButton('Optimize new plan')
            self.run.setObjectName('primaryButton')
            self.run.clicked.connect(self.start_run)
            row.addWidget(self.run)
            body.addLayout(row)
            row = QHBoxLayout()
            row.addWidget(QLabel('Fitted model'))
            self.models = QComboBox()
            row.addWidget(self.models, 1)
            refresh = QPushButton('Refresh models')
            refresh.clicked.connect(self.refresh_models)
            row.addWidget(refresh)
            body.addLayout(row)
            self.contract = note('Desired P/V/A → loaded-drone response + NN → rotated cable attachment → DDER + NN. 30 Hz commands; the complete sequence is planned before flight.')
            self.contract.setObjectName('pipelineBanner')
            body.addWidget(self.contract)
            split = QSplitter()
            body.addWidget(split, 1)
            left = QWidget()
            left_body = QVBoxLayout(left)
            left_body.setContentsMargins(0, 0, 8, 0)
            self.add_vector(left_body, 'Launch · OptiTrack tracked origin', 'launch', 'origin_m', -10, 10, 0.01, 'm')
            self.add_vector(left_body, 'Target position', 'launch', 'target_m', -10, 10, 0.01, 'm')
            group = QGroupBox('Maneuver and strike')
            form = QFormLayout(group)
            from learning.pva_success import criterion, TIP_CONTACT, LEGACY
            self.strike_form = form
            self.success_rule = QComboBox()
            self.success_rule.addItem('Tip reaches target', TIP_CONTACT)
            self.success_rule.addItem('Legacy speed / strategy / wave gates', LEGACY)
            self.success_rule.setCurrentIndex(self.success_rule.findData(criterion(self.cfg['task'])))
            form.addRow('Success condition', self.success_rule)
            self.success_note = note('')
            form.addRow(self.success_note)
            receding = self.cfg['mppi'].get('mode') == 'receding'
            self.horizon = QSpinBox()
            self.horizon.setRange(3, 150)
            self.horizon.setValue(round((self.cfg['mppi']['horizon_s'] if receding else self.cfg['task']['duration_s']) * 30))
            form.addRow('MPPI lookahead intervals (30 Hz)' if receding else '30 Hz command intervals', self.horizon)
            self.duration = note('')
            form.addRow('Lookahead' if receding else 'Duration', self.duration)
            self.horizon.valueChanged.connect(lambda n: self.duration.setText(f'{n / 30:.3f} s'))
            self.duration.setText(f'{self.horizon.value() / 30:.3f} s')
            if receding:
                self.number(form, 'Maneuver safety limit [s]', ('task', 'duration_s'), 0.1, 30.0, 1.0)
            self.number(form, 'Hit radius [m]', ('task', 'target_radius_m'), 0.005, 0.3, 0.005)
            self.number(form, 'Minimum directed tip speed [m/s]', ('task', 'minimum_directed_speed_m_s'), 0.1, 20, 0.25)
            self.number(form, 'Maximum strike angle [deg]', ('task', 'maximum_angle_deg'), 1, 90, 1)
            self.first_contact = QCheckBox('Only the first target entry can count')
            self.first_contact.setChecked(self.cfg['task']['first_contact_only'])
            form.addRow(self.first_contact)
            self.pullback = QCheckBox('Require forward pull, then backward drone motion at contact')
            self.pullback.setChecked(self.cfg['task'].get('require_pullback', False))
            form.addRow(self.pullback)
            from learning.whip_wave import DEFAULTS
            self.wave = QCheckBox('Require a travelling bend before tip contact')
            self.wave.setChecked(self.cfg['task'].get('require_wave', False))
            form.addRow(self.wave)
            for key, value in DEFAULTS.items():
                self.cfg['task'].setdefault(key, value)
                self.number(form, key.replace('wave_', 'Bend ').replace('_', ' '), ('task', key), 0.001, 3.14, 0.01)
            for key, label, value in (('minimum_pull_distance_m', 'Forward pull distance [m]', 0.25), ('minimum_pull_speed_m_s', 'Forward pull speed [m/s]', 1.0), ('minimum_backward_distance_m', 'Backward travel before contact [m]', 0.1), ('minimum_backward_speed_m_s', 'Backward drone speed at contact [m/s]', 0.5)):
                self.cfg['task'].setdefault(key, value)
                self.number(form, label, ('task', key), 0.01, 5.0, 0.05)
            self.gate_controls = [widget for (section, key), widget in self.fields.items() if section == 'task' and key not in ('target_radius_m', 'duration_s')]
            self.gate_controls += [self.first_contact] + [getattr(self, key) for key in ('pullback', 'wave') if hasattr(self, key)]
            self.success_rule.currentIndexChanged.connect(self.update_success_controls)
            self.update_success_controls()
            left_body.addWidget(group)
            self.add_vector(left_body, 'Desired strike direction', 'task', 'strike_direction', -1, 1, 0.1, 'unit vector')
            left_body.addStretch()
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(left)
            split.addWidget(scroll)
            right = QWidget()
            rb = QVBoxLayout(right)
            rb.setContentsMargins(8, 0, 0, 0)
            controls = QTabWidget()
            rb.addWidget(controls)
            reward_section = 'reward'
            for title, section in [('Rewards', reward_section), ('PVA limits', 'limits'), ('MPPI sampling', 'mppi')]:
                content = QWidget()
                v = QVBoxLayout(content)
                group = QGroupBox(title)
                form = QFormLayout(group)
                v.addWidget(group)
                labels = {'progress': 'Closest-distance progress', 'strike_quality': 'Near-target directed-speed progress', 'success': 'Valid hit bonus', 'time_per_s': 'Time cost [reward / s]', 'failure': 'Infeasible / numerical failure cost', 'invalid_contact': 'Invalid first-contact cost', 'early_hit': 'Earlier successful hit · maximum bonus', 'early_hit_scale_s': 'Earlier-hit decay time [s]', 'impact': 'Harder tip hit · maximum bonus', 'impact_scale_m_s': 'Impact speed scale [m/s]', 'displacement': 'Drone displacement cost [reward / m²·s]', 'jerk': 'Normalized jerk cost [reward / s]', 'proximity_scale_m': 'Speed-shaping distance [m]', 'pull_phase': 'Forward-pull progress reward', 'reverse_phase': 'Backward-release progress reward', 'wave_progress': 'Travelling-bend progress reward', 'reach_progress': 'Outward release progress reward (once per episode)', 'brake_progress': 'Braking after pull progress reward', 'command_acceleration': 'Specific acceleration cost [reward / s at limit]', 'command_speed': 'Commanded speed cost [reward / s at speed limit]', 'vertical_excursion': 'Vertical drone excursion cost [reward / m²·s]', 'vertical_tip_velocity': 'Near-strike vertical tip velocity cost', 'vertical_tip_alignment': 'Near-strike vertical cable alignment cost', 'horizontal_contact': 'Horizontal contact quality bonus', 'drone_approach': 'Forward drone approach cost [reward / m²·s]', 'lateral_excursion': 'Sideways drone excursion cost [reward / m²·s]', 'near_target_reach': 'Near-target cable reach shortfall cost', 'outward_contact': 'Outward cable reach at contact bonus', 'minimum_origin_z_m': 'Minimum drone height [m]', 'maximum_origin_z_m': 'Maximum drone height [m]', 'minimum_cable_z_m': 'Minimum cable height [m]', 'maximum_specific_force_m_s2': 'Maximum specific acceleration [m/s²]', 'minimum_specific_vertical_m_s2': 'Minimum upward specific acceleration [m/s²]', 'maximum_speed_m_s': 'Maximum commanded speed [m/s]', 'maximum_tilt_deg': 'Maximum reference tilt [deg]', 'seed': 'Random seed', 'samples': 'Parallel candidate trajectories', 'iterations': 'Iteration limit', 'temperature': 'MPPI temperature', 'noise_std': 'Latent action noise std', 'noise_correlation': 'Temporal noise correlation', 'minimum_iterations': 'Minimum iterations per lookahead', 'patience': 'Iterations without improvement', 'initial_minimum_iterations': 'First lookahead minimum iterations', 'initial_patience': 'First lookahead plateau patience', 'strike_exit_speed': 'Strike-exit speed cost', 'strike_exit_climb': 'Strike-exit upward speed cost', 'strike_exit_acceleration': 'Strike-exit forward acceleration cost', 'control_prior': 'Additional zero-jerk preference [0–1]', 'terminal_distance': 'Terminal tip-position guidance', 'terminal_velocity': 'Terminal strike-velocity guidance', 'terminal_anchor': 'Terminal drone-position guidance', 'terminal_command_speed': 'Terminal command-speed cost', 'terminal_command_acceleration': 'Terminal command-acceleration cost'}
                for key, value in self.cfg[section].items():
                    if section == 'trajectory_objective' and isinstance(value, str):
                        continue
                    if isinstance(value, bool):
                        checkbox = QCheckBox()
                        checkbox.setChecked(value)
                        self.fields[section, key] = checkbox
                        form.addRow(labels.get(key, key.replace('_', ' ').capitalize()), checkbox)
                        continue
                    if section == 'mppi' and key in ('mode', 'horizon_s', 'initialization'):
                        continue
                    if section == 'mppi' and key == 'noise_scales':
                        row = QHBoxLayout()
                        fields = []
                        for scale in value:
                            spin = QDoubleSpinBox()
                            spin.setDecimals(3)
                            spin.setRange(0.001, 3.0)
                            spin.setValue(scale)
                            fields.append(spin)
                            row.addWidget(spin)
                        self.fields[section, key] = fields
                        form.addRow('Mixed exploration noise scales', row)
                        continue
                    upper = 10000000 if isinstance(value, int) else 0.999 if key == 'noise_correlation' else 1.0 if key in ('control_prior', 'gae_lambda') else 10000
                    lower = -10 if key == 'initial_log_std' else 0 if key in ('seed', 'iterations') or isinstance(value, float) else 1
                    self.number(form, labels.get(key, key.replace('_', ' ').capitalize()), (section, key), lower, upper, 1 if isinstance(value, int) else 0.01, integer=isinstance(value, int))
                    if section == 'mppi' and key == 'noise_std' and self.cfg['mppi'].get('noise_scales'):
                        self.fields[section, key].setEnabled(False)
                        self.fields[section, key].setToolTip('The mixed exploration scales above replace this single-scale setting.')
                    if section == 'mppi' and key == 'iterations':
                        self.fields[section, key].setSpecialValueText('No limit')
                if section == 'mppi':
                    self.initialization = QComboBox()
                    self.initialization.addItem('Zero jerk', 'zero')
                    self.initialization.addItem('Forward/backward jerk guesses', 'pullback')
                    self.initialization.addItem('Varied whip pulse timing and lift', 'wave')
                    self.initialization.setCurrentIndex(self.initialization.findData(self.cfg['mppi'].get('initialization', 'zero')))
                    form.addRow('Initial proposal', self.initialization)
                    v.addWidget(note('Each lookahead optimizes until plateau or manual stop, then advances one 30 Hz command. The maneuver continues until a hit, failure or its separate safety limit. Terminal guidance is separate from actual strike reward.'))
                if section == 'limits':
                    v.addWidget(note('Specific acceleration includes gravity only for feasibility screening. The CSV acceleration is kinematic. These limits are provisional, not measured actuator limits.'))
                if section == 'reward':
                    v.addWidget(note('Return ends at the first valid modeled hit or the time limit. Recovery is appended after planning and receives no whip reward.'))
                if section == 'trajectory_objective':
                    v.addWidget(note('MPPI whole-trajectory score: preferred fold, contact, cast and effort. Recovery is not scored.'))
                v.addStretch()
                sc = QScrollArea()
                sc.setWidgetResizable(True)
                sc.setWidget(content)
                controls.addTab(sc, title)
            self.add_vector(rb, 'Jerk action bounds', 'action', 'jerk_limit_m_s3', 1, 300, 5, 'm/s³')
            self.model_note = note('')
            rb.addWidget(self.model_note)
            save = QPushButton('Save this method’s setup')
            save.clicked.connect(self.save_settings)
            rb.addWidget(save)
            split.addWidget(right)
            split.setSizes([460, 650])
            self.setup_note = note('Changes affect new MPPI runs; saved runs retain their model and launch.')
            body.addWidget(self.setup_note)
            quick = QPushButton('Quick setup · 256 samples')
            quick.clicked.connect(self.quick_setup)
            rb.addWidget(quick)
            self.live_enabled = QCheckBox('Record live 3D candidate snapshots')
            self.live_enabled.setChecked(self.cfg.get('visualization', {}).get('live_mppi', True))
            rb.addWidget(self.live_enabled)
            self.models.currentIndexChanged.connect(self.describe_model)
            self.refresh_models()
        progress_page = QWidget()
        pb = QVBoxLayout(progress_page)
        self.tabs.addTab(progress_page, 'Optimization progress')
        progress_scroll = QScrollArea()
        progress_scroll.setWidgetResizable(True)
        progress_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        pb.addWidget(progress_scroll)
        progress_content = QWidget()
        pb = QVBoxLayout(progress_content)
        progress_scroll.setWidget(progress_content)
        row = QHBoxLayout()
        row.addWidget(QLabel('Run'))
        self.progress_runs = QComboBox()
        self.progress_runs.setMinimumContentsLength(20)
        self.progress_runs.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.progress_runs.currentIndexChanged.connect(self.select_progress_run)
        row.addWidget(self.progress_runs, 1)
        self.stop = QPushButton('Stop after current update')
        self.stop.setEnabled(False)
        self.stop.clicked.connect(self.stop_run)
        row.addWidget(self.stop)
        pb.addLayout(row)
        details_body = pb
        from .mppi_dashboard import MPPIDashboard
        self.mppi_dashboard = MPPIDashboard()
        pb.addWidget(self.mppi_dashboard)
        self.progress_note = note('Select a saved run or start a new one.')
        self.progress_note.setObjectName('pipelineBanner')
        details_body.addWidget(self.progress_note)
        self.figure = Figure(layout='constrained', facecolor='white')
        self.canvas = FigureCanvasQTAgg(self.figure)
        details_body.addWidget(self.canvas, 1)
        self.progress_note.hide()
        self.canvas.setMinimumHeight(240)
        self.canvas.setMaximumHeight(320)
        options = QHBoxLayout()

        def change_points():
            self.last_history = None
            self.poll()
        self.mppi_chart = QComboBox()
        self.mppi_chart.addItems(['Current lookahead search', 'Committed path trends'])
        self.mppi_chart.currentIndexChanged.connect(change_points)
        options.addWidget(self.mppi_chart)
        live_button = QPushButton('Watch live 3D')
        live_button.clicked.connect(lambda: self.tabs.setCurrentWidget(self.live_view))
        options.addWidget(live_button)
        save_plot = QPushButton('Save plots…')
        save_plot.clicked.connect(self.save_plot)
        options.addWidget(save_plot)
        files = QPushButton('Open run files')
        files.clicked.connect(self.open_progress_files)
        options.addWidget(files)
        options.addStretch()
        details_body.addLayout(options)
        self.saved_log = QPlainTextEdit()
        self.saved_log.setReadOnly(True)
        self.saved_log.setMaximumHeight(150)
        self.saved_log.setMaximumBlockCount(200)
        self.saved_log.setVisible(True)
        self.saved_log.setMinimumHeight(100)
        self.log_toggle = QCheckBox('Show selected run log')
        self.log_toggle.toggled.connect(self.toggle_progress_log)
        self.follow_log = QCheckBox('Follow latest log')
        self.follow_log.setChecked(True)
        options.addWidget(self.follow_log)
        details_body.addWidget(self.log_toggle)
        details_body.addWidget(self.saved_log)
        self.job = BackgroundJob(root)
        self.job.setParent(self)
        self.job.hide()
        self.job.finished.connect(self.job_finished)
        library = QWidget()
        lb = QVBoxLayout(library)
        self.tabs.addTab(library, 'Saved plans')
        row = QHBoxLayout()
        refresh = QPushButton('Refresh library')
        refresh.clicked.connect(self.refresh_library)
        row.addWidget(refresh)
        row.addStretch()
        lb.addLayout(row)
        headers = ['Name', 'Model', 'Status', 'Attempts / iterations', 'Created']
        headers.append('Strike definition')
        self.library = QTableWidget(0, len(headers))
        self.library.setHorizontalHeaderLabels(headers)
        self.library.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.library.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.library.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.library.verticalHeader().hide()
        self.library.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        lb.addWidget(self.library, 1)
        row = QHBoxLayout()
        for label, callback in [('Show progress', self.show_selected), ('Rehearse selected', self.rehearse_selected), ('Open files', self.open_selected), ('Archive from library', self.archive_selected)]:
            button = QPushButton(label)
            button.clicked.connect(callback)
            row.addWidget(button)
        lb.addLayout(row)
        self.library_note = note('Historical force policies remain in the historical workspace. Their weights are not compatible with jerk actions.')
        lb.addWidget(self.library_note)
        rehearsal = QWidget()
        re = QVBoxLayout(rehearsal)
        self.tabs.addTab(rehearsal, 'Rehearsal and export')
        row = QHBoxLayout()
        self.rehearsal_note = note('Rehearse a completed MPPI plan.')
        row.addWidget(self.rehearsal_note, 1)
        open_saved = QPushButton('Open saved PVA rehearsal')
        open_saved.clicked.connect(self.open_saved)
        row.addWidget(open_saved)
        re.addLayout(row)
        self.inspector = RehearsalWorkspace(root, inspection_only=True)
        re.addWidget(self.inspector, 1)
        self.rehearsal_job = BackgroundJob(root)
        re.addWidget(self.rehearsal_job)
        self.rehearsal_job.finished.connect(self.rehearsal_finished)
        from .mppi_live_view import MPPILiveView
        self.live_view = MPPILiveView()
        self.tabs.addTab(self.live_view, 'Live 3D search')
        self.live_view.run_selector.currentIndexChanged.connect(self.select_live_run)
        self.timer = QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self.poll)
        self.timer.start()
        self.tabs.currentChanged.connect(lambda _: self.set_page_active(self.active))
        self.refresh_library()
        self.poll()
        self.log_toggle.setChecked(True)

    def number(self, form, label, key, low, high, step, integer=False):
        widget = QSpinBox() if integer else QDoubleSpinBox()
        if not integer:
            widget.setDecimals(6 if key[-1] in ('learning_rate', 'entropy_coefficient', 'relative_improvement') else 4)
        widget.setRange(low, high)
        widget.setSingleStep(step)
        widget.setValue(self.cfg[key[0]][key[1]])
        self.fields[key] = widget
        form.addRow(label, widget)

    def update_success_controls(self):
        simple = self.success_rule.currentData() == 'tip_contact_v1'
        self.success_note.setText('Success: the tip reaches the target. Speed, reversal and bend statistics do not gate a hit. Existing feasibility checks still apply.' if simple else 'Historical task: saved speed, direction, first-contact and enabled strategy gates all apply.')
        for widget in self.gate_controls:
            self.strike_form.setRowVisible(widget, not simple)

    def quick_setup(self):
        for key, value in dict(samples=256, initial_minimum_iterations=8, initial_patience=4, minimum_iterations=3, patience=2).items():
            self.fields['mppi', key].setValue(value)
        self.setup_note.setText('Quick setup staged for the next run: 256 samples, 8 initial / 3 later minimum iterations, shorter plateau patience. Horizon, physics and running jobs are unchanged.')

    def add_vector(self, layout, title, section, key, low, high, step, unit):
        group = QGroupBox(title + ' [' + unit + ']')
        row = QHBoxLayout(group)
        fields = []
        for i, axis in enumerate('XYZ'):
            row.addWidget(QLabel(axis))
            spin = QDoubleSpinBox()
            spin.setDecimals(2 if section == 'action' else 4)
            spin.setRange(low, high)
            spin.setSingleStep(step)
            spin.setValue(self.cfg[section][key][i])
            spin.setFixedWidth(100)
            row.addWidget(spin)
            fields.append(spin)
        self.fields[section, key] = fields
        layout.addWidget(group)

    def collect(self):
        if self.spline_setup:
            from .spline_setup import collect
            return collect(self)
        cfg = deepcopy(self.cfg)
        cfg['task']['success_criterion'] = self.success_rule.currentData()
        for (section, key), widget in self.fields.items():
            cfg[section][key] = [w.value() for w in widget] if isinstance(widget, list) else widget.isChecked() if isinstance(widget, QCheckBox) else widget.value()
        cfg['task']['first_contact_only'] = self.first_contact.isChecked()
        if hasattr(self, 'pullback'):
            cfg['task']['require_pullback'] = self.pullback.isChecked()
            cfg['task']['require_wave'] = self.wave.isChecked()
        cfg['mppi']['initialization'] = self.initialization.currentData()
        cfg.setdefault('visualization', {})['live_mppi'] = self.live_enabled.isChecked()
        if cfg['mppi'].get('mode') == 'receding':
            cfg['mppi']['horizon_s'] = self.horizon.value() / 30
        else:
            cfg['task']['duration_s'] = self.horizon.value() / 30
        cfg['model_path'] = self.models.currentData() or self.cfg['model_path']
        validate_settings(cfg)
        return cfg

    def save_settings(self):
        try:
            self.cfg = self.collect()
            atomic_json(self.root / 'config/pva/systematic_strike.json' if self.spline_setup else settings_path(self.root, 'mppi'), self.cfg)
            self.setup_note.setText('Saved MPPI setup. Existing runs are unchanged.')
            self.changed.emit()
        except (ValueError, OSError) as e:
            self.setup_note.setText(str(e))

    def refresh_models(self):
        selected = self.models.currentData() or self.cfg['model_path']
        if selected:
            p = Path(selected)
            selected = str(p if p.is_absolute() else self.root / p)
        self.models.blockSignals(True)
        self.models.clear()
        for p in model_paths(self.root):
            self.models.addItem(model_label(p), str(p.resolve()))
        index = self.models.findData(selected)
        if index < 0:
            reviewed = selected and Path(selected).is_file() and configured_development_review(self.cfg, selected)
            self.models.addItem('Reviewed M0 development · simulation only' if reviewed else 'Selected bootstrap model · fitting / unavailable' if selected else 'No fitted model selected', selected)
            index = self.models.count() - 1
        self.models.setCurrentIndex(index)
        self.models.blockSignals(False)
        self.describe_model()

    def describe_model(self):
        path = self.models.currentData()
        m = read_json(path, {}) if path else {}
        reviewed = bool(m) and configured_development_review(self.cfg, path)
        self.run.setEnabled(bool(m) and (m.get('provenance', {}).get('fit_complete') is not False or bool(reviewed)) and (not (hasattr(self, 'job') and self.job.running)))
        provenance = m.get('provenance', {})
        mass = m.get('mass_measurement', {})
        self.model_note.setText(f"{self.models.currentText()}\nDrone {mass.get('drone_mass_kg', 0) * 1000:g} g · cable {mass.get('cable_assembly_mass_kg', 0) * 1000:g} g\n" + (provenance.get('quality_note', 'Fitted model; next flights provide prospective evidence.') if provenance.get('fit_complete') else 'Unfinished or historical model; inspect its original provenance.') if m else 'No fitted model available. Collect and review the new preliminary takes before fitting M0 and planning.')
        if reviewed:
            self.model_note.setText(self.model_note.text() + '\nReviewed for offline planning with retained M0. Physical validation is pending.')

    def select_model(self, path):
        i = self.models.findData(path)
        if i < 0:
            self.models.addItem(model_label(Path(path)), path)
            i = self.models.count() - 1
        self.models.setCurrentIndex(i)
        self.tabs.setCurrentIndex(0)

    def start_run(self):
        if self.job.running:
            return
        try:
            cfg = self.collect()
            directory, command = prepare(self.root, cfg, self.run_name.text(), checkpoint=None)
            if self.spline_setup and (not cfg.get('spline_seed_directory')) and (cfg['mppi'].get('initialization', 'templates') != 'from_scratch'):
                template = Path(cfg['proposal_templates_path'])
                shutil.copy2(template if template.is_absolute() else self.root / template, directory / 'proposal_baselines.npz')
            command[0] = sys.executable
            self.current_run = directory
            self.job.start(directory, command)
            self.run.setEnabled(False)
            self.tabs.setCurrentIndex(1)
            self.refresh_library()
        except (OSError, ValueError, KeyError) as e:
            self.setup_note.setText('Could not start: ' + str(e))

    def refresh_library(self):
        selected = self.selected() if hasattr(self, 'runs') else None
        base = self.root / 'runs' / 'mppi_pva'
        self.runs = [p.parent for p in sorted(base.glob('*/identity.json'), reverse=True) if not (p.parent / 'ARCHIVED').exists()]
        self.library_fingerprint = self.run_fingerprint()
        self.library.setRowCount(len(self.runs))
        for i, p in enumerate(self.runs):
            identity = read_json(p / 'identity.json', {})
            status = read_json(p / 'status.json', {})
            values = [identity.get('name', p.name), model_label(p / 'model.json'), status.get('status', 'unknown'), str(status.get('attempts', status.get('iteration', status.get('iterations', '—')))), p.name]
            task = read_json(p / 'settings.json', {}).get('task', {})
            values.append('Targeted strike · saved fold rule' if task.get('success_criterion') == 'targeted_fold_strike_v1' else 'Tip reaches target' if task.get('success_criterion') == 'tip_contact_v1' else 'Legacy: travelling bend + pullback' if task.get('require_wave') else 'Legacy: forward/backward whip' if task.get('require_pullback') else 'Legacy: directed tip hit')
            for j, v in enumerate(values):
                self.library.setItem(i, j, QTableWidgetItem(v))
        if self.runs:
            self.library.selectRow(self.runs.index(selected) if selected in self.runs else 0)
            if self.current_run is None:
                self.current_run = self.runs[0]
        selectors = [self.progress_runs]
        selectors.append(self.live_view.run_selector)
        for selector in selectors:
            selector.blockSignals(True)
            selector.clear()
            for p in self.runs:
                identity = read_json(p / 'identity.json', {})
                status = read_json(p / 'status.json', {})
                selector.addItem(identity.get('name', p.name) + ' · ' + p.name + ' · ' + status.get('status', 'unknown'), str(p))
            selector.setCurrentIndex(selector.findData(str(self.current_run)))
            selector.blockSignals(False)

    def select_progress_run(self):
        self.select_run(self.progress_runs.currentData())

    def select_live_run(self):
        self.select_run(self.live_view.run_selector.currentData())

    def select_run(self, path):
        if not path:
            return
        self.current_run = Path(path)
        self.last_history = None
        if self.current_run in self.runs:
            self.library.selectRow(self.runs.index(self.current_run))
        self.poll()

    def open_progress_files(self):
        if self.current_run:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_run)))

    def toggle_progress_log(self, visible):
        self.saved_log.setVisible(visible)
        self.poll()

    def run_fingerprint(self):
        base = self.root / 'runs' / 'mppi_pva'
        return tuple(((str(p), p.stat().st_mtime_ns, (p.parent / 'ARCHIVED').exists()) for p in sorted(base.glob('*/status.json'))))

    def selected(self):
        row = self.library.currentRow()
        return self.runs[row] if 0 <= row < len(self.runs) else None

    def show_selected(self):
        if self.selected():
            self.current_run = self.selected()
            self.last_history = None
            self.tabs.setCurrentIndex(1)
            self.poll()

    def open_selected(self):
        if self.selected():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.selected())))

    def archive_selected(self):
        p = self.selected()
        if not p:
            return
        if read_json(p / 'status.json', {}).get('status') == 'running':
            self.library_note.setText('Stop the run before archiving it.')
            return
        (p / 'ARCHIVED').touch()
        self.library_note.setText('Hidden from this library. Files remain in ' + str(p))
        self.refresh_library()

    def stop_run(self):
        if self.current_run and read_json(self.current_run / 'status.json', {}).get('status') == 'running':
            (self.current_run / 'STOP').touch()

    def job_finished(self, code):
        self.describe_model()
        self.refresh_library()
        self.poll()

    def poll(self):
        if self.run_fingerprint() != self.library_fingerprint:
            self.refresh_library()
        p = self.current_run
        self.live_view.set_job(p)
        selector = self.live_view.run_selector
        selector.blockSignals(True)
        selector.setCurrentIndex(selector.findData(str(p)))
        selector.blockSignals(False)
        if not p:
            return
        self.progress_runs.blockSignals(True)
        self.progress_runs.setCurrentIndex(self.progress_runs.findData(str(p)))
        self.progress_runs.blockSignals(False)
        status = read_json(p / 'status.json', {})
        self.stop.setEnabled(status.get('status') == 'running')
        if (p / 'STOP').exists() and status.get('status') == 'running':
            self.stop.setText('Stop requested')
            self.stop.setEnabled(False)
        else:
            self.stop.setText('Stop after current update')
        if self.log_toggle.isChecked():
            log = p / 'console.log'
            if not log.exists():
                log = p / 'stdout.log'
            if log.exists():
                with log.open('rb') as stream:
                    stream.seek(max(0, log.stat().st_size - 14000))
                    text = stream.read().decode('utf-8', errors='replace')
                from .mppi_dashboard import readable_log
                text = readable_log(text)
                if self.saved_log.toPlainText() != text:
                    scroll = self.saved_log.verticalScrollBar()
                    position = scroll.value()
                    self.saved_log.setPlainText(text)
                    scroll.setValue(scroll.maximum() if self.follow_log.isChecked() else position)
            else:
                self.saved_log.setPlainText('No console log saved for this run.')
        receding = read_json(p / 'settings.json', {}).get('mppi', {}).get('mode') == 'receding'
        self.mppi_chart.setVisible(receding)
        identity = read_json(p / 'identity.json', {})
        amount = status.get('attempts', status.get('iteration', status.get('iterations', '—')))
        self.progress_note.setText(f"{identity.get('name', p.name)} · {status.get('status', 'unknown')} · {amount} " + 'iterations' + '\n' + status.get('stage', status.get('error', status.get('stop_reason', ''))) + ('\nSaved best plan: ' + ('infeasible' if status['best_failed'] else 'modeled valid hit' if status.get('best_success') else 'no modeled valid hit') if 'best_failed' in status else ''))
        if receding:
            clock = status.get('maneuver_time_s', status.get('time_s', status.get('maneuver_duration_s', 0.0)))
            self.progress_note.setText(self.progress_note.text() + f'\nSimulated maneuver: {clock:.3f} s · receding lookahead')
        requirement = read_json(p / 'settings.json', {}).get('task', {}).get('require_pullback', False)
        task = read_json(p / 'settings.json', {}).get('task', {})
        message = ('Fold diagnostic only; full offline targeted strike.' if read_json(p / 'settings.json', {}).get('fold_requirement') == 'diagnostic_only' else 'Travelling fold required before the scored strike; full offline trajectory.') if task.get('success_criterion') == 'targeted_fold_strike_v1' else 'Task requires forward pull, then backward drone motion at tip contact.' if requirement else 'Saved tip-hit task: backward release was not required.'
        self.progress_note.setText(self.progress_note.text() + '\n' + message)
        path = p / ('windows.json' if receding else 'history.json')
        history_key = (str(p), path.stat().st_mtime_ns if path.exists() else None)
        if getattr(self, 'history_cache_key', None) != history_key:
            self.history_cache_key = history_key
            self.history_rows = read_json(path, [])
        rows = self.history_rows
        search_view = False
        history_path = p / 'history.json'
        key = (str(p), history_path.stat().st_mtime_ns if history_path.exists() else None)
        if getattr(self, 'mppi_history_key', None) != key:
            self.mppi_history_key = key
            self.mppi_history = read_json(history_path, [])
        modified = max((f.stat().st_mtime for f in (p / 'status.json', history_path) if f.exists()), default=time.time())
        self.mppi_dashboard.update_run(status, read_json(p / 'settings.json', {}), self.mppi_history, rows if receding else [], max(0, time.time() - modified))
        search_view = receding and self.mppi_chart.currentIndex() == 0
        if search_view:
            step = status.get('command_step', status.get('command_steps', 0))
            rows = [r for r in self.mppi_history if r.get('command_step') == step]
            if not rows and status.get('status') != 'running':
                rows = self.mppi_history
            history_key = (key, step, 'search')
        if self.last_history == history_key:
            return
        self.last_history = history_key
        if not rows:
            self.figure.clear()
            ax = self.figure.subplots()
            ax.axis('off')
            ax.text(0.5, 0.5, 'Waiting for the first iteration result.' if search_view or not receding else 'Committed-path trends appear after the first command.', ha='center', va='center', wrap=True)
            self.canvas.draw_idle()
            return
        self.figure.clear()
        axes = self.figure.subplots(1, 2).ravel()
        x = [r.get('attempts', r.get('iteration')) for r in rows]
        keys = ['best_reward', 'success', 'best_minimum_tip_distance_m', 'effective_samples']
        titles = ['Saved best return', 'Modeled valid hits (sampled batch)', 'Saved best tip distance', 'Effective importance samples']
        keys = ['best_reward', 'best_minimum_tip_distance_m']
        titles = ['Best return', 'Closest tip distance']
        if read_json(p / 'settings.json', {}).get('trajectory_objective', {}).get('schema') == 'targeted_fold_strike_v1':
            keys = ['best_score', 'strike_distance_m']
            titles = ['Best feasible strike score', 'Distance at scored strike']
            if read_json(p / 'settings.json', {}).get('trajectory_objective', {}).get('free_target'):
                keys = ['best_score', 'horizontal_cable_rms_m']
                titles = ['Best feasible release score', 'Cable height RMS at release [m]']
        if receding and (not search_view):
            x = [r['time_s'] for r in rows]
            keys = ['actual_reward', 'actual_minimum_tip_distance_m']
            titles = ['Committed return', 'Committed closest tip distance']
        if search_view:
            x = [r['iteration'] for r in rows]
            keys = ['lookahead_score', 'predicted_distance_m']
            titles = ['Best lookahead score', 'Best proposal closest tip distance']
        for ax, key, title in zip(axes, keys, titles):
            factor = 100 if key in ('success', 'failures') else 1
            ax.plot(x, [(r.get(key) if r.get(key) is not None else float('nan')) * factor for r in rows], color='#2563eb', lw=1.6, label=None)
            ax.set_title(title, loc='left', fontsize=10)
            ax.set_xlabel('Simulated maneuver time [s]' if receding and (not search_view) else 'Iterations')
            ax.grid(alpha=0.2)
            ax.spines[['top', 'right']].set_visible(False)
            if key == 'actual_success':
                ax.set_ylim(0, 1)
                ax.set_yticks([0, 1], ['No', 'Yes'])
            if key in ('success', 'failures'):
                ax.set_ylabel('%')
                ax.set_ylim(0, 100)
            if 'distance' in key:
                ax.set_ylabel('m')
        self.canvas.draw_idle()

    def save_plot(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save learning plots', str(self.root / 'runs' / ('mppi' + '-progress.png')), 'PNG (*.png);;PDF (*.pdf)')
        if path:
            self.figure.savefig(path, dpi=180)

    def rehearse_selected(self):
        p = self.selected()
        if p is None or self.rehearsal_job.running:
            return
        if read_json(p / 'status.json', {}).get('status') == 'running':
            self.library_note.setText('Stop or finish the run before rehearsing.')
            return
        cfg = read_json(p / 'settings.json')
        if not (p / 'plan.npz').exists():
            self.library_note.setText('Plan not yet available.')
            return
        self.rehearsal_output = self.root / 'runs/rehearsals_pva' / (stamp() + '-' + 'mppi')
        tool = p / 'source_snapshot/tools/rehearse_pva.py'
        command = [sys.executable, '-u', str(tool), '--job', str(p), '--output', str(self.rehearsal_output)]
        self.rehearsal_note.setText('Generating from the selected run’s frozen launch and model: ' + read_json(p / 'identity.json')['name'])
        self.rehearsal_job.start(self.root / 'runs/pva_jobs' / stamp(), command)
        self.tabs.setCurrentIndex(3)
        self.inspector.clear_result()

    def rehearsal_finished(self, code):
        if code:
            self.rehearsal_note.setText('Generation failed. Open the log; no new command has been exported.')
            return
        try:
            self.inspector.load_result(self.rehearsal_output)
            m = self.inspector.metadata
            message = m['outcome'] if m.get('preview_only') else 'Complete command is ready for inspection.'
            self.rehearsal_note.setText(message)
            self.changed.emit()
        except (OSError, ValueError, KeyError) as e:
            self.rehearsal_note.setText(str(e))

    def open_saved(self):
        p = QFileDialog.getExistingDirectory(self, 'Open saved PVA rehearsal', str(self.root / 'runs/rehearsals_pva'))
        if p:
            try:
                self.inspector.clear_result()
                self.inspector.load_result(p)
            except (OSError, ValueError, KeyError) as e:
                self.rehearsal_note.setText(str(e))

    def set_page_active(self, active):
        self.active = active
        self.inspector.set_page_active(active and self.tabs.currentIndex() == 3)
        self.live_view.set_active(active and self.tabs.currentWidget() is self.live_view)

    def shutdown(self):
        self.timer.stop()
        self.live_view.shutdown()
        return self.inspector.shutdown()
