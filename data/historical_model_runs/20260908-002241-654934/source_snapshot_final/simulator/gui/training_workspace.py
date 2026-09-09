"""Matching PPO and SAC workspaces: controls, learning curves and actual flight."""
from collections import deque
from pathlib import Path
import shutil
import time

import numpy as np
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QFrame,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QSplitter, QFileDialog, QProgressBar, QLineEdit, QTabWidget)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from simulator.workflow import read_json, prepare_training, stamp
from .research_widgets import note, BackgroundJob, draw_learning, load_history, export_learning
from .process_status import process_is_running
from .live_worker import LiveFlightWorker
from .viewer_3d import create_viewer


class PolicyViewport(QWidget):
    flight_finished = Signal()

    def __init__(self, root, algorithm, parent=None):
        super().__init__(parent)
        self.root, self.algorithm = Path(root), algorithm
        self.viewer = self.worker = self.thread = None
        self.arrays, self.metadata = None, {}
        self.active = False
        self.error = None
        self.trail = deque(maxlen=5000)
        self.outer = QVBoxLayout(self)
        self.outer.setContentsMargins(0, 0, 0, 0)
        self.source = note('First validation trial · waiting for a trained policy')
        self.outer.addWidget(self.source)
        self.host = QVBoxLayout()
        self.outer.addLayout(self.host, 1)
        controls = QHBoxLayout()
        self.play = QPushButton('Replay trial')
        self.play.setEnabled(False)
        self.play.clicked.connect(self.replay)
        controls.addWidget(self.play)
        self.execute = QPushButton('Execute')
        self.execute.setEnabled(False)
        self.execute.setToolTip('Execute the learned force sequence once, then return to PID hover.')
        self.execute.clicked.connect(self.execute_policy)
        controls.addWidget(self.execute)
        self.stop = QPushButton('Stop flight')
        self.stop.setEnabled(False)
        self.stop.clicked.connect(self.stop_flight)
        controls.addWidget(self.stop)
        self.camera = QComboBox()
        self.camera.addItems(['Perspective', 'Side XZ', 'Top XY', 'Front YZ'])
        self.camera.currentTextChanged.connect(lambda value: self.viewer.set_camera_preset(value) if self.viewer else None)
        controls.addWidget(self.camera)
        controls.addStretch()
        self.outer.addLayout(controls)
        self.status = note('Initial drone + cable state → one force sequence → PID recovery')
        self.outer.addWidget(self.status)
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.tick)
        self.playing = False

    def ensure_viewer(self):
        if self.viewer is not None:
            return
        model, task = [read_json(self.root / 'config' / name) for name in ('model.json', 'task.json')]
        lengths = [length / count for length, count in zip(model['cable']['marker_interval_lengths_m'],
                   model['cable']['segments_per_marker_interval']) for _ in range(count)]
        q = np.tile(task['initial_root_position_m'], (len(lengths) + 1, 1)).astype(float)
        q[:, 2] -= np.r_[0, np.cumsum(lengths)]
        self.viewer = create_viewer(q, task['target_position_m'], task['desired_strike_direction_world'],
                                    task['success']['tip_target_distance_m'], self)
        self.viewer.set_live_flight(True)
        self.viewer.setMinimumHeight(150)
        if hasattr(self.viewer, 'plotter'):
            self.viewer.plotter.interactor.setMinimumHeight(150)
        self.host.addWidget(self.viewer)

    def set_active(self, active):
        self.active = active
        if active:
            self.ensure_viewer()
        if active or self.worker is not None:
            self.timer.start()
        else:
            self.timer.stop()

    def clear_trial(self):
        self.arrays, self.metadata = None, {}
        self.playing = False
        self.play.setEnabled(False)
        self.source.setText('First validation trial · waiting for results from this run')
        self.status.setText('Initial drone + cable state → one force sequence → PID recovery')

    def load_trial(self, path, record):
        if self.worker is not None:
            return False
        with np.load(path, allow_pickle=False) as data:
            import json
            self.metadata = json.loads(str(data['metadata']))
            if self.metadata.get('schema') != 'training_validation_trial_v1':
                raise ValueError('This is not a recorded training validation trial.')
            self.arrays = {key: data[key].copy() for key in data.files if key != 'metadata'}
        step = int(record['training_episodes'])
        self.source.setText(f'Actual validation · trial 1 · {step:,} training attempts · policy {record["checkpoint_sha256"][:8]}')
        if record.get('evaluation_reused'):
            self.source.setText(self.source.text() + '\nUnchanged policy · prior validation reused')
        self.source.setToolTip(f'Evaluation {record["evaluation_id"]}\nScenario set {record["scenario_id"]}')
        self.play.setEnabled(True)
        self.replay()
        return True

    def replay(self):
        if self.arrays is None or self.worker is not None:
            return
        self.ensure_viewer()
        meta = self.metadata
        self.viewer.set_target(meta['target_position_m'], meta['desired_strike_direction_world'], meta['target_radius_m'])
        q = self.arrays['positions_m'][:, 0]
        if hasattr(self.viewer, 'plotter'):
            finite = q[np.isfinite(q).all(axis=-1)]
            if len(finite):
                low = np.minimum(finite.min(0), meta['target_position_m']) - .25
                high = np.maximum(finite.max(0), meta['target_position_m']) + .25
                self.viewer.plotter.reset_camera(bounds=tuple(np.column_stack((low, high)).ravel()))
        self.started = time.perf_counter()
        self.playing = True
        self.timer.start()

    def start_latest(self, directory):
        if self.thread is not None:
            return
        directory = Path(directory)
        checkpoint = directory / 'checkpoints/latest.pt'
        if not checkpoint.is_file():
            raise ValueError('This run has no latest policy yet.')
        # Copy stable checkpoint bytes: training may atomically replace latest.pt.
        frozen = directory / 'manual_flights' / stamp()
        (frozen / 'checkpoints').mkdir(parents=True)
        shutil.copy2(checkpoint, frozen / 'checkpoints/latest.pt')
        for name in ('model.json', 'task.json', 'ppo.json', 'sac.json'):
            if (directory / name).is_file():
                shutil.copy2(directory / name, frozen / name)
        configs = [read_json(frozen / name) for name in ('model.json', 'task.json', 'ppo.json')]
        self.ensure_viewer()
        self.viewer.set_target(configs[1]['target_position_m'], configs[1]['desired_strike_direction_world'],
                               configs[1]['success']['tip_target_distance_m'])
        self.thread = QThread(self)
        self.worker = LiveFlightWorker(*configs, frozen / 'checkpoints/latest.pt')
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.status.connect(self.status.setText)
        self.worker.failed.connect(self.failed)
        self.worker.finished.connect(self.on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread_done)
        self.playing = False
        self.execute.setEnabled(False)
        self.error = None
        self.trail.clear()
        self.source.setText(f'{self.algorithm} · latest policy from {directory.name} · live simulated flight')
        self.play.setEnabled(False)
        self.stop.setEnabled(True)
        self.timer.start()
        self.thread.start()

    def failed(self, error):
        self.error = error
        self.execute.setEnabled(False)
        self.status.setText(error)

    def execute_policy(self):
        if self.worker is not None and self.execute.isEnabled():
            self.execute.setEnabled(False)
            self.worker.command('strike')
            self.status.setText('Preparing one policy attempt…')

    def stop_flight(self):
        self.execute.setEnabled(False)
        if self.worker is not None:
            self.worker.command('stop')
            self.status.setText('Stopping simulated flight…')

    def on_finished(self, arrays, summary):
        self.execute.setEnabled(False)
        self.stop.setEnabled(False)
        self.status.setText(self.error or 'Flight stopped. Training validation replay is available.')

    def thread_done(self):
        self.thread.deleteLater()
        self.thread = self.worker = None
        self.play.setEnabled(self.arrays is not None)
        self.flight_finished.emit()

    def tick(self):
        if self.worker is not None:
            frame = self.worker.take_frame()
            if frame is None:
                return
            self.execute.setEnabled(bool(frame['ready']) and not frame.get('preparing',False)
                                    and not self.worker.strike_pending.is_set()
                                    and not self.worker.stop_requested.is_set())
            self.trail.append(frame['positions'][[0, -1]])
            if self.active:
                trail = np.asarray(self.trail)
                self.viewer.update_state(frame['positions'], frame['command'], trail[:, 0], trail[:, 1])
                phase = {0: 'PID hover', 1: 'Open-loop strike', 2: 'PID recovery'}[frame['phase']]
                self.status.setText(f'{phase} · {frame["time_s"]:.1f} s · {frame["realtime_rate"]:.2f}× real time · {frame["message"]}')
        elif self.active and self.playing and self.arrays is not None:
            times = self.arrays['time_s']
            index = min(len(times) - 1, int(np.searchsorted(times, time.perf_counter() - self.started)))
            q = self.arrays['positions_m'][:, 0]
            self.viewer.update_state(q[index], self.arrays['commanded_force_world_n'][index, 0],
                                     q[:index+1, 0], q[:index+1, -1])
            phase = 'Open-loop strike' if self.arrays['striking'][index, 0] else 'PID recovery'
            hit = 'valid hit' if self.arrays['hit'][index, 0] else 'no valid hit'
            if not self.metadata['first_trial_planned']:
                phase = 'Plan refused'
            self.status.setText(f'{times[index]:.2f} s · {phase} · {hit} · planned cutoff {self.metadata["cutoff_s"]:.2f} s')
            if index == len(times) - 1:
                self.playing = False


class AlgorithmTrainingPage(QWidget):
    checkpoint_requested = Signal(str)

    def __init__(self, root, algorithm, parent=None):
        super().__init__(parent)
        self.root, self.algorithm = Path(root), algorithm.upper()
        self.config = read_json(self.root / 'config' / f'{algorithm.lower()}.json')
        self.directory = None
        self.last_record = None
        self.history_signature = None
        self.active = False
        self.resume_checkpoint = None
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        training_tab = QWidget()
        self.tabs.addTab(training_tab, 'Training')
        outer = QVBoxLayout(training_tab)
        outer.setContentsMargins(20, 16, 20, 18)
        outer.setSpacing(10)
        self.model_note=note('');outer.addWidget(self.model_note)
        toolbar = QHBoxLayout()
        self.runs = QComboBox()
        self.runs.setPlaceholderText('No training runs yet')
        self.runs.setMinimumWidth(210)
        self.runs.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.runs.currentIndexChanged.connect(self.select_run)
        toolbar.addWidget(QLabel('Run'))
        toolbar.addWidget(self.runs, 1)
        self.start_button = QPushButton('Train new policy')
        self.start_button.setObjectName('primaryButton')
        self.start_button.clicked.connect(lambda: self.start_training(False))
        toolbar.addWidget(self.start_button)
        self.resume_button = QPushButton('Continue latest')
        self.resume_button.clicked.connect(lambda: self.start_training(True))
        toolbar.addWidget(self.resume_button)
        self.stop_button = QPushButton('Stop training')
        self.stop_button.clicked.connect(self.stop_training)
        toolbar.addWidget(self.stop_button)
        outer.addLayout(toolbar)
        naming = QHBoxLayout()
        naming.addWidget(QLabel('New run name'))
        self.run_name = QLineEdit()
        self.run_name.setPlaceholderText('e.g. Adaptation 1 — cable model update')
        self.run_name.setMaxLength(120)
        naming.addWidget(self.run_name, 1)
        outer.addLayout(naming)
        self.resume_note = note('Continue latest creates a new run and preserves the source run.')
        outer.addWidget(self.resume_note)
        settings = QHBoxLayout()
        self.seed, self.episodes, self.batch = QSpinBox(), QSpinBox(), QSpinBox()
        for spin, maximum, value in ((self.seed, 2147483646, self.config['seed']),
                (self.episodes, 1000000000, self.config['training']['requested_episodes']),
                (self.batch, 32768, self.config['training']['collection_batch'])):
            spin.setRange(1, maximum)
            spin.setValue(int(value))
            spin.setMaximumWidth(128)
        self.seed.setMinimum(0)
        self.device = QComboBox()
        self.device.addItems(['auto', 'cuda', 'cpu'])
        self.device.setCurrentText(self.config['training']['device'])
        for text, field in [('Seed', self.seed), ('Target attempts', self.episodes), ('Batch', self.batch), ('Device', self.device)]:
            settings.addWidget(QLabel(text))
            settings.addWidget(field)
        settings.addStretch()
        self.advanced_button = QPushButton('Algorithm settings')
        self.advanced_button.setCheckable(True)
        settings.addWidget(self.advanced_button)
        outer.addLayout(settings)
        self.advanced = QFrame()
        advanced_layout = QHBoxLayout(self.advanced)
        self.hyperparameters = {}
        keys = ('learning_rate', 'entropy_coefficient', 'update_epochs') if self.algorithm == 'PPO' else (
            'actor_learning_rate', 'critic_learning_rate', 'updates_per_collection')
        for key in keys:
            form = QFormLayout()
            spin = QDoubleSpinBox()
            integer = key in ('update_epochs', 'updates_per_collection')
            spin.setDecimals(0 if integer else 7)
            spin.setRange(1 if integer else 0, 1024 if integer else 1)
            spin.setValue(self.config[self.algorithm.lower()][key])
            spin.setSingleStep(1 if integer else .00001)
            self.hyperparameters[key] = spin
            form.addRow(key.replace('_', ' ').capitalize(), spin)
            advanced_layout.addLayout(form)
        self.advanced.hide()
        self.advanced_button.toggled.connect(self.advanced.setVisible)
        outer.addWidget(self.advanced)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFormat('Ready · no training run')
        outer.addWidget(self.progress)
        self.batch_progress = QProgressBar()
        self.batch_progress.setRange(0, 1000)
        self.batch_progress.hide()
        outer.addWidget(self.batch_progress)
        self.metrics = note('Validation: — valid hit · — hit + recovery · — task return')
        outer.addWidget(self.metrics)
        self.split = QSplitter(Qt.Orientation.Horizontal)
        plot_panel = QFrame()
        plot_panel.setObjectName('contentCard')
        plots = QVBoxLayout(plot_panel)
        heading = QHBoxLayout()
        heading.addWidget(QLabel('Learning progress'), 1)
        self.export_button = QPushButton('Export figures')
        self.export_button.clicked.connect(self.export_figures)
        heading.addWidget(self.export_button)
        plots.addLayout(heading)
        self.figure = Figure(figsize=(3.6, 4.6), facecolor='white')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumWidth(270)
        self.canvas.setMinimumHeight(160)
        plots.addWidget(self.canvas, 1)
        plots.addWidget(note('Current-policy validation · all attempts counted · raw curves for this seed'))
        draw_learning(self.figure, [], self.algorithm)
        self.split.addWidget(plot_panel)
        flight_panel = QFrame()
        flight_panel.setObjectName('contentCard')
        flight = QVBoxLayout(flight_panel)
        heading = QHBoxLayout()
        heading.addWidget(QLabel('Policy in 3D'), 1)
        self.run_latest_button = QPushButton('Run latest policy')
        self.run_latest_button.setObjectName('secondaryButton')
        self.run_latest_button.clicked.connect(self.run_latest)
        heading.addWidget(self.run_latest_button)
        flight.addLayout(heading)
        self.viewport = PolicyViewport(root, self.algorithm)
        flight.addWidget(self.viewport, 1)
        self.split.addWidget(flight_panel)
        self.split.setSizes([390, 650])
        outer.addWidget(self.split, 1)
        self.job = BackgroundJob(root)
        self.job.finished.connect(lambda _: self.refresh())
        outer.addWidget(self.job)
        from .policy_library_page import PolicyLibraryPage
        self.library = PolicyLibraryPage(root)
        self.tabs.addTab(self.library, 'Policies')
        self.library.use_requested.connect(self.checkpoint_requested.emit)
        self.library.continue_requested.connect(self.choose_resume_checkpoint)
        self.tabs.currentChanged.connect(lambda _: self.viewport.set_active(self.active and self.tabs.currentIndex() == 0))
        self.timer = QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self.refresh)
        self.refresh()

    def set_page_active(self, active):
        self.active = active
        self.viewport.set_active(active and self.tabs.currentIndex() == 0)
        if active:
            self.refresh()
            self.timer.start()
        else:
            self.timer.stop()

    def select_run(self):
        self.resume_checkpoint = None
        if hasattr(self, 'resume_note'):
            self.resume_note.setText('Continue latest creates a new run and preserves the source run.')
            self.resume_button.setText('Continue latest')
        selected = self.runs.currentData()
        self.directory = Path(selected) if selected else None
        if self.directory is not None and hasattr(self, 'batch'):
            filename = f'{self.algorithm.lower()}.json'
            saved = read_json(self.directory/'launch_config'/filename,
                              read_json(self.directory/filename, {}))
            settings = saved.get('training', {})
            for key, field in (('collection_batch', self.batch), ('requested_episodes', self.episodes)):
                if key in settings:
                    field.setValue(int(settings[key]))
            if 'seed' in saved:
                self.seed.setValue(int(saved['seed']))
            if 'device' in settings:
                self.device.setCurrentText(settings['device'])
        self.last_record = self.history_signature = None
        if hasattr(self, 'viewport') and self.viewport.thread is not None:
            self.viewport.stop_flight()
        if hasattr(self, 'viewport'):
            self.viewport.clear_trial()
        if hasattr(self, 'figure'):
            self.refresh_results()

    def refresh(self):
        parent = self.root / 'runs' / self.algorithm.lower()
        runs = sorted([path for path in parent.iterdir() if path.is_dir() and (path / 'run.json').exists()],
                      key=lambda path: path.name, reverse=True) if parent.exists() else []
        values = [str(path) for path in runs]
        old_values = [self.runs.itemData(index) for index in range(self.runs.count())]
        if values != old_values:
            selected = str(self.directory) if self.directory else None
            self.runs.blockSignals(True)
            self.runs.clear()
            for path in runs:
                self.runs.addItem(read_json(path/'run.json',{}).get('display_name',path.name), str(path))
            if selected in values:
                self.runs.setCurrentIndex(values.index(selected))
            elif values:
                self.runs.setCurrentIndex(0)
            self.runs.blockSignals(False)
            self.select_run()
        self.refresh_results()
        self.library.refresh()

    def refresh_results(self):
        from simulator.cable.residual import cable_drag_description
        model_path=self.directory/'launch_config/model.json' if self.directory else self.root/'config/model.json'
        saved=read_json(model_path,{})
        settings_path=self.directory/'launch_config/ppo.json' if self.directory else self.root/'config/ppo.json'
        protocol=read_json(settings_path,{})
        scope='XYZ' if protocol.get('ppo',{}).get('stochastic_action_indices')==[0,1,2] else 'configured axes'
        mode='Refinement guard on' if protocol.get('update_guard',{}).get('enabled') else 'Acquisition · rollback guard off'
        self.model_note.setText(('Selected run uses its saved model' if self.directory else 'New run uses the active baseline')+
            f' · {cable_drag_description(saved)}. '
            'Applying another baseline does not change an existing policy run.')
        if saved.get('fullstate_execution',{}).get('enabled'):
            self.model_note.setText(self.model_note.text()+
                '\nFull model: force plan → 30 Hz full-state reference → drone tracking + NN → cable physics + NN.'
                ' Training scores the whip; recovery is exported separately and is not scored.')
        active_model=read_json(self.root/'config/model.json',{})
        if self.directory and active_model.get('fullstate_execution',{}).get('enabled'):
            self.model_note.setText(self.model_note.text()+'\nTrain new policy uses the active full-model baseline; continuing a saved policy keeps its saved model.')
        if self.algorithm=='PPO':
            acceleration=(f' · GPU graph replay · {protocol.get("training",{}).get("collection_batch",0):,} parallel environments'
                          if protocol.get('cuda_graph_physics') else '')
            self.model_note.setText(self.model_note.text()+f'\n{mode} · {scope} forces · near-hover initial states'+acceleration)
            deployment = protocol.get('deployment', {})
            if deployment.get('target_position_radius_m', 0.) or deployment.get('initial_position_radius_m', 0.):
                self.model_note.setText(self.model_note.text()+
                    f'\nTarget variation: {100*deployment.get("target_position_radius_m", 0.):g} cm radius'
                    f' · Initial attachment variation: {100*deployment.get("initial_position_radius_m", 0.):g} cm radius'
                    ' · Target known before open-loop execution')
        else:
            sac_path=self.directory/'launch_config/sac.json' if self.directory else self.root/'config/sac.json'
            sac=read_json(sac_path,{})
            axes=sac.get('sac',{}).get('stochastic_action_indices',[])
            scope=''.join('XYZ'[axis] for axis in axes)
            acceleration=' · GPU graph replay' if protocol.get('cuda_graph_physics') else ''
            initialization=' · searched strike initialization' if (sac.get('bootstrap') or {}).get('enabled') else ''
            self.model_note.setText(self.model_note.text()+f'\nSAC · {scope} forces · {sac.get("training",{}).get("collection_batch",0):,} parallel environments'+acceleration+initialization)
        status = read_json(self.directory / 'status.json', {}) if self.directory else {}
        state = status.get('status', 'READY')
        running = state in ('STARTING', 'RUNNING', 'STOPPING') and process_is_running(int(status.get('pid', 0)))
        if state in ('STARTING', 'RUNNING', 'STOPPING') and not running:
            state = 'INTERRUPTED'
        own_running = self.job.running if hasattr(self, 'job') else False
        completed = int(status.get('episodes', 0))
        if state=='STARTING' and self.directory:
            completed=max(completed,int(read_json(self.directory/'run.json',{}).get('parent_training_episodes',0)))
            # Older workers publish RUNNING only after their first full batch.
            if running and (self.directory/'validation_latest.json').exists():
                state='RUNNING'
        if not own_running and self.directory is not None:
            label='Validating the starting policy' if running and state=='STARTING' else state.title()
            if running and state=='STARTING' and (self.directory/'validation/latest.json').exists():
                label='Collecting the next training batch'
            if running and (self.directory/'STOP_REQUESTED').exists():
                label='Stopping — preserving the last saved update'
            self.job.status.setText(label)
            log_path=self.directory/'console.log'
            if log_path.exists():
                with log_path.open('rb') as stream:
                    stream.seek(max(0,log_path.stat().st_size-14000))
                    self.job.log.setPlainText(stream.read().decode('utf-8',errors='replace'))
        target = int(status.get('target_episodes', status.get('episodes_target', self.episodes.value())))
        stage = status.get('stage', '')
        if state=='RUNNING' and status.get('status')=='STARTING':
            stage='Collecting training batch'
        self.progress.setValue(min(1000, int(1000 * completed / max(1, target))))
        self.progress.setFormat(f'{state.title()} · {completed:,} / {target:,} attempts' + (f' · {stage}' if stage else ''))
        phase_total=int(status.get('phase_total',0))
        phase_step=int(status.get('phase_step',0))
        self.batch_progress.setVisible(running and phase_total>0)
        if running and phase_total>0:
            self.progress.setFormat(f'Running · {completed:,} / {target:,} completed attempts')
            self.batch_progress.setValue(min(1000,int(1000*phase_step/phase_total)))
            self.batch_progress.setFormat(f'{stage} · {phase_step:,}/{phase_total:,} steps · %p%')
            age=max(0,int(time.time()-status.get('progress_updated_at',time.time())))
            self.batch_progress.setToolTip(f'{status.get("collection_batch",0):,} parallel attempts in this batch. Last progress update {age}s ago. Steps measure simulation progress, not completed attempts.')
        exists = self.directory is not None and (self.directory / 'checkpoints/latest.pt').exists()
        self.start_button.setEnabled(not (running or own_running))
        self.resume_button.setEnabled((bool(self.resume_checkpoint) or exists) and not (running or own_running))
        stop_pending=self.directory is not None and (self.directory/'STOP_REQUESTED').exists()
        self.stop_button.setEnabled((running or (own_running and self.directory == self.job.directory)) and not stop_pending)
        if running and stop_pending:
            self.progress.setFormat(f'Stopping · {completed:,} saved attempts · discarding unfinished work')
        self.run_latest_button.setEnabled(exists and self.viewport.thread is None)
        history = load_history(self.directory)
        signature = (str(self.directory), len(history), history[-1]['record_id'] if history else None)
        if signature != self.history_signature:
            draw_learning(self.figure, history, self.algorithm)
            self.canvas.draw_idle()
            self.history_signature = signature
        self.export_button.setEnabled(bool(history))
        record = read_json(self.directory / 'validation/latest.json', {}) if self.directory else {}
        if record:
            metadata = read_json(self.directory / 'run.json', {})
            seed = metadata.get('seed', read_json(self.directory / f'{self.algorithm.lower()}.json', {}).get('seed', '—'))
            recovery_text=('recovery not scored' if saved.get('fullstate_execution',{}).get('enabled') else
                f'{100*record["hit_and_recovery_rate"]:.1f}% hit + recovery')
            self.metrics.setText(f'Seed {seed} · validation · {100*record["success_rate"]:.1f}% valid hit · '
                f'{recovery_text} · '
                f'{record["mean_episode_reward"]:.2f} task return · {record["episodes"]} trials')
            measurements=[]
            for key,label,unit in (
                ('mean_impact_speed_m_s','Impact speed','m/s'),
                ('mean_hit_time_s','Hit time','s'),
                ('mean_maximum_execution_drone_displacement_m','Max. drone travel','m')):
                value=record.get(key)
                if value is not None and np.isfinite(value):
                    measurements.append(f'{label} {value:.2f} {unit}')
            if measurements:
                self.metrics.setText(self.metrics.text()+'\n'+' · '.join(measurements))
            self.metrics.setToolTip('Validation failure causes: '+', '.join(
                f'{label} {100*record.get(key,0):.1f}%' for key,label in
                [('nonfinite_rate','nonfinite state'),('position_limit_rate','position limit'),('speed_limit_rate','speed limit')])+
                '\nImpact speed and time are means over valid hits. Maximum drone travel is the mean of each trial’s peak distance, including PID recovery.')
            if saved.get('fullstate_execution',{}).get('enabled'):
                self.metrics.setToolTip(self.metrics.toolTip().replace('including PID recovery','during the frozen whip; recovery is not scored'))
            if self.active and record['record_id'] != self.last_record:
                try:
                    if self.viewport.load_trial(self.directory / record['replay'], record):
                        self.last_record = record['record_id']
                except (OSError, ValueError, KeyError) as error:
                    self.viewport.status.setText(f'Could not load validation trial: {error}')
        else:
            self.metrics.setText('Validation: — valid hit · — hit + recovery · — task return')
        if 'nonfinite_rate' in status:
            self.metrics.setText(self.metrics.text()+'\nLast training batch: '+', '.join(
                f'{label} {100*status.get(key,0):.1f}%' for key,label in
                [('nonfinite_rate','nonfinite'),('position_limit_rate','position limit'),('speed_limit_rate','speed limit')]))

    def choose_resume_checkpoint(self, checkpoint):
        directory = str(Path(checkpoint).parent.parent)
        index = self.runs.findData(directory)
        if index < 0:
            self.job.status.setText('Refresh the run list before continuing this policy.')
            return
        self.runs.setCurrentIndex(index)
        self.resume_checkpoint = Path(checkpoint)
        self.resume_button.setText('Continue selected checkpoint')
        self.resume_note.setText(f'Continue from: {Path(checkpoint).name} · {self.runs.currentText()}. Set a new run name and total target attempts, then click Continue.')
        self.tabs.setCurrentIndex(0)
        self.refresh_results()

    def start_training(self, resume):
        try:
            if self.job.running:
                raise ValueError('Stop the current training job before starting another.')
            checkpoint = (self.resume_checkpoint or self.directory / 'checkpoints/latest.pt') if resume and self.directory else None
            if resume:
                import torch
                if checkpoint is None or not checkpoint.is_file():
                    raise ValueError('Select an existing checkpoint to continue.')
                payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
                if self.episodes.value() <= int(payload.get('episodes', 0)):
                    raise ValueError('Set total target attempts above the selected checkpoint count to continue.')
            overrides = {key: int(spin.value()) if key in ('update_epochs', 'updates_per_collection') else spin.value()
                         for key, spin in self.hyperparameters.items()}
            directory, command = prepare_training(self.root, self.algorithm, seed=self.seed.value(),
                episodes=self.episodes.value(), batch=self.batch.value(), device=self.device.currentText(),
                overrides=overrides, resume=checkpoint, run_name=self.run_name.text())
            self.job.start(directory, command)
            self.directory = directory
            self.refresh()
            self.job.status.setText('Training uses the saved baseline and task. Validation runs after each batch.')
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            self.job.status.setText(str(error))

    def stop_training(self):
        directory = self.directory
        if directory:
            (directory / 'STOP_REQUESTED').touch()
            self.job.status.setText('Stopping · unfinished batch will be discarded; last saved update is preserved')
            self.stop_button.setEnabled(False)

    def run_latest(self):
        try:
            self.viewport.start_latest(self.directory)
            self.last_record = None
            self.refresh_results()
        except (OSError, ValueError) as error:
            self.viewport.status.setText(str(error))

    def export_figures(self):
        path = QFileDialog.getExistingDirectory(self, 'Export learning figures and source CSV')
        if path:
            try:
                export_learning(self.directory, path, self.algorithm)
                self.job.status.setText('Exported PDF, SVG, 600 dpi PNG, source CSV and figure metadata.')
            except (OSError, ValueError) as error:
                self.job.status.setText(str(error))

    def shutdown(self):
        if self.viewport.thread is not None:
            self.viewport.stop_flight()
            return False
        self.timer.stop()
        self.viewport.timer.stop()
        if self.viewport.viewer is not None:
            self.viewport.viewer.close()
        return True
