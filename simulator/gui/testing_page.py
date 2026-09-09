"""Select, export and rehearse a PPO with drone-only initial measurements."""
from collections import deque
import csv
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PySide6.QtCore import QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QFileDialog, QSplitter, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QHeaderView)

from deployment.package import export_policy
from deployment.rehearsal import assumed_hanging_state
from simulator.workflow import read_json, stamp
from .rehearsal_worker import RehearsalWorker
from .research_widgets import note
from .viewer_3d import create_viewer


class TestingPage(QWidget):
    flight_finished = Signal()
    fullstate_mode = False

    def __init__(self, root):
        super().__init__()
        self.root = Path(root)
        self.viewer = self.worker = self.thread = None
        self.active = False
        self.rendered_target = None
        self.csv_path = self.directory = None
        self.trail = deque(maxlen=6000)
        self.trail_phases = deque(maxlen=6000)
        outer = QVBoxLayout(self)
        self.intro = note('Simulation rehearsal · 10 s hover → assumed hanging cable → force CSV → open-loop execution → normal control')
        outer.addWidget(self.intro)
        row = QHBoxLayout()
        self.checkpoints = QComboBox()
        self.checkpoints.setMinimumContentsLength(35)
        self.checkpoints.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        row.addWidget(QLabel('PPO checkpoint'))
        row.addWidget(self.checkpoints, 1)
        self.refresh_button = QPushButton('Refresh')
        self.refresh_button.clicked.connect(self.refresh_checkpoints)
        row.addWidget(self.refresh_button)
        self.browse = QPushButton('Browse…')
        self.browse.clicked.connect(self.browse_checkpoint)
        row.addWidget(self.browse)
        self.export = QPushButton('Export policy…')
        self.export.clicked.connect(self.export_selected)
        row.addWidget(self.export)
        self.open_policies = QPushButton('Open policy folder')
        self.open_policies.clicked.connect(self.open_policy_folder)
        row.addWidget(self.open_policies)
        outer.addLayout(row)
        coordinates = QHBoxLayout()
        self.start_spins = self.vector_controls(coordinates, 'Attachment hover XYZ (m)', [0,0,1.5])
        self.target_spins = self.vector_controls(coordinates, 'Target XYZ (m)', [1,0,1.4])
        self.apply_target = QPushButton('Apply target')
        self.apply_target.clicked.connect(self.update_target)
        coordinates.addWidget(self.apply_target)
        outer.addLayout(coordinates)
        controller_row = QHBoxLayout()
        controller_row.addWidget(QLabel('Controller mass (kg)'))
        self.controller_mass = QDoubleSpinBox()
        self.controller_mass.setDecimals(6)
        self.controller_mass.setRange(0, 100)
        self.controller_mass.setSpecialValueText('Not configured')
        self.controller_mass.setToolTip('Enter the actual mass used by the controller gravity compensation. Zero exports total thrust only.')
        controller_row.addWidget(self.controller_mass)
        controller_row.addWidget(QLabel('Controller gravity (m/s²)'))
        self.controller_gravity = QDoubleSpinBox()
        self.controller_gravity.setDecimals(6)
        self.controller_gravity.setRange(.001, 100)
        self.controller_gravity.setValue(9.80665)
        self.controller_gravity.setToolTip('Match the gravity constant used by the firmware; common world Z-up frame required.')
        controller_row.addWidget(self.controller_gravity)
        self.save_controller = QPushButton('Save controller settings')
        self.save_controller.clicked.connect(self.save_controller_settings)
        controller_row.addWidget(self.save_controller)
        outer.addLayout(controller_row)
        self.mass_note = note('')
        outer.addWidget(self.mass_note)
        self.assumption = note('GPU rehearsal · ideal OptiTrack-style positions at 100 Hz → estimated velocity → 20 Hz force controller. Cable assumed vertical. Starts suspended 10 cm below hover; ground takeoff and tracking noise/latency are not modeled.')
        outer.addWidget(self.assumption)
        self.controller_legend = QLabel(
            'Trajectory trails: <span style="color:#2563eb">● Normal controller (hover / recovery)</span>'
            ' &nbsp; <span style="color:#f97316">● Open-loop force controller</span>')
        outer.addWidget(self.controller_legend)
        self.controller_mode = QLabel('Active: normal controller')
        self.controller_mode.setStyleSheet('color: #2563eb; font-weight: bold;')
        outer.addWidget(self.controller_mode)
        controls = QHBoxLayout()
        self.start = QPushButton('Start rehearsal')
        self.start.setObjectName('primaryButton')
        self.start.clicked.connect(self.start_rehearsal)
        self.generate = QPushButton('Generate again')
        self.generate.clicked.connect(lambda: self.worker.command('plan') if self.worker else None)
        self.execute = QPushButton('Execute sequence')
        self.execute.clicked.connect(self.execute_sequence)
        self.stop = QPushButton('Stop rehearsal')
        self.stop.clicked.connect(self.stop_rehearsal)
        self.camera = QComboBox()
        self.camera.addItems(['Perspective', 'Side XZ', 'Top XY', 'Front YZ'])
        self.camera.currentTextChanged.connect(lambda text: self.viewer.set_camera_preset(text) if self.viewer else None)
        for widget in (self.start, self.generate, self.execute, self.stop, self.camera):
            controls.addWidget(widget)
        outer.addLayout(controls)
        split = QSplitter()
        self.viewer_host = QWidget()
        self.viewer_layout = QVBoxLayout(self.viewer_host)
        self.viewer_layout.setContentsMargins(0,0,0,0)
        split.addWidget(self.viewer_host)
        csv_panel = QWidget()
        csv_layout = QVBoxLayout(csv_panel)
        self.csv_kind = QComboBox()
        self.csv_kind.setEnabled(False)
        self.csv_kind.currentIndexChanged.connect(self.show_csv_selection)
        csv_layout.addWidget(self.csv_kind)
        self.csv_convention = note('Controller force CSV subtracts the configured drone weight from Fz.')
        csv_layout.addWidget(self.csv_convention)
        self.plan_note = note('The CSV appears here before execution. Commands update at 20 Hz; identical holds may be merged.')
        csv_layout.addWidget(self.plan_note)
        self.table = QTableWidget(0,5)
        self.table.setHorizontalHeaderLabels(['Start (s)', 'End (s)', 'Fx (N)', 'Fy (N)', 'Fz (N)'])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        csv_layout.addWidget(self.table,1)
        self.open_csv = QPushButton('Open CSV')
        self.open_csv.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.csv_path))) if self.csv_path else None)
        csv_layout.addWidget(self.open_csv)
        self.open_controller_csv = QPushButton('Open controller acceleration CSV')
        self.open_controller_csv.setEnabled(False)
        self.open_controller_csv.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(
            str(self.csv_path.parent/'controller_acceleration.csv'))) if self.csv_path else None)
        csv_layout.addWidget(self.open_controller_csv)
        self.open_folder = QPushButton('Open rehearsal folder')
        self.open_folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.directory))) if self.directory else None)
        csv_layout.addWidget(self.open_folder)
        split.addWidget(csv_panel)
        split.setStretchFactor(0,3)
        split.setStretchFactor(1,2)
        split.setSizes([700,430])
        outer.addWidget(split,1)
        self.telemetry = note('Drone XYZ: — · target XYZ: —')
        self.status = note('Select a saved PPO checkpoint to begin.')
        outer.addWidget(self.telemetry)
        outer.addWidget(self.status)
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.tick)
        self.checkpoints.currentIndexChanged.connect(self.selection_changed)
        self.refresh_checkpoints()
        self.set_running(False)
        self.load_controller_settings()
        self.controller_mass.editingFinished.connect(self.save_controller_settings)
        self.controller_gravity.editingFinished.connect(self.save_controller_settings)

    @staticmethod
    def vector_controls(layout, label, values):
        layout.addWidget(QLabel(label))
        spins = []
        for value in values:
            spin = QDoubleSpinBox()
            spin.setRange(-10,10)
            spin.setDecimals(3)
            spin.setSingleStep(.01)
            spin.setValue(value)
            layout.addWidget(spin)
            spins.append(spin)
        return spins

    def refresh_checkpoints(self):
        if self.thread is not None:
            return
        previous = self.checkpoints.currentData()
        self.checkpoints.blockSignals(True)
        self.checkpoints.clear()
        from simulator.policy_library import deleted_checkpoints, checkpoint_key
        deleted = deleted_checkpoints(self.root)
        for run in sorted((self.root/'runs/ppo').glob('*'), reverse=True):
            for checkpoint in sorted((run/'checkpoints').glob('*.pt'),
                                     key=lambda p: ({'best_validation.pt': 0, 'latest.pt': 1}.get(p.name, 2), p.name)):
                if checkpoint_key(self.root, checkpoint) not in deleted and all((run/f'{name}.json').is_file() for name in ('model','task','ppo')):
                    name = read_json(run/'run.json', {}).get('display_name', run.name)
                    self.checkpoints.addItem(f'{name} / {checkpoint.name}', str(checkpoint.resolve()))
        index = self.checkpoints.findData(previous)
        if index >= 0:
            self.checkpoints.setCurrentIndex(index)
        self.checkpoints.blockSignals(False)
        self.selection_changed()

    def browse_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select saved PPO checkpoint', str(self.root/'runs/ppo'), 'PyTorch checkpoint (*.pt)')
        if path:
            self.checkpoints.addItem(Path(path).name, path)
            self.checkpoints.setCurrentIndex(self.checkpoints.count()-1)

    def selection_changed(self):
        checkpoint = self.checkpoints.currentData()
        if checkpoint:
            try:
                task = read_json(Path(checkpoint).parent.parent/'task.json')
                for spins, key in ((self.start_spins,'initial_root_position_m'), (self.target_spins,'target_position_m')):
                    for spin, value in zip(spins, task[key]):
                        spin.setValue(value)
                if self.active:
                    self.reset_viewer()
            except Exception as error:
                self.status.setText(str(error))
        self.start.setEnabled(bool(checkpoint) and self.thread is None)
        self.export.setEnabled(bool(checkpoint) and self.thread is None)
        try:
            model = self.configs()[0]
            cable = model['cable']['bare_cable_mass_kg'] + sum(model['cable']['moving_marker_masses_kg'])
            self.mass_note.setText(f'Selected policy model: drone {1000*model["point_mass"]["mass_kg"]:.2f} g · cable assembly {1000*cable:.2f} g. Controller mass is a separate export setting.')
        except (OSError, ValueError, KeyError):
            self.mass_note.setText('Select a policy with saved model masses.')

    def configs(self):
        checkpoint = self.checkpoints.currentData()
        directory = Path(checkpoint).parent.parent if checkpoint else self.root/'config'
        return [read_json(directory/f'{name}.json') for name in ('model','task','ppo')]

    def reset_viewer(self):
        configs = self.configs()
        target = [spin.value() for spin in self.target_spins]
        q = assumed_hanging_state(configs[0], [s.value() for s in self.start_spins], [0,0,0]).positions_m[0].numpy()
        if self.viewer is None:
            self.viewer = create_viewer(q, target, configs[1]['desired_strike_direction_world'], configs[1]['success']['tip_target_distance_m'], self)
            self.viewer.set_live_flight(True)
            self.viewer_layout.addWidget(self.viewer)
        self.viewer.set_target(target, configs[1]['desired_strike_direction_world'], configs[1]['success']['tip_target_distance_m'])
        self.rendered_target = target
        self.set_controller_visual(False)
        self.viewer.update_state(q, np.zeros(3), q[:1], q[-1:])

    def set_controller_visual(self, force_control):
        name = 'open-loop force controller' if force_control else 'normal controller (hover / recovery)'
        color = '#f97316' if force_control else '#2563eb'
        self.controller_mode.setText(f'Active: {name}')
        self.controller_mode.setStyleSheet(f'color: {color}; font-weight: bold;')
        if self.viewer:
            self.viewer.set_controller_mode(force_control)

    def set_page_active(self, active):
        self.active = active
        if active and self.viewer is None:
            self.reset_viewer()
        if active or self.thread is not None:
            self.timer.start()
        else:
            self.timer.stop()

    def update_target(self):
        if self.worker:
            self.execute.setEnabled(False)
            self.worker.command('target', [s.value() for s in self.target_spins])
        elif self.active:
            self.reset_viewer()

    def export_selected(self):
        try:
            destination = self.root/'policies'/f'Whip-PPO-{stamp()}'
            export_policy(self.root, self.checkpoints.currentData(), destination,
                          experiment_setup=self.experiment_setup())
            self.status.setText(f'Exported policy package and ZIP: {destination}')
        except Exception as error:
            self.status.setText(f'Export failed: {error}')

    def open_policy_folder(self):
        directory = self.root/'policies'
        directory.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def experiment_setup(self):
        return dict(initial_attachment_position_m=[s.value() for s in self.start_spins],
                    target_position_m=[s.value() for s in self.target_spins],
                    controller_export=self.controller_settings(),
                    settling_duration_s=10., initialization='assumed_vertical_cable', simulation_only=True)

    def controller_settings(self):
        if self.controller_mass.value() == 0:
            return {}
        return dict(controller_mass_kg=self.controller_mass.value(),
                    controller_gravity_m_s2=self.controller_gravity.value())

    def load_controller_settings(self):
        path = self.root/'config/controller_export.json'
        if not path.is_file():
            return
        try:
            settings = read_json(path)
            mass = float(settings['controller_mass_kg'])
            gravity = float(settings['controller_gravity_m_s2'])
            if not (np.isfinite(mass) and 0 <= mass <= 100 and
                    np.isfinite(gravity) and .001 <= gravity <= 100):
                raise ValueError('Mass or gravity outside the supported range')
            self.controller_mass.setValue(mass)
            self.controller_gravity.setValue(gravity)
        except Exception as error:
            self.status.setText(f'Could not load controller settings: {error}')

    def save_controller_settings(self):
        path = self.root/'config/controller_export.json'
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            settings = dict(controller_mass_kg=self.controller_mass.value(),
                            controller_gravity_m_s2=self.controller_gravity.value())
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                             prefix='controller_export_', suffix='.tmp', delete=False) as stream:
                json.dump(settings, stream, indent=2)
                stream.write('\n')
                temporary = stream.name
            os.replace(temporary, path)
            self.status.setText('Controller mass and gravity saved for future rehearsals and exports.')
        except Exception as error:
            self.status.setText(f'Could not save controller settings: {error}')

    def set_running(self, running):
        for widget in (self.checkpoints, self.refresh_button, self.browse, self.export, self.start,
                       self.controller_mass, self.controller_gravity, self.save_controller, *self.start_spins):
            widget.setEnabled(not running)
        self.start.setEnabled(not running and bool(self.checkpoints.currentData()))
        self.export.setEnabled(not running and bool(self.checkpoints.currentData()))
        self.generate.setEnabled(False)
        self.execute.setEnabled(False)
        self.stop.setEnabled(running)
        self.open_csv.setEnabled(self.csv_path is not None)
        self.open_folder.setEnabled(self.directory is not None)
        for widget in (*self.target_spins, self.apply_target):
            widget.setEnabled(True)

    def start_rehearsal(self):
        if self.thread is not None:
            return
        try:
            self.directory = self.root/'runs/rehearsals'/stamp()
            self.directory.mkdir(parents=True)
            package = export_policy(self.root, self.checkpoints.currentData(), self.directory/'package',
                                    experiment_setup=self.experiment_setup())
            configs = [read_json(package/'policy'/f'{name}.json') for name in ('model','task','ppo')]
            configs[1]['initial_root_position_m'] = [s.value() for s in self.start_spins]
            configs[1]['target_position_m'] = [s.value() for s in self.target_spins]
            self.flight_task = configs[1]
            (self.directory/'rehearsal_task.json').write_text(json.dumps(configs[1],indent=2)+'\n',encoding='utf-8')
            self.reset_viewer()
            self.csv_path = None
            self.csv_kind.clear()
            self.csv_kind.setEnabled(False)
            self.open_controller_csv.setEnabled(False)
            self.table.setRowCount(0)
            self.trail.clear()
            self.trail_phases.clear()
            self.plan_note.setText('Waiting for 10 seconds of settled hover and planning.')
            self.worker = RehearsalWorker(configs, package/'policy/checkpoints/policy.pt', self.directory,
                                          controller_export=self.controller_settings(),
                                          fullstate_mode=self.fullstate_mode)
            self.thread = QThread(self)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.status.connect(self.status.setText)
            self.worker.failed.connect(self.status.setText)
            self.worker.plan_ready.connect(self.show_plan)
            self.worker.plan_invalidated.connect(lambda: self.execute.setEnabled(False))
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread_done)
            self.set_running(True)
            self.timer.start()
            self.thread.start()
        except Exception as error:
            self.status.setText(f'Could not start rehearsal: {error}')

    def show_plan(self, path, metadata):
        directory = Path(path).parent
        self.plan_metadata = metadata
        self.csv_kind.blockSignals(True)
        self.csv_kind.clear()
        for name, title in (
            ('controller_force.csv', 'Controller force (N) · drone weight removed'),
            ('controller_acceleration.csv', 'Controller acceleration (m/s²) · gravity removed'),
            ('commands.csv', 'Simulation total thrust (N) · includes weight')):
            if (directory/name).is_file():
                self.csv_kind.addItem(title, str(directory/name))
        self.csv_kind.setCurrentIndex(0)
        self.csv_kind.blockSignals(False)
        self.csv_kind.setEnabled(True)
        self.show_csv_selection()
        self.open_controller_csv.setEnabled((directory/'controller_acceleration.csv').is_file())
        self.execute.setEnabled(True)

    def show_csv_selection(self):
        path = self.csv_kind.currentData()
        if not path:
            return
        self.csv_path = Path(path)
        metadata = self.plan_metadata
        acceleration = self.csv_path.name == 'controller_acceleration.csv'
        total = self.csv_path.name == 'commands.csv'
        axes = ['Ax (m/s²)', 'Ay (m/s²)', 'Az (m/s²)'] if acceleration else ['Fx (N)', 'Fy (N)', 'Fz (N)']
        self.table.setHorizontalHeaderLabels(['Start (s)', 'End (s)', *axes])
        if total:
            message = 'Total simulation thrust includes gravity support.'
            if not (self.csv_path.parent/'controller_force.csv').is_file():
                message += ' Controller CSV unavailable: configure controller mass and gravity before starting a new rehearsal.'
        else:
            settings = metadata.get('controller_export', {})
            message = 'Drone gravity compensation removed once; cable support remains.'
            if 'mass_kg' in settings and 'gravity_m_s2' in settings:
                message += f' Subtracted {settings["mass_kg"]*settings["gravity_m_s2"]:.6f} N from total Fz.'
            if acceleration:
                message += ' Then divided by controller mass for cmdFullState.'
        self.csv_convention.setText(message)
        self.open_csv.setText('Open total thrust CSV' if total else
                              'Open acceleration CSV' if acceleration else 'Open controller force CSV')
        with self.csv_path.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.reader(stream))[1:]
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                self.table.setItem(i,j,QTableWidgetItem(f'{float(value):.6f}'))
        self.plan_note.setText(f'Cutoff {metadata["cutoff_s"]:.2f} s · planned in {metadata["planning_wall_seconds"]:.3f} s\nTarget {metadata["target_position_m"]} m\n{self.csv_path.parent.name}/{self.csv_path.name}')
        self.plan_note.setToolTip(str(self.csv_path))
        self.open_csv.setEnabled(True)

    def execute_sequence(self):
        if self.worker:
            self.execute.setEnabled(False)
            self.worker.command('execute')

    def stop_rehearsal(self):
        if self.worker:
            self.worker.command('stop')
            self.stop.setEnabled(False)
            self.status.setText('Stopping rehearsal and saving the recording…')

    def tick(self):
        if self.worker is None:
            return
        frame = self.worker.take_frame()
        if frame is None:
            return
        self.trail.append(frame['positions'][[0,-1]])
        self.trail_phases.append(frame.get('applied_controller_phase', frame['phase']))
        self.set_controller_visual(frame['phase'] == 1)
        self.generate.setEnabled(frame['ready'] and not frame['preparing'] and frame['attempt'] == 0)
        if frame['attempt']:
            for widget in (*self.target_spins, self.apply_target):
                widget.setEnabled(False)
        self.telemetry.setText(f'Drone attachment XYZ: {np.round(frame["positions"][0],3)} m · target: {np.round(frame["target"],3)} m')
        self.status.setText(f'{frame["message"]} · {frame["time_s"]:.2f} s · {frame["realtime_rate"]:.2f}× real time\n{frame["runtime_device"]} · tracking {frame["tracking_hz"]:.1f} Hz · control {frame["control_hz"]:.1f} Hz (wall-clock averages)')
        if self.active and self.viewer:
            trail = np.asarray(self.trail)
            if frame['target'] != self.rendered_target:
                self.viewer.set_target(frame['target'], self.flight_task['desired_strike_direction_world'], self.flight_task['success']['tip_target_distance_m'])
                self.rendered_target = frame['target']
            self.viewer.update_state(frame['positions'],frame['command'],trail[:,0],trail[:,1],
                                     trail_phases=np.asarray(self.trail_phases))

    def thread_done(self):
        self.thread.deleteLater()
        self.thread = self.worker = None
        self.set_running(False)
        if not self.active:
            self.timer.stop()
        self.flight_finished.emit()

    def shutdown(self):
        self.stop_rehearsal()
        if self.thread is None:
            self.timer.stop()
            if self.viewer is not None:
                self.viewer.close()
        return self.thread is None
