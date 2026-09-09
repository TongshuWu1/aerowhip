"""Preliminary recordings, physical inputs, and explicit DDER baseline fitting."""
from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QPushButton, QDoubleSpinBox, QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QCheckBox, QFileDialog, QScrollArea, QSplitter, QLabel, QMessageBox)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from experimental_data.io import atomic_json
from simulator.workflow import read_json, import_take, apply_baseline, prepare_data_job
from .research_widgets import note, BackgroundJob


class BaselinePage(QWidget):
    baseline_applied = Signal()

    def __init__(self, root, parent=None):
        super().__init__(parent)
        self.root = Path(root)
        self.model = read_json(self.root / 'config/model.json')
        self.candidate = None
        self.fields = {}
        self.loaded_field_values = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 16, 20, 18)
        outer.addWidget(note('Preliminary recordings establish the simulator before policy deployment. '
                             'Fit on training takes and inspect validation error before applying a baseline.'))
        split = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(split, 1)
        left = QWidget()
        column = QVBoxLayout(left)
        column.setContentsMargins(0, 0, 8, 0)
        toolbar = QHBoxLayout()
        self.import_button = QPushButton('Import take')
        self.import_button.clicked.connect(self.import_recording)
        self.process_button = QPushButton('Process enabled takes')
        self.process_button.clicked.connect(lambda: self.start_job('process'))
        toolbar.addWidget(self.import_button)
        toolbar.addWidget(self.process_button)
        column.addLayout(toolbar)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(['Use', 'Recording', 'Role', 'Duration', 'Valid state'])
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self.inspect_take)
        self.table.setMaximumHeight(295)
        column.addWidget(self.table)
        column.addWidget(note('Unchecked takes are excluded. Raw files are preserved. The protected test take stays separate.'))
        self.take_note = note('Select a recording to inspect its measured tip motion.')
        column.addWidget(self.take_note)
        self.fit_view = QComboBox()
        self.fit_view.addItems(['Validation motion', 'Fitting progress', 'Longer predictions'])
        self.fit_view.currentIndexChanged.connect(self.show_fit_view)
        self.fit_view.setEnabled(False)
        column.addWidget(self.fit_view)
        self.figure = Figure(figsize=(6, 2.7), facecolor='white')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(185)
        column.addWidget(self.canvas, 1)
        self.job = BackgroundJob(root)
        self.job.finished.connect(self.job_finished)
        self.job.progress.connect(self.fit_progress)
        self.last_fit_progress = None
        column.addWidget(self.job)
        split.addWidget(left)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(340)
        right = QWidget()
        details = QVBoxLayout(right)
        details.setContentsMargins(8, 0, 0, 0)
        self.active = note('')
        details.addWidget(self.active)
        mass = QGroupBox('Physical inputs')
        form = QFormLayout(mass)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.add_field(form, ('point_mass', 'mass_kg'), 'Drone point mass', ' kg', .001, 10, 6)
        self.add_field(form, ('cable', 'bare_cable_mass_kg'), 'Bare cable mass', ' kg', .000001, 1, 6)
        self.add_field(form, ('cable', 'EI_n_m2'), 'Stiffness · EI', '', .000000001, 1, 10)
        self.add_field(form, ('cable', 'Cb_n_m2_s'), 'Damping · Cb', '', 0, 1, 10)
        details.addWidget(mass)
        details.addWidget(note('EI [N·m²] · Cb [N·m²·s]. The point represents translation; the low-level controller handles attitude.'))
        geometry_toggle = QPushButton('Cable geometry and marker masses')
        geometry_toggle.setCheckable(True)
        details.addWidget(geometry_toggle)
        geometry = QGroupBox('Measured geometry')
        geo = QFormLayout(geometry)
        geo.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        for axis in range(3):
            self.add_field(geo, ('recorded_data', 'optitrack_to_attachment_offset_body_m', axis),
                           f'Drone → attachment {"XYZ"[axis]}', ' m', -1, 1, 4)
        for index in range(10):
            label = 'Attachment → c1' if index == 0 else f'c{index} → c{index+1}'
            self.add_field(geo, ('cable', 'marker_interval_lengths_m', index), label, ' m', .001, 2, 4)
            self.add_field(geo, ('cable', 'moving_marker_masses_kg', index), f'c{index+1} mass', ' kg', 0, 1, 9)
        details.addWidget(geometry)
        geometry.hide()
        geometry_toggle.toggled.connect(geometry.setVisible)
        details.addWidget(note('10 measured markers → 12 simulation nodes. The first 63 mm interval has one interpolated midpoint.'))
        self.fit_button = QPushButton('Fit physics + neural residual')
        self.fit_button.setObjectName('secondaryButton')
        self.fit_button.clicked.connect(lambda: self.start_job('fit'))
        details.addWidget(self.fit_button)
        self.fit_runs = QComboBox()
        self.fit_runs.addItem('No new fit available', None)
        self.fit_runs.currentIndexChanged.connect(self.select_fit)
        details.addWidget(self.fit_runs)
        self.fit_result = note('No new fit. The existing calibrated parameters are loaded above.')
        details.addWidget(self.fit_result)
        self.fit_figure = Figure(figsize=(3.2, 2.1), facecolor='white')
        self.fit_canvas = FigureCanvasQTAgg(self.fit_figure)
        self.fit_canvas.setMinimumHeight(180)
        self.fit_canvas.hide()
        details.addWidget(self.fit_canvas)
        self.export_fit_button = QPushButton('Export fit figures')
        self.export_fit_button.setEnabled(False)
        self.export_fit_button.clicked.connect(self.export_fit)
        details.addWidget(self.export_fit_button)
        self.use_residual = QCheckBox('Include validated neural correction when applying')
        self.use_residual.setEnabled(False)
        details.addWidget(self.use_residual)
        self.apply_button = QPushButton('Apply physical baseline')
        self.apply_button.setObjectName('primaryButton')
        self.apply_button.clicked.connect(self.apply_model)
        details.addWidget(self.apply_button)
        self.baseline_status = note('Applies to future runs. Each training run saves its own model.')
        details.addWidget(self.baseline_status)
        details.addStretch()
        scroll.setWidget(right)
        split.addWidget(scroll)
        split.setSizes([700, 340])
        self.refresh()
        self.refresh_fits()

    def add_field(self, form, key, label, suffix, low, high, decimals):
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(low, high)
        spin.setSuffix(suffix)
        spin.setMinimumWidth(100)
        spin.setMaximumWidth(175)
        value = self.model[key[0]][key[1]]
        spin.setValue(value if len(key) == 2 else value[key[2]])
        spin.setSingleStep(10 ** -min(decimals, 4))
        spin.valueChanged.connect(self.inputs_changed)
        form.addRow(label, spin)
        self.fields[key] = spin
        self.loaded_field_values[key] = spin.value()

    def inputs_changed(self):
        self.candidate = None
        self.fit_result.setText('Inputs edited. Fit again, or apply these values as a manually configured baseline.')
        self.fit_canvas.hide()

    def model_values(self):
        model = deepcopy(self.model)
        for key, spin in self.fields.items():
            # Display precision must not silently round a saved physical model.
            if spin.value() == self.loaded_field_values[key]:
                continue
            if len(key) == 2:
                model[key[0]][key[1]] = spin.value()
            else:
                model[key[0]][key[1]][key[2]] = spin.value()
        return model

    def refresh(self):
        self.manifest = read_json(self.root / 'data/dataset_manifest.json')
        force = read_json(self.root / 'data/force_takes/manifest.json', {})
        self.table.setRowCount(len(self.manifest['takes']))
        for index, (name, row) in enumerate(self.manifest['takes'].items()):
            protected = row['role'] == 'untouched_test'
            enabled = QCheckBox()
            enabled.setChecked(row.get('enabled', True))
            enabled.setEnabled(not protected)
            enabled.toggled.connect(lambda value, name=name: self.change_role(name, enabled=value))
            self.table.setCellWidget(index, 0, enabled)
            self.table.setItem(index, 1, QTableWidgetItem(name))
            roles = QComboBox()
            roles.addItem('Fit', 'training')
            roles.addItem('Validation', 'validation')
            if protected:
                roles.addItem('Protected test', 'untouched_test')
            roles.setCurrentIndex(roles.findData(row['role']))
            roles.setEnabled(not protected)
            roles.currentIndexChanged.connect(lambda _, name=name, box=roles: self.change_role(name, role=box.currentData()))
            self.table.setCellWidget(index, 2, roles)
            data = force.get('takes', {}).get(name, {})
            self.table.setItem(index, 3, QTableWidgetItem(f'{data["duration_s"]:.1f} s' if data else '—'))
            self.table.setItem(index, 4, QTableWidgetItem(f'{100*data["state_valid_fraction"]:.1f}%' if data else '—'))
        baseline = read_json(self.root / 'config/baseline.json', {})
        self.active.setText('Active baseline: ' + baseline.get('version', 'existing preliminary calibration'))
        residual=read_json(self.root/'config/model.json',{}).get('motion_residual',{}).get('enabled',False)
        self.active.setText(self.active.text()+(' · Cable NN on' if residual else ' · Cable NN off'))
        if self.table.rowCount():
            self.table.selectRow(0)

    def change_role(self, name, **changes):
        manifest = read_json(self.root / 'data/dataset_manifest.json')
        if manifest['takes'][name]['role'] == 'untouched_test':
            return
        manifest['takes'][name].update(changes)
        atomic_json(self.root / 'data/dataset_manifest.json', manifest)
        self.candidate = None
        self.fit_result.setText('Data selection changed. Fit again to evaluate this selection.')

    def import_recording(self):
        path = QFileDialog.getExistingDirectory(self, 'Select a take folder containing logger and Motive CSV files')
        if path:
            try:
                import_take(self.root, path)
                self.refresh()
            except (OSError, ValueError) as error:
                QMessageBox.warning(self, 'Import take', str(error))

    def inspect_take(self):
        index = self.table.currentRow()
        if index < 0 or self.table.item(index, 1) is None:
            return
        name = self.table.item(index, 1).text()
        row = self.manifest['takes'][name]
        self.take_note.setText(row.get('note', ''))
        self.figure.clear()
        ax = self.figure.subplots()
        if row['role'] == 'untouched_test':
            ax.text(.5, .5, 'Protected test recording', ha='center', transform=ax.transAxes)
        else:
            try:
                import numpy as np
                with np.load(self.root / 'data/force_takes' / name / 'take.npz') as data:
                    positions = data['cable_node_position_world_m'].copy()
                    positions[~data['state_valid']] = np.nan
                    time = data['time_s']
                    for axis, color in enumerate(('#2563b8', '#da7822', '#268271')):
                        ax.plot(time[::5], positions[::5, -1, axis], color=color, linewidth=.9, label='XYZ'[axis])
                    ax.legend(frameon=False, ncol=3, fontsize=8)
            except (OSError, KeyError, ValueError):
                ax.text(.5, .5, 'Process this take to inspect its motion', ha='center', transform=ax.transAxes)
        ax.set(xlabel='Time [s]', ylabel='Measured tip\nposition [m]')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def start_job(self, kind):
        try:
            self.job_kind = kind
            directory, command = prepare_data_job(self.root, kind, self.model_values())
            self.job.start(directory, command)
            if kind == 'fit':
                self.fit_view.setEnabled(True)
                self.fit_view.setCurrentIndex(1)
            self.table.setEnabled(False)
            self.fit_runs.setEnabled(False)
            for field in self.fields.values():
                field.setEnabled(False)
            for button in (self.fit_button, self.process_button, self.import_button, self.apply_button):
                button.setEnabled(False)
        except (OSError, ValueError) as error:
            self.job.status.setText(str(error))

    def job_finished(self, code):
        self.table.setEnabled(True)
        self.fit_runs.setEnabled(True)
        for field in self.fields.values():
            field.setEnabled(True)
        for button in (self.fit_button, self.process_button, self.import_button, self.apply_button):
            button.setEnabled(True)
        self.refresh()
        if code == 0 and self.job_kind == 'fit':
            self.refresh_fits()
            self.fit_runs.setCurrentIndex(self.fit_runs.findData(str(self.job.directory)))
            self.select_fit()

    def refresh_fits(self):
        parent = self.root / 'data/workflow_jobs'
        jobs = sorted(parent.glob('*/fit_result.json'), reverse=True) if parent.exists() else []
        selected = self.fit_runs.currentData()
        self.fit_runs.blockSignals(True)
        self.fit_runs.clear()
        self.fit_runs.addItem('Select a completed fit', None)
        for path in jobs:
            self.fit_runs.addItem(path.parent.name, str(path.parent))
        if selected:
            self.fit_runs.setCurrentIndex(max(0, self.fit_runs.findData(selected)))
        self.fit_runs.blockSignals(False)

    def select_fit(self):
        selected = self.fit_runs.currentData()
        if selected:
            self.candidate = Path(selected)
            # Restore the candidate's physical inputs so its application is explicit.
            source_model = read_json(self.candidate / 'model.json')
            self.model = source_model
            for key, spin in self.fields.items():
                value = source_model[key[0]][key[1]]
                spin.blockSignals(True)
                spin.setValue(value if len(key) == 2 else value[key[2]])
                self.loaded_field_values[key] = spin.value()
                spin.blockSignals(False)
            result = read_json(self.candidate / 'fit_result.json')
            params = result['fitted_parameters']
            self.fit_result.setText(f'Candidate: EI {params["EI_n_m2"]:.5g} N·m² · Cb {params["Cb_n_m2_s"]:.5g} N·m²·s. '
                                   'Review the errors, then apply.')
            before = result['validation']['before']['equal_take_marker_rmse_m']
            after = result['validation']['after']['equal_take_marker_rmse_m']
            assessment = 'Validation improved' if after < before else 'No validation improvement'
            solver = result.get('solver', {})
            residual = result.get('residual', {})
            self.use_residual.setChecked(False)
            self.use_residual.setEnabled(residual.get('validation_improved_at_all_horizons', False))
            self.fit_result.setText(self.fit_result.text() +
                f'\n{assessment}: {before*1000:.2f} → {after*1000:.2f} mm marker RMSE. '
                f'{solver.get("dtype", "")} / {solver.get("damping_backend", "")}.')
            if 'hybrid' in result['validation']:
                hybrid = result['validation']['hybrid']['equal_take_marker_rmse_m'] * 1000
                self.fit_result.setText(self.fit_result.text() + f'\nPhysics + NN: {hybrid:.2f} mm. '
                    + ('Longer validation rollouts also improved.' if self.use_residual.isEnabled()
                       else 'Neural correction is not ready to apply; inspect longer rollouts.'))
            self.fit_figure.clear()
            axes = self.fit_figure.subplots()
            phases = [(-.24, 'before', '#94a3b8'), (0, 'after', '#268271')]
            if 'hybrid' in result['validation']:
                phases.append((.24, 'hybrid', '#7c3aed'))
            for offset, phase, color in phases:
                values = [result[role][phase]['equal_take_marker_rmse_m'] * 1000 for role in ('training', 'validation')]
                axes.bar([offset, 1 + offset], values, .23, color=color,
                         label={'before': 'Active', 'after': 'Physics', 'hybrid': 'Physics + NN'}[phase])
            axes.set_xticks([0, 1], ['Fit takes', 'Validation'])
            axes.set_ylabel('Marker RMSE [mm]', fontsize=8)
            axes.legend(frameon=False, fontsize=8)
            axes.spines[['top', 'right']].set_visible(False)
            self.fit_figure.tight_layout()
            self.fit_canvas.show()
            self.fit_canvas.draw_idle()
            self.export_fit_button.setEnabled(True)
            self.fit_view.setEnabled(True)
            self.fit_view.setCurrentIndex(0)
            self.show_fit_motion()
        else:
            self.candidate = None
            self.use_residual.setEnabled(False)
            self.use_residual.setChecked(False)
            self.fit_canvas.hide()
            self.export_fit_button.setEnabled(False)

    def show_fit_motion(self):
        import json
        import numpy as np
        path = self.candidate / 'validation_window.npz'
        if not path.exists():
            return
        with np.load(path) as data:
            metadata = json.loads(str(data['metadata']))
            self.figure.clear()
            axes = self.figure.subplots(1, 3)
            for index, ax in enumerate(axes):
                for key, label, color, style in (('measured_markers_m', 'Measured', '#172033', '-'),
                        ('active_markers_m', 'Active', '#94a3b8', ':'),
                        ('candidate_markers_m', 'Candidate', '#268271', '--')):
                    ax.plot(data['time_s'], data[key][:, -1, index], style, color=color, linewidth=1.2, label=label)
                if 'hybrid_markers_m' in data:
                    ax.plot(data['time_s'], data['hybrid_markers_m'][:, -1, index], '-.',
                            color='#7c3aed', linewidth=1.2, label='Physics + NN')
                ax.set_title(f'Tip {"XYZ"[index]} [m]', fontsize=9)
                ax.set_xlabel('Time [s]', fontsize=8)
                ax.spines[['top', 'right']].set_visible(False)
                ax.tick_params(labelsize=8)
            axes[0].legend(frameon=False, fontsize=7)
            self.figure.tight_layout()
            self.canvas.draw_idle()
            self.take_note.setText(f'First validation window · {metadata["take"]} · start frame {metadata["start_frame"]}. '
                                   'Measured attachment motion drives both predictions.')
            offset = (data['active_markers_m'][0, -1] - data['measured_markers_m'][0, -1]) * 1000
            self.take_note.setText(self.take_note.text() +
                f' Initial tip shift from enforcing cable lengths: X {offset[0]:+.1f}, Y {offset[1]:+.1f}, Z {offset[2]:+.1f} mm.')

    def fit_progress(self, progress):
        identity = (progress.get('stage'), progress.get('update'))
        if identity != self.last_fit_progress:
            self.last_fit_progress = identity
            if self.fit_view.currentIndex() == 1:
                self.show_fit_view()

    def show_fit_view(self):
        if not hasattr(self, 'job'):
            return
        directory = self.job.directory if self.job.running else self.candidate
        if directory is None:
            return
        choice = self.fit_view.currentIndex()
        if choice == 0:
            if self.candidate:
                self.show_fit_motion()
            return
        self.figure.clear()
        if choice == 1:
            axes = self.figure.subplots(1, 2)
            for ax, stage in zip(axes, ('physics', 'residual')):
                history = read_json(directory / f'{stage}_history.json', [])
                rows = [r for r in history if 'validation_marker_rmse_m' in r]
                for role, color in (('training', '#2563b8'), ('validation', '#d97706')):
                    ax.plot([r['update'] for r in rows],
                        [1000*r[f'{role}_marker_rmse_m'] for r in rows], color=color,
                        label=role.title(), linewidth=1.3)
                selection = read_json(directory / f'{stage}_selection.json', {})
                selected = next((r for r in rows if abs(r.get('training_objective', float('inf'))
                    - selection.get('selected_training_objective', float('-inf'))) < 1e-12), None)
                if selected:
                    ax.axvline(selected['update'], color='#64748b', linestyle=':', label='Selected')
                ax.set_title('Physical parameters' if stage == 'physics' else 'Neural correction', fontsize=9)
                ax.set_xlabel('Optimizer update', fontsize=8)
                ax.set_ylabel('Mean take RMSE [mm]', fontsize=8)
                ax.legend(frameon=False, fontsize=7)
            self.take_note.setText('One-second predictions on complete fit and validation sets. '
                                   'Weights are selected using fit data; validation remains separate.')
        else:
            ax = self.figure.subplots()
            result = read_json(directory / 'fit_result.json', {})
            horizons = result.get('long_horizon', {})
            for stage, label in (('active', 'Active'), ('physics', 'Physics'), ('hybrid', 'Physics + NN')):
                times = sorted(horizons, key=float)
                ax.plot([float(t) for t in times],
                    [horizons[t][stage]['equal_take_marker_rmse_m']*1000 for t in times],
                    'o-', label=label, linewidth=1.3)
            ax.set_xlabel('Prediction window duration [s]', fontsize=8)
            ax.set_ylabel('Validation mean take RMSE [mm]', fontsize=8)
            ax.legend(frameon=False, fontsize=8)
            axes = [ax]
            self.take_note.setText('Each duration uses all available valid windows. '
                                   'All three models use the same windows at each duration.')
        for ax in axes:
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=8)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def export_fit(self):
        import shutil
        import numpy as np
        selected = self.fit_runs.currentData()
        if not selected:
            return
        path = QFileDialog.getExistingDirectory(self, 'Export baseline fitting figures')
        if path:
            try:
                self.candidate = Path(selected)
                self.show_fit_motion()
                destination = Path(path)
                for name, figure in (('marker_rmse', self.fit_figure), ('validation_tip_motion', self.figure)):
                    for extension in ('pdf', 'svg', 'png'):
                        figure.savefig(destination / f'{name}.{extension}', dpi=600, bbox_inches='tight', facecolor='white')
                shutil.copy2(self.candidate / 'fit_result.json', destination / 'source_fit_result.json')
                if (self.candidate / 'figures').exists():
                    shutil.copytree(self.candidate / 'figures', destination / 'identification', dirs_exist_ok=True)
                with np.load(self.candidate / 'validation_window.npz') as data:
                    columns = [data['time_s']]
                    names = ['time_s']
                    keys = ['measured_markers_m', 'active_markers_m', 'candidate_markers_m']
                    if 'hybrid_markers_m' in data:
                        keys.append('hybrid_markers_m')
                    for key in keys:
                        for marker in range(10):
                            for axis in range(3):
                                columns.append(data[key][:, marker, axis])
                                names.append(f'{key}_c{marker+1}_{"xyz"[axis]}')
                    np.savetxt(destination / 'source_validation_window.csv', np.column_stack(columns),
                               delimiter=',', header=','.join(names), comments='')
                self.baseline_status.setText('Exported fitting figures with measured and simulated source data.')
            except (OSError, ValueError) as error:
                self.baseline_status.setText(str(error))

    def apply_model(self):
        try:
            version = apply_baseline(self.root, self.model_values(), fit_directory=self.candidate,
                                     include_residual=self.use_residual.isChecked())
            self.model = read_json(self.root / 'config/model.json')
            for key, spin in self.fields.items():
                value = self.model[key[0]][key[1]]
                spin.blockSignals(True)
                spin.setValue(value if len(key) == 2 else value[key[2]])
                self.loaded_field_values[key] = spin.value()
                spin.blockSignals(False)
            self.candidate = None
            self.baseline_status.setText(f'Applied baseline {version}. Future runs will use it.')
            self.active.setText(f'Active baseline: {version}')
            self.baseline_applied.emit()
        except (OSError, ValueError) as error:
            self.baseline_status.setText(str(error))
