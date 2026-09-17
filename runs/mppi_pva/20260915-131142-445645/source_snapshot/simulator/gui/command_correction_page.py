"""Live, read-only comparison of command correction artifacts."""
from pathlib import Path
import numpy as np
from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QCheckBox, QGroupBox, QPlainTextEdit)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from .research_widgets import note, style_axes
from . import correction_monitor as monitor

COLORS = dict(previous='#d97706', corrected='#2563eb', reference='#64748b')
LABELS = dict(previous='Previous command', corrected='Corrected command', reference='M0 reference')


def fmt(value, unit=''):
    return f'{value:.3g}{unit}' if value is not None else '—'


class CommandCorrectionPage(QWidget):
    def __init__(self, root):
        super().__init__()
        self.root = Path(root)
        self.last_fingerprint = None
        self.view = None
        body = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel('Correction run'))
        self.jobs = QComboBox();self.jobs.setMinimumWidth(220);row.addWidget(self.jobs, 1)
        self.follow = QCheckBox('Follow active');self.follow.setChecked(True);row.addWidget(self.follow)
        refresh = QPushButton('Refresh');refresh.clicked.connect(lambda:self.poll(force=True));row.addWidget(refresh)
        folder = QPushButton('Open run');folder.clicked.connect(self.open_run);row.addWidget(folder)
        self.open_csv = QPushButton('Open CSV');self.open_csv.clicked.connect(self.open_export);row.addWidget(self.open_csv)
        body.addLayout(row)
        self.status = note('');self.status.setObjectName('pipelineBanner');body.addWidget(self.status)
        row = QHBoxLayout();self.cards = []
        for title in ('Tracking objective', 'Tip tracking RMSE', 'Tip error at strike', 'Optimization'):
            card = QGroupBox(title);layout = QVBoxLayout(card)
            value = QLabel('—');value.setStyleSheet('font-size: 19px; font-weight: 600; color: #172033;')
            detail = note('');layout.addWidget(value);layout.addWidget(detail);row.addWidget(card, 1)
            self.cards.append((value, detail))
        body.addLayout(row)
        row = QHBoxLayout();row.addWidget(QLabel('View'))
        self.mode = QComboBox();self.mode.addItems(['Commands', 'Predicted tracking', 'Convergence']);row.addWidget(self.mode)
        self.coordinate = QComboBox();self.coordinate.addItems(['X', 'Y', 'Z']);row.addWidget(self.coordinate)
        self.full = QCheckBox('Include braking and return');row.addWidget(self.full)
        self.original = QCheckBox('Show original M0 command');row.addWidget(self.original)
        row.addStretch();body.addLayout(row)
        self.figure = Figure(layout='constrained', facecolor='white')
        self.canvas = FigureCanvasQTAgg(self.figure);self.canvas.setMinimumHeight(290);body.addWidget(self.canvas, 1)
        self.toolbar = NavigationToolbar2QT(self.canvas, self);body.addWidget(self.toolbar)
        self.explanation = note('');body.addWidget(self.explanation)
        self.strike_note = note('');body.addWidget(self.strike_note)
        self.details = QPlainTextEdit();self.details.setReadOnly(True);self.details.setMaximumHeight(150);self.details.hide()
        show = QPushButton('Run details');show.setCheckable(True);show.toggled.connect(self.details.setVisible)
        body.addWidget(show);body.addWidget(self.details)
        self.jobs.activated.connect(self.select_job)
        self.follow.toggled.connect(lambda:self.poll(force=True))
        self.mode.currentIndexChanged.connect(self.draw)
        self.coordinate.currentIndexChanged.connect(self.draw)
        self.full.toggled.connect(self.draw);self.original.toggled.connect(self.draw)
        self.timer = QTimer(self);self.timer.setInterval(2000);self.timer.timeout.connect(self.poll);self.timer.start()
        self.poll(force=True)

    def select_job(self, _):
        self.follow.setChecked(False);self.poll(force=True)

    def open_run(self):
        if self.jobs.currentData():
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.jobs.currentData()))

    def open_export(self):
        if self.view and self.view['rehearsal']:
            path = self.view['rehearsal']/'fullstate_30hz.csv'
            if path.exists():QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def poll(self, force=False):
        if not force and not self.isVisible():return
        entries = monitor.discover(self.root)
        selected = self.jobs.currentData()
        if self.follow.isChecked():
            active = next((e for e in entries if e['status'].get('status') == 'running'), None)
            if active:selected = str(active['path'])
        self.jobs.blockSignals(True);self.jobs.clear()
        for entry in entries:self.jobs.addItem(entry['label'], str(entry['path']))
        self.jobs.setCurrentIndex(max(0, self.jobs.findData(selected)));self.jobs.blockSignals(False)
        path = self.jobs.currentData()
        if not path:
            self.view = None;self.status.setText('No command correction runs yet')
            self.open_csv.setEnabled(False);self.draw();return
        fingerprint = monitor.fingerprint(self.root, path)
        if not force and fingerprint == self.last_fingerprint:return
        self.last_fingerprint = fingerprint
        self.view = monitor.snapshot(self.root, path)
        v = self.view;status = v['status'];result = v['result'];opt = v['optimization']
        state = status.get('status', 'unavailable')
        self.status.setText(f"{state.upper()} · {status.get('error') or status.get('stage', 'Saved correction')}\n"
                            + f"Previous: {v['previous'].name if v['previous'] else 'unavailable'} → {Path(path).name}")
        initial, final = v['initial'], v['final']
        reduction = 100*(1-final/initial) if initial is not None and initial > 0 and final is not None else None
        self.cards[0][0].setText(f'{fmt(initial)} → {fmt(final)}')
        self.cards[0][1].setText(f'{fmt(reduction, "%")} lower · weighted loss (m²)' if reduction is not None else 'Waiting for saved loss')
        before = result.get('original_reference_tip_rmse_m');after = result.get('corrected_reference_tip_rmse_m')
        self.cards[1][0].setText(f'{fmt(before*100 if before is not None else None)} → {fmt(after*100 if after is not None else None)} cm')
        self.cards[1][1].setText('Across the whip, relative to M0')
        rows = v['strike_rows']
        self.cards[2][0].setText(f'{fmt(rows[1][1])} → {fmt(rows[2][1])} cm')
        self.cards[2][1].setText(f'Relative to M0 at {fmt(v["strike"], " s")}')
        iterations = opt.get('iterations', len(v['history']))
        self.cards[3][0].setText(f'{iterations} iterations')
        self.cards[3][1].setText(opt.get('stop_reason', 'Running' if state == 'running' else state.capitalize()))
        self.open_csv.setEnabled(bool(v['rehearsal'] and (v['rehearsal']/'fullstate_30hz.csv').exists()))
        self.strike_note.setText('At the planned strike time, distance to target centre: ' + ' · '.join(
            f'{name}: {fmt(distance, " cm")}' for name, _, distance in rows))
        weights = v['settings'].get('trajectory_objective', {}).get('weights', {})
        detail = [f'Run: {path}', f'Previous commands: {v["previous"]}',
                  f'Corrected export: {v["rehearsal"] or "Not exported"}',
                  f'Convergence history: {v["history_path"]}', f'Objective weights: {weights}',
                  f'Strike guard: {opt.get("strike_guard", v["settings"].get("correction", {}).get("optimizer", {}).get("strike_guard", "Unavailable"))}',
                  f'Rollouts: {opt.get("rollout_count", "Unavailable")}',
                  f'Full replay complete: {result.get("recovery_prediction_complete", "Not yet verified")}',
                  f'Command duration: {fmt(result.get("total_duration_s"), " s")}',
                  f'Replay backend: {opt.get("final_replay_backend", "See saved run")}',
                  f'CSV SHA-256: {result.get("csv_sha256", "Not exported")}']
        failure = v['provenance'].get('original_failed_check', {}).get('error')
        if failure:detail.append('Earlier check retained in provenance: '+failure)
        detail.extend(v['warnings']);self.details.setPlainText('\n'.join(detail))
        self.draw()

    def empty(self, ax, text):
        ax.text(.5, .5, text, ha='center', va='center', transform=ax.transAxes,
                color='#64748b', fontsize=9, wrap=True)

    def draw(self, *_):
        self.figure.clear()
        mode = self.mode.currentIndex()
        self.coordinate.setVisible(mode == 0);self.full.setVisible(mode == 0);self.original.setVisible(mode == 0)
        self.strike_note.setVisible(mode == 1)
        if not self.view:
            ax = self.figure.subplots();ax.axis('off');self.empty(ax, 'Saved and active corrections will appear here')
            self.explanation.setText('This tab reads saved artifacts; it does not start optimization or send commands.')
        elif mode == 0:self.draw_commands()
        elif mode == 1:self.draw_predictions()
        else:self.draw_convergence()
        self.toolbar.update();self.canvas.draw_idle()

    def time_mark(self, ax):
        if self.view['strike'] is not None:
            ax.axvline(self.view['strike'], color='#94a3b8', lw=1, ls=':', label='Planned strike')

    def draw_commands(self):
        v = self.view;axes = self.figure.subplots(2, 2).flat;axes = list(axes)
        dimension = self.coordinate.currentIndex();coordinate = self.coordinate.currentText()
        for ax, title, unit in zip(axes[:3], ['Position', 'Velocity', 'Acceleration'], ['m', 'm/s', 'm/s²']):
            ax.set_title(f'{coordinate} · {title}', loc='left', fontsize=10, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=9);ax.set_ylabel(unit, fontsize=9)
        axes[3].set_title('Command path · XZ', loc='left', fontsize=10, fontweight='bold')
        axes[3].set_xlabel('X (m)', fontsize=9);axes[3].set_ylabel('Z (m)', fontsize=9)
        for name, curve in v['commands'].items():
            if curve is None or (name == 'reference' and not self.original.isChecked()):continue
            time, values = curve['time'], curve['values']
            mask = np.ones(len(time), dtype=bool) if self.full.isChecked() or v['whip_end'] is None else time <= v['whip_end']+1e-9
            label = 'Unexported correction' if name == 'corrected' and v['provisional'] else LABELS[name]
            if name == 'reference':label = 'Original M0 command'
            style = '--' if name != 'corrected' else '-'
            for i, ax in enumerate(axes[:3]):
                ax.step(time[mask], values[mask, i*3+dimension], where='post', color=COLORS[name], ls=style, lw=1.5, label=label)
            axes[3].plot(values[mask, 0], values[mask, 2], color=COLORS[name], ls=style, lw=1.5, label=label)
        for ax in axes[:3]:
            if ax.lines:self.time_mark(ax)
        for ax in axes:
            if ax.lines:ax.legend(frameon=False, fontsize=8)
            else:self.empty(ax, 'No saved commands available yet')
        style_axes(axes)
        self.explanation.setText('Desired position, velocity and acceleration sent to the controller · 30 Hz held commands. '
            + ('The unexported preview covers the whip only; recovery is not yet available.' if v['provisional'] else
               'Each full command keeps its own duration; use “Include braking and return” to inspect recovery.'))

    def draw_predictions(self):
        v = self.view;axes = self.figure.subplots(2, 2)
        for col, part in enumerate(('tip', 'vehicle')):
            title = 'Cable tip' if part == 'tip' else 'Vehicle'
            error_ax, path_ax = axes[0, col], axes[1, col]
            error_ax.set_title(title+' error relative to M0', loc='left', fontsize=10, fontweight='bold')
            error_ax.set_xlabel('Time (s)', fontsize=9);error_ax.set_ylabel('3D error (cm)', fontsize=9)
            path_ax.set_title(title+' path · XZ', loc='left', fontsize=10, fontweight='bold')
            path_ax.set_xlabel('X (m)', fontsize=9);path_ax.set_ylabel('Z (m)', fontsize=9)
            reference = v['reference'][part]
            if reference is not None:
                path_ax.plot(reference['values'][:, 0], reference['values'][:, 2], color=COLORS['reference'], ls=':', lw=2, label='M0 reference')
                point = monitor.at_time(reference, v['strike'])
                if point is not None:path_ax.plot(point[0], point[2], 'o', color=COLORS['reference'], ms=5)
            if part == 'tip' and v['target'].shape == (3,):
                path_ax.plot(v['target'][0], v['target'][2], '+', color='#dc2626', ms=10, mew=1.7, label='Virtual target')
            for name in ('previous', 'corrected'):
                curve = v['errors'][name][part]
                if curve is not None:error_ax.plot(curve['time'], curve['values'], color=COLORS[name], lw=1.5, label=LABELS[name])
                pred = v['predictions'][name][part]
                if pred is not None:
                    mask = pred['time'] <= v['whip_end']+1e-9 if v['whip_end'] is not None else np.ones(len(pred['time']), bool)
                    path_ax.plot(pred['values'][mask, 0], pred['values'][mask, 2], color=COLORS[name], lw=1.5, label=LABELS[name])
                    point = monitor.at_time(pred, v['strike'])
                    if point is not None:path_ax.plot(point[0], point[2], 'o', color=COLORS[name], ms=5)
            if error_ax.lines:self.time_mark(error_ax)
            else:self.empty(error_ax, 'Waiting for saved predictions')
        for ax in axes.flat:
            if ax.lines:ax.legend(frameon=False, fontsize=8)
        style_axes(axes)
        self.explanation.setText('Previous and corrected commands are predicted using this correction run’s fitted model. '
            'M0 is the fixed desired motion. Dots mark the planned strike time. Simulation only; no measured hit results.'
            + (' Corrected prediction is not saved yet.' if v['predictions']['corrected']['tip'] is None else ''))

    def draw_convergence(self):
        v = self.view;axes = self.figure.subplots(1, 2)
        ax = axes[0];x = [];y = []
        if v['initial'] is not None:x.append(0);y.append(v['initial'])
        rows = [r for r in v['history'] if monitor.number(r.get('cost_m2')) is not None and monitor.number(r.get('iteration')) is not None]
        x.extend(r['iteration'] for r in rows);y.extend(r['cost_m2'] for r in rows)
        if x:ax.plot(x, y, color=COLORS['corrected'], lw=1.8, label='Retained objective')
        rejected = [r for r in rows if r.get('accepted') is False]
        if rejected:ax.plot([r['iteration'] for r in rejected], [r['cost_m2'] for r in rejected], 'x', color='#dc2626', label='No accepted step')
        if v['result'].get('corrected_cost_m2') is not None:
            ax.plot(v['optimization'].get('iterations', len(rows)), v['final'], '*', ms=11, color='#15803d', label='Exported replay')
        ax.set_title('Tracking objective · lower is better', loc='left', fontsize=10, fontweight='bold')
        ax.set_xlabel('Iteration', fontsize=9);ax.set_ylabel('Weighted loss (m²)', fontsize=9)
        if ax.lines:ax.legend(frameon=False, fontsize=8)
        else:self.empty(ax, 'Waiting for saved optimization history')
        ax = axes[1]
        optimal = [r for r in rows if monitor.number(r.get('command_feasible_optimality')) is not None]
        if optimal:
            values = [r['command_feasible_optimality'] for r in optimal]
            ax.plot([r['iteration'] for r in optimal], values, color='#7c3aed', lw=1.5)
            if min(values) > 0:ax.set_yscale('log')
        else:self.empty(ax, 'No saved stationarity diagnostic')
        tolerance = v['settings'].get('correction', {}).get('optimizer', {}).get('optimality_tolerance')
        if monitor.number(tolerance) is not None and tolerance > 0:
            ax.axhline(tolerance, color='#64748b', ls='--', label='Stopping threshold');ax.legend(frameon=False, fontsize=8)
        ax.set_title('Convergence check', loc='left', fontsize=10, fontweight='bold')
        ax.set_xlabel('Iteration', fontsize=9);ax.set_ylabel('Projected gradient (scaled units)', fontsize=9)
        style_axes(axes)
        self.explanation.setText('Saved optimization history · refreshes every 2 s. A flat loss or iteration limit alone does not establish convergence. '
            + ('History comes from the optimization run reused for this export.' if v['history_path'] != v['path'] else ''))

    def shutdown(self):
        self.timer.stop()
