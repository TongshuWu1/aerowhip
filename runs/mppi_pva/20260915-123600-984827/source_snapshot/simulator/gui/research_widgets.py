"""Small shared widgets for the five-page research workflow."""
import os
from pathlib import Path
import subprocess

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPlainTextEdit, QPushButton


def note(text):
    label = QLabel(text)
    label.setWordWrap(True)
    label.setObjectName('mutedText')
    return label


def style_axes(axes):
    import numpy as np
    for ax in np.asarray(axes,dtype=object).flat:
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15);ax.tick_params(labelsize=8)


class BackgroundJob(QWidget):
    finished = Signal(int)
    progress = Signal(dict)

    def __init__(self, root, parent=None):
        super().__init__(parent)
        self.root, self.process, self.directory = Path(root), None, None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.status = note('Ready')
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(160)
        self.log.setMaximumHeight(115)
        self.log.hide()
        self.details = QPushButton('Show job log')
        self.details.setCheckable(True)
        self.details.toggled.connect(self.log.setVisible)
        layout.addWidget(self.status)
        layout.addWidget(self.details)
        layout.addWidget(self.log)
        self.timer = QTimer(self)
        self.timer.setInterval(1500)
        self.timer.timeout.connect(self.poll)

    @property
    def running(self):
        return self.process is not None and self.process.poll() is None

    def start(self, directory, command):
        if self.running:
            raise ValueError('A job is already running on this page.')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        environment=os.environ.copy()
        if any(str(arg).endswith('train_ppo_isaaclab.py') for arg in command):
            for key in ('PYTHONPATH','QT_QPA_PLATFORM_PLUGIN_PATH','QT_PLUGIN_PATH','QT_QPA_PLATFORM'):environment.pop(key,None)
        with (self.directory / 'console.log').open('ab') as stream:
            self.process = subprocess.Popen(command, cwd=self.root, env=environment,stdout=stream, stderr=subprocess.STDOUT,
                                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        self.status.setText('Running…')
        self.timer.start()

    def poll(self):
        if self.directory is None:
            return
        path = self.directory / 'console.log'
        if path.exists():
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - 14000))
                text = stream.read().decode('utf-8', errors='replace')
            self.log.setPlainText(text)
            self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
        if self.running:
            import json
            try:
                progress = json.loads((self.directory / 'progress.json').read_text())
                if progress.get('label'):
                    self.status.setText(progress['label'])
                    self.progress.emit(progress)
                elif progress.get('stage') in ('physics', 'residual'):
                    label = 'Fitting physical parameters' if progress['stage'] == 'physics' else 'Learning neural correction'
                    self.status.setText(f'{label} · update {progress["update"]}')
                    self.progress.emit(progress)
            except (OSError, ValueError):
                pass
        if self.process is not None and self.process.poll() is not None:
            code = self.process.returncode
            self.timer.stop()
            self.status.setText('Completed' if code == 0 else 'Job failed — open the log for details')
            self.finished.emit(code)


def load_history(directory):
    import csv
    import json
    if directory is None:
        return []
    full=Path(directory)/'validation_history.jsonl'
    if full.exists():
        rows=[]
        for line in full.read_text(encoding='utf-8').splitlines():
            try:rows.append(json.loads(line))
            except ValueError:continue  # A concurrent writer may be finishing the last line.
        return rows
    try:
        with (Path(directory) / 'validation_history.csv').open(newline='', encoding='utf-8') as stream:
            return list(csv.DictReader(stream))
    except OSError:
        return []


def draw_learning(figure, rows, algorithm):
    figure.clear()
    axes = figure.subplots(2, 2, sharex=True).ravel()
    color = '#2563b8' if algorithm == 'PPO' else '#da7822'
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2, linewidth=.6)
        ax.tick_params(labelsize=8)
    for ax,title,ylabel in zip(axes,['Valid hits','Closest approach','Task return','Failed attempts'],['Success [%]','Median tip distance [cm]','Mean return','Attempts [%]']):
        ax.set_title(title,fontsize=10,loc='left');ax.set_ylabel(ylabel,fontsize=8)
    for ax in axes[2:]:ax.set_xlabel('Training attempts',fontsize=8)
    axes[0].set_ylim(0, 100)
    axes[3].set_ylim(0, 100)
    if rows:
        x = [int(row['training_episodes']) for row in rows]
        import numpy as np
        for ax,key,scale in [(axes[0],'success_rate',100),(axes[1],'median_minimum_tip_distance_m',100),(axes[2],'mean_episode_reward',1)]:
            values=[scale*float(row[key]) if row.get(key) is not None else np.nan for row in rows]
            ax.plot(x,values,color=color,lw=1.7)
            if not np.isfinite(values).any():ax.text(.5,.5,'Not recorded for this run',transform=ax.transAxes,ha='center',fontsize=8)
        for key,label,c in [('numerical_failure_rate','Any model / reference failure','#dc2626'),('reference_infeasible_rate','Reference infeasible','#ea580c')]:
            values=[100*float(row[key]) if row.get(key) is not None else np.nan for row in rows]
            axes[3].plot(x,values,color=c,lw=1.5,label=label)
        axes[3].legend(frameon=False,fontsize=6,loc='upper right')
    else:
        for ax in axes:
            ax.text(.5, .5, 'No validation results yet', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='#64748b')
    figure.set_layout_engine('constrained', w_pad=.04, h_pad=.04, hspace=.07)


def export_learning(directory, destination, algorithm):
    import shutil
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    rows = load_history(directory)
    if not rows:
        raise ValueError('Train a policy before exporting learning curves.')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(3.5, 4.6), facecolor='white')
    FigureCanvasAgg(figure)
    draw_learning(figure, rows, algorithm)
    for extension in ('pdf', 'svg', 'png'):
        figure.savefig(destination / f'{algorithm.lower()}_learning.{extension}', dpi=600,
                       facecolor='white', bbox_inches='tight')
    shutil.copy2(Path(directory) / 'validation_history.csv', destination / 'source_validation_history.csv')
    from experimental_data.io import atomic_json
    atomic_json(destination / 'figure_metadata.json', dict(algorithm=algorithm, run=str(directory),
        smoothing='none', aggregation='single training seed; no confidence interval',
        x_axis='simulated training attempts', y_axis='current-policy deterministic validation',
        validation_history='source_validation_history.csv'))
