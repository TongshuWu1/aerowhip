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
        with (self.directory / 'console.log').open('ab') as stream:
            self.process = subprocess.Popen(command, cwd=self.root, stdout=stream, stderr=subprocess.STDOUT,
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
    if directory is None:
        return []
    try:
        with (Path(directory) / 'validation_history.csv').open(newline='', encoding='utf-8') as stream:
            return list(csv.DictReader(stream))
    except OSError:
        return []


def draw_learning(figure, rows, algorithm):
    figure.clear()
    axes = figure.subplots(2, 1, sharex=True)
    color = '#2563b8' if algorithm == 'PPO' else '#da7822'
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2, linewidth=.6)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel('Validation\nsuccess [%]', fontsize=9)
    axes[1].set_ylabel('Mean task return', fontsize=9)
    axes[1].set_xlabel('Simulated training attempts', fontsize=9)
    axes[0].set_ylim(0, 100)
    if rows:
        x = [int(row['training_episodes']) for row in rows]
        for key, label, style in [('success_rate', 'Valid hit', '-'),
                                  ('hit_and_recovery_rate', 'Hit + recovery', '--')]:
            axes[0].plot(x, [100 * float(row[key]) for row in rows], style,
                         color=color, linewidth=1.6, marker='o', markersize=2.5, label=label)
        axes[0].legend(frameon=False, fontsize=8, loc='lower right')
        axes[1].plot(x, [float(row['mean_episode_reward']) for row in rows],
                     color=color, linewidth=1.6, marker='o', markersize=2.5)
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
