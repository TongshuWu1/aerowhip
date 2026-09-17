"""Read-only training dashboard using the selected run's frozen settings."""
import math
from planning.ppo_progress import evaluation_better

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QProgressBar, QSizePolicy
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


def progress_snapshot(status, settings, rows):
    latest = rows[-1] if rows else {}
    evaluations = [r for r in rows if r.get('evaluation_reward') is not None]
    evaluation = evaluations[-1] if evaluations else {}
    attempts = max(status.get('attempts', 0), latest.get('attempts', 0))
    training = settings.get('training', {})
    anchor, anchor_success, last_improvement = -math.inf, -math.inf, 0
    for row in evaluations:
        score = row['evaluation_reward']
        success=row.get('evaluation_success',0.)
        if evaluation_better(score,success,anchor,anchor_success,relative=training.get('relative_improvement',.005),success_priority=training.get('success_priority',False)):
            anchor, anchor_success, last_improvement = score, success, row['attempts']
    state = status.get('status', 'unknown')
    reason = status.get('stop_reason', '')
    heading = {
        'running': 'Training in progress', 'completed': 'Training finished',
        'stopped': 'Training stopped', 'failed': 'Training failed',
        'error': 'Training failed', 'queued': 'Waiting to start',
    }.get(state, 'Waiting for run status')
    if state != 'running' and reason == 'reward_plateau':
        heading = 'Stopped · policy evaluation plateau' if training.get('success_priority') else 'Stopped · evaluation score plateau'
    elif state != 'running' and reason == 'safety_ceiling':
        heading = 'Stopped · review budget reached'
    phases = {key: evaluation.get('evaluation_' + key) for key in (
        'pull_fraction', 'release_fraction', 'wave_complete_fraction', 'success')}
    elapsed = latest.get('elapsed_s')
    return dict(
        heading=heading, state=state, attempts=attempts,
        success=evaluation.get('evaluation_success'), evaluation_attempts=evaluation.get('attempts'),
        evaluation_scenarios=evaluation.get('evaluation_scenarios'),
        tip_contact=evaluation.get('evaluation_tip_contact_fraction'),
        closest_distance=evaluation.get('evaluation_minimum_tip_distance_m'),
        phases=phases, elapsed=elapsed, failures=latest.get('failures'),
        rate=latest.get('attempts', 0) / elapsed if elapsed and elapsed > 0 else None,
        maximum=training.get('maximum_attempts'), minimum=training.get('minimum_attempts'),
        patience=training.get('plateau_attempts'),
        since_improvement=attempts-last_improvement if math.isfinite(anchor) and training else None,
        error=status.get('error', ''), stage=status.get('stage', ''),
    )


def percent(value):
    return '—' if value is None else f'{100 * value:.1f}%'


class PPOTrainingDashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet('''
            QFrame#metricCard { background: white; border: 1px solid #dfe5ee; border-radius: 9px; }
            QLabel#metricValue { font-size: 22pt; font-weight: 700; color: #172b4d; }
            QLabel#dashboardHeading { font-size: 16pt; font-weight: 700; }
            QLabel#dashboardDetail { color: #59677b; font-size: 9pt; }
            QProgressBar { border: none; background: #e2e8f2; border-radius: 5px; min-height: 10px; max-height: 10px; }
            QProgressBar::chunk { background: #367bf5; border-radius: 5px; }
            QProgressBar:disabled { background: #e8ebef; }
        ''')
        body = QVBoxLayout(self); body.setContentsMargins(4, 4, 4, 4); body.setSpacing(7)
        self.heading = QLabel(); self.heading.setObjectName('dashboardHeading')
        self.outcome = QLabel(); self.outcome.setWordWrap(True)
        body.addWidget(self.heading); body.addWidget(self.outcome)
        cards = QHBoxLayout(); self.values = {}
        for key, title in [('attempts', 'Attempts'), ('success', 'Policy test hit rate'),
                           ('elapsed', 'Recorded elapsed'), ('failures', 'Batch infeasible')]:
            card = QFrame(); card.setObjectName('metricCard'); layout = QVBoxLayout(card)
            layout.addWidget(QLabel(title)); value = QLabel('—'); value.setObjectName('metricValue')
            layout.addWidget(value); self.values[key] = value; cards.addWidget(card, 1)
        body.addLayout(cards)
        self.budget_label = QLabel(); self.budget = QProgressBar(); self.budget.setTextVisible(False)
        self.plateau_label = QLabel(); self.plateau = QProgressBar(); self.plateau.setTextVisible(False)
        for label, bar in [(self.budget_label, self.budget), (self.plateau_label, self.plateau)]:
            body.addWidget(label); body.addWidget(bar)
        self.stopping_note = QLabel(); self.stopping_note.setObjectName('dashboardDetail'); self.stopping_note.setWordWrap(True)
        body.addWidget(self.stopping_note)
        self.phase_title = QLabel(); body.addWidget(self.phase_title)
        phase_row = QHBoxLayout(); self.phase_bars = {}; self.phase_labels = {}
        for key, label in [('pull_fraction', 'Forward pull'), ('release_fraction', 'Backward release'),
                           ('wave_complete_fraction', 'Travelling bend'), ('success', 'Valid hit')]:
            col = QVBoxLayout(); caption = QLabel(label + '  —'); col.addWidget(caption)
            bar = QProgressBar(); bar.setTextVisible(False); col.addWidget(bar)
            phase_row.addLayout(col, 1); self.phase_labels[key] = (caption, label); self.phase_bars[key] = bar
        body.addLayout(phase_row)
        self.figure = Figure(layout='constrained', facecolor='#f5f7fb')
        self.canvas = FigureCanvasQTAgg(self.figure); self.canvas.setMinimumHeight(160)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        body.addWidget(self.canvas, 1); self.chart_key = None
        self.detail = QLabel(); self.detail.setObjectName('dashboardDetail'); self.detail.setWordWrap(True)
        body.addWidget(self.detail)
        self.update_run({}, {}, [])

    @staticmethod
    def set_bar(bar, value, total):
        bar.setRange(0, 1000)
        bar.setValue(round(1000 * min(max(value / total, 0), 1)) if value is not None and total else 0)
        bar.setEnabled(value is not None and bool(total))

    def update_run(self, status, settings, rows):
        data = progress_snapshot(status, settings, rows)
        self.heading.setText(data['heading'])
        if data['success'] is None:
            outcome = 'Waiting for the first policy test.'
        elif data['success'] == 0:
            outcome = 'No valid hits in the latest policy test.'
        else:
            outcome = f"Valid hits in {percent(data['success'])} of the latest policy test scenarios."
        if data['error']: outcome += ' ' + str(data['error'])
        if data['tip_contact'] is not None:
            rule='feasible tip entry defines success; wave measures are diagnostics' if settings.get('task',{}).get('success_criterion')=='tip_contact_v1' else 'valid hits require all whip criteria at first contact'
            outcome += f" Tip touched target: {percent(data['tip_contact'])}; {rule}."
        if data['closest_distance'] is not None:
            outcome += f" Mean closest distance: {100*data['closest_distance']:.1f} cm."
        if data['evaluation_scenarios']==1:
            outcome += ' One fixed start/target; this is not a robustness estimate.'
        self.outcome.setText(outcome)
        self.values['attempts'].setText(f"{data['attempts']:,}")
        self.values['success'].setText(percent(data['success']))
        elapsed = data['elapsed']
        self.values['elapsed'].setText('—' if elapsed is None else f'{int(elapsed)//60}m {int(elapsed)%60:02d}s')
        self.values['failures'].setText(percent(data['failures']))
        maximum = data['maximum']; attempts = data['attempts']
        self.budget_label.setText(f'Review budget · {attempts:,} / {maximum:,} attempts  ({100*attempts/maximum:.1f}%)' if maximum else 'Review budget · saved limit unavailable')
        self.set_bar(self.budget, attempts, maximum)
        since, patience = data['since_improvement'], data['patience']
        self.plateau_label.setText(f'Plateau patience · {since:,} / {patience:,} attempts without improvement'
                                   if since is not None and patience else 'Plateau patience · waiting for settings and evaluation')
        self.set_bar(self.plateau, since, patience)
        minimum = data['minimum']
        self.stopping_note.setText((f'Plateau stop eligible after {minimum:,} attempts. ' if minimum else '')
                                  + 'Budget progress is not a measure of learning success.')
        for key, value in data['phases'].items():
            caption, label = self.phase_labels[key]; caption.setText(label + '  ' + percent(value))
            self.set_bar(self.phase_bars[key], value, 1)
        tested_at = data['evaluation_attempts']
        self.phase_title.setText('Policy test' + (f' at {tested_at:,} attempts' if tested_at is not None else '') + ' · motion diagnostics and outcome')
        rate = f"{data['rate']:.0f} attempts/s average · " if data['rate'] is not None else ''
        stage = data['stage'] + ' · ' if data['state'] == 'running' and data['stage'] else ''
        timing = settings.get('policy_timing', {})
        hold = timing.get('control_steps', 1)
        cadence = f'Policy decisions every {hold/30:.3f} s · CSV 30 Hz · ' if hold > 1 else ''
        evaluated = next((row for row in reversed(rows) if 'evaluation_objective_fold' in row), {})
        components = (f"Fold score {evaluated['evaluation_objective_fold']:.1f} · "
                      f"Impact score {evaluated.get('evaluation_objective_impact', 0):.1f} · ") if evaluated else ''
        self.detail.setText(stage + rate + cadence + components + 'Simulation development scenarios · latest saved updates')
        self.draw_trends(rows)

    def draw_trends(self, rows):
        points = tuple((r['attempts'], r.get('evaluation_reward'), r.get('evaluation_success'))
                       for r in rows if r.get('evaluation_reward') is not None)
        if self.chart_key == points: return
        self.chart_key = points; self.figure.clear()
        axes = self.figure.subplots(1, 2)
        for ax, title, column, scale in zip(axes, ['Policy test score', 'Policy test hits'], [1, 2], [1, 100]):
            available = [p for p in points if p[column] is not None]
            ax.set_facecolor('#f5f7fb'); ax.set_title(title, loc='left', fontsize=10)
            if available:
                ax.plot([p[0]/1000 for p in available], [p[column]*scale for p in available],
                        color='#2563eb', lw=2, marker='o', markersize=3)
            else:
                ax.text(.5, .5, 'Waiting for evaluation', ha='center', va='center', transform=ax.transAxes, color='#64748b')
            ax.set_xlabel('Attempts (thousands)', fontsize=8); ax.tick_params(labelsize=8)
            ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', alpha=.15)
            if column == 2: ax.set_ylim(-2, 102); ax.set_ylabel('%', fontsize=8)
        self.canvas.draw_idle()
