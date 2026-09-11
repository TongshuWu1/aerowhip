import os

from experimental_data.io import atomic_json
from simulator.gui.ppo_training_dashboard import progress_snapshot


def test_plateau_tracks_meaningful_evaluation_improvement_not_training_score():
    settings={'training':dict(relative_improvement=.005,minimum_attempts=400,plateau_attempts=300,maximum_attempts=1000)}
    rows=[dict(attempts=100,evaluation_reward=100,evaluation_success=0),
          dict(attempts=200,evaluation_reward=100.4,evaluation_success=0),
          dict(attempts=300,evaluation_reward=101,evaluation_success=0),
          dict(attempts=400,reward=9000,elapsed_s=20,failures=1)]
    state=progress_snapshot(dict(status='running',attempts=300),settings,rows)
    assert state['attempts']==400  # History can be written before status.
    assert state['since_improvement']==100
    assert state['evaluation_attempts']==300 and state['rate']==20
    rows.append(dict(attempts=600,evaluation_reward=101.4,evaluation_success=0,
                     evaluation_pull_fraction=1,evaluation_release_fraction=0,evaluation_wave_complete_fraction=1))
    state=progress_snapshot(dict(status='completed',attempts=600,stop_reason='reward_plateau'),settings,rows)
    assert state['heading']=='Stopped · evaluation score plateau'
    assert state['since_improvement']==300 and state['success']==0
    assert state['phases']['wave_complete_fraction']==1 and state['phases']['release_fraction']==0


def test_missing_evaluation_or_frozen_settings_never_invents_progress():
    state=progress_snapshot(dict(status='running'),{},[dict(attempts=100,reward=12,success=.5)])
    assert state['success'] is None and state['maximum'] is None
    assert state['since_improvement'] is None and state['elapsed'] is None
    assert all(value is None for value in state['phases'].values())


def test_hit_rate_priority_and_dashboard_patience_agree():
    from planning.ppo_progress import evaluation_better
    cfg={'training':dict(success_priority=True,relative_improvement=.005)}
    rows=[dict(attempts=100,evaluation_reward=100,evaluation_success=0),
          dict(attempts=200,evaluation_reward=50,evaluation_success=.25),
          dict(attempts=300,evaluation_reward=1000,evaluation_success=0)]
    assert evaluation_better(50,.25,100,0,success_priority=True)
    assert not evaluation_better(1000,0,50,.25,success_priority=True)
    state=progress_snapshot(dict(status='running'),cfg,rows)
    assert state['since_improvement']==100
    rows.append(dict(attempts=400,evaluation_reward=51,evaluation_success=.25))
    assert progress_snapshot(dict(status='running'),cfg,rows)['since_improvement']==0


def test_switching_runs_clears_old_metrics_and_log_and_stop_is_read_only(tmp_path):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    from simulator.gui.pva_workspace import PVAPlannerPage
    app=QApplication.instance() or QApplication([])
    complete=tmp_path/'runs/ppo_pva/2-complete'
    pending=tmp_path/'runs/ppo_pva/1-pending'
    for path in (complete,pending):atomic_json(path/'identity.json',dict(name=path.name))
    atomic_json(complete/'status.json',dict(status='completed',attempts=600,stop_reason='reward_plateau'))
    atomic_json(complete/'settings.json',dict(training=dict(maximum_attempts=1000,minimum_attempts=100,plateau_attempts=500)))
    atomic_json(complete/'history.json',[dict(attempts=600,reward=10,success=0,failures=1,elapsed_s=20,
                                           evaluation_reward=10,evaluation_success=0)])
    (complete/'console.log').write_text('completed run log',encoding='utf-8')
    atomic_json(pending/'status.json',dict(status='running',attempts=0))
    before={p:p.read_bytes() for p in tmp_path.rglob('*.json')}
    page=PVAPlannerPage(tmp_path,'ppo');page.show();app.processEvents()
    assert page.tabs.currentIndex()==1 and page.progress_views.currentIndex()==0
    assert page.dashboard.values['attempts'].text()=='600'
    assert 'No valid hits' in page.dashboard.outcome.text() and not page.stop.isEnabled()
    assert page.dashboard.budget.value()==600
    page.log_toggle.setChecked(True)
    assert page.saved_log.toPlainText()=='completed run log'
    page.progress_runs.setCurrentIndex(page.progress_runs.findData(str(pending)))
    assert page.current_run==pending and page.stop.isEnabled()
    assert page.dashboard.values['success'].text()=='—'
    assert not page.dashboard.budget.isEnabled() and not page.figure.axes
    assert 'No console log' in page.saved_log.toPlainText()
    assert all(p.read_bytes()==value for p,value in before.items())
    assert not list(tmp_path.rglob('STOP'))
    page.shutdown();page.close();app.processEvents()
