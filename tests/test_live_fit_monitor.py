import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import json
from pathlib import Path
import pytest
from PySide6.QtWidgets import QApplication
from simulator.gui import fit_monitor
from simulator.gui.pva_model_page import PVAModelPage


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def full_job(root, name, update=7):
    job=root/'runs/adaptation'/name
    write(job/'protocol.json', dict(schema='prospective_whip_adaptation_v1', takes={'whip':{}},
          full_update=dict(parent_id='M1',candidate_id='M2',
          residual_stopping=dict(minimum=40,patience=6,relative=.005,check_every=5,ceiling=None))))
    write(job/'status.json', dict(status='running',stage='cable_residual',update=update,best_loss=8.))
    write(job/'progress.json', dict(stage='obsolete_stage',update=999))
    write(job/'cable_residual/history.json', [dict(update=5,loss=8.2,selection_loss=8.,best_loss=8.,baseline_loss=10.)])
    return job


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def test_full_job_live_history_refresh_and_honest_uncapped_progress(tmp_path, app):
    job=full_job(tmp_path,'M2-current')
    page=PVAModelPage(tmp_path)
    try:
        assert page.jobs.currentData()==str(job)
        assert page.tabs.currentIndex()==1
        assert 'cable residual' in page.fit_status.text()
        assert 'Update 7' in page.fit_detail.text() and '999' not in page.fit_detail.text()
        assert 'No residual update or time cap' in page.stopping_note.text()
        assert page.fit_progress.isHidden()
        assert page.fit_stop.isHidden()
        page.stop();assert not (job/'STOP').exists()
        ax=page.figure.axes[3]
        assert list(ax.lines[0].get_ydata())==[10.,8.]
        write(job/'cable_residual/history.json', [dict(update=5,loss=8.2,selection_loss=8.,best_loss=8.,baseline_loss=10.),
              dict(update=10,loss=7.2,selection_loss=7.,best_loss=7.,baseline_loss=10.)])
        write(job/'status.json',dict(status='running',stage='cable_residual',update=11,best_loss=7.))
        page.poll()
        assert list(page.figure.axes[3].lines[0].get_ydata())==[10.,8.,7.]
        assert '30.0%' in page.stage_cards[3][2].text()
        assert '11 updates' in page.stage_cards[3][2].text()
        page.canvas.draw()
    finally:
        page.shutdown();page.close()


def test_manual_job_selection_survives_automatic_discovery(tmp_path, app):
    old=full_job(tmp_path,'older')
    page=PVAModelPage(tmp_path)
    try:
        new=full_job(tmp_path,'newer')
        os.utime(new/'status.json',ns=(10**18,10**18))
        os.utime(old/'status.json',ns=(10**17,10**17))
        page.poll(force=True)
        assert page.jobs.currentData()==str(new)
        page.jobs.setCurrentIndex(page.jobs.findData(str(old)))
        page.select_job(page.jobs.currentIndex())
        page.poll(force=True)
        assert not page.follow_fit.isChecked() and page.jobs.currentData()==str(old)
    finally:
        page.shutdown();page.close()


def test_selected_checkpoint_and_legacy_formats_remain_distinct(tmp_path):
    job=full_job(tmp_path,'full')
    write(job/'cable_residual/result.json',dict(baseline_loss=10.,best_loss=10.,selected_update=0,updates=10,
          numerically_verified=True,stop_reason='practical_plateau'))
    write(job/'status.json',dict(status='completed'))
    stage=fit_monitor.snapshot(job)['stages'][3]
    assert stage['best']==10. and stage['selected_update']==0 and stage['reduction']==0.
    assert stage['points'][-1][2]==8.  # Optimizer best is not mislabelled as selected.
    old=tmp_path/'runs/adaptation/legacy'
    write(old/'protocol.json',dict(schema='preliminary_pva_bootstrap_v1'))
    write(old/'status.json',dict(status='running'))
    write(old/'progress.json',dict(stage='drone',update=3,ceiling=100))
    write(old/'drone/history.json',[dict(update=3,loss=2.)])
    view=fit_monitor.snapshot(old)
    assert view['stop_supported'] and view['stages'][1]['points']==[(3.,2.,None)]
    (old/'drone/history.json').write_text('[',encoding='utf-8')
    assert fit_monitor.snapshot(old)['stages'][1]['points']==[]
    assert len(fit_monitor.discover(tmp_path))==2
