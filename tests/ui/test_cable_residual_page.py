import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
from PySide6.QtWidgets import QApplication
from simulator.gui.cable_residual_page import CableResidualPage
from experimental_data.io import atomic_json


def test_training_is_explicit_and_passes_update_budget(tmp_path, monkeypatch):
    from simulator.gui import cable_residual_page
    app=QApplication.instance() or QApplication([])
    page=CableResidualPage(tmp_path)
    assert not page.job.running and not page.apply.isEnabled()
    launches=[]
    prepared=[]
    job=tmp_path/'job';job.mkdir()
    def prepare(root,updates):
        prepared.append((root,updates))
        return job,['python','offline-test-only']
    monkeypatch.setattr(cable_residual_page,'prepare_residual_job',prepare)
    monkeypatch.setattr(page.job,'start',lambda directory,command:launches.append((directory,command)))
    page.updates.setValue(7)
    page.train.click()
    assert prepared==[(tmp_path,7)] and len(launches)==1
    assert not page.train.isEnabled() and page.stop.isEnabled()
    assert not (tmp_path/'config/model.json').exists()
    page.close();app.processEvents()


def test_learned_drag_result_is_distinguished_from_historical_fit(tmp_path):
    app=QApplication.instance() or QApplication([])
    job=tmp_path/'data/cable_residual_runs/example'
    atomic_json(job/'settings.json',dict(updates=24))
    atomic_json(job/'review.json',dict(accepted=False,selected_update=24,
        learned_drag=True,initial_drag_s_inv=.3,fitted_drag_s_inv=.27))
    atomic_json(job/'evaluation.json',{})
    page=CableResidualPage(tmp_path)
    assert 'learned drag 0.3 → 0.27 s⁻¹' in page.summary.text()
    assert not page.apply.isEnabled() and not page.job.running
    page.close();app.processEvents()


def test_rejected_candidate_cannot_be_applied(tmp_path):
    app=QApplication.instance() or QApplication([])
    job=tmp_path/'data/cable_residual_runs/example'
    atomic_json(job/'settings.json',dict(updates=24))
    atomic_json(job/'review.json',dict(accepted=False,selected_update=0))
    metric=dict(equal_take_marker_rmse_m=.02,equal_take_tip_rmse_m=.03)
    atomic_json(job/'evaluation.json',{'2.0':{kind:{role:metric for role in ('training','validation')}
                                             for kind in ('physics','residual')}})
    page=CableResidualPage(tmp_path)
    assert page.runs.count()==1 and page.table.rowCount()==2
    assert not page.apply.isEnabled()
    assert 'Not accepted' in page.summary.text()
    page.close();app.processEvents()
