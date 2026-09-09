import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication
from simulator.gui.drone_residual_page import DroneResidualPage
from experimental_data.io import atomic_json


def test_drone_fit_page_is_idle_and_reports_saved_candidate(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    page=DroneResidualPage(tmp_path)
    assert not page.job.running and not page.open.isEnabled()
    folder=tmp_path/'data/drone_residual_runs/example'
    atomic_json(folder/'settings.json',{})
    atomic_json(folder/'review.json',dict(validation_improved=True,selected_update=160))
    metric=dict(position_rmse_m=.04,maximum_position_error_m=.1)
    atomic_json(folder/'evaluation.json',dict(whip1_003=dict(role='validation',nominal=metric,residual=metric)))
    page.refresh()
    assert page.table.rowCount()==1 and 'not applied' in page.summary.text()
    assert not page.job.running
    page.close();app.processEvents()
