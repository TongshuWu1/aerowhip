import json
from pathlib import Path
from PySide6.QtWidgets import QApplication
from simulator.gui.flight_batch_page import FlightBatchPage




def test_stop_boundary_preserves_partial_fit(tmp_path):
    import pytest
    from experimental_data.current_adaptation_fit import note
    (tmp_path/'STOP').touch()
    with pytest.raises(InterruptedError,match='partial results preserved'):
        note(tmp_path,'next update')
    assert not (tmp_path/'progress.json').exists()






def test_replay_defaults_to_raw_and_optional_normalization_never_double_shifts(tmp_path,monkeypatch):
    import numpy as np
    from simulator.gui.adaptation_check_page import AdaptationCheckPage
    app=QApplication.instance() or QApplication([])
    w=AdaptationCheckPage(tmp_path)
    for method in ['set_range','draw_plots','redraw']:monkeypatch.setattr(w,method,lambda:None)
    raw=np.array([[0.,0.,1.3]])
    normalized=np.array([[0.,0.,1.25]])
    d=dict(measured_origin=raw,hover_normalized=dict(measured_origin=normalized),
        height_calibration=dict(bias_z_m=.05,checks_passed=True),alignment=dict(method='shared clock',offset_s=0),
        tracking_span=(-10,11),metadata=dict(whip_end_s=1),take='test',rehearsal=tmp_path)
    w.loaded(d)
    assert not w.normalize_height.isChecked()
    np.testing.assert_array_equal(w.data['measured_origin'],raw)
    w.normalize_height.setChecked(True);np.testing.assert_array_equal(w.data['measured_origin'],normalized)
    w.normalize_height.setChecked(False);np.testing.assert_array_equal(w.data['measured_origin'],raw)
    w.normalize_height.setChecked(True);w.loaded(d)
    np.testing.assert_array_equal(w.data['measured_origin'],raw)
    d.pop('hover_normalized');d.pop('height_calibration');w.loaded(d)
    np.testing.assert_array_equal(w.data['measured_origin'],raw)
    assert not w.normalize_height.isEnabled()
    assert raw[0,2]==1.3
    w.close()


def test_alignment_editor_requires_review_and_preserves_raw(tmp_path):
    app=QApplication.instance() or QApplication([])
    b=tmp_path/'rehearsal_csv_and_result_in_real_flight/policy/adp0'
    (b/'flight_take').mkdir(parents=True);(b/'simulation_csv').mkdir()
    (b/'simulation_csv/fullstate_30hz.csv').write_text('command')
    raw=b/'flight_take/take.csv';log=b/'flight_take/experiment_take.csv'
    raw.write_text('tracking');log.write_text('controller')
    w=FlightBatchPage(tmp_path);w.table.selectRow(0);w.offset.setValue(2.5)
    w.save.click();assert not (b/'time_alignment.json').exists()
    w.source.setText('Shared event measured in both time bases');w.reviewed.setChecked(True);w.save.click()
    data=json.loads((b/'time_alignment.json').read_text())['take']
    assert data['offset_s']==2.5 and not data['clock_verified']
    assert raw.read_text()=='tracking' and log.read_text()=='controller'
    from experimental_data.adaptation_check import recorded_alignment
    assert recorded_alignment(tmp_path,b,'take',raw,log)['offset_s']==2.5
    w.close()
