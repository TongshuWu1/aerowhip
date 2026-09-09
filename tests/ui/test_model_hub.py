import json
from pathlib import Path
from PySide6.QtWidgets import QApplication
from simulator.gui.model_hub import ModelHub
from simulator.gui.flight_batch_page import FlightBatchPage


def test_model_selection_stages_without_mutation_or_training(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    p=tmp_path/'config/research_30hz/model.json';p.parent.mkdir(parents=True)
    p.write_text(json.dumps({'cable':{},'recorded_data':{}}));before=p.read_bytes()
    w=ModelHub(tmp_path);received=[];w.model_requested.connect(received.append)
    monkeypatch.setattr(w.job,'start',lambda *a: (_ for _ in ()).throw(AssertionError('Unexpected job launch')))
    w.use.click()
    assert received==[str(p.resolve())]
    assert p.read_bytes()==before and not w.job.running
    assert not w.prepare.isEnabled() and not w.run.isEnabled()
    assert w.tabs.count()==4
    w.close()


def test_stop_boundary_preserves_partial_fit(tmp_path):
    import pytest
    from experimental_data.current_adaptation_fit import note
    (tmp_path/'STOP').touch()
    with pytest.raises(InterruptedError,match='partial results preserved'):
        note(tmp_path,'next update')
    assert not (tmp_path/'progress.json').exists()


def test_full_fit_action_dispatches_one_job_without_fitting(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    b=tmp_path/'rehearsal_csv_and_result_in_real_flight/policy/adp0'
    (b/'flight_take').mkdir(parents=True);(b/'simulation_csv').mkdir()
    (b/'simulation_csv/fullstate_30hz.csv').touch()
    for i in range(5):
        (b/f'flight_take/take{i}.csv').touch();(b/f'flight_take/experiment_take{i}.csv').touch()
    w=ModelHub(tmp_path);launches=[]
    monkeypatch.setattr(w.job,'start',lambda folder,cmd:launches.append(cmd))
    assert w.prepare.isEnabled()
    w.prepare.click()
    assert len(launches)==1 and 'all' in launches[0]
    assert not w.job.running
    w.close()


def test_all_fit_stages_are_sequential_without_training(tmp_path,monkeypatch):
    import tools.model_job as runner
    run=runner.run;calls=[]
    monkeypatch.setattr(runner,'run',lambda stage,job,batch=None:calls.append(stage))
    run('all',tmp_path,tmp_path/'batch')
    assert calls==['prepare','drone','cable_batched','attitude_refine','baseline','validate']


def test_replay_defaults_to_normalized_without_double_shift(tmp_path,monkeypatch):
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
    assert w.normalize_height.isChecked()
    np.testing.assert_array_equal(w.data['measured_origin'],normalized)
    w.normalize_height.setChecked(False);np.testing.assert_array_equal(w.data['measured_origin'],normalized)
    w.normalize_height.setChecked(True);w.loaded(d)
    np.testing.assert_array_equal(w.data['measured_origin'],normalized)
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
