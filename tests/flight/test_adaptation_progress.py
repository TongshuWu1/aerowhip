from pathlib import Path
import numpy as np
import pytest
from experimental_data.adaptation_progress import load_study,comparison_arrays
from simulator.workflow import prepare_training,read_json

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'runs/adaptation/20260908-adp0-first'
MODEL=ROOT/'data/model_candidates/20260908-adp0-M1/model.json'
PARENT=ROOT/'runs/ppo/20260908-195207-486249-seed655/checkpoints/best_validation.pt'


def test_saved_progress_and_separate_heldout_predictions():
    if not STUDY.exists():pytest.skip('Local adaptation study required')
    s=load_study(ROOT,STUDY)
    assert len(s['rows'])==5 and s['models']==[MODEL]
    assert s['means']['adapted_tip_whip_rmse_m']==pytest.approx(.10417897558)
    baseline,heldout=comparison_arrays(s,'whip_adp_0_001','heldout')
    _,fitted=comparison_arrays(s,'whip_adp_0_001','all_five')
    assert not np.array_equal(heldout['cable'],fitted['cable'])
    np.testing.assert_array_equal(baseline['measured_sites'],heldout['measured_sites'])


def test_model_switch_freezes_m1_and_preserves_parent(tmp_path,monkeypatch):
    if not MODEL.exists():pytest.skip('Local model required')
    import simulator.research_config
    monkeypatch.setattr(simulator.research_config,'freeze_training_source',lambda root,directory,command:command)
    (tmp_path/'config').mkdir();(tmp_path/'config/ppo.json').write_text('{"cuda_graph_physics":true}')
    original={p:p.read_bytes() for p in [PARENT,PARENT.parent.parent/'model.json',PARENT.parent.parent/'ppo.json',PARENT.parent.parent/'task.json',MODEL]}
    directory,command=prepare_training(tmp_path,'ppo',seed=655,episodes=334848,batch=2048,device='cuda',
        resume=PARENT,model_path=MODEL,run_name='test M1 continuation',keep_optimizer_state=False,live_scene=False)
    model=read_json(directory/'launch_config/model.json');config=read_json(directory/'launch_config/ppo.json')
    assert model['motion_residual']['sha256']==read_json(MODEL)['motion_residual']['sha256']
    assert model['fullstate_execution']['sha256']==read_json(MODEL)['fullstate_execution']['sha256']
    assert config['deployment']['reset_optimizer_on_resume']
    assert 'reward_plateau_resume' not in config
    assert config['reward']==read_json(PARENT.parent.parent/'ppo.json')['reward']
    assert read_json(directory/'launch_config/task.json')==read_json(PARENT.parent.parent/'task.json')
    assert read_json(directory/'run.json')['model_amendment']['optimizer_state_reset']
    assert all(p.read_bytes()==value for p,value in original.items())
    with pytest.raises(ValueError,match='convergence'):
        prepare_training(tmp_path,'ppo',seed=655,episodes=334848,batch=2048,device='cuda',resume=PARENT,model_path=MODEL,keep_stopping_history=True)


def test_progress_ui_reads_saved_evidence():
    if not STUDY.exists():pytest.skip('Local adaptation study required')
    import os
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    from simulator.gui.adaptation_progress_page import AdaptationProgressPage
    app=QApplication.instance() or QApplication([]);page=AdaptationProgressPage(ROOT)
    assert page.table.rowCount()==5 and 'DIAGNOSTIC ONLY' in page.summary.text()
    assert page.tabs.tabText(0)=='Real flight progress'
    assert 'awaiting adp1' in page.real.status.text().lower()
    assert all(page.real.data['rows'][0]['means'][key] is None for key in ('drone_rms_m','tip_rms_m')) if page.real.data else True
    assert page.use_model.isEnabled()
    page.mode.setCurrentIndex(1);page.timeline.setValue(100);app.processEvents()
    assert 'in-sample' in page.trace_note.text()
    assert len(page.trace_figure.axes)==2
    page.close();app.processEvents()


def test_measured_round_metrics_do_not_use_fitted_predictions():
    from experimental_data.adaptation_progress import measured_flight_metrics
    time=np.arange(101)/100
    data=dict(time=time,metadata=dict(whip_end_s=1.,predicted_hit_time_s=.5),
        drone_error=np.full(101,.1),tip_error=np.full(101,.2),target_error=np.full(101,.3),
        measured_cable=np.broadcast_to([.3,0,0],(101,11,3)).copy(),target=np.zeros(3),
        tracking_span=(-.1,1.1),take='flight',hashes={})
    data['hover_normalized']={k:data[k].copy() for k in ['drone_error','tip_error','target_error','measured_cable']}
    data['height_calibration']=dict(bias_z_m=.05,checks_passed=True)
    data['drone_error'][:]=100;data['tip_error'][:]=200;data['target_error'][:]=300
    result=measured_flight_metrics(data)
    assert result['drone_rms_m']==pytest.approx(.1)
    assert result['tip_rms_m']==pytest.approx(.2)
    assert result['target_at_strike_m']==pytest.approx(.3)
    assert result['evaluation_frame']=='hover_normalized_z'
    data['hover_normalized']['measured_cable'][50,-1]=np.nan
    assert measured_flight_metrics(data)['target_at_strike_m'] is None
    del data['hover_normalized']
    with pytest.raises(ValueError,match='calibration required'):measured_flight_metrics(data)


def test_no_real_m1_flights_means_no_adapted_rms(tmp_path):
    from experimental_data.adaptation_progress import recorded_flight_progress
    (tmp_path/'runs/adaptation/fake/validation').mkdir(parents=True)
    (tmp_path/'runs/adaptation/fake/validation/results.json').write_text('{"adapted_tip_whip_rmse_m":0.01}')
    data=recorded_flight_progress(tmp_path)
    assert len(data['rows'])==1 and data['rows'][0]['model']=='M1'
    assert data['rows'][0]['means']['tip_rms_m'] is None
    assert data['rows'][0]['actual_flights']==0


def test_policy_library_distinguishes_m1_from_m0_and_real_results(tmp_path):
    import json
    from PySide6.QtWidgets import QApplication
    from simulator.gui.policy_library_page import PolicyLibraryPage
    app=QApplication.instance() or QApplication([])
    for name,adaptation in [('M0',{}),('M1',dict(round='current_adp0',training=True))]:
        run=tmp_path/'runs/ppo'/name;(run/'checkpoints').mkdir(parents=True)
        (run/'checkpoints/best_validation.pt').write_bytes(b'library reads metadata only')
        (run/'model.json').write_text(json.dumps(dict(fullstate_execution=dict(schema='tracked_pose_execution_v1'),adaptation=adaptation)))
    page=PolicyLibraryPage(tmp_path)
    labels={page.table.item(i,0).text():page.table.item(i,2).text() for i in range(page.table.rowCount())}
    assert labels['M1'].startswith('M1') and labels['M0'].startswith('M0')
    assert 'Sim.' in page.table.horizontalHeaderItem(3).text()
    page.close();app.processEvents()
