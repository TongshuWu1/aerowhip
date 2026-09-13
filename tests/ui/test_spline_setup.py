import os,json
from pathlib import Path
import pytest
from experimental_data.io import atomic_json


def test_spline_setup_saves_separately_and_keeps_historical_plans(tmp_path):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    from simulator.gui.pva_workspace import PVAPlannerPage
    from planning.pva_job import load_settings
    root=Path(__file__).resolve().parents[2]
    cfg=json.loads((root/'tests/fixtures/targeted_strike_settings.json').read_text())
    for key in ('model_path','proposal_templates_path'):
        if key in cfg:cfg[key]=str(root/cfg[key])
    if not Path(cfg['model_path']).exists():pytest.skip('Private retained model unavailable')
    atomic_json(tmp_path/'config/pva/systematic_strike.json',cfg)
    legacy=load_settings(tmp_path,'mppi');atomic_json(tmp_path/'config/pva/mppi.json',legacy)
    app=QApplication.instance() or QApplication([])
    page=PVAPlannerPage(tmp_path,'mppi');page.timer.stop()
    assert page.tabs.tabText(0)=='Spline setup' and page.run.isEnabled()
    assert ('task','target_radius_m') not in page.fields
    assert ('reward','success') not in page.fields
    page.fields[('recovery','hold_s')].setValue(4.)
    page.fields[('trajectory_objective','speed_proximity_scale_m')].setValue(.35)
    page.save_settings()
    saved=json.loads((tmp_path/'config/pva/systematic_strike.json').read_text())
    assert saved['mppi']['initialization']=='from_scratch' and 'proposal_templates_path' not in saved
    assert saved['mppi']['position_noise_scales_m']==[.1,.4,1.]
    assert saved['launch']['origin_m']==[0.,0.,1.4]
    assert saved['launch']['target_m']==[1.4,0.,1.25]
    assert saved['trajectory_objective']['maximum_strike_angle_deg']==30.
    assert saved['trajectory_objective']['intensity_weight']==8.
    assert saved['trajectory_objective']['prefer_aligned_strike'] is True
    assert saved['trajectory_objective']['speed_metric']=='tip_gain_over_root'
    assert saved['trajectory_objective']['minimum_tip_speed_gain_m_s']==2.
    assert saved['trajectory_objective']['speed_proximity_scale_m']==.35
    assert saved['recovery']['hold_s']==4 and saved['mppi']['support_points']==9
    assert json.loads((tmp_path/'config/pva/mppi.json').read_text())==legacy
    assert not (tmp_path/'runs/mppi_pva').exists()
    page.close()


def test_archived_models_are_hidden_and_registered_m0_remains_available(tmp_path):
    from simulator.gui.pva_workspace import model_paths
    m0=tmp_path/'runs/rehearsals_pva/retained-M0/model.json'
    m1=tmp_path/'runs/adaptation/old-M1/candidate/model.json'
    fresh=tmp_path/'runs/adaptation/new-M1/candidate/model.json'
    for path in (m0,m1,fresh):atomic_json(path,{'provenance':{'fit_complete':True}})
    atomic_json(tmp_path/'config/evaluation/campaign.json',{'models':[
        {'id':'M0','model':str(m0)},{'id':'old-M1','model':str(m1)}]})
    (m1.parents[1]/'ARCHIVED').touch()
    paths=model_paths(tmp_path)
    assert m0.resolve() in paths and fresh.resolve() in paths
    assert m1.resolve() not in paths


def test_spline_progress_reports_completed_iterations_and_strike_speed():
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    from simulator.gui.mppi_dashboard import MPPIDashboard
    app=QApplication.instance() or QApplication([])
    cfg=json.loads((Path(__file__).resolve().parents[2]/'tests/fixtures/targeted_strike_settings.json').read_text())
    history=[dict(iteration=3,best_score=1.,strike_distance_m=.0123,directed_tip_speed_m_s=4.56,rewarded_tip_speed_m_s=2.34,root_forward_speed_m_s=2.22)]
    dashboard=MPPIDashboard()
    dashboard.update_run(dict(status='running',iteration=4),cfg,history,[])
    assert dashboard.values['iteration'].text()=='3'
    assert dashboard.values['step'].text()=='2.34 m/s'
    assert dashboard.labels['step'].text()=='Rewarded tip-speed gain'
    assert dashboard.values['distance'].text()=='1.23 cm'
    assert dashboard.work.value()==3
    dashboard.update_run(dict(status='completed',iterations=3),cfg,history,[])
    assert 'Complete motion optimized' in dashboard.detail.text()
    assert 'diagnostic only' in dashboard.detail.text()
    dashboard.close()


@pytest.mark.parametrize('requirement,failed,scored,score,expected',[
    ('diagnostic_only',False,True,1.,'scored strike; fold diagnostic only'),
    ('required',False,True,1.,'fold-qualified strike'),
    ('diagnostic_only',True,True,1.,'infeasible'),
    ('diagnostic_only',False,True,float('-inf'),'not accepted'),
    ('diagnostic_only',False,False,1.,'no scored strike'),
])
def test_live_candidate_labels_respect_fold_policy(requirement,failed,scored,score,expected):
    from simulator.gui.mppi_live_view import candidate_status
    cfg={'task':{'success_criterion':'targeted_fold_strike_v1'},'fold_requirement':requirement}
    assert candidate_status(cfg,failed,scored,score)==expected
