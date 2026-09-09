import os
from pathlib import Path
from experimental_data.io import atomic_json
from simulator.workflow import read_json
from planning.pva_job import load_settings,settings_path


def test_planner_setup_edits_are_independent_and_do_not_translate_saved_runs(tmp_path):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    from simulator.gui.pva_workspace import PVAPlannerPage
    app=QApplication.instance() or QApplication([])
    ppo=load_settings(tmp_path,'ppo');mppi=load_settings(tmp_path,'mppi')
    atomic_json(settings_path(tmp_path,'ppo'),ppo);atomic_json(settings_path(tmp_path,'mppi'),mppi)
    page=PVAPlannerPage(tmp_path,'mppi')
    page.fields[('launch','origin_m')][0].setValue(.4)
    page.fields[('launch','target_m')][0].setValue(1.7)
    page.fields[('reward','success')].setValue(240.)
    page.horizon.setValue(15)
    page.save_settings()
    assert read_json(settings_path(tmp_path,'ppo'))==ppo
    saved=read_json(settings_path(tmp_path,'mppi'))
    assert saved['launch']['origin_m'][0]==.4 and saved['launch']['target_m'][0]==1.7
    assert saved['reward']['success']==240.
    assert saved['mppi']['iterations']==0
    assert page.fields[('mppi','iterations')].text()=='No limit'
    assert saved['mppi']['horizon_s']==.5
    assert saved['task']['duration_s']==mppi['task']['duration_s']==5.
    page.shutdown();page.close();app.processEvents()


def test_new_window_is_six_page_direct_pva_workflow():
    from simulator.gui.main_window import SimulatorMainWindow
    from simulator.gui.pva_main_window import PVAResearchWindow,PAGES
    assert SimulatorMainWindow is PVAResearchWindow
    assert [name for name,_ in PAGES]==['Models & fitting','Recordings','PPO','MPPI','Rehearsals','Flight comparison']


def test_wave_setup_keeps_mixture_scales_and_task_controls(tmp_path):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    from simulator.gui.pva_workspace import PVAPlannerPage
    from learning.whip_wave import DEFAULTS
    app=QApplication.instance() or QApplication([])
    cfg=load_settings(tmp_path,'mppi');cfg['task'].update(DEFAULTS,require_wave=True)
    cfg['mppi'].update(initialization='wave',noise_scales=[.05,.15,.35])
    atomic_json(settings_path(tmp_path,'mppi'),cfg)
    page=PVAPlannerPage(tmp_path,'mppi');saved=page.collect()
    assert saved['task']['require_wave'] and saved['mppi']['initialization']=='wave'
    assert saved['mppi']['noise_scales']==[.05,.15,.35]
    page.shutdown();page.close();app.processEvents()


def test_campaign_run_appears_without_reopening_ui_and_evaluation_reward_is_visible(tmp_path):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    from simulator.gui.pva_workspace import PVAPlannerPage
    app=QApplication.instance() or QApplication([])
    page=PVAPlannerPage(tmp_path,'ppo')
    job=tmp_path/'runs/ppo_pva/campaign-run'
    atomic_json(job/'identity.json',dict(name='Fresh overnight PPO',model_source='model.json'))
    atomic_json(job/'model.json',dict(provenance=dict(label='Fresh M0')))
    atomic_json(job/'status.json',dict(status='running',attempts=1024))
    atomic_json(job/'history.json',[dict(attempts=1024,reward=10,success=.2,minimum_tip_distance_m=.1,
        failures=.05,evaluation_reward=15,evaluation_success=.4)])
    page.poll()
    assert page.current_run==job and page.library.rowCount()==1
    assert page.library.item(0,1).text()=='Fresh M0'
    assert [line.get_label() for line in page.figure.axes[0].lines]==['Training batch','Deterministic evaluation']
    atomic_json(job/'status.json',dict(status='completed',attempts=2048,stop_reason='reward_plateau'))
    page.poll()
    assert page.library.item(0,2).text()=='completed' and not page.stop.isEnabled()
    page.shutdown();page.close();app.processEvents()


def test_mppi_shows_actual_saved_plan_status_and_blocks_live_rehearsal(tmp_path):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    from simulator.gui.pva_workspace import PVAPlannerPage
    app=QApplication.instance() or QApplication([])
    job=tmp_path/'runs/mppi_pva/diagnostic'
    atomic_json(job/'identity.json',dict(name='Historical model diagnostic'))
    atomic_json(job/'status.json',dict(status='running',iteration=3,best_failed=True,best_success=False))
    page=PVAPlannerPage(tmp_path,'mppi');page.poll()
    assert 'infeasible' in page.progress_note.text()
    page.rehearse_selected()
    assert 'Stop or finish' in page.library_note.text() and not page.rehearsal_job.running
    page.job_finished(0)
    assert not page.run.isEnabled()  # Missing model cannot be enabled by completion callback.
    page.shutdown();page.close();app.processEvents()
