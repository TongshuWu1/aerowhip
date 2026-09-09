import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import json
import numpy as np
from PySide6.QtWidgets import QApplication
from simulator.gui.cem_page import CEMPage


def test_cem_inspector_uses_pva_without_fake_force_or_policy(tmp_path):
    app=QApplication.instance() or QApplication([])
    (tmp_path/'config').mkdir()
    (tmp_path/'config/launch_setup.json').write_text(json.dumps(dict(initial_tracking_origin_m=[-2,0,1.255],target_position_m=[-1,0,1.1])))
    folder=tmp_path/'runs/cem/result';folder.mkdir(parents=True)
    t=np.arange(4)/30;commands=np.zeros((4,11));commands[:,2]=1.255
    q=np.zeros((4,12,3));q[:,:,2]=1.2-np.arange(12)/12
    np.savez(folder/'rehearsal.npz',command_time_s=t,commands=commands,command_phase=np.ones(4),
        prediction_time_s=t,cable_positions_m=q,origin_positions_m=commands[:,:3],
        origin_rotations=np.tile(np.eye(3),(4,1,1)),target_position_m=[-1,0,1.1])
    (folder/'rehearsal.json').write_text(json.dumps(dict(schema='cem_fullstate_30hz_v1',planner='CEM',display_name='Test',
        initial_tracking_origin_m=[-2,.125,1.255],target_position_m=[-1,0,1.1],whip_end_s=.1,total_duration_s=.1,
        predicted_valid_hit=False,minimum_tip_distance_m=.2,recovery_prediction_complete=True,prediction_valid_through_s=.1)))
    (folder/'cem.json').write_text(json.dumps({'reward':{}}))  # Older result, before explicit launch settings.
    (folder/'task.json').write_text(json.dumps({'desired_strike_direction_world':[1,0,0],'success':dict(tip_target_distance_m=.05,minimum_directed_tip_speed_m_s=4,
        maximum_tip_velocity_to_desired_direction_error_deg=45)}))
    page=CEMPage(tmp_path);assert page.seed.count()==1 and page.results.count()==1
    assert page.start_spins[1].value()==0  # CEM defaults are independent of the old seed.
    assert page.tabs.tabText(1)=='Rehearsal & Export'
    page.tabs.setCurrentIndex(1)  # Entering the tab loads the selected saved rehearsal.
    assert page.tabs.currentIndex()==1 and page.preview.table.rowCount()==4
    assert page.preview.save.isEnabled() and page.preview.start.isHidden()
    assert page.preview.start_spins[1].value()==.125  # Saved replay retains its actual coordinates.
    assert page.preview.figure.axes[0].get_title()=='Commanded velocity'
    assert 'CEM spline' in page.preview.status.text()
    assert 'virtual_force_n' not in page.preview.arrays
    assert not page.preview.start_spins[0].isEnabled()
    page.start_rehearsal()
    assert page.preview.playing and page.preview.timeline.value()==0
    page.load_run_settings()
    assert page.start_spins[1].value()==.125  # Explicit load restores that CEM run's actual setup.
    page.shutdown();page.close();app.processEvents()


def test_shared_launch_defaults_override_checkpoint_fallback(tmp_path):
    from simulator.launch_setup import launch_positions
    folder=tmp_path/'config';folder.mkdir()
    (folder/'launch_setup.json').write_text(json.dumps(dict(initial_tracking_origin_m=[-2,0,1.255],target_position_m=[-1,0,1.1])))
    origin,target=launch_positions(tmp_path,[-2,.012874,1.255],[1,0,1.4])
    np.testing.assert_array_equal(origin,[-2,0,1.255]);np.testing.assert_array_equal(target,[-1,0,1.1])
    # The UI only applies defaults for new plans, not while opening saved artifacts.
    from simulator.gui.rehearsal_workspace import RehearsalWorkspace
    from pathlib import Path
    import shutil
    root=Path(__file__).resolve().parents[2]
    for file in ('model.json','task.json','ppo.json'):
        shutil.copy2(root/'config/research_30hz'/file,folder/file)
    app=QApplication.instance() or QApplication([])
    page=RehearsalWorkspace(tmp_path)
    np.testing.assert_array_equal([s.value() for s in page.start_spins],origin)
    page.refresh_checkpoints()
    np.testing.assert_array_equal([s.value() for s in page.target_spins],target)
    page.shutdown();page.close();app.processEvents()


def test_cem_settings_persist_without_changing_ppo_and_lock_during_run(tmp_path):
    app=QApplication.instance() or QApplication([])
    folder=tmp_path/'config';folder.mkdir();(folder/'ppo.json').write_text('{"untouched": true}')
    original=(folder/'ppo.json').read_bytes()
    page=CEMPage(tmp_path)
    assert page.tabs.tabText(2)=='Task & rewards'
    page.settings_page.reward_spins['time_weight_per_s'].setValue(75)
    page.settings_page.hit_spins['tip_target_distance_m'].setValue(.07)
    page.spins['elite_fraction'].setValue(.25);page.spins['population'].setValue(128)
    page.save_settings();saved=json.loads((folder/'cem.json').read_text())
    assert saved['reward']['time_weight_per_s']==75 and saved['success']['tip_target_distance_m']==.07
    assert saved['population']==128 and saved['elite_fraction']==.25
    page.set_running(True);assert not page.settings_page.isEnabled()
    page.set_running(False);assert page.settings_page.isEnabled()
    page.shutdown();page.close()
    restored=CEMPage(tmp_path)
    assert restored.settings_values()['reward']['time_weight_per_s']==75
    assert restored.settings_values()['success']['tip_target_distance_m']==.07
    assert (folder/'ppo.json').read_bytes()==original
    restored.shutdown();restored.close();app.processEvents()


def test_cem_launch_is_independent_and_survives_seed_refresh_and_restart(tmp_path):
    from simulator.launch_setup import launch_positions
    app=QApplication.instance() or QApplication([])
    config=tmp_path/'config';config.mkdir()
    for name in ('ppo','sac','task'):(config/f'{name}.json').write_text('{"preserved": true}')
    (config/'launch_setup.json').write_text(json.dumps(dict(initial_tracking_origin_m=[2,3,4],target_position_m=[3,4,5])))
    before={p:p.read_bytes() for p in config.iterdir()}
    launch=dict(initial_tracking_origin_m=[-3,.2,1.3],target_position_m=[-.8,.4,1.2])
    (config/'cem.json').write_text(json.dumps(dict(launch_setup=launch,reward={'time_weight_per_s':75})))
    for name in ('seed1','seed2'):
        folder=tmp_path/'runs/rehearsals'/name;folder.mkdir(parents=True)
        (folder/'rehearsal.json').write_text(json.dumps(dict(schema='research_fullstate_30hz_v1',
            initial_tracking_origin_m=[9,9,9],target_position_m=[8,8,8],whip_end_s=1)))
        (folder/'task.json').write_text(json.dumps(dict(success={},desired_strike_direction_world=[1,0,0])))
    page=CEMPage(tmp_path)
    assert page.launch_values()==launch
    page.start_spins[0].setValue(-2.5);page.target_spins[2].setValue(1.15)
    edited=page.launch_values()
    page.seed.setCurrentIndex(1-page.seed.currentIndex());page.refresh()
    assert page.launch_values()==edited
    page.save_launch.click()
    saved=json.loads((config/'cem.json').read_text())
    assert saved['launch_setup']==edited and saved['reward']['time_weight_per_s']==75
    page.set_running(True)
    assert not page.save_launch.isEnabled() and not page.start_spins[0].isEnabled()
    page.set_running(False);page.shutdown();page.close()
    restored=CEMPage(tmp_path)
    assert restored.launch_values()==edited
    restored.save_settings()
    assert json.loads((config/'cem.json').read_text())['launch_setup']==edited
    assert all(p.read_bytes()==value for p,value in before.items())
    origin,target=launch_positions(tmp_path,[0,0,0],[0,0,0])
    np.testing.assert_array_equal(origin,[2,3,4]);np.testing.assert_array_equal(target,[3,4,5])
    restored.shutdown();restored.close();app.processEvents()
