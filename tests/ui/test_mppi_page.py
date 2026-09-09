import json
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication
from simulator.gui.mppi_page import MPPIPage


def test_independent_mppi_controls_model_and_persistence(tmp_path):
    app=QApplication.instance() or QApplication([])
    config=tmp_path/'config';config.mkdir()
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    for name in ('cem','launch_setup','ppo'):(config/f'{name}.json').write_text('{"preserve":true}')
    before={p:p.read_bytes() for p in config.iterdir() if p.is_file()}
    page=MPPIPage(tmp_path)
    assert page.start.isEnabled() and page.seed.count()==0
    assert page.start.text()=='Generate new plan with MPPI'
    assert 'temperature' in page.spins and 'elite_fraction' not in page.spins
    assert 'adp0-M1' in page.model_path.text()
    page.spins['temperature'].setValue(35)
    page.start_spins[0].setValue(-2.2);page.target_spins[0].setValue(-.9)
    assert 'control_points' not in page.spins and 'horizon_s' in page.spins
    page.settings_page.force_reward_spins['time_to_success_weight_per_s'].setValue(30)
    page.save_settings();saved=json.loads((config/'mppi.json').read_text())
    assert saved['temperature']==35 and saved['optimizer']=='mppi'
    assert saved['reward']['time_to_success_weight_per_s']==30
    page.set_running(True);assert not page.model_path.isEnabled() and not page.browse_model.isEnabled()
    page.set_running(False);page.shutdown();page.close()
    restored=MPPIPage(tmp_path)
    assert restored.settings_values()==saved
    assert all(p.read_bytes()==b for p,b in before.items())
    restored.shutdown();restored.close();app.processEvents()


def test_mppi_replaces_cem_and_adaptation_check_stays_seventh():
    from simulator.gui.main_window import PAGE_DEFINITIONS
    assert [p[0] for p in PAGE_DEFINITIONS]==['Model','Recordings','PPO','Diagnostics',
        'Rehearsal & Export','MPPI Planner','Adaptation Check']


def test_changed_launch_does_not_show_old_csv_as_current(tmp_path):
    from pathlib import Path
    import shutil
    import pytest
    root=Path(__file__).resolve().parents[2]
    source=root/'runs/mppi/20260908-M1-mppi-local-check'
    if not (source/'rehearsal.json').exists():pytest.skip('Local saved replay required')
    app=QApplication.instance() or QApplication([])
    folder=tmp_path/'runs/mppi/saved';folder.mkdir(parents=True)
    for name in ('model.json','mppi.json','task.json','rehearsal.json','rehearsal.npz'):
        shutil.copy2(source/name,folder/name)
    config=tmp_path/'config';config.mkdir()
    model=root/'data/model_candidates/20260908-adp0-M1/model.json'
    settings=dict(model_path=str(model),launch_setup=dict(initial_tracking_origin_m=[0,0,1.225],target_position_m=[1.5,0,1.1]))
    (config/'mppi.json').write_text(json.dumps(settings))
    original=(folder/'rehearsal.npz').read_bytes()
    page=MPPIPage(tmp_path)
    assert page.results.count()==0
    page.inspect()
    assert page.preview.arrays is None and not page.preview.save.isEnabled()
    assert 'No completed trajectory' in page.result_banner.text()
    page.result_scope.setCurrentIndex(1);page.inspect()
    assert page.results.count()==1 and page.preview.save.isEnabled()
    assert page.preview.start_spins[0].value()==-2
    assert 'SAVED RUN' in page.result_banner.text()
    page.result_scope.setCurrentIndex(0);page.inspect()
    assert not page.preview.save.isEnabled() and page.preview.arrays is None
    page.start_spins[0].setValue(-2);page.start_spins[2].setValue(1.255);page.target_spins[0].setValue(-1)
    assert page.results.count()==0  # An old spline result is historical even at matching coordinates.
    page.target_spins[0].setValue(1.5)
    assert page.results.count()==0 and page.preview.arrays is None and not page.preview.save.isEnabled()
    assert (folder/'rehearsal.npz').read_bytes()==original
    page.shutdown();page.close();app.processEvents()
