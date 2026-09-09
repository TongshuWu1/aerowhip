import json
import os
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


def test_reward_editor_roundtrips_combined_quality_and_preparation_allowance(tmp_path):
    from PySide6.QtWidgets import QApplication
    from simulator.gui.reward_page import RewardSettingsPage
    from tools.dynamic_strike_setup import configure_dynamic_strike
    root=Path(__file__).resolve().parents[2]
    model,task,ppo=configure_dynamic_strike(*[json.loads((root/'config/research_30hz'/f'{n}.json').read_text())
        for n in ('model','task','ppo')])
    folder=tmp_path/'config';folder.mkdir()
    for n,v in zip(('model','task','ppo'),(model,task,ppo)):
        (folder/f'{n}.json').write_text(json.dumps(v))
    app=QApplication.instance() or QApplication([])
    page=RewardSettingsPage(tmp_path,ppo,config_directory=folder)
    assert not page.has_unsaved_changes
    assert page.speed_reference.currentData()=='world_and_attachment_relative'
    assert page.allowance_mode.currentData()=='required_reach_plus_margin'
    assert 'recovery is appended only during export' in page.execution_note.text()
    assert page.task_spins[('episode_duration_s',None)].value()==5.
    assert not page.task_spins[('control_dt_s',None)].isEnabled()
    page.spins['displacement_allowance_margin_m'].setValue(.3)
    page.spins['relative_directed_speed_reward_cap_m_s'].setValue(6.5)
    page.save_settings()
    assert not page.has_unsaved_changes
    saved=json.loads((folder/'ppo.json').read_text())
    assert saved['reward']['displacement_allowance_margin_m']==.3
    assert saved['reward']['relative_directed_speed_reward_cap_m_s']==6.5
    assert saved['reward']['directed_speed_shaping_reference']=='world_and_attachment_relative'
    assert saved['early_stopping']['patience_episodes']==20000
    assert 'preparation margin' in saved['reward']['equation']
    assert json.loads((folder/'task.json').read_text())['success']==task['success']
    page.close();app.processEvents()
