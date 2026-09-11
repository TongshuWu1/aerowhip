from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[2]


def test_process_liveness_check_handles_current_and_missing_processes() -> None:
    from simulator.gui.process_status import process_is_running

    assert process_is_running(os.getpid())
    assert not process_is_running(2_147_483_647)


def test_desktop_ui_exposes_ppo_without_sac(monkeypatch) -> None:
    from PySide6.QtWidgets import QApplication
    from simulator.gui.main_window import SimulatorMainWindow
    # This checks lazy scene construction, independent of a user's saved replay.
    monkeypatch.setattr(SimulatorMainWindow,'restore_replay',lambda self:None)
    from simulator.rollout import load_json
    application = QApplication.instance() or QApplication([])
    ppo = load_json(ROOT / 'config/ppo.json')
    window = SimulatorMainWindow(ROOT, load_json(ROOT / 'config/model.json'),
                                 load_json(ROOT / 'config/task.json'), ppo)
    assert window.main_tabs.count() == 6
    assert window.main_tabs.widget(3) is window.mppi_page
    assert not hasattr(window,'cem_page')
    assert 'Testing' not in [window.main_tabs.tabText(i) for i in range(window.main_tabs.count())]
    assert window.main_tabs.widget(4) is window.rehearsal_page
    assert window.main_tabs.widget(5) is window.flight_workspace
    assert window.flight_workspace.widget(0) is window.evolution_page
    assert window.ppo_page.fields[('task','maximum_angle_deg')].value() == 45
    assert window.ppo_page.fields[('reward','time_per_s')].value() > 0
    assert window.training_page.run.text() == 'Start new training'
    assert not hasattr(window, 'sac_page')
    assert 'SAC' not in [window.main_tabs.tabText(i) for i in range(window.main_tabs.count())]
    assert window.training_page.tabs.tabText(2) == 'Policy library'
    window.main_tabs.setCurrentIndex(2)
    application.processEvents()
    assert window.training_page.timer.isActive()
    window.buttons[4].click()
    application.processEvents()
    assert window.inspector.active
    assert window.title.text() == 'Rehearsals'
    assert [window.main_tabs.tabText(i) for i in range(6)]==['Models & fitting','Recordings','PPO','MPPI','Rehearsals','Flight comparison']
    assert window.inspector.viewer is None  # Scene is constructed only for a saved prediction.
    window.close()
    application.processEvents()
