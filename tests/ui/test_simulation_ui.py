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


def test_desktop_ui_exposes_ppo_without_sac() -> None:
    from PySide6.QtWidgets import QApplication
    from simulator.gui.main_window import SimulatorMainWindow
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


def test_reward_page_saves_next_run_config_and_keeps_angle_zero(tmp_path) -> None:
    from PySide6.QtWidgets import QApplication

    from simulator.gui.reward_page import RewardSettingsPage

    application = QApplication.instance() or QApplication([])
    config_directory = tmp_path / "config"
    config_directory.mkdir()
    (config_directory / 'model.json').write_text((ROOT / 'config/model.json').read_text(encoding='utf-8'), encoding='utf-8')
    ppo = json.loads((ROOT / "config/ppo.json").read_text(encoding="utf-8"))
    task = json.loads((ROOT / "config/task.json").read_text(encoding="utf-8"))
    (config_directory / "ppo.json").write_text(json.dumps(ppo), encoding="utf-8")
    (config_directory / "task.json").write_text(json.dumps(task), encoding="utf-8")
    page = RewardSettingsPage(tmp_path, ppo)
    page.spins["progress_weight"].setValue(24.0)
    page.spins["success_bonus"].setValue(120.0)
    page.spins["point_displacement_integral_weight"].setValue(18.0)
    page.save_settings()
    saved_ppo = json.loads(
        (config_directory / "ppo.json").read_text(encoding="utf-8")
    )
    saved_task = json.loads(
        (config_directory / "task.json").read_text(encoding="utf-8")
    )
    assert saved_ppo["reward"]["progress_weight"] == 24.0
    assert saved_ppo["reward"]["success_bonus"] == 120.0
    assert saved_ppo["reward"]["point_displacement_integral_weight"] == 18.0
    assert saved_ppo["reward"]["angle_shaping_weight"] == 0.0
    assert saved_task["reward"]["angle_shaping_weight"] == 0.0
    page.close()
    application.processEvents()
