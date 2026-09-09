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
    assert window.main_tabs.count() == 7
    assert window.main_tabs.widget(5) is window.mppi_page
    assert not hasattr(window,'cem_page')
    assert 'Testing' not in [window.main_tabs.tabText(i) for i in range(window.main_tabs.count())]
    assert window.main_tabs.widget(4) is window.fullstate_page
    assert window.reward_page.hit_spins['maximum_tip_velocity_to_desired_direction_error_deg'].value() == 45
    assert window.reward_page.spins['time_to_success_weight_per_s'].value() == ppo['reward']['time_to_success_weight_per_s']
    assert window.training_page.start_button.text() == 'Train new policy'
    assert not hasattr(window, 'sac_page')
    assert 'SAC' not in [window.main_tabs.tabText(i) for i in range(window.main_tabs.count())]
    assert window.training_page.tabs.tabText(1) == 'Policies'
    window.main_tabs.setCurrentIndex(2)
    application.processEvents()
    assert window.training_page.timer.isActive()
    window.navigation_buttons[4].click()
    application.processEvents()
    assert window.fullstate_page.active
    assert window.shell_page_title.text() == 'Rehearsal & Export'
    assert [window.main_tabs.tabText(i) for i in range(5)]==['Model','Recordings','PPO','Diagnostics','Rehearsal & Export']
    assert window.training_page.tabs.tabText(2)=='Task & rewards'
    assert window.fullstate_page.viewer is None  # Native scene is constructed when a prediction is available.
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
