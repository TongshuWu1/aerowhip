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


def test_desktop_ui_exposes_shared_task_and_two_training_workspaces() -> None:
    from PySide6.QtWidgets import QApplication
    from simulator.gui.main_window import SimulatorMainWindow
    from simulator.rollout import load_json
    application = QApplication.instance() or QApplication([])
    ppo = load_json(ROOT / 'config/ppo.json')
    window = SimulatorMainWindow(ROOT, load_json(ROOT / 'config/model.json'),
                                 load_json(ROOT / 'config/task.json'), ppo)
    assert window.main_tabs.count() == 5
    assert window.reward_page.hit_spins['maximum_tip_velocity_to_desired_direction_error_deg'].value() == 45
    assert window.reward_page.spins['time_to_success_weight_per_s'].value() == ppo['reward']['time_to_success_weight_per_s']
    assert window.training_page.start_button.text() == 'Train new policy'
    assert window.sac_page.start_button.text() == 'Train new policy'
    window.main_tabs.setCurrentIndex(2)
    application.processEvents()
    assert window.training_page.timer.isActive()
    assert not window.sac_page.timer.isActive()
    assert window.training_page.viewport.viewer.backend_name == 'MATPLOTLIB FALLBACK'
    window.main_tabs.setCurrentIndex(3)
    assert not window.training_page.timer.isActive()
    assert window.sac_page.timer.isActive()
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
