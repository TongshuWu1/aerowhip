from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(__file__).resolve().parents[2]


def _trajectory():
    positions = np.full((4, 4, 3), 5.0)
    positions[:, -1] = [[-.10, 0, 0], [.10, 0, 0], [.20, 0, 0], [.30, 0, 0]]
    velocities = np.zeros_like(positions)
    velocities[:, -1, 0] = 5.0
    return positions, velocities


def _diagnose(positions, velocities):
    from simulator.strike_diagnostics import replay_strike_diagnostics
    return replay_strike_diagnostics(positions, velocities, [2, 3], np.zeros(3), np.array([1, 0, 0]),
        radius_m=.05, minimum_speed_m_s=4, maximum_angle_deg=45)


def test_fast_crossing_is_a_hit_even_when_both_displayed_positions_are_outside():
    positions, velocities = _trajectory()
    # Node 0/1 are attachment/internal nodes, not the cable markers checked by training.
    positions[:, :2] = 0
    result = _diagnose(positions, velocities)
    assert result['hit'].tolist() == [False, True, False, False]
    assert result['success_so_far'].tolist() == [False, True, True, True]


def test_invalid_tip_then_other_marker_disqualifies_a_later_valid_entry():
    positions, velocities = _trajectory()
    positions[:, -1] = [[-.1, 0, 0], [.1, 0, 0], [.2, 0, 0], [-.1, 0, 0]]
    velocities[1, -1] = [2, 0, 0]
    positions[2:, 2] = 0
    result = _diagnose(positions, velocities)
    assert result['tip_contact'][1]
    assert not result['hit'].any()
    assert result['disqualified'][2:].all()


def test_later_valid_tip_entry_is_allowed_after_a_slow_tip_only_entry():
    positions, velocities = _trajectory()
    positions[3, -1] = [-.1, 0, 0]
    velocities[1, -1] = [2, 0, 0]
    result = _diagnose(positions, velocities)
    assert result['hit'].tolist() == [False, False, False, True]


def test_rule_editor_saves_draft_preserves_other_settings_and_keeps_run_snapshot(tmp_path):
    from PySide6.QtWidgets import QApplication
    from simulator.gui.reward_page import RewardSettingsPage
    app = QApplication.instance() or QApplication([])
    config = json.loads((ROOT / 'config/ppo.json').read_text(encoding='utf-8'))
    task = json.loads((ROOT / 'config/task.json').read_text(encoding='utf-8'))
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config/model.json').write_text((ROOT / 'config/model.json').read_text(encoding='utf-8'), encoding='utf-8')
    (tmp_path / 'config/ppo.json').write_text(json.dumps(config))
    (tmp_path / 'config/task.json').write_text(json.dumps(task))
    run = tmp_path / 'runs/ppo/example'
    run.mkdir(parents=True)
    (run.parent / 'ACTIVE_RUN.txt').write_text(str(run))
    (run / 'ppo.json').write_text(json.dumps(config))
    original_run_ppo = (run / 'ppo.json').read_text(encoding='utf-8')
    (run / 'task.json').write_text(json.dumps(task))
    (run / 'status.json').write_text(json.dumps({'status': 'RUNNING'}))
    page = RewardSettingsPage(tmp_path, config)
    assert not page.has_unsaved_changes
    page.hit_spins['maximum_tip_velocity_to_desired_direction_error_deg'].setValue(30)
    page.spins['terminal_displacement_weight'].setValue(100)
    assert page.has_unsaved_changes
    # A change from another settings page must survive this page's save.
    saved = json.loads((tmp_path / 'config/ppo.json').read_text(encoding='utf-8'))
    saved['training']['requested_episodes'] = 123456
    (tmp_path / 'config/ppo.json').write_text(json.dumps(saved))
    page.page_activated()
    assert page.hit_spins['maximum_tip_velocity_to_desired_direction_error_deg'].value() == 30
    page.save_settings()
    assert not page.has_unsaved_changes
    assert json.loads((tmp_path / 'config/ppo.json').read_text(encoding='utf-8'))['training']['requested_episodes'] == 123456
    saved_task=json.loads((tmp_path / 'config/task.json').read_text(encoding='utf-8'))
    assert abs(saved_task['control_dt_s']-task['control_dt_s'])<1e-12
    assert abs(page.task_spins[('control_dt_s',None)].value()-1/task['control_dt_s'])<1e-8
    assert json.loads((tmp_path / 'config/task.json').read_text(encoding='utf-8'))['success']['maximum_tip_velocity_to_desired_direction_error_deg'] == 30
    assert json.loads((run / 'task.json').read_text(encoding='utf-8'))['success']['maximum_tip_velocity_to_desired_direction_error_deg'] == task['success']['maximum_tip_velocity_to_desired_direction_error_deg']
    assert (run / 'ppo.json').read_text(encoding='utf-8') == original_run_ppo
    page.close()
    app.processEvents()
