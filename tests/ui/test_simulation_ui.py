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


def test_desktop_ui_exposes_mppi_without_ppo(monkeypatch) -> None:
    from PySide6.QtWidgets import QApplication
    from simulator.gui.main_window import SimulatorMainWindow
    monkeypatch.setattr(SimulatorMainWindow,'restore_replay',lambda self:None)
    application=QApplication.instance() or QApplication([])
    window=SimulatorMainWindow(ROOT)
    assert window.main_tabs.count()==5
    assert window.main_tabs.widget(2) is window.mppi_page
    assert window.main_tabs.widget(3) is window.rehearsal_page
    assert window.main_tabs.widget(4) is window.flight_workspace
    assert not hasattr(window,'ppo_page')
    assert ('task','maximum_angle_deg') not in window.mppi_page.fields
    assert window.mppi_page.fields[('trajectory_objective','cast')].value()==200
    window.buttons[3].click();application.processEvents()
    assert window.inspector.active and window.title.text()=='Rehearsals'
    assert window.inspector.viewer is None
    window.close();application.processEvents()
