import json
import os
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication
from simulator.gui.testing_page import TestingPage as RehearsalPage

ROOT = Path(__file__).resolve().parents[2]


def test_controller_settings_persist_across_page_restart(tmp_path):
    app = QApplication.instance() or QApplication([])
    (tmp_path/'config').mkdir()
    (tmp_path/'config/controller_export.json').write_bytes((ROOT/'config/controller_export.json').read_bytes())
    page = RehearsalPage(tmp_path)
    assert page.controller_mass.value() == .175
    assert page.controller_gravity.value() == 9.80665
    page.controller_mass.setValue(.158)
    page.controller_gravity.setValue(9.81)
    page.save_controller.click()
    page.close()
    restored = RehearsalPage(tmp_path)
    assert restored.experiment_setup()['controller_export'] == dict(
        controller_mass_kg=.158, controller_gravity_m_s2=9.81)
    restored.controller_mass.setValue(.156)
    restored.controller_mass.editingFinished.emit()
    assert json.loads((tmp_path/'config/controller_export.json').read_text())['controller_mass_kg'] == .156
    restored.close()
    app.processEvents()


def test_testing_page_handles_no_policies_and_csv_review(tmp_path):
    app = QApplication.instance() or QApplication([])
    (tmp_path/'config').mkdir()
    for name in ('model','task','ppo'):
        (tmp_path/'config'/f'{name}.json').write_bytes((ROOT/'config'/f'{name}.json').read_bytes())
    page = RehearsalPage(tmp_path)
    assert not page.start.isEnabled()
    assert not page.export.isEnabled()
    assert not page.execute.isEnabled()
    assert page.controller_settings() == {}
    page.controller_mass.setValue(.159)
    assert page.experiment_setup()['controller_export']['controller_mass_kg'] == .159
    page.set_running(True)
    assert not page.controller_mass.isEnabled()
    page.set_running(False)
    page.set_page_active(True)
    assert page.viewer is not None
    assert page.worker is None
    csv = tmp_path/'commands.csv'
    csv.write_text('time_s,until_s,fx_n,fy_n,fz_n\n0,0.05,0.1,0,1.7\n0.05,0.08,-0.1,0,1.7\n')
    page.show_plan(str(csv), dict(cutoff_s=.08, planning_wall_seconds=.3, target_position_m=[1,0,1.4]))
    assert page.table.rowCount() == 2
    assert page.table.item(1,1).text() == '0.080000'
    assert page.open_csv.isEnabled()
    assert not page.open_controller_csv.isEnabled()
    assert page.csv_path.name == 'commands.csv'
    assert 'configure controller mass' in page.csv_convention.text()
    assert page.open_csv.text() == 'Open total thrust CSV'
    (tmp_path/'controller_acceleration.csv').write_text('time_s,until_s,ax_ff_m_s2,ay_ff_m_s2,az_ff_m_s2\n')
    page.show_plan(str(csv), dict(cutoff_s=.08, planning_wall_seconds=.3, target_position_m=[1,0,1.4]))
    assert page.open_controller_csv.isEnabled()
    force_csv = tmp_path/'controller_force.csv'
    force_csv.write_text('time_s,until_s,fx_ff_n,fy_ff_n,fz_ff_n\n0,0.05,0.1,0,0.140333\n0.05,0.08,-0.1,0,-0.2\n')
    metadata = dict(cutoff_s=.08, planning_wall_seconds=.3, target_position_m=[1,0,1.4],
                    controller_export=dict(mass_kg=.159, gravity_m_s2=9.80665))
    page.show_plan(str(csv), metadata)
    assert page.csv_path == force_csv
    assert page.table.item(0,4).text() == '0.140333'
    assert page.table.item(1,4).text() == '-0.200000'
    assert page.open_csv.text() == 'Open controller force CSV'
    assert 'Subtracted' in page.csv_convention.text()
    page.csv_kind.setCurrentIndex(1)
    assert page.csv_path.name == 'controller_acceleration.csv'
    assert page.table.horizontalHeaderItem(4).text() == 'Az (m/s²)'
    page.csv_kind.setCurrentIndex(2)
    assert page.csv_path == csv
    assert page.table.item(0,4).text() == '1.700000'
    assert page.table.horizontalHeaderItem(4).text() == 'Fz (N)'
    assert page.shutdown()
    page.close()
    app.processEvents()
