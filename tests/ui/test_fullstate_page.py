import os
from pathlib import Path
import zipfile

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication, QFileDialog
from deployment.fullstate import write_fullstate
from simulator.gui.fullstate_page import FullStatePage


def test_fullstate_table_and_transfer_bundle(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    root = Path(__file__).resolve().parents[2]
    (tmp_path/'config').mkdir()
    for name in ('model', 'task', 'ppo'):
        (tmp_path/'config'/f'{name}.json').write_bytes((root/'config'/f'{name}.json').read_bytes())
    plan = tmp_path/'plan_001'
    plan.mkdir()
    (plan/'plan.npz').write_bytes(b'source')
    t = np.arange(81)*.01
    q = np.zeros((81, 2, 3))
    q[:, :, 2] = 1.5
    write_fullstate(dict(time_s=t, positions_m=q, velocities_m_s=np.zeros_like(q)),
                    plan, dict(cutoff_s=.8))
    page = FullStatePage(tmp_path)
    page.directory = tmp_path
    page.show_plan(str(plan/'commands.csv'), {})
    assert page.fullstate_mode
    assert page.table.rowCount() == 25
    assert page.table.columnCount() == 12
    assert page.csv_path.name == 'fullstate_30hz.csv'
    assert page.execute.text() == 'Execute rehearsal & create CSV'
    assert page.table.item(24, 0).text() == '0.800000'
    destination = tmp_path/'transfer.zip'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **k: (str(destination), 'ZIP'))
    page.save_reference()
    with zipfile.ZipFile(destination) as archive:
        assert archive.read('fullstate_30hz.csv') == page.csv_path.read_bytes()
        assert {'fullstate.json', 'fullstate_source.npz', 'plan.npz'} <= set(archive.namelist())
    assert page.shutdown()
    page.close()
    app.processEvents()


def test_original_recovery_export_available_only_after_execution(tmp_path):
    from deployment.fullstate import write_rehearsal_fullstate
    app = QApplication.instance() or QApplication([])
    (tmp_path/'config').mkdir()
    root = Path(__file__).resolve().parents[2]
    for name in ('model','task','ppo'):
        (tmp_path/'config'/f'{name}.json').write_bytes((root/'config'/f'{name}.json').read_bytes())
    plan = tmp_path/'plan_001'
    plan.mkdir()
    (plan/'plan.npz').write_bytes(b'source')
    page = FullStatePage(tmp_path)
    page.directory = tmp_path
    page.show_plan(str(plan/'commands.csv'),dict(cutoff_s=.8))
    assert page.execute.isEnabled()
    assert not page.export_reference.isEnabled()
    assert page.csv_path is None
    assert not hasattr(page,'recovery_spins')
    t = np.arange(1201)*.01
    q = np.zeros((len(t),2,3)); q[:,:,2] = [1.5,1.4]
    arrays = dict(time_s=t+5,cable_node_position_world_m=q,
        cable_node_velocity_world_m_s=np.zeros_like(q),
        controller_phase=np.where(t<=.8,1,np.where(t<2,2,0)))
    summary = dict(events=[dict(event='strike',time_s=5.,planned_duration_s=.8)],
                   hover_position_m=[0,0,1.5])
    write_rehearsal_fullstate(arrays,summary,plan,dict(cutoff_s=.8))
    page.show_plan(str(plan/'fullstate_30hz.csv'),{})
    assert page.table.rowCount()==361
    assert page.table.columnCount()==14
    assert page.table.item(360,13).text()=='hover_hold'
    assert page.export_reference.isEnabled()
    assert not page.execute.isEnabled()
    assert page.experiment_setup()['reference_source']=='recorded_whip_gentle_recovery'
    recorded=np.load(plan/'fullstate_source.npz')
    np.testing.assert_array_equal(recorded['positions_m'],q)
    from deployment.gentle_recovery import replace_recorded_recovery
    replace_recorded_recovery(plan)
    page.show_plan(str(plan/'fullstate_30hz.csv'),{})
    assert page.plan_metadata['schema']=='gentle_recovery_fullstate_v4'
    assert page.open_reference_plot.isEnabled()
    assert 'Whip unchanged' in page.plan_note.text()
    assert page.shutdown()
    page.close()
    app.processEvents()
