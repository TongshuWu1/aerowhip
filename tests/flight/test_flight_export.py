import hashlib
import json
import zipfile
import pytest
from deployment.flight_export import export_flights


def rehearsal(root,name='M5-plan',height=1.3):
    path=root/'runs/rehearsals_pva'/name;path.mkdir(parents=True)
    header='time_s,px_m,py_m,pz_m,vx_m_s,vy_m_s,vz_m_s,ax_m_s2,ay_m_s2,az_m_s2,yaw_rad,yaw_rate_rad_s\n'
    payload=(header+f'0,0,0,{height},0,0,0,0,0,0,0,0\n'+f'{1/30},0,0,{height},0,0,0,0,0,0,0,0\n').encode()
    (path/'fullstate_30hz.csv').write_bytes(payload)
    meta=dict(schema='pva_fullstate_30hz_v1',recovery_prediction_complete=True,
              csv_sha256=hashlib.sha256(payload).hexdigest(),total_duration_s=1/30,initial_tracking_origin_m=[0,0,height])
    (path/'rehearsal.json').write_text(json.dumps(meta))
    (path/'model.json').write_text(json.dumps(dict(provenance=dict(generation_index=5))))
    return path


def test_batch_keeps_exact_commands_and_separate_recordings(tmp_path):
    a=rehearsal(tmp_path,'Curved',1.792);b=rehearsal(tmp_path,'Horizontal',1.288)
    active=tmp_path/'exports/CURRENT_FLIGHT.json';active.parent.mkdir();active.write_text('existing selection')
    out=export_flights([a,b],tmp_path/'exports/batch')
    with zipfile.ZipFile(out/'flight_commands.zip') as z:
        assert z.testzip() is None
        for source in (a,b):
            assert (out/source.name/'flight_take/M5').is_dir()
            assert (out/source.name/'fullstate_30hz.csv').read_bytes()==(source/'fullstate_30hz.csv').read_bytes()
            assert z.read(source.name+'/fullstate_30hz.csv')==(source/'fullstate_30hz.csv').read_bytes()
    assert active.read_text()=='existing selection'
    marker=out/'Curved/flight_take/M5/recording.csv';marker.write_text('real data')
    with pytest.raises(FileExistsError):export_flights([a,b],out)
    assert marker.read_text()=='real data'


@pytest.mark.parametrize('problem',['changed_csv','incomplete','wrong_duration'])
def test_invalid_second_selection_publishes_nothing(tmp_path,problem):
    a=rehearsal(tmp_path,'good');b=rehearsal(tmp_path,'bad')
    if problem=='changed_csv':(b/'fullstate_30hz.csv').write_text('changed')
    else:
        meta=json.loads((b/'rehearsal.json').read_text())
        meta['recovery_prediction_complete' if problem=='incomplete' else 'total_duration_s']=False if problem=='incomplete' else 99
        (b/'rehearsal.json').write_text(json.dumps(meta))
    out=tmp_path/'exports/batch'
    with pytest.raises(ValueError):export_flights([a,b],out)
    assert not out.exists()


def test_dialog_exports_checked_rows_and_keeps_error_visible(tmp_path,monkeypatch):
    import os
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from simulator.gui.flight_export_dialog import FlightExportDialog
    app=QApplication.instance() or QApplication([])
    a=rehearsal(tmp_path,'Curved');b=rehearsal(tmp_path,'Horizontal')
    dialog=FlightExportDialog(tmp_path,a)
    assert dialog.selected_paths()==[a]
    for row in range(dialog.table.rowCount()):dialog.table.item(row,0).setCheckState(Qt.CheckState.Checked)
    out=tmp_path/'exported';dialog.destination.setText(str(out));dialog.export_button.click()
    assert dialog.exported==out and dialog.open_button.isEnabled()
    assert (out/'Curved/fullstate_30hz.csv').is_file() and (out/'Horizontal/fullstate_30hz.csv').is_file()
    dialog.export_button.click()
    assert 'existing recordings' in dialog.status.text()
    dialog.close()
