import csv
import json
from pathlib import Path
import numpy as np
import pytest
from experimental_data.io import atomic_json,sha256_file
from deployment.research_rehearsal import FIELDS,complete_packets,export_package
from deployment.translate_rehearsal import translate_rehearsal


@pytest.fixture
def source(tmp_path):
    d=tmp_path/'source';d.mkdir();(d/'assets').mkdir();(d/'checkpoints').mkdir()
    for name in ['cable_residual.pt','drone_model.json','drone_residual.pt']:(d/'assets'/name).write_text('frozen')
    (d/'checkpoints/policy.pt').write_text('unchanged weights')
    whip=np.zeros((31,11));whip[:,:3]=[0,0,1.5];whip[-1,3:6]=[.2,0,1.]
    t,c,phase,recovery=complete_packets(whip,[0,0,1.5]);q=np.repeat(c[:,:3,None],2,axis=2).transpose(0,2,1);q[:,1,2]-=.9
    np.savez(d/'rehearsal.npz',commands=c,command_time_s=t,command_phase=phase,prediction_time_s=t,
        origin_positions_m=c[:,:3],origin_rotations=np.tile(np.eye(3),(len(t),1,1)),cable_positions_m=q,
        target_position_m=np.array([1,0,1.4]),force_time_s=t,virtual_force_n=np.zeros((len(t),3)))
    with (d/'fullstate_30hz.csv').open('w',newline='') as stream:
        w=csv.writer(stream);w.writerow(FIELDS);w.writerows(np.c_[t,c])
    (d/'virtual_force_30hz.csv').write_text('unchanged force bytes')
    atomic_json(d/'model.json',dict(motion_residual=dict(checkpoint='old/cable.pt'),fullstate_execution=dict(checkpoint='old/drone.json')))
    atomic_json(d/'ppo.json',dict(seed=123));atomic_json(d/'task.json',dict(initial_root_position_m=[0,0,1.445],target_position_m=[1,0,1.4]))
    atomic_json(d/'rehearsal.json',dict(schema='research_fullstate_30hz_v1',recovery=recovery,
        initial_tracking_origin_m=[0,0,1.5],initial_attachment_m=[0,0,1.445],target_position_m=[1,0,1.4],
        checkpoint_sha256=sha256_file(d/'checkpoints/policy.pt')))
    return d


def test_translation_preserves_all_nonposition_data_and_source(source,tmp_path):
    hashes={str(p):sha256_file(p) for p in source.rglob('*') if p.is_file()}
    out=tmp_path/'shifted';m=translate_rehearsal(source,out)
    a=np.load(source/'rehearsal.npz');b=np.load(out/'rehearsal.npz');delta=np.array([0,0,-.3])
    for key in a.files:
        if key=='commands':
            np.testing.assert_allclose(b[key][:,:3],a[key][:,:3]+delta,atol=0,rtol=0)
            np.testing.assert_array_equal(b[key][:,3:],a[key][:,3:])
        elif key in ['origin_positions_m','cable_positions_m','target_position_m']:
            np.testing.assert_array_equal(b[key],a[key]+delta)
        else:np.testing.assert_array_equal(b[key],a[key])
    for p,h in hashes.items():assert sha256_file(Path(p))==h
    for name in ['virtual_force_30hz.csv','ppo.json','checkpoints/policy.pt']:
        assert (out/name).read_bytes()==(source/name).read_bytes()
    old=list(csv.DictReader((source/'fullstate_30hz.csv').open()));new=list(csv.DictReader((out/'fullstate_30hz.csv').open()))
    for x,y in zip(old,new):
        assert all(x[k]==y[k] for k in FIELDS if k!='pz_m')
        assert float(y['pz_m'])==float(x['pz_m'])-.3
    assert m['initial_tracking_origin_m']==[0,0,1.2]
    np.testing.assert_allclose(m['target_position_m'],[1,0,1.1])
    assert m['csv_sha256']==sha256_file(out/'fullstate_30hz.csv')
    r=json.loads((source/'rehearsal.json').read_text())['recovery']
    np.testing.assert_allclose(np.asarray(m['recovery']['turn_coefficients_normalized'])[0],np.asarray(r['turn_coefficients_normalized'])[0]+delta)
    second=translate_rehearsal(out,tmp_path/'second',[0,0,-.1])
    assert second['translation']['cumulative_offset_world_m']==[0,0,-.4]
    assert second['translation']['original_planning_origin_m']==[0,0,1.5]


def test_refuses_overwrite_nested_output_and_invalid_offsets(source,tmp_path):
    for dest,offset in [(source,[0,0,-.3]),(source/'nested',[0,0,-.3]),(tmp_path/'bad',[0,float('nan'),0])]:
        with pytest.raises(ValueError):translate_rehearsal(source,dest,offset)
    assert not (source/'nested').exists() and not (tmp_path/'bad').exists()


def test_translated_package_explains_original_planning_coordinates(source,tmp_path):
    import zipfile
    out=tmp_path/'shifted';translate_rehearsal(source,out)
    archive=export_package(out,tmp_path/'result.zip')
    with zipfile.ZipFile(archive) as z:
        assert 'ORIGINAL planning coordinates' in z.read('README.txt').decode()
        assert 'tools/translate_rehearsal.py' in z.namelist()
        assert z.read('fullstate_30hz.csv')==(out/'fullstate_30hz.csv').read_bytes()


def test_ui_shift_creates_new_result_and_disables_stale_export(source,tmp_path):
    import os,shutil
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    from simulator.gui.rehearsal_workspace import RehearsalWorkspace
    root=Path(__file__).resolve().parents[2]
    shutil.copytree(root/'config',tmp_path/'config')
    m=json.loads((source/'rehearsal.json').read_text());m.update(checkpoint='saved.pt',whip_end_s=1.,
        total_duration_s=10.,predicted_valid_hit=True,minimum_tip_distance_m=.04,recovery_prediction_complete=True)
    atomic_json(source/'rehearsal.json',m)
    app=QApplication.instance() or QApplication([]);page=RehearsalWorkspace(tmp_path)
    assert not page.shift_button.isEnabled()
    page.load_result(source);assert page.shift_button.isEnabled()
    page.shift_button.click();assert page.directory!=source
    assert page.metadata['translation']['offset_world_m']==[0,0,-.3]
    assert page.start_spins[2].value()==1.2 and page.target_spins[2].value()==1.1
    assert page.save.isEnabled() and page.viewer is None
    page.start_spins[0].setValue(.1)
    assert not page.shift_button.isEnabled() and not page.save.isEnabled()
    page.close();app.processEvents()
