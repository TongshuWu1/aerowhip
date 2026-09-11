import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experimental_data.adaptation_rounds import (read_optitrack,import_round,process_round,
    save_review,derivative,align,COMMAND_COLUMNS)


def make_pair(folder):
    folder.mkdir()
    t=np.arange(101)*.01
    # Intentionally lexicographic marker order, plus an irrelevant unlabeled marker.
    names=['cf_7']*7+[name for name in ['cable1:c1','cable1:c10',*[f'cable1:c{k}' for k in range(2,10)],'Unlabeled 123'] for _ in range(3)]
    types=['Rotation']*4+['Position']*(len(names)-4)
    axes=list('XYZW')+list('XYZ')+list('XYZ')*11
    with (folder/'sample.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['Length Units','Meters','Coordinate Space','Global','Rotation Type','Quaternion']);w.writerow([])
        w.writerow(['','']+['Marker']*len(names));w.writerow(['','']+names);w.writerow(['','']+['id']*len(names));w.writerow(['','']+types);w.writerow(['Frame','Time (Seconds)']+axes)
        for i,x in enumerate(t):
            data=[i,x,0,0,0,1,x,x*x,1.5]
            for name in ['cable1:c1','cable1:c10',*[f'cable1:c{k}' for k in range(2,10)],'Unlabeled 123']:
                k=int(name.split(':c')[-1]) if ':c' in name else 999
                data += ['', '', ''] if k==3 and i==50 else [x,0,1.5-k*.08]
            w.writerow(data)
    with (folder/'experiment_sample.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['time_s','x','y','z','cmd_age','cmd_valid',*COMMAND_COLUMNS])
        for x in np.arange(-.2,1.21,.01):
            w.writerow([x+.4,x,x*x,1.5,0 if x<.8 else 1,1,*([x,0,1.5,1,0,0,0,0,0,0,0])])
    return folder


def test_numeric_marker_order_masks_alignment_and_versioning(tmp_path):
    source=make_pair(tmp_path/'input')
    m=read_optitrack(source/'sample.csv')
    np.testing.assert_allclose(m['cable'][0,:,2],1.5-np.arange(1,11)*.08)
    assert np.isnan(m['cable'][50,2]).all()
    raw=(source/'sample.csv').read_bytes()
    r=import_round(tmp_path,source,0,notes='synthetic')
    with pytest.raises(ValueError,match='already exists'):import_round(tmp_path,source,0)
    p=process_round(r);report=json.loads((p/'processing.json').read_text())['reports'][0]
    assert report['status']!='failed',report
    assert report['alignment']['offset_s']==pytest.approx(.4,abs=1e-4)
    with np.load(p/'sample/dataset.npz',allow_pickle=False) as d:
        assert d['cable_position_m'].shape==(101,10,3)
        assert not d['cable_valid'][50,2]
        assert np.isnan(d['cable_velocity_m_s'][49:52,2]).all()
        assert not d['reference_valid'][-1]
        assert np.isnan(d['reference_fullstate'][-1]).all()
    assert (r/'raw/sample/sample.csv').read_bytes()==raw
    p2=process_round(r);assert p2!=p and p.is_dir()
    review=save_review(r,p2.name,'sample',role='adaptation',outcome='failure',notes='kept',
                      precontact_start_s=.5,precontact_end_s=.8)
    assert json.loads(review.read_text())['training_ready'] is False
    with pytest.raises(ValueError):save_review(r,p2.name,'sample',role='validation',outcome='success',notes='',precontact_start_s=-1,precontact_end_s=9)


def test_hash_tampering_and_protected_input_rejected(tmp_path):
    source=make_pair(tmp_path/'input');r=import_round(tmp_path,source,0)
    with (r/'raw/sample/sample.csv').open('a') as f:f.write('tamper')
    out=process_round(r)
    report=json.loads((out/'processing.json').read_text())['reports'][0]
    assert report['status']=='failed' and 'SHA' in report['error']
    with pytest.raises(ValueError,match='Protected'):
        read_optitrack(tmp_path/'fig8vertical_002.csv')


@pytest.mark.parametrize('label',['cf_3','cf3','cf_7'])
def test_drone_identity_auto_detection_preserves_xyz_quaternion_and_cable(tmp_path,label):
    source=make_pair(tmp_path/'input');path=source/'sample.csv'
    before=read_optitrack(path)
    rows=list(csv.reader(path.open()))
    rows[3]=[label if v=='cf_7' else v for v in rows[3]]
    # Move the seven pose columns after the cable markers: column indices are not fixed.
    order=[0,1,*range(9,len(rows[3])),*range(2,9)]
    for i in range(2,len(rows)):
        rows[i]=[rows[i][j] for j in order]
    with path.open('w',newline='') as f:csv.writer(f).writerows(rows)
    after=read_optitrack(path)
    assert after['drone_label']==label
    for key in ['time','drone','quaternion','cable']:
        np.testing.assert_array_equal(before[key],after[key])


def test_multiple_drones_require_selection_and_bind_it_to_file(tmp_path):
    from experimental_data.io import sha256_file
    source=make_pair(tmp_path/'input');path=source/'sample.csv'
    rows=list(csv.reader(path.open()))
    for i in range(2,len(rows)):
        extra=rows[i][2:9].copy()
        if i==3:extra=['cf_3']*7
        if i>=7:extra[4:]=['3','4','5']
        rows[i]+=extra
    with path.open('w',newline='') as f:csv.writer(f).writerows(rows)
    with pytest.raises(ValueError,match='explicitly'):read_optitrack(path)
    selected=read_optitrack(path,drone_label='cf_3')
    np.testing.assert_allclose(selected['drone'],np.tile([3,4,5],(101,1)))
    with pytest.raises(ValueError,match='no complete'):read_optitrack(path,drone_label='cf_9')
    path.with_suffix('.tracking.json').write_text(json.dumps(dict(drone='cf_3',optitrack_sha256=sha256_file(path))))
    assert read_optitrack(path)['drone_label']=='cf_3'
    with path.open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='different OptiTrack file'):read_optitrack(path)


def test_derivative_does_not_bridge_time_gap():
    t=np.array([0,.01,.02,.1,.11,.12]);p=np.column_stack([t,t,t])
    v=derivative(t,p)
    assert np.isnan(v[[0,2,3,5]]).all()
    np.testing.assert_allclose(v[[1,4]],1)


def test_still_recording_is_not_automatically_aligned():
    with pytest.raises(ValueError,match='Insufficient'):
        align(dict(time=np.arange(100)*.01,drone=np.zeros((100,3))),
              np.zeros(100,dtype=[(n,float) for n in ['time_s','x','y','z']]))


def test_reviews_do_not_collide_on_same_clock_tick(tmp_path,monkeypatch):
    from experimental_data import adaptation_rounds
    folder=tmp_path/'processed/version/trial';folder.mkdir(parents=True)
    (folder/'quality.json').write_text('{}')
    monkeypatch.setattr(adaptation_rounds,'stamp',lambda:'same-tick')
    a=save_review(tmp_path,'version','trial',role='unassigned',outcome='unreviewed',notes='first')
    b=save_review(tmp_path,'version','trial',role='unassigned',outcome='unreviewed',notes='second')
    assert a!=b
    assert b.name>a.name
    assert json.loads(a.read_text())['notes']=='first'
    assert json.loads(b.read_text())['notes']=='second'


def test_archived_round_rejects_processing_before_reading_metadata(tmp_path):
    import pytest
    from experimental_data.adaptation_rounds import process_round
    folder=tmp_path/'archived';folder.mkdir()
    (folder/'round.json').write_text('invalid metadata must never be read')
    (folder/'ARCHIVED.json').write_text('{}')
    with pytest.raises(ValueError,match='archived'):process_round(folder)
