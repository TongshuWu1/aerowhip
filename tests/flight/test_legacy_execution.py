import csv
import json

import numpy as np
import pytest

from experimental_data.adaptation_rounds import COMMAND_COLUMNS, import_round, process_round, read_controller
from experimental_data.legacy_execution import (CSV, PRE, POST, UNKNOWN, classify_execution,
    create_profile, read_sequence)
from tests.flight.test_adaptation_rounds import make_pair


def fixture_stream():
    t=np.arange(160)*.01+.2
    values=np.full((len(t),11),np.nan)
    fresh=(t>=.5)&(t<1.3)
    values[fresh]=[0,0,1.5,0,0,0,0,0,0,0,0]
    expected=np.array([[.2,0,1.5,0,0,0,0,0,0,0,0],
                       [.3,0,1.5,1,0,0,2,0,0,0,0],
                       [.4,0,1.5,0,0,0,0,0,0,0,0]])
    for a,b,row in zip([.8,.84,.87],[.84,.87,.9],expected):
        values[(t>=a)&(t<b)]=row
    values[(t>=.9)&fresh]=[.7,0,1.7,0,0,0,0,0,0,0,0]
    profile=dict(sequence_fullstate=expected.tolist(),sequence_time_s=[0,.04,.07],
        source_constants={'RECOVERY_TIME':5},post_sequence_source_behavior='Hold measured end position')
    return t,values,fresh,np.zeros(len(t)),profile


def test_zero_va_csv_rows_and_holds_are_distinct():
    t,v,fresh,age,profile=fixture_stream()
    phase,index,report=classify_execution(t,v,fresh,age,profile)
    for j in range(3):
        assert np.any(index==j)
        assert np.all(phase[index==j]==CSV)
    assert set(phase)=={UNKNOWN,PRE,CSV,POST}
    assert np.all(index[(phase==PRE)|(phase==POST)]==-1)
    assert report['csv_start_s']==pytest.approx(.8)
    assert report['csv_end_s']==pytest.approx(.9)
    assert report['post_hold_targets_m']==[[.7,0,1.7]]
    np.testing.assert_array_equal(v[index==2][0],profile['sequence_fullstate'][2])


@pytest.mark.parametrize('damage',['missing','gap','reordered','ambiguous','no_end'])
def test_incomplete_or_ambiguous_evidence_fails_for_review(damage):
    t,v,fresh,age,profile=fixture_stream()
    if damage=='missing':v[(t>=.84)&(t<.87),6]=9
    if damage=='gap':fresh[np.flatnonzero((t>.8)&(t<.84))[0]]=False
    if damage=='reordered':
        v[(t>=.84)&(t<.87)]=profile['sequence_fullstate'][2]
        v[(t>=.87)&(t<.9)]=profile['sequence_fullstate'][1]
    if damage=='ambiguous':
        for a,b,row in zip([1.,1.04,1.07],[1.04,1.07,1.10],profile['sequence_fullstate']):v[(t>=a)&(t<b)]=row
    if damage=='no_end':fresh[t>=.9]=False
    with pytest.raises(ValueError):classify_execution(t,v,fresh,age,profile)


def source_files(tmp_path,profile):
    seq=[(t,row[:3],row[3:6],row[6:9],row[9],row[10]) for t,row in zip(profile['sequence_time_s'],profile['sequence_fullstate'])]
    controller=tmp_path/'controller.py'
    controller.write_text(f'PVA_SEQUENCE = {seq!r}\nNOMINAL_SAMPLE_RATE = 30.\nPRE_TRAJECTORY_HOVER_TIME = 10.\nRECOVERY_TIME = 5.\nraise RuntimeError("must never execute")\n')
    logger=tmp_path/'logger.py';logger.write_text('raise RuntimeError("must never execute")')
    model=tmp_path/'model.json';model.write_text(json.dumps({'recorded_data':{'optitrack_to_attachment_offset_body_m':[.006,-.012,-.055]}}))
    return controller,logger,model


def test_versioned_round_preserves_commands_geometry_and_full_coverage(tmp_path):
    source=make_pair(tmp_path/'input')
    # Reuse the synthetic position signal for a known +0.4 s alignment.
    c=read_controller(source/'experiment_sample.csv')
    t,v,fresh,age,profile=fixture_stream()
    indices=np.rint((c['time_s']-.2)/.01).astype(int)
    command=source/'experiment_sample.csv'
    with command.open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(c.dtype.names)
        for row,k in zip(c,indices):
            writer.writerow([row['time_s'],row['x'],row['y'],row['z'],0 if fresh[k] else 1,int(fresh[k]),*v[k]])
    raw=command.read_bytes()
    directory=import_round(tmp_path,source,0)
    (directory/'ARCHIVED.json').write_text('{"reason":"historical exclusion"}')
    (directory/'PREPARATION_AUTHORIZATION.json').write_text(json.dumps(dict(
        schema='recording_preparation_authorization_v1',allow_preparation=True)))
    create_profile(directory,*source_files(tmp_path,profile))
    output=process_round(directory)
    report=json.loads((output/'processing.json').read_text())['reports'][0]
    assert report['status']!='failed',report
    assert not report['training_ready']
    with np.load(output/'sample/dataset.npz',allow_pickle=False) as d:
        assert len(d['optitrack_time_s'])==101
        assert len(d['controller_native_time_s'])==len(c)
        np.testing.assert_array_equal(d['controller_native_fullstate'],v[indices])
        np.testing.assert_array_equal(d['controller_native_command_age_s'],np.where(fresh[indices],0,1))
        expected=d['drone_position_m']+np.array([.006,-.012,-.055])
        np.testing.assert_allclose(d['attachment_position_m'],expected)
        # Commands are held from earlier rows and never transformed to attachment.
        at=np.flatnonzero((d['execution_phase']==POST)&d['reference_valid'])[0]
        np.testing.assert_array_equal(d['reference_fullstate'][at,:3],[.7,0,1.7])
        assert np.isnan(d['reference_fullstate'][~d['reference_valid']]).all()
        assert np.all(d['csv_sample_index'][~d['csv_maneuver_mask']]==-1)
        assert np.any(d['phase_boundary_uncertain'])
    assert (directory/'raw/sample/experiment_sample.csv').read_bytes()==raw
    assert (directory/'ARCHIVED.json').read_text()=='{"reason":"historical exclusion"}'
    second=process_round(directory)
    assert second!=output and (output/'sample/dataset.npz').is_file()
    # Profile checksum guards the supplied source; existing versions stay usable.
    profile_path=directory/'execution_profile.json'
    info=json.loads(profile_path.read_text())
    (directory/info['sources']['controller']['path']).write_text('tampered')
    with pytest.raises(ValueError,match='checksum'):process_round(directory)


def test_source_is_literal_and_duplicate_rows_are_not_guessed(tmp_path):
    *_,profile=fixture_stream()
    controller,_,_=source_files(tmp_path,profile)
    times,values,_=read_sequence(controller)
    assert len(times)==3 and values.shape==(3,11)
    controller.write_text('PVA_SEQUENCE = compute_sequence()')
    with pytest.raises(ValueError):read_sequence(controller)
