import numpy as np
import pytest
from experimental_data.adaptation_check import command_onset, interpolate_positions, find_rehearsal, flight_names
from experimental_data.adaptation_rounds import COMMAND_COLUMNS


def test_command_time_uses_reception_age_not_first_logged_row():
    ref = np.zeros(20,dtype=[('time_s',float)]+[(f'v{i}',float) for i in range(11)])
    ref['time_s']=np.arange(20)/30
    ref['v0']=np.linspace(-2,-1,20); ref['v6']=np.arange(20)+1
    c=np.zeros(60,dtype=[('time_s',float),('cmd_age',float),('cmd_valid',float)]+[(n,float) for n in COMMAND_COLUMNS])
    ids=np.repeat(np.arange(20),3); age=np.tile([.003,.013,.023],20)
    c['time_s']=18+ref['time_s'][ids]+age; c['cmd_age']=age; c['cmd_valid']=1
    for j,name in enumerate(COMMAND_COLUMNS): c[name]=ref[f'v{j}'][ids]
    onset,count,jitter=command_onset(c,ref)
    assert onset==pytest.approx(18); assert count==20; assert jitter<1e-10
    # Two executions mixed in one log must not silently average their onset.
    c['time_s'][30:]+=3
    with pytest.raises(ValueError,match='Ambiguous'): command_onset(c,ref)


def test_missing_observations_and_time_gaps_are_not_bridged():
    t=np.array([0,.01,.02,.08,.09])
    p=np.repeat(t[:,None],3,axis=1); p[1]=np.nan
    out=interpolate_positions(t,p,np.array([0,.005,.01,.02,.04,.085,.1]))
    np.testing.assert_array_equal(out[0],p[0]); np.testing.assert_array_equal(out[3],p[2])
    assert np.isnan(out[[1,2,4,6]]).all()
    np.testing.assert_allclose(out[5],[.085]*3)


def test_command_onset_rejects_negative_age_and_ambiguous_repeated_values():
    ref=np.zeros(20,dtype=[('time_s',float)]+[(f'v{i}',float) for i in range(11)])
    ref['time_s']=np.arange(20)/30;ref['v0']=np.arange(20);ref['v6']=1
    c=np.zeros(20,dtype=[('time_s',float),('cmd_age',float),('cmd_valid',float)]+[(n,float) for n in COMMAND_COLUMNS])
    c['time_s']=10+ref['time_s'];c['cmd_valid']=1;c['cmd_age']=-.01
    for i,n in enumerate(COMMAND_COLUMNS):c[n]=ref[f'v{i}']
    with pytest.raises(ValueError,match='No valid'):command_onset(c,ref)
    c['cmd_age']=0;ref['v0']=1;c[COMMAND_COLUMNS[0]]=1
    with pytest.raises(ValueError,match='Too few matching'):command_onset(c,ref)


def test_rehearsal_requires_exact_flown_csv(tmp_path):
    ref=tmp_path/'flight.csv'; ref.write_text('flown')
    saved=tmp_path/'saved'; saved.mkdir(); (saved/'fullstate_30hz.csv').write_text('other')
    with pytest.raises(ValueError,match='No saved prediction'):find_rehearsal(tmp_path,ref,saved)
    (saved/'fullstate_30hz.csv').write_text('flown')
    assert find_rehearsal(tmp_path,ref,saved)==saved


def test_unpaired_logs_not_selectable(tmp_path):
    folder=tmp_path/'flight_take'; folder.mkdir()
    for name in ['a','experiment_a','b','experiment_c']:(folder/(name+'.csv')).touch()
    assert flight_names(tmp_path)==['a']


def test_missing_marker_breaks_both_adjacent_segments():
    from simulator.gui.adaptation_scene import finite_segments
    p=np.ones((5,3)); p[2]=np.nan
    coords,lines=finite_segments(p)
    assert np.isfinite(coords).all()
    np.testing.assert_array_equal(lines,[2,0,1,2,3,4])
    _,lines=finite_segments(np.ones((4,3)),np.array([0,.01,.1,.11]))
    np.testing.assert_array_equal(lines,[2,0,1,2,2,3])


def test_command_reader_does_not_require_or_expose_controller_state(tmp_path):
    from experimental_data.adaptation_rounds import read_controller
    columns=['time_s','cmd_age','cmd_valid',*COMMAND_COLUMNS]
    p=tmp_path/'experiment_test.csv'
    values=np.zeros((4,len(columns))); values[:,0]=np.arange(4)*.01
    values[:,2]=1
    np.savetxt(p,values,delimiter=',',header=','.join(columns),comments='')
    expected=read_controller(p,commands_only=True)
    contaminated=np.column_stack([values,np.full((4,6),np.nan)])
    np.savetxt(p,contaminated,delimiter=',',header=','.join(columns+['x','y','z','vx','vy','vz']),comments='')
    actual=read_controller(p,commands_only=True)
    assert actual.dtype.names==tuple(columns)
    for key in columns: np.testing.assert_array_equal(actual[key],expected[key])


def test_explicit_event_log_keeps_commands_missing_from_tf_snapshots(tmp_path):
    from experimental_data.adaptation_rounds import read_controller
    from experimental_data.preliminary_prepare import recorded_packets
    p=tmp_path/'experiment_test.csv'
    columns=['time_s','cmd_age','cmd_valid',*COMMAND_COLUMNS]
    values=np.zeros((6,len(columns)));values[:,0]=10+np.arange(6)/30
    values[:,2]=1;values[:,3]=np.arange(6)
    np.savetxt(p,values[[0,2,5]],delimiter=',',header=','.join(columns),comments='')
    event_columns=['time_s','cmd_sequence','cmd_valid',*COMMAND_COLUMNS]
    events=values.copy();events[:,1]=np.arange(101,107)
    ep=p.with_suffix('.commands.csv')
    np.savetxt(ep,events,delimiter=',',header=','.join(event_columns),comments='')
    assert len(read_controller(p,commands_only=True))==3  # Legacy/default unchanged.
    actual=read_controller(p,commands_only=True,command_source='event_log')
    times,packets,_,_=recorded_packets(actual,invalid_rows_break_coverage=True)
    assert len(actual)==6
    # Final receipt has no observed hold after EOF; do not fabricate coverage.
    np.testing.assert_allclose(times,values[:-1,0]);np.testing.assert_array_equal(packets[:,0],np.arange(5))
    assert not actual['cmd_age'].any()
    with pytest.raises(ValueError,match='not measured'):
        read_controller(p,command_source='event_log')
    events[3,1]+=1
    np.savetxt(ep,events,delimiter=',',header=','.join(event_columns),comments='')
    with pytest.raises(ValueError,match='gaps or duplicates'):
        read_controller(p,commands_only=True,command_source='event_log')


def test_explicit_alignment_requires_exact_sources_and_never_fits_motion(tmp_path):
    import json
    from experimental_data.adaptation_check import recorded_alignment, sha256
    tracking=tmp_path/'take.csv'; tracking.write_text('raw tracking')
    controller=tmp_path/'experiment_take.csv'; controller.write_text('commands only')
    with pytest.raises(ValueError,match='Clock alignment required'):
        recorded_alignment(tmp_path,tmp_path,'take',tracking,controller)
    entry=dict(offset_s=2.5,source='shared timestamp event',clock_verified=True,
               optitrack_sha256=sha256(tracking),controller_sha256=sha256(controller))
    (tmp_path/'time_alignment.json').write_text(json.dumps({'take':entry}))
    result=recorded_alignment(tmp_path,tmp_path,'take',tracking,controller)
    assert result['offset_s']==2.5 and result['rms_m'] is None
    tracking.write_text('different trim')
    with pytest.raises(ValueError,match='source changed'):
        recorded_alignment(tmp_path,tmp_path,'take',tracking,controller)


def test_replay_measurement_and_errors_use_optitrack_only(tmp_path,monkeypatch):
    import json
    import experimental_data.adaptation_check as ac
    batch=tmp_path/'batch'; flights=batch/'flight_take'; flights.mkdir(parents=True)
    saved=tmp_path/'saved'; saved.mkdir(); (batch/'simulation_csv').mkdir()
    tracking=flights/'take.csv'; tracking.write_text('tracking fixture')
    ct=flights/'experiment_take.csv'
    t=np.arange(20)/30
    commands=np.zeros((20,11)); commands[:,0]=t; commands[:,6]=1
    ref=np.column_stack([t,commands]); header=','.join(['time_s']+[f'v{i}' for i in range(11)])
    for path in [saved/'fullstate_30hz.csv',batch/'simulation_csv/fullstate_30hz.csv']:
        np.savetxt(path,ref,delimiter=',',header=header,comments='')
    # No measured state columns exist in this controller file.
    np.savetxt(ct,np.column_stack([t+10,np.zeros(20),np.ones(20),commands]),delimiter=',',
               header=','.join(['time_s','cmd_age','cmd_valid',*COMMAND_COLUMNS]),comments='')
    (batch/'time_alignment.json').write_text(json.dumps({'take':dict(offset_s=10,
        source='shared clock',clock_verified=True,optitrack_sha256=ac.sha256(tracking),controller_sha256=ac.sha256(ct))}))
    model={'recorded_data':{'optitrack_to_attachment_offset_body_m':[0,0,-.055]},
           'cable':{'segments_per_marker_interval':[1]*10}}
    for name,value in [('model',model),('task',{}),('rehearsal',{})]:
        (saved/(name+'.json')).write_text(json.dumps(value))
    np.savez(saved/'rehearsal.npz',commands=commands,command_time_s=t,prediction_time_s=t,
        origin_positions_m=np.zeros((20,3)),origin_rotations=np.tile(np.eye(3),(20,1,1)),
        cable_positions_m=np.zeros((20,11,3)),target_position_m=np.zeros(3))
    measured=np.tile([2.,3.,4.],(20,1)); cable=np.ones((20,10,3)); cable[5,4]=np.nan
    monkeypatch.setattr(ac,'read_optitrack',lambda _:dict(time=t,drone=measured,
        quaternion=np.tile([0,0,0,1.],(20,1)),cable=cable))
    result=ac.load_comparison(tmp_path,batch,'take',saved)
    np.testing.assert_array_equal(result['measured_origin'],measured)
    np.testing.assert_allclose(result['drone_error'],np.sqrt(29))
    assert np.isnan(result['measured_cable'][5,5]).all()
    np.testing.assert_array_equal(result['predicted_origin'],np.zeros((20,3)))
    # The batch must choose its frozen prediction even when no folder is passed.
    protocol=batch/'protocol.json'
    protocol.write_text(json.dumps(dict(rehearsal=str(saved),forecast_sha256=ac.sha256(saved/'rehearsal.npz'))))
    implicit=ac.load_comparison(tmp_path,batch,'take')
    assert implicit['rehearsal']==saved
    assert str(protocol) in implicit['hashes']
    protocol.write_text(json.dumps(dict(rehearsal=str(saved),forecast_sha256='wrong')))
    with pytest.raises(ValueError,match='frozen forecast'):ac.load_comparison(tmp_path,batch,'take')
