import numpy as np
from experimental_data.hover_calibration import summarize_hold,combine_holds,corrected_tracking


def test_vertical_normalization_preserves_geometry_gaps_and_raw():
    drone=np.array([[0.,0.,1.20],[1,0,1.21]])
    cable=np.array([[[0.,0.,1.1],[0,0,np.nan]],[[1,0,1.11],[1,0,.8]]])
    m=dict(drone=drone,cable=cable,quaternion=np.ones((2,4)))
    corrected=corrected_tracking(m,-.05)
    np.testing.assert_allclose(corrected['drone'][:,2],[1.25,1.26])
    np.testing.assert_allclose(corrected['cable']-corrected['drone'][:,None],cable-drone[:,None],equal_nan=True)
    assert drone[0,2]==1.2 and np.isnan(corrected['cable'][0,1,2])
    np.testing.assert_array_equal(corrected['quaternion'],m['quaternion'])


def test_hold_estimator_ignores_maneuver_and_combines_phases_equally():
    t=np.arange(-10,11,.01);cmd=np.full(t.shape,1.25);z=cmd+.05
    z[(t>=0)&(t<8)]+=1
    good=np.ones(t.shape,bool)
    pre=summarize_hold(t,z,cmd,good,-8,-.02);post=summarize_hold(t,z,cmd,good,9,10.8)
    r=combine_holds([dict(take='a',pre=pre,post=post)])
    assert r['checks_passed'] and abs(r['bias_z_m']-.05)<1e-12
    bad=dict(post,median_bias_m=.15)
    assert not combine_holds([dict(take='a',pre=pre,post=bad)])['checks_passed']

def test_explicit_take_exclusion_preserves_files(tmp_path):
    import json
    from experimental_data.adaptation_check import flight_names
    folder=tmp_path/'flight_take';folder.mkdir()
    for name in ['whip_adp_1_001','whip_adp_1_005']:
        (folder/(name+'.csv')).write_text('raw')
        (folder/('experiment_'+name+'.csv')).write_text('raw')
    (tmp_path/'excluded_takes.json').write_text(json.dumps({'excluded':{'whip_adp_1_005':'user excluded'}}))
    assert flight_names(tmp_path)==['whip_adp_1_001']
    assert (folder/'whip_adp_1_005.csv').read_text()=='raw'

def test_reviewed_per_take_bias_preserves_failed_screening(tmp_path):
    import json
    import pytest
    from experimental_data.hover_calibration import load_calibration,normalized_evaluation
    f=tmp_path/'flight_take';f.mkdir()
    for n in ['a','b']:
        (f/(n+'.csv')).write_text('raw');(f/('experiment_'+n+'.csv')).write_text('raw')
    r=dict(schema='batch_hover_z_calibration_v1',checks_passed=False,warnings=['drift remains'],takes=[{'take':'a'},{'take':'b'}],source_hashes={},bias_z_m=-.08,mode='constant_per_take',bias_by_take_m={'a':-.09,'b':-.06})
    p=tmp_path/'height_calibration.json';p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='review'):load_calibration(tmp_path,'a')
    r['normalization_review']=dict(accepted=True,acknowledged_remaining_drift=True);p.write_text(json.dumps(r))
    a=load_calibration(tmp_path,'a');b=load_calibration(tmp_path,'b')
    assert a['bias_z_m']==-.09 and b['bias_z_m']==-.06 and not a['checks_passed']
    assert normalized_evaluation(dict(height_calibration=a,hover_normalized={'drone_error':[.1]}))['drone_error']==[.1]
    with pytest.raises(ValueError,match='no vertical'):load_calibration(tmp_path,'absent')
