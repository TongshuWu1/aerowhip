from pathlib import Path
import numpy as np
import pytest
from experimental_data.adaptation_progress import load_study,comparison_arrays
from simulator.workflow import read_json

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'runs/adaptation/20260908-adp0-first'
MODEL=ROOT/'data/model_candidates/20260908-adp0-M1/model.json'
PARENT=ROOT/'runs/ppo/20260908-195207-486249-seed655/checkpoints/best_validation.pt'


def test_saved_progress_and_separate_heldout_predictions():
    if not STUDY.exists():pytest.skip('Local adaptation study required')
    s=load_study(ROOT,STUDY)
    assert len(s['rows'])==5 and s['models']==[MODEL]
    assert s['means']['adapted_tip_whip_rmse_m']==pytest.approx(.10417897558)
    baseline,heldout=comparison_arrays(s,'whip_adp_0_001','heldout')
    _,fitted=comparison_arrays(s,'whip_adp_0_001','all_five')
    assert not np.array_equal(heldout['cable'],fitted['cable'])
    np.testing.assert_array_equal(baseline['measured_sites'],heldout['measured_sites'])


def test_measured_round_metrics_do_not_use_fitted_predictions():
    from experimental_data.adaptation_progress import measured_flight_metrics
    time=np.arange(101)/100
    data=dict(time=time,metadata=dict(whip_end_s=1.,predicted_hit_time_s=.5),
        drone_error=np.full(101,.1),tip_error=np.full(101,.2),target_error=np.full(101,.3),
        measured_cable=np.broadcast_to([.3,0,0],(101,11,3)).copy(),target=np.zeros(3),
        tracking_span=(-.1,1.1),take='flight',hashes={})
    data['hover_normalized']={k:data[k].copy() for k in ['drone_error','tip_error','target_error','measured_cable']}
    data['height_calibration']=dict(bias_z_m=.05,checks_passed=True)
    data['drone_error'][:]=100;data['tip_error'][:]=200;data['target_error'][:]=300
    result=measured_flight_metrics(data)
    assert result['drone_rms_m']==pytest.approx(.1)
    assert result['tip_rms_m']==pytest.approx(.2)
    assert result['target_at_strike_m']==pytest.approx(.3)
    assert result['evaluation_frame']=='hover_normalized_z'
    data['hover_normalized']['measured_cable'][50,-1]=np.nan
    assert measured_flight_metrics(data)['target_at_strike_m'] is None
    del data['hover_normalized']
    with pytest.raises(ValueError,match='calibration required'):measured_flight_metrics(data)


def test_no_real_m1_flights_means_no_adapted_rms(tmp_path):
    from experimental_data.adaptation_progress import recorded_flight_progress
    (tmp_path/'runs/adaptation/fake/validation').mkdir(parents=True)
    (tmp_path/'runs/adaptation/fake/validation/results.json').write_text('{"adapted_tip_whip_rmse_m":0.01}')
    data=recorded_flight_progress(tmp_path)
    assert len(data['rows'])==1 and data['rows'][0]['model']=='M1'
    assert data['rows'][0]['means']['tip_rms_m'] is None
    assert data['rows'][0]['actual_flights']==0
