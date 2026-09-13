import numpy as np
from deployment.pva_rehearsal import complete_pva_packets
from experimental_data.adaptation_check import find_rehearsal


def test_nearly_settled_exit_does_not_create_a_large_recovery_loop():
    hover=np.array([-2.,0.,1.255]);whip=np.zeros((7,11));whip[:,:3]=hover
    whip[-1,0]+=.001;whip[-1,3]=.005;whip[-1,6]=.01
    original=whip.copy();times,packets,phases,recovery=complete_pva_packets(whip,hover)
    np.testing.assert_array_equal(packets[:len(whip)],original)
    assert recovery['profile']=='gentle_near_hover_return'
    assert np.linalg.norm(packets[:,:3]-hover,axis=-1).max()<.01
    np.testing.assert_allclose(packets[-1,:3],hover,atol=1e-12)
    np.testing.assert_allclose(packets[-1,3:9],0,atol=1e-12)
    np.testing.assert_allclose(np.diff(times),1/30,atol=1e-12)
    assert phases[-1]==4


def test_new_flight_csv_matches_original_pva_ghost(tmp_path):
    folder=tmp_path/'runs/rehearsals_pva/a';folder.mkdir(parents=True)
    (folder/'rehearsal.npz').write_bytes(b'placeholder')
    (folder/'fullstate_30hz.csv').write_text('exact csv')
    recording=tmp_path/'flown.csv';recording.write_text('exact csv')
    assert find_rehearsal(tmp_path,recording)==folder


def test_live_plan_cannot_be_rehearsed_during_update(tmp_path):
    import pytest
    from experimental_data.io import atomic_json
    from deployment.pva_rehearsal import generate
    atomic_json(tmp_path/'settings.json',dict(method='mppi'))
    atomic_json(tmp_path/'model.json',{})
    atomic_json(tmp_path/'status.json',dict(status='running'))
    with pytest.raises(ValueError,match='Stop or finish'):generate(tmp_path,tmp_path/'new-rehearsal')
    assert not (tmp_path/'new-rehearsal').exists()


def test_partial_receding_plan_cannot_be_exported_as_a_complete_maneuver(tmp_path):
    import pytest
    from experimental_data.io import atomic_json
    from deployment.pva_rehearsal import generate
    atomic_json(tmp_path/'settings.json',dict(method='mppi'))
    atomic_json(tmp_path/'model.json',{})
    atomic_json(tmp_path/'status.json',dict(status='stopped'))
    np.savez(tmp_path/'plan.npz',normalized_jerk=np.zeros((13,3)),plan_complete=False,committed_steps=13)
    with pytest.raises(ValueError,match='partial'):generate(tmp_path,tmp_path/'new-rehearsal')
    assert not (tmp_path/'new-rehearsal').exists()


def test_fast_pva_exit_recovers_with_saved_envelope_and_unchanged_whip():
    from learning.pva_env import defaults
    from deployment.curved_recovery import DEFAULTS
    from simulator.research_reference import reference_packet_validity
    import torch
    exit=np.array([2.6298885900588966,.07992650759410418,1.9116615800919428,
        4.417229633334709,.6694186065378647,1.4167846386473886,2.672538497822331,2.1551114415506603,1.6275054326099718,0.,0.])
    whip=np.stack([exit,exit]);original=whip.copy();limits=defaults()['limits']
    times,packets,phases,recovery=complete_pva_packets(whip,np.array([0.,0.,1.225]),limits)
    np.testing.assert_array_equal(packets[:2],original)
    assert bool(reference_packet_validity(torch.tensor(packets)[None],limits)[0].all())
    assert packets[:,2].max()<limits['maximum_origin_z_m']
    assert packets[:,2].min()>limits['minimum_origin_z_m']
    np.testing.assert_allclose(packets[-1,:3],[0.,0.,1.225],atol=1e-12)
    np.testing.assert_allclose(packets[-1,3:9],0,atol=1e-12)
    assert DEFAULTS['maximum_horizontal_acceleration_m_s2']==5.  # Historical force shaping remains unchanged.


def test_recovery_search_filters_low_altitude_loops_before_selecting_a_turn():
    from learning.pva_env import defaults
    from deployment.curved_recovery import metrics
    exit=np.array([-.4002847759217911,-.49784872697110394,1.827213840115202,
        .2610491442081023,-1.0592649746834615,.3192095590342279,
        -8.447358993066146,-.3929328241363728,-2.5107795194982936,0.,0.])
    whip=np.stack([exit,exit]);limits=defaults()['limits']
    _,packets,_,details=complete_pva_packets(whip,np.array([-2.,0.,1.255]),limits)
    m=metrics(np.array(details['turn_coefficients_normalized']),details['turn_end_s'])
    assert m['minimum_height_m']>=limits['minimum_origin_z_m']-1e-8
    assert m['peak_height_m']<=limits['maximum_origin_z_m']+1e-8
    np.testing.assert_array_equal(packets[:2],whip)


def test_new_strike_cannot_export_without_fold_even_when_physics_passes(tmp_path,monkeypatch):
    import pytest
    import torch
    from types import SimpleNamespace
    from experimental_data.io import atomic_json
    from deployment import pva_rehearsal
    atomic_json(tmp_path/'settings.json',dict(method='mppi'))
    atomic_json(tmp_path/'model.json',{})
    atomic_json(tmp_path/'status.json',dict(status='completed'))
    np.savez(tmp_path/'plan.npz',normalized_jerk=np.zeros((45,3)),plan_complete=True,committed_steps=45)
    env=SimpleNamespace(steps=45,targeted_strike=True,active=torch.tensor([False]),
        tensor=torch.as_tensor,rollout=lambda **_:dict(failed=torch.tensor([False]),fold_valid=torch.tensor([False])))
    monkeypatch.setattr(pva_rehearsal,'PVAEnvironment',lambda *a,**k:env)
    with pytest.raises(ValueError,match='No verified travelling fold'):
        pva_rehearsal.generate(tmp_path,tmp_path/'export')
    assert not (tmp_path/'export').exists()
