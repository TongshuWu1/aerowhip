import numpy as np
import pytest
from experimental_data.drone_response_diagnostic import local_kinematics, sensitivity_summary, check_command_coverage


def test_native_cubic_derivatives_have_physical_units():
    # Includes a large clock offset and deliberately nonuniform native sampling.
    relative=np.arange(101)*.01+np.sin(np.arange(101))*.0002
    t=10000+relative
    p=np.c_[3+2*relative+4*relative**2+relative**3,relative**2,-2*relative]
    v,a=local_kinematics(t,p,np.ones(len(t),bool))
    good=np.isfinite(a).all(-1)
    expected_v=np.c_[2+8*relative+3*relative**2,2*relative,np.full(len(t),-2)]
    expected_a=np.c_[8+6*relative,np.full(len(t),2),np.zeros(len(t))]
    assert good.sum()>80
    np.testing.assert_allclose(v[good],expected_v[good],atol=1e-8)
    np.testing.assert_allclose(a[good],expected_a[good],atol=1e-7)
    assert np.isnan(a[:5]).all() and np.isnan(a[-5:]).all()


def test_missing_samples_and_clock_gaps_have_unfilled_derivative_neighborhoods():
    t=np.arange(101)*.01;p=np.c_[t*t,t*0,t*0];valid=np.ones(len(t),bool)
    valid[50]=False;p[50]=1e9
    _,a=local_kinematics(t,p,valid)
    assert np.isnan(a[45:56]).all()
    assert a[30,0]==pytest.approx(2)
    ids=np.r_[np.arange(45),np.arange(56,101)]
    _,gap_a=local_kinematics(t[ids],p[ids],valid[ids])
    assert np.isnan(gap_a[(t[ids]>.39)&(t[ids]<.61)]).all()


def test_no_excitation_and_confounded_sensitivities_are_reported():
    assert sensitivity_summary(np.zeros((30,3)))['numerical_rank']==0
    t=np.arange(30,dtype=float)
    report=sensitivity_summary(np.c_[t,2*t,np.zeros(30)])
    assert report['numerical_rank']==1
    assert report['cosine_similarity'][0][1]==pytest.approx(1)


def test_shifted_command_coverage_does_not_bridge_small_invalid_gap():
    import torch
    from types import SimpleNamespace
    from simulator.drone_pose_response import CommandSchedule
    packet_time=np.array([-.1,.05,.1]);until=np.array([.049,.1,.2])
    schedule=CommandSchedule(packet_time,torch.zeros(1,3,11,dtype=torch.float64),
        coverage_end_s=.2,valid_until_s=until)
    trial=SimpleNamespace(data={'packet_time':packet_time,'packet_valid_until':until},schedule=schedule)
    with pytest.raises(ValueError,match='No fresh observed command'):
        check_command_coverage(trial,np.array([0.,.1]),.02)


def test_diagnosis_never_opens_validation_take(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from experimental_data import drone_response_diagnostic as diagnostic
    p=SimpleNamespace(delay_s=.02,feedforward_xy=1.,attitude_time_constant_s=.08)
    monkeypatch.setattr(diagnostic.fitting,'load',lambda *args:({},
        {'takes':{'train':{'role':'adaptation'},'validation':{'role':'validation'}}},SimpleNamespace(drone=SimpleNamespace(parameters=p))))
    opened=[]
    def trial(job,name,*args):
        opened.append(name)
        assert name!='validation'
        raise RuntimeError('stop before replay')
    monkeypatch.setattr(diagnostic,'WhipTrial',trial)
    with pytest.raises(RuntimeError,match='stop before replay'):
        diagnostic.diagnose_drone(tmp_path,tmp_path/'diagnostic','cpu')
    assert opened==['train']
