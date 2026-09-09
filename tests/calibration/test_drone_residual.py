from pathlib import Path
import numpy as np
import pytest
import torch

from simulator.drone_tracking import DroneTrackingResidual, predict_trajectory, load_tracking_model
from experimental_data.drone_residual_fit import held_commands, nominal_prediction, batch, load_trial, SETTINGS
from experimental_data.io import atomic_json, sha256_file


def synthetic_trial():
    t=np.arange(30)*.01
    ct=np.arange(-20,40)*.01
    commands=np.zeros((len(ct),9));commands[:,:3]=[0.,0.,1.5]
    commands[(ct>=.05)&(ct<.15),6]=2.
    commands[ct>=.15,:3]=[.01,0.,1.5]  # Actual logged hold after a short maneuver.
    return dict(absolute_time=t,native_time=ct,native_commands=commands,native_valid=np.ones(len(ct),bool),
                measured_position=np.tile([0.,0.,1.5],(len(t),1)),initial_velocity=np.zeros(3))


def test_commands_are_causal_and_keep_logged_hold():
    trial=synthetic_trial();ct=trial['native_time'];commands=trial['native_commands']
    query=np.array([.049,.055,.151])
    held=held_commands(query,ct,commands,trial['native_valid'],query)
    assert held[0,6]==0 and held[1,6]==2 and held[2,6]==0
    np.testing.assert_array_equal(held[2,:3],[.01,0.,1.5])
    with pytest.raises(ValueError,match='does not cover'):
        held_commands(query,ct,commands,trial['native_valid'],np.array([-1.]))


def test_nominal_fit_and_recursive_torch_model_match():
    trial=synthetic_trial();gains=np.array([4.,5.,6.,3.,4.,5.,.8,1.,1.2])
    args,truth=batch([trial],.02,SETTINGS,'cpu')
    network=DroneTrackingResidual().double()
    assert torch.count_nonzero(network(args[0],args[1],args[2][:,0],args[3][:,0]))==0
    actual=predict_trajectory(*args,torch.tensor(gains),network)[0][0].detach().numpy()
    np.testing.assert_allclose(actual,nominal_prediction(trial,gains,.02,.05),rtol=1e-12,atol=1e-12)
    changed=dict(trial,measured_position=trial['measured_position'].copy())
    changed['measured_position'][1:]+=100
    changed_args,_=batch([changed],.02,SETTINGS,'cpu')
    other=predict_trajectory(*changed_args,torch.tensor(gains),network)[0][0].detach().numpy()
    np.testing.assert_array_equal(actual,other)


def test_residual_is_bounded_and_checkpoint_reproduces_prediction(tmp_path):
    torch.manual_seed(1)
    network=DroneTrackingResidual(hidden=8,acceleration_limit=2.).double()
    with torch.no_grad():network.net[-1].weight.normal_();network.net[-1].bias.fill_(.2)
    trial=synthetic_trial();args,_=batch([trial],.02,SETTINGS,'cpu')
    acceleration=network(args[0],args[1],args[2][:,0],args[3][:,0])
    assert acceleration.abs().max()<=2
    gains=torch.tensor([4.]*3+[3.]*3+[1.]*3,dtype=torch.float64)
    payload=dict(schema='effective_fullstate_drone_residual_v1',specification=network.specification(),
        state_dict=network.state_dict(),nominal=dict(gains=gains.tolist(),delay_s=.02),history_s=.05)
    path=tmp_path/'drone.pt';torch.save(payload,path)
    loaded,parameters,delay,history=load_tracking_model(path)
    assert delay==.02 and history==.05
    expected=predict_trajectory(*args,gains,network)
    actual=predict_trajectory(*args,parameters,loaded)
    for first,second in zip(actual,expected):torch.testing.assert_close(first,second)
    expected[0][:,-1].sum().backward()
    assert torch.isfinite(network.net[-1].weight.grad).all()


def test_load_trial_uses_logged_commands_and_causal_initial_velocity(tmp_path):
    t=np.arange(-100,151)*.01
    commands=np.zeros((len(t),11));commands[:,:3]=[0.,0.,1.5]
    commands[(t>=0)&(t<.65),6]=1.
    commands[t>=.65,:3]=[.1,0.,1.6]
    position=np.tile([0.,0.,1.5],(len(t),1));position[:,0]=.1*t
    arrays=dict(controller_time_s=t,drone_position_m=position,drone_position_valid=np.ones(len(t),bool),
        controller_native_time_s=t,controller_native_fullstate=commands,
        controller_native_valid=np.ones(len(t),bool),reference_valid=np.ones(len(t),bool))
    np.savez(tmp_path/'dataset.npz',**arrays)
    def quality():atomic_json(tmp_path/'quality.json',dict(output_files={'dataset.npz':sha256_file(tmp_path/'dataset.npz')},alignment={}))
    quality();before=load_trial(tmp_path,'whip1_001','training',SETTINGS)
    arrays['drone_position_m'][t>-.2,0]+=10
    np.savez(tmp_path/'dataset.npz',**arrays);quality()
    after=load_trial(tmp_path,'whip1_001','training',SETTINGS)
    np.testing.assert_allclose(before['initial_velocity'],[.1,0,0],atol=1e-10)
    np.testing.assert_array_equal(before['initial_velocity'],after['initial_velocity'])
    assert before['native_commands'][t>=.65,6].max()==0
