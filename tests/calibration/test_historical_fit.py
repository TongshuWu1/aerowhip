import numpy as np
import torch
from experimental_data.historical_dataset import quality_masks, FOLDS, TAKES
from experimental_data.historical_fit import choose_starts, drone_arrays, numpy_drone
from simulator.drone_tracking import predict_trajectory
from simulator.fullstate_execution import sample_kinematic_reference, attachment_to_vehicle_commands
from deployment.fullstate import sample_fullstate


def sample():
    n=120
    p=np.tile([0.,0.,1.5],(n,1))
    markers=np.tile(np.column_stack((np.zeros(10),np.zeros(10),1.5-np.arange(1,11)*.1)),(n,1,1))
    a=dict(time=np.arange(n)*.01,position=p,markers=markers,quaternion=np.tile([0.,0.,0.,1.],(n,1)),
        position_valid=np.ones(n,bool),marker_valid=np.ones((n,10),bool),
        commands=np.tile([0.,0.,1.5,0.,0.,0.,0.,0.,0.],(n,1)),
        command_valid=np.ones(n,bool),command_age=np.zeros(n))
    payload=dict(cable=dict(marker_interval_lengths_m=[.1]*10),recorded_data=dict(optitrack_to_attachment_offset_body_m=[0.,0.,0.]))
    return a,payload


def test_command_gaps_do_not_remove_cable_data():
    a,p=sample();a['command_valid'][40]=False
    cable,drone,_=quality_masks(a,p)
    assert cable.all() and not drone[40]
    assert drone.sum()==len(drone)-1


def test_contact_quarantine_and_missing_marker():
    a,p=sample();a['marker_valid'][25,2]=False
    a['markers'][80,-1,2]=.01
    cable,drone,why=quality_masks(a,p)
    assert not cable[25] and drone[25]
    assert not cable[60:101].any() and not drone[60:101].any()
    assert why['possible_contact_near_source_z_zero'][80]


def test_windows_never_bridge_invalid_history_or_future():
    a,_=sample();valid=np.ones(120,bool);valid[60]=False
    starts=choose_starts(a,valid,.2,6)
    assert len(starts)
    assert all(valid[s-20:s+21].all() for s in starts)


def test_whole_take_folds_cover_every_take_once():
    flattened=[n for f in FOLDS for n in f]
    assert len(flattened)==len(set(flattened))
    assert set(flattened)==set(TAKES)


def test_drone_delay_and_recursive_numpy_torch_agree():
    rng=np.random.default_rng(3)
    rows=[dict(position=rng.normal(size=(101,3)),velocity=rng.normal(size=3),commands=rng.normal(size=(120,9))) for _ in range(2)]
    args,_=drone_arrays(rows,.06)
    assert np.array_equal(args[2][0,0],rows[0]['commands'][14])
    assert np.array_equal(args[3][0,0],rows[0]['commands'][9])
    gains=np.array([4.]*3+[3.]*3+[1.]*3)
    expected=numpy_drone(args,gains)
    actual=predict_trajectory(*(torch.tensor(x) for x in args),torch.tensor(gains))[0].numpy()
    np.testing.assert_allclose(actual,expected,atol=1e-12)


def test_batched_reference_matches_export_pva():
    t=np.arange(81)*.01
    p=np.column_stack((t**3,.2*t**2,1.5+.1*t))
    v=np.column_stack((3*t*t,.4*t,np.full_like(t,.1)))
    q,x,y,z=sample_fullstate(t,p,v)
    result=sample_kinematic_reference(torch.tensor(p[None]),torch.tensor(v[None]),.01,torch.tensor(q[None]))
    np.testing.assert_allclose(result[0].numpy(),np.concatenate((x,y,z),axis=1),atol=2e-10)


def test_reference_offset_changes_position_only():
    cmds=torch.randn(2,30,9,dtype=torch.float64)
    offset=torch.tensor([[.01,0.,-.055],[0.,.02,-.055]],dtype=torch.float64)
    converted=attachment_to_vehicle_commands(cmds,offset)
    torch.testing.assert_close(converted[:,:,:3]+offset[:,None],cmds[:,:,:3])
    torch.testing.assert_close(converted[:,:,3:],cmds[:,:,3:])


def test_reference_uses_same_side_of_acceleration_knots_as_csv():
    rng=np.random.default_rng(13);t=np.arange(81)*.01
    p=rng.normal(size=(81,3));v=rng.normal(size=(81,3))
    q,x,y,z=sample_fullstate(t,p,v)
    result=sample_kinematic_reference(torch.tensor(p[None]),torch.tensor(v[None]),.01,torch.tensor(q[None]))
    np.testing.assert_allclose(result[0].numpy(),np.concatenate((x,y,z),axis=1),atol=1e-7)
