import numpy as np
import pytest
from experimental_data.bootstrap_combined import measured_history, marker_scores
from experimental_data.bootstrap_bundle import read
from simulator.cable import CableConfiguration
from pathlib import Path


def test_cable_initialization_ignores_future_marker_and_pose_values():
    root=Path(__file__).resolve().parents[2]
    cable=CableConfiguration.from_mapping(read(root/'config/model.json')['cable'])
    n=40;offset=np.array([.006,-.012,-.055]);time=np.arange(n)*.01
    p=np.column_stack((.03*time,time*0,time*0+1.5))
    markers=np.broadcast_to(p[:,None]+offset,(n,10,3)).copy()
    markers[:,:,2]-=np.cumsum(cable.marker_interval_lengths_m)
    data=dict(controller_time_s=time,drone_position_m=p,drone_position_valid=np.ones(n,bool),
        drone_quaternion_xyzw=np.tile([0.,0.,0.,1.],(n,1)),cable_position_m=markers,cable_valid=np.ones((n,10),bool))
    first,index=measured_history(data,.25,cable,offset)
    assert index==25 and len(first)==21
    data['cable_position_m'][26:]=np.nan;data['drone_position_m'][26:]=1e6
    data['drone_quaternion_xyzw'][26:]=np.nan
    second,_=measured_history(data,.25,cable,offset)
    np.testing.assert_array_equal(first,second)
    data['cable_valid'][24,2]=False
    with pytest.raises(ValueError,match='Invalid'):measured_history(data,.25,cable,offset)


def test_marker_metrics_use_xyz_distance_and_individual_validity():
    pred=np.zeros((3,10,3));truth=pred.copy();truth[:,:,0]=.03;truth[:,:,1]=.04
    valid=np.ones((3,10),bool);truth[0,0]=np.nan;valid[0,0]=False
    score=marker_scores(pred,truth,valid)
    assert score['marker_samples']==29 and score['tip_samples']==3
    assert score['marker_rmse_m']==pytest.approx(.05)
    assert score['tip_rmse_m']==pytest.approx(.05)
    pred[0,0]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):marker_scores(pred,truth,valid)


def test_component_bundle_resolves_after_copy_and_rejects_tampering(tmp_path):
    import shutil
    import torch
    from experimental_data.bootstrap_combined import save_bundle, load_bundle
    from experimental_data.io import atomic_json, sha256_file
    from experimental_data.differentiable_fit import save_weights
    from simulator.cable.residual import MotionResidual, FrozenMotionResidual
    from simulator.drone_pose_residual import DronePoseResidual, save_residual, load_residual
    root=Path(__file__).resolve().parents[2];payload=read(root/'config/model.json')
    payload['cable']['external_drag_s_inv']=0.
    c=tmp_path/'cable/final';d=tmp_path/'drone/final_all_three';c.mkdir(parents=True);d.mkdir(parents=True)
    save_weights(c/'nn.pt',MotionResidual(12,mode='dissipative').double())
    save_residual(d/'nn.pt',DronePoseResidual().double())
    payload['motion_residual']=dict(checkpoint=str(c/'nn.pt'),sha256=sha256_file(c/'nn.pt'))
    atomic_json(c/'candidate_model.json',payload)
    atomic_json(d/'candidate.json',dict(nominal={},residual=dict(checkpoint=str(d/'nn.pt'),sha256=sha256_file(d/'nn.pt'))))
    save_bundle(tmp_path);shutil.copytree(tmp_path/'bundle',tmp_path/'copied')
    cable,drone=load_bundle(tmp_path/'copied')
    assert Path(cable['motion_residual']['checkpoint']).parent==tmp_path/'copied'
    q=torch.zeros(1,12,3,dtype=torch.float64)
    assert not torch.count_nonzero(FrozenMotionResidual(cable['motion_residual']['checkpoint'],cable['motion_residual']['sha256'])(q,q))
    load_residual(drone['residual']['checkpoint'],drone['residual']['sha256'],'cpu')
    assert cable['schema']=='bootstrap_cable_component_v1' and 'point_mass' not in cable
    (tmp_path/'copied/cable_residual.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='hash'):load_bundle(tmp_path/'copied')
