from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from experimental_data.io import atomic_json
from experimental_data.model_evaluation import comparison_identity
from experimental_data.whip_response_fit import pose_loss,scalar_search
from experimental_data.response_update_contract import validate


def contract():
    return dict(schema='whip_response_gain_v1',parameter='feedforward_xy',bounds_scale=[.5,1.25],
        position_scale_m=.02,orientation_scale_rad=.05,clock_check_offsets_s=[-.02,.02],regularization=.1,
        reason='Adaptation-only evidence',reviewed_by='test',diagnostic_hashes={'report':'hash'})


def test_pose_loss_uses_rotations_masks_and_equal_frame_weight():
    c=contract();p=np.zeros((3,3));r=np.tile(np.eye(3),(3,1,1));prediction=dict(position_origin_m=p.copy(),rotation_tracking_to_world=r.copy())
    prediction['position_origin_m'][:2,0]=.02;prediction['position_origin_m'][2]=np.nan
    assert pose_loss(prediction,p,r,np.array([1,1,0],bool),c)==pytest.approx(.5*(np.sqrt(2)-1))
    with pytest.raises(ValueError,match='No scored'):pose_loss(prediction,p,r,np.zeros(3,bool),c)


def test_fixed_identity_rejects_drone_changes_under_old_cable_scope(tmp_path):
    def model(folder,gain=.7,damping=.4,kp=2.):
        folder.mkdir();atomic_json(folder/'drone.json',dict(nominal=dict(parameters=dict(feedforward_xy=gain,kp_xy=kp),training_takes=['a'])))
        atomic_json(folder/'model.json',dict(cable=dict(external_drag_s_inv=damping),fullstate_execution=dict(checkpoint='drone.json')))
        return folder/'model.json'
    base=model(tmp_path/'base');gain=model(tmp_path/'gain',gain=.6);cable=model(tmp_path/'cable',damping=.8);feedback=model(tmp_path/'feedback',kp=3.)
    assert comparison_identity(base)==comparison_identity(cable)
    assert comparison_identity(base)!=comparison_identity(gain)
    c=contract()
    assert comparison_identity(base,c)==comparison_identity(gain,c)
    assert comparison_identity(base,c)!=comparison_identity(cable,c)
    assert comparison_identity(base,c)!=comparison_identity(feedback,c)


def test_contract_cannot_enable_unreviewed_parameter_or_loss_changes():
    c=contract();assert validate(c)==c
    for key,value in [('parameter','delay_s'),('bounds_scale',[0.,10.]),('position_scale_m',1.)]:
        changed=deepcopy(c);changed[key]=value
        with pytest.raises(ValueError,match='single feedforward'):validate(changed)


def test_scalar_search_retains_baseline_and_finds_known_optimum(tmp_path):
    best,loss,history,reason=scalar_search(lambda x:(x-.63)**2,.75,[.375,.9375],tmp_path,tmp_path/'STOP')
    assert abs(best-.63)<.003 and loss<1e-5 and reason=='practical_plateau'
    assert all(.75 in row['gains'] for row in history)
    assert (tmp_path/'search_state.json').exists()
    assert all(b['best_loss']<=a['best_loss'] for a,b in zip(history,history[1:]))
