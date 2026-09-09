import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from experimental_data.io import sha256_file
from experimental_data.drone_pose_response_data import load_nominal_pose_trial
from simulator.geometry import attachment_positions
from simulator.drone_pose_response import PoseResponseParameters,predict_pose


def parameters():return PoseResponseParameters(4.,4.,3.,3.,1.,1.,.08,.02)


def fixture(folder,*,change_future=False,invalid_hover=False):
    folder.mkdir(parents=True,exist_ok=True)
    t=np.arange(201)*.01
    p=np.tile([0.,0.,1.5],(len(t),1));p[:,0]=.01*t
    q=np.tile([0.,0.,0.,1.],(len(t),1))
    if change_future:
        p[t>=.9]+=2
        q[t>=.9]=Rotation.from_euler('x',.7).as_quat()
    r=[.007,-.014,-.055]
    attachment,av=attachment_positions(p,q,r)
    commands=np.zeros((len(t),11));commands[:,:3]=[0,0,1.5]
    commands[(t>=1)&(t<1.1),6]=2
    phases=np.where(t<1,'pre_maneuver_fullstate_hold',np.where(t<1.1,'csv_maneuver','post_maneuver_fullstate_hold'))
    valid=np.ones(len(t),bool)
    if invalid_hover:valid[40]=False
    np.savez_compressed(folder/'dataset.npz',controller_time_s=t,drone_position_m=p,drone_quaternion_xyzw=q,
        drone_position_valid=np.ones(len(t),bool),attachment_position_m=attachment,attachment_valid=av,
        execution_phase=phases,reference_fullstate=commands,reference_valid=valid,
        controller_native_time_s=t,controller_native_fullstate=commands,controller_native_command_age_s=np.zeros(len(t)),
        controller_native_valid=np.ones(len(t),bool))
    (folder/'execution_phases.json').write_text(json.dumps(dict(csv_start_s=1.,csv_end_s=1.1)))
    (folder/'quality.json').write_text(json.dumps(dict(alignment={},output_files={n:sha256_file(folder/n) for n in ['dataset.npz','execution_phases.json']})))
    profile=folder.parent/'execution_profile.json';profile.write_text(json.dumps(dict(geometry=dict(offset_tracking_m=r))))
    (folder.parent/'processing.json').write_text(json.dumps(dict(execution_profile_sha256=sha256_file(profile))))


def test_real_data_adapter_cannot_feed_future_measurements_to_rollout(tmp_path):
    folder=tmp_path/'processed/whip';fixture(folder)
    args,truth,context=load_nominal_pose_trial(folder,parameters(),device='cpu',alignment_mode='prehover_effective_alignment',hover_history_s=.7)
    original=predict_pose(**args)
    assert args['output_time_s'][0]==args['initial'].time_s<.9
    assert not context['measured_targets_are_model_inputs']
    fixture(folder,change_future=True)
    changed,later,_=load_nominal_pose_trial(folder,parameters(),device='cpu',alignment_mode='prehover_effective_alignment',hover_history_s=.7)
    altered=predict_pose(**changed)
    assert not np.array_equal(truth['position_origin_m'],later['position_origin_m'])
    for key in original:torch.testing.assert_close(original[key],altered[key],atol=0,rtol=0)


def test_unobserved_hover_and_tampered_data_are_rejected(tmp_path):
    folder=tmp_path/'processed/whip';fixture(folder,invalid_hover=True)
    with pytest.raises(ValueError,match='pre-hover'):
        load_nominal_pose_trial(folder,parameters(),device='cpu',alignment_mode='prehover_effective_alignment',hover_history_s=.7)
    with (folder/'dataset.npz').open('ab') as stream:stream.write(b'tamper')
    with pytest.raises(ValueError,match='checksum'):
        load_nominal_pose_trial(folder,parameters(),device='cpu',alignment_mode='prehover_effective_alignment')
