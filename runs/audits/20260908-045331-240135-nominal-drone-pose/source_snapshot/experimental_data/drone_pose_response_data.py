"""Connect phase-aware logs to the new nominal pose engine, without a fit.

Measured prediction targets are returned separately from model inputs. Only
the already observed pre-hover is used to construct the initial state.
"""
import json
from pathlib import Path

import numpy as np
import torch

from .drone_pose_initialization import initialize_drone_pose
from .io import sha256_file
from simulator.geometry import normalized_rotations_xyzw
from simulator.drone_pose_response import initialize_from_hover,CommandSchedule


def load_nominal_pose_trial(folder,parameters,*,device,alignment_mode,
                            rotation_command_from_tracking=None,pre_command_margin_s=.1,
                            hover_history_s=1.,post_hold_s=.5):
    folder=Path(folder)
    report=json.loads((folder/'quality.json').read_text())
    profile_path=folder.parent/'execution_profile.json'
    processing=json.loads((folder.parent/'processing.json').read_text())
    if sha256_file(profile_path)!=processing['execution_profile_sha256']:
        raise ValueError('Execution profile checksum mismatch')
    profile=json.loads(profile_path.read_text())
    for filename in ('dataset.npz','execution_phases.json'):
        if sha256_file(folder/filename)!=report['output_files'][filename]:
            raise ValueError('Phase-aware recording checksum mismatch')
    phases=json.loads((folder/'execution_phases.json').read_text())
    offset=profile['geometry']['offset_tracking_m']
    if (not np.isfinite([pre_command_margin_s,hover_history_s,post_hold_s]).all()
            or pre_command_margin_s<0 or hover_history_s<.2 or post_hold_s<0):
        raise ValueError('Invalid initialization/prediction intervals')
    with np.load(folder/'dataset.npz',allow_pickle=False) as data:
        t=data['controller_time_s'];p=data['drone_position_m'];q=data['drone_quaternion_xyzw']
        onset=phases['csv_start_s']
        seed=initialize_drone_pose(t,p,q,cutoff_s=onset-pre_command_margin_s,offset_tracking_m=offset)
        index=int(seed['sample_indices'][-1])
        history=np.flatnonzero((t>=seed['time_s']-hover_history_s)&(t<=seed['time_s']))
        if (not np.all(data['execution_phase'][history]=='pre_maneuver_fullstate_hold')
                or not data['reference_valid'][history].all()):
            raise ValueError('Need fresh, phase-verified pre-hover command history')
        rotations,valid=normalized_rotations_xyzw(q[history])
        if not valid.all():raise ValueError('Invalid pre-hover orientation')
        tensor=lambda a:torch.as_tensor(np.array(a,copy=True),dtype=torch.float64,device=device)
        state,initialization=initialize_from_hover(t[history],tensor(p[history])[None],tensor(rotations)[None],
            tensor(data['reference_fullstate'][history])[None],parameters,alignment_mode=alignment_mode,
            rotation_command_from_tracking=rotation_command_from_tracking)
        # Independent agreement with the already tested causal pose initializer.
        np.testing.assert_allclose(state.velocity[0].detach().cpu(),seed['velocity_origin_m_s'],atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(state.omega_tracking[0].detach().cpu(),seed['omega_tracking_rad_s'],atol=1e-10,rtol=1e-10)
        ct=data['controller_native_time_s'];commands=data['controller_native_fullstate'];age=data['controller_native_command_age_s']
        fresh=data['controller_native_valid']&(age<.1)
        # Last logged row has no demonstrated interval after it. Do not invent one.
        schedule=CommandSchedule(ct[:-1],tensor(commands[:-1])[None],coverage_end_s=ct[-1],
            valid=fresh[:-1],valid_until_s=ct[:-1]+.1-age[:-1])
        ids=np.flatnonzero((np.arange(len(t))>=index)&(t<=phases['csv_end_s']+post_hold_s))
        if len(ids)<2 or t[ids[-1]]<phases['csv_end_s']:
            raise ValueError('Insufficient tracking coverage for complete observed maneuver')
        future_rotations,future_valid=normalized_rotations_xyzw(q[ids])
        # This dictionary is diagnostic truth; it is NEVER supplied to predict_pose.
        measured=dict(position_origin_m=p[ids].copy(),rotation_tracking_to_world=future_rotations,
            position_attachment_m=data['attachment_position_m'][ids].copy(),
            position_valid=data['drone_position_valid'][ids].copy(),orientation_valid=future_valid,
            attachment_valid=data['attachment_valid'][ids].copy(),execution_phase=data['execution_phase'][ids].copy())
        arguments=dict(initial=state,schedule=schedule,output_time_s=t[ids].copy(),parameters=parameters,
                       offset_tracking_m=offset)
        context=dict(trial_id=folder.name,source=str(folder),source_dataset_sha256=sha256_file(folder/'dataset.npz'),
            csv_onset_s=onset,csv_end_s=phases['csv_end_s'],initialization=initialization,
            pre_command_margin_s=pre_command_margin_s,clock=report['alignment'],
            measured_targets_are_model_inputs=False,model_fitted=False)
    return arguments,measured,context
