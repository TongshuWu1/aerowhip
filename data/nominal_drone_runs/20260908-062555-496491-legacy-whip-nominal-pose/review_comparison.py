"""Reproducible v1/v3 comparison and native hold-transition verification."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.spatial.transform import Rotation
import torch
JOB=Path(__file__).resolve().parent;ROOT=JOB.parents[2];sys.path.insert(0,str(ROOT))
from experimental_data.nominal_pose_fit import PreparedTrial,GAIN_NAMES
from experimental_data.legacy_execution import read_sequence
from experimental_data.io import sha256_file
from simulator.drone_pose_response import command_attitude,attitude_drive_acceleration,PoseResponseParameters

protocol=json.loads((JOB/'protocol.json').read_text());old=Path(protocol['frozen_translation']['source'])
newresults=json.loads((JOB/'results.json').read_text());oldresults=json.loads((old/'results.json').read_text())
controller=Path('C:/Users/wts28/Downloads/full_state_pva.py')
times,expected,constants=read_sequence(controller)
transitions={};comparison={}
for name in protocol['takes']:
    folder=JOB/'inputs'/protocol['source_version']/name
    with np.load(folder/'dataset.npz') as d:
        phase=d['controller_native_execution_phase'];cmd=d['controller_native_fullstate'];t=d['controller_native_time_s'];idx=d['controller_native_csv_sample_index']
        csv=np.flatnonzero(phase=='csv_maneuver');post=np.flatnonzero(phase=='post_maneuver_fullstate_hold');pre=np.flatnonzero(phase=='pre_maneuver_fullstate_hold')
        ids=idx[csv].astype(int)
        # Native commands are compared directly with the supplied script, not
        # with a new policy rollout or an unflown complete/recovery export.
        max_difference=float(abs(cmd[csv]-expected[ids]).max())
        assert max_difference<1e-9
        assert (cmd[post,3:9]==0).all() and (cmd[pre,3:9]==0).all()
        assert np.unique(ids).tolist()==list(range(20))
        transitions[name]=dict(csv_rows=20,script_command_max_abs_difference=max_difference,
            prehold_first_observed_s=float(t[pre[0]]),csv_first_observed_s=float(t[csv[0]]),posthold_first_observed_s=float(t[post[0]]),
            last_csv_pva=cmd[csv[-1]].tolist(),posthold_pva=cmd[post[0]].tolist(),
            posthold_target_constant=bool(np.allclose(cmd[post,:3],cmd[post[0],:3],rtol=0,atol=1e-10)),
            pre_and_post_hold_zero_velocity_acceleration=True,posthold_duration_observed_s=float(t[post[-1]]-t[post[0]]))
for label,item in newresults.items():
    params=PoseResponseParameters(**item['parameters']);g=np.array([item['parameters'][k] for k in GAIN_NAMES]);delay=params.delay_s
    assert all(item['parameters'][k]==oldresults[label]['parameters'][k] for k in [*GAIN_NAMES,'delay_s'])
    comparison[label]={}
    for name in protocol['takes']:
        trial=PreparedTrial(JOB/'inputs'/protocol['source_version']/name,review=protocol['reviews'][name])
        p,v,_=trial.linear(g,delay);cmd=np.array([trial.schedule.sample(t-delay)[0].numpy() for t in trial.time])
        kp=g[:2][[0,0,1]];kd=g[2:4][[0,0,1]];ff=g[4:][[0,0,1]]
        a=kp*(cmd[:,:3]-p)+kd*(cmd[:,3:6]-v)+ff*cmd[:,6:9]+trial.b0+trial.basis@g
        u=attitude_drive_acceleration(torch.as_tensor(a),params)
        rd=command_attitude(u,torch.as_tensor(cmd[:,9]),9.80665).numpy()
        target_steps=np.r_[0,np.degrees(Rotation.from_matrix(rd[:-1].transpose(0,2,1)@rd[1:]).magnitude())]
        specific=u.numpy()+[0,0,9.80665]
        tilt=np.degrees(np.arccos(np.clip(specific[:,2]/np.linalg.norm(specific,axis=1),-1,1)))
        with np.load(old/label/(name+'.npz')) as before,np.load(JOB/label/(name+'.npz')) as after:
            valid=np.isfinite(before['position_origin_m']).all(axis=1)
            delta=float(abs(before['position_origin_m'][valid]-after['position_origin_m'][valid]).max())
            # The rejected v1 fold saved CPU translation-only diagnostics;
            # the new valid fold saves GPU pose output. Allow only round-off.
            assert delta<1e-10
        comparison[label][name]=dict(old_scores=oldresults[label]['scores'][name],new_scores=item['scores'][name],
            origin_prediction_max_abs_change_m=delta,new_status=item['prediction_status'][name],
            sampled_target_tilt_max_deg=float(tilt.max()),sampled_target_frame_step_max_deg=float(target_steps.max()),
            sampled_specific_vertical_min_m_s2=float(specific[:,2].min()))
        np.savez_compressed(JOB/label/(name+'_attitude_drive.npz'),time_s=trial.time,
            modeled_origin_acceleration_m_s2=a,attitude_drive_acceleration_m_s2=u.numpy(),target_frame=rd,
            execution_phase=trial.truth['execution_phase'])
report=dict(source_sha256=sha256_file(__file__),controller_sha256=sha256_file(controller),
    controller_kind='Supplied colleague source reviewed as data; remote installed version not independently verified',
    transitions=transitions,comparison=comparison,
    limitation='Frame/acceleration extrema are at saved output samples; not substep or vehicle-limit certification')
with (JOB/'comparison.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
print(json.dumps(report,indent=2))
