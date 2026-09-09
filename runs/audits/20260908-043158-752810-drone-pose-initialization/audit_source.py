"""Audit legacy pose/hover initialization without fitting dynamics or changing data."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experimental_data.adaptation_rounds import write_json
from experimental_data.drone_pose_initialization import initialize_drone_pose
from experimental_data.io import sha256_file
from simulator.geometry import normalized_rotations_xyzw
from simulator.workflow import stamp


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed',type=Path,required=True)
    parser.add_argument('--pre-command-margin-s',type=float,default=.1)
    args=parser.parse_args()
    if not np.isfinite(args.pre_command_margin_s) or args.pre_command_margin_s<0:
        parser.error('Pre-command margin must be finite and nonnegative')
    source=args.processed.resolve()
    processing=json.loads((source/'processing.json').read_text())
    profile=json.loads((source/'execution_profile.json').read_text())
    if sha256_file(source/'execution_profile.json')!=processing['execution_profile_sha256']:
        raise ValueError('Execution profile checksum mismatch')
    offset=np.array(profile['geometry']['offset_tracking_m'])
    output=ROOT/'runs/audits'/(stamp()+'-drone-pose-initialization');output.mkdir(parents=True)
    originals={}
    for name in ['config/model.json','config/task.json','config/baseline.json','config/controller_export.json',
                 'runs/ppo/20260906-201957-294110-measured-mass-seed653/checkpoints/best_validation.pt']:
        path=ROOT/name
        if path.is_file():originals[str(path)]=sha256_file(path)
    records=[];plot_data=[]
    for report in processing['reports']:
        if report['status']=='failed':raise ValueError('Prepared trial requires review')
        name=report['trial_id'];folder=source/name
        for filename,checksum in report['output_files'].items():
            path=folder/filename
            if sha256_file(path)!=checksum:raise ValueError('Processed source checksum mismatch')
            originals[str(path)]=checksum
        with np.load(folder/'dataset.npz',allow_pickle=False) as d:
            t=d['controller_time_s'];p=d['drone_position_m'];q=d['drone_quaternion_xyzw']
            onset=report['execution_phases']['csv_start_s']
            state=initialize_drone_pose(t,p,q,cutoff_s=onset-args.pre_command_margin_s,offset_tracking_m=offset)
            i=int(state['sample_indices'][-1])
            if not np.all(d['execution_phase'][state['sample_indices']]=='pre_maneuver_fullstate_hold'):
                raise ValueError('Initializer is not wholly inside recorded pre-hover')
            hover=(t>=state['time_s']-1)&(t<=state['time_s'])
            maneuver=d['csv_maneuver_mask']
            rotations,valid=normalized_rotations_xyzw(q)
            if not valid[hover|maneuver].all():raise ValueError('Invalid quaternion in review interval')
            difference=np.einsum('ij,tjk->tik',state['rotation_tracking_to_world'].T,rotations[maneuver])
            angle=np.rad2deg(Rotation.from_matrix(difference).magnitude())
            offsets=np.einsum('tij,j->ti',rotations,offset)
            offset_error=np.linalg.norm(offsets-offsets[i],axis=1)
            reference=d['reference_fullstate'][i]
            if not d['reference_valid'][i]:raise ValueError('Pre-hover command is unknown')
            row=dict(trial_id=name,initial_time_s=state['time_s'],csv_onset_s=onset,
                pre_command_margin_observed_s=onset-state['time_s'],
                hover_last_second_position_span_m=np.ptp(p[hover],axis=0).tolist(),
                initial_position_error_actual_minus_command_m=(p[i]-reference[:3]).tolist(),
                initial_speed_m_s=float(np.linalg.norm(state['velocity_origin_m_s'])),
                initial_angular_speed_rad_s=float(np.linalg.norm(state['omega_tracking_rad_s'])),
                max_maneuver_orientation_change_deg=float(angle.max()),
                max_maneuver_rotating_offset_change_m=float(offset_error[maneuver].max()),
                initializer={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in state.items()},
                commands_and_measurements_preserved=True)
            records.append(row)
            plot_data.append((name,t[maneuver]-onset,angle,offset_error[maneuver]*100))
            np.savez_compressed(output/(name+'.npz'),**{k:v for k,v in state.items() if isinstance(v,np.ndarray)},
                state_time_s=state['time_s'],csv_onset_s=onset,
                command_history_time_s=t[hover],command_history_fullstate=d['reference_fullstate'][hover],
                command_history_valid=d['reference_valid'][hover],
                command_history_age_s=d['reference_age_s'][hover])
    for path,checksum in originals.items():
        if sha256_file(path)!=checksum:raise ValueError('Protected source changed during audit')
    from experimental_data import drone_pose_initialization
    shutil.copy2(__file__,output/'audit_source.py')
    shutil.copy2(drone_pose_initialization.__file__,output/'initializer_source.py')
    write_json(output/'audit.json',dict(platform=platform.platform(),device='CPU',source_processed_version=str(source),
        pre_command_margin_s=args.pre_command_margin_s,
        margin_interpretation='A preparation margin, not calibrated transport delay or guaranteed hardware-clock separation',
        attitude_interpretation='Rotation change from measured initial tracked frame, not absolute firmware roll/pitch or thrust-axis tilt',
        initialization='11-sample causal endpoint derivative; last measured P/R, no extrapolation to CSV onset',
        hidden_state='Controller memory is not observed; retained pre-hover command history is an input for later model warm-up',
        model_fit_started=False,training_started=False,records=records,
        protected_sources=originals,source_hashes={p.name:sha256_file(p) for p in output.glob('*source.py')}))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    fig=Figure(figsize=(9,6));axes=fig.subplots(2,1)
    for name,t,angle,error in plot_data:
        axes[0].plot(t,angle,label=name);axes[1].plot(t,error,label=name)
    axes[0].set_ylabel('Attitude change from initial (deg)')
    axes[1].set_ylabel('Change of rigid offset (cm)')
    axes[1].set_xlabel('Time since first observed CSV sample (s)')
    for ax in axes:ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Measured geometry during observed CSV maneuver only');fig.tight_layout()
    fig.savefig(output/'pose_review.png',dpi=150)
    lines=['# Drone pose and pre-hover initialization audit','',
        'No dynamics fitting or training. Each state is timestamped before CSV playback; no extrapolation to command onset.',
        'The 100 ms preparation margin does not establish hardware synchronization. Attitude change is relative to the measured initial pose, not absolute body tilt.','',
        '| Take | Initial speed (m/s) | Attitude change max (deg) | Rigid-offset change max (cm) |',
        '|---|---:|---:|---:|']
    for r in records:lines.append(f'| {r["trial_id"]} | {r["initial_speed_m_s"]:.4f} | {r["max_maneuver_orientation_change_deg"]:.2f} | {100*r["max_maneuver_rotating_offset_change_m"]:.2f} |')
    lines+=['','Commanded hover does not imply zero measured velocity or zero tracking error. No internal controller integral/bias is identified here.',
        'The next implementation step is a nominal P/V/attitude response with explicit command history and hidden-state initialization. Do not fit the old attachment-only surrogate unchanged.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(output)
    for row in records:print(row['trial_id'],row['initial_speed_m_s'],row['max_maneuver_orientation_change_deg'],row['max_maneuver_rotating_offset_change_m'])


if __name__=='__main__':main()
