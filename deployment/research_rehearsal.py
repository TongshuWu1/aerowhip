"""Offline 30 Hz export using exactly the reference scored during PPO.

The prefix is frozen before appending analytic recovery. There is no flight sender.
"""
from pathlib import Path
import csv
import json
import math
import shutil
import numpy as np
import torch

from experimental_data.io import atomic_json,sha256_file
from simulator.workflow import read_json
from simulator.research_pose import settled_initial
from simulator.research_reference import reference_feasibility
from simulator.research_config import snapshot_assets
from learning.point_force_env import PointForceWhipEnvironment
from learning.deployment_rollout import DeploymentBatch,plan_batch
from learning.research_rollout import execute_research_batch
from .curved_recovery import plan_curved_recovery

FIELDS=['time_s','px_m','py_m','pz_m','vx_m_s','vy_m_s','vz_m_s',
        'ax_m_s2','ay_m_s2','az_m_s2','yaw_rad','yaw_rate_rad_s']


def rehearsal_prefix(score,frames,env):
    """Round a predicted hit up to a 30 Hz boundary, then start recovery.

    The few sub-packet samples after the scored terminal event are unscored.
    Preserve every P/V/A packet of the used prefix, rather than regenerating
    its acceleration from a truncated virtual rollout.
    """
    state=score.execution_state
    if not hasattr(score,'command_cutoffs'):
        return score.reference_packets[0].cpu().numpy(),frames,state,score.predicted_pose
    end=int(score.command_cutoffs[0]);frames=list(frames)
    for i in range(len(frames)-1,end):
        if not bool(score.predicted_pose['valid'][:,i+1].all()):
            raise ValueError('Predicted pose fails before the 30 Hz recovery boundary.')
        state=env._research_cable(state,score.predicted_pose['position_attachment_m'][:,i+1])
        if not bool(torch.isfinite(state.positions_m).all()&torch.isfinite(state.velocities_m_s).all()) or bool(
            (state.positions_m.abs()>env.numerical_position_limit_m).any() | (state.velocities_m_s.norm(dim=-1)>env.numerical_speed_limit_m_s).any()):
            raise ValueError('Cable prediction fails before the 30 Hz recovery boundary.')
        frames.append(state.positions_m[0].cpu().numpy().copy())
    pose={k:v[:,:end+1] for k,v in score.predicted_pose.items()}
    whip=score.reference_packets[0,:end//env.physics_steps_per_control+1].cpu().numpy()
    return whip,frames,state,pose


def complete_packets(whip,hover,settings=None):
    whip=np.asarray(whip,dtype=float)
    if whip.ndim!=2 or whip.shape[1]!=11 or len(whip)<2 or not np.isfinite(whip).all():
        raise ValueError('A finite native 30 Hz whip reference is required.')
    sample,recovery=plan_curved_recovery(whip[-1,:3],whip[-1,3:6],whip[-1,6:9],hover,settings)
    tail_t=np.arange(1,math.ceil(recovery['total_duration_s']*30)+1)/30
    p,v,a=sample(np.minimum(tail_t,recovery['total_duration_s']))
    tail=np.c_[p,v,a,np.zeros((len(p),2))]
    packets=np.concatenate((whip,tail));times=np.arange(len(packets))/30
    end=(len(whip)-1)/30
    phases=np.where(times<=end+1e-10,1,np.where(times<=end+recovery['brake_end_s'],2,
        np.where(times<=end+recovery['return_end_s'],3,4)))
    return times,packets,phases,recovery


def generate(checkpoint,output,origin,target,device='cuda',progress=None):
    from run_ppo import build_agent,_load_checkpoint,configure_accelerator,validate_contract
    device=torch.device(device)
    configure_accelerator(device)
    checkpoint=Path(checkpoint).resolve();source=checkpoint.parent.parent
    model,task,config=[read_json(source/f'{n}.json') for n in ('model','task','ppo')]
    for spec in [model.get('motion_residual',{}),model.get('fullstate_execution',{})]:
        if spec.get('checkpoint') and not Path(spec['checkpoint']).is_absolute():spec['checkpoint']=str((source/spec['checkpoint']).resolve())
    validate_contract(model,task,config)
    if model.get('fullstate_execution',{}).get('schema')!='tracked_pose_execution_v1':
        raise ValueError('Select a native 30 Hz tracked-pose PPO. Legacy policies retain their original rehearsal.')
    if not math.isclose(task['control_dt_s'],1/30,abs_tol=1e-12):raise ValueError('Policy rate is not 30 Hz.')
    origin,target=np.asarray(origin,float),np.asarray(target,float)
    if origin.shape!=(3,) or target.shape!=(3,) or not np.isfinite([origin,target]).all():raise ValueError('Finite XYZ positions are required.')
    offset=np.asarray(model['recorded_data']['optitrack_to_attachment_offset_body_m'])
    task['initial_root_position_m']=(origin+offset).tolist();task['initial_root_velocity_m_s']=[0.,0.,0.];task['target_position_m']=target.tolist()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    # Freeze checkpoint bytes before loading: latest.pt can change during training.
    (output/'checkpoints').mkdir(exist_ok=True)
    frozen=output/'checkpoints/policy.pt';shutil.copy2(checkpoint,frozen)
    agent=build_agent(config,torch.device(device));_load_checkpoint(agent,frozen,load_optimizer=False)
    env=PointForceWhipEnvironment(model,task,config,batch_size=1,device=device)
    env.reset();state=env.state;one=torch.ones(1,device=env.device,dtype=env.dtype)
    batch=DeploymentBatch(state,state,one,one,one[:,None].expand(-1,3),one[:,None]*0,
        torch.ones(1,device=env.device,dtype=torch.bool),env.target.clone())
    frames=[state.positions_m[0].detach().cpu().numpy().copy()]
    def trace(i,force,current,*_):frames.append(current.positions_m[0].detach().cpu().numpy().copy())
    with torch.no_grad():
        forces,cutoffs=plan_batch(env,agent,batch,progress=progress)
        score=execute_research_batch(env,batch,forces,cutoffs,config['deployment'],trace=trace,progress=progress)
        if bool(score.failed[0]) or not hasattr(score,'reference_packets'):
            raise ValueError('Policy produced an infeasible reference or failed model prediction; no CSV was exported.')
        whip,frames,whip_end_state,whip_pose=rehearsal_prefix(score,frames,env)
        if hasattr(score,'command_cutoffs'):cutoffs=score.command_cutoffs
        times,packets,phases,recovery=complete_packets(whip,origin)
        dt=env.physics_dt_s;count=(len(packets)-1)*env.physics_steps_per_control
        packet_tensor=state.positions_m.new_tensor(packets)[None]
        valid,metrics=reference_feasibility(packet_tensor,cutoffs.new_tensor([count]),dt,model['fullstate_execution']['feasibility'])
        if not bool(valid.all()):raise ValueError('Complete reference exceeds the configured acceleration/tilt/speed envelope.')
        if progress:progress('Predicting complete reference including unvalidated recovery',0,count)
        hover=packet_tensor[:,0].clone();hover[:,3:9]=0
        pose=env._research_tracker.predict(settled_initial(state.positions_m[:,0],state.velocities_m_s[:,0],offset),
            packet_tensor,times,np.arange(count+1)*dt,offset,graph=bool(config['cuda_graph_physics']),hover_command=hover)
        # Same event schedule and numerical integration at the entire whip prefix.
        prefix=len(frames)
        discrepancy=(pose['position_origin_m'][:,:prefix]-whip_pose['position_origin_m']).abs().max().item()
        if discrepancy>1e-8:raise ValueError(f'Training/export prediction mismatch: {discrepancy:g} m')
        cable=list(frames);current=whip_end_state;valid_end=prefix-1
        for i in range(prefix-1,count):
            if not bool(pose['valid'][:,i+1].all()):break
            candidate=env._research_cable(current,pose['position_attachment_m'][:,i+1])
            finite=torch.isfinite(candidate.positions_m).all()&torch.isfinite(candidate.velocities_m_s).all()
            if not bool(finite) or bool((candidate.positions_m.abs()>env.numerical_position_limit_m).any()) or bool((candidate.velocities_m_s.norm(dim=-1)>env.numerical_speed_limit_m_s).any()):break
            current=candidate;cable.append(current.positions_m[0].cpu().numpy().copy());valid_end=i+1
            if progress and i%75==0:progress('Predicting recovery (outside fitted maneuver scope)',i+1,count)
    predicted=np.asarray(cable)
    # Full desired recovery remains available when its out-of-scope model prediction fails.
    # Playback ends at the last valid predicted sample; it never substitutes reference for truth.
    with (output/'fullstate_30hz.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(FIELDS);writer.writerows(np.c_[times,packets])
    force=forces[:int(cutoffs[0])].cpu().numpy()[:,0]
    with (output/'virtual_force_30hz.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['time_s','fx_total_n','fy_total_n','fz_total_n'])
        indices=np.arange(0,len(force),env.physics_steps_per_control);writer.writerows(np.c_[indices*dt,force[indices]])
    np.savez_compressed(output/'rehearsal.npz',command_time_s=times,commands=packets,command_phase=phases,
        prediction_time_s=np.arange(len(predicted))*dt,cable_positions_m=predicted,
        origin_positions_m=pose['position_origin_m'][0,:len(predicted)].cpu().numpy(),
        origin_rotations=pose['rotation_tracking_to_world'][0,:len(predicted)].cpu().numpy(),
        force_time_s=np.arange(len(force))*dt,virtual_force_n=force,target_position_m=target)
    metadata=dict(schema='research_fullstate_30hz_v1',checkpoint=str(checkpoint),checkpoint_sha256=sha256_file(frozen),
        source_job=model['fullstate_execution']['source_job'],force_rate_hz=30,command_rate_hz=30,
        initial_tracking_origin_m=origin.tolist(),initial_attachment_m=(origin+offset).tolist(),target_position_m=target.tolist(),
        frame='World XYZ, metres; FullState position refers to unshifted OptiTrack tracked origin',
        initial_state='Ideal settled level hover, zero velocity/bias, hanging planning cable',preflight_hold_s=10.,
        whip_end_s=(len(whip)-1)/30,total_duration_s=float(times[-1]),recovery=recovery,
        predicted_valid_hit=bool(score.episode_success[0]),minimum_tip_distance_m=float(score.episode_minimum_tip_distance[0]),
        termination_mode=config['deployment'].get('termination','frozen_virtual_plan'),
        predicted_hit_time_s=float(score.episode_hit_time_s[0]) if bool(score.episode_success[0]) else None,
        scored_duration_s=float(score.deployment.get('termination_time_s',score.deployment['duration_s'])[0]),
        prediction_valid_through_s=valid_end*dt,recovery_prediction_complete=valid_end==count,recovery_empirically_validated=False,
        training_export_prefix_max_difference_m=discrepancy,reference_feasible=True,
        reference_metrics={k:float(v[0]) for k,v in metrics.items()},
        virtual_force_csv='Virtual simulator input, including gravity; not a direct drone force command',
        execution='Take off; hold for 10 s; play complete CSV once at native timestamps; land. Offline artifact only.')
    atomic_json(output/'rehearsal.json',metadata)
    saved=snapshot_assets(model,output)
    for name,value in [('model',saved),('task',task),('ppo',config)]:atomic_json(output/f'{name}.json',value)
    if progress:progress('Complete — reference and predicted execution saved',count,count)
    return metadata


def export_package(directory,destination):
    """Portable offline inference source, frozen weights and exact CSV; no sender."""
    import zipfile
    directory=Path(directory);destination=Path(destination)
    if destination.resolve().is_relative_to(directory.resolve()):raise ValueError('Save the ZIP outside the generated result folder.')
    if not (directory/'rehearsal.json').is_file():raise ValueError('Generate a completed rehearsal first.')
    model=read_json(directory/'model.json')
    if model.get('motion_residual',{}).get('enabled'):
        model['motion_residual']['checkpoint']='assets/cable_residual.pt'
    model['fullstate_execution']['checkpoint']='assets/drone_model.json'
    root=Path(__file__).resolve().parents[1]
    instructions=('Native 30 Hz PPO / FullState offline package\n\n'
        'fullstate_30hz.csv is the exact complete reference; rehearsal.json describes its recovery.\n'
        'The CSV uses unshifted OptiTrack tracked-origin coordinates, metres and seconds.\n'
        'virtual_force_30hz.csv is simulator input INCLUDING gravity, not a drone force command.\n'
        'checkpoints/policy.pt and assets/ contain the frozen PPO and both model residuals.\n'
        'rehearsal.json records the desired start, target, feasibility and prediction limits.\n\n'
        'To generate another OFFLINE plan with this code and a compatible Python/PyTorch CUDA environment:\n'
        'python tools/rehearse_research.py --checkpoint checkpoints/policy.pt --output new_plan '
        '--origin X Y Z --target X Y Z --device cuda\n\n'
        'Use the saved initial_tracking_origin_m values for X Y Z. Initial velocity is zero and the cable is assumed hanging.\n'
        'Tested on Windows 11 / RTX 4080 / Python 3.12 / PyTorch 2.11.0+cu128.\n'
        'No ROS sender or aircraft interface is implemented in this package. Recovery is not empirically validated.\n')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in directory.rglob('*'):
            if path.is_file() and path.name not in ['model.json','console.log','progress.json']:archive.write(path,path.relative_to(directory).as_posix())
        archive.writestr('model.json',json.dumps(model,indent=2));archive.writestr('README.txt',instructions)
        for name in ['simulator','learning','deployment','experimental_data']:
            for path in (root/name).rglob('*.py'):archive.write(path,path.relative_to(root).as_posix())
        for name in ['run_ppo.py','tools/rehearse_research.py','requirements.txt']:
            if (root/name).is_file():archive.write(root/name,name)
    return destination
