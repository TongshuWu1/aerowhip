"""Replay direct-PVA plans, preserve the whip, append recovery and export."""
from pathlib import Path
import csv
import json
import math
import shutil
import zipfile
import numpy as np
import torch
from experimental_data.io import atomic_json,sha256_file
from simulator.workflow import read_json
from simulator.research_config import snapshot_assets
from simulator.research_reference import reference_packet_validity
from simulator.cable import DderState
from simulator.pva_commands import SCHEMA
from learning.pva_env import PVAEnvironment
from planning.pva_job import load_policy,freeze_model_assets
from deployment.research_rehearsal import complete_packets,FIELDS


def complete_pva_packets(whip,hover):
    """Avoid a large turning detour when the maneuver already ends near hover."""
    from deployment.curved_recovery import coefficients,evaluate,permitted,DEFAULTS
    p,v,a=whip[-1,:3],whip[-1,3:6],whip[-1,6:9]
    if np.linalg.norm(p-hover)>.1 or np.linalg.norm(v)>.2 or np.linalg.norm(a)>.5:
        return complete_packets(whip,hover)
    duration=2.;c=coefficients(p,v,a,np.asarray(hover),np.zeros(3),duration)
    valid,metrics=permitted(c,duration,DEFAULTS,DEFAULTS['maximum_tilt_deg'])
    if not valid:return complete_packets(whip,hover)
    tail_t=np.arange(1,151)/30;pva=evaluate(c,np.minimum(tail_t,duration),duration)
    tail=np.c_[*pva,np.zeros((len(tail_t),2))]
    packets=np.concatenate((whip,tail));times=np.arange(len(packets))/30;end=(len(whip)-1)/30
    phases=np.where(times<=end+1e-10,1,np.where(times<=end+duration,3,4))
    return times,packets,phases,dict(profile='gentle_near_hover_return',brake_end_s=duration,return_end_s=duration,
        total_duration_s=5.,metrics=metrics,coefficients=c.tolist())


def generate(job,output,*,checkpoint=None,origin=None,target=None,device='cuda',progress=None):
    job=Path(job).resolve();output=Path(output).resolve()
    cfg=read_json(job/'settings.json');model=read_json(job/'model.json')
    if read_json(job/'status.json',{}).get('status')=='running':raise ValueError('Stop or finish the run before rehearsing its saved plan')
    if origin is not None:cfg['launch']['origin_m']=list(origin)
    if target is not None:cfg['launch']['target_m']=list(target)
    if cfg['method']=='mppi' and (origin is not None or target is not None):
        frozen=read_json(job/'settings.json')['launch']
        if cfg['launch']['origin_m']!=frozen['origin_m'] or cfg['launch']['target_m']!=frozen['target_m']:
            raise ValueError('MPPI launch changed: optimize a new plan before rehearsing')
    if output.exists():raise FileExistsError('Use a new rehearsal folder')
    env=PVAEnvironment(model,cfg,root=job,device=device)
    if cfg['method']=='ppo':
        checkpoint=Path(checkpoint or job/'checkpoints/best.pt').resolve()
        agent=load_policy(checkpoint,env,cfg);result=env.rollout(policy=agent.deterministic_action,trace=True)
    else:
        with np.load(job/'plan.npz') as data:actions=env.tensor(data['normalized_jerk'])[None]
        result=env.rollout(actions=actions,trace=True)
    if bool(result['failed'][0]):raise ValueError('Whip fails the command, workspace or model envelope; no CSV exported')
    cutoff=int(result['cutoffs'][0]);whip=result['packets'][0,:cutoff+1].cpu().numpy()
    if len(whip)<2:raise ValueError('Empty maneuver')
    if progress:progress('Appending smooth recovery',0,1)
    times,packets,phases,recovery=complete_pva_packets(whip,np.asarray(cfg['launch']['origin_m']))
    valid,metrics=reference_packet_validity(env.tensor(packets)[None],cfg['limits'])
    if not bool(valid.all()):raise ValueError('Complete recovery exceeds the saved PVA command envelope')
    if packets[:,2].max()>cfg['limits']['maximum_origin_z_m']+1e-8 or packets[:,2].min()<cfg['limits']['minimum_origin_z_m']-1e-8:
        raise ValueError('Complete recovery exceeds the saved height envelope; whip is preserved, CSV not exported')
    count=round(times[-1]/env.dt);grid=np.arange(count+1)*env.dt
    pose=env.engine.drone.predict(env.initial_pose,env.tensor(packets)[None],times,grid,env.engine.offset,graph=True,hover_command=env.hover)
    prefix=min(cutoff*env.stride,len(env.frames))
    streamed=torch.stack([f['origin'] for f in env.frames[:prefix]],1)
    difference=float((streamed-pose['position_origin_m'][:,1:prefix+1]).abs().max())
    if difference>1e-9:raise AssertionError(f'PVA training/export pose mismatch: {difference}')
    # Saved ghost uses this exact command with no feedback or second conversion.
    state=env.initial_state;positions=[state.positions_m[0].cpu().numpy()]
    for k in range(count):
        if not bool(pose['valid'][:,k+1].all()):break
        proposed=env.cable_stepper(state,pose['position_attachment_m'][:,k+1])
        q,v=proposed.positions_m,proposed.velocities_m_s
        if not bool(torch.isfinite(q).all()&torch.isfinite(v).all()) or bool((q.abs()>20).any()) or bool((v.norm(dim=-1)>100).any()):break
        state=proposed;positions.append(q[0].cpu().numpy())
        if progress and k%150==0:progress('Predicting complete command sequence',k+1,count)
    positions=np.asarray(positions)
    if len(positions)<=prefix:raise ValueError('Complete replay fails inside the scored whip')
    if len(positions)!=count+1:raise ValueError('Model prediction fails during recovery; complete CSV not exported')
    streamed_q=torch.stack([f['cable'] for f in env.frames[:prefix]],1)[0].cpu().numpy()
    cable_difference=float(np.max(np.abs(positions[1:prefix+1]-streamed_q)))
    if cable_difference>1e-8:raise AssertionError(f'PVA training/export cable mismatch: {cable_difference}')
    output.mkdir(parents=True);saved=freeze_model_assets(model,output)
    atomic_json(output/'model.json',saved);atomic_json(output/'settings.json',cfg)
    atomic_json(output/'task.json',dict(desired_strike_direction_world=cfg['task']['strike_direction'],
        target_position_m=cfg['launch']['target_m'],success=dict(tip_target_distance_m=cfg['task']['target_radius_m'])))
    with (output/'fullstate_30hz.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(FIELDS);writer.writerows(np.c_[times,packets])
    jerk=result['actions'][0,:cutoff].cpu().numpy()*np.asarray(cfg['action']['jerk_limit_m_s3'])
    with (output/'jerk_30hz.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['time_s','jx_m_s3','jy_m_s3','jz_m_s3']);writer.writerows(np.c_[np.arange(len(jerk))/30,jerk])
    n=len(positions)
    np.savez_compressed(output/'rehearsal.npz',command_time_s=times,commands=packets,command_phase=phases,
        prediction_time_s=grid[:n],cable_positions_m=positions,origin_positions_m=pose['position_origin_m'][0,:n].cpu().numpy(),
        origin_rotations=pose['rotation_tracking_to_world'][0,:n].cpu().numpy(),target_position_m=cfg['launch']['target_m'],
        jerk_time_s=np.arange(len(jerk))/30,jerk_m_s3=jerk)
    metadata=dict(schema='pva_fullstate_30hz_v1',command_contract=SCHEMA,planner=cfg['method'].upper()+' PVA',
        job=str(job),checkpoint=str(checkpoint) if checkpoint else None,checkpoint_sha256=sha256_file(checkpoint) if checkpoint else None,
        initial_tracking_origin_m=cfg['launch']['origin_m'],target_position_m=cfg['launch']['target_m'],whip_end_s=cutoff/30,
        total_duration_s=float(times[-1]),predicted_valid_hit=bool(result['success'][0]),minimum_tip_distance_m=float(result['minimum_tip_distance_m'][0]),
        scored_duration_s=float(result['duration_s'][0]),predicted_hit_time_s=float(result['duration_s'][0]) if bool(result['success'][0]) else None,
        recovery=recovery,recovery_prediction_complete=n==count+1,prediction_valid_through_s=float(grid[n-1]),
        recovery_empirically_validated=False,reference_feasible=True,training_export_prefix_max_difference_m=difference,
        training_export_cable_max_difference_m=cable_difference,
        csv_sha256=sha256_file(output/'fullstate_30hz.csv'),
        command_semantics='Desired OptiTrack tracked-origin P/V/A, kinematic acceleration, 30 Hz zero-order hold, no force or mass compensation',
        execution='Take off; hold 10 s at saved start; execute complete CSV at saved timestamps; land. Offline artifact, no flight sender.',
        evidence='Fitted model simulation only; real flight performance remains to be measured')
    atomic_json(output/'rehearsal.json',metadata)
    if checkpoint:
        (output/'checkpoints').mkdir();shutil.copy2(checkpoint,output/'checkpoints/policy.pt')
    else:shutil.copy2(job/'plan.npz',output/'plan.npz')
    if progress:progress('Rehearsal saved',1,1)
    return metadata


def export_package(directory,destination):
    directory=Path(directory);destination=Path(destination)
    if destination.resolve().is_relative_to(directory.resolve()):raise ValueError('Save ZIP outside the rehearsal directory')
    meta=read_json(directory/'rehearsal.json')
    if meta['schema']!='pva_fullstate_30hz_v1':raise ValueError('Direct PVA rehearsal required')
    if sha256_file(directory/'fullstate_30hz.csv')!=meta['csv_sha256']:raise ValueError('Saved command CSV changed')
    model=read_json(directory/'model.json');model['motion_residual']['checkpoint']='assets/cable_residual.pt';model['fullstate_execution']['checkpoint']='assets/drone_model.json'
    snapshot=Path(meta['job'])/'source_snapshot'
    if not snapshot.is_dir():raise ValueError('Original run source snapshot is missing; cannot export a reproducible policy bundle')
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in directory.rglob('*'):
            if path.is_file() and path.name!='model.json':archive.write(path,path.relative_to(directory).as_posix())
        archive.writestr('model.json',json.dumps(model,indent=2))
        for path in snapshot.rglob('*.py'):archive.write(path,path.relative_to(snapshot).as_posix())
        archive.writestr('README.txt',meta['execution']+'\n\n'+meta['command_semantics']+'\n\n'+
            'The complete CSV includes whip, recovery and final hold. Jerk CSV documents reference generation; it is not sent to the aircraft.\n'+
            'The saved NPZ is the original prediction for comparison with measured flights. Do not replace it with a later fitted model.\n'+
            'This package contains commands, frozen model assets and the originating run source. Use a compatible Python 3.12/PyTorch CUDA environment.\n'+
            ('Regenerate offline: python tools/rehearse_pva.py --job . --checkpoint checkpoints/policy.pt --output new_rehearsal\n' if meta.get('checkpoint') else
             'Replay offline: python tools/rehearse_pva.py --job . --output new_rehearsal\n'))
