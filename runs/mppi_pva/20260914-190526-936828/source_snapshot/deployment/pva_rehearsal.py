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


def complete_pva_packets(whip,hover,limits=None,*,recovery_settings=None,jerk_limits=None):
    """Avoid a large turning detour when the maneuver already ends near hover."""
    if recovery_settings is not None:
        from deployment.braking_recovery import complete_packets as brake_return
        return brake_return(whip,hover,limits,jerk_limits,recovery_settings)
    from deployment.curved_recovery import coefficients,evaluate,permitted,DEFAULTS
    options=None
    if limits is not None:
        # Active PVA recovery uses this run's envelope. Legacy force shaping
        # limits are not an additional, hidden constraint on jerk/PVA plans.
        options=dict(maximum_downward_acceleration_m_s2=9.80665-limits['minimum_specific_vertical_m_s2'],
            maximum_horizontal_acceleration_m_s2=limits['maximum_specific_force_m_s2'],
            maximum_descent_speed_m_s=limits['maximum_speed_m_s'],maximum_tilt_deg=limits['maximum_tilt_deg'],
            maximum_speed_m_s=limits['maximum_speed_m_s'],maximum_specific_force_m_s2=limits['maximum_specific_force_m_s2'],
            minimum_specific_vertical_m_s2=limits['minimum_specific_vertical_m_s2'])
    p,v,a=whip[-1,:3],whip[-1,3:6],whip[-1,6:9]
    bounds=None if limits is None else (limits['minimum_origin_z_m'],limits['maximum_origin_z_m'])
    if np.linalg.norm(p-hover)>.1 or np.linalg.norm(v)>.2 or np.linalg.norm(a)>.5:
        return complete_packets(whip,hover,options,height_bounds=bounds)
    duration=2.;c=coefficients(p,v,a,np.asarray(hover),np.zeros(3),duration)
    valid,metrics=permitted(c,duration,DEFAULTS,DEFAULTS['maximum_tilt_deg'])
    if not valid:return complete_packets(whip,hover,options,height_bounds=bounds)
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
    from planning.position_spline import SCHEMA as SPLINE_SCHEMA, PositionSpline, replay as replay_spline
    spline_mode=cfg.get('command_contract')==SPLINE_SCHEMA
    if cfg['method']=='mppi':
        with np.load(job/'plan.npz') as data:
            if 'plan_complete' in data and not bool(data['plan_complete']):raise ValueError('Receding MPPI plan is partial; finish the maneuver before export')
            if spline_mode:
                free=data['position_control_points_m'].copy()
                if free.shape!=(9,3) or not np.isfinite(free).all():raise ValueError('Finite nine-point spline required')
                spline=PositionSpline(cfg['task']['duration_s'])
                saved_packets,jerk=spline.decode(torch.tensor(free),cfg['launch']['origin_m'])
                if not bool(spline.jerk_valid(torch.tensor(free),cfg['launch']['origin_m'],torch.tensor(cfg['action']['jerk_limit_m_s3']))):
                    raise ValueError('Spline exceeds the continuous jerk envelope')
                if not np.allclose(data['command_packets'],saved_packets.numpy(),atol=1e-10,rtol=0):
                    raise ValueError('Saved spline packets differ from its control points')
                saved_actions=jerk.numpy()/np.asarray(cfg['action']['jerk_limit_m_s3'])
            else:saved_actions=data['normalized_jerk'].copy()
            if 'committed_steps' in data and int(data['committed_steps'])!=len(saved_actions):raise ValueError('Saved committed plan length differs')
        if saved_actions.ndim!=2 or saved_actions.shape[1]!=3 or not len(saved_actions) or not np.isfinite(saved_actions).all() or np.abs(saved_actions).max()>1+1e-6:
            raise ValueError('Saved plan requires finite bounded XYZ jerk actions')
    env=PVAEnvironment(model,cfg,root=job,device=device)
    if cfg['method']=='ppo':
        checkpoint=Path(checkpoint or job/'checkpoints/best.pt').resolve()
        agent=load_policy(checkpoint,env,cfg);result=env.rollout(policy=agent.deterministic_action,trace=True)
    else:
        if len(saved_actions)>env.steps:raise ValueError('Saved plan exceeds its frozen maneuver time limit')
        actions=env.tensor(saved_actions)[None]
        result=replay_spline(env,env.tensor(free)[None],trace=True) if spline_mode else env.rollout(actions=actions,trace=True,max_steps=len(saved_actions))
        if bool(env.active.any()):raise ValueError('Saved plan ends before a modeled hit or maneuver time limit; no CSV exported')
    if bool(result['failed'][0]):raise ValueError('Whip fails the command, workspace or model envelope; no CSV exported')
    from planning.strike_objective import requires_fold, strike_direction_allowed, strike_angle_deg, uses_templates, rewarded_speed, strike_speed_allowed
    if env.targeted_strike and requires_fold(cfg) and not bool(result['fold_valid'][0]):
        raise ValueError('No verified travelling fold before the strike event; no CSV exported')
    if env.targeted_strike and 'maximum_strike_angle_deg' in cfg['trajectory_objective']:
        if not bool(env.strike_valid[0]) or not bool(strike_direction_allowed(env.strike_velocity,env.direction,cfg['trajectory_objective'])[0]):
            raise ValueError('No scored strike within the saved direction cone; no CSV exported')
    if env.targeted_strike and 'minimum_tip_speed_gain_m_s' in cfg['trajectory_objective']:
        if not bool(env.strike_valid[0]) or not bool(strike_speed_allowed(env.strike_velocity,env.direction,cfg['trajectory_objective'],env.strike_root_velocity)[0]):
            raise ValueError('No strike meets the saved minimum tip speed gain; no CSV exported')
    cutoff=int(result['cutoffs'][0]);whip=result['packets'][0,:cutoff+1].cpu().numpy()
    if len(whip)<2:raise ValueError('Empty maneuver')
    if progress:progress('Appending smooth recovery',0,1)
    times,packets,phases,recovery=complete_pva_packets(whip,np.asarray(cfg['launch']['origin_m']),cfg['limits'],
        recovery_settings=cfg.get('recovery'),jerk_limits=cfg['action']['jerk_limit_m_s3'])
    valid,metrics=reference_packet_validity(env.tensor(packets)[None],cfg['limits'])
    if not bool(valid.all()):raise ValueError('Complete recovery exceeds the saved PVA command envelope')
    if packets[:,2].max()>cfg['limits']['maximum_origin_z_m']+1e-8 or packets[:,2].min()<cfg['limits']['minimum_origin_z_m']-1e-8:
        raise ValueError('Complete recovery exceeds the saved height envelope; whip is preserved, CSV not exported')
    count=round(times[-1]/env.dt);grid=np.arange(count+1)*env.dt
    pose=env.engine.drone.predict(env.initial_pose,env.tensor(packets)[None],times,grid,env.engine.offset,graph=True,hover_command=env.hover,maximum_tilt_deg=cfg['limits']['maximum_tilt_deg'])
    if not bool(pose['valid'].all()):
        raise ValueError('Complete prediction exceeds the attitude or model envelope; no CSV exported')
    prefix=min(cutoff*env.stride,len(env.frames))
    streamed=torch.stack([f['origin'] for f in env.frames[:prefix]],1)
    difference=float((streamed-pose['position_origin_m'][:,1:prefix+1]).abs().max())
    if difference>1e-9:raise AssertionError(f'PVA training/export pose mismatch: {difference}')
    # Saved ghost uses this exact command with no feedback or second conversion.
    state=env.initial_state;positions=[state.positions_m[0].cpu().numpy()];velocities=[state.velocities_m_s[0].cpu().numpy()]
    for k in range(count):
        if not bool(pose['valid'][:,k+1].all()):break
        proposed=env.cable_stepper(state,pose['position_attachment_m'][:,k+1])
        q,v=proposed.positions_m,proposed.velocities_m_s
        if not bool(torch.isfinite(q).all()&torch.isfinite(v).all()) or bool((q.abs()>20).any()) or bool((v.norm(dim=-1)>100).any()):break
        state=proposed;positions.append(q[0].cpu().numpy());velocities.append(v[0].cpu().numpy())
        if progress and k%150==0:progress('Predicting complete command sequence',k+1,count)
    positions=np.asarray(positions)
    if len(positions)<=prefix:raise ValueError('Complete replay fails inside the scored whip')
    if len(positions)!=count+1:raise ValueError('Model prediction fails during recovery; complete CSV not exported')
    origin_z=pose['position_origin_m'][0,:len(positions),2]
    if (bool((origin_z<cfg['limits']['minimum_origin_z_m']).any()) or bool((origin_z>cfg['limits']['maximum_origin_z_m']).any())
            or positions[:,:,2].min()<cfg['limits']['minimum_cable_z_m']):
        raise ValueError('Complete predicted recovery leaves the saved drone/cable height envelope; no CSV exported')
    streamed_q=torch.stack([f['cable'] for f in env.frames[:prefix]],1)[0].cpu().numpy()
    cable_difference=float(np.max(np.abs(positions[1:prefix+1]-streamed_q)))
    if cable_difference>1e-8:raise AssertionError(f'PVA training/export cable mismatch: {cable_difference}')
    output.mkdir(parents=True);saved=freeze_model_assets(model,output,source_root=job)
    from learning.ppo_trajectory_reward import freeze_reference
    freeze_reference(cfg,output,source_root=job)
    atomic_json(output/'model.json',saved);atomic_json(output/'settings.json',cfg)
    atomic_json(output/'task.json',dict(desired_strike_direction_world=cfg['task']['strike_direction'],
        target_position_m=cfg['launch']['target_m'],
        **(dict(target_positions_m=cfg['task']['target_sequence_m']) if env.extra_tick_fields else {}),
        **(dict(target_marker_radius_m=.02,acceptance=('Travelling fold before the scored strike event; continuous distance and speed' if requires_fold(cfg) else 'Feasible targeted strike; continuous distance and speed; fold diagnostic only'),
                objective=cfg['trajectory_objective'],fold_constraint=cfg['fold_constraint'])
           if env.targeted_strike else dict(success=dict(tip_target_distance_m=cfg['task']['target_radius_m'])))))
    with (output/'fullstate_30hz.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(FIELDS);writer.writerows(np.c_[times,packets])
    jerk=result['actions'][0,:cutoff].cpu().numpy()*np.asarray(cfg['action']['jerk_limit_m_s3'])
    with (output/'jerk_30hz.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['time_s','jx_m_s3','jy_m_s3','jz_m_s3']);writer.writerows(np.c_[np.arange(len(jerk))/30,jerk])
    n=len(positions)
    target_arrays={}
    if env.extra_tick_fields:target_arrays=dict(target_positions_m=env.target_centers.cpu().numpy(),
        target_hit_times_s=env.target_hit_times[0].cpu().numpy())
    np.savez_compressed(output/'rehearsal.npz',command_time_s=times,commands=packets,command_phase=phases,
        prediction_time_s=grid[:n],cable_positions_m=positions,cable_velocities_m_s=np.asarray(velocities),origin_positions_m=pose['position_origin_m'][0,:n].cpu().numpy(),
        origin_velocities_m_s=pose['velocity_origin_m_s'][0,:n].cpu().numpy(),
        origin_rotations=pose['rotation_tracking_to_world'][0,:n].cpu().numpy(),target_position_m=cfg['launch']['target_m'],
        jerk_time_s=np.arange(len(jerk))/30,jerk_m_s3=jerk,**target_arrays)
    metadata=dict(schema='pva_fullstate_30hz_v1',command_contract=cfg.get('command_contract',SCHEMA),planner=cfg['method'].upper()+' PVA',
        planner_mode=cfg.get('mppi',{}).get('mode','open_loop') if cfg['method']=='mppi' else 'ppo',
        lookahead_s=cfg.get('mppi',{}).get('horizon_s') if cfg['method']=='mppi' else None,
        plan_sha256=sha256_file(job/'plan.npz') if cfg['method']=='mppi' else None,
        job=str(job),checkpoint=str(checkpoint) if checkpoint else None,checkpoint_sha256=sha256_file(checkpoint) if checkpoint else None,
        initial_tracking_origin_m=cfg['launch']['origin_m'],target_position_m=cfg['launch']['target_m'],whip_end_s=cutoff/30,
        total_duration_s=float(times[-1]),predicted_valid_hit=bool(result['success'][0]),minimum_tip_distance_m=float(result['minimum_tip_distance_m'][0]),
        scored_duration_s=float(result['duration_s'][0]),predicted_hit_time_s=float(result['duration_s'][0]) if bool(result['success'][0]) else None,
        recovery=recovery,recovery_prediction_complete=n==count+1,prediction_valid_through_s=float(grid[n-1]),
        command_height_range_m=[float(packets[:,2].min()),float(packets[:,2].max())],
        predicted_drone_height_range_m=[float(origin_z.min()),float(origin_z.max())],
        minimum_predicted_cable_height_m=float(positions[:,:,2].min()),
        recovery_empirically_validated=False,reference_feasible=True,predicted_attitude_checked=True,
        maximum_predicted_tilt_limit_deg=cfg['limits']['maximum_tilt_deg'],training_export_prefix_max_difference_m=difference,
        training_export_cable_max_difference_m=cable_difference,
        csv_sha256=sha256_file(output/'fullstate_30hz.csv'),
        command_semantics='Desired OptiTrack tracked-origin P/V/A, kinematic acceleration, 30 Hz zero-order hold, no force or mass compensation',
        execution='Take off; hold 10 s at saved start; execute complete CSV at saved timestamps; land. Offline artifact, no flight sender.',
        evidence='Fitted model simulation only; real flight performance remains to be measured')
    if spline_mode:
        metadata.update(trajectory_representation=SPLINE_SCHEMA,jerk_csv_semantics='Analytic spline jerk sampled at interval midpoints; diagnostic only, not the source of PVA packets')
    if env.targeted_strike:
        metadata.pop('predicted_valid_hit');metadata.pop('predicted_hit_time_s')
        if uses_templates(cfg):shutil.copy2(job/'proposal_baselines.npz',output/'proposal_baselines.npz')
        metadata.update(predicted_fold_valid=bool(result['fold_valid'][0]),fold_requirement=cfg.get('fold_requirement','required'),
            fold_completed_time_s=float(env.fold_completed_time_s[0]) if bool(torch.isfinite(env.fold_completed_time_s[0])) else None,
            objective_schema=cfg['trajectory_objective']['schema'],
            initialization=cfg['mppi'].get('initialization','templates'),
            proposal_baselines_sha256=sha256_file(output/'proposal_baselines.npz') if uses_templates(cfg) else None,
            strike_time_s=float(env.strike_time[0]),strike_distance_m=float(env.strike_distance[0]),
            strike_angle_deg=float(strike_angle_deg(env.strike_velocity[0],env.direction)),
            maximum_strike_angle_deg=cfg['trajectory_objective'].get('maximum_strike_angle_deg'),
            speed_metric=cfg['trajectory_objective'].get('speed_metric','absolute_tip_speed'),
            minimum_tip_speed_gain_m_s=cfg['trajectory_objective'].get('minimum_tip_speed_gain_m_s'),
            root_forward_speed_m_s=float((env.strike_root_velocity[0]*env.direction).sum()),
            rewarded_tip_speed_m_s=float(rewarded_speed(env.strike_velocity[0],env.direction,cfg['trajectory_objective'],env.strike_root_velocity[0])),
            root_velocity_m_s=env.strike_root_velocity[0].cpu().tolist(),
            directed_tip_speed_m_s=float((env.strike_velocity[0]*env.direction).sum()),
            closest_approach_time_s=float(env.encounter_time[0]))
    if cfg['task'].get('require_pullback',False):
        forward_speed=(env.pose.velocity[0]*env.direction).sum()
        position=((env.pose.position-env.origin0)*env.direction).sum(-1)[0]
        metadata['pullback']=dict(required=cfg['task'].get('success_criterion','legacy_strike_v1') not in ('tip_contact_v1','ordered_two_target_v1'),pull_completed=bool(env.pull_ready[0]),reverse_completed=bool(env.reverse_ready[0]),
            peak_forward_displacement_m=float(env.pull_peak[0]),backward_travel_m=float(env.pull_peak[0]-position),
            drone_forward_speed_at_handover_m_s=float(forward_speed),
            note='Physical simulated tracked-origin motion, not command reversal; criteria evaluated at the hit, values here at the next 30 Hz handover boundary')
        if metadata['predicted_hit_time_s'] is not None:
            hit=metadata['predicted_hit_time_s'];axis=env.direction.cpu().numpy()
            projected=(pose['position_origin_m'][0,:n].cpu().numpy()-np.asarray(cfg['launch']['origin_m']))@axis
            drone_speed=pose['velocity_origin_m_s'][0,:n].cpu().numpy()@axis
            tip_speed=np.asarray(velocities)[:,-1]@axis
            peak=float(projected[grid[:n]<=hit].max())
            metadata['pullback'].update(drone_forward_speed_at_contact_m_s=float(np.interp(hit,grid[:n],drone_speed)),
                tip_forward_speed_at_contact_m_s=float(np.interp(hit,grid[:n],tip_speed)),
                backward_travel_at_contact_m=peak-float(np.interp(hit,grid[:n],projected)),
                contact_evaluation='Linear interpolation within the intersected physics interval, consistent with sphere-entry timing')
    if cfg['task'].get('require_wave',False):
        metadata['wave']=dict(required=cfg['task'].get('success_criterion','legacy_strike_v1') not in ('tip_contact_v1','ordered_two_target_v1'),completed_stages=int(env.wave_stage[0]),
            completion_time_s=float(env.wave_completion_time[0]) if bool(torch.isfinite(env.wave_completion_time[0])) else None,
            definition='Dominant local bend persists through proximal, middle, distal material bands before contact; kinematic proxy only')
    metadata['success_criterion']=cfg['task'].get('success_criterion','legacy_strike_v1')
    metadata['success_meaning']=('Tip enters the target sphere; speed, reversal and wave are diagnostics, not pass/fail gates.'
        if metadata['success_criterion']=='tip_contact_v1' else 'Historical composite strike requirements in saved settings.')
    if env.targeted_strike:
        metadata['success_meaning']=('Predicted travelling fold before the scored encounter' if requires_fold(cfg) else 'Feasible scored strike; travelling fold is diagnostic only')+'; neither a hit label nor measured impact energy.'
    if env.extra_tick_fields:
        from learning.two_target_whip import details
        metadata.update(details(env,0),target_positions_m=cfg['task']['target_sequence_m'],
            planner='MPPI PVA · two targets',
            success_meaning='Tip enters two distinct virtual target spheres in order, in one continuous maneuver; no state reset or collision response.')
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
    model=read_json(directory/'model.json')
    if model['motion_residual'].get('enabled') is not False:
        model['motion_residual']['checkpoint']='assets/cable_residual.pt'
    model['fullstate_execution']['checkpoint']='assets/drone_model.json'
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
