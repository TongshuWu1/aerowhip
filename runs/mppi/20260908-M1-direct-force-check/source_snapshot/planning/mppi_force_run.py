"""Immutable direct-force MPPI jobs and PPO-equivalent FullState export."""
from pathlib import Path
from copy import deepcopy
import csv
import shutil
import sys
import os
import time
import numpy as np
import torch
from experimental_data.io import atomic_json,sha256_file
from simulator.workflow import read_json
from simulator.research_config import snapshot_assets,validate_research_contract
from simulator.research_pose import settled_initial
from simulator.research_reference import reference_feasibility
from deployment.research_rehearsal import rehearsal_prefix,complete_packets,FIELDS
from .mppi import optimize
from .mppi_force import ForceEvaluator
from .cem_launch import resolve_launch_setup

DEFAULTS=dict(optimizer='mppi',representation='ppo_force_30hz_v1',population=128,iterations=12,
    temperature=20.,action_std=.05,horizon_s=1.,random_seed=655,device='cuda',
    maximum_height_m=2.8,minimum_height_m=.08,display_name='M1 MPPI force sequence',
    model_path='data/model_candidates/20260908-adp0-M1/model.json')


def validate_settings(settings):
    for key in ('population','iterations','random_seed'):
        if int(settings[key])!=settings[key]:raise ValueError(f'{key} must be an integer.')
    if not 4<=settings['population']<=2048 or not 1<=settings['iterations']<=1000:
        raise ValueError('Use 4–2048 candidates and 1–1000 iterations.')
    if not .2<=settings['horizon_s']<=5 or not np.isclose(settings['horizon_s']*30,round(settings['horizon_s']*30),atol=1e-7):
        raise ValueError('Horizon must be 0.2–5 s on a 30 Hz boundary.')
    if settings['action_std']<=0 or settings['temperature']<=0:raise ValueError('Exploration and temperature must be positive.')
    if any(not np.isfinite(v) for v in settings.values() if isinstance(v,(float,int))):raise ValueError('Settings must be finite.')
    if not 0<=settings['minimum_height_m']<settings['maximum_height_m']:raise ValueError('Invalid height bounds.')
    if settings.get('launch_setup'):resolve_launch_setup(settings['launch_setup'])


def prepare_job(root,seed_directory,output,settings,origin=None,target=None):
    root,seed_directory,output=[Path(p).resolve() for p in (root,seed_directory,output)]
    settings=dict(DEFAULTS,**settings);validate_settings(settings)
    if output.exists() and any(output.iterdir()):raise ValueError('Choose an empty new run directory.')
    with np.load(seed_directory/'rehearsal.npz') as data:
        if 'virtual_force_n' not in data:raise ValueError('Choose a PPO or direct-force MPPI rehearsal containing its force sequence.')
        forces,times=data['virtual_force_n'].copy(),data['force_time_s'].copy()
    task,config=[read_json(seed_directory/f'{n}.json') for n in ('task','ppo')]
    model_path=Path(settings['model_path'])
    if not model_path.is_absolute():model_path=root/model_path
    model_path=model_path.resolve();model=read_json(model_path)
    for name in ('motion_residual','fullstate_execution'):
        p=Path(model[name]['checkpoint'])
        if not p.is_absolute():model[name]['checkpoint']=str(model_path.parent/p)
    reward=deepcopy(config['reward'])
    overrides=settings.get('ppo_reward',{})
    for key,value in overrides.items():
        if key not in reward or not isinstance(value,(int,float)) or isinstance(value,bool) or not np.isfinite(value) or value<0:
            raise ValueError(f'Invalid PPO reward override: {key}')
    changed=any(reward[k]!=v for k,v in overrides.items())
    reward.update(overrides)
    if changed:reward.pop('equation',None)  # Do not retain stale prose after an explicit edit.
    config['reward']=reward
    from .cem_objective import task_with_settings
    task=task_with_settings(task,settings)
    launch=resolve_launch_setup(settings.get('launch_setup'))
    if origin is not None:launch['initial_tracking_origin_m']=origin
    if target is not None:launch['target_position_m']=target
    launch=resolve_launch_setup(launch);settings['launch_setup']=launch
    task['initial_root_position_m']=(np.array(launch['initial_tracking_origin_m'])+model['recorded_data']['optitrack_to_attachment_offset_body_m']).tolist()
    task['initial_root_velocity_m_s']=[0.,0.,0.];task['target_position_m']=launch['target_position_m']
    task['episode_duration_s']=settings['horizon_s']
    validate_research_contract(model,task,config)
    settings.update(representation='ppo_force_30hz_v1',optimizer='mppi',model_path=str(model_path),
        success=task['success'],desired_strike_direction_world=task['desired_strike_direction_world'])
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'model.json',snapshot_assets(model,output))
    for n,value in (('task',task),('ppo',config),('mppi',settings)):atomic_json(output/f'{n}.json',value)
    np.savez_compressed(output/'seed_force.npz',forces=forces,times=times)
    atomic_json(output/'model_provenance.json',dict(source=str(model_path),source_sha256=sha256_file(model_path)))
    atomic_json(output/'seed_provenance.json',dict(source=str(seed_directory),
        rehearsal_sha256=sha256_file(seed_directory/'rehearsal.npz'),ppo_sha256=sha256_file(seed_directory/'ppo.json'),
        usage='Force initialization and PPO action/reward/termination contract; no actor training or spatial translation'))
    snapshot=output/'source_snapshot';snapshot.mkdir()
    for folder in ('planning','simulator','learning','deployment','experimental_data','tools'):
        for p in (root/folder).rglob('*'):
            if p.is_file() and p.suffix in ('.py','.cu','.cuh','.h') and '__pycache__' not in p.parts:
                dest=snapshot/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for name in ('run_ppo.py','requirements.txt'):shutil.copy2(root/name,snapshot/name)
    atomic_json(output/'source_manifest.json',{p.relative_to(snapshot).as_posix():sha256_file(p) for p in snapshot.rglob('*') if p.is_file()})
    atomic_json(output/'run.json',dict(status='PREPARED',display_name=settings['display_name']))
    return [sys.executable,'-u',str(snapshot/'tools/plan_mppi_force.py'),'--job',str(output)]


@torch.no_grad()
def complete_prediction(evaluation,origin,model,config,settings):
    env,batch,forces,cutoffs,score,frames=[evaluation[k] for k in ('env','batch','forces','cutoffs','score','frames')]
    if bool(score.failed[0]) or not hasattr(score,'reference_packets'):raise ValueError('PPO execution rejected this force sequence.')
    whip,frames,current,whip_pose=rehearsal_prefix(score,frames,env)
    if hasattr(score,'command_cutoffs'):cutoffs=score.command_cutoffs
    times,packets,phases,recovery=complete_packets(whip,origin)
    count=(len(packets)-1)*env.physics_steps_per_control;dt=env.physics_dt_s
    tensor=batch.truth.positions_m.new_tensor(packets)[None]
    valid,metrics=reference_feasibility(tensor,cutoffs.new_tensor([count]),dt,model['fullstate_execution']['feasibility'])
    if not bool(valid.all()):raise ValueError('Complete FullState reference exceeds the PPO command envelope.')
    offset=model['recorded_data']['optitrack_to_attachment_offset_body_m']
    hover=tensor[:,0].clone();hover[:,3:9]=0
    pose=env._research_tracker.predict(settled_initial(batch.truth.positions_m[:,0],batch.truth.velocities_m_s[:,0],offset),
        tensor,times,np.arange(count+1)*dt,offset,graph=bool(config['cuda_graph_physics']),hover_command=hover)
    prefix=len(frames)
    error=float((pose['position_origin_m'][:,:prefix]-whip_pose['position_origin_m']).abs().max())
    if error>1e-8:raise ValueError('PPO scoring/export prefix mismatch.')
    if not bool(pose['valid'].all()):raise ValueError('Complete drone prediction leaves its model domain.')
    for i in range(prefix-1,count):
        env._mppi_cancel_check()
        candidate=env._research_cable(current,pose['position_attachment_m'][:,i+1])
        if not bool(torch.isfinite(candidate.positions_m).all()&torch.isfinite(candidate.velocities_m_s).all()) or bool(
            (candidate.positions_m.abs()>env.numerical_position_limit_m).any()|(candidate.velocities_m_s.norm(dim=-1)>env.numerical_speed_limit_m_s).any()):
            raise ValueError('Complete cable prediction leaves its model domain.')
        current=candidate;frames.append(current.positions_m[0].cpu().numpy().copy())
    cable=np.asarray(frames);origins=pose['position_origin_m'][0].cpu().numpy()
    lo,hi=settings['minimum_height_m'],settings['maximum_height_m']
    if any(a.min()<lo or a.max()>hi for a in (packets[:,2],origins[:,2],cable[:,:,2])):
        raise ValueError('Complete command or predicted geometry exceeds the configured height limits.')
    force=forces[:int(cutoffs[0]),0].cpu().numpy()
    arrays=dict(command_time_s=times,commands=packets,command_phase=phases,
        prediction_time_s=np.arange(len(cable))*dt,cable_positions_m=cable,
        origin_positions_m=origins,origin_rotations=pose['rotation_tracking_to_world'][0].cpu().numpy(),
        force_time_s=np.arange(len(force))*dt,virtual_force_n=force,target_position_m=env.target[0].cpu().numpy())
    meta=dict(schema='mppi_force_fullstate_30hz_v1',planner='MPPI 30 Hz force sequence',optimizer='mppi',representation='ppo_force_30hz_v1',
        command_rate_hz=30,force_rate_hz=30,prediction_rate_hz=150,
        initial_tracking_origin_m=list(origin),target_position_m=arrays['target_position_m'].tolist(),
        frame='World XYZ in metres; unshifted OptiTrack tracked origin',preflight_hold_s=10,
        whip_end_s=(len(whip)-1)/30,total_duration_s=float(times[-1]),
        predicted_valid_hit=bool(score.episode_success[0]),minimum_tip_distance_m=float(score.episode_minimum_tip_distance[0]),
        predicted_hit_time_s=float(score.episode_hit_time_s[0]) if bool(score.episode_success[0]) else None,
        objective_score=float(score.episode_reward[0]),reward_components={k:float(v[0]) for k,v in score.episode_component_sums.items()},
        termination_mode=config['deployment'].get('termination','frozen_virtual_plan'),
        recovery=recovery,recovery_prediction_complete=True,recovery_empirically_validated=False,
        prediction_valid_through_s=float(times[-1]),training_export_prefix_max_difference_m=error,reference_feasible=True,
        maximum_command_height_m=float(packets[:,2].max()),maximum_predicted_drone_height_m=float(origins[:,2].max()),
        maximum_predicted_cable_height_m=float(cable[:,:,2].max()),
        virtual_force_csv='Virtual simulator input INCLUDING gravity, not a direct drone force command',
        execution='Take off; settle 10 seconds; execute complete fixed 30 Hz FullState CSV; land. Offline only.')
    return arrays,meta


def run_job(output):
    from .cem_execution import Cancelled
    output=Path(output).resolve();settings=read_json(output/'mppi.json');validate_settings(settings)
    if read_json(output/'run.json')['status']!='PREPARED':raise ValueError('Create a new job; existing runs are immutable.')
    with (output/'worker.lock').open('x') as f:f.write(str(os.getpid()))
    model,task,config=[read_json(output/f'{n}.json') for n in ('model','task','ppo')]
    for name,file in (('motion_residual','cable_residual.pt'),('fullstate_execution','drone_model.json')):model[name]['checkpoint']=str(output/'assets'/file)
    start=time.perf_counter();atomic_json(output/'run.json',dict(status='RUNNING',pid=os.getpid(),display_name=settings['display_name']))
    try:
        evaluator=ForceEvaluator(model,task,config,settings,settings['device'],lambda:(output/'STOP_REQUESTED').exists())
        with np.load(output/'seed_force.npz') as data:mean=evaluator.seed_actions(data['forces'],data['times'])
        history=[]
        def progress(row,best,bank):
            history.append(row);atomic_json(output/'history.json',history)
            np.savez_compressed(output/'candidates.npz',scores=[s for s,_ in bank],vectors=[v for _,v in bank])
            atomic_json(output/'progress.json',dict(row,label=f'MPPI force iteration {row["iteration"]}/{settings["iterations"]} · PPO return {row["best_score"]:.2f}',step=row['iteration'],total=settings['iterations']))
            print(row,flush=True)
        _,_,bank=optimize(mean,np.full_like(mean,settings['action_std']),evaluator,population=settings['population'],
            iterations=settings['iterations'],temperature=settings['temperature'],seed=settings['random_seed'],progress=progress,cancelled=evaluator.cancelled)
        evaluator.check_cancelled();rejected=[];selected=None
        for _,vector in bank:
            evaluator.check_cancelled();_,_,evaluation=evaluator.evaluate([vector],record=True)
            evaluation['env']._mppi_cancel_check=evaluator.check_cancelled
            try:
                arrays,meta=complete_prediction(evaluation,settings['launch_setup']['initial_tracking_origin_m'],model,config,settings)
            except ValueError as error:rejected.append(str(error));continue
            selected=(vector,arrays,meta);break
        atomic_json(output/'recovery_rejections.json',rejected)
        if selected is None:raise ValueError('No force candidate passed complete rehearsal checks. History saved; no CSV exported.')
        vector,arrays,meta=selected
        np.savez_compressed(output/'actions.npz',normalized_actions=np.clip(vector.reshape(-1,3),-1,1),time_s=np.arange(evaluator.steps)/30)
        np.savez_compressed(output/'rehearsal.npz',**arrays)
        with (output/'fullstate_30hz.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream);writer.writerow(FIELDS);writer.writerows(np.c_[arrays['command_time_s'],arrays['commands']])
        with (output/'virtual_force_30hz.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream);writer.writerow(['time_s','fx_total_n','fy_total_n','fz_total_n'])
            writer.writerows(np.c_[arrays['force_time_s'][::5],arrays['virtual_force_n'][::5]])
        meta.update(display_name=settings['display_name'],elapsed_s=time.perf_counter()-start,model_provenance=read_json(output/'model_provenance.json'))
        atomic_json(output/'rehearsal.json',meta)
        atomic_json(output/'run.json',dict(status='COMPLETED',display_name=settings['display_name'],elapsed_s=meta['elapsed_s']))
        return meta
    except Cancelled as error:
        atomic_json(output/'run.json',dict(status='STOPPED',display_name=settings['display_name'],message=str(error)))
    except Exception as error:
        atomic_json(output/'run.json',dict(status='FAILED',display_name=settings['display_name'],message=str(error)));raise
