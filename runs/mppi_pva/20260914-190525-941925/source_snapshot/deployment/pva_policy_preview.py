"""Inspect any PPO policy outcome without pretending it is a complete flight."""
from pathlib import Path
import numpy as np
import torch
from simulator.workflow import read_json
from experimental_data.io import atomic_json
from planning.pva_job import load_policy
from learning.pva_env import PVAEnvironment
from learning.pva_success import criterion,TIP_CONTACT


def generate_preview(job,output,*,device='cuda'):
    job=Path(job).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Use a new preview folder')
    cfg=read_json(job/'settings.json');model=read_json(job/'model.json')
    env=PVAEnvironment(model,cfg,root=job,device=device)
    checkpoint=job/'checkpoints/policy.pt';policy=load_policy(checkpoint,env,cfg)
    result=env.rollout(policy=policy.deterministic_action,trace=True)
    cutoff=int(result['cutoffs'][0]);frames=env.frames[:cutoff*env.stride]
    commands=result['packets'][0,:cutoff+1].cpu().numpy()
    def series(initial,key):
        return np.asarray([initial.cpu().numpy(),*[f[key][0].cpu().numpy() for f in frames]])
    times=np.asarray([0.,*[f['time_s'] for f in frames]])
    jerk=result['actions'][0,:cutoff].cpu().numpy()*np.asarray(cfg['action']['jerk_limit_m_s3'])
    provenance=read_json(job/'policy_snapshot.json')
    hit=bool(result['success'][0]);failed=bool(result['failed'][0])
    touched=bool(getattr(env,'tip_contact',env.contact)[0])
    tip_task=criterion(cfg['task'])==TIP_CONTACT
    outcome='valid hit' if hit else 'infeasible attempt' if failed else 'invalid first contact' if bool(env.contact[0]) and not tip_task else 'miss / time limit'
    output.mkdir(parents=True)
    atomic_json(output/'settings.json',cfg)
    from learning.ppo_trajectory_reward import freeze_reference
    freeze_reference(cfg,output,source_root=job)
    atomic_json(output/'task.json',dict(desired_strike_direction_world=cfg['task']['strike_direction'],
        target_position_m=cfg['launch']['target_m'],success=dict(tip_target_distance_m=cfg['task']['target_radius_m'])))
    np.savez_compressed(output/'rehearsal.npz',command_time_s=np.arange(len(commands))/30,commands=commands,
        command_phase=np.ones(len(commands),dtype=int),prediction_time_s=times,
        cable_positions_m=series(env.initial_state.positions_m[0],'cable'),
        cable_velocities_m_s=series(env.initial_state.velocities_m_s[0],'cable_velocity'),
        origin_positions_m=series(env.initial_pose.position[0],'origin'),
        origin_velocities_m_s=series(env.initial_pose.velocity[0],'origin_velocity'),
        origin_rotations=series(env.initial_pose.rotation[0],'rotation'),
        target_position_m=cfg['launch']['target_m'],jerk_time_s=np.arange(len(jerk))/30,jerk_m_s3=jerk)
    metadata=dict(schema='pva_policy_preview_v1',preview_only=True,planner='PPO policy preview',job=str(job),
        policy_snapshot=provenance,checkpoint=str(checkpoint),checkpoint_sha256=provenance['checkpoint_sha256'],
        initial_tracking_origin_m=cfg['launch']['origin_m'],target_position_m=cfg['launch']['target_m'],
        whip_end_s=cutoff/30,total_duration_s=float(times[-1]),predicted_valid_hit=hit,predicted_failed=failed,
        outcome=outcome,tip_touched_target=touched,minimum_tip_distance_m=float(result['minimum_tip_distance_m'][0]),
        predicted_hit_time_s=float(result['duration_s'][0]) if hit else None,
        termination_time_s=float(result['duration_s'][0]),prediction_valid_through_s=float(times[-1]),
        recovery_prediction_complete=False,reference_feasible=False,
        success_criterion=criterion(cfg['task']),
        pullback=dict(required=cfg['task'].get('require_pullback',False) and not tip_task),
        wave=dict(required=cfg['task'].get('require_wave',False) and not tip_task,completed_stages=int(env.wave_stage[0])),
        evidence='Frozen deterministic policy preview; simulation only, no complete recovery or executable flight CSV')
    if 'objective_terms' in result:
        metadata.update(trajectory_score=float(result['reward'][0]),
            objective_terms={key:float(value[0]) for key,value in result['objective_terms'].items()})
    atomic_json(output/'rehearsal.json',metadata)
    return metadata
