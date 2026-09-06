"""Local differentiable force correction, independent of PPO/SAC weight updates."""
from copy import deepcopy
from pathlib import Path
import math
import time
import numpy as np
import torch
from simulator.cable import DderState
from simulator.workflow import read_json, atomic_json
from experimental_data.flight_adaptation import load_trial, coupled_rollout


def corrected_forces(prior, knots, limit=.15):
    correction=torch.nn.functional.interpolate(knots.T[None],size=len(prior),mode='linear',align_corners=True)[0].T
    return prior+limit*torch.tanh(correction)


def sequence_loss(q,v,task,force,prior):
    target=q.new_tensor(task['target_position_m'])
    direction=q.new_tensor(task['desired_strike_direction_world']); direction=direction/direction.norm()
    tip=q[:,-1,-1]; speed=v[:,-1,-1]
    directed=(speed*direction).sum(-1)
    lateral=speed-directed[:,None]*direction
    # Smooth pre-contact surrogate only. Existing discrete task scoring is
    # applied independently afterwards and is never differentiated.
    distance=(tip-target).square().sum(-1).mean()/.05**2
    speed_loss=torch.nn.functional.softplus(task['success']['minimum_directed_tip_speed_m_s']-directed).square().mean()
    travel=(q[:,:,0]-q[:,:1,0]).square().sum(-1).mean()
    return distance+speed_loss+.05*lateral.square().mean()+10*travel+2*(force-prior).square().mean()


@torch.no_grad()
def rescore(payload, task, ppo, initial, forces):
    from learning.point_force_env import PointForceWhipEnvironment
    from learning.deployment_rollout import DeploymentBatch,execute_batch
    task=deepcopy(task);task['initial_root_position_m']=initial.positions_m[0,0].tolist()
    config=deepcopy(ppo);config['cuda_graph_physics']=False
    settings=deepcopy(config['deployment'])
    # This report is explicitly nominal. Independent uncertainty validation and
    # the real actuator response remain necessary before flight release.
    settings.update(launch_position_drift_m=0.,launch_velocity_drift_m_s=0.)
    env=PointForceWhipEnvironment(payload,task,config,batch_size=1,device=torch.device('cpu'))
    env.reset(initial)
    ones=torch.ones(1,dtype=torch.float64)
    batch=DeploymentBatch(initial,initial,ones,ones,torch.ones(1,3,dtype=torch.float64),
                          torch.zeros(1,1,dtype=torch.float64),torch.ones(1,dtype=torch.bool))
    score=execute_batch(env,batch,forces[:,None],torch.tensor([len(forces)]),settings)
    return dict(valid_hit=bool(score.episode_success[0]),recovered=bool(score.deployment['recovered'][0]),
                hit_and_recovery=bool(score.deployment['joint_success'][0]),
                task_return=float(score.episode_reward[0]),numerical_failure=bool(score.failed[0]),
                maximum_drone_displacement_m=float(score.deployment['maximum_execution_drone_displacement_m'][0]),
                validation='nominal exact first-contact and PID recovery only; not robust/hardware validation')


def refine(directory,payload,ppo,output,*,updates=12,progress=None):
    started=time.perf_counter();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    meta,data,take=load_trial(directory);task=read_json(Path(directory)/'task.json')
    dt=float(payload['simulation']['dt_s']);control_dt=float(task['control_dt_s'])
    ratio=round(control_dt/dt)
    if ratio<1 or not math.isclose(ratio*dt,control_dt,abs_tol=1e-8):raise ValueError('Command period must be integer physics steps')
    cutoff=meta['planned_cutoff_s']-meta['strike_start_s'];steps=round(cutoff/dt)
    if steps<1 or not math.isclose(steps*dt,cutoff,abs_tol=1e-6):raise ValueError('Cutoff must align with physics timestep')
    command_times=np.arange(math.ceil(steps/ratio))*control_dt
    indices=np.searchsorted(data['command_time_s'],command_times+1e-9,side='right')-1
    if (indices<0).any():raise ValueError('Recorded commands do not cover the maneuver')
    prior=torch.tensor(data['command_values_n'][indices],dtype=torch.float64)
    initial=DderState(take.initial_positions_m,take.initial_velocities_m_s)
    knots=torch.nn.Parameter(torch.zeros(min(4,len(prior)),3,dtype=torch.float64))
    optimizer=torch.optim.Adam([knots],lr=.1)
    best=prior.clone();best_loss=float('inf');history=[]
    # Limits are simulation development settings, not asserted aircraft limits.
    limit=float(ppo['action']['maximum_force_norm_n'])
    for update in range(updates+1):
        optimizer.zero_grad();commands=corrected_forces(prior,knots)
        force=commands.repeat_interleave(ratio,dim=0)[:steps]
        q,v=coupled_rollout(payload,initial,force[:,None],gradients=True)
        loss=sequence_loss(q,v,task,commands,prior)
        feasible=bool(torch.isfinite(q).all() and torch.isfinite(v).all() and torch.isfinite(loss)
                      and (commands.norm(dim=-1)<=limit).all() and (commands[:,2]>=0).all())
        value=float(loss.detach())
        if feasible and value<best_loss:best_loss=value;best=commands.detach().clone()
        history.append(dict(update=update,surrogate_loss=value,force_feasible=feasible))
        atomic_json(output/'history.json',history)
        if progress:progress(f'Refining fixed force sequence · {update} / {updates}')
        if update<updates:
            if not torch.isfinite(loss):raise ValueError('Nonfinite differentiable rollout')
            loss.backward()
            if not torch.isfinite(knots.grad).all():raise ValueError('Nonfinite sequence derivative')
            torch.nn.utils.clip_grad_norm_([knots],10.);optimizer.step()
    if not np.isfinite(best_loss):raise ValueError('No feasible sequence under simulation force limits')
    if progress:progress('Checking strict first contact and full PID recovery')
    base_score=rescore(payload,task,ppo,initial,prior.repeat_interleave(ratio,0)[:steps])
    candidate_score=rescore(payload,task,ppo,initial,best.repeat_interleave(ratio,0)[:steps])
    # Never silently replace the original on a surrogate-only improvement.
    improved=(candidate_score['hit_and_recovery'] and not candidate_score['numerical_failure']
              and candidate_score['task_return']>=base_score['task_return'])
    np.savetxt(output/'candidate_forces.csv',np.column_stack((command_times,best.numpy())),delimiter=',',
               header='time_s,fx_n,fy_n,fz_n',comments='')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure=Figure(figsize=(9,3),layout='constrained');FigureCanvasAgg(figure)
    for i,axis in enumerate(figure.subplots(1,3)):
        times=np.r_[command_times,cutoff]
        axis.step(times,np.r_[prior[:,i].numpy(),prior[-1,i].item()],where='post',label='Recorded sequence')
        axis.step(times,np.r_[best[:,i].numpy(),best[-1,i].item()],where='post',label='Candidate')
        axis.set(xlabel='Time after launch [s]',ylabel=f'Force {"XYZ"[i]} [N]')
        axis.spines[['top','right']].set_visible(False)
        if i==0:axis.legend(fontsize=8)
    for ext in ('png','pdf'):figure.savefig(output/f'force_comparison.{ext}',dpi=200)
    np.savez_compressed(output/'initial_state.npz',positions_m=initial.positions_m.numpy(),velocities_m_s=initial.velocities_m_s.numpy())
    atomic_json(output/'model.json',payload);atomic_json(output/'task.json',task)
    summary=dict(status='SIMULATION_CANDIDATE' if improved else 'REJECTED_BY_NOMINAL_TASK_CHECK',
        flight_ready=False,policy_weights_changed=False,initial_state_source=str(directory),
        command_rate_hz=1/control_dt,cutoff_s=cutoff,correction_knots=len(knots),component_correction_limit_n=.15,
        baseline=base_score,candidate=candidate_score,elapsed_s=time.perf_counter()-started,
        limitations=['Candidate uses the recorded launch state; regenerate for the actual next launch state',
                    'Real Lee-controller force limits and response are not verified',
                    'Independent uncertainty acceptance has not been run',
                    'Cutoff fixed before execution; no actual-hit feedback; actor weights unchanged'])
    from experimental_data.io import sha256_file
    summary['artifact_sha256']={name:sha256_file(output/name) for name in
        ('candidate_forces.csv','initial_state.npz','model.json','task.json')}
    atomic_json(output/'result.json',summary)
    return summary
