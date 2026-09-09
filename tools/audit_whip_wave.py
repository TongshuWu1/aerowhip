"""Inspect saved cable shapes; optionally replay a new plan into a NEW audit.

Never overwrite a historical forecast. The independent NumPy stage evaluator
uses saved geometry/velocities and labels retrospective application explicitly.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation,PillowWriter
from learning.whip_wave import DEFAULTS
from learning.pva_env import PVAEnvironment
from experimental_data.io import atomic_json
from simulator.workflow import read_json


def diagnostics(arrays,metadata,settings):
    cfg={**DEFAULTS,**settings['task']}
    stop=metadata.get('predicted_hit_time_s') or metadata['whip_end_s']
    mask=arrays['prediction_time_s']<=stop+1e-12
    t=arrays['prediction_time_s'][mask];q=arrays['cable_positions_m'][mask]
    v=arrays['cable_velocities_m_s'][mask];origin=arrays['origin_positions_m'][mask]
    axis=np.asarray(cfg['strike_direction'],float);axis/=np.linalg.norm(axis)
    edge=np.diff(q,axis=1);tangent=edge/np.maximum(np.linalg.norm(edge,axis=-1,keepdims=True),1e-12)
    angles=np.arctan2(np.linalg.norm(np.cross(tangent[:,:-1],tangent[:,1:]),axis=-1),
        np.clip(np.sum(tangent[:,:-1]*tangent[:,1:],axis=-1),-1,1))
    # Initial hanging geometry gives the material rest coordinates in saved ghosts.
    lengths=np.linalg.norm(edge[0],axis=-1);material=np.cumsum(lengths)[:-1]/sum(lengths)
    peak=angles.max(1);location=material[angles.argmax(1)]
    displacement=(origin-np.asarray(settings['launch']['origin_m']))@axis
    drone_speed=arrays['origin_velocities_m_s'][mask]@axis
    pull=np.maximum.accumulate((displacement>=cfg.get('minimum_pull_distance_m',.25))&(drone_speed>=cfg.get('minimum_pull_speed_m_s',1.)))
    stage=0;dwell=0.;times=[];thresholds=[cfg['wave_proximal_angle_rad'],cfg['wave_middle_angle_rad'],cfg['wave_distal_angle_rad']]
    for i in range(1,len(t)):
        region=0 if location[i]<cfg['wave_proximal_end'] else 2 if location[i]>=cfg['wave_distal_start'] else 1
        held=pull[i] and stage<3 and region==stage and peak[i]>=thresholds[min(stage,2)]
        dwell=dwell+t[i]-t[i-1] if held else 0.
        if held and dwell>=cfg['wave_dwell_s']-1e-12:
            stage+=1;times.append(float(t[i]));dwell=0.
    report=dict(evidence='Saved model geometry; kinematic travelling-bend proxy, not energy-transfer or flight proof',
        task_required_wave=settings['task'].get('require_wave',False),retrospective_diagnostic=not settings['task'].get('require_wave',False),
        stages_before_contact=stage,stage_completion_times_s=times,maximum_local_turn_angle_rad=float(peak.max()),
        peak_tip_speed_m_s=float(np.linalg.norm(v[:,-1],axis=-1).max()),
        peak_forward_tip_speed_m_s=float((v[:,-1]@axis).max()),
        criteria={k:cfg[k] for k in DEFAULTS},predicted_valid_hit=metadata.get('predicted_valid_hit',False))
    return report,(t,q,v,angles,location,material,drone_speed,axis)


def render(output,arrays,metadata,settings,*,animate=True):
    report,(t,q,v,angles,location,material,drone_speed,axis)=diagnostics(arrays,metadata,settings)
    figure=Figure(figsize=(12,7),layout='constrained');grid=figure.add_gridspec(2,2,width_ratios=[1,1.7])
    axes=[figure.add_subplot(grid[:,0]),figure.add_subplot(grid[0,1]),figure.add_subplot(grid[1,1])]
    selected=np.linspace(max(0,len(t)//3),len(t)-1,7).astype(int)
    import matplotlib
    colors=matplotlib.colormaps['viridis'](np.linspace(0,1,len(selected)))
    for i,c in zip(selected,colors):axes[0].plot(q[i]@axis,q[i,:,2],'-o',ms=3,color=c,label=f'{t[i]:.2f} s')
    target=np.asarray(settings['launch']['target_m']);axes[0].plot(target@axis,target[2],'rx',ms=10)
    axes[0].set(xlabel='Position along strike axis [m]',ylabel='Height [m]',title='Cable shape progression — frozen-model simulation')
    axes[0].set_aspect('equal',adjustable='box');axes[0].legend(ncol=2,fontsize=8)
    mesh=axes[1].pcolormesh(t,material,angles.T,shading='nearest',cmap='magma')
    axes[1].plot(t,np.where(angles.max(1)>=.05,location,np.nan),color='cyan',lw=1,label='Dominant bend (angle ≥ 0.05 rad)')
    axes[1].set(xlabel='Time [s]',ylabel='Material position: attachment → tip')
    figure.colorbar(mesh,ax=axes[1],label='Local turning angle [rad]')
    axes[2].plot(t,drone_speed,label='Drone');axes[2].plot(t,v[:,-1]@axis,label='Tip')
    axes[2].axhline(0,color='gray',lw=.6);axes[2].legend();axes[2].set(xlabel='Time [s]',ylabel='Forward speed [m/s]')
    for time in report['stage_completion_times_s']:axes[1].axvline(time,color='cyan',ls=':',lw=1)
    figure.savefig(output/'wave.png',dpi=150)
    if animate:
        fig=Figure(figsize=(7,5),layout='constrained');ax=fig.subplots()
        x=q@axis;line,=ax.plot([],[],'-o',color='#ea580c',ms=4);drone,=ax.plot([],[],'s',color='#2563eb',ms=9)
        ax.plot(target@axis,target[2],'rx',ms=12,label='Target')
        ax.set(xlim=(min(x.min(),target@axis)-.15,max(x.max(),target@axis)+.15),
            ylim=(q[:,:,2].min()-.15,q[:,:,2].max()+.15),xlabel='Strike axis [m]',ylabel='Height [m]')
        ax.set_aspect('equal');ax.grid(alpha=.2)
        frames=np.arange(0,len(t),3)
        def draw(i):
            line.set_data(x[i],q[i,:,2]);drone.set_data([x[i,0]],[q[i,0,2]])
            ax.set_title(f'Model simulation | {t[i]:.2f} s | quarter speed')
            return line,drone
        animation=FuncAnimation(fig,draw,frames=frames,interval=80,blit=False)
        animation.save(output/'wave.gif',writer=PillowWriter(fps=12.5))
    atomic_json(output/'wave_metrics.json',report)
    return report


def replay(job,*,preview=False):
    cfg=read_json(job/'settings.json');env=PVAEnvironment(read_json(job/'model.json'),cfg,root=job)
    with np.load(job/'plan.npz') as z:
        sequence=z['normalized_jerk'].copy();committed=len(sequence)
        if preview:sequence=np.concatenate((sequence,np.tanh(z['proposal_mean'])))[:env.steps]
    actions=env.tensor(sequence)[None]
    env.rollout(actions=actions,max_steps=actions.shape[1],trace=True)
    frames=env.frames
    def stack(key,initial):return np.concatenate((initial[None],torch.stack([f[key][0] for f in frames]).cpu().numpy()))
    arrays=dict(prediction_time_s=np.r_[0.,[f['time_s'] for f in frames]],
        cable_positions_m=stack('cable',env.initial_state.positions_m[0].cpu().numpy()),
        cable_velocities_m_s=stack('cable_velocity',env.initial_state.velocities_m_s[0].cpu().numpy()),
        origin_positions_m=stack('origin',env.initial_pose.position[0].cpu().numpy()),
        origin_velocities_m_s=stack('origin_velocity',env.initial_pose.velocity[0].cpu().numpy()))
    metadata=dict(whip_end_s=env.index/30,predicted_valid_hit=bool(env.success[0]),
        predicted_hit_time_s=float(env.termination_time[0]) if env.success[0] else None,
        provisional_preview=preview,committed_steps_at_snapshot=committed,
        wave_stages=int(env.wave_stage[0]),wave_completion_time_s=float(env.wave_completion_time[0]) if torch.isfinite(env.wave_completion_time[0]) else None)
    arrays['normalized_jerk']=sequence
    return arrays,metadata,cfg


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--rehearsal',type=Path);group.add_argument('--job',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--no-animation',action='store_true')
    parser.add_argument('--preview',action='store_true',help='Replay committed prefix plus current proposal into an explicitly provisional audit')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    if args.job:
        arrays,metadata,settings=replay(args.job,preview=args.preview)
        np.savez_compressed(args.output/'diagnostic_replay.npz',**arrays)
        atomic_json(args.output/'replay_metadata.json',metadata)
    else:
        with np.load(args.rehearsal/'rehearsal.npz') as z:arrays={k:z[k].copy() for k in z.files}
        metadata=read_json(args.rehearsal/'rehearsal.json');settings=read_json(args.rehearsal/'settings.json')
    print(render(args.output,arrays,metadata,settings,animate=not args.no_animation))
