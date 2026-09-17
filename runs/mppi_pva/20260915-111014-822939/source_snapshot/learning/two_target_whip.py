"""Ordered, distinct virtual tip targets; contact never changes physical state.

The same cable tip must visit T1 then T2 in a continuous rollout. No reset or
extra launch is inserted at T1. Collision forces are not part of this model.
"""
import numpy as np
import torch
from simulator.pva_commands import sphere_entry
from learning.pva_success import TWO_TARGET,impact_bonus

FIELDS=('target_hits','target_hit_times','target_hit_velocities','target_minimum_distances')


def enabled(cfg):return cfg['task'].get('success_criterion')==TWO_TARGET


def validate(cfg):
    if not enabled(cfg):
        if cfg['task'].get('target_sequence_m') is not None:raise ValueError('Target sequence requires the two-target criterion')
        return
    targets=np.asarray(cfg['task'].get('target_sequence_m'),dtype=float)
    if targets.shape!=(2,3) or not np.isfinite(targets).all():raise ValueError('Two finite XYZ targets are required')
    if cfg['method']!='mppi' or cfg['mppi'].get('mode')!='open_loop':raise ValueError('Two-target whip requires offline MPPI')
    if cfg['mppi'].get('parameterization')!='control_points' or cfg.get('trajectory_objective',{}).get('schema')!='preferred_fold_v1':
        raise ValueError('Two-target whip requires the complete-trajectory preferred-fold objective')
    if not np.array_equal(targets[0],cfg['launch']['target_m']):raise ValueError('First target and launch target must agree')
    if np.linalg.norm(targets[1]-targets[0])<=2*cfg['task']['target_radius_m']:
        raise ValueError('The two target spheres must be distinct and non-overlapping')
    if cfg['launch'].get('target_radius_m',0):raise ValueError('Freeze the two target centers for this trial')
    reward=cfg['task'].get('two_target_reward','partial_v1')
    if reward not in ('partial_v1','completion_v2'):raise ValueError('Unknown two-target reward version')
    if reward=='completion_v2':
        w=cfg['trajectory_objective']
        if not np.isfinite(w.get('sequence_progress',np.nan)) or w['sequence_progress']<=0:
            raise ValueError('Positive finite sequence progress weight required')
        if not 0<=w.get('sequence_style_fraction',-1)<=1:raise ValueError('Sequence style fraction must be in [0,1]')


def initialize(env):
    env.extra_tick_fields=FIELDS if enabled(env.settings) else ()
    if not env.extra_tick_fields:return
    validate(env.settings)
    env.target_centers=env.tensor(env.settings['task']['target_sequence_m'])
    env.target_hits=torch.zeros((env.batch_size,2),device=env.device,dtype=torch.bool)
    env.target_hit_times=env.total.new_full((env.batch_size,2),torch.inf)
    env.target_hit_velocities=env.total.new_zeros((env.batch_size,2,3))
    env.target_minimum_distances=(env.state.positions_m[:,-1,None]-env.target_centers).norm(dim=-1)


def crossings(previous,current,previous_velocity,velocity,targets,radius,hits,hit_times,hit_velocities,minima,running,left,dt):
    """Vectorized swept tip entries. Both entries in one tick retain their order."""
    a=previous[:,None];b=current[:,None];delta=current-previous
    entries=[];distances=[];fractions=[]
    for target in targets:
        center=target[None].expand(len(a),-1)
        entries.append(sphere_entry(a,b,center,radius)[:,0])
        fraction=((center-previous)*delta).sum(-1)/delta.square().sum(-1).clamp_min(1e-20)
        fractions.append(fraction)
        distances.append((previous+fraction.clamp(0,1)[:,None]*delta-center).norm(dim=-1))
    entry=torch.stack(entries,-1);distance=torch.stack(distances,-1)
    new_first=running&~hits[:,0]&torch.isfinite(entry[:,0])
    # A visit to T2 before T1 earns no credit, including reversed order in a tick.
    first_available=hits[:,0]|(new_first&(entry[:,1]>entry[:,0]))
    new_second=running&~hits[:,1]&first_available&torch.isfinite(entry[:,1])
    fresh=torch.stack((new_first,new_second),-1)
    times=torch.where(fresh,left[:,None]+entry*dt,hit_times)
    vv=previous_velocity[:,None]+entry.clamp(0,1)[:,:,None]*(velocity-previous_velocity)[:,None]
    velocities=torch.where(fresh[:,:,None],vv,hit_velocities)
    # Second-target shaping begins only after T1; a reversed visit cannot satisfy
    # either its hit state or the minimum-distance guidance account.
    eligible=torch.stack((running,running&(hits[:,0]|new_first)),-1)
    start=torch.where(hits[:,0],torch.zeros_like(entry[:,0]),entry[:,0].clamp(0,1))
    second_fraction=torch.maximum(fractions[1].clamp(0,1),start)
    distance[:,1]=(previous+second_fraction[:,None]*delta-targets[1]).norm(dim=-1)
    nearest=torch.where(eligible,torch.minimum(minima,distance),minima)
    return hits|fresh,times,velocities,nearest,new_second,entry[:,1]


def advance(env,previous,q,v,running,left):
    result=crossings(previous.positions_m[:,-1],q[:,-1],previous.velocities_m_s[:,-1],v[:,-1],
        env.target_centers,env.settings['task']['target_radius_m'],env.target_hits,env.target_hit_times,
        env.target_hit_velocities,env.target_minimum_distances,running,left,env.dt)
    env.target_hits,env.target_hit_times,env.target_hit_velocities,env.target_minimum_distances,finished,fraction=result
    return finished,fraction


def score_terms(env,weights):
    proximity=torch.exp(-(env.target_minimum_distances[:,1]/weights['proximity_scale_m']).square())
    terms=dict(second_target=weights['contact']*proximity-weights['miss']*(1-proximity),
        first_target_credit=weights['contact']*env.target_hits[:,0].double())
    # Equal contact-speed preference per target, latched once at each true entry.
    terms['impact']=sum(impact_bonus(env.settings['reward'],env.target_hits[:,k],env.failed,
        env.target_hit_velocities[:,k],env.direction) for k in range(2))/2
    return terms


def completion_terms(env,weights,style):
    """Bounded ordered progress guides misses; speed is credited after both hits."""
    if env.settings['task'].get('two_target_reward')!='completion_v2':return style
    proximity=torch.exp(-(env.target_minimum_distances/weights['proximity_scale_m']).square())
    progress=weights['sequence_progress']*(proximity[:,0]+env.target_hits[:,0]*proximity[:,1])
    complete=env.target_hits.all(-1)&~env.failed
    quality={k:v*weights['sequence_style_fraction'] for k,v in style.items()
             if k not in ('impact','second_target','first_target_credit')}
    quality.update(sequence_progress=progress,impact=torch.where(complete,style['impact'],torch.zeros_like(progress)))
    return quality


def details(env,row):
    times=env.target_hit_times[row].detach().cpu().tolist()
    return dict(target_hits=(env.target_hits[row]&~env.failed[row]).cpu().tolist(),
        target_hit_times_s=[t if np.isfinite(t) else None for t in times],
        target_minimum_distances_m=env.target_minimum_distances[row].cpu().tolist(),
        target_directed_speeds_m_s=(env.target_hit_velocities[row]@env.direction).cpu().tolist())
