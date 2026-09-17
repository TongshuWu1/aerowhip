"""A resolved travelling-bend proxy, not an energy-transfer measurement.

Track the dominant local turning angle through three material-coordinate bands.
Persistence and ordered bands reject rigid swings, stationary bends and one-frame
peak jumps. Thresholds are engineering task choices for the saved discretization.
"""
import torch

FIELDS=('wave_stage','wave_dwell','wave_credit','wave_completion_time')
DEFAULTS=dict(wave_proximal_end=.45,wave_distal_start=.75,
              wave_proximal_angle_rad=.15,wave_middle_angle_rad=.25,
              wave_distal_angle_rad=.5,wave_dwell_s=.04)


def bend_features(q,material):
    edge=q[:,1:]-q[:,:-1]
    tangent=edge/edge.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    cosine=(tangent[:,1:]*tangent[:,:-1]).sum(-1).clamp(-1,1)
    sine=torch.linalg.cross(tangent[:,:-1],tangent[:,1:]).norm(dim=-1)
    angles=torch.atan2(sine,cosine)
    peak,index=angles.max(-1)
    return angles,peak,material[index]


def advance(stage,dwell,credit,peak,location,eligible,dt,task):
    """Advance at most one stage per tick; each stage needs consecutive dwell."""
    proximal=location<task['wave_proximal_end']
    distal=location>=task['wave_distal_start']
    region=torch.where(proximal,0,torch.where(distal,2,1))
    threshold=torch.where(stage==0,task['wave_proximal_angle_rad'],
        torch.where(stage==1,task['wave_middle_angle_rad'],task['wave_distal_angle_rad']))
    amplitude=(peak/threshold).clamp(0,1)
    valid=eligible&(stage<3)&(region==stage)
    held=valid&(amplitude>=1)
    next_dwell=torch.where(held,dwell+dt,torch.zeros_like(dwell))
    completed=held&(next_dwell>=task['wave_dwell_s']-1e-12)
    # Bounded progress gives partial credit for forming a bend in the next band.
    partial=valid*(.5*amplitude+.5*(next_dwell/task['wave_dwell_s']).clamp(0,1))
    current=(stage+partial)/3
    next_credit=torch.where(eligible,torch.maximum(credit,current),credit)
    next_stage=stage+completed.long()
    next_dwell=torch.where(completed,torch.zeros_like(next_dwell),next_dwell)
    return next_stage,next_dwell,next_credit,next_credit-credit


def update(env,running,right):
    _,peak,location=bend_features(env.state.positions_m,env.wave_material)
    stage,dwell,credit,bonus=advance(env.wave_stage,env.wave_dwell,env.wave_credit,
        peak,location,running&env.pull_ready,env.dt,env.settings['task'])
    completed=(env.wave_stage<3)&(stage==3)
    env.wave_completion_time=torch.where(completed,right,env.wave_completion_time)
    env.wave_stage,env.wave_dwell,env.wave_credit=stage,dwell,credit
    return env.settings['reward'].get('wave_progress',0.)*bonus
