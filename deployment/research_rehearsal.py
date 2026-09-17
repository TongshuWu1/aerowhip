"""Shared 30 Hz command fields and analytic recovery helpers."""
import math
import numpy as np
import torch
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


def complete_packets(whip,hover,settings=None,*,height_bounds=None):
    whip=np.asarray(whip,dtype=float)
    if whip.ndim!=2 or whip.shape[1]!=11 or len(whip)<2 or not np.isfinite(whip).all():
        raise ValueError('A finite native 30 Hz whip reference is required.')
    sample,recovery=plan_curved_recovery(whip[-1,:3],whip[-1,3:6],whip[-1,6:9],hover,settings,height_bounds=height_bounds)
    tail_t=np.arange(1,math.ceil(recovery['total_duration_s']*30)+1)/30
    p,v,a=sample(np.minimum(tail_t,recovery['total_duration_s']))
    tail=np.c_[p,v,a,np.zeros((len(p),2))]
    packets=np.concatenate((whip,tail));times=np.arange(len(packets))/30
    end=(len(whip)-1)/30
    phases=np.where(times<=end+1e-10,1,np.where(times<=end+recovery['brake_end_s'],2,
        np.where(times<=end+recovery['return_end_s'],3,4)))
    return times,packets,phases,recovery
