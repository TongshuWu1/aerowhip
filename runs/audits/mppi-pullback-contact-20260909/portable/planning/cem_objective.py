"""Explicit, saved CEM reward configuration. Defaults reproduce the original score."""
from copy import deepcopy
import numpy as np

REWARD_DEFAULTS=dict(success_bonus=1000.,distance_weight=100.,speed_weight=10.,
    time_weight_per_s=25.,jerk_weight=.001,invalid_contact_penalty=100.,
    height_weight=20.,preferred_height_m=2.6,distance_cap_m=10.,jerk_cap=10000.)


def resolve_reward(value=None):
    value=value or {}
    if set(value)-set(REWARD_DEFAULTS):raise ValueError('Unknown CEM reward setting.')
    result=dict(REWARD_DEFAULTS,**value)
    if any(isinstance(v,bool) or not np.isfinite(v) or v<0 for v in result.values()):
        raise ValueError('CEM reward values must be finite and nonnegative.')
    if result['distance_cap_m']<=0 or result['jerk_cap']<=0:
        raise ValueError('Reward clipping caps must be positive.')
    return result


def task_with_settings(task,settings):
    """Apply explicit next-run task settings without editing the seed task."""
    task=deepcopy(task)
    success=settings.get('success',{})
    allowed={'tip_target_distance_m','minimum_directed_tip_speed_m_s',
             'maximum_tip_velocity_to_desired_direction_error_deg','first_contact_only',
             'tip_must_enter_before_other_markers'}
    if set(success)-allowed:raise ValueError('Unknown CEM hit criterion.')
    gate=dict(task['success'],**success)
    radius,speed,angle=[gate[k] for k in ('tip_target_distance_m','minimum_directed_tip_speed_m_s',
                                       'maximum_tip_velocity_to_desired_direction_error_deg')]
    if not np.isfinite([radius,speed,angle]).all() or radius<=0 or speed<0 or not 0<=angle<=180:
        raise ValueError('Hit radius must be positive, speed nonnegative and angle between 0 and 180 degrees.')
    for key in ('first_contact_only','tip_must_enter_before_other_markers'):
        if key in gate and not isinstance(gate[key],bool):raise ValueError('Contact options must be boolean.')
    direction=np.asarray(settings.get('desired_strike_direction_world',task.get('desired_strike_direction_world',[1,0,0])),float)
    if direction.shape!=(3,) or not np.isfinite(direction).all() or np.linalg.norm(direction)<1e-9:
        raise ValueError('Strike direction must be a finite nonzero XYZ vector.')
    task['success']=gate;task['desired_strike_direction_world']=(direction/np.linalg.norm(direction)).tolist()
    return task


def reward_components(result,durations,jerk,minimum_speed,settings=None):
    w=resolve_reward(settings)
    terminal=np.where(result['success'],result['hit_time'],durations)
    return dict(success=w['success_bonus']*result['success'],
        distance=-w['distance_weight']*np.clip(result['distance'],0,w['distance_cap_m']),
        speed=w['speed_weight']*np.clip(result['closest_speed']/max(minimum_speed,.01),0,1),
        time=-w['time_weight_per_s']*terminal,
        jerk=-w['jerk_weight']*np.minimum(jerk,w['jerk_cap']),
        invalid_contact=-w['invalid_contact_penalty']*result['invalid_contact'],
        height=-w['height_weight']*np.maximum(result['peak']-w['preferred_height_m'],0))
