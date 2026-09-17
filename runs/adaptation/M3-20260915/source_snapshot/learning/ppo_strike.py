"""Continuous PPO guidance toward simultaneous strike conditions.

This score is only a learning signal. The environment's exact first-contact,
speed, angle, reversal, wave-causality and feasibility tests still decide hits.
"""
import math
import torch


def joint_quality(distance,tip_velocity,drone_velocity,backward_distance,
                  pull_ready,wave_credit,direction,task,proximity_scale):
    forward=(tip_velocity*direction).sum(-1)
    speed=(forward/task['minimum_directed_speed_m_s']).clamp(0,1)
    alignment=(forward/tip_velocity.norm(dim=-1).clamp_min(1e-12)/
               math.cos(math.radians(task['maximum_angle_deg']))).clamp(0,1).square()
    reverse_speed=(-(drone_velocity*direction).sum(-1)/task['minimum_backward_speed_m_s']).clamp(0,1)
    reverse_distance=(backward_distance/task['minimum_backward_distance_m']).clamp(0,1)
    # Partial guidance exists before all gates pass; the maximum requires the
    # actual backward speed/travel, a completed bend, and a directed near-hit.
    release=.2+.8*torch.minimum(reverse_speed,reverse_distance)
    wave=.2+.8*wave_credit.clamp(0,1)
    near=.4*torch.exp(-(distance/proximity_scale).square())
    near+=.6*torch.exp(-(distance/(2*task['target_radius_m'])).square())
    return near*speed*alignment*release*wave*pull_ready
