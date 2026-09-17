"""Desired tracked-origin P/V/A from bounded piecewise-constant XYZ jerk.

Acceleration is kinematic m/s² (gravity is not added). The continuous reference
is derivative consistent; the aircraft receives its sampled 30 Hz ZOH packets.
"""
import math
import torch

SCHEMA='bounded_jerk_pva_30hz_v1'
RATE_HZ=30.


def integrate_jerk(packet,jerk,dt=1/RATE_HZ):
    if packet.shape[-1]!=11 or jerk.shape!=packet.shape[:-1]+(3,):raise ValueError('Expected PVA/yaw packet and XYZ jerk')
    if not math.isfinite(dt) or dt<=0:raise ValueError('Positive integration interval required')
    p,v,a=packet[...,:3],packet[...,3:6],packet[...,6:9]
    return torch.cat((p+dt*v+.5*dt**2*a+dt**3/6*jerk,
        v+dt*a+.5*dt**2*jerk,a+dt*jerk,packet[...,9:]),-1)


def jerk_packets(initial,jerks,dt=1/RATE_HZ):
    """[batch, intervals, 3] -> packets including both interval endpoints."""
    values=[initial]
    for j in jerks.unbind(1):values.append(integrate_jerk(values[-1],j,dt))
    return torch.stack(values,1)


def sphere_entry(previous,current,target,radius):
    """First intersection fraction per vertex, infinity for no sphere entry."""
    d=current-previous;r=previous-target[:,None]
    a=d.square().sum(-1);b=(r*d).sum(-1);c=r.square().sum(-1)-radius**2
    discriminant=b.square()-a*c
    entry=(-b-torch.sqrt(discriminant.clamp_min(0)))/a.clamp_min(1e-20)
    return torch.where(c<=0,torch.zeros_like(entry),torch.where(
        (a>1e-20)&(discriminant>=0)&(entry>=0)&(entry<=1),entry,torch.full_like(entry,torch.inf)))
