"""One 30 Hz PVA construction shared by PPO, rehearsal and CSV export."""
import math
import torch


def packet_cutoffs(forces,cutoffs,steps_per_packet,maximum_steps):
    end=torch.where(cutoffs>0,((cutoffs+steps_per_packet-1)//steps_per_packet)*steps_per_packet,cutoffs).clamp_max(maximum_steps)
    if not bool(end.max()):return forces,end
    i=torch.arange(int(end.max()),device=forces.device)[:,None]
    i=torch.minimum(i,(cutoffs-1).clamp_min(0)[None]).clamp_max(len(forces)-1)
    return forces[i,torch.arange(len(cutoffs),device=forces.device)[None]],end


def reference_packets(positions,velocities,dt,cutoffs,offset,rate_hz=30.):
    """Integrate linear velocity segments, avoiding inconsistent Hermite P/V.

    The virtual attachment displacement defines desired tracked-origin
    displacement using the initial offset. Future attachment rotation belongs
    to the execution model, not a measured future pose in the reference.
    """
    stride=round(1/(rate_hz*dt))
    if stride<1 or not math.isclose(stride*dt,1/rate_hz,abs_tol=1e-10):raise ValueError('Packet/physics clocks must align')
    if torch.any(cutoffs%stride):raise ValueError('Whip cutoffs must land on packet boundaries')
    increments=.5*dt*(velocities[:,1:]+velocities[:,:-1])
    p=torch.cat((positions[:,:1],positions[:,:1]+increments.cumsum(1)),1)
    a=(velocities[:,1:]-velocities[:,:-1])/dt
    a=torch.cat((a,a[:,-1:]),1)
    # A0 -> O0 is a constant translation in the virtual reference generator.
    origin=p-p.new_tensor(offset)
    indices=torch.arange(0,positions.shape[1],stride,device=p.device)
    indices=torch.minimum(indices[None],cutoffs[:,None])
    batch=torch.arange(len(p),device=p.device)[:,None]
    packets=torch.cat((origin[batch,indices],velocities[batch,indices],a[batch,indices],p.new_zeros(len(p),indices.shape[1],2)),-1)
    within=torch.arange(p.shape[1],device=p.device)[None]<=cutoffs[:,None]
    correction=torch.where(within,(p-positions).norm(dim=-1),0.).amax(-1)
    return packets,dict(position_origin_m=origin,velocity_origin_m_s=velocities,acceleration_origin_m_s2=a,
        integration_position_correction_m=correction)


def reference_packet_validity(packets,limits):
    acceleration=packets[:,:,6:9];specific=acceleration+packets.new_tensor([0.,0.,9.80665])
    norm=specific.norm(dim=-1);vertical=specific[:,:,2]
    tilt=torch.rad2deg(torch.atan2(specific[:,:,:2].norm(dim=-1),vertical))
    speed=packets[:,:,3:6].norm(dim=-1)
    bad=(~torch.isfinite(packets).all(-1)|(vertical<limits['minimum_specific_vertical_m_s2'])|
        (tilt>limits['maximum_tilt_deg'])|(norm>limits['maximum_specific_force_m_s2'])|(speed>limits['maximum_speed_m_s']))
    return ~bad,dict(tilt=tilt,norm=norm,vertical=vertical,speed=speed)


def reference_feasibility(packets,cutoffs,dt,limits):
    valid,metrics=reference_packet_validity(packets,limits)
    tilt,norm,vertical,speed=(metrics[n] for n in ('tilt','norm','vertical','speed'))
    valid_time=torch.arange(packets.shape[1],device=packets.device)[None]/30.<=cutoffs[:,None]*dt+1e-10
    bad=~valid&valid_time
    return ~bad.any(-1),dict(maximum_reference_tilt_deg=torch.where(valid_time,tilt,0.).amax(-1),
        maximum_reference_specific_force_m_s2=torch.where(valid_time,norm,0.).amax(-1),
        minimum_reference_specific_vertical_m_s2=torch.where(valid_time,vertical,torch.inf).amin(-1),
        maximum_reference_speed_m_s=torch.where(valid_time,speed,0.).amax(-1))
