"""Versioned soft preferred-shape prior, with one causal encounter per rollout.

This is a user preference from an archived simulation, not a learned physical
model or a validated wave detector. All operations batch on the rollout device.
"""
import numpy as np
import torch
from planning.mppi_trajectory import ObjectiveCapture, OBJECTIVE

WAVE_OBJECTIVE = dict(OBJECTIVE, schema='preferred_fold_v1', distance=0., joint=0.,
    pull=0., release=0., wave=0., hit=0., vertical=0., approach=15., lateral=30.,
    contact=450., fold=600., cast=200., miss=300., shape_sigma=.12,
    tangent_sigma=.55, reference_file='wave_reference.npz')

ENCOUNTER_FIELDS = ('encounter_time','encounter_distance','encounter_q',
    'encounter_tip_velocity','encounter_drone_velocity','encounter_origin',
    'encounter_backward','encounter_tip_first')


def record_encounter(env, previous, previous_origin, previous_velocity, q, v,
                     entry, nearest_fraction, running, clock_left):
    """150 Hz swept-segment first ANY contact, else closest approach so far.

Called before the legacy contact latch changes; never overwrites first contact.
All stored kinematics use the same interpolated interval fraction.
"""
    first=entry.amin(-1); touch=torch.isfinite(first)
    f=torch.where(touch,first,nearest_fraction).clamp(0,1)
    cq=previous.positions_m+f[:,None,None]*(q-previous.positions_m)
    distance=(cq[:,-1]-env.target).norm(dim=-1)
    accept=running&~env.contact&(touch|(distance<env.encounter_distance))
    origin=previous_origin+f[:,None]*(env.pose.position-previous_origin)
    values=dict(encounter_time=clock_left+f*env.dt,encounter_distance=distance,encounter_q=cq,
        encounter_tip_velocity=previous.velocities_m_s[:,-1]+f[:,None]*(v[:,-1]-previous.velocities_m_s[:,-1]),
        encounter_drone_velocity=previous_velocity+f[:,None]*(env.pose.velocity-previous_velocity),
        encounter_origin=origin,
        encounter_backward=env.pull_peak-((origin-env.origin0)*env.direction).sum(-1),
        encounter_tip_first=touch&(entry[:,-1]<entry[:,:-1].amin(-1)))
    for name,value in values.items():
        mask=accept.reshape((-1,)+(1,)*(value.ndim-1))
        setattr(env,name,torch.where(mask,value,getattr(env,name)))


def material_resample(q,material,points=21):
    """Interpolate by rest arclength, making subdivision of an edge invariant."""
    s=torch.linspace(0,1,points,device=q.device,dtype=q.dtype)
    right=torch.searchsorted(material,s).clamp(1,len(material)-1);left=right-1
    f=(s-material[left])/(material[right]-material[left])
    return q[...,left,:]*(1-f[:,None])+q[...,right,:]*f[:,None]


def sample_history(q,times,end,phases,end_q):
    """Batch variable-end histories; no state beyond the encounter is used."""
    query=end[:,None]*phases[None]
    left=(torch.searchsorted(times,query.contiguous(),right=True)-1).clamp(0,len(times)-2)
    right=left+1;batch=torch.arange(len(q),device=q.device)[:,None]
    q0=q[batch,left];q1=q[batch,right]
    beyond=times[right]>end[:,None]
    q1=torch.where(beyond[:,:,None,None],end_q[:,None],q1)
    t1=torch.minimum(times[right],end[:,None]);t0=times[left]
    f=((query-t0)/(t1-t0).clamp_min(1e-12)).clamp(0,1)
    return q0+(q1-q0)*f[:,:,None,None]


def fold_quality(q,times,end,end_q,material,length,reference,shape_sigma=.12,tangent_sigma=.55):
    """Full-coverage monotone timing choices, with a small warp penalty.

    Compare position AND unit tangent histories, preserving world strike axis.
    Every phase contributes; static folds cannot choose only a matching frame.
    Curvature is represented by the evolution of spatially resolved tangents,
    rather than a mesh-sensitive maximum local turning angle.
    """
    phases=torch.linspace(0,1,len(reference),device=q.device,dtype=q.dtype)
    scores=[]
    ref_edges=reference[:,1:]-reference[:,:-1]
    ref_tangent=ref_edges/ref_edges.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    for exponent in (.75,1.,1.3333333333333333):
        sampled=sample_history(q,times,end,phases.pow(exponent),end_q)
        normalized=(sampled-sampled[:,:,:1])/length
        shape=material_resample(normalized,material,reference.shape[1])
        edge=shape[:,:,1:]-shape[:,:,:-1]
        tangent=edge/edge.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        position_error=(shape-reference).square().sum(-1).mean((1,2))
        tangent_error=(tangent-ref_tangent).square().sum(-1).mean((1,2))
        scores.append(torch.exp(-position_error/shape_sigma**2-tangent_error/tangent_sigma**2
                                -.1*abs(np.log(exponent))))
    return torch.stack(scores).amax(0)* (end>0).to(q.dtype)


def encounter_quality(distance,tip_v,drone_v,backward,reach,direction,tip_first,contact,task,scale):
    proximity=torch.exp(-(distance/scale).square())
    speed=((tip_v*direction).sum(-1)/task['minimum_directed_speed_m_s']).clamp(0,1)
    reverse=(-(drone_v*direction).sum(-1)/task['minimum_backward_speed_m_s']).clamp(0,1)
    travel=(backward/task['minimum_backward_distance_m']).clamp(0,1)
    tip_factor=torch.where(contact,tip_first.to(distance.dtype),torch.ones_like(distance))
    contact_quality=proximity*(.2+.8*speed)*(.2+.8*reverse)*(.2+.8*travel)*tip_factor
    alignment=((tip_v*direction).sum(-1)/tip_v.norm(dim=-1).clamp_min(1e-12)).clamp(0,1).square()
    cast=proximity*speed*reach.clamp(0,1)*alignment*tip_factor
    return contact_quality,cast,proximity


class PreferredFoldCapture(ObjectiveCapture):
    def __init__(self,reference):super().__init__();self.reference=reference

    def score(self,env,result,actions,weights):
        # Existing effort/recovery costs retain their meanings. Legacy rewards
        # and strict wave success remain telemetry, not lexicographic priority.
        _,terms=super().score(env,result,actions,weights)
        q=torch.stack([f[2] for f in self.frames],1)
        times=env.tensor([row['time'] for row in self.metrics])
        material=torch.cat((env.wave_material.new_zeros(1),env.wave_material,env.wave_material.new_ones(1)))
        fold=fold_quality(q,times,env.encounter_time,env.encounter_q,material,env.cable_length,
            self.reference,weights['shape_sigma'],weights['tangent_sigma'])
        reach=((env.encounter_q[:,-1]-env.encounter_q[:,0])*env.direction).sum(-1)/env.cable_length
        contact,cast,proximity=encounter_quality(env.encounter_distance,env.encounter_tip_velocity,
            env.encounter_drone_velocity,env.encounter_backward,reach,env.direction,
            env.encounter_tip_first,env.contact,env.settings['task'],weights['proximity_scale_m'])
        terms.update(contact=weights['contact']*contact,fold=weights['fold']*fold*proximity,
            cast=weights['cast']*cast,miss=-weights['miss']*(1-proximity))
        total=sum(terms.values())
        if not bool(torch.isfinite(total).all()):raise ValueError('Nonfinite preferred-fold score')
        return total.masked_fill(result['failed'],-torch.inf),terms
