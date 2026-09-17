"""Observed virtual-target encounters and gap-aware offline tip-speed estimates."""
import numpy as np


def local_velocity(time,position,width=5):
    """Centered quadratic derivative on contiguous native observations; no gap fill.

    At 100 Hz, width 5 spans 40 ms. This is an offline measurement diagnostic,
    never a causal initializer or an independently measured impact force.
    """
    t=np.asarray(time,dtype=float);q=np.asarray(position,dtype=float)
    if width<3 or width%2!=1:raise ValueError('Odd window of at least three samples required')
    if q.shape!=(len(t),3) or len(t)<3 or np.any(np.diff(t)<=0):raise ValueError('Increasing XYZ observations required')
    v=np.full_like(q,np.nan);half=width//2;nominal=np.median(np.diff(t))
    for i in range(half,len(t)-half):
        s=slice(i-half,i+half+1);x=t[s]-t[i];y=q[s]
        if not np.isfinite(y).all() or np.diff(t[s]).max()>1.5*nominal:continue
        # Scale times for numerical conditioning; coefficient 1 is d(position)/dt.
        scale=x[-1]-x[0];u=x/scale
        v[i]=np.linalg.lstsq(np.c_[np.ones(width),u,u*u],y,rcond=None)[0][1]/scale
    return v


def encounter(time,tip,target,radius,velocity=None,direction=(1.,0.,0.)):
    """Adjacent observed segments only; return first sphere entry and nearest point.

    A swept intersection is geometric evidence, not confirmation of a physical
    collision. An unobserved gap cannot establish either contact or absence.
    """
    t=np.asarray(time,float);q=np.asarray(tip,float);target=np.asarray(target,float)
    direction=np.asarray(direction,float);direction=direction/np.linalg.norm(direction)
    if len(t)<2 or q.shape!=(len(t),3) or np.any(np.diff(t)<=0) or radius<=0:raise ValueError('Invalid encounter inputs')
    valid=np.isfinite(q).all(1);step=np.diff(t);nominal=np.median(step)
    minimum=float('inf');nearest=None;first=None
    def event(i,f):
        pos=q[i]+f*(q[i+1]-q[i]);item=dict(time_s=float(t[i]+f*step[i]),position_m=pos.tolist(),distance_m=float(np.linalg.norm(pos-target)))
        if velocity is not None and np.isfinite(velocity[i:i+2]).all():
            v=velocity[i]+f*(velocity[i+1]-velocity[i]);item.update(outward_speed_m_s=float(v@direction),total_speed_m_s=float(np.linalg.norm(v)))
        return item
    for i in range(len(t)-1):
        if not (valid[i] and valid[i+1]) or step[i]>1.5*nominal:continue
        d=q[i+1]-q[i];r=q[i]-target;a=float(d@d)
        f=float(np.clip(-(r@d)/a,0,1)) if a>1e-20 else 0.
        item=event(i,f)
        if item['distance_m']<minimum:minimum=item['distance_m'];nearest=item
        c=float(r@r-radius**2);b=2*float(r@d)
        fraction=0. if c<=0 else None
        discriminant=b*b-4*a*c
        if fraction is None and a>1e-20 and discriminant>=0:
            value=(-b-np.sqrt(discriminant))/(2*a)
            if 0<=value<=1:fraction=float(value)
        if first is None and fraction is not None:first=event(i,fraction)
    return dict(nearest=nearest,first_entry=first,observed_sphere_entry=first is not None,
        sample_coverage=float(valid.mean()),contiguous_segment_coverage=float(np.mean(valid[:-1]&valid[1:]&(step<=1.5*nominal))))
