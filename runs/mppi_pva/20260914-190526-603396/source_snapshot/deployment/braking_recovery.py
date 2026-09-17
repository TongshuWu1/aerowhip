"""Continuous-PVA braking, followed by a rest-to-rest return and final hold.

The stop position is determined by the exit PVA and candidate braking duration.
No position, velocity, or acceleration is reset at the whip/recovery boundary.
Historical curved recovery remains available for historical saved settings.
"""
import numpy as np
from numpy.polynomial import polynomial as poly
from .curved_recovery import DEFAULTS as ENVELOPE, coefficients, evaluate, permitted, extrema

SCHEMA='brake_return_hold_v1'
DEFAULTS=dict(schema=SCHEMA,minimum_brake_s=.4,maximum_brake_s=6.,
    minimum_return_s=2.,maximum_return_s=12.,hold_s=3.,
    braking_horizontal_acceleration_m_s2=5.,braking_downward_acceleration_m_s2=3.5,
    braking_tilt_deg=45.,return_speed_m_s=1.)


def validate(settings):
    if set(settings)!=set(DEFAULTS) or settings['schema']!=SCHEMA:
        raise ValueError('Explicit brake-return-hold recovery settings required')
    if any(not np.isfinite(v) or v<=0 or v>120 for k,v in settings.items() if k!='schema'):
        raise ValueError('Recovery settings must be finite, positive and <=120')
    if settings['braking_tilt_deg']>=90 or settings['minimum_brake_s']>settings['maximum_brake_s'] or settings['minimum_return_s']>settings['maximum_return_s']:
        raise ValueError('Invalid recovery duration or tilt bounds')


def brake_coefficients(p,v,a,duration):
    # Cubic velocity satisfies v(0)=v, a(0)=a, v(T)=a(T)=0.
    c=np.zeros((5,3));c[0]=p;c[1]=v*duration;c[2]=a*duration**2/2
    c[3]=-v*duration-2*a*duration**2/3;c[4]=v*duration/2+a*duration**2/4
    return c


def plan(position,velocity,acceleration,hover,limits,jerk_limits,settings):
    validate(settings)
    p,v,a,target=[np.asarray(x,float) for x in (position,velocity,acceleration,hover)]
    if any(x.shape!=(3,) or not np.isfinite(x).all() for x in (p,v,a,target)):
        raise ValueError('Finite XYZ recovery boundary conditions required')
    jerk_limits=np.asarray(jerk_limits,float)
    if jerk_limits.shape!=(3,) or not np.isfinite(jerk_limits).all() or (jerk_limits<=0).any():raise ValueError('Positive XYZ jerk bounds required')
    initial_tilt=np.degrees(np.arctan2(np.linalg.norm(a[:2]),a[2]+9.80665))
    options=dict(ENVELOPE,maximum_speed_m_s=limits['maximum_speed_m_s'],
        maximum_specific_force_m_s2=limits['maximum_specific_force_m_s2'],
        minimum_specific_vertical_m_s2=limits['minimum_specific_vertical_m_s2'],
        maximum_descent_speed_m_s=limits['maximum_speed_m_s'],
        maximum_horizontal_acceleration_m_s2=max(settings['braking_horizontal_acceleration_m_s2'],np.linalg.norm(a[:2])),
        maximum_downward_acceleration_m_s2=min(9.80665-limits['minimum_specific_vertical_m_s2'],
            max(settings['braking_downward_acceleration_m_s2'],-a[2])))
    tilt=min(limits['maximum_tilt_deg'],max(settings['braking_tilt_deg'],initial_tilt))
    def valid(c,t,opts,angle):
        ok,m=permitted(c,t,opts,angle)
        jerk=poly.polyder(c,m=3,axis=0)/t**3
        peak=np.array([max(abs(x) for x in extrema(axis)) for axis in jerk.T])
        ok &= bool((peak<=jerk_limits+1e-8).all())
        ok &= m['minimum_height_m']>=limits['minimum_origin_z_m']-1e-8 and m['peak_height_m']<=limits['maximum_origin_z_m']+1e-8
        m['peak_absolute_jerk_m_s3']=peak.tolist()
        return ok,m
    selected=None
    for frames in range(int(np.ceil(settings['minimum_brake_s']*30)),int(np.floor(settings['maximum_brake_s']*30))+1):
        duration=frames/30;curve=brake_coefficients(p,v,a,duration)
        ok,brake_metrics=valid(curve,duration,options,tilt)
        # Do not reverse along the exit-velocity direction during braking.
        if np.linalg.norm(v)>1e-8:
            forward=poly.polyder(curve,axis=0)@v/duration
            ok &= extrema(forward)[0]>=-1e-8
        if ok:
            stop=curve.sum(0)
            selected=(duration,curve,stop,brake_metrics);break
    if selected is None:raise ValueError('No smooth braking segment meets the saved recovery envelope')
    brake_s,brake,stop,brake_metrics=selected
    return_options=dict(options,maximum_speed_m_s=min(settings['return_speed_m_s'],limits['maximum_speed_m_s']),
        maximum_horizontal_acceleration_m_s2=settings['braking_horizontal_acceleration_m_s2'])
    selected=None
    for frames in range(int(np.ceil(settings['minimum_return_s']*30)),int(np.floor(settings['maximum_return_s']*30))+1):
        duration=frames/30;curve=coefficients(stop,np.zeros(3),np.zeros(3),target,np.zeros(3),duration)
        ok,return_metrics=valid(curve,duration,return_options,min(settings['braking_tilt_deg'],limits['maximum_tilt_deg']))
        if ok:selected=(duration,curve,return_metrics);break
    if selected is None:raise ValueError('No smooth return segment meets the saved recovery envelope')
    return_s,return_curve,return_metrics=selected
    return_end=brake_s+return_s;end=return_end+settings['hold_s']
    def sample(time):
        t=np.asarray(time,float)
        if t.ndim!=1 or not np.isfinite(t).all() or (t<0).any() or (t>end+1e-9).any():raise ValueError('Invalid recovery sample times')
        pp=np.broadcast_to(target,(len(t),3)).copy();vv=np.zeros_like(pp);aa=np.zeros_like(pp)
        first=t<brake_s;second=(t>=brake_s)&(t<return_end)
        pp[first],vv[first],aa[first]=evaluate(brake,t[first],brake_s)
        pp[second],vv[second],aa[second]=evaluate(return_curve,t[second]-brake_s,return_s)
        return pp,vv,aa
    details=dict(schema=SCHEMA,settings=settings,brake_end_s=brake_s,return_s=return_s,
        return_end_s=return_end,total_duration_s=end,hold_s=settings['hold_s'],
        hover_position_m=target.tolist(),braking_position_m=stop.tolist(),
        brake_coefficients_normalized=brake.tolist(),return_coefficients_normalized=return_curve.tolist(),
        brake_metrics=brake_metrics,return_metrics=return_metrics,stationary_braking_waypoint=True,
        phase_description='Smooth braking, rest-to-rest return, final hold',vehicle_limits_verified=False)
    return sample,details


def complete_packets(whip,hover,limits,jerk_limits,settings):
    whip=np.asarray(whip,float)
    if whip.ndim!=2 or whip.shape[1]!=11 or len(whip)<2 or not np.isfinite(whip).all():raise ValueError('Finite full whip PVA required')
    sample,details=plan(whip[-1,:3],whip[-1,3:6],whip[-1,6:9],hover,limits,jerk_limits,settings)
    tail_t=np.arange(1,round(details['total_duration_s']*30)+1)/30
    tail=np.c_[*sample(tail_t),np.tile(whip[-1,9:11],(len(tail_t),1))]
    packets=np.concatenate((whip,tail));times=np.arange(len(packets))/30;end=(len(whip)-1)/30
    phases=np.where(times<=end+1e-10,1,np.where(times<=end+details['brake_end_s']+1e-10,2,
        np.where(times<=end+details['return_end_s']+1e-10,3,4)))
    return times,packets,phases,details
