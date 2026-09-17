"""Continuous moving turn, descent and slow approach after an unchanged whip.

Two quintics join with a nonzero approach velocity. No stationary braking
waypoint, circular aerobatics, or change to the whip. Limits describe reference
shaping only; the complete reference still passes the saved model's envelope.
"""
import numpy as np
from numpy.polynomial import polynomial as poly


DEFAULTS = dict(minimum_turn_s=1., maximum_turn_s=8., approach_speed_m_s=.35,
    approach_distance_m=.8, approach_height_m=.4, hold_s=3.,
    maximum_downward_acceleration_m_s2=3.5, maximum_horizontal_acceleration_m_s2=5.,
    maximum_descent_speed_m_s=1.25, maximum_tilt_deg=45.,
    maximum_speed_m_s=5., maximum_specific_force_m_s2=18.28571428571429,
    minimum_specific_vertical_m_s2=2.)


def coefficients(p, v, a, end, end_v, duration):
    """Quintic in normalized time, matching initial PVA and final PV/zero A."""
    c = np.zeros((6, 3))
    c[:3] = p, v*duration, a*duration**2/2
    c[3:] = np.linalg.solve([[1,1,1],[3,4,5],[6,12,20]],
        np.stack((end-c[:3].sum(0), end_v*duration-c[1]-2*c[2], -2*c[2])))
    return c


def evaluate(c, time, duration):
    u = np.asarray(time, float)/duration
    return tuple(poly.polyval(u, poly.polyder(c, m=k, axis=0)).T/duration**k for k in range(3))


def extrema(c):
    """Exact polynomial extrema over normalized time [0, 1]."""
    roots = poly.polyroots(poly.polyder(c))
    real = roots.real[(abs(roots.imag)<1e-7)&(roots.real>0)&(roots.real<1)]
    values = poly.polyval(np.r_[0., real, 1.], c)
    return float(values.min()), float(values.max())


def squared_norm(c):
    return sum((np.convolve(axis, axis) for axis in c.T), np.zeros(2*len(c)-1))


def metrics(c, duration):
    v = poly.polyder(c, axis=0)/duration
    a = poly.polyder(v, axis=0)/duration
    specific = a.copy();specific[0,2] += 9.80665
    return dict(peak_height_m=extrema(c[:,2])[1], minimum_height_m=extrema(c[:,2])[0],
        peak_speed_m_s=np.sqrt(max(0.,extrema(squared_norm(v))[1])),
        minimum_vz_m_s=extrema(v[:,2])[0], minimum_az_m_s2=extrema(a[:,2])[0],
        peak_horizontal_acceleration_m_s2=np.sqrt(max(0.,extrema(squared_norm(a[:,:2]))[1])),
        peak_specific_force_m_s2=np.sqrt(max(0.,extrema(squared_norm(specific))[1])),
        minimum_specific_vertical_m_s2=extrema(specific[:,2])[0])


def permitted(c, duration, options, tilt_limit):
    m = metrics(c, duration)
    a = poly.polyder(c, m=2, axis=0)/duration**2
    vertical = a[:,2].copy();vertical[0] += 9.80665
    tilt_polynomial = poly.polysub(squared_norm(a[:,:2]),
        np.tan(np.deg2rad(tilt_limit))**2*poly.polymul(vertical, vertical))
    valid = (m['minimum_az_m_s2'] >= -options['maximum_downward_acceleration_m_s2']-1e-8
        and m['peak_horizontal_acceleration_m_s2'] <= options['maximum_horizontal_acceleration_m_s2']+1e-8
        and m['minimum_vz_m_s'] >= -options['maximum_descent_speed_m_s']-1e-8
        and m['peak_speed_m_s'] <= options['maximum_speed_m_s']+1e-8
        and m['peak_specific_force_m_s2'] <= options['maximum_specific_force_m_s2']+1e-8
        and m['minimum_specific_vertical_m_s2'] >= options['minimum_specific_vertical_m_s2']-1e-8
        and extrema(tilt_polynomial)[1] <= 1e-7)
    return valid, m


def plan_curved_recovery(position, velocity, acceleration, hover, settings=None, *, height_bounds=None):
    options = dict(DEFAULTS, **(settings or {}))
    if set(options)!=set(DEFAULTS) or any(not np.isfinite(v) or v<=0 or v>120 for v in options.values()):
        raise ValueError('Recovery settings must be known, finite, positive and <=120.')
    if options['maximum_tilt_deg']>=90 or options['minimum_turn_s']>options['maximum_turn_s']:
        raise ValueError('Invalid recovery tilt or duration interval.')
    p,v,a,target = [np.asarray(x,float) for x in (position,velocity,acceleration,hover)]
    if any(x.shape!=(3,) or not np.isfinite(x).all() for x in (p,v,a,target)):
        raise ValueError('Recovery requires finite XYZ boundary states.')
    if height_bounds is not None:
        if len(height_bounds)!=2 or not np.isfinite(height_bounds).all() or height_bounds[0]>=height_bounds[1]:
            raise ValueError('Recovery height bounds must be a finite increasing pair.')
    def height_valid(m):
        return height_bounds is None or (m['minimum_height_m']>=height_bounds[0]-1e-8 and m['peak_height_m']<=height_bounds[1]+1e-8)
    # The handover acceleration cannot be reset. Permit its existing demand,
    # while disallowing a more aggressive turn than either it or our defaults.
    options['maximum_downward_acceleration_m_s2'] = max(options['maximum_downward_acceleration_m_s2'], -a[2])
    options['maximum_horizontal_acceleration_m_s2'] = max(options['maximum_horizontal_acceleration_m_s2'],np.linalg.norm(a[:2]))
    options['maximum_descent_speed_m_s'] = max(options['maximum_descent_speed_m_s'], -v[2])
    initial_tilt = np.degrees(np.arctan2(np.linalg.norm(a[:2]),a[2]+9.80665))
    tilt_limit = max(options['maximum_tilt_deg'],initial_tilt)
    if tilt_limit>=90 or a[2]+9.80665<options['minimum_specific_vertical_m_s2']:
        raise ValueError('Whip exit is outside the positive-thrust recovery envelope.')
    moving = np.linalg.norm(v)>1e-6
    if moving:
        direction = v[:2] if np.linalg.norm(v[:2])>.05 else (p-target)[:2]
        if np.linalg.norm(direction)<1e-9:direction=np.array([1.,0.])
        direction=direction/np.linalg.norm(direction)
        waypoint=target+np.r_[direction*options['approach_distance_m'],
            min(options['approach_height_m'],max(0.,p[2]-target[2]))]
        distance=np.linalg.norm(target-waypoint)
        approach_v=(target-waypoint)/distance*options['approach_speed_m_s']
        approach_s=np.ceil(2*distance/options['approach_speed_m_s']*30)/30
        approach=coefficients(waypoint,approach_v,np.zeros(3),target,np.zeros(3),approach_s)
        approach_ok,approach_metrics=permitted(approach,approach_s,options,tilt_limit)
        if not approach_ok or not height_valid(approach_metrics):
            raise ValueError('The curved recovery approach exceeds its reference envelope.')
    else:
        # A stationary exit needs no manufactured loop.
        waypoint=target;approach_v=np.zeros(3);approach_s=0.;approach=None
    candidates=[]
    for frames in range(int(np.ceil(options['minimum_turn_s']*30)),int(np.floor(options['maximum_turn_s']*30))+1):
        duration=frames/30
        c=coefficients(p,v,a,waypoint,approach_v,duration)
        ok,m=permitted(c,duration,options,tilt_limit)
        if ok and height_valid(m):candidates.append((m['peak_height_m'],duration,c,m))
    if not candidates:
        raise ValueError('No curved recovery meets the reference limits for this whip exit; no CSV generated.')
    # Lower ascent without an abrupt velocity/acceleration reset. Duration is
    # selected on the native clock; no numerical optimizer or model refit.
    _,turn_s,turn,turn_metrics=min(candidates,key=lambda row:(row[0],row[1]))
    return_end=turn_s+approach_s;end=return_end+options['hold_s']

    def sample(time):
        t=np.asarray(time,float)
        if t.ndim!=1 or not np.isfinite(t).all() or np.any(t < -1e-10) or np.any(t>end+1e-10):
            raise ValueError('Recovery samples must be within the planned interval.')
        pp=np.broadcast_to(target,(len(t),3)).copy();vv=np.zeros_like(pp);aa=np.zeros_like(pp)
        first=t<turn_s
        pp[first],vv[first],aa[first]=evaluate(turn,t[first],turn_s)
        second=(t>=turn_s)&(t<return_end)
        if second.any():pp[second],vv[second],aa[second]=evaluate(approach,t[second]-turn_s,approach_s)
        return pp,vv,aa

    return sample,dict(schema='curved_moving_recovery_v1',settings=options,height_bounds_m=height_bounds,
        turn_end_s=turn_s,brake_end_s=turn_s,return_s=approach_s,return_end_s=return_end,
        total_duration_s=end,hold_s=options['hold_s'],hover_position_m=target.tolist(),
        approach_position_m=waypoint.tolist(),approach_velocity_m_s=approach_v.tolist(),
        turn_coefficients_normalized=turn.tolist(),
        approach_coefficients_normalized=approach.tolist() if approach is not None else None,
        turn_metrics=turn_metrics,maximum_tilt_limit_deg=float(tilt_limit),
        phase_description='Moving curved turn/descent, slow approach, stationary final hold',
        stationary_braking_waypoint=False,vehicle_limits_verified=False)
