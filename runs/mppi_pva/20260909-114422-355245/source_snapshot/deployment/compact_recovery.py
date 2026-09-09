"""Native FullState recovery with earlier vertical braking and descent.

Analytic P/V/A references only. The learned whip is never modified here.
Horizontal and vertical returns start independently, with shared conservative
speed/acceleration budgets. These are shaping limits, not measured vehicle limits.
"""
import numpy as np
from .gentle_recovery import DEFAULTS,plan_recovery


COMPACT_DEFAULTS=dict(DEFAULTS,vertical_braking_m_s2=3.5,
    vertical_transition_s=.2,vertical_ramp_out_s=.2,vertical_return_s=4.)
RETURN_SPEED=.4
RETURN_ACCELERATION=.3
VERTICAL_RETURN_SPEED=.3
VERTICAL_RETURN_ACCELERATION=.2


def plan_compact_recovery(position,velocity,acceleration,hover,settings=None):
    options=dict(COMPACT_DEFAULTS,**(settings or {}))
    if set(options)!=set(COMPACT_DEFAULTS) or any(not np.isfinite(v) or v<=0 or v>120 for v in options.values()):
        raise ValueError('Recovery settings must be known, finite, positive and <=120.')
    p,v,a,target=[np.asarray(x,dtype=float) for x in (position,velocity,acceleration,hover)]
    if any(x.shape!=(3,) or not np.isfinite(x).all() for x in (p,v,a,target)):
        raise ValueError('Recovery requires finite XYZ boundary states.')
    base={k:options[k] for k in DEFAULTS}
    xy_speed=np.sqrt(RETURN_SPEED**2-VERTICAL_RETURN_SPEED**2)
    xy_acceleration=np.sqrt(RETURN_ACCELERATION**2-VERTICAL_RETURN_ACCELERATION**2)

    def group(mask,settings,speed_limit,acceleration_limit):
        # First obtain the exact free stopping point. Then size the rest-to-rest
        # return using analytic quintic extrema and the assigned vector budget.
        args=[x*mask for x in (p,v,a,target)]
        _,meta=plan_recovery(*args,settings)
        distance=np.linalg.norm(args[-1]-np.asarray(meta['brake_position_m']))
        settings=dict(settings,return_s=max(settings['return_s'],1.875*distance/speed_limit,
            np.sqrt((10/np.sqrt(3))*distance/acceleration_limit)))
        return plan_recovery(*args,settings)

    horizontal,hmeta=group(np.array([1.,1.,0.]),base,xy_speed,xy_acceleration)
    vertical,vmeta=group(np.array([0.,0.,1.]),dict(base,
        transition_s=options['vertical_transition_s'],ramp_out_s=options['vertical_ramp_out_s'],
        return_s=options['vertical_return_s']),VERTICAL_RETURN_SPEED,VERTICAL_RETURN_ACCELERATION)
    return_end=max(hmeta['return_end_s'],vmeta['return_end_s'])
    end=return_end+options['hold_s']

    def sample(time):
        t=np.asarray(time,dtype=float)
        if t.ndim!=1 or not np.isfinite(t).all() or np.any(t < -1e-10) or np.any(t>end+1e-10):
            raise ValueError('Recovery samples must be within the planned interval.')
        hp,hv,ha=horizontal(np.clip(t,0,hmeta['total_duration_s']))
        zp,zv,za=vertical(np.clip(t,0,vmeta['total_duration_s']))
        return hp+zp,hv+zv,ha+za

    def axes(key):return [*hmeta[key][:2],vmeta[key][2]]
    details=dict(schema='independent_vertical_recovery_v1',settings=options,
        transition_s=options['transition_s'],vertical_transition_s=options['vertical_transition_s'],
        brake_end_s=max(hmeta['brake_end_s'],vmeta['brake_end_s']),
        axis_plateau_end_times_s=axes('axis_plateau_end_times_s'),axis_stop_times_s=axes('axis_stop_times_s'),
        brake_position_m=axes('brake_position_m'),braking_acceleration_m_s2=axes('braking_acceleration_m_s2'),
        return_start_times_s=[hmeta['brake_end_s'],hmeta['brake_end_s'],vmeta['brake_end_s']],
        axis_return_durations_s=[hmeta['return_s'],hmeta['return_s'],vmeta['return_s']],
        return_end_s=return_end,return_s=max(hmeta['return_s'],vmeta['return_s']),
        total_duration_s=end,hold_s=options['hold_s'],hover_position_m=target.tolist(),
        return_peak_speed_limit_m_s=RETURN_SPEED,return_peak_acceleration_limit_m_s2=RETURN_ACCELERATION,
        horizontal_return_speed_limit_m_s=xy_speed,horizontal_return_acceleration_limit_m_s2=xy_acceleration,
        vertical_return_speed_limit_m_s=VERTICAL_RETURN_SPEED,
        vertical_return_acceleration_limit_m_s2=VERTICAL_RETURN_ACCELERATION,vehicle_limits_verified=False)
    return sample,details
