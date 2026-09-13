"""Analytic, continuous-PVA recovery references; no vehicle or cable simulation."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from .fullstate_recovery import quintic, evaluate

# Reference shaping choices, NOT measured vehicle capabilities.
DEFAULTS = dict(transition_s=.3, ramp_out_s=.3, horizontal_braking_m_s2=2.5,
                vertical_braking_m_s2=1.5, return_s=10., hold_s=3.)


def ramp(p0, v0, a0, a1, duration, time):
    """Smoothstep acceleration with analytically integrated velocity/position."""
    t = np.asarray(time, dtype=float)
    T = float(duration)
    delta = a1-a0
    a = a0 + delta*(3*t*t/T**2-2*t**3/T**3)
    v = v0+a0*t+delta*(t**3/T**2-t**4/(2*T**3))
    p = p0+v0*t+a0*t*t/2+delta*(t**4/(4*T**2)-t**5/(10*T**3))
    return p, v, a


def plan_recovery(position, velocity, acceleration, hover, settings=None):
    options = dict(DEFAULTS, **(settings or {}))
    if set(options) != set(DEFAULTS) or any(not np.isfinite(v) or v <= 0 or v > 120 for v in options.values()):
        raise ValueError('Recovery settings must be known, finite, positive and <=120.')
    p0, v0, a0, hover = [np.asarray(x, dtype=float) for x in (position, velocity, acceleration, hover)]
    if any(x.shape != (3,) or not np.isfinite(x).all() for x in (p0,v0,a0,hover)):
        raise ValueError('Recovery requires finite XYZ boundary states.')
    transition, release = options['transition_s'], options['ramp_out_s']
    # Choose the braking plateau so the integral of acceleration cancels v0.
    # The initial acceleration is respected; it is never reset at handover.
    effective_v = v0 + transition*a0/2
    base_duration = (transition+release)/2
    horizontal_hold = max(0., np.linalg.norm(effective_v[:2])/options['horizontal_braking_m_s2']-base_duration)
    vertical_hold = max(0., abs(effective_v[2])/options['vertical_braking_m_s2']-base_duration)
    holds = np.array([horizontal_hold, horizontal_hold, vertical_hold])
    braking = -effective_v/(base_duration+holds)
    axis_ends = transition+holds+release
    brake_end = float(axis_ends.max())
    stops = np.empty(3)
    boundaries = []
    for axis in range(3):
        p1,v1,_ = ramp(p0[axis],v0[axis],a0[axis],braking[axis],transition,transition)
        h = holds[axis]
        p2,v2 = p1+v1*h+braking[axis]*h*h/2, v1+braking[axis]*h
        p3,v3,a3 = ramp(p2,v2,braking[axis],0.,release,release)
        if abs(v3)>1e-9 or abs(a3)>1e-9:
            raise ValueError('Recovery braking boundary is not stationary.')
        stops[axis] = p3
        boundaries.append((p1,v1,p2,v2))
    back = options['return_s']
    # Bound the rest-to-rest return's peak acceleration and speed by lengthening
    # it. Exact quintic extrema: max |a| = 10/sqrt(3)*distance/T², max |v|=1.875*d/T.
    distance = float(np.linalg.norm(hover-stops))
    back = max(back, np.sqrt((10/np.sqrt(3))*distance/.3), 1.875*distance/.4)
    back = np.ceil(back*30)/30  # Adaptive duration is recorded explicitly.
    c = quintic(stops, np.zeros(3), np.zeros(3), hover, back)
    return_end = brake_end+back
    end = return_end+options['hold_s']

    def sample(time):
        t = np.asarray(time, dtype=float)
        if t.ndim != 1 or not np.isfinite(t).all() or np.any(t < -1e-10) or np.any(t > end+1e-10):
            raise ValueError('Recovery samples must be within the planned interval.')
        p = np.broadcast_to(hover,(len(t),3)).copy()
        v,a = np.zeros_like(p),np.zeros_like(p)
        for axis in range(3):
            p1,v1,p2,v2 = boundaries[axis]
            first = t < transition
            plateau = (t>=transition)&(t<transition+holds[axis])
            last = (t>=transition+holds[axis])&(t<axis_ends[axis])
            stopped = (t>=axis_ends[axis])&(t<brake_end)
            p[first,axis],v[first,axis],a[first,axis] = ramp(p0[axis],v0[axis],a0[axis],braking[axis],transition,t[first])
            local = t[plateau]-transition
            p[plateau,axis] = p1+v1*local+braking[axis]*local**2/2
            v[plateau,axis] = v1+braking[axis]*local
            a[plateau,axis] = braking[axis]
            p[last,axis],v[last,axis],a[last,axis] = ramp(p2,v2,braking[axis],0.,release,t[last]-transition-holds[axis])
            p[stopped,axis] = stops[axis]
        returning = (t>=brake_end)&(t<return_end)
        p[returning],v[returning],a[returning] = evaluate(c,t[returning]-brake_end)
        return p,v,a

    details = dict(settings=options, transition_s=transition, brake_end_s=brake_end,
        return_end_s=float(return_end), total_duration_s=float(end), return_s=float(back),
        hold_s=options['hold_s'], hover_position_m=hover.tolist(), brake_position_m=stops.tolist(),
        braking_acceleration_m_s2=braking.tolist(), axis_stop_times_s=axis_ends.tolist(),
        axis_plateau_end_times_s=(transition+holds).tolist(),
        return_peak_acceleration_limit_m_s2=.3, return_peak_speed_limit_m_s=.4,
        vehicle_limits_verified=False)
    return sample, details


def replace_recorded_recovery(directory, settings=None):
    """Replace recovery in a NEW export, preserving all original whip CSV rows.

    The original recording/export is retained as provenance inside that bundle.
    Call on a newly written export or an explicitly copied bundle, never originals.
    """
    directory = Path(directory)
    meta_path, csv_path = directory/'fullstate.json', directory/'fullstate_30hz.csv'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    if meta['schema'] != 'recorded_rehearsal_fullstate_v3':
        raise ValueError('Gentle recovery expects an original recorded rehearsal export.')
    with csv_path.open(newline='',encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    cutoff = float(meta['whip_end_s'])
    whip = [r for r in rows if float(r['time_s'])<=cutoff+1e-10]
    if not whip or abs(float(whip[-1]['time_s'])-cutoff)>1e-9:
        raise ValueError('Missing exact whip boundary; cannot preserve it.')
    keys = [('px_m','py_m','pz_m'),('vx_m_s','vy_m_s','vz_m_s'),('ax_m_s2','ay_m_s2','az_m_s2')]
    initial = [np.array([float(whip[-1][k]) for k in group]) for group in keys]
    sample, details = plan_recovery(*initial,meta['hover_position_m'],settings)
    end = cutoff+details['total_duration_s']
    times = np.arange(int(np.ceil(end*30))+1)/30
    times = times[(times>cutoff+1e-10)&(times<end-1e-10)]
    times = np.r_[times,end]
    p,v,a = sample(times-cutoff)
    tail=[]
    for i,time in enumerate(times):
        local = time-cutoff
        phase = ('recovery_transition' if local<details['transition_s']-1e-10 else
                 'recovery_brake' if local<details['brake_end_s']-1e-10 else
                 'recovery_return' if local<details['return_end_s']-1e-10 else 'hover_hold')
        row = dict(zip(fields, [time,*p[i],*v[i],*a[i],float(whip[-1]['yaw_rad']),0.,len(whip)+i,phase]))
        tail.append(row)
    # Dense diagnostics include boundaries; these are reference demands only.
    dense_t = np.unique(np.r_[np.arange(0,details['total_duration_s'],.001),
        details['transition_s'],details['axis_plateau_end_times_s'],details['axis_stop_times_s'],
        details['brake_end_s'],details['return_end_s'],details['total_duration_s']])
    dp,dv,da = sample(dense_t)
    gravity = float(meta.get('source_force_plan',{}).get('controller_export',{}).get('gravity_m_s2',9.80665))
    specific = da+[0,0,gravity]
    if np.min(specific[:,2])<=0:
        raise ValueError('Recovery includes nonpositive vertical feedforward thrust; reference refused.')
    tilt = np.rad2deg(np.arctan2(np.linalg.norm(specific[:,:2],axis=1),specific[:,2]))
    returning = dense_t>=details['brake_end_s']
    directions = specific/np.linalg.norm(specific,axis=1)[:,None]
    steps = np.rad2deg(np.arccos(np.clip(np.sum(directions[1:]*directions[:-1],axis=1),-1,1)))
    details.update(position_min_m=dp.min(0).tolist(), position_max_m=dp.max(0).tolist(),
        minimum_az_m_s2=float(da[:,2].min()), peak_tilt_surrogate_deg=float(tilt.max()),
        return_peak_tilt_surrogate_deg=float(tilt[returning].max()),
        peak_direction_rate_surrogate_deg_s=float((steps/np.diff(dense_t)).max()),
        diagnostic_sample_interval_s=.001)
    backups = ('recorded_pid_reference.csv','recorded_pid_reference.json')
    if any((directory/name).exists() for name in backups):
        raise ValueError('Recovery has already been rewritten in this directory.')
    shutil.copy2(csv_path,directory/backups[0])
    shutil.copy2(meta_path,directory/backups[1])
    with csv_path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(whip+tail)
    meta.update(schema='gentle_recovery_fullstate_v4',source='unchanged recorded whip + analytic gentle recovery',
        sample_count=len(whip)+len(tail), total_duration_s=float(end), recovery=details,
        phases=['whip','recovery_transition','recovery_brake','recovery_return','hover_hold'],
        final_position_m=meta['hover_position_m'],final_velocity_m_s=[0.,0.,0.],final_acceleration_m_s2=[0.,0.,0.],
        interpolation='Original whip rows; analytic acceleration ramps/braking and rest-to-rest quintic return',
        endpoint='Stationary commanded hover; cable settling not simulated for the new recovery',
        original_whip_rows=len(whip), recovery_preview='analytic drone reference only; original cable preview is not this recovery',
        limitations=['Recovery shaping defaults are not verified vehicle limits.',
            'Transition retains the whip terminal acceleration; braking still requires tilt.',
            'Actual attitude tracking, cable loads and workspace clearance are not validated.',
            'The 3D source rehearsal retains its original PID recovery; inspect the new CSV reference plot.'])
    all_rows=whip+tail
    all_a=np.array([[float(r[k]) for k in keys[2]] for r in all_rows])
    all_v=np.array([[float(r[k]) for k in keys[1]] for r in all_rows])
    meta.update(peak_acceleration_m_s2=float(np.linalg.norm(all_a,axis=1).max()),
                peak_speed_m_s=float(np.linalg.norm(all_v,axis=1).max()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(9,6))
    ax=fig.add_subplot(111,projection='3d')
    wp=np.array([[float(r[k]) for k in keys[0]] for r in whip])
    ax.plot(*wp.T,color='#f59e0b',label='Unchanged whip')
    ax.plot(*dp.T,color='#3b82f6',label='Gentle recovery reference')
    ax.scatter(*meta['hover_position_m'],color='green',label='Hover')
    ax.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title='Exported drone reference (not measured flight)')
    bounds=np.vstack((wp,dp)); span=np.maximum(np.ptp(bounds,axis=0),.4)
    center=(bounds.max(0)+bounds.min(0))/2
    ax.set_xlim(center[0]-span[0]/2,center[0]+span[0]/2)
    ax.set_ylim(center[1]-span[1]/2,center[1]+span[1]/2)
    ax.set_zlim(center[2]-span[2]/2,center[2]+span[2]/2)
    ax.set_box_aspect(span)
    ax.legend();fig.tight_layout();fig.savefig(directory/'recovery_reference.png',dpi=140);plt.close(fig)
    files=('fullstate_30hz.csv','fullstate_source.npz',*backups,'recovery_reference.png')
    meta['files']={name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in files}
    meta['csv_sha256']=meta['files']['fullstate_30hz.csv']
    meta_path.write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    return meta
