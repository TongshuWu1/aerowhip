"""Offline kinematic references from the frozen force plan; no vehicle sender."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def simulate_trajectory(plan, physics, cancel_requested=None):
    q = plan.initial_state.positions_m.clone()
    v = plan.initial_state.velocities_m_s.clone()
    positions, velocities = [q[0].cpu().numpy().copy()], [v[0].cpu().numpy().copy()]
    for force in plan.forces_world_n:
        if cancel_requested and cancel_requested():
            raise InterruptedError('Trajectory preparation cancelled')
        q, v = physics(q, v, force[None])
        positions.append(q[0].cpu().numpy().copy())
        velocities.append(v[0].cpu().numpy().copy())
    return dict(time_s=np.arange(len(positions))*plan.dt_s,
                positions_m=np.asarray(positions), velocities_m_s=np.asarray(velocities))


def sample_fullstate(time_s, positions, velocities, rate_hz=30., *, sample_times=None):
    """Cubic Hermite p/v interpolation, with its analytic first/second derivatives.

    Samples at k/rate strictly before cutoff, followed by an exact terminal
    boundary sample. Never extend a strike to round its duration to a tick.
    """
    t, p, v = (np.asarray(x, dtype=float) for x in (time_s, positions, velocities))
    if (t.ndim != 1 or len(t) < 2 or p.shape != (len(t), 3) or v.shape != p.shape
            or not all(np.isfinite(x).all() for x in (t, p, v))
            or t[0] != 0 or np.any(np.diff(t) <= 0)
            or not np.isfinite(rate_hz) or rate_hz <= 0):
        raise ValueError('Expected finite increasing trajectory times, Nx3 states and positive rate')
    times = np.arange(int(np.ceil(t[-1]*rate_hz)))/rate_hz
    times = np.r_[times[times < t[-1]-1e-12], t[-1]]
    if sample_times is not None:
        times = np.asarray(sample_times,dtype=float)
        if (times.ndim!=1 or not len(times) or not np.isfinite(times).all()
                or np.any(np.diff(times)<=0) or times[0]<t[0] or times[-1]>t[-1]):
            raise ValueError('Sample times must increase within the recorded trajectory')
    i = np.clip(np.searchsorted(t, times, side='right')-1, 0, len(t)-2)
    h = (t[i+1]-t[i])[:, None]
    u = ((times-t[i])[:, None])/h
    # Polynomial in physical interval time, preserving endpoint p and v.
    c0, c1 = p[i], v[i]
    c2 = 3*(p[i+1]-p[i])/h**2-(2*v[i]+v[i+1])/h
    c3 = -2*(p[i+1]-p[i])/h**3+(v[i]+v[i+1])/h**2
    x = u*h
    return times, c0+c1*x+c2*x*x+c3*x**3, c1+2*c2*x+3*c3*x*x, 2*c2+6*c3*x


def write_fullstate(trajectory, directory, source_metadata):
    if 'reference' in trajectory:
        return write_complete_fullstate(trajectory, directory, source_metadata)
    directory = Path(directory)
    path = directory/'fullstate_30hz.csv'
    if path.exists():
        raise ValueError('Full-state export already exists')
    times, p, v, a = sample_fullstate(trajectory['time_s'],
        trajectory['positions_m'][:, 0], trajectory['velocities_m_s'][:, 0])
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_s', 'px_m', 'py_m', 'pz_m', 'vx_m_s', 'vy_m_s', 'vz_m_s',
                         'ax_m_s2', 'ay_m_s2', 'az_m_s2', 'yaw_rad', 'yaw_rate_rad_s'])
        for row in zip(times, p, v, a):
            writer.writerow([row[0], *row[1], *row[2], *row[3], 0., 0.])
    np.savez_compressed(directory/'fullstate_source.npz', **trajectory)
    metadata = dict(source_metadata, schema='simulated_fullstate_reference_v1',
        flight_ready=False, sample_rate_hz=30., reference_point='simulated_cable_attachment',
        acceleration_convention='kinematic second derivative; no gravity subtraction or mass division',
        interpolation='piecewise cubic Hermite position/velocity; acceleration can jump at physics knots',
        heading='constant zero yaw reference; no simulated attitude or body-rate prediction',
        terminal_sample='exact cutoff boundary for recovery handover; not an extra held command',
        execution='normal position/velocity feedback remains enabled; absolute start-relative deadlines',
        recovery='not included; terminal position and velocity are supplied for a separate recovery',
        limitations=['source trajectory preview, not a flight-controller tracking simulation',
            'verify world frame and tracked-point offset before mapping to vehicle',
            'verify installed cmdFullState interface, body rates and loaded-vehicle feedforward convention'],
        csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_plan_sha256=hashlib.sha256((directory/'plan.npz').read_bytes()).hexdigest(),
        peak_speed_m_s=float(np.linalg.norm(v, axis=1).max()),
        peak_acceleration_m_s2=float(np.linalg.norm(a, axis=1).max()))
    metadata['source_force_plan'] = {key: metadata.pop(key) for key in
        ('force_units', 'gravity_convention', 'controller_export') if key in metadata}
    (directory/'fullstate.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    return metadata


def write_complete_fullstate(trajectory, directory, source_metadata):
    """Versioned complete export. Legacy strike-only files remain untouched."""
    directory = Path(directory)
    filenames = ('fullstate_30hz.csv','fullstate_source.npz','fullstate_preview.npz','fullstate.json')
    if any((directory/name).exists() for name in filenames):
        raise ValueError('Full-state export already exists; choose a new plan directory')
    ref, recovery = trajectory['reference'], trajectory['recovery']
    t,p,v,a = [ref[k] for k in ('time_s','position_m','velocity_m_s','acceleration_m_s2')]
    if (not all(np.isfinite(x).all() for x in (t,p,v,a)) or np.any(np.diff(t)<=0)
            or not np.allclose(p[-1], recovery['hover_position_m'], atol=1e-10)
            or np.max(abs(v[-1]))>1e-10 or np.max(abs(a[-1]))>1e-10):
        raise ValueError('Invalid complete PVA reference or nonstationary hover endpoint')
    path = directory/'fullstate_30hz.csv'
    with path.open('x',newline='',encoding='utf-8') as stream:
        writer = csv.writer(stream)
        # Keep the original numeric columns first for named-column readers.
        writer.writerow(['time_s','px_m','py_m','pz_m','vx_m_s','vy_m_s','vz_m_s',
                         'ax_m_s2','ay_m_s2','az_m_s2','yaw_rad','yaw_rate_rad_s','sample_index','phase'])
        for i in range(len(t)):
            writer.writerow([t[i],*p[i],*v[i],*a[i],0.,0.,i,ref['phase'][i]])
    np.savez_compressed(directory/'fullstate_source.npz',
        **{k:trajectory[k] for k in ('time_s','positions_m','velocities_m_s')})
    np.savez_compressed(directory/'fullstate_preview.npz',**trajectory['preview'])
    meta = dict(source_metadata, schema='complete_fullstate_reference_v2',flight_ready=False,
        sample_rate_hz=30.,sample_count=len(t),cutoff_s=recovery['whip_end_s'],
        whip_end_s=recovery['whip_end_s'],total_duration_s=float(t[-1]),recovery=recovery,
        reference_point='simulated_cable_attachment',
        acceleration_convention='kinematic second derivative; no gravity subtraction or mass division',
        interpolation='unchanged strike cubic Hermite; C2 quintic braking and return; stationary hold',
        terminal_sample='final hover at total_duration_s; consume once, no extra sample-period wait',
        execution='takeoff and initial hold externally; entire CSV on absolute deadlines; then land',
        phases=['whip','recovery_brake','recovery_return','hover_hold'],
        peak_speed_m_s=float(np.linalg.norm(v,axis=1).max()),
        peak_acceleration_m_s2=float(np.linalg.norm(a,axis=1).max()),
        position_min_m=p.min(axis=0).tolist(),position_max_m=p.max(axis=0).tolist(),
        limitations=['Recovery preview assumes ideal attachment tracking, not a firmware controller.',
          'Final hold duration does not certify physical cable settling; inspect residual cable speed.',
          'Verify tracked-point/body reference mapping and vehicle motion limits before flight.'])
    meta['source_force_plan'] = {key:meta.pop(key) for key in
        ('force_units','gravity_convention','controller_export') if key in meta}
    meta['files'] = {name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in filenames[:-1]}
    meta['csv_sha256'] = meta['files']['fullstate_30hz.csv']
    meta['source_plan_sha256'] = hashlib.sha256((directory/'plan.npz').read_bytes()).hexdigest()
    (directory/'fullstate.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    import shutil
    for name in ('fullstate_playback.py','FULLSTATE_PLAYBACK.md'):
        shutil.copyfile(Path(__file__).parent/name, directory/name)
    return meta


def write_rehearsal_fullstate(arrays, summary, directory, source_metadata, *, gentle_recovery=False,
                             initial_world_attachment_offset=None, attachment_offset_tracking_m=None,
                             initial_tracking_orientation_xyzw=None, initial_pose_source=None):
    """Export recorded original whip/PID recovery. No physics or controller runs."""
    pose_metadata = None
    if attachment_offset_tracking_m is not None or initial_tracking_orientation_xyzw is not None:
        if (attachment_offset_tracking_m is None or initial_tracking_orientation_xyzw is None
                or initial_world_attachment_offset is not None):
            raise ValueError('Provide either a world offset or a tracking-frame offset with its initial orientation')
        from simulator.geometry import initial_world_offset
        initial_world_attachment_offset = initial_world_offset(
            attachment_offset_tracking_m, initial_tracking_orientation_xyzw)
        pose_metadata = dict(offset_tracking_frame_m=np.asarray(attachment_offset_tracking_m).tolist(),
            initial_tracking_to_world_xyzw=np.asarray(initial_tracking_orientation_xyzw).tolist(),
            initial_pose_source=initial_pose_source or 'caller supplied; acquisition not verified')
    directory = Path(directory)
    path = directory/'fullstate_30hz.csv'
    outputs = ('fullstate_30hz.csv','fullstate_source.npz','fullstate.json')
    if any((directory/name).exists() for name in outputs):
        raise ValueError('Full-state export exists; use a new completed rehearsal')
    strikes = [e for e in summary['events'] if e['event']=='strike']
    if len(strikes)!=1:
        raise ValueError('Expected one recorded strike')
    event = strikes[0]
    start = float(event['time_s'])
    absolute_times = np.asarray(arrays['time_s'],dtype=float)
    matches = np.flatnonzero(np.isclose(absolute_times,start,rtol=0,atol=1e-9))
    if len(matches)!=1:
        raise ValueError('Exact launch frame is missing from the recording')
    index = int(matches[0])
    t = absolute_times[index:]-absolute_times[index]
    if initial_world_attachment_offset is not None:
        # The new training execution model samples this same canonical 100 Hz
        # physics grid. Remove accumulated clock roundoff at acceleration knots.
        canonical=np.arange(len(t),dtype=float)*.01
        if not np.allclose(t,canonical,rtol=0,atol=1e-8):
            raise ValueError('Full-model export requires the recorded 100 Hz physics grid')
        t=canonical
    q = np.asarray(arrays['cable_node_position_world_m'])[index:]
    velocity = np.asarray(arrays['cable_node_velocity_world_m_s'])[index:]
    source_phase = np.asarray(arrays['controller_phase'])[index:]
    cutoff = float(event['planned_duration_s'])
    if t[-1]<=cutoff or source_phase[-1]!=0:
        raise ValueError('Recording does not include return to hover')
    if not all(np.isfinite(x).all() for x in (t,q,velocity)):
        raise ValueError('Nonfinite recorded trajectory')
    regular,_,_,_ = sample_fullstate(t,q[:,0],velocity[:,0])
    times = regular
    if not np.any(abs(times-cutoff)<1e-10):
        times = np.sort(np.r_[times,cutoff])
    times,p,v,a = sample_fullstate(t,q[:,0],velocity[:,0],sample_times=times)
    if gentle_recovery:
        # The cutoff belongs to the whip. Floating timestamps must not select
        # the first PID interval and inject its force reversal into this row.
        boundary = np.flatnonzero(np.isclose(t,cutoff,rtol=0,atol=1e-9))
        target = np.flatnonzero(np.isclose(times,cutoff,rtol=0,atol=1e-10))
        if len(boundary)!=1 or boundary[0]<1 or len(target)!=1:
            raise ValueError('Exact end-of-whip state is required for gentle recovery.')
        b,k = int(boundary[0]),int(target[0])
        local = t[b-1:b+1]-t[b-1]
        _,_,_,left_a = sample_fullstate(local,q[b-1:b+1,0],velocity[b-1:b+1,0],
                                        sample_times=np.array([local[-1]]))
        p[k],v[k],a[k] = q[b,0],velocity[b,0],left_a[-1]
    source_indices = np.clip(np.searchsorted(t,times,side='right')-1,0,len(t)-1)
    phases = np.where(times<=cutoff+1e-10,'whip',
        np.where(source_phase[source_indices]==0,'hover_hold','pid_recovery'))
    with path.open('x',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream)
        writer.writerow(['time_s','px_m','py_m','pz_m','vx_m_s','vy_m_s','vz_m_s',
                         'ax_m_s2','ay_m_s2','az_m_s2','yaw_rad','yaw_rate_rad_s','sample_index','phase'])
        for i in range(len(times)):
            writer.writerow([times[i],*p[i],*v[i],*a[i],0.,0.,i,phases[i]])
    np.savez_compressed(directory/'fullstate_source.npz',time_s=t,positions_m=q,
                        velocities_m_s=velocity,controller_phase=source_phase)
    metadata = dict(source_metadata,schema='recorded_rehearsal_fullstate_v3',flight_ready=False,
        source='existing GPU rehearsal recording; original force strike and PID recovery',
        sample_rate_hz=30.,sample_count=len(times),cutoff_s=cutoff,whip_end_s=cutoff,
        total_duration_s=float(t[-1]),recording_start_time_s=start,
        hover_position_m=summary['hover_position_m'],phases=['whip','pid_recovery','hover_hold'],
        recovery='original rehearsal PID and settling behavior; no added trajectory or new simulation',
        reference_point='simulated_cable_attachment',
        acceleration_convention='kinematic derivative of recorded p/v; no gravity subtraction or mass division',
        interpolation='same cubic Hermite p/v and analytic derivatives as the original exporter',
        endpoint='actual recorded settled state; no snapping to hover or zeroing derivatives',
        final_position_m=p[-1].tolist(),final_velocity_m_s=v[-1].tolist(),final_acceleration_m_s2=a[-1].tolist(),
        peak_speed_m_s=float(np.linalg.norm(v,axis=1).max()),
        peak_acceleration_m_s2=float(np.linalg.norm(a,axis=1).max()),
        csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_plan_sha256=hashlib.sha256((directory/'plan.npz').read_bytes()).hexdigest(),
        limitations=['Recorded source simulation, not a real full-state tracking-controller model.',
            'PID acceleration can change at control updates; export does not change or smooth the simulation.',
            'Verify controlled reference point and actual vehicle tracking/landing conditions.'])
    if gentle_recovery:
        metadata['cutoff_acceleration'] = 'left-hand derivative of final whip interval; excludes PID recovery'
    metadata['source_force_plan']={key:metadata.pop(key) for key in
        ('force_units','gravity_convention','controller_export') if key in metadata}
    metadata['files']={name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in outputs[:-1]}
    (directory/'fullstate.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    import shutil
    for name in ('fullstate_playback.py','FULLSTATE_PLAYBACK.md'):
        shutil.copyfile(Path(__file__).parent/name,directory/name)
    if gentle_recovery:
        from .gentle_recovery import replace_recorded_recovery
        metadata = replace_recorded_recovery(directory)
    if initial_world_attachment_offset is not None:
        metadata = convert_attachment_reference(directory, metadata, initial_world_attachment_offset,
                                                pose_metadata=pose_metadata)
    return metadata


def convert_attachment_reference(directory, metadata, offset, *, pose_metadata=None):
    """Map a newly generated attachment reference to the cf_7 command origin.

Keep the original generated attachment CSV. Only positions change; force plan,
time, velocity and acceleration are identical. Historical exports are not used
as input to this worker-only step.
"""
    import shutil
    directory=Path(directory);offset=np.asarray(offset,dtype=float)
    if offset.shape!=(3,) or not np.isfinite(offset).all():raise ValueError('Expected finite world attachment offset')
    source=directory/'fullstate_30hz.csv';backup=directory/'attachment_reference.csv'
    if backup.exists():raise ValueError('Attachment reference was already converted')
    with source.open(newline='',encoding='utf-8') as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
    shutil.copy2(source,backup)
    for row in rows:
        for k,value in zip(('px_m','py_m','pz_m'),offset):row[k]=format(float(row[k])-value,'.17g')
    with source.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    metadata=dict(metadata)
    metadata['reference_point']='OptiTrack_cf7_origin'
    metadata['reference_mapping']=dict(initial_world_attachment_offset_m=offset.tolist(),
        mode='fixed_initial_translation', dynamic_rigid_body_conversion=False,
        position='vehicle_command = virtual_attachment_reference - initial_world_attachment_offset',
        velocity_acceleration='unchanged because this mapping is a constant translation',
        original_attachment_csv=backup.name,
        initial_pose_assumption='World offset supplied explicitly; orientation provenance unspecified',
        limitation='Not the inverse of rotating attachment PVA; dynamic conversion requires orientation and angular derivatives',
        controller_tracking_frame_mapping='flight-side bridge and installed firmware not verified')
    if pose_metadata is not None:
        metadata['reference_mapping'].update(pose_metadata)
        metadata['reference_mapping']['initial_pose_assumption']=pose_metadata['initial_pose_source']
    for key in ('hover_position_m','final_position_m'):
        if key in metadata:metadata[key]=(np.asarray(metadata[key])-offset).tolist()
    metadata['initial_vehicle_position_m']=[float(rows[0][k]) for k in ('px_m','py_m','pz_m')]
    if isinstance(metadata.get('recovery'),dict):
        metadata['recovery']=dict(metadata['recovery'])
        for key in ('hover_position_m','brake_position_m'):
            if key in metadata['recovery']:metadata['recovery'][key]=(np.asarray(metadata['recovery'][key])-offset).tolist()
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig=Figure(figsize=(9,6),layout='constrained');FigureCanvasAgg(fig);ax=fig.add_subplot(111,projection='3d')
    points=np.array([[float(r[k]) for k in ('px_m','py_m','pz_m')] for r in rows])
    times=np.array([float(r['time_s']) for r in rows]);strike=times<=metadata['cutoff_s']+1e-10
    ax.plot(*points[strike].T,color='#f59e0b',label='Whip reference')
    ax.plot(*points[~strike].T,color='#3b82f6',label='Gentle recovery')
    ax.scatter(*metadata['hover_position_m'],color='green',label='Vehicle hover')
    span=np.maximum(np.ptp(points,axis=0),.4);center=(points.max(0)+points.min(0))/2
    ax.set_xlim(center[0]-span[0]/2,center[0]+span[0]/2);ax.set_ylim(center[1]-span[1]/2,center[1]+span[1]/2);ax.set_zlim(center[2]-span[2]/2,center[2]+span[2]/2)
    ax.set_box_aspect(span);ax.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title='Exported cf_7 position reference');ax.legend()
    fig.savefig(directory/'recovery_reference.png',dpi=140)
    metadata.setdefault('files',{}).update({name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
        for name in ('fullstate_30hz.csv','attachment_reference.csv','recovery_reference.png')})
    metadata['csv_sha256']=metadata['files']['fullstate_30hz.csv']
    (directory/'fullstate.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    return metadata
