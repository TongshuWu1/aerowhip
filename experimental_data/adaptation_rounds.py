"""Versioned recording preparation. Does not fit physics or train policies."""
from pathlib import Path
import csv
import json
import re
import shutil
from uuid import uuid4
from time import perf_counter_ns

import numpy as np
from scipy.optimize import minimize_scalar

from .io import sha256_file, utc_now
from simulator.workflow import stamp
from simulator.geometry import attachment_positions, normalized_rotations_xyzw
from . import legacy_execution

COMMAND_COLUMNS = ['cmd_'+n for n in ('x','y','z','vx','vy','vz','ax','ay','az','yaw','yaw_rate')]


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def safe_name(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
        raise ValueError('Use letters, digits, underscores and hyphens for trial names')
    if 'fig8vertical_002' in name.lower():
        raise ValueError('Protected recording cannot enter adaptation')
    return name


def read_optitrack(path, drone_label=None, *, allow_external_filename=False):
    if 'fig8vertical_002' in str(path).lower():raise ValueError('Protected recording cannot enter adaptation')
    if not allow_external_filename:safe_name(Path(path).stem)
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.reader(stream))
    meta = dict(zip(rows[0][::2], rows[0][1::2]))
    if meta.get('Length Units') != 'Meters' or meta.get('Coordinate Space') != 'Global':
        raise ValueError('Expected OptiTrack global coordinates in meters; no implicit frame conversion')
    if meta.get('Rotation Type') != 'Quaternion':
        raise ValueError('Expected quaternion rotation export')
    if len(rows) < 9 or rows[6][:2] != ['Frame','Time (Seconds)']:
        raise ValueError('Unsupported OptiTrack header')
    # A complete pose identifies a rigid body; never select individual markers,
    # a prop body, or whichever set of XYZ columns happens to appear first.
    candidates=sorted({name for name in rows[3] if re.fullmatch(r'cf_?\d+',name,re.IGNORECASE)
        and all(any(n==name and rows[5][i]==kind and rows[6][i]==axis
                    for i,n in enumerate(rows[3]))
                for kind,axes in [('Position','XYZ'),('Rotation','XYZW')] for axis in axes)})
    identity_path=Path(path).with_suffix('.tracking.json')
    if drone_label is None and identity_path.exists():
        identity=json.loads(identity_path.read_text(encoding='utf-8'))
        if identity.get('optitrack_sha256')!=sha256_file(path):
            raise ValueError('Drone identity belongs to a different OptiTrack file; review the selection.')
        drone_label=identity['drone']
    if drone_label is None:
        if len(candidates)!=1:
            raise ValueError('Select the drone rigid body explicitly: expected one cf pose, found '+str(candidates)+
                             '. Use Recordings → Drone rigid body to save the selection.')
        drone_label=candidates[0]
    if drone_label not in candidates:
        raise ValueError(f'Drone rigid body {drone_label!r} has no complete XYZ/quaternion pose; available: {candidates}')
    def columns(name, kind, axes):
        result = []
        for axis in axes:
            found = [i for i,v in enumerate(rows[3]) if v==name and rows[5][i]==kind and rows[6][i]==axis]
            if len(found)!=1:
                raise ValueError(f'Missing or ambiguous {name} {kind} {axis}')
            result.append(found[0])
        return result
    ids = [0,1]+columns(drone_label,'Position','XYZ')+columns(drone_label,'Rotation','XYZW')
    ids += [i for k in range(1,11) for i in columns(f'cable1:c{k}','Position','XYZ')]
    a = np.asarray([[float(r[i]) if i<len(r) and r[i].strip() else np.nan for i in ids] for r in rows[7:] if r])
    validate_times(a[:,1])
    return dict(frame=a[:,0], time=a[:,1], drone=a[:,2:5], quaternion=a[:,5:9],
                cable=a[:,9:].reshape(-1,10,3), metadata=meta,drone_label=drone_label)


def validate_times(t):
    if len(t)<3 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('Need at least three strictly increasing finite timestamps')


def command_log_path(path, source='snapshots'):
    if source not in ('snapshots','event_log'):raise ValueError('Unknown command source')
    return Path(path).with_suffix('.commands.csv') if source=='event_log' else Path(path)


def read_controller(path, *, commands_only=False, allow_external_filename=False, command_source='snapshots'):
    if 'fig8vertical_002' in str(path).lower():raise ValueError('Protected recording cannot enter adaptation')
    if not allow_external_filename:safe_name(Path(path).stem)
    if command_source!='snapshots' and not commands_only:
        raise ValueError('Event logs contain commands, not measured vehicle poses')
    c = np.genfromtxt(command_log_path(path,command_source), delimiter=',', names=True, encoding='utf-8-sig')
    required = ['time_s','cmd_age','cmd_valid',*COMMAND_COLUMNS]
    if command_source=='event_log':
        if not {'time_s','cmd_sequence','cmd_valid',*COMMAND_COLUMNS}.issubset(c.dtype.names or []):
            raise ValueError('Command event log is missing receipt/sequence/command fields')
        validate_times(c['time_s'])
        seq=c['cmd_sequence']
        if not np.isfinite(seq).all() or np.any(seq!=np.floor(seq)) or np.any(np.diff(seq)!=1):
            raise ValueError('Command event sequence has gaps or duplicates')
        events=np.zeros(len(c),dtype=[(k,'f8') for k in required])
        for k in required:
            if k!='cmd_age':events[k]=c[k]
        # Each row is a command receipt, so its age is zero. Do not infer receipts
        # from TF snapshots, which can miss commands during measurement gaps.
        return events
    if not commands_only:
        required += ['x','y','z']
    if not set(required).issubset(c.dtype.names or []):
        raise ValueError('Controller CSV is missing required command/timing fields or legacy diagnostic XYZ')
    validate_times(c['time_s'])
    if commands_only:
        return c[required].copy()
    if not np.isfinite(np.column_stack([c[n] for n in ('x','y','z')])).all():
        raise ValueError('Nonfinite controller measured positions; review logger before alignment')
    return c


def import_round(root, source, number, *, notes='', parent=None, reference=None, provenance=None):
    root, source = Path(root), Path(source)
    if isinstance(number, bool) or int(number)!=number or number<0:
        raise ValueError('Round number must be a nonnegative integer')
    name=f'adaptation{int(number)}'
    destination=root/'data/adaptation_rounds'/name
    if destination.exists():
        raise ValueError(f'{name} already exists; originals cannot be replaced')
    if parent and not (root/'data/adaptation_rounds'/safe_name(parent)/'round.json').is_file():
        raise ValueError('Parent round does not exist')
    pairs=[]
    for controller in sorted(source.glob('experiment_*.csv')):
        trial=safe_name(controller.stem.removeprefix('experiment_'))
        tracking=source/(trial+'.csv')
        if not tracking.is_file():
            raise ValueError(f'Missing matching OptiTrack file: {tracking.name}')
        read_optitrack(tracking);read_controller(controller)
        pairs.append((trial,tracking,controller))
    if not pairs:
        raise ValueError('No experiment_<trial>.csv / <trial>.csv pairs in this folder')
    if reference:
        reference=Path(reference)
        if not reference.is_file(): raise ValueError('Reference CSV does not exist')
        r=np.genfromtxt(reference,delimiter=',',names=True)
        validate_times(r['time_s'])
        if not set(['px_m','py_m','pz_m','vx_m_s','vy_m_s','vz_m_s','ax_m_s2','ay_m_s2','az_m_s2']).issubset(r.dtype.names):
            raise ValueError('Reference must be a full-state trajectory CSV')
    destination.mkdir(parents=True)
    records=[]
    for trial, tracking, controller in pairs:
        folder=destination/'raw'/trial;folder.mkdir(parents=True)
        sources={}
        for kind, src in [('optitrack',tracking),('controller',controller)]:
            dst=folder/src.name;shutil.copy2(src,dst)
            sources[kind]=dict(path=dst.relative_to(destination).as_posix(),sha256=sha256_file(dst),original_name=src.name)
        identity=read_optitrack(tracking)['drone_label']
        write_json((folder/tracking.name).with_suffix('.tracking.json'),
                   dict(drone=identity,optitrack_sha256=sha256_file(tracking)))
        records.append(dict(trial_id=trial,sources=sources,drone_rigid_body=identity))
    ref_info=None
    if reference:
        folder=destination/'reference';folder.mkdir()
        dst=folder/reference.name;shutil.copy2(reference,dst)
        ref_info=dict(path=dst.relative_to(destination).as_posix(),sha256=sha256_file(dst),association='user supplied / prefix match only; not proof of flight firmware or complete execution')
    context=destination/'import_context';context.mkdir()
    for n in ('model','task','controller_export','baseline'):
        src=root/'config'/f'{n}.json'
        if src.is_file():shutil.copy2(src,context/src.name)
    write_json(destination/'round.json',dict(schema='aerial_whip_recording_round_v1',round=name,
        parent_round=parent,created_utc=utc_now(),notes=notes,provenance=provenance or {},
        model_adaptation_performed=False,policy_training_performed=False,
        execution='precomputed full-state reference tracked by vehicle; no onboard PPO',
        identity=dict(drone_by_trial={r['trial_id']:r['drone_rigid_body'] for r in records},cable=[f'cable1:c{i}' for i in range(1,11)],unlabeled='ignored'),
        import_context='Current workspace at import, NOT verified flight model/controller snapshots',
        reference=ref_info,trials=records))
    return destination


def align(m,c,override=None):
    ct=c['time_s'];cp=np.column_stack([c[n] for n in ('x','y','z')])
    good=np.isfinite(m['drone']).all(axis=1)
    t=m['time'][good];p=m['drone'][good]
    if len(t)<30 or np.max(np.ptp(p,axis=0))<.05:
        raise ValueError('Insufficient valid motion to identify an alignment')
    lo,hi=ct[0]-t[0],ct[-1]-t[-1]
    if hi<lo:raise ValueError('Controller log must cover the trimmed OptiTrack duration')
    def cost(offset, mask=None):
        tt,pp=(t,p) if mask is None else (t[mask],p[mask])
        fitted=np.column_stack([np.interp(tt+offset,ct,cp[:,j]) for j in range(3)])
        return float(np.mean(np.sum((fitted-pp)**2,axis=1)))
    if override is None:
        grid=np.linspace(lo,hi,max(2,int(np.ceil((hi-lo)/.01))+1))
        vals=np.array([cost(o) for o in grid]);best=grid[vals.argmin()]
        # Sample-and-hold logger positions create several local minima per tick.
        # A fine local scan is needed before bounded scalar refinement.
        fine=np.linspace(max(lo,best-.03),min(hi,best+.03),121)
        best=fine[np.argmin([cost(o) for o in fine])]
        result=minimize_scalar(cost,bounds=(max(lo,best-.0005),min(hi,best+.0005)),method='bounded')
        offset=float(result.x)
    else:
        offset=float(override)
        if not np.isfinite(offset) or not lo<=offset<=hi:raise ValueError('Offset leaves tracking outside controller coverage')
    halves=[]
    for mask in (np.arange(len(t))<len(t)//2,np.arange(len(t))>=len(t)//2):
        fit=minimize_scalar(lambda o:cost(o,mask),bounds=(max(lo,offset-.15),min(hi,offset+.15)),method='bounded')
        halves.append(float(fit.x))
    return dict(offset_s=offset,rms_m=cost(offset)**.5,half_offsets_s=halves,
        method='manual offset' if override is not None else 'measured XYZ constant-offset least squares; no spatial fit',
        clock_verified=False,includes_logging_latency=True)


def derivative(t,p):
    """Centered offline differences; do not bridge missing samples or time gaps."""
    out=np.full_like(p,np.nan)
    dt=np.diff(t);nominal=np.median(dt)
    valid=np.isfinite(p[:-2]).all(axis=-1)&np.isfinite(p[1:-1]).all(axis=-1)&np.isfinite(p[2:]).all(axis=-1)
    gaps=(dt[:-1]<1.5*nominal)&(dt[1:]<1.5*nominal)
    valid &= gaps.reshape((-1,)+(1,)*(valid.ndim-1))
    slope=(p[2:]-p[:-2])/(t[2:]-t[:-2]).reshape((-1,)+(1,)*(p.ndim-1))
    out[1:-1]=np.where(valid[...,None],slope,np.nan)
    return out


def preparation_enabled(directory):
    """A later explicit preparation decision can supersede an old archive flag.

    It does not alter fit roles, old study splits, or the archive record itself.
    """
    directory=Path(directory)
    if not (directory/'ARCHIVED.json').exists():
        return True
    path=directory/'PREPARATION_AUTHORIZATION.json'
    if not path.is_file():
        return False
    value=json.loads(path.read_text())
    return value.get('schema')=='recording_preparation_authorization_v1' and value.get('allow_preparation') is True


def process_round(directory, overrides=None):
    if not preparation_enabled(directory):
        raise ValueError('This recording round is archived and excluded from the active adaptation workflow.')
    directory=Path(directory);meta=json.loads((directory/'round.json').read_text())
    output=directory/'processed'/stamp();output.mkdir(parents=True)
    shutil.copy2(__file__,output/'processor_snapshot.py')
    shutil.copy2(legacy_execution.__file__,output/'legacy_execution_snapshot.py')
    from simulator import geometry
    shutil.copy2(geometry.__file__,output/'geometry_snapshot.py')
    profile=legacy_execution.load_profile(directory,output)
    authorization=directory/'PREPARATION_AUTHORIZATION.json'
    if authorization.is_file():shutil.copy2(authorization,output/authorization.name)
    reports=[]
    reference=None
    if meta.get('reference'):
        ref=directory/meta['reference']['path']
        if sha256_file(ref)!=meta['reference']['sha256']:raise ValueError('Reference hash changed')
        reference=np.genfromtxt(ref,delimiter=',',names=True)
    for trial in meta['trials']:
        name=safe_name(trial['trial_id'])
        try:
            files={}
            for kind,info in trial['sources'].items():
                p=(directory/info['path']).resolve()
                if not p.is_relative_to(directory.resolve()) or sha256_file(p)!=info['sha256']:
                    raise ValueError('Raw-file path or SHA-256 check failed')
                files[kind]=p
            m=read_optitrack(files['optitrack']);c=read_controller(files['controller'])
            sync=align(m,c,(overrides or {}).get(name));t=m['time']+sync['offset_s'];ct=c['time_s']
            raw_idx=np.searchsorted(ct,t,side='right')-1
            idx=np.clip(raw_idx,0,len(ct)-1)
            vals=np.column_stack([c[n] for n in COMMAND_COLUMNS])
            valid_c=(c['cmd_valid']==1)&np.isfinite(vals).all(axis=1)&np.isfinite(c['cmd_age'])&(c['cmd_age']>=0)&(c['cmd_age']<=.1)
            cmd_age=t-ct[idx]+c['cmd_age'][idx]
            valid=valid_c[idx]&(cmd_age>=0)&(cmd_age<=.1)&(raw_idx>=0)&(t<=ct[-1])
            aligned=vals[idx].copy();aligned[~valid]=np.nan
            active=valid_c&(c['cmd_age']<=.1)&(np.linalg.norm(np.nan_to_num(vals[:,3:9]),axis=1)>1e-3)
            ai=np.flatnonzero(active)
            # Preserve all segments, rather than treating arbitrary motions as one strike.
            segments=[]
            if len(ai):
                starts=np.r_[ai[0],ai[1:][np.diff(ai)>1]];ends=np.r_[ai[:-1][np.diff(ai)>1],ai[-1]]
                for a,b in zip(starts,ends):
                    segments.append(dict(start_s=float(ct[a]),end_s=float(ct[b+1] if b+1<len(ct) else ct[b]),reviewed=False))
            unique=vals[active,:9]
            if len(unique):unique=unique[np.r_[True,np.any(np.diff(unique,axis=0)!=0,axis=1)]]
            ref_match=None
            if reference is not None and len(unique):
                refvalues=np.column_stack([reference[n] for n in ('px_m','py_m','pz_m','vx_m_s','vy_m_s','vz_m_s','ax_m_s2','ay_m_s2','az_m_s2')])
                match=len(unique)<=len(refvalues) and np.allclose(unique,refvalues[:len(unique)],rtol=0,atol=1e-10)
                ref_match=dict(prefix_matches=bool(match),recorded_distinct_samples=len(unique),reference_samples=len(refvalues),
                    absent_tail_times_s=reference['time_s'][len(unique):].tolist() if match else None)
            _,qvalid=normalized_rotations_xyzw(m['quaternion'])
            dvalid=np.isfinite(m['drone']).all(axis=1)
            cvalid=np.isfinite(m['cable']).all(axis=2)
            phase_report=None;extra={}
            if profile:
                native_phase,native_index,phase_report=legacy_execution.classify_execution(ct,vals,valid_c,c['cmd_age'],profile)
                phase=native_phase[idx].copy();phase[~valid]=legacy_execution.UNKNOWN
                sample_index=native_index[idx].copy();sample_index[~valid]=-1
                onset=phase_report['csv_start_s']
                boundary=np.zeros(len(t),dtype=bool)
                transitions=np.flatnonzero(native_phase[1:]!=native_phase[:-1])+1
                for i in transitions:boundary|=(t>=ct[i-1])&(t<=ct[i])
                attachment,attachment_valid=attachment_positions(m['drone'],m['quaternion'],profile['geometry']['offset_tracking_m'])
                extra=dict(execution_phase=phase,csv_sample_index=sample_index,time_from_csv_onset_s=t-onset,
                    csv_maneuver_mask=phase==legacy_execution.CSV,
                    non_csv_fullstate_mask=valid&(phase!=legacy_execution.CSV),
                    phase_boundary_uncertain=boundary,
                    attachment_position_m=attachment,attachment_valid=attachment_valid,
                    attachment_velocity_m_s=derivative(m['time'],attachment),
                    controller_native_execution_phase=native_phase,controller_native_csv_sample_index=native_index,
                    controller_native_time_from_csv_onset_s=ct-onset)
                phase_report['tracking_coverage_controller_s']=[float(t[0]),float(t[-1])]
                phase_report['tracking_phase_counts']={str(p):int(np.sum(phase==p)) for p in np.unique(phase)}
                phase_report['tracking_boundary_uncertain_frames']=int(boundary.sum())
                phase_report['tracking_covers_complete_csv']=bool(t[0]<=onset and t[-1]>=phase_report['csv_end_s'])
                phase_report['boundary_uncertainty']='Previous-to-first logger-row bracket around phase transitions only; excludes unknown clock/measurement latency.'
                segments=[dict(start_s=onset,end_s=phase_report['csv_end_s'],reviewed=False,
                               source='Exact supplied controller sequence matched in logged order')]
                if reference is not None:
                    refvalues=np.column_stack([reference[n] for n in ('px_m','py_m','pz_m','vx_m_s','vy_m_s','vz_m_s','ax_m_s2','ay_m_s2','az_m_s2')])
                    observed=np.asarray(profile['sequence_fullstate'])[:,:9]
                    count=len(observed)
                    match=count<=len(refvalues) and np.allclose(observed,refvalues[:count],rtol=0,atol=1e-10)
                    ref_match=dict(prefix_matches=bool(match),recorded_distinct_samples=count,reference_samples=len(refvalues),
                        absent_tail_times_s=reference['time_s'][count:].tolist() if match else None,
                        method='All CSV row P/V/A values matched to supplied controller and log, including zero V/A rows')
            folder=output/name;folder.mkdir()
            np.savez_compressed(folder/'dataset.npz',optitrack_time_s=m['time'],controller_time_s=t,
                frame=m['frame'],drone_position_m=m['drone'],drone_quaternion_xyzw=m['quaternion'],
                drone_position_valid=dvalid,drone_orientation_valid=qvalid,cable_position_m=m['cable'],cable_valid=cvalid,
                drone_velocity_m_s=derivative(m['time'],m['drone']),cable_velocity_m_s=derivative(m['time'],m['cable']),
                reference_fullstate=aligned,reference_valid=valid,reference_age_s=cmd_age,
                controller_native_time_s=ct,controller_native_fullstate=vals,controller_native_valid=valid_c,
                controller_native_command_age_s=c['cmd_age'],controller_native_cmd_valid=c['cmd_valid'],
                controller_native_logged_columns=np.array(c.dtype.names),
                controller_native_logged_values=np.column_stack([c[n] for n in c.dtype.names]),
                controller_native_position_m=np.column_stack([c[n] for n in ('x','y','z')]),
                cable_names=np.array(meta['identity']['cable']),reference_columns=np.array(COMMAND_COLUMNS),**extra)
            if phase_report:
                write_json(folder/'execution_phases.json',phase_report)
                with (folder/'phases.csv').open('x',newline='') as stream:
                    w=csv.writer(stream);w.writerow(['optitrack_time_s','controller_time_s','time_from_csv_onset_s','phase','csv_sample_index','command_valid','phase_boundary_uncertain'])
                    w.writerows(zip(m['time'],t,t-onset,phase,sample_index,valid.astype(int),boundary.astype(int)))
                with (folder/'controller_phases.csv').open('x',newline='') as stream:
                    w=csv.writer(stream);w.writerow(['controller_time_s','time_from_csv_onset_s','phase','csv_sample_index','command_fresh'])
                    w.writerows(zip(ct,ct-onset,native_phase,native_index,valid_c.astype(int)))
            with (folder/'tracking.csv').open('x',newline='') as stream:
                w=csv.writer(stream);w.writerow(['optitrack_time_s','controller_time_s',
                    'drone_x','drone_y','drone_z','qx','qy','qz','qw','drone_position_valid','drone_orientation_valid']+
                    [f'c{k}_{f}' for k in range(1,11) for f in ('x','y','z','valid')])
                for j in range(len(t)):
                    cable=np.column_stack([m['cable'][j],cvalid[j]]).ravel()
                    w.writerow([m['time'][j],t[j],*m['drone'][j],*m['quaternion'][j],int(dvalid[j]),int(qvalid[j]),*cable])
            with (folder/'fullstate_aligned.csv').open('x',newline='') as stream:
                w=csv.writer(stream);w.writerow(['controller_time_s',*COMMAND_COLUMNS,'valid','age_s'])
                for j in range(len(t)):w.writerow([t[j],*aligned[j],int(valid[j]),cmd_age[j]])
            changed=np.r_[False,np.any(np.diff(np.column_stack([c[n] for n in ('x','y','z')]),axis=0)!=0,axis=1)]
            updates=ct[changed];hz=float(1/np.median(np.diff(updates))) if len(updates)>2 else None
            report=dict(trial_id=name,status='processed — review required',alignment=sync,
                tracking_frames=len(t),missing_cable_samples_per_marker=(~cvalid).sum(axis=0).tolist(),
                missing_drone_positions=int((~dvalid).sum()),logged_position_update_hz=hz,
                candidate_motion_intervals=segments,reference_comparison=ref_match,
                execution_phases=phase_report,
                geometry=profile['geometry'] if profile else None,
                training_ready=False,role='unassigned',outcome='unreviewed',
                required_review=['time alignment','marker direction and attachment offset','contact/intervention interval','outcome and dataset split'],
                conventions=dict(position='source global meters; no frame transform',quaternion='source OptiTrack XYZW; mapping to body frame unverified',
                    cable='c1..c10 numerically ordered; physical endpoint mapping unverified',
                    velocity='offline centered difference; NaN at gaps and endpoints; not controller velocity',
                    reference='Original kinematic full-state commands, not forces; causal hold with 100 ms age limit; no retrospective position offset',
                    attachment='Derived measured attachment uses archived geometry and measured orientation; commands remain at cf_7 origin' if profile else 'Not reconstructed'),
                output_files={p.name:sha256_file(p) for p in folder.iterdir() if p.is_file()})
            write_json(folder/'quality.json',report)
            plot_review(folder,m,t,c,segments,phase_report)
            reports.append(report)
        except Exception as error:
            reports.append(dict(trial_id=name,status='failed',error=str(error),training_ready=False))
    write_json(output/'processing.json',dict(schema='aerial_whip_processed_round_v2' if profile else 'aerial_whip_processed_round_v1',created_utc=utc_now(),
        round=meta['round'],round_manifest_sha256=sha256_file(directory/'round.json'),
        execution_profile_sha256=sha256_file(output/'execution_profile.json') if profile else None,
        source_snapshots={p.name:sha256_file(p) for p in output.iterdir() if p.is_file()},
        processor_sha256=sha256_file(output/'processor_snapshot.py'),reports=reports))
    return output


def plot_review(folder,m,t,c,segments,phase_report=None):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    fig=Figure(figsize=(10,9) if phase_report else (10,7))
    if phase_report:
        grid=fig.add_gridspec(3,2,height_ratios=[1,1,.65])
        axes=np.array([[fig.add_subplot(grid[i,j]) for j in range(2)] for i in range(2)])
    else:axes=fig.subplots(2,2)
    for j,n in enumerate(('x','y','z')):
        ax=axes.flat[j];ax.plot(t,m['drone'][:,j],label='OptiTrack')
        ax.plot(c['time_s'],c[n],label='Controller measurement',alpha=.65)
        good=(c['cmd_valid']==1)&(c['cmd_age']>=0)&(c['cmd_age']<=.1)
        desired=np.where(good,c['cmd_'+n],np.nan)
        ax.step(c['time_s'],desired,where='post',label='Logged desired',alpha=.7)
        ax.set_ylabel(n+' (m)');ax.set_xlabel('Controller log time (s)');ax.grid(alpha=.2)
        ax.set_xlim(t[0],t[-1])
        if phase_report:
            colors={legacy_execution.PRE:'#489bd1',legacy_execution.CSV:'#ed8b24',legacy_execution.POST:'#69ab68',
                    legacy_execution.UNKNOWN:'#999999',legacy_execution.OTHER:'#b071c3'}
            for seg in phase_report['intervals']:
                ax.axvspan(seg['start_s'],seg['end_s'],color=colors[seg['phase']],alpha=.12)
        else:
            for seg in segments:ax.axvspan(seg['start_s'],seg['end_s'],color='orange',alpha=.12)
    ax=axes.flat[3]
    ax.imshow(np.isfinite(m['cable']).all(axis=2).T,aspect='auto',interpolation='nearest',vmin=0,vmax=1,
              extent=[t[0],t[-1],10.5,.5],cmap='RdYlGn')
    ax.set_ylabel('Cable marker number');ax.set_xlabel('Controller log time (s)');ax.set_title('Marker availability (green = present)')
    axes.flat[0].legend(fontsize=8)
    if phase_report:
        ax=fig.add_subplot(grid[2,:])
        ax.plot(c['time_s'],c['z'],label='Controller measured Z',alpha=.7)
        ax.plot(t,m['drone'][:,2],label='Available OptiTrack Z')
        ax.step(c['time_s'],np.where(good,c['cmd_z'],np.nan),where='post',label='Logged desired Z',alpha=.7)
        for seg in phase_report['intervals']:
            ax.axvspan(seg['start_s'],seg['end_s'],color=colors[seg['phase']],alpha=.12)
        ax.set_xlim(c['time_s'][0],c['time_s'][-1]);ax.set_xlabel('Entire controller recording (s)')
        ax.set_ylabel('Z (m)');ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper right')
    title=' — blue: pre-hold; orange: CSV; green: post-hold; gray: unknown command' if phase_report else ' — alignment and data quality'
    fig.suptitle(folder.name+title,fontsize=10);fig.tight_layout()
    fig.savefig(folder/'review.png',dpi=130)


def save_review(directory, version, trial, *, role, outcome, notes, offset_verified=False,
                precontact_start_s=None,precontact_end_s=None):
    directory=Path(directory);safe_name(trial);safe_name(version)
    if not (directory/'processed'/version/trial/'quality.json').is_file():raise ValueError('Select a processed trial')
    if role not in ('unassigned','adaptation','validation'):raise ValueError('Unsupported role')
    if outcome not in ('unreviewed','success','failure','aborted'):raise ValueError('Unsupported outcome')
    if (precontact_start_s is None)!=(precontact_end_s is None):raise ValueError('Supply both interval endpoints or neither')
    if precontact_start_s is not None:
        t=np.load(directory/'processed'/version/trial/'dataset.npz')['controller_time_s']
        if not t[0]<=precontact_start_s<precontact_end_s<=t[-1]:raise ValueError('Review interval must be inside the tracking recording')
    folder=directory/'reviews';folder.mkdir(exist_ok=True)
    path=folder/(stamp()+f'-{perf_counter_ns():020d}-'+uuid4().hex[:8]+'.json')
    write_json(path,dict(processed_version=version,trial_id=trial,role=role,outcome=outcome,notes=notes,
        alignment_reviewed=bool(offset_verified),precontact_start_s=precontact_start_s,precontact_end_s=precontact_end_s,
        created_utc=utc_now(),training_ready=False))
    return path
