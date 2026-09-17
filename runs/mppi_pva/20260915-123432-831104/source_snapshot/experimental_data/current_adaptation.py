"""Current complete-CSV adp0 protocol; independent of legacy whip phase rules.

Raw measurements and original flight predictions are never modified. Each
trial uses its own fixed measured-stream alignment and causal initialization.
"""
from copy import deepcopy
from dataclasses import replace, asdict
from pathlib import Path
import json
import shutil
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from .adaptation_check import load_comparison, flight_names, interpolate_positions
from .adaptation_rounds import read_optitrack, read_controller, COMMAND_COLUMNS
from .io import atomic_json, sha256_file
from .nominal_pose_fit import GAIN_NAMES, parameters, integration_steps, linear_prediction
from .state_initialization import project_state
from simulator.cable import CableConfiguration, DderModel, DderState
from simulator.drone_pose_response import initialize_from_hover, CommandSchedule
from simulator.geometry import attachment_positions, normalized_rotations_xyzw

ROOT=Path(__file__).resolve().parents[1]
BATCH=ROOT/'rehearsal_csv_and_result_in_real_flight/20260908-195207-486249-seed655_best_validation/adp0'
SOURCE=ROOT/'data/model_candidates/20260908-smooth-differentiable/model.json'
JOB=ROOT/'runs/adaptation/20260908-adp0-first'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def finite_json(value):
    if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite_json(v) for v in value]
    if isinstance(value,np.ndarray):return finite_json(value.tolist())
    if isinstance(value,np.generic):return finite_json(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def save(path,value):
    atomic_json(Path(path),finite_json(value))


def phase_weights(times, valid):
    """Equal trial weight; 80% whip, 20% early recovery when both are present."""
    w=np.zeros(len(times))
    for lo,hi,weight in [(0.,1.,.8),(1.,3.,.2)]:
        mask=valid&(times>=lo)&(times<hi if lo==0 else times<=hi)
        if mask.any():w[mask]=weight/mask.sum()
    if w.sum()==0:raise ValueError('No eligible samples in fixed fit phases')
    return w/w.sum()


def packet_receipts(c, reference, onset):
    """Use exact matching moving packets and their observed receipt timestamps.

    Initial/final identical holds cannot be distinguished; retain nominal CSV
    timing for those, documenting the ambiguity instead of fitting time shifts.
    """
    values=np.column_stack([reference[n] for n in reference.dtype.names[1:]])
    commands=np.column_stack([c[n] for n in COMMAND_COLUMNS])
    good=np.isfinite(commands).all(1)&np.isfinite(c['cmd_age'])&(c['cmd_valid']>.5)
    distance,ids=cKDTree(values).query(commands[good])
    receipt=(c['time_s']-c['cmd_age'])[good]-onset
    dynamic=np.linalg.norm(values[:,3:9],axis=1)>1e-5
    times=reference['time_s'].copy()
    for j in np.flatnonzero(dynamic):
        mask=(distance<1e-8)&(ids==j)
        if not mask.any():raise ValueError(f'Missing dynamic command {j}')
        times[j]=np.median(receipt[mask])
    if np.any(np.diff(times)<=0):raise ValueError('Recorded command times are not monotone')
    return times,values


def causal_history_indices(times,cutoff,history_s):
    """Cover the requested past duration, ending strictly before the cutoff."""
    times=np.asarray(times)
    if not np.isfinite(history_s) or history_s<=0:raise ValueError('Positive cable history required')
    end=int(np.searchsorted(times,cutoff,side='left'))-1
    if end<0:raise ValueError('Missing causal cable history')
    start=int(np.searchsorted(times,times[end]-history_s+1e-9,side='right'))-1
    if start<0 or end-start<10 or times[end]-times[start]<history_s-1e-8:
        raise ValueError('Missing causal cable history')
    ids=np.arange(start,end+1);dt=np.diff(times[ids])
    if not np.isfinite(times[ids]).all() or np.any(dt<=0) or dt.max()>1.5*np.median(dt):
        raise ValueError('Missing causal cable history')
    return ids


def endpoint_velocity(times,values,weight_tau_s=None):
    """Causal quadratic endpoint derivative, optionally emphasizing recent data."""
    times=np.asarray(times);values=np.asarray(values)
    duration=times[-1]-times[0];x=(times-times[-1])/duration
    design=np.c_[np.ones(len(x)),x,x*x]
    if weight_tau_s is None:weights=np.ones(len(x))
    else:
        if not np.isfinite(weight_tau_s) or weight_tau_s<=0:raise ValueError('Positive velocity weight time constant required')
        weights=np.exp((times-times[-1])/(2*weight_tau_s))
    row=(np.linalg.pinv(design*weights[:,None])[1]*weights)/duration
    return np.einsum('t,t...->...',row,values)


def nodes_from_sites(sites, cable):
    parts=[sites[...,:1,:]]
    for i,n in enumerate(cable.interval_subdivisions):
        for j in range(1,n+1):
            parts.append((sites[...,i:i+1,:]*(1-j/n)+sites[...,i+1:i+2,:]*(j/n)))
    return np.concatenate(parts,axis=-2)


def observed_cable_history(data, ids, cable, minimum_fraction=None):
    """Keep missing history observations masked; require a fully observed endpoint.

    The default preserves the original strict history rule. An explicitly
    reviewed partial-history policy permits regression on observed samples,
    independently for each node, without filling missing positions.
    """
    sites=data['sites'][ids].copy()
    if len(ids)<11 or not data['pose_valid'][ids].all():
        raise ValueError('Missing causal cable history')
    sites[:,1:][~data['marker_valid'][ids]]=np.nan
    nodes=nodes_from_sites(sites,cable)
    valid=np.isfinite(nodes).all(-1)
    if minimum_fraction is None:
        if not valid.all():raise ValueError('Missing causal cable history')
    else:
        if not .8<=minimum_fraction<=1.:raise ValueError('Invalid history observation fraction')
        if not valid[-1].all():raise ValueError('Missing causal cable endpoint; no extrapolated position allowed')
        if np.any(valid.mean(0)<minimum_fraction) or np.any(valid.sum(0)<11):
            raise ValueError('Insufficient observed causal cable history')
    return nodes,valid


def observed_endpoint_velocity(times, nodes, valid, weight_tau_s=None):
    if valid.all():return endpoint_velocity(times,nodes,weight_tau_s)
    # Missing coordinates do not enter any least-squares term. The last
    # position/time is still observed and identical for every node.
    return np.stack([endpoint_velocity(times[valid[:,i]],nodes[valid[:,i],i],weight_tau_s)
                     for i in range(nodes.shape[1])])


def prepare(job=JOB,batch=BATCH,source=SOURCE):
    job=Path(job);batch=Path(batch);source=Path(source)
    if job.exists():raise FileExistsError('Adaptation job already exists; use its frozen inputs')
    from .hover_calibration import load_calibration
    if load_calibration(batch) is None:
        raise ValueError('Calibrate batch hover Z before preparing a normalized model fit.')
    job.mkdir(parents=True)
    model=read(source);cable=CableConfiguration.from_mapping(model['cable'])
    protocol=dict(schema='complete_csv_adp0_adaptation_v1',seed=20260908,
        source_model=str(source),source_sha256=sha256_file(source),batch=str(batch),
        training='five current complete-CSV flights only; old data enters only through the M0 prior',
        heldout='leave one whole flight out; each fold excludes it from all parameter/checkpoint selection',
        phases={'whip':[0,1],'early_recovery':[1,3],'late_recovery_hold':[3,11.2]},
        fit_phase_weights={'whip':.8,'early_recovery':.2},
        geometry_and_mass='frozen; no separate fixed drag; no target loss or time-shift fitting',
        initialization='last 21 native pose/cable samples before CSV onset; initial time is last sample, no extrapolation',
        missingness='finite adjacent interpolation only; native jump masks; no gap bridging',
        cable_fit='measured rotated attachment; short-window BPTT plus complete-whip training-only selection',
        combined_validation='predicted attachment only after initialization; saved original forecast reported separately',
        contact='user confirmed all five free flight; virtual target only',
        drone_updates=80,cable_updates=24,cable_learning_rate=.002,
        cable_window_s=.12,cable_window_starts_s=[0,.15,.30,.45,.60,.75,.9,1.2,1.6,2.0,2.4],
        model_selected=False,policy_training=False,prospective_validation=False)
    from .hover_calibration import load_calibration
    calibration=load_calibration(batch)
    if calibration is not None:
        protocol['vertical_calibration']=calibration
        protocol['heldout']='Maneuver fitting excludes each held-out flight; shared batch nuisance Z calibration uses pre/post holds from all takes. Conditional retrospective diagnostic, not fully independent or prospective validation.'
        protocol['vertical_convention']='Subtract batch hover bias once from measured drone/cable Z before initialization/fitting; commands unchanged. No second subtraction in residual or rollout.'
    protected={}
    for base in ['config','runs/ppo','runs/rehearsals','runs/cem','policies',str(batch.relative_to(ROOT)),str(source.parent.relative_to(ROOT))]:
        for p in (ROOT/base).rglob('*'):
            if p.is_file():protected[str(p.resolve())]=sha256_file(p)
    save(job/'protected_before.json',protected)
    save(job/'protocol.json',protocol)
    shutil.copytree(source.parent,job/'source_candidate')
    reports={}
    for name in flight_names(batch):
        comp=load_comparison(ROOT,batch,name)
        m=read_optitrack(batch/'flight_take'/f'{name}.csv')
        from .hover_calibration import load_calibration,corrected_tracking
        height_calibration=load_calibration(batch,name)
        raw_position=m['drone'].copy();raw_cable=m['cable'].copy()
        if height_calibration is not None:
            m=corrected_tracking(m,height_calibration['bias_z_m'])
        c=read_controller(batch/'flight_take'/f'experiment_{name}.csv', commands_only=True)
        ref=np.genfromtxt(batch/'simulation_csv/fullstate_30hz.csv',delimiter=',',names=True)
        t=m['time']+comp['alignment']['offset_s']-comp['onset']
        r,rv=normalized_rotations_xyzw(m['quaternion'])
        a,av=attachment_positions(m['drone'],m['quaternion'],model['recorded_data']['optitrack_to_attachment_offset_body_m'])
        sites=np.concatenate([a[:,None],m['cable']],1)
        pre=np.flatnonzero(t<0)[-21:]
        if len(pre)!=21 or np.max(np.diff(t[pre]))>.015 or not np.isfinite(sites[pre]).all() or not rv[pre].all():
            raise ValueError(name+': insufficient clean causal initialization history')
        # Verify the hold using actual controller packets, not the desired CSV.
        ci=np.searchsorted(c['time_s'],t[pre]+comp['onset'],side='right')-1
        hc=np.column_stack([c[n][ci] for n in COMMAND_COLUMNS])
        if not np.isfinite(hc).all() or not np.allclose(hc,hc[:1],atol=1e-8,rtol=0) or np.max(np.abs(hc[:,3:9]))>1e-8:
            raise ValueError(name+': pre-onset interval was not a constant hold')
        pt,packets=packet_receipts(c,ref,comp['onset'])
        pvalid=np.isfinite(m['drone']).all(1)
        pjump=np.linalg.norm(np.diff(m['drone'],axis=0),axis=1)>.1
        pvalid[:-1]&=~pjump;pvalid[1:]&=~pjump
        marker_valid=np.isfinite(m['cable']).all(-1)
        jumps=np.linalg.norm(np.diff(m['cable'],axis=0),axis=-1)>.15
        marker_valid[:-1]&=~jumps;marker_valid[1:]&=~jumps
        # Keep raw arrays and explicit masks; never silently move measured sites.
        folder=job/'inputs'/name;folder.mkdir(parents=True)
        np.savez_compressed(folder/'data.npz',time=t,position=m['drone'],rotation=r,sites=sites,
            raw_optitrack_position=raw_position,raw_optitrack_cable=raw_cable,
            pose_valid=pvalid&rv,marker_valid=marker_valid,pre_indices=pre,
            packet_time=pt,packets=packets,hover_commands=hc,
            saved_time=comp['time'],saved_origin=comp['predicted_origin'],saved_cable=comp['predicted_cable'])
        whip=(t>=0)&(t<=1)
        lengths=np.linalg.norm(np.diff(sites,axis=1),axis=-1)
        reports[name]=dict(measured_state_source=comp['measured_state_source'],controller_source=comp['controller_source'],
            vertical_calibration=height_calibration,
            alignment=comp['alignment'],onset_controller_s=comp['onset'],hashes=comp['hashes'],
            span_s=[t[0],t[-1]],initial_time_s=t[pre[-1]],history_s=t[pre[-1]]-t[pre[0]],
            whip_pose_samples=int((whip&pvalid&rv).sum()),whip_marker_valid_counts=marker_valid[whip].sum(0),
            interval_length_median_m=np.nanmedian(lengths[whip],axis=0),
            packet_jitter_span_s=comp['packet_jitter_s'],exact_dynamic_packets=comp['packet_count'])
        save(folder/'provenance.json',reports[name])
    save(job/'preparation.json',reports)
    return job


class Trial:
    """Explicit current-flight adapter for shared nominal recurrence helpers."""
    def __init__(self,job,name,model,*,end=3.,device='cuda'):
        self.name=name;self.device=device;self.model=model
        with np.load(Path(job)/'inputs'/name/'data.npz') as z:self.data={k:z[k].copy() for k in z.files}
        d=self.data;pre=d['pre_indices'];self.hover_time=d['time'][pre]
        self.hover=(d['position'][pre][None],d['rotation'][pre][None],d['hover_commands'][None])
        self.offset=model['recorded_data']['optitrack_to_attachment_offset_body_m']
        self.time=d['time'][(d['time']>=self.hover_time[-1])&(d['time']<=end)]
        ids=np.searchsorted(d['time'],self.time)
        self.truth=dict(position_origin_m=d['position'][ids],rotation_tracking_to_world=d['rotation'][ids])
        self.masks={'fit_position':d['pose_valid'][ids]&(self.time>=0),'fit_orientation':d['pose_valid'][ids]&(self.time>=0)}
        self.weights=phase_weights(self.time,self.masks['fit_position'])
        self.context={'csv_end_s':float(end)}
        ptime=np.r_[-2.,d['packet_time']]
        values=np.concatenate([d['hover_commands'][:1],d['packets']],axis=0)
        self.schedule=CommandSchedule(ptime,torch.tensor(values[None],dtype=torch.float64),coverage_end_s=12.)
        self.p0=self.hover[0][0,-1].copy()
        zero=np.array([0,0,0,0,1,1.]);s=self.state(zero,'cpu')
        self.v0=s.velocity[0].numpy();self.b0=s.compensation[0].numpy();self.basis=np.zeros((3,6))
        for j in range(4):
            x=zero.copy();x[j]=1
            self.basis[:,j]=self.state(x,'cpu').compensation[0].numpy()-self.b0
        self.plans={}

    def state(self,gains,device,tau=.08,delay=.02,attitude_scales=(1.,1.)):
        h=[torch.as_tensor(a,dtype=torch.float64,device=device) for a in self.hover]
        return initialize_from_hover(self.hover_time,*h,parameters(gains,tau,delay,attitude_scales),alignment_mode='prehover_effective_alignment')[0]

    def initial_pose(self,params):
        return self.state([getattr(params,k) for k in GAIN_NAMES],self.device,params.attitude_time_constant_s,
            params.delay_s,(params.attitude_acceleration_scale_xy,params.attitude_acceleration_scale_z))

    def linear(self,gains,delay):
        if delay not in self.plans:self.plans[delay]=integration_steps(self.schedule,self.time,delay)
        return linear_prediction(gains,self.p0,self.v0,self.b0,self.basis,self.plans[delay])

    def grid(self,end=1.):
        dt=self.model['simulation']['dt_s'];start=self.hover_time[-1]
        return start+np.arange(int(np.ceil((end-start)/dt))+1)*dt

    def measured(self,times):
        d=self.data
        good=d['pose_valid'];r=np.full((len(times),3,3),np.nan)
        # Only adjacent finite samples may supply an orientation interpolation.
        valid=np.isfinite(interpolate_positions(d['time'],np.where(good[:,None],d['position'],np.nan),times)).all(-1)
        if valid.any():r[valid]=Slerp(d['time'][good],Rotation.from_matrix(d['rotation'][good]))(times[valid]).as_matrix()
        p=interpolate_positions(d['time'],np.where(good[:,None],d['position'],np.nan),times)
        sites=d['sites'].copy();sites[:,1:][~d['marker_valid']]=np.nan
        site=interpolate_positions(d['time'],sites,times)
        # Recompute root from interpolated rigid pose, never interpolate the offset in world axes.
        site[:,0]=p+np.einsum('tij,j->ti',r,self.offset)
        return p,r,site

    def cable_state(self,physics,*,cutoff=None,history_s=None,velocity_weight_tau_s=None,
                    minimum_history_observation_fraction=None):
        d=self.data
        if history_s is None:ids=d['pre_indices'] if cutoff is None else np.flatnonzero(d['time']<cutoff)[-11:]
        else:
            end=float(d['time'][d['pre_indices'][-1]])
            ids=causal_history_indices(d['time'],np.nextafter(end,np.inf) if cutoff is None else cutoff,history_s)
            if not d['pose_valid'][ids].all():raise ValueError('Missing causal cable history')
        t=d['time'][ids]
        nodes,valid=observed_cable_history(d,ids,CableConfiguration.from_mapping(self.model['cable']),
                                          minimum_history_observation_fraction)
        q=torch.tensor(nodes[-1: ],dtype=torch.float64,device=self.device)
        v=torch.tensor(observed_endpoint_velocity(t,nodes,valid,velocity_weight_tau_s)[None],dtype=q.dtype,device=q.device)
        with torch.no_grad():state=project_state(physics,q,v)
        return state,float(t[-1]),float((state.positions_m-q).abs().max())
