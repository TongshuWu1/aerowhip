"""Source-preserving preparation of arbitrary-duration PVA preliminary motions."""
from pathlib import Path
from copy import deepcopy
import shutil
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation
from .adaptation_rounds import read_controller,read_optitrack,COMMAND_COLUMNS
from .current_adaptation import Trial,read,save
from .nominal_pose_fit import parameters,integration_steps,linear_prediction
from .io import sha256_file
from simulator.geometry import attachment_positions,normalized_rotations_xyzw
from simulator.drone_pose_response import PoseResponseState,CommandSchedule,command_attitude


def measured_clock_alignment(m,c):
    """Match two recorded position streams; never use desired command motion."""
    xyz=np.column_stack([c[k] for k in ('x','y','z')])
    changed=np.r_[False,(np.diff(xyz,axis=0)!=0).any(1)];ids=np.flatnonzero(changed)
    dist,nearest=cKDTree(m['drone']).query(xyz[ids]);close=dist<.003
    delta=c['time_s'][ids[close]]-m['time'][nearest[close]]
    bins=np.round(delta/.05)*.05;values,counts=np.unique(bins,return_counts=True)
    if len(counts)==0:raise ValueError('No measured-stream timing correspondences')
    guess=values[np.argmax(counts)];ids=ids[close][abs(delta-guess)<.08]
    if len(ids)<30:raise ValueError('Insufficient measured-stream clock correspondences')
    def score(offset,subset=ids):
        t=c['time_s'][subset]-offset
        pred=np.stack([np.interp(t,m['time'],m['drone'][:,j]) for j in range(3)],1)
        return float(np.mean(np.sum((pred-xyz[subset])**2,axis=1)))
    fit=minimize_scalar(score,bounds=(guess-.04,guess+.04),method='bounded',options={'xatol':1e-9})
    chunks=[]
    for part in np.array_split(ids,4):
        f=minimize_scalar(lambda x:score(x,part),bounds=(fit.x-.04,fit.x+.04),method='bounded')
        chunks.append(dict(offset_s=float(f.x),rmse_m=float(np.sqrt(f.fun)),samples=len(part)))
    if np.sqrt(fit.fun)>.01:raise ValueError('Measured streams do not agree within 1 cm')
    return dict(offset_s=float(fit.x),rmse_m=float(np.sqrt(fit.fun)),matched_updates=len(ids),chunks=chunks,
        convention='controller time = OptiTrack time + offset',method='Measured OptiTrack XYZ matched to controller-cached OptiTrack XYZ',
        limitation='Estimated stream alignment; cache/transport latency is not independently known. Fitted command delay is effective, not measured actuator latency.')


def recorded_packets(c):
    values=np.column_stack([c[k] for k in COMMAND_COLUMNS]);receipt=c['time_s']-c['cmd_age']
    good=(c['cmd_valid']>.5)&np.isfinite(values).all(1)&np.isfinite(receipt)&(c['cmd_age']>=0)&(c['cmd_age']<=.2)
    ids=np.flatnonzero(good)
    # The same cached receipt can differ by clock roundoff in adjacent log rows.
    split=(np.diff(receipt[ids])>.001)|(np.diff(values[ids],axis=0)!=0).any(1)
    starts=np.flatnonzero(np.r_[True,split]);ends=np.r_[starts[1:],len(ids)]
    times=np.array([np.median(receipt[ids[a:b]]) for a,b in zip(starts,ends)])
    knots=values[ids[starts]]
    if len(times)<3 or np.any(np.diff(times)<=0):raise ValueError('Nonmonotone command receipts')
    end=float(c['time_s'][-1]);valid_until=np.minimum(np.r_[times[1:],end],times+.2)
    return times,knots,valid_until,end


def prepare(job,root,*,cable_velocity_weight_tau_s=None):
    root=Path(root).resolve();job=Path(job).resolve()
    if job.exists():raise FileExistsError('Fit job already exists')
    from .pva_bootstrap import cold_seed
    job.mkdir(parents=True);cold_seed(job/'source_candidate')
    model=read(job/'source_candidate/model.json');model['provenance'].update(fit_state='Native global OptiTrack; no retrospective height normalization',source_drone='cf_3')
    save(job/'source_candidate/model.json',model)
    names=['figure8_001','figure8_002','osci_001','osci_002','vertical_figure8_001']
    batch=root/'rehearsal_csv_and_result_in_real_flight/preliminary1';summary={};windows=[];protected={}
    for name in names:
        mp=batch/(name+'.csv');cp=batch/('experiment_'+name+'.csv')
        protected[str(mp)]=sha256_file(mp);protected[str(cp)]=sha256_file(cp)
        m=read_optitrack(mp);c=read_controller(cp);alignment=measured_clock_alignment(m,c)
        t=m['time']+alignment['offset_s'];r,rv=normalized_rotations_xyzw(m['quaternion'])
        anchor,av=attachment_positions(m['drone'],m['quaternion'],model['recorded_data']['optitrack_to_attachment_offset_body_m'])
        sites=np.concatenate([anchor[:,None],m['cable']],1)
        pose_valid=np.isfinite(m['drone']).all(1)&rv&av
        marker_valid=np.isfinite(m['cable']).all(-1)
        pj=np.linalg.norm(np.diff(m['drone'],axis=0),axis=-1)>.1
        pose_valid[:-1]&=~pj;pose_valid[1:]&=~pj
        jumps=np.linalg.norm(np.diff(m['cable'],axis=0),axis=-1)>.15
        marker_valid[:-1]&=~jumps;marker_valid[1:]&=~jumps
        chords=np.linalg.norm(np.diff(sites,axis=1),axis=-1)
        too_long=chords>np.asarray(model['cable']['marker_interval_lengths_m'])[None]+.015
        marker_valid&=~too_long
        marker_valid[:,:-1]&=~too_long[:,1:]
        pt,packets,until,end=recorded_packets(c)
        schedule=CommandSchedule(pt,torch.tensor(packets[None],dtype=torch.float64),coverage_end_s=end,valid_until_s=until)
        folder=job/'inputs'/name;folder.mkdir(parents=True)
        np.savez_compressed(folder/'data.npz',time=t,optitrack_time=m['time'],position=m['drone'],quaternion=m['quaternion'],rotation=r,sites=sites,
            pose_valid=pose_valid,marker_valid=marker_valid,position_jump_edges=pj,marker_jump_edges=jumps,chord_overlength=too_long,
            packet_time=pt,packets=packets,packet_valid_until=until,command_coverage_end=end)
        role='validation' if name=='figure8_002' else 'training'
        selected=[];rejected=[]
        # Fixed 2 s windows tile all recorded motion. A separate 0.4 s past
        # interval supplies causal state/memory estimates, never future samples.
        for index in range(41,len(t)-202,200):
            start=t[index];last=start+2.;hist=slice(index-41,index)
            if not pose_valid[index-41:index+201].all():
                rejected.append(dict(start_s=float(start),reason='Invalid/jump-masked drone history or rollout'));continue
            try:
                # Ensure every possible delay candidate has command support.
                checks=np.unique(np.r_[t[index-41:index+201],pt[(pt>=start-.6)&(pt<=last)],until[(until>=start-.6)&(until<=last)]])
                for a,b in zip(checks[:-1],checks[1:]):
                    if a<start-.53 or a>last:continue
                    schedule.sample((a+b)*.5-.12);schedule.sample((a+b)*.5)
                for x in (t[index-41]-.12,last):schedule.sample(x)
            except ValueError as exc:
                rejected.append(dict(start_s=float(start),reason=str(exc)));continue
            item=dict(take=name,role=role,index=index,start_s=float(start),end_s=float(last),name=name+f'-{index:05d}')
            selected.append(item);windows.append(item)
        summary[name]=dict(alignment=alignment,role=role,source_hashes={mp.name:protected[str(mp)],cp.name:protected[str(cp)]},drone=m['drone_label'],
            raw_time_span_s=t[[0,-1]],marker_masked_samples=(~marker_valid).sum(0),chord_overlength_samples=too_long.sum(0),
            accepted_drone_windows=len(selected),rejected_drone_windows=rejected,normalization='None; raw measured and commanded global coordinates retained')
        save(folder/'provenance.json',summary[name])
    if any(not any(w['take']==n for w in windows) for n in names):raise ValueError('A take has no usable drone windows')
    protocol=dict(schema='preliminary_pva_bootstrap_v1',training_takes=[n for n in names if n!='figure8_002'],validation_takes=['figure8_002'],
        window_s=2.,window_stride_s=2.,history_s=.4,cable_history_s=1.,cable_velocity_weight_tau_s=cable_velocity_weight_tau_s,cable_window_s=1.,source_batch=str(batch),normalization_applied=False,
        initial_memory='Causal moving-history mean acceleration minus nominal feedback/feedforward; fixed during each rollout; not observed firmware integral',
        alignment='Causal moving-history effective attitude alignment; not mounting calibration',
        selection='Training-only plateau; whole figure8_002 excluded from optimization and checkpoint selection',
        evidence='Retrospective short-window identification; final whole-take checks are not prospective flight evidence',
        stopping=dict(drone=dict(minimum=40,check_every=5,patience=5,relative=.005,ceiling=400),cable=dict(minimum=12,check_every=3,patience=4,relative=.005,ceiling=120)))
    save(job/'protocol.json',protocol);save(job/'windows.json',windows);save(job/'preparation.json',summary);save(job/'protected_before.json',protected)
    save(job/'status.json',dict(status='prepared',stage='preliminary inputs frozen'))
    return job


class PreliminaryTrial(Trial):
    """Short recursive prediction starting from actual moving past history."""
    def __init__(self,job,window,model,device='cuda'):
        self.name=window['name'];self.take=window['take'];self.role=window['role'];self.device=device;self.model=model
        # Absent in historical jobs: preserve their saved initialization semantics.
        self.cable_history_s=read(Path(job)/'protocol.json').get('cable_history_s')
        self.cable_velocity_weight_tau_s=read(Path(job)/'protocol.json').get('cable_velocity_weight_tau_s')
        with np.load(Path(job)/'inputs'/self.take/'data.npz') as z:self.data={k:z[k].copy() for k in z.files}
        d=self.data;index=window['index'];origin=window['start_s'];d['time']-=origin;d['packet_time']-=origin;d['packet_valid_until']-=origin
        d['pre_indices']=np.arange(index-41,index);self.hover_time=d['time'][d['pre_indices']]
        self.schedule=CommandSchedule(d['packet_time'],torch.tensor(d['packets'][None],dtype=torch.float64),coverage_end_s=float(d['command_coverage_end'])-origin,valid_until_s=d['packet_valid_until'])
        ids=np.arange(index-1,index+201);self.time=d['time'][ids];self.context={'csv_end_s':float(self.time[-1])}
        self.truth=dict(position_origin_m=d['position'][ids],rotation_tracking_to_world=d['rotation'][ids]);self.masks={'fit_position':d['pose_valid'][ids]&(self.time>=0),'fit_orientation':d['pose_valid'][ids]&(self.time>=0)}
        self.weights=self.masks['fit_position'].astype(float);self.weights/=self.weights.sum()
        self.offset=model['recorded_data']['optitrack_to_attachment_offset_body_m'];self.p0=d['position'][index-1].copy()
        h=d['position'][d['pre_indices']];tt=self.hover_time
        def derivative(times,values,at_start=False):
            duration=times[-1]-times[0];x=(times-times[-1])/duration
            coefficients=np.linalg.pinv(np.c_[np.ones(len(x)),x,x*x])@values
            return (coefficients[1]-(2*coefficients[2] if at_start else 0))/duration
        self.v0=derivative(tt[-11:],h[-11:]);first=derivative(tt[:11],h[:11],at_start=True);self.b0=(self.v0-first)/(tt[-1]-tt[0])
        rr=d['rotation'][d['pre_indices']];self.omega=derivative(tt[-11:],Rotation.from_matrix(rr[-1].T@rr[-11:]).as_rotvec())
        self.mean_rotation=Rotation.from_matrix(rr).mean().as_matrix();self.history_position=h;self.bases={};self.plans={}
        commands=np.stack([self.schedule.sample(t)[0].numpy() for t in tt]);d['hover_commands']=commands
        self.hover=(h[None],rr[None],commands[None]);self.basis=self.basis_for(.02)

    def cable_state(self,physics,*,cutoff=None):
        return super().cable_state(physics,cutoff=cutoff,history_s=self.cable_history_s,velocity_weight_tau_s=self.cable_velocity_weight_tau_s)

    def basis_for(self,delay):
        if delay not in self.bases:
            commands=np.stack([self.schedule.sample(t-delay)[0].numpy() for t in self.hover_time]);h=self.history_position;t=self.hover_time;dt=np.diff(t);duration=t[-1]-t[0]
            avg=lambda a:((a[:-1]+a[1:])*.5*dt[:,None]).sum(0)/duration
            ep=avg(commands[:,:3]-h);ev=avg(commands[:,3:6])-(h[-1]-h[0])/duration;acc=avg(commands[:,6:9])
            basis=np.zeros((3,6))
            for j,k in enumerate([0,0,1]):basis[j,k]=-ep[j];basis[j,k+2]=-ev[j];basis[j,k+4]=-acc[j]
            self.bases[delay]=basis
        return self.bases[delay]

    def state(self,gains,device,tau=.08,delay=.02,attitude_scales=(1.,1.)):
        tensor=lambda a:torch.tensor(np.asarray(a)[None],dtype=torch.float64,device=device)
        mean=tensor(self.b0*np.array([attitude_scales[0],attitude_scales[0],attitude_scales[1]]))
        alignment=command_attitude(mean,tensor([self.hover[2][0,0,9]])[:,0],9.80665).transpose(-1,-2)@tensor(self.mean_rotation)
        return PoseResponseState(tensor(self.p0),tensor(self.v0),tensor(self.data['rotation'][self.data['pre_indices'][-1]]),tensor(self.omega),
            tensor(self.b0+self.basis_for(delay)@np.asarray(gains)),alignment,'causal_motion_effective_alignment',float(self.time[0]))

    def linear(self,gains,delay):
        if delay not in self.plans:self.plans[delay]=integration_steps(self.schedule,self.time,delay)
        return linear_prediction(gains,self.p0,self.v0,self.b0,self.basis_for(delay),self.plans[delay])
