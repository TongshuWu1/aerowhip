"""Immutable full-model adaptation inputs; no optimization or role reassignment."""
from pathlib import Path
from copy import deepcopy
import shutil
import numpy as np
import torch
from .io import atomic_json,sha256_file
from .whip_adaptation import WhipTrial,verify_hashes,SCHEMA
from .preliminary_prepare import PreliminaryTrial
from .current_adaptation import causal_history_indices
from simulator.workflow import read_json
from simulator.research_execution import ResearchExecutionModel
from planning.pva_job import freeze_model_assets

FULL_SCHEMA='whip_full_model_v1'
ROOT=Path(__file__).resolve().parents[1]


def default_contract():
    return dict(schema=FULL_SCHEMA,stages=['drone_nominal','drone_residual','attitude_refinement','cable_physics','cable_residual','combined_validation'],
        replay_weight=.5,position_scale_m=.02,orientation_scale_rad=.05,cable_scale_m=.02,
        nominal_prior=.03,residual_magnitude=.01,residual_change=.01,
        delay_candidates_s=[0.,.01,.02,.03,.04,.06,.08,.10,.12],
        drone_gain_bounds=[[.05,.05,.05,.05,.01,.01],[80.,80.,20.,20.,3.,3.]],
        attitude_bounds=[[.1,.05,.02],[3.,3.,.3]],
        cable_bounds=[[1e-10,1e-8,1e-4],[1e-4,1e-3,2.]],
        cable_residual=dict(hidden=32,acceleration_limit=.5,mode='acceleration'),
        cable_log_difference_step=1e-6,
        nominal_stopping=dict(minimum=6,patience=5,relative=.001,ceiling=80),
        residual_stopping=dict(minimum=40,patience=6,relative=.005,check_every=5,ceiling=None),
        seed=20260910,cable_history_s=1.,cable_velocity_weight_tau_s=.02,
        training='Whole adaptation takes and all eligible preliminary training windows; no validation gradients or selection',
        candidate_id='M1-full',parent_id='M0',promotion=False,
        interpretation='Effective loaded PVA response and cable dynamics; no identified motor ceiling or explicit cable reaction added')


def prepare(job,whip_source,preliminary_source,contract=None):
    job=Path(job).resolve();src=Path(whip_source).resolve();pre=Path(preliminary_source).resolve()
    c=deepcopy(contract or default_contract())
    source_protocol=read_json(src/'protocol.json')
    if not any(r['role']=='validation' for r in source_protocol['takes'].values()):
        c['evaluate_before_training']=True
    for label,budget in c.get('stage_budgets',{}).items():
        if label not in ('drone_residual','cable_residual') or type(budget['maximum_updates']) is not int or budget['maximum_updates']<1 or not np.isfinite(budget['maximum_seconds']) or budget['maximum_seconds']<=0:
            raise ValueError('Invalid residual stage budget')
    if 'stage_budgets' in c and set(c['stage_budgets'])!={'drone_residual','cable_residual'}:
        raise ValueError('Specify budgets for both residual stages')
    if read_json(src/'protocol.json').get('diagnostics_only'):raise ValueError('Final diagnostic recordings cannot enter a fit')
    if c.get('schema')!=FULL_SCHEMA:raise ValueError('Expected full-model contract')
    for name in ('prepared_hashes.json','source_hashes.json'):verify_hashes(read_json(src/name))
    # Parent source snapshot is evidence, not an assertion that later GUI code is identical.
    verify_hashes({str(src/'source_snapshot'/n):h for n,h in read_json(src/'code_hashes.json').items()})
    pp=read_json(pre/'protocol.json')
    if pp.get('normalization_applied') is not False:raise ValueError('Raw preliminary inputs required')
    windows=read_json(pre/'windows.json');train=[w for w in windows if w['role']=='training']
    if {w['take'] for w in train}!=set(pp['training_takes']):raise ValueError('Preliminary role mismatch')
    # Portable imports use POSIX separators on every OS. Compare path keys
    # consistently without changing either the preserved hashes or raw bytes.
    frozen={}
    for original,h in read_json(pre/'protected_before.json').items():
        key=original.replace('\\','/')
        if key in frozen and frozen[key]!=h:raise ValueError('Conflicting preliminary source hashes')
        frozen[key]=h
    sources=read_json(src/'source_hashes.json')
    all_takes=pp['training_takes']+pp['validation_takes']
    for take in all_takes:
        provenance=read_json(pre/'inputs'/take/'provenance.json')
        for filename,h in provenance['source_hashes'].items():
            path=(Path(pp['source_batch'])/filename).as_posix()
            if frozen.get(path)!=h:raise ValueError('Preliminary raw provenance mismatch')
            sources[path]=h
    verify_hashes(sources)
    job.mkdir(parents=True,exist_ok=False);shutil.copytree(src/'inputs',job/'inputs')
    (job/'source_candidate').mkdir()
    model=freeze_model_assets(read_json(src/'source_candidate/model.json'),job/'source_candidate',source_root=src/'source_candidate')
    atomic_json(job/'source_candidate/model.json',model)
    # Preserve the original raw-whip review as a separate immutable source.
    shutil.copy2(src/'review.json',job/'review.json')
    p=read_json(src/'protocol.json');p.pop('response_update',None)
    p.pop('candidate_bounds_s_inv',None)
    p.update(full_update=c,neural_training=True,model_update='Full drone nominal/residual and cable physics/residual',
        source_whip_job=str(src),preliminary_source=str(pre),
        validation_qualification='Whole validation takes excluded from updates and selection; outcomes previously inspected in method development.')
    p['parent_generation']=int(model.get('provenance',{}).get('generation_index',p.get('parent_generation',0)))
    if c['candidate_id']==c['parent_id']:raise ValueError('Candidate must have a new model ID')
    prior=[]
    for index,prior_source in enumerate(c.get('prior_whip_sources',[])):
        old=Path(prior_source).resolve();op=read_json(old/'protocol.json')
        verify_hashes(read_json(old/'prepared_hashes.json'));verify_hashes(read_json(old/'source_hashes.json'))
        sources.update(read_json(old/'source_hashes.json'))
        folder=job/'prior_whip'/str(index);folder.mkdir(parents=True)
        shutil.copy2(old/'protocol.json',folder/'protocol.json')
        names=[n for n,r in op['takes'].items() if r['role']=='adaptation']
        if not names:raise ValueError('Prior whip replay has no adaptation takes')
        for name in names:shutil.copytree(old/'inputs'/name,folder/'inputs'/name)
        # Prior validation remains excluded from training; its exact input is retained for post-fit checks.
        for name,r in op['takes'].items():
            if r['role']=='validation':shutil.copytree(old/'inputs'/name,folder/'inputs'/name)
        prior.append(dict(folder=str(folder),source=str(old),training_takes=names))
        for f in (old/'protocol.json',old/'prepared_hashes.json'):sources[str(f)]=sha256_file(f)
    p['prior_whip_replay']=prior
    replay=job/'replay';replay.mkdir()
    rp=deepcopy(pp);rp.update(cable_history_s=1.,cable_velocity_weight_tau_s=.02)
    atomic_json(replay/'protocol.json',rp);atomic_json(replay/'windows.json',train)
    atomic_json(replay/'validation_windows.json',[w for w in windows if w['role']=='validation'])
    for take in all_takes:
        shutil.copytree(pre/'inputs'/take,replay/'inputs'/take)
    for f in (pre/'protocol.json',pre/'windows.json',pre/'preparation.json'):
        sources[str(f)]=sha256_file(f)
    # Hash the input arrays too: raw hashes alone do not protect a prepared array.
    for take in all_takes:
        f=pre/'inputs'/take/'data.npz';sources[str(f)]=sha256_file(f)
    atomic_json(job/'protocol.json',p);atomic_json(job/'source_hashes.json',sources)
    code={}
    for base in ('experimental_data','simulator','planning','learning','tools'):
        for f in (ROOT/base).rglob('*'):
            if not f.is_file() or f.suffix not in ('.py','.cu','.cuh','.h','.cpp'):continue
            rel=f.relative_to(ROOT);dest=job/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
            code[rel.as_posix()]=sha256_file(f)
    atomic_json(job/'code_hashes.json',code)
    files={str(f):sha256_file(f) for d in ('inputs','replay','prior_whip','source_candidate') for f in (job/d).rglob('*') if f.is_file()}
    files.update({str(job/n):sha256_file(job/n) for n in ('protocol.json','review.json','source_hashes.json','code_hashes.json')})
    atomic_json(job/'prepared_hashes.json',files)
    atomic_json(job/'status.json',dict(status='prepared',stage='Full adaptation inputs frozen',model_selected=False))
    return job


def load(job,device='cuda'):
    job=Path(job).resolve()
    for n in ('prepared_hashes.json','source_hashes.json'):verify_hashes(read_json(job/n))
    code=read_json(job/'code_hashes.json')
    verify_hashes({str(job/'source_snapshot'/n):h for n,h in code.items()})
    verify_hashes({str(ROOT/n):h for n,h in code.items() if 'gui' not in Path(n).parts})
    p=read_json(job/'protocol.json')
    if p.get('schema')!=SCHEMA or p.get('full_update',{}).get('schema')!=FULL_SCHEMA:raise ValueError('Full prepared job required')
    m=read_json(job/'source_candidate/model.json')
    return m,p,ResearchExecutionModel.from_mapping(m,root=job/'source_candidate',device=device)


class WhipPoseTrial(WhipTrial):
    def __init__(self,job,name,model,device='cuda'):
        super().__init__(job,name,model,device);self.take=name;self.role=self.protocol['takes'][name]['role'];self.category='whip'
        d=self.data;ids=np.flatnonzero((d['time']>=self.hover_time[-1]-1e-10)&(d['time']<self.end-1e-9))
        self.time=d['time'][ids];self.context={'csv_end_s':float(self.time[-1])}
        self.truth={'position_origin_m':d['position'][ids],'rotation_tracking_to_world':d['rotation'][ids]}
        mask=(self.time>=0)&d['pose_valid'][ids];self.masks={'fit_position':mask,'fit_orientation':mask}
        self.weights=mask.astype(float)/mask.sum();self.bases={}
        self.p0=self.hover[0][0,-1].copy()
        # Recover affine initialization coefficients using the actual initializer.
        zero=np.zeros(6);s=self.state(zero,'cpu');self.v0=s.velocity[0].numpy();self.b0=s.compensation[0].numpy()
        self._basis=np.stack([self.state(np.eye(6)[i],'cpu').compensation[0].numpy()-self.b0 for i in range(6)],axis=1)
        from simulator.drone_pose_response import command_attitude
        a=torch.tensor(self.b0[None]);yaw=torch.tensor([self.hover[2][0,0,9]])
        self.mean_rotation=(command_attitude(a,yaw,9.80665)@s.rotation_command_from_tracking)[0].numpy()

    def basis_for(self,delay):return self._basis


def drone_trials(job,model,device='cuda',validation=False):
    p=read_json(Path(job)/'protocol.json');role='validation' if validation else 'adaptation'
    trials=[WhipPoseTrial(job,n,model,device) for n,r in p['takes'].items() if r['role']==role]
    if not validation:
        for previous in p.get('prior_whip_replay',[]):
            for name in previous['training_takes']:
                t=WhipPoseTrial(previous['folder'],name,model,device)
                if t.role!='adaptation':raise ValueError('Prior validation entered training replay')
                t.category='prior_whip';trials.append(t)
        for w in read_json(Path(job)/'replay/windows.json'):
            if w['role']!='training':raise ValueError('Validation entered training replay')
            t=PreliminaryTrial(Path(job)/'replay',w,model,device);t.category='preliminary';trials.append(t)
        weights=group_weights(trials,p['full_update']['replay_weight'])
        for t,w in zip(trials,weights):t.weights*=len(trials)*w
    return trials


def group_weights(rows,replay_weight=.5):
    """Equal takes within each data family; window count cannot change take weight."""
    get=lambda r,k:getattr(r,k) if not isinstance(r,dict) else r[k]
    families={get(r,'category') for r in rows};mass={f:(1. if f=='whip' else replay_weight) for f in families};total=sum(mass.values())
    result=[]
    for r in rows:
        f=get(r,'category');take=get(r,'take');takes={get(x,'take') for x in rows if get(x,'category')==f}
        n=sum(get(x,'take')==take and get(x,'category')==f for x in rows)
        result.append(mass[f]/total/len(takes)/n)
    return np.asarray(result)


@torch.no_grad()
def cable_rows(job,model,engine,trials,*,review_path=None):
    rows=[];excluded=[]
    for t in trials:
        is_whip=isinstance(t,WhipPoseTrial)
        for cutoff in ([0.] if is_whip else [0.,1.]):
            try:
                if is_whip:state,start,projection=t.cable_state(engine.physics)
                else:state,start,projection=t.cable_state(engine.physics,cutoff=cutoff)
                end=t.end if is_whip else cutoff+1.
                grid=start+np.arange(int(np.floor((end-start)/engine.dt_s+1e-8))+1)*engine.dt_s
                origin,rotation,truth=t.measured(grid)
                if not np.isfinite(truth[:,0]).all():raise ValueError('Missing measured attachment')
                considered=(grid>=cutoff)&(grid<end-1e-9)
                valid=np.isfinite(truth[:,1:]).all(-1)&considered[:,None]
                if valid[considered].mean()<.8 or valid[considered,-1].mean()<.8:raise ValueError('Insufficient marker coverage')
                rows.append(dict(name=t.name+f'-{cutoff:g}',take=t.take,category=t.category,trial=t,grid=grid,end=end,cutoff=cutoff,
                    origin=origin,rotation=rotation,truth=truth,valid=valid,q=state.positions_m[0],v=state.velocities_m_s[0],
                    roots=torch.as_tensor(truth[:,0],device=t.device,dtype=torch.float64),projection_m=projection))
            except (ValueError,IndexError) as exc:
                if is_whip:raise
                excluded.append(dict(name=t.name,cutoff=cutoff,reason=str(exc)))
    atomic_json(review_path or Path(job)/'cable_window_review.json',dict(accepted=[{k:r[k] for k in ('name','take','category','projection_m')} for r in rows],excluded=excluded))
    return rows


def cable_data(rows,replay_weight=.5):
    length=max(len(r['grid']) for r in rows)
    def pad(x):return torch.cat([x,x[-1:].expand((length-len(x),)+x.shape[1:])])
    q=torch.stack([r['q'] for r in rows]);device=q.device
    truth=torch.stack([pad(q.new_tensor(np.nan_to_num(r['truth'][:,1:]))) for r in rows])
    mask=torch.zeros((len(rows),length,truth.shape[2]),device=device,dtype=torch.bool)
    for i,r in enumerate(rows):mask[i,:len(r['valid'])]=torch.as_tensor(r['valid'],device=device)
    return dict(q=q,v=torch.stack([r['v'] for r in rows]),roots=torch.stack([pad(r['roots']) for r in rows]),truth=truth,mask=mask,
        weights=q.new_tensor(group_weights(rows,replay_weight)))
