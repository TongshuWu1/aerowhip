"""Prospective raw-coordinate M0 checks and reviewed, immutable whip inputs.

No legacy height normalization, cold start, automatic role assignment or fitting.
"""
from pathlib import Path
from copy import deepcopy
import shutil
import numpy as np
import torch
from .io import atomic_json,sha256_file
from simulator.workflow import read_json
from .adaptation_check import load_comparison,flight_names
from .adaptation_rounds import read_optitrack,read_controller,COMMAND_COLUMNS
from .preliminary_prepare import recorded_packets
from .current_adaptation import Trial,causal_history_indices
from simulator.geometry import normalized_rotations_xyzw,attachment_positions
from simulator.drone_pose_response import CommandSchedule
from planning.pva_job import freeze_model_assets

SCHEMA='prospective_whip_adaptation_v1'

def verify_hashes(hashes):
    for p,h in hashes.items():
        if not Path(p).is_file() or sha256_file(p)!=h:raise ValueError('Source changed: '+str(p))


def setup(root,batch,selection_path=None):
    root=Path(root).resolve();batch=Path(batch).resolve()
    selection=read_json(selection_path or root/'config/pva/flight_selection.json')
    rehearsal=Path(selection['rehearsal']);package=Path(selection['package'])
    # Each selected package binds its own forecast; never reuse an M0 audit for M2.
    manifest=read_json(package/'manifest.json')
    verify_hashes({str(package/name):h for name,h in manifest['files'].items()})
    hashes={str(f):sha256_file(f) for f in rehearsal.rglob('*') if f.is_file() and f.name!='ARCHIVED'}
    verify_hashes(hashes)
    if sha256_file(rehearsal/'rehearsal.npz')!=selection['forecast_sha256']:raise ValueError('Selected forecast changed')
    if sha256_file(package/'fullstate_30hz.csv')!=selection['command_sha256']:raise ValueError('Selected command changed')
    batch.mkdir(parents=True,exist_ok=False);(batch/'flight_take').mkdir();(batch/'simulation_csv').mkdir()
    shutil.copy2(package/'fullstate_30hz.csv',batch/'simulation_csv/fullstate_30hz.csv')
    model=read_json(rehearsal/'model.json')
    generation=model.get('provenance',{}).get('generation_index',0)
    if type(generation) is not int or generation<0:raise ValueError('Invalid model generation')
    atomic_json(batch/'protocol.json',dict(schema=SCHEMA,rehearsal=str(rehearsal),frozen_hashes=hashes,
        parent_generation=generation,parent_forecast_model_sha256=sha256_file(rehearsal/'model.json'),
        command_sha256=selection['command_sha256'],forecast_sha256=selection['forecast_sha256'],
        frame='raw_global_xyz',cable_history_s=1.,drone_history_s=.4,cable_velocity_weight_tau_s=.02,
        planned_roles={'whip_001':'adaptation','whip_002':'adaptation','whip_003':'validation'},
        model_update='cable external_drag_s_inv only, if reviewed diagnostics support this scope',
        candidate_bounds_s_inv=[0.,2.],neural_training=False,automatic_selection=False,
        normalization=False,role_unit='whole take',minimum_observation_fraction=.8))
    atomic_json(batch/'time_alignment.json',{})
    return batch


def protocol(batch):
    batch=Path(batch);p=read_json(batch/'protocol.json')
    if p['schema']!=SCHEMA or p['frame']!='raw_global_xyz' or p['normalization']:
        raise ValueError('Expected raw-coordinate prospective whip protocol')
    verify_hashes(p['frozen_hashes'])
    if sha256_file(batch/'simulation_csv/fullstate_30hz.csv')!=p['command_sha256']:
        raise ValueError('Flown command differs from the frozen selection')
    return p


def rms_summary(error):
    error=np.asarray(error);valid=np.isfinite(error)
    return dict(rmse_m=float(np.sqrt(np.mean(error[valid]**2))) if valid.any() else None,
        valid_count=int(valid.sum()),total_count=int(error.size),coverage=float(valid.mean()))


def compare(root,batch,output):
    """Read original ghost before any fitting; native frames, no reinitialization."""
    batch=Path(batch).resolve();output=Path(output).resolve();p=protocol(batch)
    names=flight_names(batch)
    if not names:raise ValueError('No paired flight data yet')
    output.mkdir(parents=True,exist_ok=False);hashes={str(batch/'protocol.json'):sha256_file(batch/'protocol.json')}
    rows={};reviews={}
    for name in names:
        d=load_comparison(root,batch,name,p['rehearsal']);hashes.update(d['hashes'])
        mask=d['time']<=d['metadata']['whip_end_s']+1e-10
        rows[name]=dict(drone=rms_summary(d['drone_error'][mask]),tip=rms_summary(d['tip_error'][mask]),
            target_distance=rms_summary(d['target_error'][mask]),
            minimum_observed_tip_target_m=float(np.nanmin(d['target_error'][mask])) if np.isfinite(d['target_error'][mask]).any() else None,
            alignment=d['alignment'],onset_s=d['onset'],packet_jitter_s=d['packet_jitter_s'],packet_count=d['packet_count'],
            forecast_sha256=p['forecast_sha256'],frame='raw_global_xyz',
            interpretation='Observed target distance is sampled geometry, not independently verified physical contact; gaps are not inferred misses.')
        np.savez_compressed(output/(name+'.npz'),time_s=d['time'],measured_origin=d['measured_origin'],
            measured_cable=d['measured_cable'],predicted_origin=d['predicted_origin'],predicted_cable=d['predicted_cable'],
            drone_error=d['drone_error'],tip_error=d['tip_error'],target_distance=d['target_error'])
        reviews[name]=dict(role='unassigned',accepted=False,reviewed_by='',clock_reviewed=False,
            same_controller_and_hardware=None,no_intervention=None,physical_contact='unknown',
            free_motion_end_s=None,cable_only_update_justification='',notes='')
        reviews[name]['role']=p.get('planned_roles',{}).get(name,'unassigned')
    atomic_json(output/'source_hashes.json',hashes);verify_hashes(hashes)
    atomic_json(output/'report.json',dict(schema=SCHEMA,batch=str(batch),takes=rows,
        evidence='Frozen '+p.get('model_id',f"M{p.get('parent_generation',0)}")+' forecast comparison before adaptation; timing alignment is estimated unless independently verified.',new_rollouts=0))
    atomic_json(output/'review.template.json',dict(schema=SCHEMA,comparison=str(output),
        report_sha256=sha256_file(output/'report.json'),takes=reviews))
    return rows


def prepare(root,batch,comparison,review_path,job,*,full_model=False,diagnostics_only=False):
    """Freeze reviewed whole takes and raw masks; never change M0 or fit roles."""
    batch=Path(batch).resolve();comparison=Path(comparison).resolve();job=Path(job).resolve()
    p=protocol(batch);review=read_json(review_path);source_hashes=read_json(comparison/'source_hashes.json')
    verify_hashes(source_hashes)
    if review.get('schema')!=SCHEMA or Path(review['comparison']).resolve()!=comparison or review['report_sha256']!=sha256_file(comparison/'report.json'):
        raise ValueError('Review must refer to this exact frozen M0 comparison')
    if set(review['takes'])!=set(flight_names(batch)):raise ValueError('Batch membership changed after review')
    chosen={n:r for n,r in review['takes'].items() if r['role']!='excluded'}
    response=review.get('response_update')
    if response is not None:
        from .response_update_contract import validate
        validate(response);verify_hashes(response['diagnostic_hashes'])
        source_hashes.update(response['diagnostic_hashes'])
        p['response_update']=response
        p['model_update']='drone nominal feedforward_xy only; cable, delay, feedback, attitude, geometry and residual fixed'
    if diagnostics_only:
        if not chosen or any(r['role']!='validation' for r in chosen.values()):raise ValueError('Diagnostic preparation requires only whole validation takes')
    elif not any(r['role']=='adaptation' for r in chosen.values()):raise ValueError('Assign at least one whole adaptation take')
    for name,r in chosen.items():
        planned=p.get('planned_roles',{}).get(name)
        if planned is not None and r['role']!=planned:raise ValueError(name+': role differs from the predeclared whole-take split')
        if r['role'] not in ('adaptation','validation') or r['accepted'] is not True or not r['reviewed_by'].strip():
            raise ValueError(name+': explicit take review and whole-take role required')
        if not all(r[k] is True for k in ('clock_reviewed','same_controller_and_hardware','no_intervention')):
            raise ValueError(name+': clock, hardware/controller and intervention review required')
        end=r.get('free_motion_end_s')
        if r['physical_contact'] not in ('none','at_or_after_end') or not isinstance(end,(float,int)) or not np.isfinite(end) or end<=0:
            raise ValueError(name+': specify a reviewed free-motion interval before physical contact')
    model=read_json(Path(p['rehearsal'])/'model.json')
    if model.get('motion_residual',{}).get('enabled') and not full_model:
        raise ValueError('Scalar preparation requires cable NN disabled; use full-model preparation for a learned parent')
    job.mkdir(parents=True,exist_ok=False);(job/'source_candidate').mkdir()
    saved=freeze_model_assets(model,job/'source_candidate',source_root=Path(p['rehearsal']));atomic_json(job/'source_candidate/model.json',saved)
    report=read_json(comparison/'report.json');prepared={}
    for name,r in chosen.items():
        m=read_optitrack(batch/'flight_take'/f'{name}.csv')
        c=read_controller(batch/'flight_take'/f'experiment_{name}.csv',commands_only=True)
        check=report['takes'][name];t=m['time']+check['alignment']['offset_s']-check['onset_s']
        rotation,rv=normalized_rotations_xyzw(m['quaternion'])
        anchor,av=attachment_positions(m['drone'],m['quaternion'],model['recorded_data']['optitrack_to_attachment_offset_body_m'])
        sites=np.concatenate([anchor[:,None],m['cable']],1)
        pose_valid=np.isfinite(m['drone']).all(1)&rv&av;marker_valid=np.isfinite(m['cable']).all(-1)
        pj=np.linalg.norm(np.diff(m['drone'],axis=0),axis=-1)>.1
        mj=np.linalg.norm(np.diff(m['cable'],axis=0),axis=-1)>.15
        pose_valid[:-1]&=~pj;pose_valid[1:]&=~pj;marker_valid[:-1]&=~mj;marker_valid[1:]&=~mj
        too_long=np.linalg.norm(np.diff(sites,axis=1),axis=-1)>np.asarray(model['cable']['marker_interval_lengths_m'])+.015
        marker_valid&=~too_long;marker_valid[:,:-1]&=~too_long[:,1:]
        pre=causal_history_indices(t,0.,p['drone_history_s']);cpre=causal_history_indices(t,0.,p['cable_history_s'])
        if not pose_valid[cpre].all() or not marker_valid[cpre].all():raise ValueError(name+': missing clean one-second causal initialization')
        pt,packets,until,end=recorded_packets(c,invalid_rows_break_coverage=True);pt-=check['onset_s'];until-=check['onset_s'];end-=check['onset_s']
        schedule=CommandSchedule(pt,torch.tensor(packets[None],dtype=torch.float64),coverage_end_s=end,valid_until_s=until)
        hold=np.stack([schedule.sample(x)[0].numpy() for x in t[pre]])
        if not np.allclose(hold,hold[:1],atol=1e-8,rtol=0) or np.max(np.abs(hold[:,3:9]))>1e-8:
            raise ValueError(name+': drone initializer requires the recorded preflight hold')
        whip_end=read_json(Path(p['rehearsal'])/'rehearsal.json')['whip_end_s']
        stop=float(r['free_motion_end_s']) if diagnostics_only else min(float(r['free_motion_end_s']),whip_end)
        if stop>t[-1]:raise ValueError(name+': incomplete reviewed free-motion interval')
        # Validate actual command support including delay, all receipt/gap edges.
        delay=read_json(Path(saved['fullstate_execution']['checkpoint']))['nominal']['parameters']['delay_s']
        edges=np.unique(np.r_[t[pre[0]],stop,pt[(pt>=t[pre[0]]-.2)&(pt<=stop)],until[(until>=t[pre[0]]-.2)&(until<=stop)]])
        for x in np.r_[t[pre],t[(t>=0)&(t<=stop)],(edges[:-1]+edges[1:])*.5]:
            if t[pre[0]]<=x<=stop:schedule.sample(x-delay);schedule.sample(x)
        folder=job/'inputs'/name;folder.mkdir(parents=True)
        np.savez_compressed(folder/'data.npz',time=t,position=m['drone'],quaternion=m['quaternion'],rotation=rotation,
            sites=sites,pose_valid=pose_valid,marker_valid=marker_valid,pre_indices=pre,hover_commands=hold,
            packet_time=pt,packets=packets,packet_valid_until=until,command_coverage_end=end)
        prepared[name]=dict(role=r['role'],end_s=stop,review=r,masked_markers=(~marker_valid).sum(0).tolist())
    parent_generation=model.get('provenance',{}).get('generation_index',0)
    if p.get('parent_generation',parent_generation)!=parent_generation:raise ValueError('Protocol and frozen model generation disagree')
    p.update(parent_generation=parent_generation,takes=prepared,comparison=str(comparison),review_path=str(Path(review_path).resolve()),
        preparation_scope='full_model' if full_model else 'scalar',diagnostics_only=diagnostics_only,
        evidence=f'M{parent_generation+1} development; training takes cannot independently validate the updated model')
    atomic_json(job/'protocol.json',p);shutil.copy2(review_path,job/'review.json')
    shutil.copy2(comparison/'report.json',job/'frozen_M0_comparison.json')
    source_hashes[str(Path(review_path).resolve())]=sha256_file(review_path)
    atomic_json(job/'source_hashes.json',source_hashes);verify_hashes(source_hashes)
    code_hashes={}
    for base in ('experimental_data','simulator','planning','learning'):
        for source in (Path(root)/base).rglob('*'):
            if not source.is_file() or source.suffix not in ('.py','.cu','.cuh','.h','.cpp'):continue
            relative=source.relative_to(root);dest=job/'source_snapshot'/relative
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
            code_hashes[relative.as_posix()]=sha256_file(dest)
    atomic_json(job/'code_hashes.json',code_hashes)
    files={str(f):sha256_file(f) for folder in (job/'inputs',job/'source_candidate') for f in folder.rglob('*') if f.is_file()}
    files.update({str(job/n):sha256_file(job/n) for n in ('protocol.json','review.json','frozen_M0_comparison.json','source_hashes.json','code_hashes.json')})
    atomic_json(job/'prepared_hashes.json',files)
    atomic_json(job/'status.json',dict(status='prepared',fit_started=False,model_selected=False))
    return job


class WhipTrial(Trial):
    """Shared causal state/geometry routines with actual bounded packet coverage."""
    def __init__(self,job,name,model,device='cuda'):
        self.name=name;self.device=device;self.model=model
        self.protocol=read_json(Path(job)/'protocol.json');self.end=self.protocol['takes'][name]['end_s']
        with np.load(Path(job)/'inputs'/name/'data.npz') as z:self.data={k:z[k].copy() for k in z.files}
        d=self.data;pre=d['pre_indices'];self.hover_time=d['time'][pre]
        self.hover=(d['position'][pre][None],d['rotation'][pre][None],d['hover_commands'][None])
        self.offset=model['recorded_data']['optitrack_to_attachment_offset_body_m']
        self.schedule=CommandSchedule(d['packet_time'],torch.tensor(d['packets'][None],dtype=torch.float64),
            coverage_end_s=float(d['command_coverage_end']),valid_until_s=d['packet_valid_until'])

    def cable_state(self,physics,**kwargs):
        return super().cable_state(physics,history_s=self.protocol['cable_history_s'],
            velocity_weight_tau_s=self.protocol['cable_velocity_weight_tau_s'],**kwargs)
