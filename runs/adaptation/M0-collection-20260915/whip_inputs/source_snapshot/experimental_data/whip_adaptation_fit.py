"""Small, GPU-batched physical update after the frozen forecast/data review.

Only effective scalar cable damping changes. Drone, EI/Cb, geometry and neural
weights stay fixed. No automatic promotion or planner launch.
"""
from pathlib import Path
from copy import deepcopy
from dataclasses import replace
import time
import numpy as np
import torch
from .whip_adaptation import WhipTrial,verify_hashes,rms_summary,SCHEMA
from .io import atomic_json,sha256_file
from .plateau import Plateau
from simulator.workflow import read_json
from simulator.research_execution import ResearchExecutionModel
from simulator.research_physics import ResearchPhysics
from simulator.cable import DderState
from planning.pva_job import freeze_model_assets


def load(job,device):
    job=Path(job);verify_hashes(read_json(job/'prepared_hashes.json'));verify_hashes(read_json(job/'source_hashes.json'))
    root=Path(__file__).resolve().parents[1]
    code=read_json(job/'code_hashes.json')
    verify_hashes({str(root/name):h for name,h in code.items()})
    verify_hashes({str(job/'source_snapshot'/name):h for name,h in code.items()})
    p=read_json(job/'protocol.json')
    if p['schema']!=SCHEMA:raise ValueError('Expected reviewed whip protocol')
    model=read_json(job/'source_candidate/model.json')
    if model.get('motion_residual',{}).get('enabled'):raise ValueError('Cable residual must remain disabled')
    return model,p,ResearchExecutionModel.from_mapping(model,root=job/'source_candidate',device=device)


@torch.no_grad()
def records(job,names,model,engine,device):
    result=[]
    for name in names:
        t=WhipTrial(job,name,model,device);state,start,projection=t.cable_state(engine.physics)
        count=int(np.floor((t.end-start)/engine.dt_s+1e-8))
        if count<15:raise ValueError(name+': reviewed free-motion interval is too short')
        grid=start+np.arange(count+1)*engine.dt_s
        origin,rotation,truth=t.measured(grid)
        if not np.isfinite(truth[:,0]).all():raise ValueError(name+': attachment has gaps; conditional fitting cannot bridge them')
        valid=np.isfinite(truth[:,1:]).all(-1)&(grid[:,None]>=0)&(grid[:,None]<t.end-1e-9)
        considered=(grid>=0)&(grid<t.end-1e-9)
        threshold=t.protocol['minimum_observation_fraction']
        if not considered.any() or valid[considered].mean()<threshold or valid[considered,-1].mean()<threshold:
            raise ValueError(name+': insufficient observed marker/tip coverage in reviewed interval')
        result.append(dict(name=name,trial=t,grid=grid,origin=origin,rotation=rotation,truth=truth,valid=valid,
            q=state.positions_m[0],v=state.velocities_m_s[0],roots=torch.as_tensor(truth[:,0],device=device,dtype=torch.float64),projection_m=projection))
    return result


class CableBatch:
    """Candidates × takes share one captured physics kernel; padding is unscored."""
    def __init__(self,engine,rows,count):
        self.rows=rows;self.count=count;self.engine=engine
        q=torch.stack([r['q'] for r in rows]).repeat_interleave(count,0)
        v=torch.stack([r['v'] for r in rows]).repeat_interleave(count,0)
        self.initial=DderState(q,v);self.rates=q.new_zeros(len(q))
        length=max(len(r['grid']) for r in rows)
        self.roots=torch.stack([torch.cat([r['roots'],r['roots'][-1:].expand(length-len(r['grid']),-1)]) for r in rows]).repeat_interleave(count,0)
        constants=replace(engine.physics.runtime_constants(q),external_drag_s_inv=self.rates)
        self.step=ResearchPhysics(engine.physics,self.initial,engine.dt_s,constants=constants,
            graph=True,fast_solve=q.is_cuda,fast_geometry=q.is_cuda)

    @torch.no_grad()
    def __call__(self,rates):
        if len(rates)!=self.count or not np.isfinite(rates).all() or np.min(rates)<0:raise ValueError('Invalid damping candidates')
        self.rates.copy_(self.rates.new_tensor(rates).repeat(len(self.rows)))
        state=self.initial;positions=[state.positions_m]
        for root in self.roots[:,1:].unbind(1):
            state=self.step(state,root);positions.append(state.positions_m)
        q=torch.stack(positions,1)
        if not bool(torch.isfinite(q).all()):raise ValueError('Nonfinite physical candidate; stop and inspect forward model')
        return q.reshape(len(self.rows),self.count,*q.shape[1:])


def losses(q,rows,marker_ids):
    per_take=[]
    for i,r in enumerate(rows):
        pred=q[i,:,:len(r['grid']),marker_ids,:]
        truth=pred.new_tensor(np.nan_to_num(r['truth'][:,1:]));mask=torch.as_tensor(r['valid'],device=pred.device)
        distance=(pred-truth).norm(dim=-1)
        robust=torch.sqrt(1+(distance/.02)**2)-1
        # Equal weight per take and fixed observed masks, never candidate-dependent masks.
        all_markers=(robust*mask).sum((1,2))/mask.sum()
        tip=(robust[:,:,-1]*mask[:,-1]).sum(1)/mask[:,-1].sum()
        per_take.append(.5*(all_markers+tip))
    return torch.stack(per_take).mean(0)


@torch.no_grad()
def evaluate(rows,engine,rate,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    conditional=CableBatch(engine,rows,1)([rate]);report={}
    for i,r in enumerate(rows):
        t=r['trial'];d=t.data;state=DderState(r['q'][None],r['v'][None])
        coupled=engine.predict(t.initial_pose(engine.drone.parameters),state,
            torch.as_tensor(d['packets'][None],device=t.device,dtype=torch.float64),d['packet_time'],r['grid'],graph=True,
            hover_command=torch.as_tensor(d['hover_commands'][-1:],device=t.device,dtype=torch.float64))
        cq=conditional[i,0,:len(r['grid'])].cpu().numpy();pq=coupled['cable_positions_m'][0].cpu().numpy()
        po=coupled['position_origin_m'][0].cpu().numpy();ids=list(engine.cable.marker_node_indices[1:])
        if not np.isfinite(pq).all() or not bool(coupled['valid'].all()):raise ValueError('Invalid coupled diagnostic')
        def cable_error(q):
            error=np.linalg.norm(q[:,ids]-r['truth'][:,1:],axis=-1);error[~r['valid']]=np.nan
            # Initialization frames are outside the evaluation denominator.
            return dict(markers=rms_summary(error[pose_mask]),tip=rms_summary(error[pose_mask,-1]))
        pose_mask=(r['grid']>=0)&(r['grid']<t.end-1e-9)
        report[r['name']]=dict(drone=rms_summary(np.linalg.norm(po[pose_mask]-r['origin'][pose_mask],axis=-1)),
            conditional_cable=cable_error(cq),command_driven=cable_error(pq),projection_m=r['projection_m'],
            role=t.protocol['takes'][r['name']]['role'],start_s=float(r['grid'][0]),end_s=float(r['grid'][-1]))
        np.savez_compressed(folder/(r['name']+'.npz'),time_s=r['grid'],measured_sites=r['truth'],mask=r['valid'],
            conditional_cable=cq,coupled_cable=pq,measured_origin=r['origin'],predicted_origin=po)
    atomic_json(folder/'metrics.json',report)
    return report


def diagnose(job,device='cuda'):
    job=Path(job);model,p,engine=load(job,device)
    names=[name for name,row in p['takes'].items() if row['role']=='adaptation']
    if not names:raise ValueError('Pre-fit diagnosis requires adaptation takes')
    rows=records(job,names,model,engine,device)
    result=evaluate(rows,engine,model['cable']['external_drag_s_inv'],job/'baseline')
    atomic_json(job/'fit_review.template.json',dict(baseline_metrics_sha256=sha256_file(job/'baseline/metrics.json'),
        reviewed_by='',accept_cable_only_scope=False,reason='',
        diagnostic_takes=names,diagnostic_role='adaptation',
        note='Use adaptation takes to choose the update scope. Validation model diagnostics are deferred until selection is frozen. If drone execution or initial-state mismatch dominates, do not make cable damping absorb it.'))
    return result


def fit(job,review_path,device='cuda'):
    job=Path(job);model,p,engine=load(job,device);review=read_json(review_path)
    if p.get('response_update') is not None:raise ValueError('This job requires the reviewed response fitter, not cable fitting')
    if (review.get('accept_cable_only_scope') is not True or not review.get('reviewed_by','').strip() or
        not review.get('reason','').strip() or review.get('baseline_metrics_sha256')!=sha256_file(job/'baseline/metrics.json')):
        raise ValueError('Review the exact baseline diagnostics and justify a cable-only update first')
    names=[n for n,r in p['takes'].items() if r['role']=='adaptation']
    rows=records(job,names,model,engine,device);folder=job/'fit';folder.mkdir(exist_ok=False)
    # Freeze fitting code as well as inputs; no source-dependent automatic resume.
    for filename in ('whip_adaptation.py','whip_adaptation_fit.py'):
        shutil_source=Path(__file__).parent/filename
        (folder/filename).write_bytes(shutil_source.read_bytes())
    atomic_json(folder/'review.json',review)
    atomic_json(job/'status.json',dict(status='running',stage='cable damping only',model_selected=False))
    start=time.perf_counter();base=float(model['cable']['external_drag_s_inv']);best=base;best_loss=float('inf')
    low,high=p['candidate_bounds_s_inv'];width=(high-low)/2
    if not low<=base<=high:raise ValueError('M0 damping lies outside declared search bounds')
    batch=CableBatch(engine,rows,9);ids=list(engine.cable.marker_node_indices[1:]);stop=Plateau(minimum=3,patience=3,relative=.005)
    history=[];reason='safety_ceiling'
    try:
        for update in range(1,13):
            if (job/'STOP').exists():raise InterruptedError('Manual stop; search state retained')
            # Base/incumbent are explicitly present; no parameter randomization campaign.
            grid=np.linspace(low,high,7) if update==1 else np.linspace(max(low,best-width),min(high,best+width),7)
            rates=np.r_[base,best,grid]
            scores=losses(batch(rates),rows,ids).cpu().numpy()
            index=int(np.argmin(scores))
            if scores[index]<best_loss:best_loss=float(scores[index]);best=float(rates[index])
            _,done=stop.observe(update,best_loss)
            history.append(dict(update=update,rates_s_inv=rates.tolist(),scores=scores.tolist(),best=best,
                best_loss=best_loss,baseline_loss=float(scores[0]),elapsed_s=time.perf_counter()-start))
            atomic_json(folder/'history.json',history)
            atomic_json(folder/'search_state.json',dict(update=update,best=best,best_loss=best_loss,width=width,
                plateau=stop.__dict__,training_takes=names,automatic_resume=False))
            print(f'CABLE update {update}: damping={best:.6g}/s loss={best_loss:.6g}',flush=True)
            if done:reason='practical_plateau';break
            width*=.5
        candidate=deepcopy(model);candidate['cable']['external_drag_s_inv']=best
        generation=int(p.get('parent_generation',0))+1
        candidate.setdefault('provenance',{}).update(label=f'M{generation} development - reviewed whip scalar damping',
            adaptation_generation=f'M{generation} development',generation_index=generation,
            parent_model_sha256=sha256_file(job/'source_candidate/model.json'),
            source_job=str(job),training_takes=names,updated_parameters=['cable.external_drag_s_inv'],
            drone_and_neural_weights_inherited=True,fit_complete=False,selected_model=False,flight_ready=False,
            prospective_flight_evidence=False,quality_note='Post-fit development candidate; promotion and prospective check pending')
        out=job/'candidate';out.mkdir();candidate=freeze_model_assets(candidate,out);atomic_json(out/'model.json',candidate)
        # Re-evaluate the frozen scalar with an independent batch-one rollout.
        check_engine=ResearchExecutionModel.from_mapping(candidate,root=out,device=device)
        check_rows=records(job,names,candidate,check_engine,device)
        independent=[]
        for row in check_rows:
            independent.append(float(losses(CableBatch(check_engine,[row],1)([best]),[row],ids)[0]))
        difference=abs(float(np.mean(independent))-best_loss)
        if difference>1e-7:raise ValueError('Batched and independent selection losses disagree')
        atomic_json(folder/'selection_frozen.json',dict(candidate_model_sha256=sha256_file(out/'model.json'),
            damping_s_inv=best,training_takes=names,selected_loss=best_loss,
            note='Parameter selection is complete before parent/candidate validation diagnostics.'))
        # Whole validation takes are first evaluated after parameter selection is frozen.
        validation_names=[n for n,r in p['takes'].items() if r['role']=='validation']
        if validation_names:
            validation_rows=records(job,validation_names,model,engine,device)
            evaluate(validation_rows,engine,base,job/'baseline_validation')
        all_rows=records(job,list(p['takes']),candidate,check_engine,device)
        final=evaluate(all_rows,check_engine,best,job/'candidate_diagnostics')
        verify_hashes(read_json(job/'prepared_hashes.json'));verify_hashes(p['frozen_hashes'])
        summary=dict(status='completed',stop_reason=reason,updates=len(history),elapsed_s=time.perf_counter()-start,
            candidate_hashes={str(path):sha256_file(path) for folder in (out,job/'candidate_diagnostics',job/'baseline_validation') for path in folder.rglob('*') if path.is_file()},
            selection_frozen_sha256=sha256_file(folder/'selection_frozen.json'),
            baseline_damping_s_inv=base,candidate_damping_s_inv=best,baseline_loss=history[0]['baseline_loss'],
            selected_loss=best_loss,independent_loss_difference=difference,
            training_takes=names,validation_takes=[n for n,r in p['takes'].items() if r['role']=='validation'],
            model_selected=False,drone_fitted=False,neural_training=False,physical_validation=False,
            at_parameter_boundary=bool(best==low or best==high),
            evidence='Training improvement is in-sample. A validation repeat of this same maneuver is not proof of new-motion generalization. A later prospective M1 flight is required.')
        atomic_json(folder/'result.json',summary);atomic_json(job/'status.json',summary)
        return summary
    except BaseException as exc:
        atomic_json(job/'status.json',dict(status='stopped' if isinstance(exc,InterruptedError) else 'failed',error=str(exc),
            model_selected=False,automatic_resume=False));raise
