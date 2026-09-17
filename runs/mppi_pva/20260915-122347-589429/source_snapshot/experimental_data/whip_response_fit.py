"""One reviewed drone response gain, fixed cable/NN/delay, prospective validation."""
from pathlib import Path
from copy import copy,deepcopy
from dataclasses import replace
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from . import whip_adaptation_fit as fitting
from .whip_adaptation import WhipTrial,verify_hashes
from .current_adaptation import causal_history_indices
from .drone_response_diagnostic import check_command_coverage
from .response_update_contract import validate
from .plateau import Plateau
from .io import atomic_json,sha256_file
from .model_evaluation import comparison_identity
from simulator.workflow import read_json
from simulator.research_execution import ResearchExecutionModel
from planning.pva_job import freeze_model_assets


def pose_loss(predicted,position,rotation,mask,contract):
    """Equal take weight outside; pseudo-Huber position and orientation within."""
    if not np.any(mask):raise ValueError('No scored native pose')
    dp=np.linalg.norm(predicted['position_origin_m'][mask]-position[mask],axis=-1)/contract['position_scale_m']
    relative=rotation[mask].transpose(0,2,1)@predicted['rotation_tracking_to_world'][mask]
    dr=np.linalg.norm(Rotation.from_matrix(relative).as_rotvec(),axis=-1)/contract['orientation_scale_rad']
    return float(np.mean(.5*(np.sqrt(1+dp*dp)-1+np.sqrt(1+dr*dr)-1)))


def pose_trial(job,name,model,engine,device,clock_shift=0.):
    trial=WhipTrial(job,name,model,device)
    if clock_shift:
        d=trial.data=trial.data.copy();d['time']=d['time']+clock_shift
        pre=causal_history_indices(d['time'],0.,trial.protocol['drone_history_s'])
        trial.hover_time=d['time'][pre]
        hold=np.stack([trial.schedule.sample(float(t))[0].cpu().numpy() for t in trial.hover_time])
        if not d['pose_valid'][pre].all() or np.max(abs(hold[:,3:9]))>1e-8:raise ValueError('Clock scenario lacks valid stationary past history')
        trial.hover=(d['position'][pre][None],d['rotation'][pre][None],hold[None])
    d=trial.data;ids=np.flatnonzero((d['time']>=trial.hover_time[-1]-1e-10)&(d['time']<trial.end-1e-9))
    t=d['time'][ids];mask=(t>=0)&d['pose_valid'][ids]
    check_command_coverage(trial,t,engine.drone.parameters.delay_s)
    return dict(trial=trial,time=t,position=d['position'][ids],rotation=d['rotation'][ids],mask=mask,
        packets=torch.as_tensor(d['packets'][None],device=device,dtype=torch.float64),
        hold=torch.as_tensor(trial.hover[2][0,-1:],device=device,dtype=torch.float64))


@torch.no_grad()
def predict(row,engine,gain,graph=True):
    drone=copy(engine.drone);drone.parameters=replace(engine.drone.parameters,feedforward_xy=float(gain))
    trial=row['trial']
    value=drone.predict(trial.initial_pose(drone.parameters),row['packets'],trial.data['packet_time'],row['time'],
        engine.offset,graph=graph,hover_command=row['hold'])
    if not bool(value['valid'].all()):raise ValueError('Invalid response prediction')
    return {k:v[0].cpu().numpy() for k,v in value.items()}


def scalar_search(objective,base,bounds,folder,stop_file):
    """Cached bounded refinement with baseline/incumbent retained and saved state."""
    low,high=bounds;best=base;best_loss=float('inf');width=(high-low)/2
    plateau=Plateau(minimum=3,patience=3,relative=.005);history=[];cache={};started=time.perf_counter()
    def score(value):
        key=float(value)
        if key not in cache:cache[key]=float(objective(key))
        if not np.isfinite(cache[key]):raise ValueError('Nonfinite response loss')
        return cache[key]
    reason='safety_ceiling'
    for update in range(1,13):
        if Path(stop_file).exists():raise InterruptedError('Manual stop; scalar state preserved')
        grid=np.linspace(low,high,7) if update==1 else np.linspace(max(low,best-width),min(high,best+width),7)
        gains=np.r_[base,best,grid];scores=np.array([score(g) for g in gains]);index=int(scores.argmin())
        if scores[index]<best_loss:best=float(gains[index]);best_loss=float(scores[index])
        _,done=plateau.observe(update,best_loss)
        history.append(dict(update=update,parameter='feedforward_xy',gains=gains.tolist(),scores=scores.tolist(),
            best=best,best_loss=best_loss,baseline_loss=score(base),elapsed_s=time.perf_counter()-started))
        atomic_json(Path(folder)/'history.json',history)
        atomic_json(Path(folder)/'search_state.json',dict(best=best,best_loss=best_loss,width=width,
            cache={str(k):v for k,v in cache.items()},plateau=plateau.__dict__,automatic_resume=False))
        print(f'DRONE update {update}: feedforward_xy={best:.6g} loss={best_loss:.6g}',flush=True)
        if done:reason='practical_plateau';break
        width*=.5
    return best,best_loss,history,reason


@torch.no_grad()
def fit(job,device='cuda'):
    job=Path(job).resolve();model,p,engine=fitting.load(job,device)
    contract=validate(p.get('response_update',{}));verify_hashes(contract['diagnostic_hashes'])
    names=[n for n,r in p['takes'].items() if r['role']=='adaptation']
    validation=[n for n,r in p['takes'].items() if r['role']=='validation']
    folder=job/'fit';folder.mkdir(exist_ok=False);started=time.perf_counter()
    atomic_json(job/'status.json',dict(status='running',stage='drone horizontal response gain',model_selected=False))
    try:
        rows=[pose_trial(job,n,model,engine,device) for n in names]
        base=float(engine.drone.parameters.feedforward_xy)
        bounds=np.array(contract['bounds_scale'])*base
        def objective(gain):
            losses=[pose_loss(predict(r,engine,gain),r['position'],r['rotation'],r['mask'],contract) for r in rows]
            return float(np.mean(losses)+contract['regularization']*((gain/base)-1)**2)
        best,loss,history,reason=scalar_search(objective,base,bounds,folder,job/'STOP')
        independent=float(np.mean([pose_loss(predict(r,engine,best,False),r['position'],r['rotation'],r['mask'],contract) for r in rows])
            +contract['regularization']*((best/base)-1)**2)
        if abs(independent-loss)>1e-7:raise ValueError('Captured and independent eager pose selection losses differ')
        # Review robustness only on adaptation recordings before accessing validation.
        timing=[]
        for shift in contract['clock_check_offsets_s']:
            shifted=[pose_trial(job,n,model,engine,device,shift) for n in names]
            losses={}
            for label,gain in [('M0',base),('candidate',best)]:
                losses[label]=[pose_loss(predict(r,engine,gain),r['position'],r['rotation'],r['mask'],contract) for r in shifted]
            timing.append(dict(clock_shift_s=shift,losses=losses,mean_improved=bool(np.mean(losses['candidate'])<np.mean(losses['M0']))))
        atomic_json(folder/'clock_sensitivity.json',timing)
        out=job/'candidate';out.mkdir();candidate=freeze_model_assets(deepcopy(model),out)
        component=Path(candidate['fullstate_execution']['checkpoint']);drone=read_json(component)
        drone['nominal']['parameters']['feedforward_xy']=best
        drone['nominal']['training_takes']=list(dict.fromkeys(drone['nominal'].get('training_takes',[])+names))
        atomic_json(component,drone);candidate['fullstate_execution']['sha256']=sha256_file(component)
        generation=int(p.get('parent_generation',0))+1
        candidate.setdefault('provenance',{}).update(label=f'M{generation} development - horizontal drone response',
            adaptation_generation=f'M{generation} development',generation_index=generation,
            parent_model_sha256=sha256_file(job/'source_candidate/model.json'),source_job=str(job),training_takes=names,
            updated_parameters=['drone.nominal.parameters.feedforward_xy'],drone_fitted=True,neural_weights_inherited=True,
            cable_parameters_inherited=True,fit_complete=True,selected_model=False,flight_ready=False,
            prospective_flight_evidence=False,quality_note='One effective closed-loop gain; no actuator-limit identification. Validation/prospective review separate.')
        atomic_json(out/'model.json',candidate)
        if comparison_identity(out/'model.json',contract)!=comparison_identity(job/'source_candidate/model.json',contract):
            raise ValueError('Candidate changed fields outside the reviewed single-gain scope')
        check_engine=ResearchExecutionModel.from_mapping(candidate,root=out,device=device)
        # Past-state reconstruction is repeated under each model. For this gain-only
        # update, a zero-acceleration hold must make measured initialization identical.
        initial_max=0.
        for row in rows:
            a=row['trial'].initial_pose(engine.drone.parameters);b=row['trial'].initial_pose(check_engine.drone.parameters)
            for field in ('position','velocity','rotation','omega_tracking','compensation','rotation_command_from_tracking'):
                initial_max=max(initial_max,float((getattr(a,field)-getattr(b,field)).abs().max()))
            if a.time_s!=b.time_s:raise ValueError('Candidate initial time differs')
        if initial_max>1e-10:raise ValueError('Gain update changed the fixed measured initial state')
        native={}
        for name,row in zip(names,rows):
            before=predict(row,engine,base);after=predict(row,check_engine,best)
            np.savez_compressed(folder/(name+'-pose.npz'),time_s=row['time'],mask=row['mask'],
                measured_origin=row['position'],baseline_origin=before['position_origin_m'],candidate_origin=after['position_origin_m'])
            native[name]=dict(baseline_loss=pose_loss(before,row['position'],row['rotation'],row['mask'],contract),
                candidate_loss=pose_loss(after,row['position'],row['rotation'],row['mask'],contract))
        atomic_json(folder/'adaptation_pose.json',native)
        cable_rows=fitting.records(job,names,model,engine,device)
        fitting.evaluate(cable_rows,engine,model['cable']['external_drag_s_inv'],job/'baseline')
        adaptation_result=fitting.evaluate(cable_rows,check_engine,candidate['cable']['external_drag_s_inv'],job/'candidate_adaptation')
        atomic_json(folder/'selection_frozen.json',dict(candidate_model_sha256=sha256_file(out/'model.json'),
            parameter='feedforward_xy',value=best,selected_loss=loss,training_takes=names,
            initial_state_max_difference=initial_max,clock_sensitivity_sha256=sha256_file(folder/'clock_sensitivity.json'),
            note='Parameter selection and adaptation-only diagnostics frozen before validation. No tuning on validation.'))
        # First reinitialized validation model diagnostics occur after the freeze.
        if validation:
            val_rows=fitting.records(job,validation,model,engine,device)
            fitting.evaluate(val_rows,engine,model['cable']['external_drag_s_inv'],job/'baseline_validation')
        all_rows=fitting.records(job,list(p['takes']),model,engine,device)
        fitting.evaluate(all_rows,check_engine,candidate['cable']['external_drag_s_inv'],job/'candidate_diagnostics')
        verify_hashes(read_json(job/'prepared_hashes.json'));verify_hashes(p['frozen_hashes'])
        result=dict(status='completed',stop_reason=reason,updates=len(history),elapsed_s=time.perf_counter()-started,
            parameter='feedforward_xy',baseline_gain=base,candidate_gain=best,baseline_loss=history[0]['baseline_loss'],
            selected_loss=loss,independent_loss_difference=abs(independent-loss),initial_state_max_difference=initial_max,
            clock_checks_improved=all(r['mean_improved'] for r in timing),at_parameter_boundary=bool(best in bounds),
            training_takes=names,validation_takes=validation,model_selected=False,drone_fitted=True,
            cable_fitted=False,neural_training=False,physical_validation=False,device=device,dtype='float64',
            selection_frozen_sha256=sha256_file(folder/'selection_frozen.json'),
            candidate_hashes={str(path):sha256_file(path) for sub in (out,job/'candidate_diagnostics',job/'baseline_validation',job/'candidate_adaptation') for path in sub.rglob('*') if path.is_file()},
            evidence='Effective horizontal response over observed conditions. Same-command held-out repetition; new M1 prospective flight still needed.')
        atomic_json(folder/'result.json',result);atomic_json(job/'status.json',result)
        return result
    except BaseException as exc:
        atomic_json(job/'status.json',dict(status='stopped' if isinstance(exc,InterruptedError) else 'failed',error=str(exc),model_selected=False,automatic_resume=False))
        raise
