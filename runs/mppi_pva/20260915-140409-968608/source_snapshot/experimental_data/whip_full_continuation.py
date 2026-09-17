"""Reviewed continuation of the full fit, preserving failed and completed stages."""
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict
import gc,itertools,time,shutil
import numpy as np
import torch
from . import whip_full_data as data
from . import whip_full_fit as full
from .whip_full_optim import FullTranslation,fit_attitude,progress
from .whip_full_cable import make_residual,residual_objective,gradient_check
from .plateau import Plateau
from .io import atomic_json,sha256_file
from .whip_adaptation import verify_hashes
from simulator.workflow import read_json
from simulator.research_execution import ResearchExecutionModel
from .model_evaluation import model_identity


def prepare(source,job,numerical_review):
    source=Path(source).resolve();job=Path(job).resolve();review=Path(numerical_review).resolve()
    p=read_json(source/'protocol.json');c=deepcopy(p['full_update'])
    if read_json(source/'status.json')['status']!='failed':raise ValueError('Expected preserved failed full run')
    if (source/'candidate/model.json').exists():raise ValueError('Do not silently revise a frozen candidate')
    rows=read_json(review);valid=[r for r in rows.values() if r.get('gradient',{}).get('passed')]
    if not valid:raise ValueError('No numerically verified physical checkpoint')
    chosen=min(valid,key=lambda r:r['training_loss'])
    history=read_json(source/'cable_physics/history.json');saved=next(r for r in history if r['update']==chosen['update'])
    if saved['values']!=chosen['parameters'] or saved['best_loss']!=chosen['training_loss']:raise ValueError('Review is not bound to the saved physical iterate')
    paths=[source/'drone_residual/state.pt',source/'drone_residual/history.json',source/'drone_residual/result.json',source/'drone_nominal/parameters.json',review]
    drone_result=read_json(source/'drone_residual/result.json')
    reuse_drone=(drone_result.get('stop_reason')=='practical_plateau' and
                 drone_result.get('numerically_verified') is True and
                 (source/'stages/drone_residual/model.json').is_file())
    retained=('stages/drone_nominal','drone_nominal','cable_physics')
    if reuse_drone:retained+=('stages/drone_residual','drone_residual','attitude_refinement','adapted_drone')
    paths += [f for folder in retained for f in (source/folder).rglob('*') if f.is_file()]
    hashes={str(f):sha256_file(f) for f in paths}
    c['residual_stopping']['ceiling']=None
    c['continuation']=dict(source=str(source),hashes=hashes,selected_physical=chosen,reuse_completed_drone=reuse_drone,
        reason='Final physical iterate failed numerical validation; best verified earlier saved iterate retained. Completed, verified drone stages are reused when already stopped at their declared plateau.',
        validation_used_for_selection=False)
    c['implementation']='whip_full_continuation_v1'
    return data.prepare(job,source,p['preliminary_source'],c)


def train_to_plateau(net,objective,contract,folder,job,label,*,resume=None):
    """No routine update ceiling. Preserve the original objective and Adam state."""
    folder=Path(folder);folder.mkdir(parents=True);settings=contract['residual_stopping']
    optimizer=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=1e-4)
    started=time.perf_counter();start_update=0;history=[]
    if resume is not None:
        source=Path(resume);saved=torch.load(source/'state.pt',map_location='cuda',weights_only=True)
        net.load_state_dict(saved['current']);optimizer.load_state_dict(saved['optimizer'])
        best=deepcopy(saved['best']);best_update=saved['best_update'];start_update=saved['update']
        stop=Plateau(**saved['plateau']);history=read_json(source/'history.json');baseline=history[0]['baseline_loss']
        score=float(objective().detach())
        if not np.isclose(score,history[-1]['selection_loss'],rtol=1e-7,atol=1e-8):raise ValueError('Resumed neural objective differs from the frozen run')
        atomic_json(folder/'resume.json',dict(source=str(source),source_state_sha256=sha256_file(source/'state.pt'),update=start_update,loss=score,optimizer_preserved=True))
    else:
        score=float(objective().detach());baseline=score;best=deepcopy(net.state_dict());best_update=0
        stop=Plateau(settings['minimum'],settings['patience'],settings['relative']);stop.observe(0,score)
    atomic_json(folder/'history.json',history)
    torch.save(dict(update=best_update,loss=stop.best,state_dict=best),folder/f'best-{best_update:06d}.pt')
    for update in itertools.count(start_update+1):
        progress(job,label,update=update,best_loss=stop.best,stopping='practical plateau; no update ceiling')
        optimizer.zero_grad();loss=objective()
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite '+label)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if update%settings['check_every']:
            continue
        with torch.no_grad():score=float(objective())
        improved,done=stop.observe(update,score)
        if improved:best=deepcopy(net.state_dict());best_update=update
        history.append(dict(update=update,loss=float(loss.detach()),best_loss=stop.best,selection_loss=score,
            baseline_loss=baseline,gradient_norm=float(norm),elapsed_s=time.perf_counter()-started,continued_update=update-start_update))
        atomic_json(folder/'history.json',history)
        state=dict(update=update,current=net.state_dict(),best=best,optimizer=optimizer.state_dict(),best_update=best_update,plateau=asdict(stop))
        torch.save(state,folder/'state.tmp');(folder/'state.tmp').replace(folder/'state.pt')
        # Keep independent best checkpoints for numerical verification, not validation selection.
        if improved:torch.save(dict(update=update,loss=score,state_dict=best),folder/f'best-{update:06d}.pt')
        if done:break
    net.load_state_dict(best)
    result=dict(updates=update,additional_updates=update-start_update,selected_update=best_update,best_loss=stop.best,
        baseline_loss=baseline,stop_reason='practical_plateau',elapsed_s=time.perf_counter()-started,ceiling=None)
    atomic_json(folder/'result.json',result);return result


def verify_selected_residual(net,objective,folder,result):
    """Select by training loss subject to numerical validity, never validation data."""
    folder=Path(folder);candidates=[];checks=[]
    for f in folder.glob('best-*.pt'):
        value=torch.load(f,map_location='cuda',weights_only=True);candidates.append((float(value['loss']),f,value))
    for loss,path,value in sorted(candidates,key=lambda item:item[0]):
        net.load_state_dict(value['state_dict'])
        try:
            check=gradient_check(net,objective,folder/f'gradient-{value["update"]:06d}.json')
            checks.append(dict(update=value['update'],loss=loss,passed=True))
            result.update(optimizer_best_loss=result['best_loss'],best_loss=loss,selected_update=value['update'],numerically_verified=True)
            atomic_json(folder/'numerical_selection.json',checks);atomic_json(folder/'result.json',result);return result
        except ValueError as exc:
            checks.append(dict(update=value['update'],loss=loss,passed=False,error=str(exc)))
            atomic_json(folder/'numerical_selection.json',checks)
    raise ValueError('No neural checkpoint passed the full-rollout numerical check')


def run(job):
    job=Path(job).resolve();model,p,engine=data.load(job);c=p['full_update'];continuation=c['continuation'];source=Path(continuation['source'])
    verify_hashes(continuation['hashes']);(job/'fit').mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(c['seed']);started=time.perf_counter()
    try:
        trials=data.drone_trials(job,model);rows=data.cable_rows(job,model,engine,trials);cd=data.cable_data(rows,c['replay_weight'])
        shutil.copytree(source/'drone_nominal',job/'drone_nominal')
        nominal=ResearchExecutionModel.from_mapping(read_json(source/'stages/drone_nominal/model.json'),root=source/'stages/drone_nominal',device='cuda')
        engine.drone=nominal.drone
        full.save_model(model,engine,job/'stages/drone_nominal',job)
        if continuation.get('reuse_completed_drone'):
            # A cable-stage numerical failure must not restart a completed
            # translation fit or silently change its selected checkpoint.
            fitted=ResearchExecutionModel.from_mapping(read_json(source/'stages/drone_residual/model.json'),
                root=source/'stages/drone_residual',device='cuda')
            engine.drone=fitted.drone
            dr=read_json(source/'drone_residual/result.json')
            if dr.get('stop_reason')!='practical_plateau' or dr.get('numerically_verified') is not True:
                raise ValueError('Completed verified drone fit required for reuse')
            for folder in ('drone_residual','attitude_refinement','adapted_drone'):
                shutil.copytree(source/folder,job/folder)
            if (source/'training_windows.json').exists():shutil.copy2(source/'training_windows.json',job/'training_windows.json')
            progress(job,'Reusing completed drone fit',selected_update=dr['selected_update'])
            full.save_model(model,engine,job/'stages/drone_residual',job)
        else:
            engine.drone.residual.requires_grad_(True)
            progress(job,'drone_residual graph preparation',windows=len(trials))
            # Capture the change regularizer against the inherited residual before resuming.
            batch=FullTranslation(trials,engine.drone.parameters,engine.drone.residual,c,count=1);objective=batch.residual_block()
            dr=train_to_plateau(engine.drone.residual,objective,c,job/'drone_residual',job,'drone_residual',resume=source/'drone_residual')
            dr=verify_selected_residual(engine.drone.residual,objective,job/'drone_residual',dr)
            engine.drone.residual.requires_grad_(False);del objective,batch;gc.collect();torch.cuda.empty_cache()
            engine.drone.parameters=fit_attitude(trials,engine.drone.parameters,engine.drone.residual,c,job/'attitude_refinement',job)
            full.save_model(model,engine,job/'stages/drone_residual',job)
            full.compare_drone_runtime(trials,engine,c,job/'adapted_drone')
        cable=np.array(continuation['selected_physical']['parameters'])
        shutil.copytree(source/'cable_physics',job/'cable_physics');cr=read_json(job/'cable_physics/result.json')
        cr.update(best=np.log(cable).tolist(),best_loss=continuation['selected_physical']['training_loss'],selected_update=continuation['selected_physical']['update'],
            selection='Best training loss among numerically verified physical iterates',rejected_final=True,original_result=str(source/'cable_physics/result.json'))
        atomic_json(job/'cable_physics/result.json',cr)
        physical=full.save_model(model,engine,job/'stages/cable_physics',job,cable_parameters=cable)
        del engine;gc.collect();torch.cuda.empty_cache()
        engine=ResearchExecutionModel.from_mapping(physical,root=job/'stages/cable_physics',device='cuda');net=make_residual(engine,c)
        progress(job,'cable_residual graph preparation',windows=len(rows))
        objective,accelerator=residual_objective(engine,cd,cable,c)
        gradient_check(net,objective,job/'cable_residual_initial_gradient_check.json')
        rr=train_to_plateau(net,objective,c,job/'cable_residual',job,'cable_residual')
        rr=verify_selected_residual(net,objective,job/'cable_residual',rr)
        net.requires_grad_(False)
        candidate=full.save_model(model,engine,job/'candidate',job,cable_parameters=cable,cable_net=net)
        candidate['provenance']['fit_complete']=True;atomic_json(job/'candidate/model.json',candidate)
        if full.immutable_identity(job/'candidate/model.json')!=full.immutable_identity(job/'source_candidate/model.json'):raise ValueError('Candidate changed unreviewed fields')
        full.validate_candidate(job/'candidate/model.json',c)
        selection=dict(schema=data.FULL_SCHEMA,stages_complete=True,candidate_sha256=sha256_file(job/'candidate/model.json'),
            training_takes=[n for n,r in p['takes'].items() if r['role']=='adaptation'],validation_used_for_selection=False,
            drone_parameters=asdict(engine.drone.parameters),cable_parameters=cable.tolist(),drone_residual=dr,cable_physics=cr,cable_residual=rr)
        atomic_json(job/'fit/selection_frozen.json',selection)
        del objective,accelerator,net,engine,cd;gc.collect();torch.cuda.empty_cache()
        output=data.ROOT/'runs/evaluation'/job.name;progress(job,'combined_validation')
        full.evaluate_pair(job,output,{'M0':job/'source_candidate/model.json',c['candidate_id']:job/'candidate/model.json'})
        full.evaluate_preliminary_retention(job,{'M0':job/'source_candidate/model.json',c['candidate_id']:job/'candidate/model.json'},c)
        _,hashes=model_identity(job/'candidate/model.json')
        for f in (job/'fit/selection_frozen.json',output/'report.json',job/'preliminary_validation/report.json'):hashes[str(f)]=sha256_file(f)
        result=dict(status='completed',schema=data.FULL_SCHEMA,training_takes=selection['training_takes'],candidate_hashes=hashes,stages=selection,
            comparison=str(output),elapsed_s=time.perf_counter()-started,model_selected=False,continuation=str(source))
        atomic_json(job/'fit/result.json',result);registration=full.register(job,output);atomic_json(job/'registration.json',registration)
        atomic_json(job/'status.json',dict(status='completed',stage='Full adaptation and combined validation complete',elapsed_s=result['elapsed_s'],registration=registration,model_selected=False))
        return result
    except BaseException as exc:
        atomic_json(job/'status.json',dict(status='stopped' if isinstance(exc,InterruptedError) else 'failed',error=str(exc),model_selected=False));raise
