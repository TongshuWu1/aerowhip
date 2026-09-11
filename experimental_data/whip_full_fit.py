"""Complete staged M0 adaptation with frozen selection before validation."""
from pathlib import Path
from dataclasses import asdict,replace
from copy import deepcopy
from datetime import datetime,timezone
import gc,time
import numpy as np
import torch
from . import whip_full_data as data
from .whip_full_optim import FullTranslation,fit_nominal,fit_attitude,train_residual,progress
from .whip_full_cable import fit_physics,make_residual,residual_objective,gradient_check
from .whip_adaptation import verify_hashes
from . import whip_adaptation_fit as old_evaluation
from .io import atomic_json,sha256_file
from .model_evaluation import model_identity,load_catalog,save_catalog,summarize_diagnostics,REPORT,digest
from simulator.workflow import read_json
from simulator.research_execution import ResearchExecutionModel
from simulator.drone_pose_residual import save_residual
from planning.pva_job import freeze_model_assets


def save_model(model,engine,folder,job,*,cable_parameters=None,cable_net=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    m=freeze_model_assets(deepcopy(model),folder)
    dp=Path(m['fullstate_execution']['checkpoint']);d=read_json(dp)
    save_residual(dp.parent/'drone_residual.pt',engine.drone.residual)
    d['nominal']['parameters']=asdict(engine.drone.parameters)
    p=read_json(Path(job)/'protocol.json')
    names=[n for n,r in p['takes'].items() if r['role']=='adaptation']
    d['nominal']['training_takes']=list(dict.fromkeys(d['nominal'].get('training_takes',[])+names))
    d['residual'].update(checkpoint='drone_residual.pt',sha256=sha256_file(dp.parent/'drone_residual.pt'),specification=engine.drone.residual.specification())
    atomic_json(dp,d);m['fullstate_execution']['sha256']=sha256_file(dp)
    if cable_parameters is not None:m['cable'].update(dict(zip(('EI_n_m2','Cb_n_m2_s','external_drag_s_inv'),map(float,cable_parameters))))
    if cable_net is not None:
        file=folder/'cable_residual.pt'
        torch.save(dict(specification=cable_net.specification(),state_dict={k:v.detach().cpu() for k,v in cable_net.state_dict().items()}),file)
        m['motion_residual']=dict(enabled=True,checkpoint=str(file),sha256=sha256_file(file),specification=cable_net.specification())
    c=p['full_update']
    m.setdefault('provenance',{}).update(label=c['candidate_id']+' full adaptation candidate',generation_index=int(p['parent_generation'])+1,
        parent_model_sha256=sha256_file(Path(job)/'source_candidate/model.json'),source_job=str(job),training_takes=names,
        fit_complete=False,selected_model=False,flight_ready=False,prospective_flight_evidence=False,
        update_schema=data.FULL_SCHEMA,candidate_variant='full')
    atomic_json(folder/'model.json',m);return m


def immutable_identity(path):
    """Only explicitly fitted quantities may differ; geometry and numerical model stay bound."""
    path=Path(path);m=deepcopy(read_json(path));m.pop('provenance',None)
    for k in ('EI_n_m2','Cb_n_m2_s','external_drag_s_inv'):m['cable'].pop(k)
    m.pop('motion_residual',None)
    spec=m['fullstate_execution'];dp=Path(spec.pop('checkpoint'));dp=dp if dp.is_absolute() else path.parent/dp
    spec.pop('sha256',None)
    d=read_json(dp);d['nominal'].pop('training_takes',None)
    for k in ('kp_xy','kp_z','kd_xy','kd_z','feedforward_xy','feedforward_z','delay_s','attitude_time_constant_s','attitude_acceleration_scale_xy','attitude_acceleration_scale_z'):
        d['nominal']['parameters'].pop(k)
    d['residual'].pop('checkpoint');d['residual'].pop('sha256')
    spec['component']=d
    return digest(m)


def validate_candidate(path,contract):
    m=read_json(path);engine=ResearchExecutionModel.from_mapping(m,root=Path(path).parent,device='cuda')
    spec=deepcopy(m['motion_residual']['specification']);spec.setdefault('mode','acceleration');expected=contract['cable_residual']
    if not m['motion_residual'].get('enabled') or any(spec.get(k)!=v for k,v in expected.items()):raise ValueError('Cable residual differs from frozen architecture')
    if spec.get('learn_drag') or spec.get('initial_drag_s_inv',0)!=0:raise ValueError('Unexpected second cable damping term')
    from .nominal_pose_fit import GAIN_NAMES
    groups=[([getattr(engine.drone.parameters,k) for k in GAIN_NAMES],contract['drone_gain_bounds']),
        ([engine.drone.parameters.attitude_acceleration_scale_xy,engine.drone.parameters.attitude_acceleration_scale_z,engine.drone.parameters.attitude_time_constant_s],contract['attitude_bounds']),
        ([m['cable'][k] for k in ('EI_n_m2','Cb_n_m2_s','external_drag_s_inv')],contract['cable_bounds'])]
    for values,bounds in groups:
        if not np.isfinite(values).all() or np.any(np.asarray(values)<np.asarray(bounds[0])*(1-1e-8)) or np.any(np.asarray(values)>np.asarray(bounds[1])*(1+1e-8)):raise ValueError('Fitted values outside declared model bounds')
    if engine.drone.parameters.delay_s not in contract['delay_candidates_s']:raise ValueError('Undeclared effective delay')
    return engine


def compare_drone_runtime(trials,engine,contract,folder):
    folder=Path(folder);folder.mkdir(exist_ok=False);reports={}
    with torch.no_grad():
        batch=FullTranslation(trials,engine.drone.parameters,engine.drone.residual,contract,count=1)
        prediction=batch.rollout()[0][0].cpu().numpy()
        for i,t in enumerate(trials):
            d=t.data;result=engine.drone.predict(t.initial_pose(engine.drone.parameters),
                torch.as_tensor(d['packets'][None],device='cuda',dtype=torch.float64),d['packet_time'],t.time,engine.offset,graph=True,
                hover_command=torch.as_tensor(d['hover_commands'][-1:],device='cuda',dtype=torch.float64))
            position=result['position_origin_m'][0].cpu().numpy();diff=float(abs(position-prediction[i,:len(t.time)]).max())
            if diff>3e-5:raise ValueError(f'{t.name}: full drone training/production mismatch {diff}')
            mask=t.masks['fit_position'];e=position[mask]-t.truth['position_origin_m'][mask]
            from scipy.spatial.transform import Rotation
            dr=Rotation.from_matrix(t.truth['rotation_tracking_to_world'][mask].transpose(0,2,1)@result['rotation_tracking_to_world'][0,mask].cpu().numpy()).as_rotvec()
            reports[t.name]=dict(take=t.take,category=t.category,position_rms_xyz_m=np.sqrt(np.mean(e*e,axis=0)).tolist(),
                position_rms_m=float(np.sqrt(np.mean(np.sum(e*e,axis=1)))),attitude_rms_rad=float(np.sqrt(np.mean(np.sum(dr*dr,axis=1)))),
                final_position_error_m=e[-1].tolist(),production_max_difference_m=diff)
    atomic_json(folder/'report.json',reports);return reports


def evaluate_pair(job,output,models,device='cuda'):
    job=Path(job).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    model,p,baseline=data.load(job,device)
    rows=old_evaluation.records(job,list(p['takes']),model,baseline,device);ids=list(baseline.cable.marker_node_indices[1:])
    report=dict(schema=REPORT,job=str(job),protocol=p,models={},device=device,
        evidence='Reinitialized same-flight predictions. Whole validation takes excluded from fitting; prior method-development inspection disclosed.',
        aggregation='Equal take weight, fixed masks; no frame-level confidence intervals',comparison_key=digest(read_json(job/'prepared_hashes.json')))
    hashes={};identity=immutable_identity(job/'source_candidate/model.json')
    for name,path in models.items():
        if (output/'STOP').exists():raise InterruptedError('Comparison stopped between models')
        atomic_json(output/'progress.json',dict(label='Evaluating '+name,completed=len(report['models']),total=len(models)))
        print('combined validation',name,flush=True)
        if immutable_identity(path)!=identity:raise ValueError('Full comparison changes geometry or unreviewed fields')
        value=read_json(path);engine=ResearchExecutionModel.from_mapping(value,root=Path(path).parent,device=device)
        old_evaluation.evaluate(rows,engine,value['cable']['external_drag_s_inv'],output/name)
        native=[data.WhipPoseTrial(job,n,value,device) for n in p['takes']]
        pose=compare_drone_runtime(native,engine,p['full_update'],output/name/'native_pose')
        metrics=summarize_diagnostics(output/name,p,ids)
        for t,r in metrics.items():
            r['data_use']='Adaptation training' if r['role']=='adaptation' else 'Held out of optimization; method-development reuse disclosed'
            r['native_pose']=pose[t]
        signature,mhash=model_identity(path);hashes.update(mhash)
        report['models'][name]=dict(model=dict(id=name,model=str(path),signature=signature),takes=metrics,marker_ids=ids)
    for n in ('prepared_hashes.json','source_hashes.json','code_hashes.json'):hashes[str(job/n)]=sha256_file(job/n)
    for f in output.glob('**/*'):
        if f.is_file():hashes[str(f)]=sha256_file(f)
    atomic_json(output/'report.json',report);hashes[str(output/'report.json')]=sha256_file(output/'report.json')
    atomic_json(output/'evidence_hashes.json',hashes);atomic_json(output/'status.json',dict(status='completed'))
    return report


def evaluate_preliminary_retention(job,models,contract):
    """Post-selection check on the original whole-take preliminary holdout."""
    job=Path(job);out=job/'preliminary_validation';out.mkdir();report={}
    source=read_json(job/'source_candidate/model.json')
    trials=[]
    for w in read_json(job/'replay/validation_windows.json'):
        if w['role']!='validation':raise ValueError('Unexpected preliminary validation role')
        t=data.PreliminaryTrial(job/'replay',w,source,'cuda');t.category='preliminary';trials.append(t)
    baseline=ResearchExecutionModel.from_mapping(source,root=job/'source_candidate',device='cuda')
    rows=data.cable_rows(job,source,baseline,trials,review_path=out/'cable_window_review.json')
    for name,path in models.items():
        progress(job,'preliminary validation',model=name,windows=len(trials))
        m=read_json(path);engine=ResearchExecutionModel.from_mapping(m,root=Path(path).parent,device='cuda')
        pose=compare_drone_runtime(trials,engine,contract,out/name)
        q=old_evaluation.CableBatch(engine,rows,1)([m['cable']['external_drag_s_inv']])[:,0];metrics=[]
        for i,r in enumerate(rows):
            pred=q[i,:len(r['grid']),list(engine.cable.marker_node_indices[1:])].cpu().numpy()
            error=np.linalg.norm(pred-r['truth'][:,1:],axis=-1);mask=r['valid']
            metrics.append(dict(window=r['name'],take=r['take'],conditional_markers_rms_m=float(np.sqrt(np.mean(error[mask]**2))),
                conditional_tip_rms_m=float(np.sqrt(np.mean(error[:,-1][mask[:,-1]]**2)))))
        report[name]=dict(drone=pose,conditional_cable=metrics,qualification='Original preliminary validation take; excluded from all updates and selection, previously inspected during M0 development')
    atomic_json(out/'report.json',report);return report


def validation_changes(report,protocol,parent,candidate):
    names=[t for t,v in protocol['takes'].items() if v['role']=='validation']
    if not names:raise ValueError('Full model registration needs held-out takes')
    by_take={t:{metric:[report['models'][k]['takes'][t]['metrics'][metric]['rmse_m'] for k in (parent,candidate)]
        for metric in ('drone','command_driven_tip')} for t in names}
    mean={metric:np.mean([r[metric] for r in by_take.values()],axis=0).tolist() for metric in ('drone','command_driven_tip')}
    return dict(takes=by_take,equal_take_mean=mean)


def evaluate_prior_retention(job,models,device='cuda'):
    job=Path(job);p=read_json(job/'protocol.json');report={}
    for index,previous in enumerate(p.get('prior_whip_replay',[])):
        folder=Path(previous['folder']);op=read_json(folder/'protocol.json')
        names=[n for n,r in op['takes'].items() if r['role']=='validation']
        baseline=read_json(job/'source_candidate/model.json')
        engine=ResearchExecutionModel.from_mapping(baseline,root=job/'source_candidate',device=device)
        rows=old_evaluation.records(folder,names,baseline,engine,device)
        result={}
        for name,path in models.items():
            progress(job,'prior whip validation',model=name,takes=names)
            m=read_json(path);e=ResearchExecutionModel.from_mapping(m,root=Path(path).parent,device=device)
            result[name]=old_evaluation.evaluate(rows,e,m['cable']['external_drag_s_inv'],job/'prior_validation'/str(index)/name)
        report[str(index)]=dict(source=previous['source'],takes=names,models=result,
            qualification='Prior whole-take holdout; excluded from replay, already inspected during development')
    atomic_json(job/'prior_validation/report.json',report)


def register(job,comparison):
    job=Path(job).resolve();p=read_json(job/'protocol.json');result=read_json(job/'fit/result.json')
    if result['status']!='completed':raise ValueError('Incomplete fit')
    verify_hashes(result['candidate_hashes']);catalog=load_catalog(data.ROOT)
    name=p['full_update']['candidate_id'];parent=p['full_update']['parent_id']
    existing=next((m for m in catalog['models'] if m['id']==name),None)
    if existing:
        if existing['signature']!=model_identity(job/'candidate/model.json')[0]:raise ValueError('A different candidate variant already uses this ID')
        return dict(id=name,status=existing['status'],promoted=False,already_registered=True)
    baseline=next(m for m in catalog['models'] if m['id']==parent)
    if model_identity(job/'source_candidate/model.json')[0]!=baseline['signature']:raise ValueError('Wrong catalog parent')
    signature,hashes=model_identity(job/'candidate/model.json');hashes.update(result['candidate_hashes'])
    sources=set(baseline['training_sources']);raw=read_json(job/'source_hashes.json')
    for take in result['training_takes']:
        found=[h for f,h in raw.items() if Path(f).name==take+'.csv']
        if not found:raise ValueError('Missing training-source ancestry')
        sources.update(found)
    r=read_json(Path(comparison)/'report.json');changes=validation_changes(r,p,parent,name)
    improved=all(b<a for a,b in changes['equal_take_mean'].values())
    state='Candidate - validation improved; prospective flight pending' if improved else 'Not promoted - validation did not improve both drone and tip'
    generation=int(baseline['generation_index'])+1
    if generation!=int(p['parent_generation'])+1:raise ValueError('Wrong parent generation')
    catalog['models'].append(dict(id=name,parent=parent,generation_index=generation,candidate_variant='full',model=str(job/'candidate/model.json'),
        signature=signature,hashes=hashes,job=str(job),training_sources=sorted(sources),status=state,created_at=datetime.now(timezone.utc).isoformat()))
    save_catalog(data.ROOT,catalog);return dict(id=name,status=state,validation=changes,promoted=False)


def fit(job,device='cuda'):
    if device!='cuda' or not torch.cuda.is_available():raise ValueError('Full adaptation requires the reviewed CUDA runtime')
    job=Path(job).resolve();model,p,engine=data.load(job,device);contract=p['full_update'];started=time.perf_counter()
    (job/'fit').mkdir(exist_ok=False);torch.set_num_threads(4);torch.manual_seed(contract['seed'])
    try:
        from .whip_full_continuation import train_to_plateau,verify_selected_residual
        def train(net,objective,folder,label):
            if contract['residual_stopping']['ceiling'] is not None:
                return train_residual(net,objective,contract,folder,job,label)
            result=train_to_plateau(net,objective,contract,folder,job,label)
            result=verify_selected_residual(net,objective,folder,result)
            net.requires_grad_(False)
            return result
        progress(job,'preparing full model training');trials=data.drone_trials(job,model,device)
        rows=data.cable_rows(job,model,engine,trials);cd=data.cable_data(rows,contract['replay_weight'])
        atomic_json(job/'training_windows.json',[dict(name=t.name,take=t.take,category=t.category,role=t.role,weight=float(t.weights.sum()/len(trials))) for t in trials])
        # Record baseline before any candidate update, training data only.
        compare_drone_runtime(trials,engine,contract,job/'baseline_drone')
        nominal=fit_nominal(trials,engine,contract,job/'drone_nominal',job);engine.drone.parameters=nominal
        save_model(model,engine,job/'stages/drone_nominal',job)
        engine.drone.residual.requires_grad_(True)
        progress(job,'drone_residual graph preparation',windows=len(trials))
        batch=FullTranslation(trials,nominal,engine.drone.residual,contract,count=1);objective=batch.residual_block()
        gradient_check(engine.drone.residual,objective,job/'drone_residual_gradient_check.json')
        dr=train(engine.drone.residual,objective,job/'drone_residual','drone_residual')
        del objective,batch;gc.collect();torch.cuda.empty_cache()
        engine.drone.parameters=fit_attitude(trials,nominal,engine.drone.residual,contract,job/'attitude_refinement',job)
        drone_model=save_model(model,engine,job/'stages/drone_residual',job)
        compare_drone_runtime(trials,engine,contract,job/'adapted_drone')
        cable,cr=fit_physics(engine,cd,model,contract,job/'cable_physics',job)
        physical=save_model(model,engine,job/'stages/cable_physics',job,cable_parameters=cable)
        del engine;gc.collect();torch.cuda.empty_cache()
        engine=ResearchExecutionModel.from_mapping(physical,root=job/'stages/cable_physics',device=device)
        net=make_residual(engine,contract)
        progress(job,'cable_residual graph preparation',windows=len(rows))
        objective,accelerator=residual_objective(engine,cd,cable,contract)
        gradient_check(net,objective,job/'cable_residual_gradient_check.json')
        rr=train(net,objective,job/'cable_residual','cable_residual')
        candidate=save_model(model,engine,job/'candidate',job,cable_parameters=cable,cable_net=net)
        candidate['provenance']['fit_complete']=True;atomic_json(job/'candidate/model.json',candidate)
        if immutable_identity(job/'candidate/model.json')!=immutable_identity(job/'source_candidate/model.json'):raise ValueError('Candidate changed unreviewed fields')
        validate_candidate(job/'candidate/model.json',contract)
        # Native fit states, current/best checkpoints and the exact frozen selection are retained.
        selection=dict(schema=data.FULL_SCHEMA,stages_complete=True,candidate_sha256=sha256_file(job/'candidate/model.json'),
            training_takes=[n for n,r in p['takes'].items() if r['role']=='adaptation'],validation_used_for_selection=False,
            drone_parameters=asdict(engine.drone.parameters),cable_parameters=cable.tolist(),drone_residual=dr,cable_physics=cr,cable_residual=rr)
        atomic_json(job/'fit/selection_frozen.json',selection)
        del objective,accelerator,net,engine,cd;gc.collect();torch.cuda.empty_cache()
        output=data.ROOT/'runs/evaluation'/job.name
        progress(job,'combined_validation')
        models={contract['parent_id']:job/'source_candidate/model.json',contract['candidate_id']:job/'candidate/model.json'}
        report=evaluate_pair(job,output,models,device)
        evaluate_preliminary_retention(job,models,contract)
        evaluate_prior_retention(job,models,device)
        _,hashes=model_identity(job/'candidate/model.json')
        hashes.update({str(job/'fit/selection_frozen.json'):sha256_file(job/'fit/selection_frozen.json'),str(output/'report.json'):sha256_file(output/'report.json'),
            str(job/'preliminary_validation/report.json'):sha256_file(job/'preliminary_validation/report.json'),
            str(job/'prior_validation/report.json'):sha256_file(job/'prior_validation/report.json')})
        result=dict(status='completed',schema=data.FULL_SCHEMA,training_takes=selection['training_takes'],candidate_hashes=hashes,
            stages=selection,comparison=str(output),elapsed_s=time.perf_counter()-started,model_selected=False)
        atomic_json(job/'fit/result.json',result)
        registration=register(job,output);atomic_json(job/'registration.json',registration)
        atomic_json(job/'status.json',dict(status='completed',stage='All adaptation stages and combined validation complete',
            elapsed_s=result['elapsed_s'],model_selected=False,registration=registration))
        return result
    except BaseException as exc:
        atomic_json(job/'status.json',dict(status='stopped' if isinstance(exc,InterruptedError) else 'failed',error=str(exc),model_selected=False))
        raise
