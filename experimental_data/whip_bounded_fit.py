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


from .whip_full_fit import save_model, immutable_identity, validate_candidate, compare_drone_runtime, evaluate_pair, evaluate_preliminary_retention, evaluate_prior_retention, register

def fit(job,device='cuda'):
    if read_json(Path(job)/'protocol.json').get('diagnostics_only'):raise ValueError('Final diagnostic recordings cannot enter a fit')
    if device!='cuda' or not torch.cuda.is_available():raise ValueError('Full adaptation requires the reviewed CUDA runtime')
    job=Path(job).resolve();model,p,engine=data.load(job,device);contract=p['full_update'];started=time.perf_counter()
    (job/'fit').mkdir(exist_ok=False);torch.set_num_threads(4);torch.manual_seed(contract['seed'])
    try:
        from .whip_bounded_training import train_bounded
        def train(net,objective,folder,label):
            return train_bounded(net,objective,contract,folder,job,label)
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
