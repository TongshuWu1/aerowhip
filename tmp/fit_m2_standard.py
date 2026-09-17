import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
"""Complete staged M0 adaptation with frozen selection before validation."""
from pathlib import Path
from dataclasses import asdict,replace
from copy import deepcopy
from datetime import datetime,timezone
import gc,time
import numpy as np
import torch
from experimental_data import whip_full_data as data
from experimental_data.whip_full_optim import FullTranslation,fit_nominal,fit_attitude,train_residual,progress
from experimental_data.whip_full_cable import fit_physics,make_residual,residual_objective,gradient_check
from experimental_data.whip_adaptation import verify_hashes
from experimental_data import whip_adaptation_fit as old_evaluation
from experimental_data.io import atomic_json,sha256_file
from experimental_data.model_evaluation import model_identity,load_catalog,save_catalog,summarize_diagnostics,REPORT,digest
from simulator.workflow import read_json
from simulator.research_execution import ResearchExecutionModel
from simulator.drone_pose_residual import save_residual
from planning.pva_job import freeze_model_assets


from experimental_data.whip_full_fit import save_model, immutable_identity, validate_candidate, compare_drone_runtime, evaluate_pair, evaluate_preliminary_retention, evaluate_prior_retention, register

def fit(job,device='cuda'):
    if read_json(Path(job)/'protocol.json').get('diagnostics_only'):raise ValueError('Final diagnostic recordings cannot enter a fit')
    if device!='cuda' or not torch.cuda.is_available():raise ValueError('Full adaptation requires the reviewed CUDA runtime')
    job=Path(job).resolve();model,p,engine=data.load(job,device);contract=p['full_update'];started=time.perf_counter()
    assert sha256_file(Path(__file__))==contract['runner_sha256']
    (job/'fit').mkdir(exist_ok=False);torch.set_num_threads(4);torch.manual_seed(contract['seed'])
    try:
        from experimental_data.whip_full_continuation import train_to_plateau,verify_selected_residual
        def train(net,objective,folder,label):
            result=train_to_plateau(net,objective,contract,folder,job,label)
            result=verify_selected_residual(net,objective,folder,result)
            net.requires_grad_(False)
            return result
        progress(job,'preparing full model training');trials=data.drone_trials(job,model,device)
        rows=data.cable_rows(job,model,engine,trials);cd=data.cable_data(rows,contract['replay_weight'])
        atomic_json(job/'training_windows.json',[dict(name=t.name,take=t.take,category=t.category,role=t.role,weight=float(t.weights.sum()/len(trials))) for t in trials])
        # Record baseline before any candidate update, training data only.
        nominal=fit_nominal(trials,engine,contract,job/'drone_nominal',job);engine.drone.parameters=nominal
        save_model(model,engine,job/'stages/drone_nominal',job)
        engine.drone.residual.requires_grad_(True)
        progress(job,'drone_residual graph preparation',windows=len(trials))
        batch=FullTranslation(trials,nominal,engine.drone.residual,contract,count=1);objective=batch.residual_block()
        dr=train(engine.drone.residual,objective,job/'drone_residual','drone_residual')
        del objective,batch;gc.collect();torch.cuda.empty_cache()
        engine.drone.parameters=fit_attitude(trials,nominal,engine.drone.residual,contract,job/'attitude_refinement',job)
        drone_model=save_model(model,engine,job/'stages/drone_residual',job)
        cable,cr=fit_physics(engine,cd,model,contract,job/'cable_physics',job)
        physical=save_model(model,engine,job/'stages/cable_physics',job,cable_parameters=cable)
        del engine;gc.collect();torch.cuda.empty_cache()
        engine=ResearchExecutionModel.from_mapping(physical,root=job/'stages/cable_physics',device=device)
        net=make_residual(engine,contract)
        progress(job,'cable_residual graph preparation',windows=len(rows))
        objective,accelerator=residual_objective(engine,cd,cable,contract)
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
        # User explicitly deferred performance comparisons; no placeholder scores.
        _,hashes=model_identity(job/'candidate/model.json')
        hashes[str(job/'fit/selection_frozen.json')]=sha256_file(job/'fit/selection_frozen.json')
        hashes[str(Path(__file__).resolve())]=sha256_file(Path(__file__))
        result=dict(status='completed',schema=data.FULL_SCHEMA,training_takes=selection['training_takes'],
            candidate_hashes=hashes,stages=selection,comparison=None,performance_evaluation='deferred_by_user',
            elapsed_s=time.perf_counter()-started,model_selected=False)
        atomic_json(job/'fit/result.json',result)
        catalog=load_catalog(data.ROOT)
        parent=next(m for m in catalog['models'] if m['id']==contract['parent_id'])
        assert model_identity(job/'source_candidate/model.json')[0]==parent['signature']
        assert not any(m['id']==contract['candidate_id'] for m in catalog['models'])
        raw=read_json(job/'source_hashes.json');sources=set(parent['training_sources'])
        for name in selection['training_takes']:
            found=[h for f,h in raw.items() if Path(f).name==name+'.csv']
            assert found;sources.update(found)
        signature,_=model_identity(job/'candidate/model.json')
        state='Candidate - performance evaluation deferred; next flight pending'
        catalog['models'].append(dict(id=contract['candidate_id'],parent=contract['parent_id'],
            generation_index=parent['generation_index']+1,candidate_variant='full',model=str(job/'candidate/model.json'),
            signature=signature,hashes=hashes,job=str(job),training_sources=sorted(sources),status=state,
            created_at=datetime.now(timezone.utc).isoformat()))
        save_catalog(data.ROOT,catalog)
        registration=dict(id=contract['candidate_id'],status=state,promoted=False)
        atomic_json(job/'registration.json',registration)
        atomic_json(job/'status.json',dict(status='completed',stage='M2 fit complete with standard stopping; performance comparison deferred',
            elapsed_s=result['elapsed_s'],model_selected=False,registration=registration))
        return result
    except BaseException as exc:
        atomic_json(job/'status.json',dict(status='stopped' if isinstance(exc,InterruptedError) else 'failed',error=str(exc),model_selected=False))
        raise

if __name__=="__main__":
    fit(Path(__file__).resolve().parents[1]/"runs/adaptation/M2-20260915")
