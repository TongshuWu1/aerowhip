"""Finalize M3's saved best checkpoint after the user's explicit stop request."""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
import sys,gc,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experimental_data import whip_full_data as data,whip_full_fit as full
from experimental_data.whip_full_cable import make_residual,residual_objective
from experimental_data.whip_full_continuation import verify_selected_residual
from experimental_data.io import atomic_json,sha256_file
from experimental_data.model_evaluation import model_identity,load_catalog,save_catalog
from simulator.research_execution import ResearchExecutionModel
from simulator.workflow import read_json

torch.set_num_threads(4)
job=ROOT/'runs/adaptation/M3-20260915';started=time.perf_counter()
assert not (job/'candidate').exists()
history=read_json(job/'cable_residual/history.json')
saved=torch.load(job/'cable_residual/state.pt',map_location='cpu',weights_only=True)
assert saved['update']==history[-1]['update']
receipt=dict(reason='User requested stopping because remaining improvement is small; retain best saved checkpoint.',
    termination='user_requested_early_stop',convergence_claimed=False,
    last_saved_update=saved['update'],best_saved_update=saved['best_update'],
    state_sha256=sha256_file(job/'cable_residual/state.pt'),history_sha256=sha256_file(job/'cable_residual/history.json'),
    original_status=read_json(job/'status.json'),finalizer_sha256=sha256_file(Path(__file__)),
    interrupted_worker_pid=45400,unsaved_updates_discarded=True)
atomic_json(job/'fit/user_stop.json',receipt)
atomic_json(job/'status.json',dict(status='running',stage='Verifying best checkpoint after user-requested stop',
                                 training_stopped=True,model_selected=False))
try:
    model,p,parent=data.load(job);contract=p['full_update']
    trials=data.drone_trials(job,model)
    cd=data.cable_data(data.cable_rows(job,model,parent,trials),contract['replay_weight'])
    path=job/'stages/cable_physics/model.json';physical=read_json(path)
    engine=ResearchExecutionModel.from_mapping(physical,root=path.parent,device='cuda')
    net=make_residual(engine,contract)
    cable=[physical['cable'][key] for key in ('EI_n_m2','Cb_n_m2_s','external_drag_s_inv')]
    # Construct the change penalty against inherited M2 weights before loading
    # trained weights, so finalization reproduces the original fitting loss.
    objective,accelerator=residual_objective(engine,cd,cable,contract)
    net.load_state_dict(saved['best'])
    with torch.no_grad():reproduced=float(objective())
    assert np.isclose(reproduced,saved['plateau']['best'],atol=1e-8,rtol=1e-7)
    receipt['reproduced_best_loss']=reproduced;atomic_json(job/'fit/user_stop.json',receipt)
    rr=dict(updates=saved['update'],selected_update=saved['best_update'],best_loss=saved['plateau']['best'],
        baseline_loss=history[0]['baseline_loss'],elapsed_s=history[-1]['elapsed_s'],
        stop_reason='user_requested_early_stop',ceiling=None)
    rr=verify_selected_residual(net,objective,job/'cable_residual',rr);net.requires_grad_(False)
    candidate=full.save_model(model,engine,job/'candidate',job,cable_parameters=cable,cable_net=net)
    candidate['provenance'].update(fit_complete=True,termination='User-requested early stop; best numerically verified saved checkpoint')
    atomic_json(job/'candidate/model.json',candidate)
    assert full.immutable_identity(job/'candidate/model.json')==full.immutable_identity(job/'source_candidate/model.json')
    full.validate_candidate(job/'candidate/model.json',contract)
    dr=read_json(job/'drone_residual/result.json');cr=read_json(job/'cable_physics/result.json')
    selection=dict(schema=data.FULL_SCHEMA,stages_complete=True,candidate_sha256=sha256_file(job/'candidate/model.json'),
        training_takes=[n for n,r in p['takes'].items() if r['role']=='adaptation'],validation_used_for_selection=False,
        drone_parameters=asdict(engine.drone.parameters),cable_parameters=cable,drone_residual=dr,
        cable_physics=cr,cable_residual=rr,termination_override=receipt)
    atomic_json(job/'fit/selection_frozen.json',selection)
    _,hashes=model_identity(job/'candidate/model.json')
    for f in (job/'fit/user_stop.json',job/'fit/selection_frozen.json',job/'cable_residual/numerical_selection.json',
              job/'drone_residual/result.json',job/'cable_residual/result.json',Path(__file__)):
        hashes[str(f)]=sha256_file(f)
    result=dict(status='completed',schema=data.FULL_SCHEMA,training_takes=selection['training_takes'],candidate_hashes=hashes,
        stages=selection,comparison=None,performance_evaluation='deferred_by_user',model_selected=False,
        termination='user_requested_early_stop',convergence_claimed=False,
        elapsed_s=time.perf_counter()-started,elapsed_scope='Finalization only; training times retained per stage')
    atomic_json(job/'fit/result.json',result)
    catalog=load_catalog(ROOT);parent_row=next(m for m in catalog['models'] if m['id']==contract['parent_id'])
    assert model_identity(job/'source_candidate/model.json')[0]==parent_row['signature']
    assert not any(m['id']==contract['candidate_id'] for m in catalog['models'])
    raw=read_json(job/'source_hashes.json');sources=set(parent_row['training_sources'])
    for name in selection['training_takes']:
        found=[h for f,h in raw.items() if Path(f).name==name+'.csv'];assert found;sources.update(found)
    signature,_=model_identity(job/'candidate/model.json')
    state='Candidate - user-requested early stop; next-flight evaluation pending'
    catalog['models'].append(dict(id=contract['candidate_id'],parent=contract['parent_id'],
        generation_index=parent_row['generation_index']+1,candidate_variant='full',model=str(job/'candidate/model.json'),
        signature=signature,hashes=hashes,job=str(job),training_sources=sorted(sources),status=state,
        created_at=datetime.now(timezone.utc).isoformat()))
    save_catalog(ROOT,catalog)
    registration=dict(id=contract['candidate_id'],status=state,promoted=False)
    atomic_json(job/'registration.json',registration)
    atomic_json(job/'status.json',dict(status='completed',stage='M3 saved after user-requested early stop',
        training_stopped=True,model_selected=False,registration=registration,selected_cable_update=rr['selected_update']))
    print('M3 saved:',job/'candidate/model.json',flush=True)
    print('Cable result:',rr,flush=True)
except BaseException as exc:
    atomic_json(job/'status.json',dict(status='failed',stage='Finalization after user stop',training_stopped=True,error=str(exc)))
    raise
