"""Finalize saved training weights after the user's explicit early-stop request."""
from pathlib import Path
import sys, gc, time
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from experimental_data import whip_full_data as data, whip_full_fit as full
from experimental_data.whip_full_cable import make_residual,residual_objective
from experimental_data.whip_full_continuation import verify_selected_residual
from experimental_data.io import atomic_json,sha256_file
from experimental_data.model_evaluation import model_identity
from simulator.workflow import read_json
from simulator.research_execution import ResearchExecutionModel

torch.set_num_threads(4)
job=ROOT/'runs/adaptation/M1-20260915-probe-refined'
started=time.perf_counter()
model,p,parent=data.load(job);c=p['full_update']
history=read_json(job/'cable_residual/history.json')
saved=torch.load(job/'cable_residual/state.pt',map_location='cpu',weights_only=True)
assert saved['update']==history[-1]['update']
receipt=dict(reason='User requested stopping because remaining improvement is small; keep the best saved checkpoint.',
             last_saved_update=saved['update'],best_saved_update=saved['best_update'],
             state_sha256=sha256_file(job/'cable_residual/state.pt'),
             original_status=read_json(job/'status.json'),finalizer_sha256=sha256_file(Path(__file__)),
             optional_preliminary_retention='Deferred at user request to prioritize next collection; not claimed as checked.')
atomic_json(job/'fit/user_stop.json',receipt)
full.progress(job,'Finalizing best checkpoint after user-requested stop')
trials=data.drone_trials(job,model)
cd=data.cable_data(data.cable_rows(job,model,parent,trials),c['replay_weight'])
path=job/'stages/cable_physics/model.json';physical=read_json(path)
engine=ResearchExecutionModel.from_mapping(physical,root=path.parent,device='cuda')
net=make_residual(engine,c)
cable=[physical['cable'][k] for k in ('EI_n_m2','Cb_n_m2_s','external_drag_s_inv')]
# Capture the regularizer against the inherited zero extension, then restore
# trained weights; otherwise the fitting objective would silently change.
objective,accelerator=residual_objective(engine,cd,cable,c)
rr=dict(updates=saved['update'],selected_update=saved['best_update'],
        best_loss=saved['plateau']['best'],baseline_loss=history[0]['baseline_loss'],
        elapsed_s=history[-1]['elapsed_s'],stop_reason='user_requested_early_stop',ceiling=None)
rr=verify_selected_residual(net,objective,job/'cable_residual',rr)
net.requires_grad_(False)
candidate=full.save_model(model,engine,job/'candidate',job,cable_parameters=cable,cable_net=net)
candidate['provenance']['fit_complete']=True
candidate['provenance']['termination']='User-requested early stop; best saved training checkpoint'
atomic_json(job/'candidate/model.json',candidate)
assert full.immutable_identity(job/'candidate/model.json')==full.immutable_identity(job/'source_candidate/model.json')
full.validate_candidate(job/'candidate/model.json',c)
dr=read_json(job/'drone_residual/result.json');cr=read_json(job/'cable_physics/result.json')
selection=dict(schema=data.FULL_SCHEMA,stages_complete=True,candidate_sha256=sha256_file(job/'candidate/model.json'),
    training_takes=[n for n,r in p['takes'].items() if r['role']=='adaptation'],validation_used_for_selection=False,
    drone_parameters=asdict(engine.drone.parameters),cable_parameters=cable,drone_residual=dr,cable_physics=cr,cable_residual=rr,
    termination_override=receipt)
atomic_json(job/'fit/selection_frozen.json',selection)
del objective,accelerator,net,engine,cd,parent,trials,saved;gc.collect();torch.cuda.empty_cache()
output=ROOT/'runs/evaluation'/job.name
full.progress(job,'Saving M1 prediction comparison')
report=full.evaluate_pair(job,output,{c['candidate_id']:job/'candidate/model.json'})
before=read_json(job/'before_update/report.json')
assert before['comparison_key']==report['comparison_key']
report['models'][c['parent_id']]=before['models'][c['parent_id']]
report['baseline_reuse']='Exact frozen M0 evaluation on identical prepared data, saved before fitting.'
atomic_json(output/'report.json',report)
hashes=read_json(output/'evidence_hashes.json');hashes[str(output/'report.json')]=sha256_file(output/'report.json')
hashes.update(read_json(job/'before_update/evidence_hashes.json'))
atomic_json(output/'evidence_hashes.json',hashes)
_,candidate_hashes=model_identity(job/'candidate/model.json')
candidate_hashes.update(hashes)
for f in (job/'fit/user_stop.json',job/'fit/selection_frozen.json',job/'cable_residual/numerical_selection.json',Path(__file__)):
    candidate_hashes[str(f)]=sha256_file(f)
result=dict(status='completed',schema=data.FULL_SCHEMA,training_takes=selection['training_takes'],
    candidate_hashes=candidate_hashes,stages=selection,comparison=str(output),
    elapsed_s=time.perf_counter()-started,model_selected=False,termination='user_requested_early_stop',
    elapsed_scope='Finalization only; individual training stage times preserved in stages',
    preliminary_retention='Not run in this finalization; deferred to prioritize requested next collection')
atomic_json(job/'fit/result.json',result)
registration=full.register(job,output);atomic_json(job/'registration.json',registration)
atomic_json(job/'status.json',dict(status='completed',stage='M1 saved after user-requested early stop',model_selected=False,registration=registration))
print('M1 fitting completed:',job/'candidate/model.json',flush=True)
