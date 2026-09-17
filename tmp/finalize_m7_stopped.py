"""Finalize the saved M7 checkpoint after an explicit user stop; no training."""
from resume_m7_horizontal import *

started=time.perf_counter();torch.set_num_threads(4)
model,p,engine=data.load(JOB);c=p['full_update']
verify_hashes(c['m7_continuation']['hashes'])
saved=torch.load(JOB/'cable_residual/state.pt',map_location='cpu',weights_only=True)
stop_record=dict(reason='user_requested_stop',request='just stop and give me the csv',
    saved_update=saved['update'],best_update=saved['best_update'],
    convergence_reached=False,state_sha256=sha256_file(JOB/'cable_residual/state.pt'),
    finalization_runner=str(Path(__file__)),runner_sha256=sha256_file(Path(__file__)))
atomic_json(JOB/'user_stop.json',stop_record)
try:
    progress(JOB,'Finalizing saved checkpoint; training stopped',update=saved['update'])
    trials=data.drone_trials(JOB,model);rows=data.cable_rows(JOB,model,engine,trials)
    cd=data.cable_data(rows,c['replay_weight'])
    del engine;gc.collect();torch.cuda.empty_cache()
    physical_path=JOB/'stages/cable_physics/model.json';physical=read_json(physical_path)
    engine=ResearchExecutionModel.from_mapping(physical,root=physical_path.parent,device='cuda')
    cable=np.array([physical['cable'][k] for k in ('EI_n_m2','Cb_n_m2_s','external_drag_s_inv')])
    net=make_residual(engine,c)
    objective,accelerator=residual_objective(engine,cd,cable,c)
    history=read_json(JOB/'cable_residual/history.json')
    rr=dict(updates=saved['update'],selected_update=saved['best_update'],
        best_loss=saved['plateau']['best'],baseline_loss=history[0]['baseline_loss'],
        stop_reason='user_requested_stop',convergence_reached=False,
        elapsed_s=history[-1]['elapsed_s'],ceiling=None)
    progress(JOB,'Verifying saved cable checkpoint',update=saved['best_update'])
    rr=verify_selected_residual(net,objective,JOB/'cable_residual',rr);net.requires_grad_(False)
    candidate=full.save_model(model,engine,JOB/'candidate',JOB,cable_parameters=cable,cable_net=net)
    candidate['provenance'].update(fit_complete=True,training_stop_reason='user_requested_stop',convergence_reached=False)
    atomic_json(JOB/'candidate/model.json',candidate)
    assert full.immutable_identity(JOB/'candidate/model.json')==full.immutable_identity(JOB/'source_candidate/model.json')
    full.validate_candidate(JOB/'candidate/model.json',c)
    dr=read_json(JOB/'drone_residual/result.json');cr=read_json(JOB/'cable_physics/result.json')
    selection=dict(schema=data.FULL_SCHEMA,stages_complete=True,candidate_sha256=sha256_file(JOB/'candidate/model.json'),
        training_takes=[n for n,r in p['takes'].items() if r['role']=='adaptation'],validation_used_for_selection=False,
        drone_parameters=asdict(engine.drone.parameters),cable_parameters=cable.tolist(),
        drone_residual=dr,cable_physics=cr,cable_residual=rr,user_stop=stop_record)
    atomic_json(JOB/'fit/selection_frozen.json',selection)
    del objective,accelerator,net,engine,cd,saved;gc.collect();torch.cuda.empty_cache()
    output=ROOT/'runs/evaluation'/JOB.name;progress(JOB,'combined_validation')
    models={'M6':JOB/'source_candidate/model.json','M7':JOB/'candidate/model.json'}
    full.evaluate_pair(JOB,output,models)
    full.evaluate_preliminary_retention(JOB,models,c);full.evaluate_prior_retention(JOB,models)
    _,hashes=model_identity(JOB/'candidate/model.json')
    hashes.update(read_json(JOB/'before_update/evidence_hashes.json'))
    for path in (JOB/'fit/selection_frozen.json',output/'report.json',JOB/'preliminary_validation/report.json',
                 JOB/'prior_validation/report.json',JOB/'user_stop.json',Path(__file__)):
        hashes[str(path)]=sha256_file(path)
    result=dict(status='completed',schema=data.FULL_SCHEMA,training_takes=selection['training_takes'],
        candidate_hashes=hashes,stages=selection,comparison=str(output),elapsed_s=time.perf_counter()-started,
        model_selected=False,continuation=str(SOURCE),training_stop_reason='user_requested_stop')
    atomic_json(JOB/'fit/result.json',result)
    registration=full.register(JOB,output);atomic_json(JOB/'registration.json',registration)
    atomic_json(JOB/'status.json',dict(status='completed',stage='M7 saved checkpoint verified and registered',
        registration=registration,training_stop_reason='user_requested_stop'))
    print('M7 saved checkpoint finalized',flush=True)
except BaseException as exc:
    atomic_json(JOB/'status.json',dict(status='failed',error=str(exc),model_selected=False));raise
