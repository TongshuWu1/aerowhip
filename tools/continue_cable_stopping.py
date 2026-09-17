"""Resume a stopped cable fit with an explicit, cable-only stopping amendment."""
from pathlib import Path
import argparse
from copy import deepcopy
from dataclasses import asdict
import gc
import itertools
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from experimental_data import whip_full_data as data
from experimental_data import whip_full_fit as full
from experimental_data.io import atomic_json, sha256_file
from experimental_data.model_evaluation import model_identity
from experimental_data.plateau import Plateau
from experimental_data.whip_adaptation import verify_hashes
from experimental_data.whip_full_cable import make_residual, residual_objective
from experimental_data.whip_full_continuation import verify_selected_residual
from experimental_data.whip_full_optim import progress
from simulator.research_execution import ResearchExecutionModel
from simulator.workflow import read_json


def amended_plateau(settings, saved, score):
    """Start fresh patience at resumption; retain the best weights/loss so far."""
    old = saved['plateau']
    if not old['relative'] <= settings['relative'] < 1:
        raise ValueError('Expected a looser finite relative threshold below one')
    if not 1 <= settings['patience'] <= old['patience']:
        raise ValueError('Expected positive patience no longer than the original')
    if settings['minimum'] != old['minimum']:
        raise ValueError('Minimum updates must remain unchanged')
    return Plateau(settings['minimum'], settings['patience'], settings['relative'],
                   best=old['best'], anchor=score, stale=0)


def prepare(source, job, relative, patience):
    if read_json(source/'status.json')['status'] != 'stopped':
        raise ValueError('Source must be gracefully stopped before continuation')
    if (source/'candidate/model.json').exists():
        raise ValueError('Cannot amend an already frozen candidate')
    # Verify the original frozen inputs/code before constructing new evidence.
    _, p, engine = data.load(source)
    del engine
    c = deepcopy(p['full_update'])
    settings = dict(c['residual_stopping'], relative=relative, patience=patience)
    state = torch.load(source/'cable_residual/state.pt', map_location='cpu', weights_only=True)
    history = read_json(source/'cable_residual/history.json')
    if state['update'] != history[-1]['update']:
        raise ValueError('Checkpoint and history do not end at the same update')
    amended_plateau(settings, state, history[-1]['selection_loss'])
    dr = read_json(source/'drone_residual/result.json')
    if dr.get('numerically_verified') is not True or dr['stop_reason'] != 'practical_plateau':
        raise ValueError('Require a completed, verified vehicle fit')
    required = ('cable_physical_gradient_check.json', 'cable_residual_gradient_check.json',
                'cable_capture_eager_check.json')
    physical_check = read_json(source/required[0])
    if (physical_check.get('full_window') is not True or
            not 0 <= physical_check.get('relative_error', float('inf')) <= .02):
        raise ValueError('Source physical sensitivity check did not pass its original tolerance')
    for name in required[1:]:
        if read_json(source/name).get('passed') is not True:
            raise ValueError('Source numerical check did not pass: '+name)
    retained = ('drone_nominal', 'drone_residual', 'attitude_refinement', 'adapted_drone',
                'cable_physics', 'stages', 'cable_residual', 'before_update')
    files = [f for d in retained for f in (source/d).rglob('*') if f.is_file()]
    files += [source/n for n in (*required, 'protocol.json', 'prepared_hashes.json',
              'source_hashes.json', 'code_hashes.json', 'training_windows.json',
              'before_update_ancestry.json')]
    c['cable_residual_stopping'] = settings
    c['cable_stopping_amendment'] = dict(
        source=str(source), hashes={str(f): sha256_file(f) for f in files},
        source_update=state['update'], original_settings=c['residual_stopping'],
        amended_settings=settings,
        reason='User requested looser convergence for cable residual only during fitting.',
        patience_policy='Reset patience at the resumed checkpoint; do not apply the new rule retroactively.',
        optimizer_preserved=True, objective_unchanged=True, validation_used_for_selection=False)
    return data.prepare(job, source, p['preliminary_source'], c)


def train(net, objective, settings, source, folder, job):
    folder.mkdir()
    saved = torch.load(source/'state.pt', map_location='cuda', weights_only=True)
    history = read_json(source/'history.json')
    net.load_state_dict(saved['current'])
    optimizer = torch.optim.Adam(net.parameters(), lr=.001, weight_decay=1e-4)
    optimizer.load_state_dict(saved['optimizer'])
    with torch.no_grad():
        score = float(objective())
    if not np.isclose(score, history[-1]['selection_loss'], rtol=1e-7, atol=1e-8):
        raise ValueError('Resumed objective differs from the frozen original objective')
    stop = amended_plateau(settings, saved, score)
    best = deepcopy(saved['best'])
    best_update = saved['best_update']
    start_update = saved['update']
    baseline = history[0]['baseline_loss']
    for f in source.glob('best-*.pt'):
        shutil.copy2(f, folder/f.name)
    atomic_json(folder/'resume.json', dict(source=str(source),
        source_state_sha256=sha256_file(source/'state.pt'), update=start_update,
        verified_resumed_loss=score, optimizer_preserved=True,
        stopping=settings, plateau_before=saved['plateau'], plateau_after=asdict(stop)))
    started = time.perf_counter()
    for update in itertools.count(start_update+1):
        progress(job, 'cable_residual', update=update, best_loss=stop.best,
                 stopping='User-amended cable-only plateau: 1% / 3 checks' if
                 settings['relative'] == .01 and settings['patience'] == 3 else settings)
        optimizer.zero_grad()
        loss = objective()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite cable residual objective')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if update % settings['check_every']:
            continue
        with torch.no_grad():
            score = float(objective())
        improved, done = stop.observe(update, score)
        if improved:
            best = deepcopy(net.state_dict())
            best_update = update
        history.append(dict(update=update, loss=float(loss.detach()), best_loss=stop.best,
            selection_loss=score, baseline_loss=baseline, gradient_norm=float(norm),
            elapsed_s=time.perf_counter()-started, continued_update=update-start_update,
            phase='user_amended_cable_stopping'))
        atomic_json(folder/'history.json', history)
        torch.save(dict(update=update, current=net.state_dict(), best=best,
            optimizer=optimizer.state_dict(), best_update=best_update, plateau=asdict(stop)), folder/'state.tmp')
        (folder/'state.tmp').replace(folder/'state.pt')
        if improved:
            torch.save(dict(update=update, loss=score, state_dict=best), folder/f'best-{update:06d}.pt')
        if done:
            break
    net.load_state_dict(best)
    result = dict(updates=update, additional_updates=update-start_update,
        selected_update=best_update, best_loss=stop.best, baseline_loss=baseline,
        stop_reason='practical_plateau', stopping=settings,
        elapsed_s=time.perf_counter()-started, ceiling=None,
        amendment=str(job/'protocol.json'))
    atomic_json(folder/'result.json', result)
    return verify_selected_residual(net, objective, folder, result)


def run(job):
    model, p, parent = data.load(job)
    c = p['full_update']
    amendment = c['cable_stopping_amendment']
    source = Path(amendment['source'])
    verify_hashes(amendment['hashes'])
    (job/'fit').mkdir()
    started = time.perf_counter()
    try:
        torch.set_num_threads(4)
        torch.manual_seed(c['seed'])
        progress(job, 'Reusing completed vehicle and cable-physics stages')
        trials = data.drone_trials(job, model)
        rows = data.cable_rows(job, model, parent, trials)
        cd = data.cable_data(rows, c['replay_weight'])
        del parent
        for name in ('drone_nominal', 'drone_residual', 'attitude_refinement', 'adapted_drone',
                     'cable_physics', 'before_update'):
            shutil.copytree(source/name, job/name)
        for name in ('before_update_ancestry.json', 'training_windows.json',
                     'cable_physical_gradient_check.json', 'cable_residual_gradient_check.json',
                     'cable_capture_eager_check.json'):
            shutil.copy2(source/name, job/name)
        physical_path = source/'stages/cable_physics/model.json'
        physical = read_json(physical_path)
        engine = ResearchExecutionModel.from_mapping(physical, root=physical_path.parent, device='cuda')
        cable = np.array([physical['cable'][k] for k in ('EI_n_m2', 'Cb_n_m2_s', 'external_drag_s_inv')])
        net = make_residual(engine, c)
        # Capture the inherited M5 residual reference BEFORE loading M6 optimizer weights.
        objective, accelerator = residual_objective(engine, cd, cable, c)
        rr = train(net, objective, c['cable_residual_stopping'], source/'cable_residual',
                   job/'cable_residual', job)
        net.requires_grad_(False)
        candidate = full.save_model(model, engine, job/'candidate', job,
                                    cable_parameters=cable, cable_net=net)
        candidate['provenance']['fit_complete'] = True
        atomic_json(job/'candidate/model.json', candidate)
        if full.immutable_identity(job/'candidate/model.json') != full.immutable_identity(job/'source_candidate/model.json'):
            raise ValueError('Candidate changed an unreviewed field')
        full.validate_candidate(job/'candidate/model.json', c)
        selection = dict(schema=data.FULL_SCHEMA, stages_complete=True,
            candidate_sha256=sha256_file(job/'candidate/model.json'),
            training_takes=[n for n,r in p['takes'].items() if r['role']=='adaptation'],
            validation_used_for_selection=False, drone_parameters=asdict(engine.drone.parameters),
            cable_parameters=cable.tolist(), drone_residual=read_json(job/'drone_residual/result.json'),
            cable_physics=read_json(job/'cable_physics/result.json'), cable_residual=rr,
            cable_stopping_amendment=amendment)
        atomic_json(job/'fit/selection_frozen.json', selection)
        del objective, accelerator, net, engine, cd
        gc.collect()
        torch.cuda.empty_cache()
        output = ROOT/'runs/evaluation'/job.name
        progress(job, 'combined_validation')
        models = {c['parent_id']: job/'source_candidate/model.json', c['candidate_id']: job/'candidate/model.json'}
        full.evaluate_pair(job, output, models)
        full.evaluate_preliminary_retention(job, models, c)
        full.evaluate_prior_retention(job, models)
        _, hashes = model_identity(job/'candidate/model.json')
        hashes.update(amendment['hashes'])
        hashes.update(read_json(job/'before_update/evidence_hashes.json'))
        for f in (job/'fit/selection_frozen.json', output/'report.json',
                  job/'preliminary_validation/report.json', job/'prior_validation/report.json',
                  job/'protocol.json', job/'cable_residual/resume.json'):
            hashes[str(f)] = sha256_file(f)
        result = dict(status='completed', schema=data.FULL_SCHEMA,
            training_takes=selection['training_takes'], candidate_hashes=hashes,
            stages=selection, comparison=str(output), elapsed_s=time.perf_counter()-started,
            model_selected=False, continuation=str(source))
        atomic_json(job/'fit/result.json', result)
        registration = full.register(job, output)
        atomic_json(job/'registration.json', registration)
        atomic_json(job/'status.json', dict(status='completed', stage='All stages and evaluation complete',
            elapsed_s=result['elapsed_s'], model_selected=False, registration=registration))
    except BaseException as exc:
        atomic_json(job/'status.json', dict(status='stopped' if isinstance(exc, InterruptedError) else 'failed',
            error=str(exc), model_selected=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--relative', type=float, default=.01)
    parser.add_argument('--patience', type=int, default=3)
    args = parser.parse_args()
    job = args.job.resolve()
    prepare(args.source.resolve(), job, args.relative, args.patience)
    run(job)
