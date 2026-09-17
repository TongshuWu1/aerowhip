"""Finalize a user-stopped cable residual from saved checkpoints, without updates.

Preserves the original frozen protocol and records the stopping deviation
separately. Checkpoint selection uses the existing training-loss/numerical rule;
held-out observations are evaluated only after the candidate is frozen.
"""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import gc
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experimental_data import whip_full_data as data
from experimental_data import whip_full_fit as full
from experimental_data.whip_full_cable import make_residual, residual_objective
from experimental_data.whip_full_continuation import verify_selected_residual
from experimental_data.whip_adaptation import verify_hashes
from experimental_data.io import atomic_json, sha256_file
from experimental_data.model_evaluation import model_identity
from simulator.research_execution import ResearchExecutionModel
from simulator.workflow import read_json


def run(job, reason):
    job = Path(job).resolve()
    if read_json(job/'status.json').get('status') != 'stopped' or not (job/'STOP').is_file():
        raise ValueError('A cooperatively stopped fit with its STOP request is required')
    if any((job/n).exists() for n in ('candidate', 'fit/selection_frozen.json', 'user_stop_finalization.json')):
        raise ValueError('Finalization already attempted or candidate exists; inspect before proceeding')
    model, protocol, parent = data.load(job)
    contract = protocol['full_update']
    folder = job/'cable_residual'
    saved = torch.load(folder/'state.pt', map_location='cpu', weights_only=True)
    history = read_json(folder/'history.json')
    if saved['update'] != history[-1]['update']:
        raise ValueError('History and last durable optimizer state disagree')
    if not np.isclose(saved['plateau']['best'], history[-1]['best_loss'], rtol=1e-10):
        raise ValueError('Saved best-loss records disagree')
    dr = read_json(job/'drone_residual/result.json')
    if dr.get('numerically_verified') is not True:
        raise ValueError('Completed numerically verified quadrotor stage required')
    cr = read_json(job/'cable_physics/result.json')
    physical_path = job/'stages/cable_physics/model.json'
    physical = read_json(physical_path)
    parameters = np.array([physical['cable'][n] for n in ('EI_n_m2','Cb_n_m2_s','external_drag_s_inv')])
    frozen = {str(f): sha256_file(f) for f in [folder/'state.pt', folder/'history.json',
        job/'drone_residual/result.json', job/'cable_physics/result.json', Path(__file__),
        *sorted(folder.glob('best-*.pt'))]}
    _, component_hashes = model_identity(physical_path)
    frozen.update(component_hashes)
    receipt = dict(reason=reason, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        original_status=read_json(job/'status.json'), original_stopping=contract['residual_stopping'],
        actual_stop_reason='user_requested_stop', saved_update=saved['update'],
        optimizer_best_update=saved['best_update'], optimizer_best_loss=saved['plateau']['best'],
        baseline_loss=history[0]['baseline_loss'], training_elapsed_s=history[-1]['elapsed_s'],
        checkpoint_hashes=frozen, training_restarted=False, additional_optimizer_updates=0,
        validation_used_for_stop_or_selection=False)
    atomic_json(job/'user_stop_finalization.json', receipt)
    # Keep the stop request and stopped status as evidence. Archiving the flag
    # permits the existing evaluation progress hook; no training function runs.
    (job/'STOP').rename(job/'STOP.training-user')
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(contract['seed'])
    def status(stage):
        atomic_json(job/'status.json', dict(status='running',stage=stage,
            activity='finalization_only',additional_optimizer_updates=0,model_selected=False))
        print(stage, flush=True)
    try:
        status('Reconstructing the unchanged training objective; no optimization')
        trials = data.drone_trials(job, model)
        rows = data.cable_rows(job, model, parent, trials)
        cd = data.cable_data(rows, contract['replay_weight'])
        del parent
        engine = ResearchExecutionModel.from_mapping(physical, root=physical_path.parent, device='cuda')
        net = make_residual(engine, contract)
        # Construct before loading learned weights: the change penalty must
        # still reference the exact inherited/zero network used during training.
        objective, accelerator = residual_objective(engine, cd, parameters, contract)
        with torch.no_grad():
            initial = float(objective())
        if not np.isclose(initial, history[0]['baseline_loss'], rtol=1e-7, atol=1e-8):
            raise ValueError(f'Reconstructed baseline differs: {initial}')
        net.load_state_dict(saved['best'])
        with torch.no_grad():
            reproduced = float(objective())
        if not np.isclose(reproduced, saved['plateau']['best'], rtol=1e-7, atol=1e-8):
            raise ValueError(f'Saved best loss cannot be reproduced: {reproduced}')
        rr = dict(updates=saved['update'], additional_updates=0,
            selected_update=saved['best_update'], best_loss=saved['plateau']['best'],
            baseline_loss=history[0]['baseline_loss'], stop_reason='user_requested_stop',
            elapsed_s=history[-1]['elapsed_s'], ceiling=None, practical_plateau_reached=False,
            stop_receipt=str(job/'user_stop_finalization.json'))
        status('Numerically checking saved cable checkpoints; no optimization')
        rr = verify_selected_residual(net, objective, folder, rr)
        verify_hashes(frozen)
        net.requires_grad_(False)
        candidate = full.save_model(model, engine, job/'candidate', job,
            cable_parameters=parameters, cable_net=net)
        candidate['provenance'].update(fit_complete=True,
            cable_residual_stop_reason='user_requested_stop',
            stopping_deviation_record=str(job/'user_stop_finalization.json'))
        atomic_json(job/'candidate/model.json', candidate)
        if full.immutable_identity(job/'candidate/model.json') != full.immutable_identity(job/'source_candidate/model.json'):
            raise ValueError('Candidate changes unreviewed geometry or numerical settings')
        full.validate_candidate(job/'candidate/model.json', contract)
        selection = dict(schema=data.FULL_SCHEMA, stages_complete=True,
            candidate_sha256=sha256_file(job/'candidate/model.json'),
            training_takes=[n for n,r in protocol['takes'].items() if r['role']=='adaptation'],
            validation_used_for_selection=False, drone_parameters=asdict(engine.drone.parameters),
            cable_parameters=parameters.tolist(), drone_residual=dr, cable_physics=cr,
            cable_residual=rr, user_stop_receipt_sha256=sha256_file(job/'user_stop_finalization.json'))
        atomic_json(job/'fit/selection_frozen.json', selection)
        del objective, accelerator, net, engine, cd, saved
        gc.collect()
        torch.cuda.empty_cache()
        status('Evaluating frozen M0/M1 predictions')
        output = ROOT/'runs/evaluation'/job.name
        models = {'M0':job/'source_candidate/model.json', contract['candidate_id']:job/'candidate/model.json'}
        full.evaluate_pair(job, output, models)
        status('Checking retained preliminary validation')
        full.evaluate_preliminary_retention(job, models, contract)
        _, hashes = model_identity(job/'candidate/model.json')
        hashes.update(frozen)
        for f in (job/'fit/selection_frozen.json', output/'report.json',
                  job/'preliminary_validation/report.json', job/'user_stop_finalization.json'):
            hashes[str(f)] = sha256_file(f)
        verify_hashes(hashes)
        elapsed = time.perf_counter()-started
        result = dict(status='completed', schema=data.FULL_SCHEMA,
            training_takes=selection['training_takes'], candidate_hashes=hashes, stages=selection,
            comparison=str(output), elapsed_s=history[-1]['elapsed_s']+elapsed,
            finalization_elapsed_s=elapsed, additional_optimizer_updates=0,
            model_selected=False, stopping='user_requested_stop',
            continuation=contract['continuation']['source'])
        atomic_json(job/'fit/result.json', result)
        registration = full.register(job, output)
        atomic_json(job/'registration.json', registration)
        atomic_json(job/'status.json', dict(status='completed',
            stage='Stopped checkpoint finalized and evaluated; no further training',
            model_selected=False,registration=registration,additional_optimizer_updates=0))
        print('Completed', job, flush=True)
        return result
    except BaseException as exc:
        atomic_json(job/'status.json', dict(status='failed',stage='Stopped-fit finalization',
            error=str(exc),training_restarted=False,model_selected=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True, type=Path)
    parser.add_argument('--reason', required=True)
    args = parser.parse_args()
    run(args.job, args.reason)
