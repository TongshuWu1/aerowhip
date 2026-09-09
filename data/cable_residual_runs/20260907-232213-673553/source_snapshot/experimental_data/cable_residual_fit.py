"""Fit cable damping and a bounded NN to preliminary motion.

No commanded-force inference, drone fitting, whip data, or protected recording
is used. Candidate selection uses training takes; validation is an acceptance
check, not an optimizer target. Active model and policy files are never changed.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from simulator.cable import CableConfiguration, DderModel
from simulator.cable.residual import MotionResidual
from simulator.workflow import read_json, stamp
from .io import atomic_json, canonical_json_hash, sha256_file
from .force_dataset import _normalized_rotations, reconstruct_dder_nodes
from .state_initialization import causal_state
from .cable_fit import PreparedTake, contiguous_window_starts
from .differentiable_fit import rollout, prediction_loss, save_weights, subset
from .constrained_identification import combine, evaluate_together

PRELIMINARY_TAKES = {'fig8_001', 'fig8_002', 'fig8_003', 'fig8vertical_001',
                     'osc_001', 'osc_002', 'osc_003'}
DATA_KEYS = ('time_s', 'uav_position_m', 'uav_orientation_xyzw', 'uav_valid',
             'cable_marker_positions_m', 'cable_marker_valid', 'auto_frame_valid')


def permitted_takes(manifest):
    selected = {}
    for name, row in manifest['takes'].items():
        # Filter before any source path is constructed or opened.
        if name == 'fig8vertical_002' or not row.get('enabled', True):
            continue
        if row['role'] not in ('training', 'validation'):
            continue
        if name not in PRELIMINARY_TAKES:
            raise ValueError(f'{name} is not an authorized preliminary cable recording.')
        selected[name] = deepcopy(row)
    if {r['role'] for r in selected.values()} != {'training', 'validation'}:
        raise ValueError('Separate preliminary training and validation takes are required.')
    return selected


def prepare_residual_job(root, *, updates=None):
    root = Path(root).resolve()
    model = read_json(root/'config/model.json')
    if model.get('motion_residual', {}).get('enabled'):
        raise ValueError('Select a physical-only calibration before fitting a new cable residual.')
    selected = permitted_takes(read_json(root/'data/dataset_manifest.json'))
    sources = {name: root/'data/processed_takes'/name/'take.npz' for name in selected}
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise ValueError('Process preliminary recordings first: '+', '.join(missing))
    settings = read_json(root/'config/cable_residual.json')
    if updates is not None:
        settings['updates'] = int(updates)
    if settings['updates'] < 1:
        raise ValueError('Use at least one optimizer update.')
    job = root/'data/cable_residual_runs'/stamp()
    job.mkdir(parents=True, exist_ok=False)
    atomic_json(job/'model.json', model)
    atomic_json(job/'settings.json', settings)
    atomic_json(job/'dataset_manifest.json', dict(takes=selected))
    hashes = {}
    for name, source in sources.items():
        destination = job/'processed_takes'/name/'take.npz'
        destination.parent.mkdir(parents=True)
        shutil.copy2(source, destination)
        hashes[name] = sha256_file(destination)
    code_hashes = {}
    project = Path(__file__).resolve().parents[1]
    for name in ('experimental_data/cable_residual_fit.py', 'experimental_data/differentiable_fit.py',
                 'experimental_data/constrained_identification.py', 'experimental_data/state_initialization.py',
                 'experimental_data/cable_fit.py', 'experimental_data/force_dataset.py',
                 'simulator/cable/residual.py', 'simulator/cable/dder.py', 'simulator/cable/config.py',
                 'simulator/point_mass.py'):
        destination = job/'source_snapshot'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project/name, destination)
        code_hashes[name] = sha256_file(destination)
    atomic_json(job/'provenance.json', dict(model_sha256=canonical_json_hash(model),
        manifest_sha256=canonical_json_hash(dict(takes=selected)), settings_sha256=canonical_json_hash(settings),
        input_sha256=hashes, source_sha256=code_hashes, protected_test_used=False,
        whip_data_used=False, drone_residual_trained=False,
        physics_frozen=not settings.get('learn_drag', False),
        learned_drag=settings.get('learn_drag', False),
        frozen_parameters='Masses, geometry, EI and internal bending damping',
        initialization='Past 0.2 seconds; quadratic endpoint velocity and length projection',
        boundary='Measured attachment from calibrated body offset and measured orientation'))
    return job, [sys.executable, '-u', '-m', 'experimental_data.cable_residual_fit', '--job', str(job)]


def prepare_takes(job, payload, horizon, maximum, device, *, training_only=False):
    cable = CableConfiguration.from_mapping(payload['cable'])
    model = DderModel(cable.dder_parameters(EI=payload['cable']['EI_n_m2'], Cb=payload['cable']['Cb_n_m2_s']))
    takes = []
    windows = []
    roles = permitted_takes(read_json(job/'dataset_manifest.json'))
    for name, row in roles.items():
        if training_only and row['role'] != 'training':
            continue
        with np.load(job/'processed_takes'/name/'take.npz', allow_pickle=False) as loaded:
            data = {k: loaded[k] for k in DATA_KEYS}
        t = data['time_s']
        dt = float(np.median(np.diff(t)))
        if not np.isfinite(t).all() or dt <= 0 or not np.allclose(np.diff(t), dt, atol=1e-9, rtol=1e-5):
            raise ValueError(f'{name}: uniform finite timestamps required.')
        rotation, rotation_valid = _normalized_rotations(data['uav_orientation_xyzw'])
        root = data['uav_position_m'] + np.einsum('tij,j->ti', rotation,
                    np.asarray(payload['recorded_data']['optitrack_to_attachment_offset_body_m']))
        root_valid = data['uav_valid'] & rotation_valid & data['auto_frame_valid']
        nodes, node_valid = reconstruct_dder_nodes(root, root_valid, data['cable_marker_positions_m'],
            data['cable_marker_valid'] & data['auto_frame_valid'][:, None], cable)
        valid = node_valid.all(axis=1)
        after, history = round(horizon/dt), max(10, round(.2/dt))
        starts = [s+history for s in contiguous_window_starts(valid,
            horizon_steps=after+history, stride_steps=max(1, after))]
        if not starts:
            raise ValueError(f'{name}: no complete valid {horizon:g}-second windows.')
        starts = np.asarray(starts)[np.linspace(0, len(starts)-1, min(maximum, len(starts)), dtype=int)]
        histories = torch.tensor(np.stack([nodes[s-history:s+1] for s in starts]), dtype=torch.float64)
        with torch.no_grad():
            state = causal_state(histories, dt, model)
        truth = np.stack([data['cable_marker_positions_m'][s:s+after+1] for s in starts])
        initialization = float(np.sqrt(np.mean(np.sum((state.positions_m[:, cable.marker_node_indices[1:]].numpy()-truth[:,0])**2, axis=-1))))
        takes.append(PreparedTake(name, row['role'], state.positions_m.to(device), state.velocities_m_s.to(device),
            torch.tensor(np.stack([root[s:s+after+1] for s in starts]), dtype=torch.float64, device=device),
            torch.tensor(truth, dtype=torch.float64, device=device), tuple(int(s) for s in starts), initialization, dt))
        windows.append(dict(take=name, role=row['role'], starts=starts.tolist(), horizon_s=after*dt))
    if len({round(t.dt_s, 9) for t in takes}) != 1:
        raise ValueError('All takes must share a sample interval.')
    return cable, model, takes, windows


def gradient_check(item, model, cable, parameters, network):
    """Check a training-only loss derivative before trusting short-rollout BPTT."""
    variables = list(network.parameters())
    prediction = rollout(item, model, cable, parameters, gradients=True)
    loss = prediction_loss(prediction, item.measured_marker_positions_m)
    gradients = torch.autograd.grad(loss, variables)
    norm = torch.sqrt(sum(g.square().sum() for g in gradients))
    if not torch.isfinite(norm) or norm <= 1e-12:
        return dict(passed=False, reason='Nonfinite or negligible gradient')
    direction = [g/norm for g in gradients]
    original = [v.detach().clone() for v in variables]
    epsilon = 1e-3
    values = []
    try:
        with torch.no_grad():
            for sign in (1, -1):
                for value, base, delta in zip(variables, original, direction):
                    value.copy_(base+sign*epsilon*delta)
                p = rollout(item, model, cable, parameters)
                values.append(float(prediction_loss(p, item.measured_marker_positions_m)))
    finally:
        with torch.no_grad():
            for value, base in zip(variables, original):
                value.copy_(base)
    finite = (values[0]-values[1])/(2*epsilon)
    analytic = float(norm)
    relative = abs(finite-analytic)/max(abs(finite), abs(analytic), 1e-10)
    return dict(passed=bool(np.isfinite(relative) and relative < .05), relative_error=relative,
                analytic=analytic, finite_difference=finite, epsilon=epsilon)


def acceptance(evaluation):
    checks = {}
    for horizon, pair in evaluation.items():
        baseline, candidate = pair['physics']['validation'], pair['residual']['validation']
        checks[horizon] = (all(candidate[k] < baseline[k] for k in ('equal_take_marker_rmse_m','equal_take_tip_rmse_m'))
            and all(candidate['per_take'][name]['marker_rmse_m'] <= 1.05*row['marker_rmse_m']
                    for name, row in baseline['per_take'].items()))
    return bool(checks and all(checks.values())), checks


@torch.no_grad()
def training_objective(takes, model, cable, parameters):
    if not takes or any(t.role != 'training' for t in takes):
        raise ValueError('Checkpoint selection accepts training takes only.')
    item = combine(takes)
    prediction = rollout(item, model, cable, parameters)
    squared = (prediction[:,1:]-item.measured_marker_positions_m[:,1:]).square().sum(-1)
    if not torch.isfinite(squared).all():
        raise ValueError('Nonfinite candidate selection rollout.')
    loss = .002*(torch.sqrt(1+squared/.002**2)-1)
    means = []
    start = 0
    for take in takes:
        means.append(float(loss[start:start+take.window_count].mean()))
        start += take.window_count
    return float(np.mean(means))


def fit(job):
    job = Path(job).resolve()
    payload, settings = read_json(job/'model.json'), read_json(job/'settings.json')
    provenance = read_json(job/'provenance.json')
    if canonical_json_hash(payload) != provenance['model_sha256']:
        raise ValueError('Saved calibration changed since preparation.')
    for filename, key in (('dataset_manifest.json', 'manifest_sha256'), ('settings.json', 'settings_sha256')):
        if key in provenance and canonical_json_hash(read_json(job/filename)) != provenance[key]:
            raise ValueError(f'Saved {filename} changed since preparation.')
    selected = permitted_takes(read_json(job/'dataset_manifest.json'))
    for name in selected:
        if sha256_file(job/'processed_takes'/name/'take.npz') != provenance['input_sha256'][name]:
            raise ValueError(f'Saved input changed: {name}')
    device = settings['device']
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; select an installed CUDA-enabled Python.')
    torch.set_num_threads(1)
    torch.manual_seed(settings['seed'])
    rng = np.random.default_rng(settings['seed'])
    started = time.perf_counter()
    def progress(label, **extra):
        value = dict(label=label, elapsed_s=time.perf_counter()-started, **extra)
        atomic_json(job/'progress.json', value)
        print(json.dumps(value), flush=True)
    progress('Preparing preliminary cable motion; masses, geometry, EI and internal damping frozen')
    cable, model, training, windows = prepare_takes(job, payload, settings['training_horizon_s'],
        settings['training_windows_per_take'], device, training_only=True)
    network = MotionResidual(cable.node_count, hidden=settings['hidden'],
        acceleration_limit=settings['acceleration_limit_m_s2'],
        learn_drag=settings.get('learn_drag', False),
        initial_drag_s_inv=payload['cable']['external_drag_s_inv']).to(device=device, dtype=torch.float64)
    model.motion_residual = network
    parameters = training[0].initial_positions_m.new_tensor([payload['cable'][k] for k in ('EI_n_m2','Cb_n_m2_s','external_drag_s_inv')])
    baseline_parameters = parameters.clone()
    if network.learn_drag:
        parameters[2] = 0.0  # Damping now lives entirely inside the residual.
    probe = combine([subset(t, [0]) for t in training])
    progress('Checking cable-residual gradients on training takes')
    check = gradient_check(probe, model, cable, parameters, network)
    atomic_json(job/'gradient_check.json', check)
    if not check['passed']:
        raise ValueError('Cable-residual gradient check failed; see gradient_check.json. No calibration was applied.')
    evaluations = {}
    evaluation_takes = {}
    for horizon in settings['evaluation_horizons_s']:
        progress(f'Preparing {horizon:g}-second evaluation windows')
        _, _, takes, records = prepare_takes(job, payload, horizon, settings['evaluation_windows_per_take'], device)
        evaluation_takes[str(horizon)] = takes
        windows.extend(records)
    atomic_json(job/'windows.json', windows)
    model.motion_residual = None
    for horizon, takes in evaluation_takes.items():
        progress(f'Evaluating physical baseline at {horizon} seconds')
        metrics, prediction, truth = evaluate_together(takes, model, cable, baseline_parameters)
        evaluations[horizon] = dict(physics=metrics)
        np.savez_compressed(job/f'physics_{horizon}s.npz', prediction=prediction, measured=truth)
    # Selection uses the shortest full training evaluation horizon, never validation.
    select_horizon = str(min(settings['evaluation_horizons_s']))
    selection_takes = [t for t in evaluation_takes[select_horizon] if t.role == 'training']
    best_loss = evaluations[select_horizon]['physics']['training']['objective']
    initial_loss = best_loss
    best = deepcopy(network.state_dict())
    best_update = 0
    model.motion_residual = network
    # Check the candidate's own initial state, which includes trainable damping.
    # Selection still compares training takes only, never validation.
    best_loss = training_objective(selection_takes, model, cable, parameters)
    optimizer = torch.optim.Adam(network.parameters(), lr=settings['learning_rate'])
    history = []
    for update in range(1, settings['updates']+1):
        if (job/'STOP_REQUESTED').exists():
            raise InterruptedError('Cable residual stopped; active calibration unchanged.')
        batch = combine([subset(t, rng.integers(0, t.window_count, size=settings['batch_windows_per_take'])) for t in training])
        optimizer.zero_grad()
        prediction = rollout(batch, model, cable, parameters, gradients=True)
        correction = network(batch.initial_positions_m, batch.initial_velocities_m_s)
        loss = prediction_loss(prediction, batch.measured_marker_positions_m) + settings['penalty']*correction.square().mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss; active calibration unchanged.')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(network.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        row = dict(update=update, loss=float(loss.detach()), gradient_norm=float(norm))
        if network.learn_drag:
            row['drag_s_inv'] = float(network.drag_coefficient().detach())
        if update % settings['evaluate_every'] == 0 or update == settings['updates']:
            progress(f'Selecting update {update} using training takes only')
            current = training_objective(selection_takes, model, cable, parameters)
            row['selection_training_objective'] = current
            if current < best_loss:
                best_loss, best_update = current, update
                best = deepcopy(network.state_dict())
        history.append(row)
        atomic_json(job/'history.json', history)
        progress(f'Training cable residual: {update}/{settings["updates"]}', **row)
    network.load_state_dict(best)
    save_weights(job/'residual_candidate.pt', network)
    for horizon, takes in evaluation_takes.items():
        progress(f'Checking selected residual at {horizon} seconds')
        metrics, prediction, truth = evaluate_together(takes, model, cable, parameters)
        evaluations[horizon]['residual'] = metrics
        np.savez_compressed(job/f'residual_{horizon}s.npz', prediction=prediction, measured=truth)
    accepted, checks = acceptance(evaluations)
    candidate = deepcopy(payload)
    if network.learn_drag:
        candidate['cable']['external_drag_s_inv'] = 0.0
    candidate['motion_residual'] = dict(enabled=True, checkpoint=str(job/'residual_candidate.pt'),
        sha256=sha256_file(job/'residual_candidate.pt'), specification=network.specification())
    atomic_json(job/'candidate_model.json', candidate)
    atomic_json(job/'evaluation.json', evaluations)
    atomic_json(job/'review.json', dict(accepted=accepted and best_update>0, validation_checks=checks,
        selected_update=best_update, initial_training_objective=initial_loss, selected_training_objective=best_loss,
        selection='Training-only 2-second rollout objective, including initialized residual',
        physics_frozen=not network.learn_drag, learned_drag=network.learn_drag,
        initial_drag_s_inv=payload['cable']['external_drag_s_inv'],
        fitted_drag_s_inv=float(network.drag_coefficient().detach()) if network.learn_drag else None,
        frozen_parameters='Masses, geometry, EI and internal bending damping',
        active_model_changed=False, drone_residual_trained=False,
        model_sha256=canonical_json_hash(payload), candidate_sha256=canonical_json_hash(candidate),
        hardware=torch.cuda.get_device_name() if device=='cuda' else 'CPU',
        validation_scope='Preliminary development takes; not independent final or whip evidence'))
    progress('Complete: residual passes validation' if accepted and best_update>0 else 'Complete: keep physical baseline; residual not accepted')


def apply_candidate(root, job):
    root, job = Path(root), Path(job)
    review = read_json(job/'review.json')
    if not review['accepted']:
        raise ValueError('This residual did not pass validation. Keep the physical baseline.')
    payload = read_json(root/'config/model.json')
    if canonical_json_hash(payload) != review['model_sha256']:
        raise ValueError('Active calibration changed. Fit a residual for that calibration first.')
    candidate = read_json(job/'candidate_model.json')
    if canonical_json_hash(candidate) != review['candidate_sha256']:
        raise ValueError('Candidate changed after validation.')
    from simulator.point_mass import ForceControlledPointCable
    ForceControlledPointCable.from_mapping(candidate, root=root)
    folder = root/'data/baselines'/stamp()
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copy2(job/'residual_candidate.pt', folder/'motion_residual.pt')
    candidate['motion_residual']['checkpoint'] = (folder/'motion_residual.pt').relative_to(root).as_posix()
    for name in ('review.json','evaluation.json','provenance.json'):
        shutil.copy2(job/name, folder/name)
    atomic_json(folder/'model.json', candidate)
    atomic_json(folder/'manifest.json', dict(provenance='Validated preliminary cable residual; see review for learned parameters', source=str(job)))
    atomic_json(root/'config/model.json', candidate)
    atomic_json(root/'config/baseline.json', dict(version=folder.name, model_sha256=canonical_json_hash(candidate)))
    return folder.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.job/'status.json', dict(status='RUNNING'))
    try:
        fit(args.job)
    except BaseException as error:
        atomic_json(args.job/'status.json', dict(status='STOPPED' if isinstance(error, InterruptedError) else 'FAILED', error=str(error)))
        raise
    atomic_json(args.job/'status.json', dict(status='COMPLETED'))


if __name__ == '__main__':
    main()
