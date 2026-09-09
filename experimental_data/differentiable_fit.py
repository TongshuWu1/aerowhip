"""Staged differentiable physical identification and neural motion correction.

Measured attachment trajectories are inputs; interior measurements are targets
only. Rollouts never reset to measured states after initialization. Selection
uses training loss; validation and the protected test cannot train the model.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import csv
import json
import shutil
import time
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from simulator.cable import CableConfiguration, DderModel, DderState, START_PINNED_FREE_END
from simulator.cable.residual import MotionResidual
from .cable_fit import (_prepare_take, _pseudo_huber, fit_pivot_cable,
                        record_validation_window)
from .io import atomic_json, canonical_json_hash, sha256_file


def subset(take, indices):
    return replace(take, initial_positions_m=take.initial_positions_m[indices],
        initial_velocities_m_s=take.initial_velocities_m_s[indices],
        root_positions_m=take.root_positions_m[indices],
        measured_marker_positions_m=take.measured_marker_positions_m[indices],
        starts=tuple(take.starts[int(i)] for i in indices))


def rollout(take, model, cable, parameters, *, gradients=False, block_steps=10):
    """Full backpropagation through time; checkpointing saves memory, not history."""
    q, v = take.initial_positions_m, take.initial_velocities_m_s
    dt = q.new_full((len(q),), take.dt_s)
    constants = replace(model.runtime_constants(q),
        bending_stiffness_n_m2=parameters[0].expand(len(q)),
        bending_damping_n_m2_s=parameters[1].expand(len(q)))
    if len(parameters) == 3:
        constants = replace(constants, external_drag_s_inv=parameters[2].expand(len(q)))
    marker_nodes = torch.tensor(cable.marker_node_indices[1:], device=q.device)
    outputs = [q.index_select(1, marker_nodes)]

    def advance(q, v, boundary_block):
        predictions = []
        for step in range(boundary_block.shape[1]):
            state = model.step_runtime(DderState(q, v), boundary_block[:, step, None],
                dt, constants, iterative_damping=False,
                pinned_endpoints=START_PINNED_FREE_END, create_graph=gradients,
                dense_constraint_solve=True, analytic_bending=True)
            q, v = state.positions_m, state.velocities_m_s
            predictions.append(q.index_select(1, marker_nodes))
        return q, v, torch.stack(predictions, dim=1)

    for start in range(1, take.horizon_steps + 1, block_steps):
        boundary = take.root_positions_m[:, start:start + block_steps]
        if gradients:
            q, v, predicted = checkpoint(advance, q, v, boundary, use_reentrant=False)
        else:
            q, v, predicted = advance(q, v, boundary)
        outputs.extend(predicted.unbind(1))
    return torch.stack(outputs, dim=1)


def prediction_loss(predicted, measured, scale=.002):
    # Initialization projection is fixed and therefore excluded from the loss.
    distance = torch.linalg.vector_norm(predicted[:, 1:] - measured[:, 1:], dim=-1)
    return _pseudo_huber(distance, scale).mean() / scale


@torch.no_grad()
def evaluate(takes, model, cable, parameters, *, batch_size=24):
    rows = {}
    for take in takes:
        loss, square, tip_square, count, tip_count = 0., 0., 0., 0, 0
        lead = {}
        for start in range(0, take.window_count, batch_size):
            item = subset(take, range(start, min(start + batch_size, take.window_count)))
            prediction = rollout(item, model, cable, parameters)
            delta = prediction[:, 1:] - item.measured_marker_positions_m[:, 1:]
            if not bool(torch.isfinite(delta).all()):
                raise ValueError(f'Nonfinite rollout while evaluating {take.take_id}.')
            sq = delta.square().sum(-1)
            distance = sq.sqrt()
            loss += float((_pseudo_huber(distance, .002) / .002).sum())
            square += float(sq.sum())
            tip_square += float(sq[:, :, -1].sum())
            count += sq.numel()
            tip_count += sq[:, :, -1].numel()
            for seconds in (.1, .25, .5, .7, 1., 2., 3., 5.):
                frame = round(seconds / take.dt_s)
                if frame <= take.horizon_steps:
                    row = lead.setdefault(str(seconds), [0., 0., 0])
                    row[0] += float(sq[:, frame - 1].sum())
                    row[1] += float(sq[:, frame - 1, -1].sum())
                    row[2] += len(sq)
        rows[take.take_id] = dict(objective=loss / count,
            marker_rmse_m=(square / count)**.5, tip_rmse_m=(tip_square / tip_count)**.5,
            windows=take.window_count,
            lead_times={t: dict(marker_rmse_m=(s / (n*cable.moving_marker_count))**.5,
                                tip_rmse_m=(tip/n)**.5) for t, (s, tip, n) in lead.items()})
    return dict(aggregation='arithmetic_mean_of_per_take_RMSE',
        objective=float(np.mean([r['objective'] for r in rows.values()])),
        equal_take_marker_rmse_m=float(np.mean([r['marker_rmse_m'] for r in rows.values()])),
        equal_take_tip_rmse_m=float(np.mean([r['tip_rmse_m'] for r in rows.values()])),
        per_take=rows)


def prepare(job, model, cable, *, horizon, device):
    manifest = json.loads((job / 'force_takes/manifest.json').read_text())
    objective = json.loads((job / 'fit_config.json').read_text())['objective']
    result = []
    for name, row in manifest['takes'].items():
        if row['role'] not in ('training', 'validation'):
            continue
        try:
            result.append(_prepare_take(name, row['role'], job / 'force_takes' / row['path'],
                model=model, cable=cable, device=device, dtype=torch.float64,
                horizon_s=horizon, stride_s=horizon, projection_passes=8,
                initialization=objective.get('initialization', 'offline_centered'),
                initialization_samples=int(objective.get('initialization_samples', 11))))
        except ValueError as error:
            if 'no complete valid' not in str(error):
                raise
            print(f'Skipped {name} at {horizon}s: no complete valid windows', flush=True)
    return tuple(t for t in result if t.role == 'training'), tuple(t for t in result if t.role == 'validation')


def save_weights(path, network):
    payload = dict(specification=network.specification(),
                   state_dict={k: v.detach().cpu() for k, v in network.state_dict().items()})
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def train_stage(stage, takes, validation, model, cable, parameters, settings, job, network=None):
    torch.manual_seed(int(settings['seed']))
    rng = np.random.default_rng(int(settings['seed']))
    physical = stage == 'physics'
    log_parameters = torch.nn.Parameter(parameters.log().clone())
    if physical:
        trainables = [log_parameters]
    else:
        trainables = list(network.parameters())
    optimizer = torch.optim.Adam(trainables, lr=float(settings[f'{stage}_learning_rate']))
    lower = parameters.new_tensor(settings['parameter_bounds']).T[0].log()
    upper = parameters.new_tensor(settings['parameter_bounds']).T[1].log()
    initial = evaluate(takes, model, cable, parameters)
    initial_validation = evaluate(validation, model, cable, parameters)
    best_loss = initial['objective']
    best_params = parameters.detach().clone()
    best_weights = deepcopy(network.state_dict()) if network else None
    history = [dict(stage=stage, update=0, batch_loss=initial['objective'],
        training_objective=initial['objective'], training_marker_rmse_m=initial['equal_take_marker_rmse_m'],
        validation_marker_rmse_m=initial_validation['equal_take_marker_rmse_m'],
        gradient_norm=0., EI=float(parameters[0]), Cb=float(parameters[1]), elapsed_s=0.,
        training_horizon_s=settings['horizon_s'])]
    stale, previous_horizon, best_update = 0, None, 0
    updates = int(settings[f'{stage}_updates'])
    queue = {t.take_id: [] for t in takes}
    started = time.perf_counter()
    for update in range(1, updates + 1):
        optimizer.zero_grad()
        batch_loss = 0.
        items = []
        for take in takes:
            indices = []
            for _ in range(int(settings['windows_per_take'])):
                if not queue[take.take_id]:
                    queue[take.take_id] = rng.permutation(take.window_count).tolist()
                indices.append(queue[take.take_id].pop())
            item = subset(take, indices)
            items.append(item)
        # Each take contributes the same number of windows, preserving equal-take
        # weighting while sharing one batched differentiable rollout.
        if max(t.dt_s for t in items) - min(t.dt_s for t in items) > 1e-8:
            raise ValueError('Batched identification requires a shared sample interval.')
        item = replace(items[0],
            initial_positions_m=torch.cat([t.initial_positions_m for t in items]),
            initial_velocities_m_s=torch.cat([t.initial_velocities_m_s for t in items]),
            root_positions_m=torch.cat([t.root_positions_m for t in items]),
            measured_marker_positions_m=torch.cat([t.measured_marker_positions_m for t in items]))
        horizon = next(h for fraction, h in settings['curriculum'] if update / updates <= fraction)
        if horizon != previous_horizon:
            # A new curriculum stage gets its own patience budget. Otherwise
            # short-window stagnation can stop the first full-horizon update.
            stale = 0
            previous_horizon = horizon
        frames = min(item.horizon_steps, round(horizon / item.dt_s))
        item = replace(item, root_positions_m=item.root_positions_m[:, :frames + 1],
                       measured_marker_positions_m=item.measured_marker_positions_m[:, :frames + 1])
        current = log_parameters.exp() if physical else parameters
        prediction = rollout(item, model, cable, current, gradients=True)
        loss = prediction_loss(prediction, item.measured_marker_positions_m)
        if network:
            acceleration = network(item.initial_positions_m, item.initial_velocities_m_s)
            loss = loss + float(settings['residual_penalty']) * acceleration.square().mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError(f'{stage}: nonfinite training loss at update {update}.')
        loss.backward()
        batch_loss = float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(trainables, float(settings['gradient_clip']),
                                             error_if_nonfinite=True)
        optimizer.step()
        if physical:
            with torch.no_grad():
                log_parameters.clamp_(lower, upper)
        current = log_parameters.detach().exp() if physical else parameters
        row = dict(stage=stage, update=update, batch_loss=batch_loss,
            gradient_norm=float(norm), EI=float(current[0]), Cb=float(current[1]),
            training_horizon_s=frames * item.dt_s,
            elapsed_s=time.perf_counter() - started)
        if update % int(settings['evaluate_every']) == 0 or update == updates:
            metrics = evaluate(takes, model, cable, current)
            held_out = evaluate(validation, model, cable, current)
            row.update(training_objective=metrics['objective'],
                training_marker_rmse_m=metrics['equal_take_marker_rmse_m'],
                validation_marker_rmse_m=held_out['equal_take_marker_rmse_m'])
            if metrics['objective'] < best_loss:
                improvement = best_loss - metrics['objective']
                best_loss, best_params = metrics['objective'], current.clone()
                best_update = update
                best_weights = deepcopy(network.state_dict()) if network else None
                stale = 0 if improvement > float(settings['minimum_improvement']) else stale + 1
                if network:
                    save_weights(job / 'residual_candidate.pt', network)
            else:
                stale += 1
        history.append(row)
        atomic_json(job / f'{stage}_history.json', history)
        atomic_json(job / 'progress.json', row)
        print(json.dumps(row), flush=True)
        if horizon >= settings['horizon_s'] and stale >= int(settings['patience_evaluations']):
            break
    if network:
        network.load_state_dict(best_weights)
        save_weights(job / 'residual_candidate.pt', network)
    atomic_json(job / f'{stage}_selection.json', dict(initial_training_objective=initial['objective'],
        selected_training_objective=best_loss, selection='minimum full training objective',
        validation_used_for_selection=False, updates=len(history) - 1, selected_update=best_update,
        stopped_for_patience=stale >= int(settings['patience_evaluations']),
        fitted_parameters=best_params.cpu().tolist()))
    return best_params, history


def fit_physics_and_residual(job):
    job = Path(job).resolve()
    source_files = ('experimental_data/differentiable_fit.py', 'experimental_data/cable_fit.py',
        'experimental_data/state_initialization.py',
        'simulator/cable/dder.py', 'simulator/cable/cuda_fixed_pcg.py',
        'simulator/cable/residual.py', 'simulator/cable/config.py')
    project = Path(__file__).resolve().parents[1]
    source_hashes = {}
    for name in source_files:
        snapshot = job / 'source_snapshot' / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / name, snapshot)
        source_hashes[name] = sha256_file(snapshot)
    atomic_json(job / 'source_manifest.json', source_hashes)
    config = json.loads((job / 'fit_config.json').read_text())
    settings = config['differentiable']
    model_payload = json.loads((job / 'model.json').read_text())
    torch.set_num_threads(int(settings.get('cpu_threads', 1)))
    torch.manual_seed(int(settings['seed']))
    device = torch.device(settings['device'])
    cable = CableConfiguration.from_mapping(model_payload['cable'])
    model = DderModel(cable.dder_parameters(EI=model_payload['cable']['EI_n_m2'],
                                          Cb=model_payload['cable']['Cb_n_m2_s']))
    stable = model.maximum_stable_bending_stiffness(model_payload['simulation']['dt_s'],
                                                    pinned_endpoints=START_PINNED_FREE_END)
    if settings['parameter_bounds'][0][1] >= stable:
        raise ValueError('Physical fit EI bound exceeds the explicit bending stability limit.')
    training, validation = prepare(job, model, cable, horizon=settings['horizon_s'], device=device)
    if not training or not validation:
        raise ValueError('Separate training and validation takes are required.')
    initial = torch.tensor([model_payload['cable']['EI_n_m2'], model_payload['cable']['Cb_n_m2_s']],
                           dtype=torch.float64, device=device)
    # A reproducible broad initialization prevents SGD depending on an arbitrary
    # material guess. This remains a reference against which refinement is tested.
    warm = job / 'grid_reference'
    warm.mkdir(exist_ok=True)
    warm_start = settings.get('warm_start_job')
    if warm_start:
        parent = Path(warm_start).resolve()
        previous_objective = json.loads((parent / 'fit_config.json').read_text())['objective']
        for key, default in [('initialization', 'offline_centered'), ('initialization_samples', 11)]:
            if previous_objective.get(key, default) != config['objective'].get(key, default):
                raise ValueError('Neural refinement requires unchanged state initialization.')
        for name in ('model.json', 'dataset_manifest.json'):
            if canonical_json_hash(json.loads((parent / name).read_text())) != canonical_json_hash(
                    json.loads((job / name).read_text())):
                raise ValueError(f'Neural refinement requires unchanged {name}.')
        shutil.copy2(parent / 'grid_reference/fit_result.json', warm / 'fit_result.json')
    grid_config = deepcopy(config)
    grid_config.pop('differentiable', None)
    grid_config['objective'].update(horizon_s=settings['horizon_s'], stride_s=settings['horizon_s'])
    grid_config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    grid_config['validation_lead_times_s'] = [.1, .25, .5, .7, 1.]
    atomic_json(warm / 'fit_config.json', grid_config)
    if not (warm / 'fit_result.json').exists():
        fit_pivot_cable(model_path=job / 'model.json', fit_config_path=warm / 'fit_config.json',
            data_root=job / 'force_takes', output_root=warm)
    grid_result = json.loads((warm / 'fit_result.json').read_text())
    start = initial.new_tensor([grid_result['fitted_parameters'][k] for k in ('EI_n_m2', 'Cb_n_m2_s')])
    if warm_start:
        fitted = json.loads((parent / 'physical_parameters.json').read_text())
        physical = initial.new_tensor([fitted['EI_n_m2'], fitted['Cb_n_m2_s']])
        physical_history = json.loads((parent / 'physics_history.json').read_text())
        for name in ('physics_history.json', 'physics_selection.json'):
            shutil.copy2(parent / name, job / name)
    else:
        physical, physical_history = train_stage('physics', training, validation, model, cable,
                                                 start, settings, job)
    atomic_json(job / 'physical_parameters.json', dict(EI_n_m2=float(physical[0]), Cb_n_m2_s=float(physical[1])))
    network = MotionResidual(cable.node_count, hidden=settings['hidden'],
        acceleration_limit=settings['acceleration_limit_m_s2']).to(device=device, dtype=torch.float64)
    if warm_start:
        payload = torch.load(parent / 'residual_candidate.pt', map_location=device, weights_only=True)
        if payload['specification'] != network.specification():
            raise ValueError('Neural refinement requires the same network specification.')
        network.load_state_dict(payload['state_dict'])
        atomic_json(job / 'warm_start.json', dict(parent=str(parent),
            checkpoint_sha256=sha256_file(parent / 'residual_candidate.pt'), optimizer='fresh Adam',
            physical_parameters='frozen inherited fit'))
    model.motion_residual = network
    _, residual_history = train_stage('residual', training, validation, model, cable,
                                       physical, settings, job, network)
    network.eval().requires_grad_(False)
    checkpoint_path = job / 'residual_candidate.pt'
    save_weights(checkpoint_path, network)
    report = deepcopy(grid_result)
    report.update(schema='differentiable_physics_residual_fit_v1',
        fitted_parameters=dict(EI_n_m2=float(physical[0]), Cb_n_m2_s=float(physical[1])),
        selection=dict(method='grid initialization then Adam on full recursive prediction loss',
            validation_used_for_selection=False, untouched_test_used=False),
        solver=dict(device=str(device), dtype='float64', damping_backend='direct',
            constraint_solver='dense', substeps=model.parameters.substeps,
            constraint_iterations=model.parameters.constraint_iterations),
        residual=dict(checkpoint='residual_candidate.pt', sha256=sha256_file(checkpoint_path),
            specification=network.specification(), automatically_applied=False,
            interpretation='bounded cable acceleration discrepancy, not material parameters'),
        long_horizon={}, differentiable_settings=settings, provenance={**grid_result['provenance'],
            'source_snapshot_sha256': source_hashes,
            'differentiable_fit_sha256': sha256_file(Path(__file__)),
            'residual_source_sha256': sha256_file(Path(__file__).parents[1] / 'simulator/cable/residual.py'),
            'fit_config_sha256': canonical_json_hash(config)})
    for horizon in settings['evaluation_horizons_s']:
        fits, held_out = prepare(job, model, cable, horizon=horizon, device=device)
        if not held_out:
            continue
        stages = {}
        for name, parameters, residual in [('active', initial, None), ('physics', physical, None),
                                           ('hybrid', physical, network)]:
            model.motion_residual = residual
            stages[name] = evaluate(held_out, model, cable, parameters)
        report['long_horizon'][str(horizon)] = stages
        print(f'Validation at {horizon}s: ' + json.dumps({k:v['equal_take_marker_rmse_m'] for k,v in stages.items()}), flush=True)
    for role, takes in [('training', training), ('validation', validation)]:
        model.motion_residual = None
        before = evaluate(takes, model, cable, initial)
        after = evaluate(takes, model, cable, physical)
        model.motion_residual = network
        hybrid = evaluate(takes, model, cable, physical)
        report[role] = dict(before=before, after=after, hybrid=hybrid)
    # A candidate can be reviewed without silently changing the active simulator.
    report['residual']['validation_improved_at_all_horizons'] = all(
        stages['hybrid']['equal_take_marker_rmse_m'] < stages['physics']['equal_take_marker_rmse_m']
        for stages in report['long_horizon'].values()) and (
            len(report['long_horizon']) == len(settings['evaluation_horizons_s']))
    model.motion_residual = None
    record_validation_window(job / 'validation_window.npz', validation[0], model=model, cable=cable,
                             before=initial, after=physical, use_optimized_cuda=False)
    with np.load(job / 'validation_window.npz') as data:
        arrays = {k: data[k].copy() for k in data.files}
    model.motion_residual = network
    with torch.no_grad():
        arrays['hybrid_markers_m'] = rollout(subset(validation[0], [0]), model, cable, physical)[0].cpu().numpy()
    np.savez_compressed(job / 'validation_window.npz', **arrays)
    atomic_json(job / 'fit_result.json', report)
    export_figures(job, report, physical_history + residual_history)
    return report


def export_figures(job, report, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = job / 'figures'
    output.mkdir(exist_ok=True)
    with (output / 'learning.csv').open('w', newline='') as stream:
        fields = sorted(set().union(*(r.keys() for r in history)))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), layout='constrained')
    for ax, stage in zip(axes, ('physics', 'residual')):
        rows = [r for r in history if r['stage'] == stage and 'validation_marker_rmse_m' in r]
        for name, color in [('training', '#2563b8'), ('validation', '#d97706')]:
            ax.plot([r['update'] for r in rows], [1000*r[f'{name}_marker_rmse_m'] for r in rows],
                    label=name.title(), color=color)
        selection = json.loads((job / f'{stage}_selection.json').read_text())
        selected = next((r for r in rows if abs(r['training_objective'] - selection['selected_training_objective']) < 1e-12), None)
        if selected:
            ax.axvline(selected['update'], color='#64748b', linestyle=':', label='Selected')
        ax.set(title='Physical parameters' if stage == 'physics' else 'Neural residual',
               xlabel='Optimizer update', ylabel='Mean take RMSE [mm]')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'learning.{ext}', dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.2, 3.5), layout='constrained')
    horizons = sorted(report['long_horizon'], key=float)
    rows = []
    for stage, label in [('active', 'Active physics'), ('physics', 'Fitted physics'), ('hybrid', 'Physics + NN')]:
        values = [report['long_horizon'][h][stage]['equal_take_marker_rmse_m']*1000 for h in horizons]
        ax.plot(list(map(float, horizons)), values, 'o-', label=label)
        rows.extend(dict(horizon_s=h, model=stage, marker_rmse_mm=v) for h, v in zip(horizons, values))
    ax.set(xlabel='Prediction window duration [s]', ylabel='Validation marker RMSE [mm]')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'validation_horizons.{ext}', dpi=300)
    plt.close(fig)
    with (output / 'validation_horizons.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['horizon_s', 'model', 'marker_rmse_mm'])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    try:
        atomic_json(args.job / 'status.json', dict(status='RUNNING'))
        fit_physics_and_residual(args.job)
        atomic_json(args.job / 'status.json', dict(status='COMPLETED'))
    except BaseException as error:
        atomic_json(args.job / 'status.json', dict(status='FAILED', error=str(error)))
        raise
