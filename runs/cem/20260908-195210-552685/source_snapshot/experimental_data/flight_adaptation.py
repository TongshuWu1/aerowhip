"""Measured-boundary identification and independent coupled flight replay.

Candidates never overwrite the active physical baseline. Aircraft response is
diagnosed separately; commanded forces are not treated as measured thrust.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import time
import numpy as np
import torch
from scipy.optimize import least_squares
from simulator.cable import CableConfiguration, DderModel, DderState, START_PINNED_FREE_END, FREE_ENDPOINTS
from simulator.point_mass import ForceControlledPointCable
from simulator.workflow import read_json, atomic_json
from experimental_data.io import sha256_file, canonical_json_hash
from experimental_data.cable_fit import PreparedTake
from experimental_data.differentiable_fit import rollout


def load_trial(directory):
    directory = Path(directory)
    meta = read_json(directory/'trial.json')
    if meta['role'] not in ('adaptation', 'validation'):
        raise ValueError('Protected trials cannot enter adaptation')
    manifest = read_json(directory/'import.json')
    for name,digest in manifest.get('snapshot_sha256',{}).items():
        if sha256_file(directory/name)!=digest:raise ValueError(f'Flight snapshot changed since import: {name}')
    if sha256_file(directory/'prepared.npz') != manifest['prepared_sha256']:
        raise ValueError('Prepared flight data changed since import')
    with np.load(directory/'prepared.npz', allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    tensor = lambda key: torch.tensor(data[key], dtype=torch.float64)
    take = PreparedTake(meta['trial_id'], 'training' if meta['role']=='adaptation' else 'validation',
        tensor('initial_positions_m'), tensor('initial_velocities_m_s'), tensor('root_positions_m'),
        tensor('measured_marker_positions_m'), (0,), read_json(directory/'diagnostics.json')['initialization_marker_rmse_m'],
        float(np.diff(data['time_s']).mean()))
    return meta, data, take


def physics(payload):
    if payload.get('motion_residual', {}).get('enabled'):
        raise ValueError('This physics-only adaptation path requires residual disabled')
    cable = CableConfiguration.from_mapping(payload['cable'])
    rod = DderModel(cable.dder_parameters(EI=payload['cable']['EI_n_m2'], Cb=payload['cable']['Cb_n_m2_s']))
    return cable, rod


def parameters(payload):
    return torch.tensor([payload['cable']['EI_n_m2'], payload['cable']['Cb_n_m2_s'],
                         payload['cable'].get('external_drag_s_inv', 0.)], dtype=torch.float64)


def coupled_rollout(payload, initial, forces, *, gradients=False):
    """Independent force-driven trajectory, with a differentiable dense path."""
    model = ForceControlledPointCable.from_mapping(payload)
    state = initial
    dt = state.positions_m.new_full((len(state.positions_m),), payload['simulation']['dt_s'])
    constants = model.dder.runtime_constants(state.positions_m)
    positions, velocities = [state.positions_m], [state.velocities_m_s]
    for force in forces:
        state = model.dder.step_runtime(state, state.positions_m[:, :0], dt, constants,
            external_force_world_n=model.controller.node_forces(force, validate=False),
            pinned_endpoints=FREE_ENDPOINTS, create_graph=gradients, dense_constraint_solve=True,
            analytic_bending=True, iterative_damping=False)
        positions.append(state.positions_m); velocities.append(state.velocities_m_s)
    return torch.stack(positions, 1), torch.stack(velocities, 1)


def errors(prediction, truth):
    delta = prediction[:, 1:]-truth[:, 1:]
    if not bool(torch.isfinite(delta).all()):
        raise ValueError('Nonfinite replay')
    square = delta.square().sum(-1)
    return dict(marker_rmse_m=float(square.mean().sqrt()), tip_rmse_m=float(square[:, :, -1].mean().sqrt()))


@torch.no_grad()
def replay(directory, payload, output=None):
    meta, data, take = load_trial(directory)
    cable, rod = physics(payload)
    boundary = rollout(take, rod, cable, parameters(payload))
    force = torch.tensor(data['commanded_force_world_n'], dtype=torch.float64)[:, None]
    q, _ = coupled_rollout(payload, DderState(take.initial_positions_m, take.initial_velocities_m_s), force)
    coupled = q[:, :, cable.marker_node_indices[1:]]
    metrics = dict(trial_id=meta['trial_id'], role=meta['role'], boundary=errors(boundary, take.measured_marker_positions_m),
                   coupled=errors(coupled, take.measured_marker_positions_m),
                   root_rmse_m=float((q[:, 1:, 0]-take.root_positions_m[:, 1:]).square().sum(-1).mean().sqrt()),
                   duration_s=float(data['time_s'][-1]), controller_response='instantaneous force assumption; not an identified Lee controller')
    if output:
        output = Path(output); output.mkdir(parents=True, exist_ok=False)
        atomic_json(output/'metrics.json', metrics)
        np.savez_compressed(output/'replay.npz', time_s=data['time_s'], measured=take.measured_marker_positions_m.numpy(),
                            boundary=boundary.numpy(), coupled=coupled.numpy(), root=q[:, :, 0].numpy(),
                            measured_root=take.root_positions_m.numpy())
        plot_replay(output)
    return metrics


def plot_replay(output):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    with np.load(output/'replay.npz') as data:
        figure = Figure(figsize=(9, 3), layout='constrained'); FigureCanvasAgg(figure)
        figure.suptitle(read_json(output/'metrics.json')['trial_id'],fontsize=10)
        axes = figure.subplots(1, 3)
        for i, axis in enumerate(axes):
            for name, label, style in [('measured','Measured','-'),('boundary','Measured attachment replay','--'),('coupled','Force-driven replay',':')]:
                axis.plot(data['time_s'], data[name][0, :, -1, i], label=label, linestyle=style, lw=1.4)
            axis.set(xlabel='Time after launch [s]', ylabel=f'Tip {"XYZ"[i]} [m]')
            axis.spines[['top','right']].set_visible(False)
        axes[0].legend(fontsize=7)
        for ext in ('png','pdf'): figure.savefig(output/f'tip_replay.{ext}', dpi=200)


def fit_candidate(directories, payload, output, *, selected=('external_drag_s_inv',), max_evaluations=12, progress=None):
    """Bounded log-parameter TRF least squares; JVPs use differentiable DDER.

Each complete trial contributes equal total weight. Validation cannot affect
the optimizer. Default drag-only reflects the baseline identifiability audit.
"""
    started = time.perf_counter()
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'baseline_model.json',payload)
    directories = [Path(p) for p in directories]
    loaded = [load_trial(p) for p in directories]
    if len({row[0]['trial_id'] for row in loaded}) != len(loaded):
        raise ValueError('A complete flight trial cannot occur in multiple roles')
    fit = [row[2] for row in loaded if row[0]['role']=='adaptation']
    if not fit: raise ValueError('Choose at least one adaptation trial')
    # A different geometry would invalidate prepared initialization; changing
    # physics parameters across sequential model versions is permitted.
    def fixed(model):
        model=deepcopy(model)
        for key in ('EI_n_m2','Cb_n_m2_s','external_drag_s_inv','parameter_source','previous_parameter_source'):
            model['cable'].pop(key,None)
        return model
    for directory in directories:
        if canonical_json_hash(fixed(read_json(directory/'model.json'))) != canonical_json_hash(fixed(payload)):
            raise ValueError('Flight geometry/mass/timestep differs; reprocess with explicit compatible initialization')
    cable, rod = physics(payload)
    base = parameters(payload)
    names = ('EI_n_m2','Cb_n_m2_s','external_drag_s_inv')
    indices = [names.index(name) for name in selected]
    if not indices or len(indices)!=len(set(indices)): raise ValueError('Choose distinct physical parameters')
    initial = base[indices].clamp_min(1e-8).log()
    cache = {}; history=[]
    def residual(logs):
        values = base.scatter(0, torch.tensor(indices), logs.exp())
        terms=[]
        for take in fit:
            prediction=rollout(take, rod, cable, values, gradients=True)
            error=(prediction[:,1:]-take.measured_marker_positions_m[:,1:]).flatten()/.002
            # Signed pseudo-Huber residual; sum-of-squares is robust position loss.
            robust=error*torch.sqrt(2/(torch.sqrt(1+error.square())+1))
            terms.append(robust/(error.numel()*len(fit))**.5)
        terms.append(.05*(logs-initial))
        return torch.cat(terms)
    def compute(x):
        if cache.get('x') is not None and np.array_equal(cache['x'],x): return
        logs=torch.tensor(x,dtype=torch.float64,requires_grad=True)
        columns=[]; value=None
        for i in range(len(x)):
            direction=torch.zeros_like(logs); direction[i]=1
            value, derivative=torch.autograd.functional.jvp(residual,logs,direction)
            columns.append(derivative.detach().numpy())
        y=value.detach().numpy(); jac=np.stack(columns,axis=1)
        if not np.isfinite(y).all() or not np.isfinite(jac).all(): raise ValueError('Nonfinite fitting loss or derivative')
        cache.update(x=x.copy(),y=y,jac=jac)
        history.append(dict(evaluation=len(history)+1,objective=float(y@y),parameters=np.exp(x).tolist()))
        atomic_json(output/'history.json',history)
        if progress: progress(f'Fitting cable dynamics · evaluation {len(history)} / {max_evaluations}')
    def fun(x): compute(x); return cache['y']
    def jac(x): compute(x); return cache['jac']
    result=least_squares(fun, initial.numpy(), jac=jac, method='trf',
        bounds=(initial.numpy()-np.log(2),initial.numpy()+np.log(2)),max_nfev=max_evaluations,
        ftol=1e-5,xtol=1e-5,gtol=1e-5)
    candidate=deepcopy(payload)
    for name,value in zip(selected,np.exp(result.x)):candidate['cable'][name]=float(value)
    atomic_json(output/'model.json',candidate)
    metrics={}
    for directory in directories:
        meta=read_json(directory/'trial.json')
        if progress:progress(f'Validating free replay · {meta["trial_id"]}')
        metrics[meta['trial_id']]={label:replay(directory,model,output/f'{meta["trial_id"]}-{label}')
            for label,model in [('baseline',payload),('candidate',candidate)]}
    validation=[value for value in metrics.values() if value['baseline']['role']=='validation']
    improves = bool(validation) and all(
        row['candidate']['boundary']['marker_rmse_m'] < row['baseline']['boundary']['marker_rmse_m']
        and row['candidate']['boundary']['tip_rmse_m'] <= row['baseline']['boundary']['tip_rmse_m']
        and row['candidate']['coupled']['marker_rmse_m'] <= row['baseline']['coupled']['marker_rmse_m']
        for row in validation)
    singular=np.linalg.svd(result.jac,compute_uv=False)
    summary=dict(status='CANDIDATE_REVIEW' if improves else 'NOT_VALIDATED',
        selected_parameters=list(selected),parameter_values={name:candidate['cable'][name] for name in selected},
        solver='scipy least_squares TRF; autodiff JVP Jacobian; log bounds 0.5–2x baseline; prior weight 0.05',
        solver_converged=bool(result.success),solver_message=result.message,
        elapsed_s=time.perf_counter()-started,evaluations=result.nfev,
        jacobian_singular_values=singular.tolist(),validation_improved=improves,metrics=metrics,
        active_baseline_changed=False,flight_ready=False,
        limitations=['No preliminary-data forgetting check yet','No identified aircraft response; coupled force model is provisional',
                    'No NN fitted; use only after repeatable held-out residual error',
                    'Jacobian includes regularization; singular values do not establish physical identifiability'],
        input_hashes={str(p):sha256_file(p/'prepared.npz') for p in directories})
    atomic_json(output/'result.json',summary)
    return summary
