"""Small, frozen cable-only identification diagnostic; never publishes/selects M0."""
from pathlib import Path
import argparse
import copy
import itertools
import json
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experimental_data.current_adaptation import read, save, nodes_from_sites
from experimental_data.current_adaptation_fit import cable_windows, join_windows, cable_objectives
from experimental_data.preliminary_prepare import PreliminaryTrial
from experimental_data.preliminary_fit import CableForward
from experimental_data.state_initialization import project_state
from experimental_data.io import sha256_file
from experimental_data.plateau import Plateau
from simulator.research_execution import ResearchExecutionModel

SOURCE = ROOT / 'runs/adaptation/20260909-preliminary1-M0-v2'
METHODS = ('causal_uniform_1s', 'causal_weighted_1s', 'offline_centered_110ms')


def centered_velocity(times, positions, at):
    """Offline quadratic derivative; caller must label its future-sample access."""
    x = np.asarray(times) - at
    if len(x) != 11 or not np.allclose(np.diff(x), .01, atol=1e-6):
        raise ValueError('Offline reference needs eleven contiguous native samples')
    if not np.isfinite(positions).all():
        raise ValueError('Offline reference contains missing markers')
    weights = np.linalg.pinv(np.c_[np.ones(len(x)), x, x*x])[1]
    return np.einsum('t,t...->...', weights, positions)


def engine(model, substeps=8):
    modified = copy.deepcopy(model)
    modified['motion_residual']['enabled'] = False
    modified['cable']['substeps'] = substeps
    result = ResearchExecutionModel.from_mapping(modified, root=SOURCE/'candidate', device='cuda')
    assert result.physics.motion_residual is None
    return result


def status(out, stage, **kwargs):
    save(out/'status.json', dict(status='running', stage=stage, **kwargs))
    print(json.dumps(dict(stage=stage, **kwargs)), flush=True)
    if (out/'STOP').exists():
        raise InterruptedError('Stopped by request')


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    model = read(SOURCE/'candidate/model.json')
    source_model = read(SOURCE/'source_candidate/model.json')
    protected_paths = list((SOURCE/'inputs').glob('*/data.npz'))
    protected_paths += list((SOURCE/'candidate').glob('*'))
    protected_paths += [SOURCE/'protocol.json', SOURCE/'windows.json']
    protected_paths += list((ROOT/'config').glob('*.json'))
    protected_paths += [Path(p) for p in read(SOURCE/'protected_before.json')]
    protected = {str(p): sha256_file(p) for p in protected_paths if p.is_file()}
    save(out/'protected_before.json', protected)
    for directory in ('experimental_data', 'simulator'):
        for p in (ROOT/directory).rglob('*.py'):
            dest = out/'code_snapshot'/p.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
    shutil.copy2(__file__, out/'diagnostic_source.py')
    save(out/'source_model.json', model)
    protocol = dict(scope='Small physical-only cable diagnostic; no M0 publication or selection',
        source_job=str(SOURCE), model_sha256=sha256_file(SOURCE/'candidate/model.json'),
        hardware=dict(os=platform.platform(), gpu=torch.cuda.get_device_name(), dtype='float64'),
        methods=METHODS, history_s=1., weighted_tau_s=.02, offline_future_s=.05,
        initial_positions='Same raw positions and length projection for all methods; only velocity estimate differs',
        training_window_selection='Three evenly spaced eligible windows per training take, before prediction errors',
        validation_window_selection='Three evenly spaced eligible windows from figure8_002; excluded from search',
        horizons_s=[.25,.5,1.,2.], training_horizon_s=1.,
        fitting_initializer='causal_weighted_1s; tau frozen from prior training-only diagnostic',
        parameter_bounds=[[1e-9,1e-4],[1e-9,1e-3]], grid_shape=[7,7],
        search=dict(population=16, seed=9409, minimum=6, patience=5, relative=.005, ceiling=24),
        evidence='Conditional on measured attachment. Offline reference is retrospective, not deployable.',
        stopping='Small finite grid plus practical plateau; ceiling is not convergence')
    save(out/'protocol.json', protocol)
    e = engine(model)
    eligible, rejected = {}, []
    for w in read(SOURCE/'windows.json'):
        t = PreliminaryTrial(SOURCE, w, source_model)
        t.cable_history_s = 1.
        t.cable_velocity_weight_tau_s = .02
        records, errors = cable_windows([t], e.physics, [0.], 2.)
        if errors:
            rejected.extend(errors)
            continue
        end = t.data['pre_indices'][-1]
        ids = np.arange(end-5,end+6)
        if not t.data['pose_valid'][ids].all() or not t.data['marker_valid'][ids].all():
            rejected.append(dict(name=t.name, reason='Offline-reference neighborhood masked'))
            continue
        eligible.setdefault(t.take, []).append((w, t, records[0]))
    selected, dataset = [], {m: [] for m in METHODS}
    for take, values in sorted(eligible.items()):
        for i in np.unique(np.rint(np.linspace(0, len(values)-1, 3)).astype(int)):
            w, t, record = values[i]
            selected.append(dict(**w, projection_m=record['projection_m'], start_time_s=float(record['time'][0])))
            for method in METHODS:
                r = dict(record)
                if method == METHODS[0]:
                    t.cable_velocity_weight_tau_s = None
                    state, _, _ = t.cable_state(e.physics)
                elif method == METHODS[1]:
                    t.cable_velocity_weight_tau_s = .02
                    state, _, _ = t.cable_state(e.physics)
                else:
                    end = t.data['pre_indices'][-1]
                    ids = np.arange(end-5,end+6)
                    sites = t.data['sites'][ids]
                    nodes = nodes_from_sites(sites, e.cable)
                    v = centered_velocity(t.data['time'][ids], nodes, t.data['time'][end])
                    # Use the same raw q as causal initializers; isolate velocity.
                    q = torch.tensor(nodes[5:6], device='cuda', dtype=torch.float64)
                    state = project_state(e.physics, q, torch.tensor(v[None],device='cuda',dtype=torch.float64))
                r.update(q=state.positions_m, v=state.velocities_m_s)
                dataset[method].append(r)
    if len(selected) != 15:
        raise ValueError('Expected three eligible windows for each of five takes')
    save(out/'windows.json', dict(selected=selected, rejected=rejected,
        eligible_per_take={k:len(v) for k,v in eligible.items()}))
    arrays = {}
    for method, records in dataset.items():
        for key, value in join_windows(records).items():
            arrays[method+'__'+key] = value.cpu().numpy()
    np.savez_compressed(out/'states.npz', **arrays)
    states = {m:join_windows(v) for m,v in dataset.items()}
    assert torch.equal(states[METHODS[0]]['q'], states[METHODS[1]]['q'])
    assert torch.equal(states[METHODS[1]]['q'], states[METHODS[2]]['q'])
    save(out/'preparation_checks.json', dict(identical_initial_positions=True,
        selected_before_prediction=True, parameters_fit_from_training_only=True,
        max_position_projection_m=max(w['projection_m'] for w in selected)))
    return model, selected, states


def load(out):
    z = np.load(out/'states.npz')
    return read(out/'source_model.json'), read(out/'windows.json')['selected'], {
        m:{k:torch.tensor(z[m+'__'+k],device='cuda',dtype=torch.float64)
           for k in ('q','v','roots','truth')} for m in METHODS}


def select_data(data, ids, steps=None):
    return {k:v[ids, :steps+1] if steps is not None and k in ('roots','truth') else v[ids]
            for k,v in data.items()}


def metrics(q, data, selected, marker_indices):
    result = []
    errors = (q[:,:,marker_indices]-data['truth']).square().sum(-1).cpu().numpy()
    for w, error in zip(selected, errors):
        row = dict(name=w['name'], take=w['take'], role=w['role'], horizons={})
        for h in (.25,.5,1.,2.):
            n = min(int(round(h*150)),len(error)-1)
            row['horizons'][str(h)] = dict(actual_last_time_s=n/150,
                all_marker_rmse_m=float(np.sqrt(error[1:n+1].mean())),
                tip_rmse_m=float(np.sqrt(error[1:n+1,-1].mean())))
        result.append(row)
    return result


def checks(out, model, selected, states):
    params = [model['cable']['EI_n_m2'],model['cable']['Cb_n_m2_s']]
    train_ids = [i for i,w in enumerate(selected) if w['role']=='training']
    rows, predictions = {}, {}
    for substeps in (8,16,32):
        e = engine(model, substeps)
        for method in METHODS if substeps==8 else (METHODS[1],):
            status(out,'forward checks',substeps=substeps,method=method)
            data = select_data(states[method],train_ids)
            forward = CableForward(e,data)
            q, _ = forward(params)
            if substeps==8:
                repeat,_ = forward(params)
                if not torch.equal(q,repeat):
                    raise AssertionError('Identical input does not give identical prediction')
            key = method+'_'+str(substeps)
            predictions[key] = q.cpu().numpy()
            rows[key] = metrics(q,data,[selected[i] for i in train_ids],list(e.cable.marker_node_indices[1:]))
            del forward
    comparisons = {}
    for a,b in ((8,16),(16,32)):
        err = predictions[METHODS[1]+'_'+str(a)]-predictions[METHODS[1]+'_'+str(b)]
        comparisons[f'{a}_vs_{b}'] = [dict(name=selected[i]['name'],
            tip_trajectory_difference_rms_m=float(np.sqrt(np.mean(np.sum(x[1:,-1]**2,-1)))))
            for i,x in zip(train_ids,err)]
    np.savez_compressed(out/'forward_check_predictions.npz', **predictions)
    save(out/'forward_checks.json',dict(parameters=params,metrics=rows,substep_comparisons=comparisons,
        exact_repeat=True,scope='Training windows only; no parameter selection'))


def search(out, model, selected, states):
    protocol = read(out/'protocol.json')
    ids = [i for i,w in enumerate(selected) if w['role']=='training']
    assert all(selected[i]['take']!='figure8_002' for i in ids)
    data = select_data(states[METHODS[1]],ids,150)
    e = engine(model)
    marker_indices = list(e.cable.marker_node_indices[1:])
    # Equal windows per take: averaging twelve windows gives equal take weights.
    assert len(ids)==12 and len({selected[i]['take'] for i in ids})==4
    size = 16
    expanded = {k:v.repeat_interleave(size,0) for k,v in data.items()}
    forward = CableForward(e,expanded)
    evaluations = []
    def evaluate(logs, stage):
        losses = []
        for start in range(0,len(logs),size):
            group = logs[start:start+size]
            padded = np.concatenate([group,np.repeat(group[-1:],size-len(group),axis=0)])
            status(out,stage,candidates_completed=len(evaluations))
            begin = time.perf_counter()
            q,_ = forward(np.tile(10**padded,(len(ids),1)).T)
            values = cable_objectives(q,expanded['truth'],marker_indices).reshape(len(ids),size).mean(0).cpu().numpy()[:len(group)]
            if not np.isfinite(values).all():
                raise FloatingPointError('Nonfinite physical objective')
            for x,y in zip(group,values):
                evaluations.append(dict(log10_parameters=x,objective=float(y)))
            losses.extend(values)
            save(out/'evaluations.json',evaluations)
            print(json.dumps(dict(batch_seconds=time.perf_counter()-begin,batch_best=float(min(values)))),flush=True)
        return np.array(losses)
    grid = np.array(list(itertools.product(np.linspace(-9,-4,7),np.linspace(-9,-3,7))))
    old = np.log10([model['cable']['EI_n_m2'],model['cable']['Cb_n_m2_s']])
    grid = np.vstack([grid, old, [-6.,-6.]])
    scores = evaluate(grid,'physical loss surface')
    save(out/'grid.json',dict(log10_parameters=grid,objectives=scores))
    rng = np.random.default_rng(protocol['search']['seed'])
    population = rng.uniform([-9,-9],[-4,-3],(size,2))
    population[:4] = grid[np.argsort(scores)[:4]]
    values = evaluate(population,'initialize differential evolution')
    tracker = Plateau(minimum=6,patience=5,relative=.005)
    tracker.observe(0,float(values.min()))
    history = []
    reason = 'safety_ceiling'
    for generation in range(1,25):
        proposals = []
        for i in range(size):
            a,b,c = rng.choice(np.delete(np.arange(size),i),3,replace=False)
            mutant = np.clip(population[a]+.7*(population[b]-population[c]),[-9,-9],[-4,-3])
            cross = rng.random(2)<.8
            cross[rng.integers(2)] = True
            proposals.append(np.where(cross,mutant,population[i]))
        proposals = np.array(proposals)
        candidate_scores = evaluate(proposals,'differential evolution generation '+str(generation))
        accept = candidate_scores<values
        population[accept],values[accept] = proposals[accept],candidate_scores[accept]
        _,stop = tracker.observe(generation,float(values.min()))
        history.append(dict(generation=generation,best=float(values.min()),stale=tracker.stale))
        save(out/'optimizer_state.json',dict(population=population,scores=values,
            rng_state=rng.bit_generator.state,history=history,plateau=vars(tracker)))
        best = int(values.argmin())
        save(out/'best_physics.json',dict(EI_n_m2=float(10**population[best,0]),
            Cb_n_m2_s=float(10**population[best,1]),objective=float(values[best]),generation=generation,
            selected_model=False,training_windows=[selected[i]['name'] for i in ids]))
        if stop:
            reason = 'practical_plateau'
            break
    save(out/'stopping.json',dict(reason=reason,history=history,global_convergence=False))
    del forward
    best = read(out/'best_physics.json')
    params = [best['EI_n_m2'],best['Cb_n_m2_s']]
    report = {}
    # Only after freezing selection do we evaluate the separate take.
    for label,physical in [('published_M0_physics',10**old),('pilot_physics',params)]:
        for method in METHODS:
            status(out,'frozen evaluation',label=label,method=method)
            forward = CableForward(e,states[method]);q,_ = forward(physical)
            report[label+'__'+method] = metrics(q,states[method],selected,marker_indices)
            np.savez_compressed(out/(label+'__'+method+'.npz'),q=q.cpu().numpy())
            del forward
    # Verify the selected physical point with finer integration without reselecting.
    for n in (16,32):
        status(out,'selected-parameter numerical check',substeps=n)
        forward = CableForward(engine(model,n),states[METHODS[1]])
        q,_ = forward(params)
        report['pilot_physics__substeps_'+str(n)] = metrics(q,states[METHODS[1]],selected,marker_indices)
        np.savez_compressed(out/f'pilot_physics__substeps_{n}.npz',q=q.cpu().numpy())
        del forward
    save(out/'evaluation.json',report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['checks','search'],required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = args.output.resolve()
    try:
        if args.stage=='checks':
            model,selected,states = prepare(out)
            checks(out,model,selected,states)
        else:
            if not (out/'forward_checks.json').exists() or (out/'evaluations.json').exists():
                raise ValueError('Search requires reviewed checks and no previous search')
            model,selected,states = load(out)
            search(out,model,selected,states)
        before = read(out/'protected_before.json')
        changed = [p for p,h in before.items() if sha256_file(p)!=h]
        save(out/'integrity.json',dict(protected_files=len(before),changed=changed))
        if changed:
            raise AssertionError('Protected input changed')
        save(out/'status.json',dict(status='completed',stage=args.stage,selected_model=False))
    except Exception as exc:
        if out.exists():
            save(out/'status.json',dict(status='stopped' if isinstance(exc,InterruptedError) else 'failed',
                stage=args.stage,error=str(exc)))
        raise


if __name__=='__main__':
    main()
