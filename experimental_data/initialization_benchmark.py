"""Controlled, physical-only initialization benchmark on frozen recording roles."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from simulator.cable import CableConfiguration, DderModel
from .cable_fit import contiguous_window_starts
from .io import atomic_json, sha256_file
from .state_initialization import (HistoryFitSettings, causal_state, endpoint_velocity,
    history_rollout, physics_assisted_state, project_state)


METHODS = ('offline_centered', 'causal_polynomial', 'causal_der_history')


def select_indices(scores, thresholds, per_bin=4):
    bins = np.digitize(scores, thresholds)
    chosen = []
    for b in range(3):
        eligible = np.flatnonzero(bins == b)
        if len(eligible):
            chosen.extend(eligible[np.linspace(0, len(eligible)-1, min(per_bin, len(eligible))).round().astype(int)].tolist())
    return sorted(chosen)


def run(source, output, *, horizon=2., per_bin=4, updates=15, take_filter=None):
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'status.json', dict(status='RUNNING'))
    settings = HistoryFitSettings(updates=updates)
    model_payload = json.loads((source/'model.json').read_text())
    fit = json.loads((source/'fit_result.json').read_text())['fitted_parameters']
    cable = CableConfiguration.from_mapping(model_payload['cable'])
    model = DderModel(cable.dder_parameters(EI=fit['EI_n_m2'], Cb=fit['Cb_n_m2_s']))
    torch.set_num_threads(1)
    manifest = json.loads((source/'dataset_manifest.json').read_text())
    inputs = {}
    for name in ['model.json', 'dataset_manifest.json', 'fit_result.json']:
        shutil.copy2(source/name, output/name)
        inputs[name] = sha256_file(source/name)
    for name in ['experimental_data/state_initialization.py', 'experimental_data/initialization_benchmark.py',
                 'simulator/cable/dder.py', 'simulator/cable/config.py']:
        original = Path(__file__).resolve().parents[1]/name
        target = output/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original,target)
        inputs[name] = sha256_file(original)
    prepared = []
    for name, entry in manifest['takes'].items():
        if entry['role'] not in ('training','validation') or not entry.get('enabled',True):
            continue
        if take_filter and name not in take_filter:
            continue
        path = source/'force_takes'/name/'take.npz'
        inputs[str(path.relative_to(source))] = sha256_file(path)
        with np.load(path, allow_pickle=False) as loaded:
            a = {k:loaded[k] for k in loaded.files}
        dt = float(np.median(np.diff(a['time_s'])))
        before = settings.history_steps+settings.derivative_samples-1
        after = round(horizon/dt)
        # Future validity only defines an evaluable offline cohort. Neither causal
        # estimator receives future values or future validity flags.
        first = contiguous_window_starts(a['state_valid'],horizon_steps=before+after,stride_steps=round(1/dt))
        starts = np.asarray(first,dtype=int)+before
        if not len(starts):
            print(f'Skipped {name}: no full history and prediction windows',flush=True)
            continue
        history = torch.tensor(np.stack([a['cable_node_position_world_m'][s-before:s+1] for s in starts]),dtype=torch.float64)
        velocities = endpoint_velocity(history,dt,settings.derivative_samples)
        # State-only intensity measure available before prediction: relative tip speed.
        scores = torch.linalg.vector_norm(velocities[:,-1]-velocities[:,0],dim=-1).numpy()
        prepared.append((name,entry['role'],a,dt,starts,history,scores,after))
    training_scores = [item[6] for item in prepared if item[1]=='training']
    if not training_scores:
        raise ValueError('Need training takes to set motion-intensity thresholds.')
    thresholds = np.quantile(np.concatenate(training_scores),[1/3,2/3])
    atomic_json(output/'protocol.json',dict(schema='initialization_benchmark_v1',source_job=str(source),
        physical_parameters=fit, residual_enabled=False,history_settings=asdict(settings),horizon_s=horizon,
        samples_per_take_per_intensity_bin=per_bin,seed='deterministic temporal spread within each bin',
        intensity='pre-handover causal tip-relative speed',intensity_thresholds_m_s=thresholds.tolist(),
        threshold_source='training candidates only',methods=METHODS,inputs_sha256=inputs,
        notes=['Physical coefficients and neural weights are never trained in this benchmark.',
               'Measured future attachment positions are boundary inputs for conditional cable prediction.',
               'Offline centered velocity uses future samples; it is a reference, not a deployable estimator.',
               'Within each run all methods share starts and future targets. Different horizon runs may select different starts.',
               'Window selection balances three intensity bins and preserves the saved whole-take roles.',
               'Windows can overlap; window counts are not independent experimental repetitions.',
               'No initializer tuning is performed using future validation error.']))
    rows, diagnostics, traces = [], {}, {}
    begin = time.perf_counter()
    for name,role,a,dt,starts,histories,scores,after in prepared:
        keep = select_indices(scores,thresholds,per_bin)
        starts = starts[keep];histories=histories[keep];scores=scores[keep]
        root = torch.tensor(np.stack([a['root_position_world_m'][s:s+after+1] for s in starts]),dtype=torch.float64)
        truth = torch.tensor(np.stack([a['cable_node_position_world_m'][s:s+after+1,cable.marker_node_indices[1:]] for s in starts]),dtype=torch.float64)
        print(f'{name}: {len(starts)} windows, estimating states',flush=True)
        initializers = {}
        times = {}
        with torch.no_grad():
            stamp=time.perf_counter()
            initializers['offline_centered'] = project_state(model,histories[:,-1],
                torch.tensor(a['cable_node_velocity_world_m_s'][starts],dtype=torch.float64))
            times['offline_centered']=time.perf_counter()-stamp
            stamp=time.perf_counter()
            initializers['causal_polynomial']=causal_state(histories,dt,model)
            times['causal_polynomial']=time.perf_counter()-stamp
        initializers['causal_der_history'],diagnostics[name]=physics_assisted_state(histories,dt,model,
            cable.marker_node_indices[1:],settings)
        times['causal_der_history']=diagnostics[name]['elapsed_s']
        for method,state in initializers.items():
            with torch.no_grad():
                prediction,_=history_rollout(model,state,root,dt)
                prediction=prediction[:,:,cable.marker_node_indices[1:]]
                sq=(prediction-truth).square().sum(-1)
            traces[f'{name}__{method}']=prediction.numpy()
            for j,s in enumerate(starts):
                row=dict(take=name,role=role,method=method,start_frame=int(s),
                    start_time_s=float(a['time_s'][s]),intensity=['low','medium','high'][np.digitize(scores[j],thresholds)],
                    initial_tip_relative_speed_m_s=float(scores[j]),
                    initialization_marker_rmse_m=float(sq[j,0].mean().sqrt()),
                    initialization_velocity_difference_from_offline_m_s=float(torch.linalg.vector_norm(
                        state.velocities_m_s[j]-initializers['offline_centered'].velocities_m_s[j],dim=-1).mean()),
                    marker_rmse_m=float(sq[j,1:].mean().sqrt()),tip_rmse_m=float(sq[j,1:,-1].mean().sqrt()),
                    initialization_batch_s=times[method],initialization_batch_size=len(starts),
                    initialization_amortized_s=times[method]/len(starts),finite=bool(torch.isfinite(sq[j]).all()))
                for lead in [.1,.25,.5,1.,2.,5.]:
                    frame=round(lead/dt)
                    if frame<=after:
                        row[f'marker_at_{lead:g}s_m']=float(sq[j,frame].mean().sqrt())
                        row[f'tip_at_{lead:g}s_m']=float(sq[j,frame,-1].sqrt())
                rows.append(row)
        traces[f'{name}__measured']=truth.numpy()
        traces[f'{name}__starts']=starts
        atomic_json(output/'estimator_diagnostics.json',diagnostics)
        atomic_json(output/'progress.json',dict(take=name,completed_takes=len(diagnostics),elapsed_s=time.perf_counter()-begin))
        print(f'{name}: DER history fit {times["causal_der_history"]:.1f}s / batch',flush=True)
    with (output/'windows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(output/'predictions.npz',**traces)
    groups=[]
    for role in ('training','validation'):
        for method in METHODS:
            for intensity in ('all','low','medium','high'):
                selected=[r for r in rows if r['role']==role and r['method']==method and (intensity=='all' or r['intensity']==intensity)]
                if not selected:continue
                per_take={}
                metrics=['marker_rmse_m','tip_rmse_m','initialization_marker_rmse_m']+[k for k in rows[0] if k.startswith(('marker_at_','tip_at_'))]
                for take in sorted({r['take'] for r in selected}):
                    sub=[r for r in selected if r['take']==take]
                    per_take[take]={k:float(np.sqrt(np.mean([r[k]**2 for r in sub]))) for k in metrics}
                    per_take[take]['windows']=len(sub)
                group=dict(role=role,method=method,intensity=intensity,windows=len(selected),takes=len(per_take),per_take=per_take,
                           **{k:float(np.mean([p[k] for p in per_take.values()])) for k in metrics})
                groups.append(group)
    report=dict(schema='initialization_benchmark_result_v1',aggregation='RMSE over equal-duration windows within each take, then arithmetic mean across takes',
        elapsed_s=time.perf_counter()-begin,groups=groups,protected_test_used=False,
        nonfinite_windows=sum(not r['finite'] for r in rows),applied_to_runtime=False)
    atomic_json(output/'result.json',report)
    figures(output,report,horizon)
    atomic_json(output/'status.json',dict(status='COMPLETED'))
    print(json.dumps([g for g in groups if g['intensity']=='all'],indent=2),flush=True)
    return report


def figures(output,report,horizon):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Centered reference','Past-only polynomial','Past-only DER fit']
    colors=['#777777','#0072B2','#D55E00']
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for col,role in enumerate(['training','validation']):
        for method,label,color in zip(METHODS,labels,colors):
            group=next(g for g in report['groups'] if g['role']==role and g['method']==method and g['intensity']=='all')
            leads=[v for v in [.1,.25,.5,1.,2.,5.] if v<=horizon]
            for row,part in enumerate(['marker','tip']):
                axes[row,col].plot(leads,[1000*group[f'{part}_at_{v:g}s_m'] for v in leads],'-o',label=label,color=color,ms=4)
                axes[row,col].set_ylabel(f'Mean take {part} RMSE [mm]')
                axes[row,col].set_xlabel('Prediction lead time [s]');axes[row,col].grid(alpha=.2)
        axes[0,col].set_title(f'{role.capitalize()} recordings')
    axes[0,0].legend(fontsize=8)
    for ext in ['png','pdf','svg']:fig.savefig(output/f'prediction_error.{ext}',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--horizon',type=float,default=2.)
    parser.add_argument('--per-bin',type=int,default=4)
    parser.add_argument('--updates',type=int,default=15)
    parser.add_argument('--takes',nargs='+')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=args.output or root/'data/calibration_audits'/datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-initialization')
    try:
        run(args.source.resolve(),output.resolve(),horizon=args.horizon,per_bin=args.per_bin,updates=args.updates,take_filter=args.takes)
    except Exception as error:
        if output.exists():atomic_json(output/'status.json',dict(status='FAILED',error=str(error)))
        raise
