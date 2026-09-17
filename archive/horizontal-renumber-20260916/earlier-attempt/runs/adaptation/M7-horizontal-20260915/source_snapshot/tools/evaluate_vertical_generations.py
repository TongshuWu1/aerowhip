"""Paired selected-model evaluation on original vertical-whip collections only."""
from pathlib import Path
from copy import deepcopy
import argparse
import gc
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experimental_data import whip_adaptation_fit as replay
from experimental_data.io import atomic_json, sha256_file
from experimental_data.model_evaluation import (load_catalog, model_identity, metric_arrays,
    summarize_diagnostics, paired_summary, load_evaluation, REPORT, digest)
from experimental_data.whip_adaptation import verify_hashes
from experimental_data.whip_full_fit import immutable_identity
from simulator.workflow import read_json
from simulator.research_execution import ResearchExecutionModel


def at_time(t, xyz, valid, when):
    """Interpolate only between adjacent valid samples; never bridge gaps."""
    t = np.asarray(t)
    i = int(np.searchsorted(t, when))
    if i < len(t) and abs(t[i]-when) < 1e-10:
        return xyz[i].copy() if valid[i] else None
    if i == 0 or i == len(t) or not (valid[i-1] and valid[i]):
        return None
    f = (when-t[i-1])/(t[i]-t[i-1])
    return xyz[i-1]*(1-f)+xyz[i]*f


def closest_approach(t, xyz, valid, target):
    """Minimum on observed piecewise-linear segments, with explicit censoring."""
    t, xyz, valid, target = map(np.asarray, (t, xyz, valid, target))
    ids = np.flatnonzero(valid & np.isfinite(xyz).all(-1))
    if not len(ids):
        return None
    candidates = [(float(np.linalg.norm(xyz[i]-target)), float(t[i]), xyz[i], i, 0.) for i in ids]
    for i in range(len(t)-1):
        if not (valid[i] and valid[i+1]):
            continue
        d = xyz[i+1]-xyz[i]
        f = float(np.clip(np.dot(target-xyz[i], d)/max(np.dot(d,d), 1e-30), 0, 1))
        q = xyz[i]+f*d
        candidates.append((float(np.linalg.norm(q-target)), float(t[i]+f*(t[i+1]-t[i])), q, i, f))
    distance, clock, point, index, fraction = min(candidates, key=lambda r:(r[0],r[1]))
    return dict(distance_m=distance, time_s=clock, position_m=point.tolist(),
                at_observed_window_boundary=bool(abs(clock-t[ids[0]])<1e-9 or abs(clock-t[ids[-1]])<1e-9),
                incomplete_observation=bool(not valid.all()))


def event_metrics(arrays, marker_ids, strike, target, end):
    t = arrays['time_s']
    keep = (t>=0)&(t<end-1e-9)
    t = t[keep]
    truth = arrays['measured_sites'][keep,-1]
    prediction = arrays['coupled_cable'][keep,marker_ids[-1]]
    valid = arrays['mask'][keep,-1] & np.isfinite(truth).all(-1)
    measured = closest_approach(t, truth, valid, target)
    predicted = closest_approach(t, prediction, valid, target)
    q = at_time(t, truth, valid, strike)
    p = at_time(t, prediction, valid, strike)
    result = dict(event='Closest approach to a virtual fixed target in the reviewed command interval',
        interval_s=[0.,end], planned_time_s=strike, measured=measured, predicted=predicted,
        planned_time_prediction_error_m=float(np.linalg.norm(p-q)) if q is not None and p is not None else None,
        measured_planned_time_target_distance_m=float(np.linalg.norm(q-target)) if q is not None else None,
        predicted_planned_time_target_distance_m=float(np.linalg.norm(p-target)) if p is not None else None)
    if measured is not None and predicted is not None:
        result.update(closest_approach_time_error_s=predicted['time_s']-measured['time_s'],
            closest_approach_position_error_m=float(np.linalg.norm(np.array(predicted['position_m'])-measured['position_m'])),
            closest_approach_distance_error_m=predicted['distance_m']-measured['distance_m'],
            timing_censored=measured['at_observed_window_boundary'] or predicted['at_observed_window_boundary'])
    return result


def mean_metric(report, model, names, metric):
    return float(np.mean([report['models'][model]['takes'][n]['metrics'][metric]['rmse_m'] for n in names]))


def render(out, report, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models = list(report['models'])
    collections = list(summary['collections'])
    lines = [f'# Original vertical whip: paired {models[0]}–{models[-1]} evaluation', '',
        '58 repetitions, 15 recording groups, five executed command versions (M0–M4). '
        'No horizontal/curved-side recordings or M6 model are included.', '',
        '## Command-driven tip prediction RMSE (cm)', '',
        '| Predictor | '+' | '.join(collections)+' |', '|---|'+'---:|'*len(collections)]
    for m in models:
        cells=[]
        for c in collections:
            v=summary['collections'][c]['models'][m]
            cells.append(f'{100*v["command_driven_tip"]:.2f}'+(' †' if v['training_exposed_takes'] else ''))
        lines.append('| '+m+' | '+' | '.join(cells)+' |')
    lines += ['', '† The recordings occur in the model’s registered training ancestry: these cells are training/retention diagnostics. '
        'Other cells have no registered training-source overlap; this retrospective audit does not make them a pristine final test. '
        'Compare models within columns. Commands differ between columns; diagonal trends cannot isolate model accuracy.', '',
        '## Evaluation contract', '',
        '- Each model predicts the same recorded commands using the same measured physical initial state and causal pre-command initialization procedure. Model-specific hover compensation uses that same pre-command history.',
        '- Primary scores preserve the historical pre-strike window [0, 1.1138000791100293) s; no model-specific temporal shifting or spatial registration.',
        '- Tip and all-ten-marker 3D RMSE are separate. All-marker scoring weights each valid marker-time observation equally, with no extra tip weight.',
        '- Conditional cable scores use measured attachment motion; these are component diagnostics, not full command-to-motion prediction.',
        '- Missing observations use identical masks for every model. Per-whip values and per-recording-group means are retained; no frame-level confidence intervals or significance claims.',
        '- Event diagnostics use the shared reviewed command interval [0, 1.1333333333333333) s. Closest approach is found on adjacent valid piecewise-linear segments only; endpoint minima are flagged as censored.',
        '- Fixed planned-time prediction error is separate from closest-approach timing/position error. Event matching does not shift trajectory RMSE.',
        '- The target was virtual, not physically present. Distances are not contact or hit-rate evidence. Velocity-based event scores are omitted rather than introducing an unreviewed derivative filter.',
        '- All models and raw/prepared inputs are frozen and hash-verified. Current inference code is recorded. This is new retrospective inference, not a recovered original preflight forecast.', '',
        '## Paired comparisons on the next collection (no registered training overlap for either model)', '',
        '| Earlier → updated model | Collection | Tip RMSE before → after (cm) | Paired mean change (cm) |',
        '|---|---|---:|---:|']
    for row in summary['forward_pairs']:
        lines.append(f'| {row["earlier"]} → {row["updated"]} | {row["collection"]} | '
            f'{100*row["before_m"]:.2f} → {100*row["after_m"]:.2f} | {100*row["paired_change_m"]:+.2f} |')
    lines += ['', '**M5 has no unseen original-vertical-whip collection here.** It was trained on the latest M4 flights and inherits earlier training ancestry. '
        'A new original-whip collection is needed to independently assess M4→M5. Existing data cannot manufacture this missing comparison.', '',
        '## Measured task outcomes (virtual target; not physical hit rates)', '',
        '| Command collection | Planned-time target distance (cm) | Minimum target distance (cm) | Endpoint minima |',
        '|---|---:|---:|---:|']
    for c,v in summary['measured_task_outcomes'].items():
        lines.append(f'| {c} | {100*v["planned_time_mean_m"]:.2f} | {100*v["minimum_distance_mean_m"]:.2f} | {v["endpoint_minima"]}/{v["takes"]} |')
    lines += ['', 'These are measured outcomes of different commands, stored independently of which model is being scored. '
        'Collection order and session effects are not randomized retrospectively. They do not isolate the effect of model refinement.', '',
        'Artifacts: `report.json` (per-whip model errors and events), `summary.json` (collection/session means and paired differences), '
        '`measured_task_outcomes.json` (measured virtual-target outcomes, stored once per flight), '
        '`evaluation.png` (tip and all-marker matrices).', '',
        'UI: Evaluation → refresh → this report → All takes · development. Inspect the training-exposure labels in this report.']
    (out/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    fig, axes = plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax, metric, title in zip(axes, ('command_driven_tip','command_driven_markers'), ('Tip prediction RMSE (cm)','All-marker prediction RMSE (cm)')):
        a=np.array([[100*summary['collections'][c]['models'][m][metric] for c in collections] for m in models])
        im=ax.imshow(a,cmap='YlOrRd',vmin=0,aspect='auto')
        ax.set(xticks=range(len(collections)),xticklabels=collections,yticks=range(len(models)),yticklabels=models,title=title,xlabel='Executed command collection',ylabel='Frozen predictor')
        for i,m in enumerate(models):
            for j,c in enumerate(collections):
                mark='*' if summary['collections'][c]['models'][m]['training_exposed_takes'] else ''
                ax.text(j,i,f'{a[i,j]:.2f}{mark}',ha='center',va='center',fontsize=9)
        fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('Original vertical whip only · 58 repetitions\n* training ancestry; compare predictors within each column',fontsize=12)
    fig.savefig(out/'evaluation.png',dpi=170)
    plt.close(fig)


@torch.no_grad()
def main(out, model_ids=None):
    started=time.perf_counter()
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    atomic_json(out/'status.json',dict(status='running',stage='Verifying frozen inputs'))
    try:
        model_ids = model_ids or [f'M{i}' for i in range(6)]
        if len(set(model_ids)) != len(model_ids) or len(model_ids) < 2:
            raise ValueError('Choose at least two distinct model IDs')
        catalog = {m['id']:m for m in load_catalog(ROOT)['models']}
        selected = {name:catalog[name] for name in model_ids}
        baseline_name = next(iter(selected))
        protected={str(Path(__file__)):sha256_file(__file__)}
        for base in ('experimental_data','simulator','planning'):
            protected.update({str(f):sha256_file(f) for f in (ROOT/base).rglob('*.py')})
        identity=None
        for m in selected.values():
            verify_hashes(m['hashes'])
            sig,h=model_identity(m['model']);assert sig==m['signature'];protected.update(h)
            current=immutable_identity(m['model'])
            if identity is not None:assert identity==current
            identity=current
        reference=ROOT/'runs/reference_tracking/M0-2cm-brake-1p3s-fixed-reference/reference.json'
        strike=read_json(reference)['planned_strike_time_s']
        protected[str(reference)]=sha256_file(reference)
        report=dict(schema=REPORT,job=str(out),models={},protocol=dict(takes={},recording_groups={},diagnostics_only=True),
            device='cuda',evidence='Retrospective same-flight inference; per-model training ancestry explicitly labeled.',
            aggregation='Equal-whip means; per-recording-group means also retained; no inferential independence claim.',
            task_family='original_vertical_whip',scored_interval_s=[0.,strike],physical_target_present=False)
        outcomes={};collections={};groups={};raw_by_take={}
        sources={f'M{i}':ROOT/f'runs/adaptation/M{i}-collection-20260915/whip_inputs' for i in range(5)}
        for collection,job in sources.items():
            p=read_json(job/'protocol.json');verify_hashes(read_json(job/'prepared_hashes.json'))
            raw=read_json(job/'source_hashes.json');verify_hashes(raw)
            protected.update(read_json(job/'prepared_hashes.json'));protected.update(raw)
            rehearsal=Path(p['rehearsal']);meta=read_json(rehearsal/'rehearsal.json')
            planned=meta.get('strike_time_s',meta.get('predicted_hit_time_s'))
            assert abs(planned-strike)<1e-10 and p.get('physical_target_present') is False
            target=np.array(meta['target_position_m'])
            protected[str(rehearsal/'rehearsal.json')]=sha256_file(rehearsal/'rehearsal.json')
            scored=deepcopy(p)
            for n,r in scored['takes'].items():
                assert r['end_s']>strike
                r['end_s']=strike
                report['protocol']['takes'][n]=r
                groups[n]=p['recording_groups'][n];collections[n]=collection
                raw_by_take[n]={h for f,h in raw.items() if Path(f).name==n+'.csv'}
                assert raw_by_take[n]
            basepath=job/'source_candidate/model.json';model=read_json(basepath)
            base=ResearchExecutionModel.from_mapping(model,root=basepath.parent,device='cuda')
            rows=replay.records(job,list(p['takes']),model,base,'cuda')
            ids=list(base.cable.marker_node_indices[1:]);assert len(ids)==10
            states={r['name']:tuple(getattr(r['trial'].initial_pose(base.drone.parameters),k).clone()
                for k in ('position','velocity','rotation','omega_tracking')) for r in rows}
            del base;gc.collect();torch.cuda.empty_cache()
            for name,item in selected.items():
                progress=dict(label=f'{name} on original {collection} flights',
                    completed=sum(len(v['takes']) for v in report['models'].values()),total=58*len(selected))
                atomic_json(out/'progress.json',progress);print(progress['label'],flush=True)
                path=Path(item['model']);value=read_json(path)
                engine=ResearchExecutionModel.from_mapping(value,root=path.parent,device='cuda')
                for r in rows:
                    pose=r['trial'].initial_pose(engine.drone.parameters)
                    for key,expected in zip(('position','velocity','rotation','omega_tracking'),states[r['name']]):
                        assert torch.equal(getattr(pose,key),expected), 'Unequal measured physical initial state'
                folder=out/'traces'/collection/name
                replay.evaluate(rows,engine,value['cable']['external_drag_s_inv'],folder)
                metrics=summarize_diagnostics(folder,scored,ids)
                report['models'].setdefault(name,dict(model=dict(id=name,model=str(path),signature=item['signature']),marker_ids=ids,takes={}))
                for take,row in metrics.items():
                    a,score,errors=metric_arrays(folder/(take+'.npz'),ids,strike)
                    if name!=baseline_name:
                        with np.load(out/baseline_name/(take+'.npz')) as old:
                            for key in ('time_s','measured_sites','mask','measured_origin'):
                                np.testing.assert_allclose(a[key],old[key],atol=0,rtol=0,equal_nan=True)
                    seen=bool(raw_by_take[take]&set(item['training_sources']))
                    event=event_metrics(a,ids,strike,target,p['takes'][take]['end_s'])
                    row.update(collection=collection,recording_group=groups[take],training_exposure=seen,
                        data_use='Training ancestry / retention diagnostic' if seen else 'No registered training overlap; retrospective evaluation',
                        event_metrics=event)
                    row['per_marker_rmse_m']={mode:np.sqrt(np.nanmean(errors[mode+'_markers'][score]**2,axis=0)).tolist() for mode in ('command_driven','conditional_cable')}
                    if take not in outcomes:
                        outcomes[take]=dict(collection=collection,recording_group=groups[take],virtual_target_m=target.tolist(),
                            closest_approach=event['measured'],planned_time_target_distance_m=event['measured_planned_time_target_distance_m'],physical_contact='not_tested_target_absent')
                    else:assert outcomes[take]['closest_approach']==event['measured']
                    (out/name).mkdir(exist_ok=True)
                    import shutil
                    shutil.copy2(folder/(take+'.npz'),out/name/(take+'.npz'))
                    report['models'][name]['takes'][take]=row
                atomic_json(out/'partial_report.json',report)
                del engine;gc.collect();torch.cuda.empty_cache()
            del rows,states;gc.collect();torch.cuda.empty_cache()
        assert len(collections)==58 and len(set(groups.values()))==15
        report['protocol']['recording_groups']=groups
        report.update(collections=collections,recording_groups=groups,comparison_key=digest(protected))
        names,_=paired_summary(report,'command_driven_tip','all');assert len(names)==58
        paired_summary(report,'command_driven_markers','all')
        metrics=('drone','command_driven_tip','command_driven_markers','conditional_cable_tip','conditional_cable_markers')
        summary=dict(collections={},sessions={},forward_pairs=[],measured_task_outcomes={},
                     takes=58,recording_groups=15,model_ids=list(selected))
        for c in sources:
            ns=[n for n in names if collections[n]==c]
            summary['collections'][c]=dict(takes=len(ns),models={m:dict(
                **{metric:mean_metric(report,m,ns,metric) for metric in metrics},
                training_exposed_takes=sum(report['models'][m]['takes'][n]['training_exposure'] for n in ns)) for m in selected})
            fixed=[outcomes[n]['planned_time_target_distance_m'] for n in ns if outcomes[n]['planned_time_target_distance_m'] is not None]
            minimum=[outcomes[n]['closest_approach']['distance_m'] for n in ns if outcomes[n]['closest_approach'] is not None]
            summary['measured_task_outcomes'][c]=dict(takes=len(ns),planned_time_valid_count=len(fixed),
                planned_time_mean_m=float(np.mean(fixed)) if fixed else None,
                minimum_distance_mean_m=float(np.mean(minimum)) if minimum else None,
                endpoint_minima=sum(outcomes[n]['closest_approach']['at_observed_window_boundary'] for n in ns if outcomes[n]['closest_approach']))
            for m in selected:
                events=[report['models'][m]['takes'][n]['event_metrics'] for n in ns]
                uncensored=[abs(e['closest_approach_time_error_s']) for e in events if 'closest_approach_time_error_s' in e and not e['timing_censored'] and not e['measured']['incomplete_observation']]
                fixed=[e['planned_time_prediction_error_m'] for e in events if e['planned_time_prediction_error_m'] is not None]
                summary['collections'][c]['models'][m]['event_diagnostics']=dict(
                    mean_planned_time_prediction_error_m=float(np.mean(fixed)) if fixed else None,
                    planned_time_valid_count=len(fixed),
                    uncensored_complete_timing_mean_absolute_error_s=float(np.mean(uncensored)) if uncensored else None,
                    uncensored_complete_timing_count=len(uncensored),
                    timing_censored_count=sum(e.get('timing_censored',False) for e in events))
        for g in sorted(set(groups.values())):
            ns=[n for n in names if groups[n]==g]
            summary['sessions'][g]=dict(takes=ns,models={m:{metric:mean_metric(report,m,ns,metric) for metric in metrics} for m in selected})
        for i in range(1,5):
            before,after,c=f'M{i-1}',f'M{i}',f'M{i}'
            if before not in selected or after not in selected:
                continue
            ns=[n for n in names if collections[n]==c]
            assert not any(report['models'][m]['takes'][n]['training_exposure'] for m in (before,after) for n in ns)
            differences={n:report['models'][after]['takes'][n]['metrics']['command_driven_tip']['rmse_m']-report['models'][before]['takes'][n]['metrics']['command_driven_tip']['rmse_m'] for n in ns}
            summary['forward_pairs'].append(dict(earlier=before,updated=after,collection=c,
                before_m=mean_metric(report,before,ns,'command_driven_tip'),after_m=mean_metric(report,after,ns,'command_driven_tip'),
                paired_change_m=float(np.mean(list(differences.values()))),per_take_change_m=differences,
                per_session_change_m={g:float(np.mean([v for n,v in differences.items() if groups[n]==g])) for g in sorted({groups[n] for n in ns})}))
        atomic_json(out/'report.json',report);atomic_json(out/'summary.json',summary)
        atomic_json(out/'measured_task_outcomes.json',outcomes)
        render(out,report,summary)
        verify_hashes(protected)
        atomic_json(out/'input_and_code_hashes.json',protected)
        hashes={str(f):sha256_file(f) for f in out.rglob('*') if f.is_file() and f.name not in ('status.json','progress.json')}
        hashes.update(protected);atomic_json(out/'evidence_hashes.json',hashes)
        atomic_json(out/'status.json',dict(status='completed',models=list(selected),takes=58,elapsed_s=time.perf_counter()-started))
        atomic_json(out/'progress.json',dict(label='Comparison complete',completed=58*len(selected),total=58*len(selected)))
        load_evaluation(out)
        print('COMPLETED',out,flush=True)
    except BaseException as exc:
        atomic_json(out/'status.json',dict(status='failed',error=str(exc)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--models',nargs='+',choices=[f'M{i}' for i in range(6)],default=[f'M{i}' for i in range(6)])
    args=parser.parse_args();main(args.output,args.models)
