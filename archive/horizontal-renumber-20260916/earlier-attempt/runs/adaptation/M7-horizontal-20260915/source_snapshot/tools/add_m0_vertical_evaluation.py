"""Add M0 inference to the immutable, completed M1--M5 vertical comparison."""
from copy import deepcopy
from pathlib import Path
import argparse
import gc
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import evaluate_vertical_generations as ev


def event_summary(events):
    fixed = [e['planned_time_prediction_error_m'] for e in events
             if e['planned_time_prediction_error_m'] is not None]
    timing = [abs(e['closest_approach_time_error_s']) for e in events
              if 'closest_approach_time_error_s' in e and not e['timing_censored']
              and not e['measured']['incomplete_observation']]
    return dict(mean_planned_time_prediction_error_m=float(np.mean(fixed)) if fixed else None,
                planned_time_valid_count=len(fixed),
                uncensored_complete_timing_mean_absolute_error_s=float(np.mean(timing)) if timing else None,
                uncensored_complete_timing_count=len(timing),
                timing_censored_count=sum(e.get('timing_censored', False) for e in events))


@torch.no_grad()
def main(source, out):
    started = time.perf_counter()
    source, out = source.resolve(), out.resolve()
    old = ev.load_evaluation(source)  # Includes all original code/input/trace hashes.
    assert list(old['models']) == ['M1', 'M2', 'M3', 'M4', 'M5']
    assert old['task_family'] == 'original_vertical_whip'
    protected = ev.read_json(source/'evidence_hashes.json')
    protected[str(source/'evidence_hashes.json')] = ev.sha256_file(source/'evidence_hashes.json')
    protected[str(Path(__file__).resolve())] = ev.sha256_file(__file__)
    catalog = {m['id']:m for m in ev.load_catalog(ROOT)['models']}
    item = catalog['M0']
    ev.verify_hashes(item['hashes'])
    signature, hashes = ev.model_identity(item['model'])
    assert signature == item['signature']
    protected.update(hashes)
    for name, value in old['models'].items():
        assert catalog[name]['signature'] == value['model']['signature']
        assert ev.immutable_identity(item['model']) == ev.immutable_identity(value['model']['model'])
    out.mkdir(parents=True, exist_ok=False)
    ev.atomic_json(out/'status.json', dict(status='running', stage='Adding M0 to verified M1–M5 results'))
    torch.set_num_threads(4)
    try:
        report = deepcopy(old)
        report.update(job=str(out), reuse_provenance=dict(source=str(source),
            source_report_sha256=ev.sha256_file(source/'report.json'),
            reused_models=list(old['models']), newly_evaluated_models=['M0']))
        report['models'] = {'M0':dict(model=dict(id='M0', model=item['model'], signature=signature), takes={}),
                            **report['models']}
        for name in old['models']:
            shutil.copytree(source/name, out/name)
        summary = deepcopy(ev.read_json(source/'summary.json'))
        summary['model_ids'] = list(report['models'])
        outcomes = ev.read_json(source/'measured_task_outcomes.json')
        strike = report['scored_interval_s'][1]
        metrics = ('drone', 'command_driven_tip', 'command_driven_markers',
                   'conditional_cable_tip', 'conditional_cable_markers')
        for collection in summary['collections']:
            job = ROOT/f'runs/adaptation/{collection}-collection-20260915/whip_inputs'
            protocol = ev.read_json(job/'protocol.json')
            raw = ev.read_json(job/'source_hashes.json')
            ev.verify_hashes(raw)
            ev.verify_hashes(ev.read_json(job/'prepared_hashes.json'))
            metadata = ev.read_json(Path(protocol['rehearsal'])/'rehearsal.json')
            target = np.asarray(metadata['target_position_m'])
            scored = deepcopy(protocol)
            names = list(protocol['takes'])
            for name in names:
                assert old['models']['M1']['takes'][name]['collection'] == collection
                assert old['protocol']['takes'][name]['end_s'] == strike
                scored['takes'][name]['end_s'] = strike
            basepath = job/'source_candidate/model.json'
            basevalue = ev.read_json(basepath)
            base = ev.ResearchExecutionModel.from_mapping(basevalue, root=basepath.parent, device='cuda')
            rows = ev.replay.records(job, names, basevalue, base, 'cuda')
            path = Path(item['model'])
            value = ev.read_json(path)
            engine = ev.ResearchExecutionModel.from_mapping(value, root=path.parent, device='cuda')
            ids = list(engine.cable.marker_node_indices[1:])
            assert ids == old['models']['M1']['marker_ids'] and engine.dt_s == base.dt_s
            report['models']['M0']['marker_ids'] = ids
            for row in rows:
                p = row['trial'].initial_pose(engine.drone.parameters)
                q = row['trial'].initial_pose(base.drone.parameters)
                for key in ('position', 'velocity', 'rotation', 'omega_tracking'):
                    assert torch.equal(getattr(p, key), getattr(q, key))
                with np.load(source/'M1'/(row['name']+'.npz')) as previous:
                    for key, current in [('time_s', row['grid']), ('measured_sites', row['truth']),
                                         ('mask', row['valid']), ('measured_origin', row['origin'])]:
                        np.testing.assert_allclose(current, previous[key], atol=0, rtol=0, equal_nan=True)
            label = f'M0 on original {collection} flights'
            ev.atomic_json(out/'progress.json', dict(label=label,
                completed=len(report['models']['M0']['takes']), total=58, reused_predictions=290))
            print(label, flush=True)
            folder = out/'traces'/collection/'M0'
            ev.replay.evaluate(rows, engine, value['cable']['external_drag_s_inv'], folder)
            scored_rows = ev.summarize_diagnostics(folder, scored, ids)
            for name, row in scored_rows.items():
                arrays, score, errors = ev.metric_arrays(folder/(name+'.npz'), ids, strike)
                raw_hashes = {h for f,h in raw.items() if Path(f).name == name+'.csv'}
                assert raw_hashes
                seen = bool(raw_hashes & set(item['training_sources']))
                event = ev.event_metrics(arrays, ids, strike, target, protocol['takes'][name]['end_s'])
                assert event['measured'] == outcomes[name]['closest_approach']
                assert event['measured_planned_time_target_distance_m'] == outcomes[name]['planned_time_target_distance_m']
                row.update(collection=collection, recording_group=old['recording_groups'][name],
                    training_exposure=seen, data_use='Training ancestry / retention diagnostic' if seen else
                    'No registered training overlap; retrospective evaluation', event_metrics=event)
                row['per_marker_rmse_m'] = {mode:np.sqrt(np.nanmean(errors[mode+'_markers'][score]**2, axis=0)).tolist()
                                           for mode in ('command_driven', 'conditional_cable')}
                (out/'M0').mkdir(exist_ok=True)
                shutil.copy2(folder/(name+'.npz'), out/'M0'/(name+'.npz'))
                report['models']['M0']['takes'][name] = row
            entry = dict(**{metric:ev.mean_metric(report, 'M0', names, metric) for metric in metrics},
                training_exposed_takes=sum(scored_rows[n]['training_exposure'] for n in names),
                event_diagnostics=event_summary([scored_rows[n]['event_metrics'] for n in names]))
            summary['collections'][collection]['models'] = {'M0':entry, **summary['collections'][collection]['models']}
            ev.atomic_json(out/'partial_report.json', report)
            del rows, engine, base
            gc.collect(); torch.cuda.empty_cache()
        for group, entry in summary['sessions'].items():
            entry['models'] = {'M0':{metric:ev.mean_metric(report, 'M0', entry['takes'], metric) for metric in metrics}, **entry['models']}
        names = [n for n,c in old['collections'].items() if c == 'M1']
        assert not any(report['models'][m]['takes'][n]['training_exposure'] for m in ('M0', 'M1') for n in names)
        differences = {n:report['models']['M1']['takes'][n]['metrics']['command_driven_tip']['rmse_m']
                       -report['models']['M0']['takes'][n]['metrics']['command_driven_tip']['rmse_m'] for n in names}
        summary['forward_pairs'].insert(0, dict(earlier='M0', updated='M1', collection='M1',
            before_m=ev.mean_metric(report, 'M0', names, 'command_driven_tip'),
            after_m=ev.mean_metric(report, 'M1', names, 'command_driven_tip'),
            paired_change_m=float(np.mean(list(differences.values()))), per_take_change_m=differences,
            per_session_change_m={g:float(np.mean([v for n,v in differences.items() if old['recording_groups'][n]==g]))
                                  for g in sorted({old['recording_groups'][n] for n in names})}))
        for metric in metrics:
            paired_names, _ = ev.paired_summary(report, metric, 'all')
            assert len(paired_names) == 58
        for name in old['models']:
            assert report['models'][name] == old['models'][name]
            for take in old['models'][name]['takes']:
                assert ev.sha256_file(source/name/(take+'.npz')) == ev.sha256_file(out/name/(take+'.npz'))
        report['comparison_key'] = ev.digest(protected)
        ev.atomic_json(out/'report.json', report)
        ev.atomic_json(out/'summary.json', summary)
        ev.atomic_json(out/'measured_task_outcomes.json', outcomes)
        ev.render(out, report, summary)
        ev.verify_hashes(protected)
        ev.atomic_json(out/'input_and_code_hashes.json', protected)
        hashes = {str(p):ev.sha256_file(p) for p in out.rglob('*') if p.is_file()
                  and p.name not in ('status.json', 'progress.json')}
        hashes.update(protected)
        ev.atomic_json(out/'evidence_hashes.json', hashes)
        ev.atomic_json(out/'status.json', dict(status='completed', models=list(report['models']), takes=58,
            newly_evaluated_predictions=58, reused_predictions=290, elapsed_s=time.perf_counter()-started))
        ev.atomic_json(out/'progress.json', dict(label='Comparison complete', completed=348, total=348))
        ev.load_evaluation(out)
        print('COMPLETED', out, flush=True)
    except BaseException as exc:
        ev.atomic_json(out/'status.json', dict(status='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.output)
