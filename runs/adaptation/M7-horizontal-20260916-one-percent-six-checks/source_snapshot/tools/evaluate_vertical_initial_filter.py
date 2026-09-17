"""Outcome-independent initial-state filter, followed by paired M0--M4 scoring."""
from copy import deepcopy
from pathlib import Path
import argparse
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experimental_data.current_adaptation import (causal_history_indices, observed_cable_history,
    observed_endpoint_velocity, endpoint_velocity)
from experimental_data.io import atomic_json, sha256_file
from experimental_data.whip_adaptation import verify_hashes, rms_summary
from experimental_data.model_evaluation import load_evaluation, metric_arrays, paired_summary, digest
from simulator.cable import CableConfiguration
from simulator.workflow import read_json

LIMITS = dict(cable_shape_rms_m=.03, relative_tip_speed_m_s=.10,
              vehicle_position_error_m=.05, vehicle_speed_m_s=.05)
MODELS = [f'M{i}' for i in range(5)]
METRICS = ('command_driven_tip', 'command_driven_markers', 'drone',
           'conditional_cable_tip', 'conditional_cable_markers')


def gate(values):
    failures = [key for key, limit in LIMITS.items()
                if not np.isfinite(values[key]) or values[key] > limit]
    return not failures, failures


def initial_metrics(data, protocol, cable, nominal, origin):
    """Uses strictly pre-command observations; no outcome or predictor input."""
    ids = causal_history_indices(data['time'], 0., protocol['cable_history_s'])
    nodes, valid = observed_cable_history(data, ids, cable,
        protocol.get('cable_history_missingness', {}).get('minimum_observation_fraction'))
    tau = protocol['cable_velocity_weight_tau_s']
    velocities = observed_endpoint_velocity(data['time'][ids], nodes, valid, tau)
    q = nodes[-1]
    markers = list(cable.marker_node_indices[1:])
    shape = (q-q[0])[markers]-(nominal-nominal[0])[markers]
    pi = causal_history_indices(data['time'], 0., protocol['drone_history_s'])
    if not data['pose_valid'][pi].all():
        raise ValueError('Invalid pre-command vehicle observations')
    vq = endpoint_velocity(data['time'][pi], data['position'][pi], tau)
    assert ids[-1] == pi[-1] and data['time'][ids[-1]] < 0.
    return dict(cable_shape_rms_m=float(np.sqrt(np.mean(np.sum(shape**2, axis=-1)))),
        relative_tip_speed_m_s=float(np.linalg.norm(velocities[-1]-velocities[0])),
        vehicle_position_error_m=float(np.linalg.norm(data['position'][ids[-1]]-origin)),
        vehicle_speed_m_s=float(np.linalg.norm(vq)),
        last_observation_time_s=float(data['time'][ids[-1]]),
        tip_shape_error_m=float(np.linalg.norm(shape[-1])),
        cable_history_s=protocol['cable_history_s'], vehicle_history_s=protocol['drone_history_s'],
        velocity_weight_tau_s=tau)


def aggregate(values):
    a = np.asarray(values, dtype=float)
    assert np.isfinite(a).all()
    return dict(n=len(a), mean_cm=float(100*a.mean()) if len(a) else None,
                sd_cm=float(100*a.std(ddof=1)) if len(a)>1 else None)


def render(out, result):
    def pair(before, after):
        return f'{before["mean_cm"]:.2f} / {after["mean_cm"]:.2f}' if after['n'] else f'{before["mean_cm"]:.2f} / n/a'
    lines = ['# Vertical M0-M4: initial-state-conditioned comparison', '',
        'Exploratory post-hoc analysis. Thresholds were agreed before inspecting filtered outcome errors. '
        'No horizontal data, refitting, command changes, or manuscript edits.', '',
        '## Fixed filter', '',
        '- Cable shape RMS <= 3 cm across the ten tracked markers, relative to the measured attachment and the nominal hanging shape.',
        '- Tip speed relative to the attachment <= 10 cm/s.',
        '- Tracked vehicle position deviation from nominal hover <= 5 cm.',
        '- Vehicle speed <= 5 cm/s. All four conditions must pass.',
        '- Positions use the last observation strictly before command onset. Velocities use causal quadratic endpoint regression: existing 1 s cable / 0.4 s vehicle histories, 0.02 s weighting time constant.',
        '- Shape is not projected by any candidate model. Filter membership is identical for every predictor. No outcome, target error, or hit success enters the filter.', '',
        '## Retention', '', '| Command collection | Kept / original | Recording groups retained |', '|---|---:|---:|']
    for c, row in result['retention'].items():
        lines.append(f'| {c} | {row["kept"]} / {row["total"]} | {row["recording_groups_kept"]} / {row["recording_groups_total"]} |')
    lines += ['', '## Primary: common final M4-command collection', '',
        'Each row predicts the exact same flights. This collection is outside the registered training ancestry of M0-M4. '
        'It remains retrospective and conditions on near-nominal starts; it is not a new independent test.', '',
        '| Predictor | Tip RMSE: all / filtered (cm) | All-marker RMSE: all / filtered (cm) |', '|---|---:|---:|']
    for m, row in result['prediction']['M4'].items():
        lines.append(f'| {m} | {pair(row["all"]["command_driven_tip"],row["filtered"]["command_driven_tip"])} | '
                     f'{pair(row["all"]["command_driven_markers"],row["filtered"]["command_driven_markers"])} |')
    lines += ['', '## Secondary: all five command collections', '',
        'This aggregate mixes training-exposed and unexposed recordings; it is a descriptive comparison, not a held-out ranking.', '',
        '| Predictor | Tip RMSE: all / filtered (cm) | All-marker RMSE: all / filtered (cm) |', '|---|---:|---:|']
    for m, row in result['prediction']['all'].items():
        lines.append(f'| {m} | {pair(row["all"]["command_driven_tip"],row["filtered"]["command_driven_tip"])} | '
                     f'{pair(row["all"]["command_driven_markers"],row["filtered"]["command_driven_markers"])} |')
    lines += ['', '## Physical outcomes by executed command', '',
        'These compare measured motion with the unchanged M0 reference, not prediction with measurement.', '',
        '| Command collection | Reference RMSE: all / filtered (cm) | Fixed-time target distance: all / filtered (cm) | Minimum distance: all / filtered (cm) |', '|---|---:|---:|---:|']
    for c, row in result['physical'].items():
        lines.append('| '+c+' | '+' | '.join(pair(row['all'][k],row['filtered'][k]) for k in
            ('reference_rmse_m','planned_target_distance_m','minimum_target_distance_m'))+' |')
    lines += ['', 'All trajectory scores use [0, 1.1333333333333333) s, matching the rewritten experiment tables. '
        'The fixed strike time remains 1.1138000791100293 s. Values are equal-flight means of per-flight 3D RMSE or event distance. '
        'Sample SDs, per-recording means, and paired model differences are stored in summary.json.', '',
        'No time shifting or spatial registration is applied to outcome scoring. Only the filter shape calculation subtracts attachment position '
        'to separate cable shape from vehicle translation; vehicle position is gated independently.', '',
        'The existing predictors already use measured initialization. Changes after filtering indicate performance on a conditional subset, '
        'not proof that nominal initialization or downwash caused the remaining error. Small samples and session/command composition limit interpretation.', '',
        'The separate 17/20 M4 contact test is untouched and is not recomputed from these target-free flights.', '',
        'Artifacts: filter.json freezes inclusion/exclusion and reasons before scoring; summary.json contains full aggregates; '
        'report.json is a UI-compatible filtered M0-M4 comparison with 58-flight unfiltered baselines retained in summary.json.', '',
        'UI: Evaluation -> Refresh -> this report -> All takes - development. That UI aggregate covers all retained command collections; '
        'use the common-M4 table above for the primary comparison.']
    (out/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main(out):
    out = out.resolve(); out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'status.json', dict(status='running', stage='Pre-command state selection only'))
    try:
        refdir = ROOT/'runs/reference_tracking/M0-2cm-brake-1p3s-fixed-reference'
        meta = read_json(refdir/'reference.json')
        assert sha256_file(refdir/'reference.npz') == meta['reference_sha256']
        with np.load(refdir/'reference.npz') as z:
            nominal, rt, rq = z['cable_position_m'][0].copy(), z['time_s'].copy(), z['tip_position_m'].copy()
        protected = {str(p):sha256_file(p) for p in (Path(__file__),refdir/'reference.json',refdir/'reference.npz')}
        rows = {}
        for c in MODELS:
            job = ROOT/f'runs/adaptation/{c}-collection-20260915/whip_inputs'
            p = read_json(job/'protocol.json'); hashes = read_json(job/'prepared_hashes.json')
            verify_hashes(hashes); protected.update(hashes)
            cable = CableConfiguration.from_mapping(read_json(job/'source_candidate/model.json')['cable'])
            for name in p['takes']:
                with np.load(job/'inputs'/name/'data.npz') as z:
                    data = {key:z[key].copy() for key in z.files}
                values = initial_metrics(data,p,cable,nominal,np.asarray(meta['launch_origin_m']))
                keep, reasons = gate(values)
                rows[name] = dict(collection=c,recording_group=p['recording_groups'][name],
                                  values=values,keep=keep,exclusion_reasons=reasons)
        assert len(rows)==58
        atomic_json(out/'filter.json', dict(limits=LIMITS,takes=rows,
            selection='Strictly pre-command measurements; thresholds agreed before filtered errors were calculated.'))
        filter_hash = sha256_file(out/'filter.json')
        print('Selection frozen:', {c:sum(r['keep'] for r in rows.values() if r['collection']==c) for c in MODELS},flush=True)

        # Only after freezing inclusion do we open prediction/outcome artifacts.
        source = ROOT/'runs/evaluation/vertical-only-M0-M5-paired-complete-20260915'
        original = load_evaluation(source)
        protected.update(read_json(source/'evidence_hashes.json'))
        protected[str(source/'evidence_hashes.json')] = sha256_file(source/'evidence_hashes.json')
        names = list(rows); kept = [n for n in names if rows[n]['keep']]
        assert set(names)==set(original['models']['M0']['takes'])
        end = meta['interval_s'][1]
        report = deepcopy(original)
        report.update(job=str(out), models={}, scored_interval_s=[0.,end], initial_state_filter=str(out/'filter.json'),
            evidence='Exploratory initial-state-conditioned retrospective comparison; training exposure remains labeled.')
        report['protocol']['takes'] = {n:{**original['protocol']['takes'][n],'end_s':end} for n in kept}
        for key in ('collections','recording_groups'):
            report[key] = {n:original[key][n] for n in kept}
        report['protocol']['recording_groups'] = report['recording_groups']
        physical, scores = {}, {m:{} for m in MODELS}
        for m in MODELS:
            entry = deepcopy(original['models'][m]); entry['takes'] = {}
            (out/m).mkdir()
            for n in names:
                a, mask, errors = metric_arrays(source/m/(n+'.npz'),entry['marker_ids'],end)
                scored = {key:rms_summary(errors[key][mask]) for key in METRICS}
                scores[m][n] = {key:scored[key]['rmse_m'] for key in METRICS}
                assert all(v is not None and np.isfinite(v) for v in scores[m][n].values())
                if rows[n]['keep']:
                    row = deepcopy(original['models'][m]['takes'][n])
                    row.update(end_s=end, metrics=scored, initial_state_filter_pass=True)
                    entry['takes'][n] = row
                    shutil.copy2(source/m/(n+'.npz'),out/m/(n+'.npz'))
                if m=='M0':
                    t = a['time_s'][mask]
                    assert t.min()>=rt[0]-1e-9 and t.max()<=rt[-1]+1e-9
                    ref = np.stack([np.interp(t,rt,rq[:,j]) for j in range(3)],axis=1)
                    valid = a['mask'][mask,-1]
                    delta = a['measured_sites'][mask,-1]-ref
                    event = original['models'][m]['takes'][n]['event_metrics']
                    physical[n] = dict(reference_rmse_m=float(np.sqrt(np.mean(np.sum(delta[valid]**2,axis=-1)))),
                        planned_target_distance_m=event['measured_planned_time_target_distance_m'],
                        minimum_target_distance_m=event['measured']['distance_m'])
            report['models'][m] = entry
        result = dict(limits=LIMITS,interval_s=[0.,end],total=58,retained=len(kept),
            retention={},prediction={},physical={},per_flight_prediction=scores,per_flight_physical=physical,
            recording_group_means={},primary_common_m4_paired_differences={})
        for c in MODELS:
            ns = [n for n in names if rows[n]['collection']==c]; ks = [n for n in ns if rows[n]['keep']]
            result['retention'][c] = dict(total=len(ns),kept=len(ks),retained_takes=ks,
                recording_groups_total=len({rows[n]['recording_group'] for n in ns}),
                recording_groups_kept=len({rows[n]['recording_group'] for n in ks}))
            result['physical'][c] = {label:{key:aggregate([physical[n][key] for n in subset]) for key in physical[ns[0]]}
                                     for label,subset in [('all',ns),('filtered',ks)]}
        for c in ['all']+MODELS:
            ns = names if c=='all' else [n for n in names if rows[n]['collection']==c]
            ks = [n for n in ns if rows[n]['keep']]
            result['prediction'][c] = {m:{label:{key:aggregate([scores[m][n][key] for n in subset]) for key in METRICS}
                for label,subset in [('all',ns),('filtered',ks)]} for m in MODELS}
        for g in sorted({r['recording_group'] for r in rows.values()}):
            ns = [n for n in names if rows[n]['recording_group']==g]
            result['recording_group_means'][g] = {label:dict(n=len(subset),
                prediction={m:{key:aggregate([scores[m][n][key] for n in subset]) for key in METRICS} for m in MODELS},
                physical={key:aggregate([physical[n][key] for n in subset]) for key in physical[ns[0]]})
                for label,subset in [('all',ns),('filtered',[n for n in ns if rows[n]['keep']])]}
        primary = [n for n in kept if rows[n]['collection']=='M4']
        assert not any(original['models'][m]['takes'][n]['training_exposure'] for m in MODELS for n in primary)
        for before, after in zip(MODELS[:-1],MODELS[1:]):
            result['primary_common_m4_paired_differences'][before+'->'+after] = {
                key:aggregate([scores[after][n][key]-scores[before][n][key] for n in primary]) for key in METRICS}
        for key in METRICS:
            assert set(paired_summary(report,key,'all')[0])==set(kept)
        # Independent check against the already published common-interval scoring.
        paper = read_json(ROOT/'output/paper/experiments_m0_m4_20260915/metrics.json')
        for c in MODELS:
            for key in result['physical'][c]['all']:
                np.testing.assert_allclose(result['physical'][c]['all'][key]['mean_cm'],paper['physical'][c][key]['mean_cm'],atol=1e-12)
        for m in MODELS:
            for key in ('command_driven_tip','command_driven_markers','drone'):
                np.testing.assert_allclose(result['prediction']['M4'][m]['all'][key]['mean_cm'],paper['prediction'][m][key]['mean_cm'],atol=1e-12)
        assert sha256_file(out/'filter.json')==filter_hash
        report['comparison_key'] = digest(dict(inputs=protected,filter_sha256=filter_hash))
        atomic_json(out/'report.json',report); atomic_json(out/'summary.json',result)
        render(out,result)
        verify_hashes(protected)
        evidence = {str(p):sha256_file(p) for p in out.rglob('*') if p.is_file() and p.name!='status.json'}
        evidence.update(protected); atomic_json(out/'evidence_hashes.json',evidence)
        atomic_json(out/'status.json',dict(status='completed',models=MODELS,original_takes=58,retained_takes=len(kept),
            fitting_performed=False,inference_reused=True,manuscript_changed=False))
        load_evaluation(out)
        print('COMPLETED',out,flush=True)
    except BaseException as exc:
        atomic_json(out/'status.json',dict(status='failed',error=str(exc)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
