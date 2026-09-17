"""Freeze an outcome-independent five-flight physical subset, then rescore it."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import argparse
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experimental_data.io import atomic_json, sha256_file
from experimental_data.whip_adaptation import verify_hashes
from experimental_data.model_evaluation import metric_arrays
from simulator.workflow import read_json

SOURCE = ROOT/'runs/evaluation/vertical-M0-M4-initial-state-filter-20260915'
SEED = 20260915
MODELS = [f'M{i}' for i in range(5)]
KEYS = ('reference_rmse_m', 'planned_target_distance_m', 'minimum_target_distance_m')


def select_flights(rows, model, seed, count=5):
    """Only names, filter-pass flags, collection and recording labels are used."""
    groups = defaultdict(list)
    for name in sorted(rows):
        row = rows[name]
        if row['collection'] == model and row['keep']:
            groups[row['recording_group']].append(name)
    if sum(map(len, groups.values())) < count or count < len(groups):
        raise ValueError('Insufficient flights or count too small to cover all recordings')
    rng = np.random.Generator(np.random.PCG64(seed))
    quota = {g: 0 for g in sorted(groups)}
    for _ in range(count):
        eligible = [g for g in quota if quota[g] < len(groups[g])]
        level = min(quota[g] for g in eligible)
        ties = [g for g in eligible if quota[g] == level]
        quota[ties[int(rng.integers(len(ties)))]] += 1
    selected = {}
    for g, n in quota.items():
        ids = rng.choice(len(groups[g]), size=n, replace=False)
        selected[g] = sorted(groups[g][int(i)] for i in ids)
    return dict(seed=seed, quotas=quota, selected_by_recording=selected,
                selected_takes=sorted(n for names in selected.values() for n in names))


def aggregate(values):
    a = np.asarray(values, dtype=float)
    if len(a) < 2 or not np.isfinite(a).all():
        raise ValueError('At least two finite flight values required')
    return dict(n=len(a), mean_cm=float(100*a.mean()), sd_cm=float(100*a.std(ddof=1)))


def freeze(out):
    out.mkdir(parents=True, exist_ok=False)
    source = SOURCE/'filter.json'
    evidence = read_json(SOURCE/'evidence_hashes.json')
    assert sha256_file(source) == evidence[str(source)]
    rows = read_json(source)['takes']
    selection = {m: select_flights(rows, m, SEED+i) for i, m in enumerate(MODELS)}
    assert all(sorted(s['quotas'].values()) == [1, 2, 2] for s in selection.values())
    atomic_json(out/'selection.json', dict(
        schema='five_flight_physical_subset_v1', frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        algorithm='Allocate five slots to the least-filled eligible recording, with fixed-seed random tie breaking; sample uniformly without replacement within each recording.',
        rng='NumPy PCG64', numpy_version=np.__version__, base_seed=SEED,
        selection_inputs='Flight name, collection, recording group, and frozen pre-command filter-pass flag only. No outcome values.',
        scope='Physical execution only. The ten-flight common-M4 prediction set and separate 17/20 contact test are unchanged.',
        source_filter_sha256=sha256_file(source), selector_sha256=sha256_file(Path(__file__)),
        collections=selection))
    atomic_json(out/'status.json', dict(status='selection_frozen', scoring_performed=False))
    print('FROZEN', out/'selection.json')
    for m, row in selection.items():
        print(m, row['quotas'], row['selected_takes'])


def score(out):
    # No selection is performed in this stage. A mismatched selection must fail.
    from tools.evaluate_vertical_generations import event_metrics
    if (out/'summary.json').exists():
        raise FileExistsError('This subset has already been scored')
    manifest_path = out/'selection.json'
    manifest = read_json(manifest_path)
    selection_hash = sha256_file(manifest_path)
    assert manifest['selector_sha256'] == sha256_file(Path(__file__))
    assert manifest['source_filter_sha256'] == sha256_file(SOURCE/'filter.json')
    verify_hashes(read_json(SOURCE/'evidence_hashes.json'))
    rows = read_json(SOURCE/'filter.json')['takes']
    for i, m in enumerate(MODELS):
        assert manifest['collections'][m] == select_flights(rows, m, SEED+i)
    original = read_json(SOURCE/'summary.json')
    report = read_json(SOURCE/'report.json')
    refdir = ROOT/'runs/reference_tracking/M0-2cm-brake-1p3s-fixed-reference'
    meta = read_json(refdir/'reference.json')
    assert meta['reference_sha256'] == sha256_file(refdir/'reference.npz')
    with np.load(refdir/'reference.npz') as z:
        rt, rq = z['time_s'].copy(), z['tip_position_m'].copy()
    end, strike = meta['interval_s'][1], meta['planned_strike_time_s']
    marker_ids = report['models']['M0']['marker_ids']
    per_take, physical, groups = {}, {}, {}
    for model, selected in manifest['collections'].items():
        for name in selected['selected_takes']:
            arrays, mask, _ = metric_arrays(SOURCE/'M0'/(name+'.npz'), marker_ids, end)
            t = arrays['time_s'][mask]
            assert t.min() >= rt[0]-1e-9 and t.max() <= rt[-1]+1e-9
            reference = np.stack([np.interp(t, rt, rq[:, j]) for j in range(3)], axis=1)
            valid = arrays['mask'][mask, -1]
            delta = arrays['measured_sites'][mask, -1]-reference
            events = event_metrics(arrays, marker_ids, strike, np.asarray(meta['physical_target_m']), end)
            value = dict(reference_rmse_m=float(np.sqrt(np.mean(np.sum(delta[valid]**2, axis=-1)))),
                planned_target_distance_m=events['measured_planned_time_target_distance_m'],
                minimum_target_distance_m=events['measured']['distance_m'])
            for key in KEYS:
                np.testing.assert_allclose(value[key], original['per_flight_physical'][name][key], atol=1e-12, rtol=0)
            per_take[name] = dict(collection=model, recording_group=rows[name]['recording_group'], **value)
        physical[model] = {k: aggregate([per_take[n][k] for n in selected['selected_takes']]) for k in KEYS}
        for group, names in selected['selected_by_recording'].items():
            groups[group] = dict(n=len(names), mean_cm={k:float(100*np.mean([per_take[n][k] for n in names])) for k in KEYS})
    common = original['retention']['M4']['retained_takes']
    assert len(common) == 10
    prediction = {m: original['prediction']['M4'][m]['filtered']['command_driven_tip'] for m in MODELS}
    for model in MODELS:
        values = []
        for name in common:
            assert not report['models'][model]['takes'][name]['training_exposure']
            _, mask, errors = metric_arrays(SOURCE/model/(name+'.npz'), report['models'][model]['marker_ids'], end)
            values.append(float(np.sqrt(np.nanmean(errors['command_driven_tip'][mask]**2))))
        checked = aggregate(values)
        for key in ('mean_cm', 'sd_cm'):
            np.testing.assert_allclose(checked[key], prediction[model][key], atol=1e-12, rtol=0)
    assert sha256_file(manifest_path) == selection_hash
    result = dict(interval_s=[0., end], planned_strike_time_s=strike,
        selection_sha256=selection_hash, physical=physical, per_flight_physical=per_take,
        recording_group_means=groups, prediction=prediction, prediction_takes=common,
        full_filtered_physical={m:original['physical'][m]['filtered'] for m in MODELS},
        unfiltered_physical={m:original['physical'][m]['all'] for m in MODELS},
        physical_contact_test='Unchanged author-confirmed 17/20 separate fixed-M4 trials; trial-level contacts not independently verified.')
    atomic_json(out/'summary.json', result)
    lines = ['# Five-flight physical evaluation: vertical M0-M4', '',
        'Retrospective subset: five filtered physical flights per executed command, balanced 2/2/1 across its three recording groups. '
        'Selection uses fixed seeds 20260915+i for Mi and was saved before calculating these subset results. '
        'It uses no prediction, tracking, targeting, or contact outcome. Earlier full-cohort outcomes were already known; this is not a preregistered prospective study.', '',
        '| Command | n | Tip-reference RMSE (cm) | Fixed-time distance (cm) | Minimum distance (cm) |', '|---|---:|---:|---:|---:|']
    for m, row in physical.items():
        cells = [f"{row[k]['mean_cm']:.2f} +/- {row[k]['sd_cm']:.2f}" for k in KEYS]
        lines.append('| '+m+' | 5 | '+' | '.join(cells)+' |')
    lines += ['', 'Means and sample standard deviations are across flights, not frames. Recording-group means and exact IDs are saved. '
        'Physical metrics were recomputed from immutable measured traces with the existing time window, reference, masks, and target-event definitions; all per-flight values matched the earlier evaluation within 1e-12 m.', '',
        'The common ten-M4-flight tip prediction evaluation is unchanged and was independently rechecked for every predictor. '
        'No model fitting, command changes, horizontal analysis, or contact-test filtering was performed. Full filtered and unfiltered physical baselines remain in summary.json.']
    (out/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    source_files = [SOURCE/'evidence_hashes.json', SOURCE/'filter.json', SOURCE/'summary.json', SOURCE/'report.json',
                    refdir/'reference.json', refdir/'reference.npz', Path(__file__), manifest_path, out/'summary.json', out/'README.md']
    atomic_json(out/'evidence_hashes.json', {str(p):sha256_file(p) for p in source_files})
    atomic_json(out/'status.json', dict(status='completed', physical_flights=25, prediction_flights=10,
        selection_sha256=selection_hash, fitting_performed=False, inference_reused=True))
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('select', 'score'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (freeze if args.stage == 'select' else score)(args.output.resolve())
