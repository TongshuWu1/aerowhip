"""Rescore saved vertical traces over the fixed correction interval; no fitting."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experimental_data.model_evaluation import load_evaluation, metric_arrays
from experimental_data.io import atomic_json, sha256_file
from simulator.workflow import read_json


def summary(values):
    a = np.asarray(values, dtype=float)
    assert len(a) > 1 and np.isfinite(a).all()
    return dict(n=len(a), mean_cm=float(100*a.mean()), sd_cm=float(100*a.std(ddof=1)))


def main():
    source = ROOT/'runs/evaluation/vertical-only-M0-M5-paired-complete-20260915'
    report = load_evaluation(source)
    reference = ROOT/'runs/reference_tracking/M0-2cm-brake-1p3s-fixed-reference'
    meta = read_json(reference/'reference.json')
    assert sha256_file(reference/'reference.npz') == meta['reference_sha256']
    with np.load(reference/'reference.npz') as z:
        rt, rq = z['time_s'].copy(), z['tip_position_m'].copy()
    end = meta['interval_s'][1]
    result = dict(interval_s=[0.,end], endpoint_policy='left inclusive, right exclusive',
        planned_strike_time_s=meta['planned_strike_time_s'], physical={}, prediction={},
        physical_per_take={}, prediction_per_take={}, recording_group_means={},
        hit_test=dict(model='M4', successful_contacts=17, attempts=20,
            source='User confirmation in current conversation; separate fixed-M4-command repeatability test',
            trial_level_records_verified=False))
    first = report['models']['M0']
    for name, row in first['takes'].items():
        a, score, _ = metric_arrays(source/'M0'/(name+'.npz'), first['marker_ids'], end)
        t = a['time_s'][score]
        assert t.min() >= rt[0]-1e-9 and t.max() <= rt[-1]+1e-9
        ref = np.stack([np.interp(t, rt, rq[:,j]) for j in range(3)], axis=1)
        valid = a['mask'][score,-1]
        delta = a['measured_sites'][score,-1]-ref
        events = row['event_metrics']
        result['physical_per_take'][name] = dict(collection=row['collection'],
            recording_group=row['recording_group'], valid_tip_samples=int(valid.sum()),
            reference_rmse_m=float(np.sqrt(np.mean(np.sum(delta[valid]**2, axis=-1)))),
            planned_target_distance_m=events['measured_planned_time_target_distance_m'],
            minimum_target_distance_m=events['measured']['distance_m'])
    for collection in [f'M{i}' for i in range(5)]:
        rows = [r for r in result['physical_per_take'].values() if r['collection']==collection]
        result['physical'][collection] = {key:summary([r[key] for r in rows]) for key in
            ('reference_rmse_m','planned_target_distance_m','minimum_target_distance_m')}
    for group in sorted({r['recording_group'] for r in result['physical_per_take'].values()}):
        rows = [r for r in result['physical_per_take'].values() if r['recording_group']==group]
        result['recording_group_means'][group] = {key:float(np.mean([r[key] for r in rows])) for key in
            ('reference_rmse_m','planned_target_distance_m','minimum_target_distance_m')}
    for model in [f'M{i}' for i in range(5)]:
        rows = report['models'][model]
        values = {}
        for name, row in rows['takes'].items():
            if row['collection'] != 'M4':
                continue
            assert not row['training_exposure']
            a, score, errors = metric_arrays(source/model/(name+'.npz'), rows['marker_ids'], end)
            values[name] = {k:float(np.sqrt(np.nanmean(errors[k][score]**2)))
                           for k in ('command_driven_tip','command_driven_markers','drone')}
        assert len(values)==12
        result['prediction_per_take'][model] = values
        result['prediction'][model] = {k:summary([r[k] for r in values.values()])
                                      for k in ('command_driven_tip','command_driven_markers','drone')}
    result['source_hashes'] = {str(p):sha256_file(p) for p in
        (source/'evidence_hashes.json', reference/'reference.json', reference/'reference.npz', Path(__file__))}
    out = ROOT/'output/paper/experiments_m0_m4_20260915'
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'metrics.json', result)
    print('Physical:', result['physical'])
    print('Prediction:', result['prediction'])


if __name__ == '__main__':
    main()
