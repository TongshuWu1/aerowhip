"""Extend the frozen paper tip comparison to all ten observed cable markers.

Replays fixed models on the same recorded commands, initial states, masks, and
time grids. Every replay must reproduce the archived tip prediction. No fitting,
selection, command generation, data editing, or manuscript editing occurs.
"""
from pathlib import Path
import argparse
import csv
import gc
import json
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experimental_data.io import atomic_json, sha256_file
from experimental_data.model_evaluation import model_identity
from experimental_data.whip_adaptation import verify_hashes
from experimental_data.whip_adaptation_fit import records
from experimental_data.whip_full_fit import immutable_identity
from simulator.cable import DderState
from simulator.research_execution import ResearchExecutionModel
from simulator.workflow import read_json


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=ROOT / 'runs/evaluation/paper_M0_M1_takes_M0_M1_M2_20260913')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    baseline = args.baseline.resolve()
    contract = read_json(baseline / 'contract.json')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    atomic_json(output / 'status.json', {'status': 'running'})
    models = {name: Path(row['path']) for name, row in contract['models'].items()}
    assert list(models) == ['M0', 'M1', 'M2']
    jobs = {name: Path(path) for name, path in contract['batches'].items()}
    protected = {str(Path(__file__).resolve()): sha256_file(__file__),
                 str(baseline / 'contract.json'): sha256_file(baseline / 'contract.json')}
    common_geometry = immutable_identity(models['M0'])
    for name, path in models.items():
        identity, hashes = model_identity(path)
        assert identity == contract['models'][name]['signature'], name + ': model identity changed'
        assert immutable_identity(path) == common_geometry
        verify_hashes(contract['models'][name]['hashes'])
        protected.update(hashes)
    for folder in ('simulator', 'experimental_data'):
        for path in (ROOT / folder).rglob('*.py'):
            protected[str(path)] = sha256_file(path)
    for batch, job in jobs.items():
        saved = read_json(job / 'prepared_hashes.json')
        for filename in ('protocol.json', 'prepared_hashes.json'):
            protected[str(job / filename)] = sha256_file(job / filename)
        for number in range(1, 6):
            path = job / 'inputs' / f'{batch}_{number:03d}' / 'data.npz'
            expected = saved[str(path)]
            assert sha256_file(path) == expected
            protected[str(path)] = expected
    shutil.copy2(__file__, output / 'evaluation_source.py')
    report = {
        'schema': 'frozen_paper_all_marker_comparison_v1',
        'baseline_contract': contract,
        'metric': 'Per-flight 3D RMSE over all valid marker-time observations; equal-flight mean.',
        'weighting': 'Every valid observation has equal weight; no additional tip weighting.',
        'marker_scope': 'Ten non-attachment cable markers, including the tip; the attachment site is reconstructed from vehicle pose.',
        'coordinates': 'World frame; no spatial registration, attachment-relative alignment, or new time alignment.',
        'evidence': 'Retrospective development comparison on the same five M0 and five M1 recordings.',
        'interval_s': contract['interval_s'],
        'takes': {},
    }
    try:
        for batch, job in jobs.items():
            names = [f'{batch}_{number:03d}' for number in range(1, 6)]
            base_model = read_json(models['M0'])
            base = ResearchExecutionModel.from_mapping(base_model, root=models['M0'].parent, device=args.device)
            rows = records(job, names, base_model, base, args.device)
            marker_nodes = list(base.cable.marker_node_indices[1:])
            assert len(marker_nodes) == 10 and marker_nodes[-1] == 11
            report['marker_node_indices'] = marker_nodes
            report['marker_distances_m'] = np.cumsum(base_model['cable']['marker_interval_lengths_m']).tolist()
            physical_states = {}
            for row in rows:
                assert row['trial'].end == contract['interval_s'][1]
                pose = row['trial'].initial_pose(base.drone.parameters)
                physical_states[row['name']] = tuple(getattr(pose, key).clone() for key in ('position', 'velocity', 'rotation', 'omega_tracking'))
                report['takes'][row['name']] = {'batch': batch, 'models': {}}
            del base
            for model_name, path in models.items():
                engine = ResearchExecutionModel.from_mapping(read_json(path), root=path.parent, device=args.device)
                assert list(engine.cable.marker_node_indices[1:]) == marker_nodes
                folder = output / 'predictions' / batch / model_name
                folder.mkdir(parents=True)
                for row in rows:
                    name = row['name']
                    trial = row['trial']
                    data = trial.data
                    pose = trial.initial_pose(engine.drone.parameters)
                    for key, expected in zip(('position', 'velocity', 'rotation', 'omega_tracking'), physical_states[name]):
                        assert torch.equal(getattr(pose, key), expected)
                    prediction = engine.predict(
                        pose, DderState(row['q'][None].clone(), row['v'][None].clone()),
                        torch.as_tensor(data['packets'][None], device=trial.device, dtype=torch.float64),
                        data['packet_time'], row['grid'], graph=True,
                        hover_command=torch.as_tensor(data['hover_commands'][-1:], device=trial.device, dtype=torch.float64),
                    )
                    predicted_nodes = prediction['cable_positions_m'][0].cpu().numpy()
                    assert np.isfinite(predicted_nodes).all() and bool(prediction['valid'].all())
                    predicted = predicted_nodes[:, marker_nodes]
                    measured = row['truth'][:, 1:]
                    valid = row['valid'].copy()
                    score = (row['grid'] >= 0) & (row['grid'] < trial.end - 1e-9)
                    error = np.linalg.norm(predicted - measured, axis=-1)
                    error[~valid] = np.nan
                    archived_path = baseline / 'predictions' / batch / model_name / (name + '.npz')
                    protected[str(archived_path)] = sha256_file(archived_path)
                    with np.load(archived_path) as old:
                        for key, actual in [('time_s', row['grid']), ('measured_tip', measured[:, -1]),
                                            ('valid', valid[:, -1]), ('score', score),
                                            ('initial_cable_position', row['q'].cpu().numpy()),
                                            ('initial_cable_velocity', row['v'].cpu().numpy())]:
                            assert np.array_equal(actual, old[key], equal_nan=True), name + ': changed ' + key
                        tip_difference = float(np.max(np.abs(predicted[:, -1] - old['predicted_tip'])))
                        assert tip_difference <= 1e-8, f'{name}/{model_name}: archived tip mismatch {tip_difference}'
                    scored = error[score]
                    counts = np.isfinite(scored).sum(axis=0)
                    assert np.all(counts > 0)
                    mse_by_marker = np.nanmean(scored ** 2, axis=0)
                    metric = {
                        'all_marker_rmse_cm': float(100 * np.sqrt(np.nanmean(scored ** 2))),
                        'all_marker_mean_distance_cm': float(100 * np.nanmean(scored)),
                        'equal_marker_rmse_cm': float(100 * np.sqrt(mse_by_marker.mean())),
                        'per_marker_rmse_cm': (100 * np.sqrt(mse_by_marker)).tolist(),
                        'per_marker_mean_distance_cm': (100 * np.nanmean(scored, axis=0)).tolist(),
                        'per_marker_valid_count': counts.tolist(),
                        'scored_frames': int(score.sum()),
                        'tip_reproduction_max_abs_m': tip_difference,
                    }
                    np.savez_compressed(folder / (name + '.npz'), time_s=row['grid'],
                        predicted_markers_m=predicted, measured_markers_m=measured,
                        predicted_nodes_m=predicted_nodes, valid=valid, score=score,
                        error_m=error, marker_node_indices=marker_nodes)
                    report['takes'][name]['models'][model_name] = metric
                    print(f'{name} {model_name}: all-marker RMSE {metric["all_marker_rmse_cm"]:.4f} cm; tip replay difference {tip_difference:.2e} m', flush=True)
                    atomic_json(output / 'progress.json', {'completed': sum(len(x['models']) for x in report['takes'].values()), 'total': 30, 'elapsed_s': time.perf_counter() - started})
                del engine
                gc.collect()
                if args.device == 'cuda':
                    torch.cuda.empty_cache()
        names = list(report['takes'])
        assert len(names) == 10
        report['aggregates'] = {}
        for model in models:
            metrics = [report['takes'][name]['models'][model] for name in names]
            rmse = np.array([m['all_marker_rmse_cm'] for m in metrics])
            markers = np.array([m['per_marker_rmse_cm'] for m in metrics])
            report['aggregates'][model] = {
                'n_flights': len(names), 'mean_all_marker_rmse_cm': float(rmse.mean()),
                'sample_sd_all_marker_rmse_cm': float(rmse.std(ddof=1)),
                'mean_marker_rmse_cm': markers.mean(axis=0).tolist(),
                'sample_sd_marker_rmse_cm': markers.std(axis=0, ddof=1).tolist(),
                'mean_distance_cm': float(np.mean([m['all_marker_mean_distance_cm'] for m in metrics])),
                'equal_marker_mean_rmse_cm': float(np.mean([m['equal_marker_rmse_cm'] for m in metrics])),
            }
        for name in names:
            counts = [report['takes'][name]['models'][m]['per_marker_valid_count'] for m in models]
            assert counts[0] == counts[1] == counts[2]
        report['elapsed_s'] = time.perf_counter() - started
        atomic_json(output / 'report.json', report)
        with (output / 'per_flight_marker_errors.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['flight', 'model', 'all_marker_rmse_cm', 'mean_distance_cm', *[f'marker_{j}_rmse_cm' for j in range(1, 11)]])
            for name in names:
                for model, metric in report['takes'][name]['models'].items():
                    writer.writerow([name, model, metric['all_marker_rmse_cm'], metric['all_marker_mean_distance_cm'], *metric['per_marker_rmse_cm']])
        verify_hashes(protected)
        atomic_json(output / 'input_and_code_hashes.json', protected)
        atomic_json(output / 'output_hashes.json', {str(p): sha256_file(p) for p in output.rglob('*') if p.is_file() and p.name != 'status.json'})
        atomic_json(output / 'status.json', {'status': 'completed', 'elapsed_s': report['elapsed_s'],
            'all_30_tip_predictions_reproduced': True, 'input_hashes_verified': True})
        print(json.dumps(report['aggregates'], indent=2), flush=True)
    except Exception as error:
        atomic_json(output / 'status.json', {'status': 'failed', 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
