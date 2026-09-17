"""Frozen M0/M1/M2 inference on the ten retained M0/M1 development takes.

No fitting, selection, flight commands, original forecasts, or data are changed.
The existing reviewed masks, causal initialization and targeting interval apply.
"""
from pathlib import Path
import argparse
import csv
import gc
import json
import platform
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experimental_data.io import atomic_json, sha256_file
from experimental_data.model_evaluation import model_identity
from experimental_data.whip_adaptation import verify_hashes, rms_summary
from experimental_data.whip_adaptation_fit import records
from experimental_data.whip_full_fit import immutable_identity
from simulator.cable import DderState
from simulator.research_execution import ResearchExecutionModel
from simulator.workflow import read_json

MODELS = {
    'M0': ROOT/'runs/rehearsals_pva/20260913-012740-484590-M0-slower-brake-1s/model.json',
    'M1': ROOT/'runs/rehearsals_pva/20260913-032256-816506-M1-local-fixed-tip-reference/model.json',
    'M2': ROOT/'runs/adaptation/M2-selected-20260913/candidate/model.json',
}
JOBS = {
    'M0': ROOT/'runs/adaptation/M1-paper-20260913-verified-physical',
    'M1': ROOT/'runs/adaptation/M2-paper-20260913',
}


def data_use(take):
    batch, number = take.split('_')
    if number in ('001', '002', '004'):
        return 'M1 fitting and M2 replay' if batch == 'M0' else 'M2 fitting'
    return 'Development validation' if batch == 'M0' else 'M2 model selection'


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    protected = {str(Path(__file__).resolve()): sha256_file(__file__)}
    identities = {}
    common_identity = immutable_identity(MODELS['M0'])
    for name, path in MODELS.items():
        signature, hashes = model_identity(path)
        protected.update(hashes)
        assert immutable_identity(path) == common_identity
        identities[name] = dict(path=str(path), signature=signature, hashes=hashes)
    for folder in ('simulator', 'experimental_data'):
        for path in (ROOT/folder).rglob('*.py'):
            protected[str(path)] = sha256_file(path)
    for batch, job in JOBS.items():
        saved = read_json(job/'prepared_hashes.json')
        for filename in ('protocol.json', 'prepared_hashes.json'):
            protected[str(job/filename)] = sha256_file(job/filename)
        for number in range(1, 6):
            path = job/'inputs'/f'{batch}_{number:03d}'/'data.npz'
            expected = saved.get(str(path))
            if expected is None:
                raise ValueError('Prepared input missing from frozen manifest: '+str(path))
            assert sha256_file(path) == expected
            protected[str(path)] = expected
    atomic_json(out/'inputs_and_code_hashes.json', protected)
    contract = dict(
        schema='frozen_paper_tip_comparison_v1', models=identities,
        batches={k: str(v) for k, v in JOBS.items()},
        interval_s=[0., 34/30], rate_hz=150.,
        evidence='Retrospective development comparison; no independent final-test claim.',
        initialization='Identical observed causal history and projected cable state for each take. Vehicle position, velocity, rotation and angular velocity are shared; effective hover compensation and alignment are inferred for each model using the same pre-maneuver-only procedure.',
        scoring='3D tip RMSE on identical timestamps and valid masks; no spatial registration or model-specific time alignment. Average per-take RMSEs equally.',
        excluded='Old M2 physical flights and protected recordings are not accessed.',
        predictions='New retrospective predictions, not original preflight forecasts.',
        device=args.device, os=platform.platform(),
        gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None,
    )
    atomic_json(out/'contract.json', contract)
    report = dict(contract=contract, takes={}, aggregates={})
    for batch, job in JOBS.items():
        names = [f'{batch}_{i:03d}' for i in range(1, 6)]
        base_model = read_json(MODELS['M0'])
        base = ResearchExecutionModel.from_mapping(base_model, root=MODELS['M0'].parent, device=args.device)
        rows = records(job, names, base_model, base, args.device)
        initial_shared = {}
        for row in rows:
            assert row['trial'].end == 34/30
            t = row['trial']
            pose = t.initial_pose(base.drone.parameters)
            initial_shared[row['name']] = tuple(getattr(pose, key).clone() for key in ('position', 'velocity', 'rotation', 'omega_tracking'))
            report['takes'][row['name']] = dict(data_use=data_use(row['name']), original_role=t.protocol['takes'][row['name']]['role'], models={})
        del base
        for model_name, path in MODELS.items():
            value = read_json(path)
            engine = ResearchExecutionModel.from_mapping(value, root=path.parent, device=args.device)
            folder = out/'predictions'/batch/model_name
            folder.mkdir(parents=True)
            for row in rows:
                name = row['name']
                trial = row['trial']
                d = trial.data
                pose = trial.initial_pose(engine.drone.parameters)
                for actual, expected in zip((getattr(pose, key) for key in ('position', 'velocity', 'rotation', 'omega_tracking')), initial_shared[name]):
                    assert torch.equal(actual, expected), 'Unequal measured initial physical state'
                prediction = engine.predict(
                    pose, DderState(row['q'][None].clone(), row['v'][None].clone()),
                    torch.as_tensor(d['packets'][None], device=trial.device, dtype=torch.float64),
                    d['packet_time'], row['grid'], graph=True,
                    hover_command=torch.as_tensor(d['hover_commands'][-1:], device=trial.device, dtype=torch.float64),
                )
                tip = prediction['cable_positions_m'][0, :, -1].cpu().numpy()
                assert np.isfinite(tip).all() and bool(prediction['valid'].all())
                truth = row['truth'][:, -1]
                valid = row['valid'][:, -1]
                score = (row['grid'] >= 0) & (row['grid'] < trial.end-1e-9)
                error = np.linalg.norm(tip-truth, axis=-1)
                error[~valid] = np.nan
                metric = rms_summary(error[score])
                np.savez_compressed(folder/(name+'.npz'), time_s=row['grid'], predicted_tip=tip,
                    measured_tip=truth, valid=valid, score=score, error_m=error,
                    initial_cable_position=row['q'].cpu().numpy(), initial_cable_velocity=row['v'].cpu().numpy(),
                    initial_hover_compensation=pose.compensation.cpu().numpy(), initial_alignment=pose.rotation_command_from_tracking.cpu().numpy())
                report['takes'][name]['models'][model_name] = metric
                print(f'{name} {model_name}: tip RMSE {100*metric["rmse_m"]:.4f} cm; coverage {metric["coverage"]:.3f}', flush=True)
                atomic_json(out/'progress.json', dict(takes=report['takes'], elapsed_s=time.perf_counter()-started))
            del engine
            gc.collect()
            if args.device == 'cuda':
                torch.cuda.empty_cache()
    for batch in JOBS:
        for role, numbers in [('fitting', (1, 2, 4)), ('development', (3, 5)), ('all', (1, 2, 3, 4, 5))]:
            names = [f'{batch}_{n:03d}' for n in numbers]
            group = dict(takes=names, n=len(names), models={})
            for model in MODELS:
                values = np.array([report['takes'][n]['models'][model]['rmse_m'] for n in names])
                group['models'][model] = dict(mean_m=float(values.mean()), sample_sd_m=float(values.std(ddof=1)), per_take_m=values.tolist())
            report['aggregates'][batch+'_'+role] = group
    report['elapsed_s'] = time.perf_counter()-started
    atomic_json(out/'report.json', report)
    with (out/'per_take_tip_rmse.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['take', 'data_use', 'M0_rmse_cm', 'M1_rmse_cm', 'M2_rmse_cm', 'tip_coverage'])
        for name, row in report['takes'].items():
            coverage = [row['models'][m]['coverage'] for m in MODELS]
            assert len(set(coverage)) == 1
            writer.writerow([name, row['data_use'], *[100*row['models'][m]['rmse_m'] for m in MODELS], coverage[0]])
    verify_hashes(protected)
    hashes = {str(p): sha256_file(p) for p in out.rglob('*') if p.is_file()}
    atomic_json(out/'output_hashes.json', hashes)
    atomic_json(out/'status.json', dict(status='completed', elapsed_s=report['elapsed_s'], input_hashes_verified=True))
    print(json.dumps(report['aggregates'], indent=2), flush=True)


if __name__ == '__main__':
    main()

