"""Diagnose attachment-offset identifiability and initialization velocity choices."""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from experimental_data.io import atomic_json
from experimental_data.force_dataset import (_normalized_rotations, local_polynomial_derivative,
                                             DifferentiationSettings)
from tools.audit_calibration import trajectory


def fit_offset(points):
    offset = np.array([0., 0., -.055])
    cost = lambda x: np.mean(np.sqrt(1 + ((np.linalg.norm(points-x, axis=1)-.063)/.001)**2)-1)
    for _ in range(100):
        delta = points-offset
        distance = np.linalg.norm(delta, axis=1)
        residual = distance-.063
        jacobian = -delta/distance[:, None]
        weight = (1+(residual/.001)**2)**(-.25)
        step = np.linalg.lstsq(jacobian*weight[:, None], -residual*weight, rcond=None)[0]
        old_cost = cost(offset)
        for exponent in range(16):
            candidate = np.clip(offset + step*2.**(-exponent), [-.02,-.02,-.08], [.02,.02,-.03])
            if cost(candidate) <= old_cost:
                break
        if np.linalg.norm(candidate-offset) < 1e-10:
            break
        offset = candidate
    return offset, float(np.linalg.cond(jacobian*weight[:, None]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit-job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((args.fit_job / 'force_takes/manifest.json').read_text())
    model = json.loads((args.fit_job / 'model.json').read_text())
    training, observations = [], {}
    for name, row in manifest['takes'].items():
        with np.load(root / 'data/processed_takes' / name / 'take.npz') as data:
            rotation, valid = _normalized_rotations(data['uav_orientation_xyzw'])
            valid &= data['auto_frame_valid'] & data['uav_valid'] & data['cable_marker_valid'][:,0]
            body = np.einsum('tji,tj->ti', rotation[valid],
                data['cable_marker_positions_m'][valid,0]-data['uav_position_m'][valid])
        observations[name] = body
        if row['role'] == 'training':
            training.append(body[np.linspace(0,len(body)-1,min(1000,len(body))).astype(int)])
    offset, condition = fit_offset(np.concatenate(training))
    report = dict(assumption='fixed straight 63mm first link; curvature and origin calibration confound this estimate',
                  fitted_offset_body_m=offset.tolist(), jacobian_condition=condition, per_take={})
    for name, points in observations.items():
        report['per_take'][name] = {label: dict(
            median_residual_mm=float(np.median(np.linalg.norm(points-value,axis=1)-.063)*1000),
            rmse_mm=float(np.sqrt(np.mean((np.linalg.norm(points-value,axis=1)-.063)**2))*1000))
            for label,value in [('configured',np.array([0.,0.,-.055])),('estimated',offset)]}
    atomic_json(args.output / 'attachment_offset_diagnostic.json', report)
    print(json.dumps(report), flush=True)
    rows = []
    with tempfile.TemporaryDirectory() as temp:
        for name in ('fig8_001','fig8_003','osc_003'):
            with np.load(args.fit_job / 'force_takes' / name / 'take.npz') as source:
                arrays = {key: source[key].copy() for key in source.files}
            velocities = {}
            common = arrays['state_valid'].copy()
            for window in (7,11,21,31):
                velocity, valid = local_polynomial_derivative(arrays['cable_node_position_world_m'],
                    arrays['cable_node_valid'],dt_s=.01,derivative_order=1,
                    settings=DifferentiationSettings(window_samples=window))
                velocities[window] = velocity
                common &= valid.all(axis=1)
            destination = Path(temp) / name / 'take.npz'
            destination.parent.mkdir()
            for window, velocity in velocities.items():
                current = {**arrays, 'cable_node_velocity_world_m_s': velocity,
                    'root_velocity_world_m_s': velocity[:,0], 'state_valid':common}
                np.savez_compressed(destination, **current)
                _, result = trajectory(model, destination)
                row = dict(take=name, derivative_window=window, **result)
                rows.append(row)
                print(json.dumps(row), flush=True)
                atomic_json(args.output / 'velocity_sensitivity.json', rows)


if __name__ == '__main__':
    main()
