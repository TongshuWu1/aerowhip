"""Training-only attachment calibration with fixed ruler geometry and sensitivity profiles."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .force_dataset import _normalized_rotations
from .io import atomic_json, sha256_file


def load_geometry_data(source, root):
    roles = json.loads((source / 'dataset_manifest.json').read_text())['takes']
    data, hashes = {}, {}
    for name, row in roles.items():
        if row['role'] not in ('training', 'validation') or not row.get('enabled', True):
            continue
        processed = source/'processed_takes' if (source/'processed_takes').exists() else root/'data/processed_takes'
        paths = [processed / name / 'take.npz', source / 'force_takes' / name / 'take.npz']
        with np.load(paths[0]) as p, np.load(paths[1]) as f:
            np.testing.assert_allclose(p['time_s'], f['time_s'])
            rotation, valid_r = _normalized_rotations(p['uav_orientation_xyzw'])
            valid = f['state_valid'] & valid_r
            delta = p['cable_marker_positions_m'][:, 0] - p['uav_position_m']
            tangent = p['cable_marker_positions_m'][:, 1] - p['cable_marker_positions_m'][:, 0]
            tangent /= np.linalg.norm(tangent, axis=1)[:, None]
            body_c1 = np.einsum('tji,tj->ti', rotation, delta)
            body_tangent = np.einsum('tji,tj->ti', rotation, tangent)
            velocity = f['cable_node_velocity_world_m_s']
            quiet = valid & (np.linalg.norm(velocity[:, 0], axis=1) < .05)
            quiet &= np.linalg.norm(velocity[:, -1] - velocity[:, 0], axis=1) < .1
            data[name] = dict(role=row['role'], c1=body_c1[valid],
                              quiet_c1=body_c1[quiet], quiet_tangent=body_tangent[quiet])
        hashes.update({str(p): sha256_file(p) for p in paths})
    return data, hashes


def robust(distance, scale=.002):
    return scale * (torch.sqrt(1 + (distance / scale).square()) - 1)


def fit_offset(training, *, height=.055, length=.063, exclude=None):
    """Fit x/y only. Profile, rather than identify, confounded height and first span."""
    groups = []
    for name, row in training.items():
        if row['role'] != 'training' or name == exclude:
            continue
        def spread(a, count):
            return torch.tensor(a[np.linspace(0, len(a)-1, min(count, len(a)), dtype=int)], dtype=torch.float64)
        c1 = spread(row['c1'], 800)
        quiet_c1 = spread(row['quiet_c1'], 300)
        quiet_tangent = spread(row['quiet_tangent'], 300)
        groups.append((c1, quiet_c1, quiet_tangent))
    if not groups:
        raise ValueError('Geometry fitting requires training observations.')
    raw = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    optimizer = torch.optim.LBFGS([raw], lr=1., max_iter=100, tolerance_grad=1e-10,
                                tolerance_change=1e-12, line_search_fn='strong_wolfe')
    def objective():
        xy = .02 * raw.tanh()
        offset = torch.cat((xy, xy.new_tensor([-height])))
        losses = []
        for c1, quiet_c1, tangent in groups:
            excess = (torch.linalg.vector_norm(c1-offset, dim=-1)-length-.002).clamp_min(0)
            loss = robust(excess).mean()
            if len(quiet_c1):
                error = torch.linalg.vector_norm(quiet_c1-length*tangent-offset, dim=-1)
                loss = loss + .25 * robust(error).mean()
            losses.append(loss)
        return torch.stack(losses).mean() + .0002 * (xy/.02).square().mean()
    def closure():
        optimizer.zero_grad()
        value = objective()
        value.backward()
        return value
    optimizer.step(closure)
    offset = [*(.02 * raw.detach().tanh()).tolist(), -height]
    return dict(offset_body_m=offset, first_span_m=length, objective_m=float(objective().detach()))


def consistency(data, offset, length):
    result = {}
    for name, row in data.items():
        chord = np.linalg.norm(row['c1']-np.array(offset), axis=1)
        result[name] = dict(role=row['role'], frames=len(chord),
            chord_p95_m=float(np.percentile(chord, 95)),
            excess_over_2mm_percent=float(100*(chord>length+.002).mean()),
            mean_excess_m=float(np.maximum(chord-length-.002, 0).mean()))
    return result


def calibrate(source, output, root=None):
    root = root or Path(__file__).resolve().parents[1]
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    data, hashes = load_geometry_data(source, root)
    selected = fit_offset(data)
    profiles = [dict(height_m=h, **fit_offset(data, height=h, length=l))
                for h in (.052, .055, .058) for l in (.060, .063, .066)]
    leave_out = {name: fit_offset(data, exclude=name) for name, r in data.items() if r['role']=='training'}
    report = dict(selected=selected, profiles=profiles, leave_one_training_take_out=leave_out,
        original_consistency=consistency(data, [0, 0, -.055], .063),
        candidate_consistency=consistency(data, selected['offset_body_m'], .063),
        applied=False, protected_test_used=False,
        protocol=dict(source=str(source), input_hashes=hashes,
            solver='float64 LBFGS, strong Wolfe, 100 maximum iterations',
            selected_geometry='Fix height 55 mm and first span 63 mm; optimize bounded lateral offset only.',
            lateral_bounds_m=[-.02, .02], feasibility_tolerance_m=.002,
            quiet_tangent_loss_weight=.25, lateral_prior_weight_m=.0002,
            sampling='At most 800 valid and 300 quiet frames per training take, equally spaced; equal take loss.',
            sensitivity='Height 52/55/58 and span 60/63/66 mm are assumed sensitivity ranges, not ruler uncertainty or confidence intervals.',
            limitations=['Quiet first-span tangent extrapolation is approximate.',
                'Geometry is a constrained candidate, not certified attachment metrology.',
                'Profiles and leave-one-take-out checks do not use validation for selection.']))
    atomic_json(output/'result.json', report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    calibrate(a.source.resolve(), a.output.resolve())
