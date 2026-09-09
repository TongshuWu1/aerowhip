"""Read-only calibration audit; write diagnostics to a separate output directory."""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from experimental_data.io import atomic_json
from experimental_data.cable_fit import _prepare_take
from experimental_data.force_dataset import _normalized_rotations
from simulator.cable import CableConfiguration, START_PINNED_FREE_END
from simulator.cable.dder import DderModel


@torch.no_grad()
def trajectory(model, path, *, substeps=3, iterations=4, dtype=torch.float64, device='cpu',
               optimized=False, zero_velocity=False):
    payload = deepcopy(model['cable'])
    payload.update(substeps=substeps, constraint_iterations=iterations)
    cable = CableConfiguration.from_mapping(payload)
    dder = DderModel(cable.dder_parameters(EI=payload['EI_n_m2'], Cb=payload['Cb_n_m2_s']))
    take = _prepare_take(path.parent.name, 'diagnostic', path, model=dder, cable=cable,
        device=torch.device(device), dtype=dtype, horizon_s=.7, stride_s=.7, projection_passes=8)
    # First three complete windows. Fixed samples across all numerical variants.
    q, v = take.initial_positions_m[:3], take.initial_velocities_m_s[:3]
    if zero_velocity:
        v = torch.zeros_like(v)
    state = dder.initial_state(q, v)
    constants = dder.runtime_constants(q)
    frames = [q.clone()]
    errors = []
    for step in range(1, take.horizon_steps + 1):
        state = dder.step_runtime(state, take.root_positions_m[:3, step, None],
            q.new_full((len(q),), take.dt_s), constants, iterative_damping=optimized,
            damping_backend='pcg32_experimental' if optimized else 'pcg60_reference',
            pinned_endpoints=START_PINNED_FREE_END, create_graph=False)
        frames.append(state.positions_m.clone())
        errors.append(dder.maximum_segment_error_m(state.positions_m))
    predicted = torch.stack(frames).cpu().numpy().transpose(1, 0, 2, 3)
    measured = take.measured_marker_positions_m[:3].cpu().numpy()
    difference = predicted[:, :, cable.marker_node_indices[1:]] - measured
    result = dict(marker_rmse_mm=float(np.sqrt(np.mean(np.sum(difference[:, 1:]**2, axis=-1)))*1000),
        tip_rmse_mm=float(np.sqrt(np.mean(np.sum(difference[:, 1:, -1]**2, axis=-1)))*1000),
        maximum_segment_error_mm=float(torch.stack(errors).max().cpu())*1000,
        initial_marker_rmse_mm=float(np.sqrt(np.mean(np.sum(difference[:, 0]**2, axis=-1)))*1000))
    return predicted, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit-job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = json.loads((args.fit_job / 'model.json').read_text())
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.fit_job / 'force_takes/manifest.json').read_text())
    geometry = {}
    for name, row in manifest['takes'].items():
        with np.load(args.fit_job / 'force_takes' / row['path']) as data:
            q = data['cable_node_position_world_m'][data['state_valid']]
            sites = q[:, [0, *range(2, 12)]]
            lengths = np.linalg.norm(np.diff(sites, axis=1), axis=-1)
        geometry[name] = dict(role=row['role'], frames=len(q),
            interval_percentiles_mm=(np.percentile(lengths, [5, 50, 95], axis=0)*1000).tolist())
    atomic_json(args.output / 'geometry.json', geometry)
    results = []
    for name in ('fig8_001', 'fig8_003', 'osc_003'):
        path = args.fit_job / 'force_takes' / name / 'take.npz'
        trajectories = {}
        variants = [(3, 4), (3, 12), (6, 4), (12, 4), (24, 4)]
        for steps, iterations in variants:
            start = time.perf_counter()
            q, metrics = trajectory(model, path, substeps=steps, iterations=iterations)
            trajectories[f's{steps}_i{iterations}'] = q
            row = dict(take=name, substeps=steps, iterations=iterations, **metrics,
                       runtime_s=time.perf_counter()-start)
            results.append(row)
            print(json.dumps(row), flush=True)
            atomic_json(args.output / 'convergence.json', results)
        reference = trajectories['s24_i4']
        for row in results:
            if row['take'] == name:
                q = trajectories[f's{row["substeps"]}_i{row["iterations"]}']
                row['tip_difference_from_24_substeps_mm'] = float(np.sqrt(np.mean(np.sum((q[:,:,-1]-reference[:,:,-1])**2,axis=-1)))*1000)
        np.savez_compressed(args.output / f'{name}_convergence.npz', **trajectories)
        atomic_json(args.output / 'convergence.json', results)


if __name__ == '__main__':
    main()
