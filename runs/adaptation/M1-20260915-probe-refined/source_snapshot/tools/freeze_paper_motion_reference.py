"""Freeze the user's original M0 physical-motion reference; never re-simulate it."""
from pathlib import Path
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experimental_data.io import atomic_json, sha256_file
from planning.position_spline import PositionSpline


def main():
    source = ROOT / 'runs/rehearsals_pva/20260913-012740-484590-M0-slower-brake-1s'
    export = ROOT / 'exports/M0_Bspline_slower_brake_1s/fullstate_30hz.csv'
    out = ROOT / 'runs/reference_tracking/M0-paper-fixed-reference'
    expected_csv = '49f9fd4b8e356543cb5523abca866646968f58b603f73bb94466f0734a1e8d90'
    if sha256_file(export) != expected_csv or sha256_file(source / 'fullstate_30hz.csv') != expected_csv:
        raise ValueError('Original executed command identity changed')
    meta = json.loads((source / 'rehearsal.json').read_text())
    cfg = json.loads((source / 'settings.json').read_text())
    end = 34 / 30
    strike = 1.1172482457473654
    if not np.isclose(meta['whip_end_s'], end) or not np.isclose(meta['predicted_hit_time_s'], strike):
        raise ValueError('Original strike or handover time changed')
    with np.load(source / 'rehearsal.npz', allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    with np.load(source / 'plan.npz', allow_pickle=False) as saved:
        controls = saved['position_control_points_m'].copy()
        spline = PositionSpline(cfg['task']['duration_s'])
        packets, _ = spline.decode(torch.tensor(controls), cfg['launch']['origin_m'])
        if not np.allclose(saved['command_packets'], packets.numpy(), atol=1e-10, rtol=0):
            raise ValueError('Original command spline cannot be reproduced')
    csv = np.loadtxt(export, delimiter=',', skiprows=1)
    if not np.allclose(csv, np.c_[arrays['command_time_s'], arrays['commands']], atol=1e-12, rtol=0):
        raise ValueError('Saved forecast command differs from the flown CSV')
    count = round(end * 30) + 1
    if not np.allclose(arrays['commands'][:count], packets.numpy()[:count], atol=1e-10, rtol=0):
        raise ValueError('Executed prefix differs from the original command spline')
    grid = arrays['prediction_time_s']
    keep = grid <= end + 1e-10
    if not np.isclose(grid[keep][-1], end) or not np.allclose(np.diff(grid), 1/150, atol=1e-10, rtol=0):
        raise ValueError('Reference time grid differs from the frozen physics schedule')
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Reference contains nonfinite values')
    sources = [source / n for n in ('rehearsal.npz', 'rehearsal.json', 'settings.json', 'plan.npz', 'model.json')]
    sources += [export, Path(__file__)]
    hashes = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sources}
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / 'reference.npz',
        time_s=grid[keep], tip_position_m=arrays['cable_positions_m'][keep, -1],
        quadrotor_position_m=arrays['origin_positions_m'][keep],
        cable_position_m=arrays['cable_positions_m'][keep],
        command_time_s=arrays['command_time_s'][:count],
        original_command_packets=arrays['commands'][:count],
        original_position_control_points_m=controls,
        original_spline_knots_s=spline.knots,
        original_full_prediction_time_s=grid,
        original_full_cable_positions_m=arrays['cable_positions_m'],
        original_full_quadrotor_positions_m=arrays['origin_positions_m'],
        target_position_m=arrays['target_position_m'])
    atomic_json(out / 'reference.json', dict(
        schema='frozen_m0_physical_motion_reference_v1',
        source_rehearsal=source.relative_to(ROOT).as_posix(), source_hashes=hashes,
        reference_sha256=sha256_file(out / 'reference.npz'),
        physical_target_m=cfg['launch']['target_m'], launch_origin_m=cfg['launch']['origin_m'],
        interval_s=[0., end], planned_strike_time_s=strike,
        original_spline_domain_s=[0., cfg['task']['duration_s']],
        priority='Tip trajectory primary; quadrotor trajectory soft secondary; original timestamps fixed.',
        reference_kind='Original M0 predicted physical motion; not command positions or a new simulation.',
        recovery_settings=cfg['recovery'], limits=cfg['limits'],
        jerk_limits_m_s3=cfg['action']['jerk_limit_m_s3'],
        numerical_objective_settings=None,
        correction_status='Reference frozen only; correction optimizer not yet implemented or run.'))
    with np.load(out / 'reference.npz') as saved:
        np.testing.assert_array_equal(saved['tip_position_m'], arrays['cable_positions_m'][keep, -1])
        np.testing.assert_array_equal(saved['quadrotor_position_m'], arrays['origin_positions_m'][keep])
    for path, digest in hashes.items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError(f'Source changed while freezing: {path}')
    print(out.as_posix())
    print(f'Frozen {int(keep.sum())} physics samples and {count} command packets; exact source arrays verified.')


if __name__ == '__main__':
    main()
