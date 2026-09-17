"""Reproduce the 2026-09-13 M0 intake and frozen-forecast evaluation.

Specific study record, not an automatic contact review or a model fit.
Original files are copied without changing bytes. Timing uses measured drone
streams only; neither the model prediction nor target determines alignment.
"""
from pathlib import Path
import sys
import csv
import shutil
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experimental_data.io import atomic_json, sha256_file
from experimental_data.adaptation_rounds import read_optitrack, read_controller, align, COMMAND_COLUMNS
from experimental_data.adaptation_check import command_onset, load_comparison, interpolate_positions
from experimental_data import whip_adaptation as adaptation
from experimental_data.flight_performance import encounter, local_velocity
from experimental_data.current_adaptation import finite_json
from simulator.workflow import read_json

EXPORT = ROOT / 'exports/M0_Bspline_slower_brake_1s'
BATCH = ROOT / 'data/flight_batches/M0_paper_slower_brake_20260913'
OUT = ROOT / 'runs/data_review/M0-paper-20260913'
REHEARSAL = ROOT / 'runs/rehearsals_pva/20260913-012740-484590-M0-slower-brake-1s'
ROLES = {f'M0_{i:03d}': ('adaptation' if i in (1, 2, 4) else 'validation') for i in range(1, 6)}


def save(path, value):
    atomic_json(path, finite_json(value))


def clip_observations(t, p, lo, hi):
    interior = (t > lo) & (t < hi)
    grid = np.r_[lo, t[interior], hi]
    return grid, interpolate_positions(t, p, grid)


def nearest(t, p, target):
    # Radius is only required by the existing utility; no sphere outcome is used.
    result = encounter(t, p, target, radius=1e-9)
    return {k: result[k] for k in ('nearest', 'sample_coverage', 'contiguous_segment_coverage')}


def rms(error):
    return adaptation.rms_summary(error)


def main():
    if BATCH.exists() or OUT.exists():
        raise FileExistsError('Study artifacts already exist; do not overwrite frozen evidence.')
    BATCH.mkdir(parents=True)
    (BATCH / 'flight_take').mkdir()
    (BATCH / 'simulation_csv').mkdir()
    frozen = {str(f): sha256_file(f) for f in REHEARSAL.rglob('*') if f.is_file() and f.name != 'ARCHIVED'}
    raw_hashes = {}
    for name in ROLES:
        for filename in (name + '.csv', 'experiment_' + name + '.csv'):
            source = EXPORT / 'flight_take' / filename
            raw_hashes[str(source)] = sha256_file(source)
            dest = BATCH / 'flight_take' / filename
            shutil.copy2(source, dest)
            assert sha256_file(dest) == raw_hashes[str(source)]
    shutil.copy2(EXPORT / 'fullstate_30hz.csv', BATCH / 'simulation_csv/fullstate_30hz.csv')
    reference = np.genfromtxt(BATCH / 'simulation_csv/fullstate_30hz.csv', delimiter=',', names=True)
    metadata = read_json(REHEARSAL / 'rehearsal.json')
    assert sha256_file(EXPORT / 'fullstate_30hz.csv') == metadata['csv_sha256']
    protocol = dict(schema=adaptation.SCHEMA, model_id='M0', rehearsal=str(REHEARSAL), frozen_hashes=frozen,
        parent_generation=0, parent_forecast_model_sha256=sha256_file(REHEARSAL / 'model.json'),
        command_sha256=metadata['csv_sha256'], forecast_sha256=sha256_file(REHEARSAL / 'rehearsal.npz'),
        frame='raw_global_xyz', normalization=False, cable_history_s=1., drone_history_s=.4,
        cable_velocity_weight_tau_s=.02, planned_roles=ROLES, role_unit='whole take',
        minimum_observation_fraction=.8, automatic_selection=False, neural_training=True,
        model_update='Existing staged full identification: new whip adaptation takes plus preliminary training only',
        role_source='docs/paper/PAPER_EXPERIMENT_PROTOCOL.md (001/002/004 adaptation; 003/005 operational validation)',
        endpoints=dict(planned_strike_time_s=metadata['predicted_hit_time_s'], target_m=[1.25, 0., 1.25],
            common_evaluation_interval_s=[0., 1.5], fit_end_s=metadata['whip_end_s'],
            no_binary_paper_hit_threshold=True, timing_sensitivity_offsets_s=[-.01, 0., .01]),
        operator_review='User confirms all five are clean takes with no physical contact, abort or intervention; same M0 setup.',
        qualification='This is the update batch, not the final independent physical comparison. Clock offsets are estimates.')
    save(BATCH / 'protocol.json', protocol)
    save(BATCH / 'original_raw_hashes.json', raw_hashes)
    alignments = {}
    quality = {}
    values = np.column_stack([reference[n] for n in reference.dtype.names[1:]])
    dynamic = np.linalg.norm(values[:, 3:9], axis=1) > 1e-5
    for name in ROLES:
        mp = BATCH / 'flight_take' / (name + '.csv')
        cp = BATCH / 'flight_take' / ('experiment_' + name + '.csv')
        m, c = read_optitrack(mp), read_controller(cp)
        a = align(m, c)
        onset, count, jitter = command_onset(c, reference)
        commands = np.column_stack([c[n] for n in COMMAND_COLUMNS])
        good = np.isfinite(commands).all(1) & (c['cmd_valid'] > .5) & (c['cmd_age'] >= 0)
        distances, ids = cKDTree(values).query(commands[good])
        matched = set(ids[distances < 1e-8].tolist())
        missing = [int(i) for i in np.flatnonzero(dynamic) if i not in matched]
        if missing:
            raise ValueError(f'{name}: missing dynamic CSV packets {missing}')
        t = m['time'] + a['offset_s'] - onset
        quality[name] = dict(alignment=a, onset_s=onset, matched_dynamic_packets=count,
            expected_dynamic_packets=int(dynamic.sum()), missing_dynamic_packets=missing,
            packet_timing_spread_s=jitter, relative_tracking_span_s=t[[0, -1]].tolist(),
            marker_missing_by_column=(~np.isfinite(m['cable']).all(-1)).sum(0).tolist(),
            strike_marker_coverage=float(np.isfinite(m['cable'][(t >= 0) & (t <= 1.5)]).all(-1).mean()),
            strike_tip_coverage=float(np.isfinite(m['cable'][(t >= 0) & (t <= 1.5), -1]).all(-1).mean()))
        alignments[name] = dict(offset_s=a['offset_s'], source=a['method'], clock_verified=False,
            optitrack_sha256=sha256_file(mp), controller_sha256=sha256_file(cp), diagnostic=a)
        save(mp.with_suffix('.tracking.json'), dict(drone=m['drone_label'], optitrack_sha256=sha256_file(mp)))
    save(BATCH / 'time_alignment.json', alignments)
    save(BATCH / 'intake_quality.json', quality)
    adaptation.compare(ROOT, BATCH, OUT)
    rows = {}
    hit_time = protocol['endpoints']['planned_strike_time_s']
    target = np.asarray(protocol['endpoints']['target_m'])
    for name in ROLES:
        d = load_comparison(ROOT, BATCH, name)
        t, tip = d['time'], d['measured_cable'][:, -1]
        fixed = interpolate_positions(t, tip, np.array([hit_time]))[0]
        ct, cp = clip_observations(t, tip, 0., 1.5)
        closest = nearest(ct, cp, target)
        errors = {}
        predicted_markers = d['predicted_cable'][:, d['predicted_marker_indices']]
        for label, end in [('strike', metadata['whip_end_s']), ('common', 1.5)]:
            mask = (t >= 0.) & (t <= end)
            errors[label] = dict(drone=rms(d['drone_error'][mask]), tip=rms(d['tip_error'][mask]),
                markers=rms(np.linalg.norm(predicted_markers[mask] - d['measured_cable'][mask, 1:], axis=-1)))
        velocity = local_velocity(t, tip)
        vt = interpolate_positions(t, velocity, np.array([hit_time]))[0]
        sensitivity = []
        for shift in (-.01, 0., .01):
            p = interpolate_positions(t + shift, tip, np.array([hit_time]))[0]
            sensitivity.append(dict(offset_change_s=shift, fixed_time_error_m=float(np.linalg.norm(p - target))))
        rows[name] = dict(role=ROLES[name], planned_time_s=hit_time,
            fixed_time_error_m=float(np.linalg.norm(fixed-target)), fixed_time_tip_position_m=fixed.tolist(),
            fixed_time_error_xyz_m=(fixed-target).tolist(), closest=closest, original_forecast_errors=errors,
            fixed_time_tip_speed_m_s=float(np.linalg.norm(vt)), timing_sensitivity=sensitivity, quality=quality[name])
    aggregate = {}
    metrics = dict(fixed_time_error_m=lambda r: r['fixed_time_error_m'],
        closest_error_m=lambda r: r['closest']['nearest']['distance_m'],
        common_drone_rmse_m=lambda r: r['original_forecast_errors']['common']['drone']['rmse_m'],
        common_tip_rmse_m=lambda r: r['original_forecast_errors']['common']['tip']['rmse_m'],
        common_markers_rmse_m=lambda r: r['original_forecast_errors']['common']['markers']['rmse_m'])
    for key, getter in metrics.items():
        v = np.asarray([getter(r) for r in rows.values()], float)
        aggregate[key] = dict(mean=float(np.mean(v)), sample_sd=float(np.std(v, ddof=1)), n=len(v), per_take=v.tolist())
    save(OUT / 'paper_metrics.json', dict(protocol=protocol['endpoints'], takes=rows, aggregate=aggregate,
        definition='3D Euclidean errors; RMSE=sqrt(mean squared Euclidean error); equal-take mean and sample SD.',
        limitations=['Estimated timing includes logging latency; +/-10 ms sensitivity reported.',
            'Closest approach is piecewise linear between adjacent valid 100 Hz observations.',
            'Original forecast uses planned initial state, not measured preflight state.',
            'No physical impact or binary success is inferred from distance.']))
    with (OUT / 'per_take.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['take', 'role', *metrics])
        writer.writeheader()
        for name, row in rows.items():
            writer.writerow(dict(take=name, role=row['role'], **{k: g(row) for k, g in metrics.items()}))
    review = read_json(OUT / 'review.template.json')
    for name, row in review['takes'].items():
        row.update(accepted=True, reviewed_by='Operator clean-take confirmation; Codex command/identity/timing audit',
            clock_reviewed=True, same_controller_and_hardware=True, no_intervention=True, physical_contact='none',
            free_motion_end_s=1.5, notes='Clock estimated from measured streams, not verified hardware sync. Fit capped at planned whip end; roles predeclared.')
    save(OUT / 'review.json', review)
    adaptation.verify_hashes(raw_hashes)
    print(finite_json(dict(aggregate=aggregate, quality=quality)), flush=True)


if __name__ == '__main__':
    main()
