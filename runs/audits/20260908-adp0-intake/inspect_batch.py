"""Read-only, batch-specific intake; does not approve fit intervals or modify raw data."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experimental_data.adaptation_rounds import COMMAND_COLUMNS, align, read_controller, read_optitrack

OUT = Path(__file__).resolve().parent
BATCH = ROOT / 'rehearsal_csv_and_result_in_real_flight/20260908-195207-486249-seed655_best_validation/adp0'
REHEARSAL = ROOT / 'runs/rehearsals/20260908-203914-039721'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources = sorted(BATCH.rglob('*.csv'))
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}
    reference_path = BATCH / 'simulation_csv/fullstate_30hz.csv'
    assert digest(reference_path) == digest(REHEARSAL / 'fullstate_30hz.csv')
    reference = np.genfromtxt(reference_path, delimiter=',', names=True)
    rt = reference['time_s']
    rv = np.column_stack([reference[n] for n in reference.dtype.names[1:]])
    tree = cKDTree(rv)
    dynamic = np.linalg.norm(rv[:, 3:9], axis=1) > 1e-5
    results = []
    for number in range(1, 6):
        name = f'whip_adp_0_{number:03d}'
        if not (BATCH / 'flight_take' / f'{name}.csv').exists():
            results.append(dict(take=f'{number:03d}', status='OptiTrack file currently absent; awaiting replacement', fit_approved=False))
            continue
        m = read_optitrack(BATCH / 'flight_take' / f'{name}.csv')
        c = read_controller(BATCH / 'flight_take' / f'experiment_{name}.csv')
        sync = align(m, c)
        cv = np.column_stack([c[n] for n in COMMAND_COLUMNS])
        valid = np.isfinite(cv).all(axis=1) & np.isfinite(c['cmd_age']) & (c['cmd_valid'] > .5)
        distance, index = tree.query(cv[valid])
        matched = distance < 1e-10
        active = matched & dynamic[index]
        receipt = (c['time_s'] - c['cmd_age'])[valid][active]
        ids = index[active]
        # One estimate per distinct dynamic reference row, avoiding logger repeat weighting.
        estimates = np.array([np.median(receipt[ids == j]) - rt[j] for j in np.unique(ids)])
        onset = float(np.median(estimates))
        relative = m['time'] + sync['offset_s'] - onset
        whip = (relative >= 0) & (relative <= 1.0)
        markers_valid = np.isfinite(m['cable']).all(axis=2)
        quaternion_valid = np.isfinite(m['quaternion']).all(axis=1) & (np.linalg.norm(m['quaternion'], axis=1) > 1e-6)
        results.append(dict(
            take=f'{number:03d}', alignment=sync, controller_rows=len(c), optitrack_rows=len(relative),
            controller_span_s=[float(c['time_s'][0]), float(c['time_s'][-1])],
            csv_onset_controller_s=onset, onset_estimate_range_s=[float(estimates.min()), float(estimates.max())],
            valid_command_rows=int(valid.sum()), exact_reference_matches=int(matched.sum()),
            dynamic_reference_rows_seen=int(len(np.unique(ids))), dynamic_reference_rows_expected=int(dynamic.sum()),
            optitrack_csv_relative_span_s=[float(relative[0]), float(relative[-1])],
            complete_whip_coverage=bool(relative[0] <= 0 and relative[-1] >= 1),
            whip_samples=int(whip.sum()), marker_valid_fraction=markers_valid.mean(axis=0).tolist(),
            whip_marker_valid_fraction=markers_valid[whip].mean(axis=0).tolist() if whip.any() else None,
            quaternion_valid_fraction=float(quaternion_valid.mean()),
            contact_review='User confirmed no object/target contact or intervention; deliberately free-flight virtual-target trials', fit_approved=False,
        ))
    report = dict(
        scope='Policy-scoped adp0; separate from historical data/adaptation_rounds/adaptation0',
        policy='runs/ppo/20260908-195207-486249-seed655/checkpoints/best_validation.pt',
        policy_sha256=digest(ROOT / 'runs/ppo/20260908-195207-486249-seed655/checkpoints/best_validation.pt'),
        rehearsal=str(REHEARSAL.relative_to(ROOT)), source_hashes=hashes,
        reference_matches_saved_rehearsal=True, csv_duration_s=float(rt[-1]), whip_end_s=1.0,
        notes=[
            'User confirmed all five flights had no object/target contact or intervention. Target assessment is geometric, not observed physical impact.',
            'Evaluate tip error at the saved predicted strike time separately from closest approach during the whip and its timing; do not time-shift to minimize target error.',
            'User identifies simulation_csv as the flown command; dynamic PVA packets match exactly in all five controller logs.',
            'Repeated stationary holds cannot be uniquely identified by command values alone.',
            'CSV onset estimated from command reception age; not a hardware actuation timestamp.',
            'OptiTrack alignment uses measured drone XYZ only, no desired-trajectory fit or spatial transformation.',
            'Alignment includes logging latency; half-window offsets differ by up to about one 100 Hz sample.',
            'OptiTrack wall-clock metadata has a 1969 date and is not used for synchronization.',
            'Only cf_7 and cable1:c1 through c10 are parsed; unlabeled markers are ignored.',
            'Raw logs preserved; no missing cable samples interpolated, no fit/training or old legacy phase classification.',
            'Take 002 was replaced during inspection: the initial trim began about 2.60 s after CSV onset; the current file begins about 0.25 s before onset and covers the whip. Hashes describe the current files.',
        ], takes=results,
    )
    assert hashes == {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}
    (OUT / 'intake.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    lines = ['# Real-flight adp0 intake', '',
             'Five controller takes; supplied 30 Hz CSV matches saved rehearsal 20260908-203914-039721 byte-for-byte.',
             'The scored whip is CSV time 0–1 s. The complete CSV lasts 11.2 s and includes recovery and final hold.', '',
             '| Take | OptiTrack coverage relative to CSV (s) | Complete whip | Measurement-stream alignment RMS (cm) |',
             '|---|---|---|---|']
    for item in results:
        if 'status' in item:
            lines.append(f"| {item['take']} | OptiTrack file currently absent | Pending replacement | — |")
            continue
        start, end = item['optitrack_csv_relative_span_s']
        lines.append(f"| {item['take']} | {start:.3f} to {end:.3f} | {'Yes' if item['complete_whip_coverage'] else 'NO'} | {100*item['alignment']['rms_m']:.2f} |")
    lines += ['', 'Alignment RMS is agreement between two measured position streams, **not tracking or hitting error**.', '',
              '## Interpretation', ''] + ['- ' + note for note in report['notes']]
    lines += ['', 'Contact review complete: user confirms all five are free-flight trials. Fit selection remains a separate quality-review step; no fitting has been performed.',
              'Detailed per-marker whole-take and whip validity, timing offsets and all input hashes are in intake.json.',
              'Reproduce on the project Python environment with this directory’s inspect_batch.py.']
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
