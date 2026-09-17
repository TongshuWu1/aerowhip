"""Replay an unchanged saved whip with a longer braking tail; never activate it."""
from pathlib import Path
import argparse
import json
import shutil
import sys

import numpy as np
from numpy.polynomial import polynomial as poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experimental_data.io import atomic_json, sha256_file
from experimental_data.model_evaluation import model_identity
from planning.pva_job import freeze_model_assets
from deployment.pva_rehearsal import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--brake-seconds', type=float, required=True)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--export', type=Path, required=True)
    args = parser.parse_args()
    source, job, output, export = [p.resolve() for p in
                                  (args.source, args.job, args.output, args.export)]
    destinations = (job, output, export)
    if any(p.exists() for p in destinations):
        raise ValueError('Use new job, rehearsal and export directories')
    if len(set(destinations)) != 3 or any(
            a in b.parents for a in destinations for b in destinations if a != b):
        raise ValueError('Output directories must be distinct and non-nested')
    cfg = json.loads((source/'settings.json').read_text())
    previous = json.loads((source/'rehearsal.json').read_text())
    if not previous.get('recovery_prediction_complete'):
        raise ValueError('Source requires a complete saved recovery prediction')
    if not previous['recovery']['brake_end_s'] < args.brake_seconds <= cfg['recovery']['maximum_brake_s']:
        raise ValueError('Requested braking duration must be longer and within existing bounds')
    hashes = {str(p): sha256_file(p) for p in source.rglob('*') if p.is_file()}
    old_identity = model_identity(source/'model.json')[0]
    cfg['recovery']['minimum_brake_s'] = args.brake_seconds
    job.mkdir(parents=True)
    frozen = freeze_model_assets(json.loads((source/'model.json').read_text()),
                                 job, source_root=source, portable=True)
    atomic_json(job/'model.json', frozen)
    cfg['model_path'] = 'model.json'
    atomic_json(job/'settings.json', cfg)
    shutil.copy2(source/'plan.npz', job/'plan.npz')
    shutil.copy2(__file__, job/'export_braking_variant.py')
    atomic_json(job/'source_hashes.json', hashes)
    atomic_json(job/'status.json', dict(status='prepared', source=str(source)))
    summary = generate(job, output, progress=lambda *v: print(*v, flush=True))
    summary['execution'] = ('Controller-managed takeoff and 15 s settling at the saved start; '
                            'play the complete CSV at native timestamps, with 15 s settling '
                            'between repetitions; land after the final repetition. '
                            'Four repetitions are intended; the revised command is not flight-tested. '
                            'Offline artifact, no flight sender.')
    atomic_json(output/'rehearsal.json', summary)
    with np.load(source/'rehearsal.npz') as z:
        old = {k: z[k] for k in z.files}
    with np.load(output/'rehearsal.npz') as z:
        new = {k: z[k] for k in z.files}
    cutoff = round(previous['whip_end_s']*30)+1
    assert summary['whip_end_s'] == previous['whip_end_s']
    np.testing.assert_array_equal(new['commands'][:cutoff], old['commands'][:cutoff])
    prefix = np.count_nonzero(old['prediction_time_s'] <= previous['whip_end_s']+1e-10)
    prefix_difference = {}
    for key in ('origin_positions_m', 'cable_positions_m'):
        error = float(np.max(np.abs(new[key][:prefix]-old[key][:prefix])))
        prefix_difference[key] = error
        np.testing.assert_allclose(new[key][:prefix], old[key][:prefix], atol=1e-8, rtol=0)
    recovery = summary['recovery']
    brake = np.asarray(recovery['brake_coefficients_normalized'])
    ret = np.asarray(recovery['return_coefficients_normalized'])
    origin = np.asarray(cfg['launch']['origin_m'])
    for derivative in range(3):
        bd = poly.polyder(brake, m=derivative, axis=0)/recovery['brake_end_s']**derivative
        rd = poly.polyder(ret, m=derivative, axis=0)/recovery['return_s']**derivative
        np.testing.assert_allclose(poly.polyval(0., bd),
                                   new['commands'][cutoff-1, 3*derivative:3*derivative+3], atol=1e-9)
        np.testing.assert_allclose(poly.polyval(1., bd), poly.polyval(0., rd), atol=1e-9)
        np.testing.assert_allclose(poly.polyval(1., rd), origin if derivative == 0 else np.zeros(3), atol=1e-9)
    rows = np.loadtxt(output/'fullstate_30hz.csv', delimiter=',', skiprows=1)
    np.testing.assert_array_equal(rows[:, 1:], new['commands'])
    np.testing.assert_allclose(np.diff(rows[:, 0]), 1/30, atol=1e-12, rtol=0)
    np.testing.assert_allclose(rows[-1, 1:4], origin, atol=1e-12)
    np.testing.assert_allclose(rows[-1, 4:], 0., atol=1e-12)
    assert model_identity(output/'model.json')[0] == old_identity
    assert sha256_file(output/'plan.npz') == hashes[str(source/'plan.npz')]
    assert all(sha256_file(Path(p)) == value for p, value in hashes.items())
    verification = dict(source=str(source), model_identity=old_identity,
                        original_files_unchanged=True, whip_commands_exactly_unchanged=True,
                        prefix_prediction_max_difference_m=prefix_difference,
                        continuous_position_velocity_acceleration=True,
                        rows=len(rows), command_rate_hz=30,
                        original_braking_position_m=previous['recovery']['braking_position_m'],
                        new_braking_position_m=recovery['braking_position_m'],
                        full_replay_passed=True, physical_flight_tested=False)
    atomic_json(output/'verification.json', verification)
    export.mkdir(parents=True)
    shutil.copy2(output/'fullstate_30hz.csv', export/'fullstate_30hz.csv')
    atomic_json(export/'export.json', dict(summary, rehearsal=str(output), recovery_job=str(job)))
    atomic_json(export/'verification.json', verification)
    (export/'flight_take').mkdir()
    (export/'README.md').write_text(
        '# M0 with longer braking\n\n'
        f'Braking duration: {recovery["brake_end_s"]:.3f} s. '
        f'Complete CSV duration: {summary["total_duration_s"]:.3f} s, at 30 Hz.\n\n'
        'Execute the complete CSV, including return and final hold. The controller supplies '
        'the separate 15 s settling intervals and repeated playback. Four repetitions per '
        'battery were reported by the user for the previous CSV; this revised motion has not '
        'been flight-tested.\n\n'
        'Start/end tracked origin: (0, 0, 1.4) m; yaw zero. '
        f'Commanded braking stop: {recovery["braking_position_m"]} m.\n\n'
        'The M0 model and all command samples through the original whip handover are unchanged. '
        'Complete simulation and command-envelope checks passed. Room clearance is not modeled. '
        'Keep these recordings separate from the previous command version.\n\n'
        f'Rehearsal: `{output}`.\nCSV SHA256: `{summary["csv_sha256"]}`.\n', encoding='utf-8')
    atomic_json(job/'status.json', dict(status='completed', rehearsal=str(output), export=str(export)))
    print(json.dumps(dict(verification, total_duration_s=summary['total_duration_s'],
                         recovery=recovery), indent=2), flush=True)


if __name__ == '__main__':
    main()
