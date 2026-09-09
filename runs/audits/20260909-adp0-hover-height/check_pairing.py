"""Independently check all 25 file pairings without a spatial transform."""
from pathlib import Path
import sys
import csv
import json

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.optimize import minimize_scalar
from experimental_data.adaptation_check import load_comparison, flight_names, read_optitrack, read_controller, sha256

OUTPUT = Path(__file__).resolve().parent
BATCH = ROOT / 'rehearsal_csv_and_result_in_real_flight/20260908-195207-486249-seed655_best_validation/adp0'


def direct_read(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.reader(stream))
    indices = []
    for axis in 'XYZ':
        matches = [i for i in range(len(rows[3])) if rows[2][i] == 'Rigid Body'
                   and rows[3][i] == 'cf_7' and rows[5][i] == 'Position' and rows[6][i] == axis]
        assert len(matches) == 1
        indices.append(matches[0])
    values = np.array([[float(r[i]) if r[i].strip() else np.nan for i in [1] + indices]
                       for r in rows[7:] if r])
    return values, indices


def match(mt, measured, controller, axes=(0, 1, 2)):
    ct = controller['time_s']
    cp = np.column_stack([controller[k] for k in ('x', 'y', 'z')])
    good = np.isfinite(measured).all(1)
    mt = mt[good]; measured = measured[good]
    lower, upper = ct[0] - mt[0], ct[-1] - mt[-1]
    def cost(shift):
        delta = np.column_stack([np.interp(mt + shift, ct, cp[:, a]) - measured[:, a] for a in axes])
        return float(np.mean(np.sum(delta ** 2, axis=1)))
    grid = np.arange(lower, upper, .01)
    best = grid[np.argmin([cost(value) for value in grid])]
    fine = np.linspace(max(lower, best - .02), min(upper, best + .02), 161)
    best = fine[np.argmin([cost(value) for value in fine])]
    refined = minimize_scalar(cost, bounds=(max(lower, best - .00025), min(upper, best + .00025)), method='bounded')
    return dict(offset_s=float(refined.x), rms_m=float(np.sqrt(refined.fun)))


def main():
    names = flight_names(BATCH)
    measured = []; controllers = []; rows = []; input_hashes = {}
    for name in names:
        mp = BATCH / 'flight_take' / (name + '.csv')
        cp = BATCH / 'flight_take' / ('experiment_' + name + '.csv')
        input_hashes.update({str(mp): sha256(mp), str(cp): sha256(cp)})
        raw, columns = direct_read(mp)
        parsed = read_optitrack(mp)
        np.testing.assert_array_equal(raw[:, 0], parsed['time'])
        np.testing.assert_array_equal(raw[:, 1:], parsed['drone'])
        measured.append(raw)
        controllers.append(read_controller(cp))
        rows.append(dict(take=name, rigid_body='cf_7', position_columns_zero_based=columns,
                         direct_read_matches_production=True))
    matrix = np.zeros((len(names), len(names)))
    for i, m in enumerate(measured):
        for j, c in enumerate(controllers):
            result = match(m[:, 0], m[:, 1:], c)
            matrix[i, j] = result['rms_m']
            if i == j:
                comparison = load_comparison(ROOT, BATCH, names[i])
                rows[i]['independent_xyz_alignment'] = result
                rows[i]['production_xyz_alignment'] = comparison['alignment']
                rows[i]['independent_minus_production_offset_s'] = result['offset_s'] - comparison['alignment']['offset_s']
                rows[i]['xy_only_alignment'] = match(m[:, 0], m[:, 1:], c, (0, 1))
                rows[i]['z_only_alignment'] = match(m[:, 0], m[:, 1:], c, (2,))
                rows[i]['exact_distinct_dynamic_csv_packets'] = comparison['packet_count']
                rows[i]['command_onset_s'] = comparison['onset']
                rows[i]['packet_onset_spread_s'] = comparison['packet_jitter_s']
        rows[i]['lowest_error_controller_pair'] = names[int(np.argmin(matrix[i]))]
        rows[i]['correct_pair_is_best'] = int(np.argmin(matrix[i])) == i
        print(names[i], np.round(matrix[i] * 100, 3).tolist(), flush=True)
    assert all(sha256(p) == value for p, value in input_hashes.items())
    result = dict(rows=rows, row_optitrack_take=names, column_controller_take=names,
                  all_pairing_rms_cm=(matrix * 100).tolist(), input_hashes=input_hashes,
                  inputs_unchanged=True, spatial_shift_or_rotation_applied=False,
                  source_semantics='Both measured XYZ streams derive from OptiTrack; logger is not independent onboard estimator telemetry.')
    (OUTPUT / 'pairing_check.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
