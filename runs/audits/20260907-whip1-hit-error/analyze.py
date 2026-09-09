"""Read-only hit-error audit of the three authorized adaptation0 whip recordings.

Run with Python providing NumPy and Matplotlib (project .venv on this Windows host;
the bundled runtime lacks Matplotlib). Outputs are confined to this audit directory.
No physics fitting, policy execution, source mutation or flight commands.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ROUND = ROOT / 'data/adaptation_rounds/adaptation0'
PROCESSED = ROUND / 'processed/20260907-194802-922485'
PLAN = ROOT / 'runs/rehearsals/20260906-222405-804438/plan_001'
SOURCE_FILES = []


def record(path):
    path = Path(path)
    assert 'fig8vertical_002' not in str(path).lower()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    SOURCE_FILES.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': digest})
    return digest


def read_json(path):
    record(path)
    return json.loads(Path(path).read_text(encoding='utf-8'))


def raw_optitrack(path):
    record(path)
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.reader(f))
    meta = dict(zip(rows[0][::2], rows[0][1::2]))
    assert meta['Length Units'] == 'Meters' and meta['Coordinate Space'] == 'Global'
    def columns(name):
        out = []
        for axis in 'XYZ':
            ids = [i for i, n in enumerate(rows[3])
                   if n == name and rows[5][i] == 'Position' and rows[6][i] == axis]
            assert len(ids) == 1, (name, axis, ids)
            out += ids
        return out
    ids = [0, 1] + columns('cf_7')
    ids += [i for j in range(1, 11) for i in columns(f'cable1:c{j}')]
    a = np.array([[float(row[i]) if row[i].strip() else np.nan for i in ids]
                  for row in rows[7:] if row])
    return a[:, 1], a[:, 2:5], a[:, 5:].reshape(-1, 10, 3)


def interpolate(t, q, at):
    # Interpolate only adjacent complete tracking frames, never bridge gaps.
    i = int(np.searchsorted(t, at, side='right') - 1)
    if i == len(t)-1 and abs(t[i]-at) < 1e-9:
        return q[i].copy()
    assert 0 <= i < len(t)-1 and t[i+1]-t[i] <= .010001
    assert np.isfinite(q[i:i+2]).all(), 'Missing tracking data at requested instant'
    a = (at-t[i])/(t[i+1]-t[i])
    return q[i] + a*(q[i+1]-q[i])


def closest(t, q, target, lo, hi):
    best = None
    for i in range(len(t)-1):
        if t[i+1] < lo or t[i] > hi or t[i+1]-t[i] > .010001:
            continue
        if not np.isfinite(q[i:i+2]).all():
            continue
        step = q[i+1]-q[i]
        denom = np.dot(step, step)
        a = np.dot(target-q[i], step)/denom if denom > 0 else 0.
        lower = max(0., (lo-t[i])/(t[i+1]-t[i]))
        upper = min(1., (hi-t[i])/(t[i+1]-t[i]))
        a = np.clip(a, lower, upper)
        point = q[i]+a*step
        d = float(np.linalg.norm(point-target))
        if best is None or d < best['distance_m']:
            best = dict(time_s=float(t[i]+a*(t[i+1]-t[i])), position_m=point.tolist(),
                        error_xyz_m=(point-target).tolist(), distance_m=d,
                        segment_index=i, fraction=float(a))
    assert best is not None
    return best


manifest = read_json(ROUND/'round.json')
plan_meta = read_json(PLAN/'plan.json')
fullstate_meta = read_json(PLAN/'fullstate.json')
task = read_json(PLAN.parent/'rehearsal_task.json')
target = np.array(plan_meta['target_position_m'])
radius = task['success']['tip_target_distance_m']
assert np.array_equal(target, task['target_position_m'])
assert record(ROUND/'reference/fullstate_30hz.csv') == manifest['reference']['sha256']
assert record(PLAN/'fullstate_30hz.csv') == fullstate_meta['csv_sha256'] == manifest['reference']['sha256']
assert record(PLAN/'plan.npz') == fullstate_meta['source_plan_sha256']
record(PLAN/'fullstate_source.npz')
source = np.load(PLAN/'fullstate_source.npz', allow_pickle=False)
st, sq, sv = source['time_s'], source['positions_m'], source['velocities_m_s']
tip_source = sq[:, -1]
ref = np.genfromtxt(PLAN/'fullstate_30hz.csv', delimiter=',', names=True)
ref_values = np.column_stack([ref[n] for n in ref.dtype.names[1:10]])

# Match the simulator's segment-sphere entry and end-of-step velocity gates.
marker_q = sq[:, 2:]
crossings = np.zeros(marker_q.shape[:2], dtype=bool)
for i in range(1, len(st)):
    for j in range(10):
        crossings[i, j] = closest(st[i-1:i+1], marker_q[i-1:i+1, j], target,
                                   st[i-1], st[i])['distance_m'] <= radius
disqualified = np.cumsum(crossings[:, :-1].any(axis=1)) > 0
direction = np.array(task['desired_strike_direction_world'])
projected = sv[:, -1] @ direction
angle = np.degrees(np.arccos(np.clip(projected/np.maximum(np.linalg.norm(sv[:, -1],axis=1),1e-12), -1, 1)))
valid_hit = (crossings[:, -1] & ~disqualified & ~crossings[:, :-1].any(axis=1)
             & (projected >= task['success']['minimum_directed_tip_speed_m_s'])
             & (angle <= task['success']['maximum_tip_velocity_to_desired_direction_error_deg']))
hit_index = int(np.flatnonzero(valid_hit)[0])
hit_time = float(st[hit_index])
source_closest = closest(st, tip_source, target, 0., float(st[-1]))

results, plot_data = [], []
for trial in manifest['trials']:
    name = trial['trial_id']
    assert name in ('whip1_001', 'whip1_002', 'whip1_003')
    for info in trial['sources'].values():
        assert record(ROUND/info['path']) == info['sha256']
    mt, drone, cable = raw_optitrack(ROUND/trial['sources']['optitrack']['path'])
    controller = np.genfromtxt(ROUND/trial['sources']['controller']['path'], delimiter=',', names=True)
    quality = read_json(PROCESSED/name/'quality.json')
    assert record(PROCESSED/name/'dataset.npz') == quality['output_files']['dataset.npz']
    prepared = np.load(PROCESSED/name/'dataset.npz', allow_pickle=False)
    np.testing.assert_array_equal(mt, prepared['optitrack_time_s'])
    np.testing.assert_allclose(cable, prepared['cable_position_m'], rtol=0, atol=0, equal_nan=True)
    ct = controller['time_s']
    values = np.column_stack([controller['cmd_'+n] for n in ('x','y','z','vx','vy','vz','ax','ay','az')])
    active = ((controller['cmd_valid'] == 1) & (controller['cmd_age'] < .1)
              & (np.linalg.norm(np.nan_to_num(values[:, 3:]), axis=1) > 1e-7))
    ids = np.flatnonzero(active)
    assert len(ids) and np.all(np.diff(ids) == 1)
    first, stop = ids[0], ids[-1]+1
    selected = values[ids]
    unique = selected[np.r_[True, np.any(np.diff(selected, axis=0) != 0, axis=1)]]
    np.testing.assert_allclose(unique, ref_values[:len(unique)], rtol=0, atol=1e-12)
    start = float(ct[first]-controller['cmd_age'][first])
    end = float(ct[stop]-controller['cmd_age'][stop])
    offset = float(quality['alignment']['offset_s'])
    t = mt+offset-start
    tip = cable[:, -1]
    at = interpolate(t, tip, hit_time)
    expected = tip_source[hit_index]
    # Inspect marker direction during the last half-second before launch.
    pre = (t >= -.5) & (t < 0)
    root_distances = np.nanmedian(np.linalg.norm(cable[pre]-drone[pre,None], axis=2),axis=0)
    assert np.all(np.diff(root_distances) > 0), root_distances
    clipped = closest(t, tip, target, 0., float(st[-1]))
    extended = closest(t, tip, target, 0., float(st[-1])+.5)
    live = closest(t, tip, target, 0., end-start)
    # Independent dense interpolation checks in complete-data metric windows.
    dense_t = np.linspace(0., 1.3, 130001)
    complete_window = (t >= -.01) & (t <= 1.31)
    assert np.isfinite(tip[complete_window]).all()
    dense_q = np.column_stack([np.interp(dense_t, t, tip[:, j]) for j in range(3)])
    dense_min = np.linalg.norm(dense_q-target, axis=1).min()
    assert abs(dense_min-extended['distance_m']) < 5e-5
    np.testing.assert_allclose(at, [np.interp(hit_time,t,tip[:,j]) for j in range(3)], atol=1e-12)
    sensitivity = [float(np.linalg.norm(interpolate(t, tip, hit_time+dt)-target))
                   for dt in np.linspace(-.05,.05,101)]
    at_closest_time = interpolate(t, tip, source_closest['time_s'])
    event_band = (t >= hit_time-.05) & (t <= hit_time+.05)
    item = dict(trial=name, target_m=target.tolist(), tip_marker='cable1:c10',
        start_controller_s=start, start_optitrack_s=start-offset,
        offset_s=offset, alignment_rms_m=quality['alignment']['rms_m'],
        observed_reference_duration_s=end-start, recorded_reference_samples=len(unique),
        intended_reference_samples=len(ref), expected_hit_time_s=hit_time,
        measured_tip_at_expected_hit_m=at.tolist(), target_error_xyz_m=(at-target).tolist(),
        target_error_3d_m=float(np.linalg.norm(at-target)),
        planned_tip_at_expected_hit_m=expected.tolist(),
        measured_minus_planned_tip_xyz_m=(at-expected).tolist(),
        measured_minus_planned_tip_3d_m=float(np.linalg.norm(at-expected)),
        error_at_source_closest_time_m=float(np.linalg.norm(at_closest_time-target)),
        closest_during_intended_sequence=clipped, closest_during_recorded_commands=live,
        closest_through_half_second_after_cutoff=extended,
        closest_time_minus_planned_hit_s=extended['time_s']-hit_time,
        timing_sensitivity_plus_minus_50ms_error_range_m=[min(sensitivity), max(sensitivity)],
        timing_sensitivity_note='Illustrative time shifts, not a confidence interval or clock-error bound',
        missing_tip_samples_near_expected_hit=int((~np.isfinite(tip[event_band]).all(axis=1)).sum()),
        median_prelaunch_drone_to_markers_m=root_distances.tolist())
    results.append(item)
    plot_data.append((t, tip, end-start))

payload = dict(target_m=target.tolist(), hit_radius_m=radius,
    expected_hit_time_s=hit_time, source_tip_at_hit_m=tip_source[hit_index].tolist(),
    source_tip_target_error_at_hit_m=float(np.linalg.norm(tip_source[hit_index]-target)),
    source_closest=source_closest, planned_cutoff_s=float(st[-1]),
    method='Raw named c10 marker, common world coordinates; fixed measured-position clock offsets; adjacent-frame linear interpolation only',
    limitations=['Target is the saved intended coordinate, not an independently tracked physical target.',
                 'Timing includes unmeasured measurement/logging latency; no clock synchronization verified.',
                 'Closest pass after 0.67 s includes the subsequent position-hold response.',
                 'c10 endpoint mapping is supported by numeric naming and prelaunch geometry, not a new physical measurement.',
                 'No physical-contact labels or model/policy adaptation inferred.'],
    results=results, sources=SOURCE_FILES)
(HERE/'results.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n', encoding='utf-8')

colors = ['#2563eb','#d97706','#7c3aed']
fig, axes = plt.subplots(1, 2, figsize=(13, 6.0), layout='constrained')
for i, ((t, tip, duration), r) in enumerate(zip(plot_data, results)):
    use = (t >= 0) & (t <= 1.3)
    dist = np.linalg.norm(tip-target, axis=1)
    axes[0].plot(t[use], dist[use]*100, color=colors[i], label=r['trial'])
    axes[0].scatter([hit_time], [r['target_error_3d_m']*100], color=colors[i], s=45, zorder=5)
    before = (t >= 0) & (t <= .8)
    after = (t >= .8) & (t <= 1.3)
    axes[1].plot(tip[before,0],tip[before,2],color=colors[i],label=r['trial'])
    axes[1].plot(tip[after,0],tip[after,2],color=colors[i],ls=':',alpha=.85)
    point = r['measured_tip_at_expected_hit_m']
    axes[1].scatter([point[0]],[point[2]],color=colors[i],s=45,zorder=5)
axes[0].plot(st,np.linalg.norm(tip_source-target,axis=1)*100,color='#111827',ls='--',label='Planned simulation')
axes[0].axvline(hit_time,color='#111827',ls=':',label='Planned hit: 0.78 s')
axes[0].axvline(.668,color='#dc2626',alpha=.5,label='Recorded switch to hold: ~0.67 s')
axes[0].axhline(radius*100,color='#16a34a',ls=':',label='5 cm hit radius')
axes[0].set(xlabel='Time from first maneuver command (s)',ylabel='Tip distance to target (cm)',
            title='3D distance and planned hit instant',xlim=(0,1.3))
axes[0].legend(fontsize=8,loc='upper right')
axes[1].plot(tip_source[:,0],tip_source[:,2],color='#111827',ls='--',label='Planned simulation')
axes[1].scatter([target[0]],[target[2]],marker='*',s=180,color='#16a34a',label='Target',zorder=6)
axes[1].add_patch(Circle((target[0],target[2]),radius,fill=False,color='#16a34a',ls=':'))
axes[1].set(xlabel='World X (m)',ylabel='World Z (m)',
            title='Cable-tip path, X–Z projection\nDots: planned hit time; dotted paths: after 0.80 s')
axes[1].set_aspect('equal',adjustable='datalim')
axes[1].legend(fontsize=8,loc='upper left')
for ax in axes:
    ax.grid(alpha=.18)
fig.suptitle('adaptation0: cable-tip miss relative to target [1, 0, 1.4] m',fontsize=14)
fig.savefig(HERE/'hit_error.png',dpi=180)

lines = ['# Whip1 cable-tip hit error', '',
    f'Target: {target.tolist()} m in the saved world frame. Cable-tip marker: cable1:c10.', '',
    f'The saved source simulation first satisfies its 5 cm, speed and direction hit gates at {hit_time:.2f} s. '
    f'Its nearest target-center pass is {source_closest["time_s"]:.4f} s, '
    f'{source_closest["distance_m"]*100:.2f} cm from the center. The planned cutoff is {st[-1]:.2f} s.', '',
    '| Trial | Error at planned hit (cm) | X error (cm) | Y error (cm) | Z error (cm) | Closest in 0–0.80 s (cm) | Closest in 0–1.30 s (cm) | Time of latter (s) |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for r in results:
    e = np.array(r['target_error_xyz_m'])*100
    lines.append(f'| {r["trial"]} | {r["target_error_3d_m"]*100:.1f} | {e[0]:+.1f} | {e[1]:+.1f} | {e[2]:+.1f} | '
                 f'{r["closest_during_intended_sequence"]["distance_m"]*100:.1f} | '
                 f'{r["closest_through_half_second_after_cutoff"]["distance_m"]*100:.1f} | '
                 f'{r["closest_through_half_second_after_cutoff"]["time_s"]:.3f} |')
lines += ['', 'Errors are measured tip minus target. Negative X means short of the target along the desired +X strike direction; positive Z means above it.', '',
    '![Cable-tip distance and paths](hit_error.png)', '',
    '## Method and interpretation', '',
    '- Read all three original OptiTrack CSVs and their paired controller CSVs. Verified raw SHA-256 hashes and exact agreement between named cable coordinates and the prepared NPZ arrays.',
    '- Ignored unlabeled markers. Prelaunch drone-to-marker distances increase monotonically from c1 to c10 in every trial, consistent with c10 being the free tip.',
    '- The copied reference hash matches the saved simulation export. The first 20 logged p/v/a commands match its first 20 rows. The reference target is used directly, with no spatial fit to make the tip agree.',
    '- Time zero is the first command reception estimate (controller time minus command age). Existing constant offsets were determined by matching measured drone positions, not by optimizing tip error. All distances use three coordinates, including Y.',
    '- The hit-instant value uses linear interpolation between adjacent 100 Hz tracking frames. Closest distances minimize distance along adjacent frame segments. No interpolation over missing frames is permitted.',
    '- The 0–0.80 s window is the intended maneuver. The 0–1.30 s window adds a fixed 0.50 s to inspect a delayed pass; it includes the position-hold response and is not a completed open-loop strike.',
    '- Logs switch from the reference to hold after approximately 0.67 s, before the planned 0.78 s hit. These errors describe the trajectory actually recorded, including that change.',
    '- Clock synchronization and physical target location were not independently verified. At-time errors are approximate. The +/-50 ms sensitivity below is illustrative, not a statistical interval. Geometric miss distances are less dependent on clock alignment when the nearest pass lies inside the search window.',
    '- No missing c10 frames occur within 50 ms of the planned hit in any trial. Original raw data, configs, policies and review labels were left unchanged.', '',
    '## Timing sensitivity', '',
    '| Trial | Hit-instant error range for illustrative +/-50 ms shift (cm) |', '|---|---:|']
for r in results:
    low, high = r['timing_sensitivity_plus_minus_50ms_error_range_m']
    lines.append(f'| {r["trial"]} | {low*100:.1f}–{high*100:.1f} |')
lines += ['', 'Full numerical values and source hashes are in results.json. The analysis script is analyze.py. '
          'This is a Windows offline measurement audit; no physics fitting, retraining or flight execution was performed.', '']
(HERE/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps({k:payload[k] for k in ('expected_hit_time_s','source_closest','results')},indent=2))
