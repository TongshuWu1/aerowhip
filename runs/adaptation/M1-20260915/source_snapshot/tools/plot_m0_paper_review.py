"""Render paper-review artifacts from the immutable M0 measurements."""
from pathlib import Path
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experimental_data.adaptation_check import load_comparison


def main():
    out = ROOT / 'runs/data_review/M0-paper-20260913'
    batch = ROOT / 'data/flight_batches/M0_paper_slower_brake_20260913'
    report = json.loads((out / 'paper_metrics.json').read_text())
    endpoint = report['protocol']; hit = endpoint['planned_strike_time_s']
    target = np.asarray(endpoint['target_m'])
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    names = list(report['takes'])
    for i, name in enumerate(names):
        d = load_comparison(ROOT, batch, name)
        mask = (d['time'] >= 0) & (d['time'] <= 1.5)
        t = d['time'][mask]; tip = d['measured_cable'][mask, -1]
        axes[0, 0].plot(tip[:, 0], tip[:, 2], color=colors[i], label=name)
        axes[0, 1].plot(t, d['target_error'][mask] * 100, color=colors[i])
        axes[1, 0].plot(t, d['drone_error'][mask] * 100, color=colors[i])
        axes[1, 1].plot(t, d['tip_error'][mask] * 100, color=colors[i])
        if i == 0:
            pred = d['predicted_cable'][mask, -1]
            axes[0, 0].plot(pred[:, 0], pred[:, 2], 'k--', lw=1.3, label='Original M0 forecast')
            axes[0, 1].plot(t, np.linalg.norm(pred - target, axis=1) * 100, 'k--', lw=1.3)
    axes[0, 0].scatter(target[0], target[2], marker='+', s=95, color='k', zorder=10, label='Target')
    axes[0, 0].set(xlabel='x (m)', ylabel='z (m)', title='(a) Tip trajectory, x–z projection')
    axes[0, 0].set_aspect('equal', adjustable='datalim')
    axes[0, 0].legend(fontsize=7, loc='best')
    for ax, title in [(axes[0, 1], '(b) Measured 3D tip–target distance'),
                      (axes[1, 0], '(c) Quadrotor forecast error'),
                      (axes[1, 1], '(d) Tip forecast error')]:
        ax.set(xlabel='Time from CSV onset (s)', ylabel='Distance (cm)', title=title, xlim=(0, 1.5))
        ax.axvline(hit, ls=':', color='0.35', lw=1)
        ax.grid(alpha=.18)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(out / ('M0_frozen_forecast.' + ext), dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 3.3), layout='constrained')
    x = np.arange(len(names))
    fixed = [report['takes'][n]['fixed_time_error_m'] * 100 for n in names]
    closest = [report['takes'][n]['closest']['nearest']['distance_m'] * 100 for n in names]
    ax.bar(x-.18, fixed, .36, color='#0072B2', label=f'At planned strike time ({hit:.3f} s)')
    ax.bar(x+.18, closest, .36, color='#D55E00', label='Closest approach in 0–1.5 s')
    ax.set(xticks=x, xticklabels=names, ylabel='3D tip–target distance (cm)', ylim=(0, 43))
    ax.legend(frameon=False, fontsize=8)
    for xx, yy in zip(x-.18, fixed): ax.text(xx, yy+.6, f'{yy:.1f}', ha='center', fontsize=8)
    for xx, yy in zip(x+.18, closest): ax.text(xx, yy+.6, f'{yy:.1f}', ha='center', fontsize=8)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(out / ('M0_target_errors.' + ext), dpi=180)
    plt.close(fig)
    lines = ['# M0 physical-flight evaluation — 13 September 2026', '',
        '**Five clean executions of the same slower-braking CSV. No model fitting is used in these scores.**', '',
        'Target: (1.25, 0, 1.25) m. Commanded launch: (0, 0, 1.4) m. Raw global coordinates are retained.', '',
        f'The fixed strike time is **{hit:.9f} s** after CSV onset, taken from the original saved M0 forecast before scoring these takes. It is the planner’s predicted target-sphere entry time, not a retrospectively selected measured time. The physical target metric has no binary radius threshold.', '',
        '| Take | Error at planned strike (cm) | Closest distance (cm) | Closest time (s) | Drone forecast RMSE (cm) | Tip forecast RMSE (cm) |',
        '|---|---:|---:|---:|---:|---:|']
    for name, row in report['takes'].items():
        common = row['original_forecast_errors']['common']; near = row['closest']['nearest']
        lines.append(f"| {name} | {row['fixed_time_error_m']*100:.2f} | {near['distance_m']*100:.2f} | {near['time_s']:.4f} | {common['drone']['rmse_m']*100:.2f} | {common['tip']['rmse_m']*100:.2f} |")
    lines += ['', '## Aggregate results', '', '| Metric | Equal-take mean ± sample SD (cm), n=5 |', '|---|---:|']
    labels = {'fixed_time_error_m': 'Error at planned strike time', 'closest_error_m': 'Closest approach, 0–1.5 s',
        'common_drone_rmse_m': 'Quadrotor forecast RMSE, 0–1.5 s', 'common_tip_rmse_m': 'Tip forecast RMSE, 0–1.5 s',
        'common_markers_rmse_m': 'All 10 cable markers forecast RMSE, 0–1.5 s'}
    for key, label in labels.items():
        row = report['aggregate'][key]
        lines.append(f"| {label} | {row['mean']*100:.2f} ± {row['sample_sd']*100:.2f} |")
    lines += ['', 'RMSE is the square root of the mean squared **3D Euclidean** error, computed within each take and then averaged equally across takes. The SD describes variation across five takes; it is not a confidence interval. Marker RMSE pools the 10 observed marker sites within each take, excludes the constructed attachment site, and retains native gaps.', '',
        '## Interpretation', '',
        'Every take is short of the target in x at the planned strike time. Closest approach follows later, between 1.129 and 1.170 s. The smaller closest-distance error therefore does not imply accurate strike timing. Residual lateral and vertical error remains at closest approach. These observations describe geometry; they do not establish impact, impact energy, or a causal explanation of the model error.', '',
        'The original forecast assumes the planned launch state. Its errors include differences in actual initial state as well as model error. The later M0/M1 comparison must use the same causal measured initialization, schedules and masks for both models and must be reported separately.', '',
        '## Data and timing audit', '',
        '- All 122 dynamic CSV packets are present in each controller log; packet reception timing spread is 0.265–0.432 ms.',
        '- OptiTrack rigid body: cf_3. Cable: cable1:c1 through c10 in numeric order. Unlabeled markers are ignored.',
        '- Tip tracking is 100% complete over 0–1.5 s in all five takes. All-marker coverage is at least 99.93%. No overlength segments were found in this interval under the existing preparation rule.',
        '- Closest approach uses adjacent valid 100 Hz observations with piecewise-linear interpolation. No interpolation bridges missing samples or large timestamp gaps.',
        '- Clock offsets are estimated by matching the two measured quadrotor position streams without spatial fitting. Alignment residuals are 1.24–1.58 cm; estimates include logging latency. Half-record offset spread reaches 9.2 ms for take 003.',
        '- Shifting the estimated alignment by ±10 ms gives an equal-take mean fixed-time error range of 20.45–28.65 cm. This is a sensitivity check, not a statistical uncertainty interval. Closest approach is substantially less sensitive to a constant time shift because the encounter remains inside the fixed window.',
        '- Recordings 001, 002 and 004 end before the final 7.133 s hold finishes. This does not truncate the reported 0–1.5 s endpoints; no full-duration recovery score is claimed.', '',
        '## M1 preparation', '',
        'The original split is adaptation 001/002/004 and operational validation 003/005. Take 002 has four missing tip observations and invalid adjacent-marker geometry at approximately −0.75 to −0.72 s. It fails the unchanged one-second causal initialization requirement. Its M0 task measurements above remain valid. A separate fitting review records whether it is excluded or supplied as a corrected tracking export; validation takes must not be moved into fitting to replace it.', '',
        'The existing staged update uses current whip training data plus the retained preliminary training takes, with training-only stopping. M0 is preserved. M1 enables a cable residual absent in M0, so the comparison includes a model-capacity change. This update batch is not the final independent physical test.', '',
        '## Artifacts', '',
        '- [Per-take metrics](per_take.csv), [complete numerical report](paper_metrics.json), and [source checksums](source_hashes.json).',
        '- [Trajectory and forecast plots (PDF)](M0_frozen_forecast.pdf) / [editable SVG](M0_frozen_forecast.svg).',
        '- [Target-error chart (PDF)](M0_target_errors.pdf) / [editable SVG](M0_target_errors.svg).', '',
        'Generated on Windows using the project runtime. These are measured update-batch results, with the limitations above; no M1 result is implied.']
    # Compute the sensitivity aggregate from stored per-take values, never hand-copy it.
    means = [np.mean([r['timing_sensitivity'][i]['fixed_time_error_m'] for r in report['takes'].values()])*100 for i in (0, 2)]
    lines = [s.replace('20.45–28.65', f'{min(means):.2f}–{max(means):.2f}') for s in lines]
    (out / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(out / 'REPORT.md')


if __name__ == '__main__':
    main()
