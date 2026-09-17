"""Plot the verified frozen-model comparison without modifying the manuscript."""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Output path prefix')
    args = parser.parse_args()
    folder = args.evaluation.resolve()
    report = json.loads((folder / 'report.json').read_text())
    assert json.loads((folder / 'status.json').read_text())['status'] == 'completed'
    models = ['M0', 'M1', 'M2']
    means = np.array([report['aggregates'][m]['mean_all_marker_rmse_cm'] for m in models])
    sd = np.array([report['aggregates'][m]['sample_sd_all_marker_rmse_cm'] for m in models])
    profiles = np.array([report['aggregates'][m]['mean_marker_rmse_cm'] for m in models])
    prefix = args.output.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    colors = ['#596574', '#277DA8', '#D18400']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.titlesize': 10, 'axes.labelsize': 9,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.15), sharey=True,
                             gridspec_kw={'width_ratios': [1, 1.7]})
    fig.subplots_adjust(left=.095, right=.985, bottom=.22, top=.82, wspace=.28)
    fig.suptitle('Cable prediction error on the same ten flights', fontsize=11, y=.98)
    x = np.arange(3)
    axes[0].bar(x, means, width=.62, color=colors, alpha=.85,
                yerr=sd, capsize=3, error_kw={'elinewidth': 1, 'ecolor': '#303030'})
    axes[0].set_xticks(x, models)
    axes[0].set_title('(a) Whole cable', loc='left', pad=9)
    axes[0].set_ylabel('3D position RMSE (cm)')
    axes[0].set_xlabel('Prediction model')
    for idx, (mean, spread) in enumerate(zip(means, sd)):
        axes[0].text(idx, mean + spread + .7, f'{mean:.2f}', ha='center', va='bottom', fontsize=9)
    markers = np.arange(1, 11)
    for model, color, shape, values in zip(models, colors, ['s', 'o', '^'], profiles):
        axes[1].plot(markers, values, color=color, marker=shape, markersize=4,
                     linewidth=1.5, label=model)
    axes[1].set_title('(b) Individual cable markers', loc='left', pad=9)
    axes[1].set_xlabel('Marker index (attachment to tip)')
    axes[1].set_xticks(markers)
    axes[1].set_xlim(.7, 10.3)
    axes[1].legend(frameon=False, ncol=3, fontsize=8, loc='upper left',
                   handlelength=1.5, columnspacing=1.2)
    limit = max(float(np.max(means + sd)) + 3, float(profiles.max()) + 4)
    axes[0].set_ylim(0, np.ceil(limit / 5) * 5)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#DCE1E7', linewidth=.6)
        ax.tick_params(length=3, color='#5D6773')
        ax.spines['left'].set_color('#8A929A')
        ax.spines['bottom'].set_color('#8A929A')
    fig.text(.095, .035,
             '0–1.133 s; ten cable markers including the tip. Bars: mean ± sample SD across flights.',
             fontsize=8, color='#505860')
    fig.savefig(prefix.with_suffix('.pdf'), metadata={'Title': 'Whole-cable prediction error: M0 to M2'})
    fig.savefig(prefix.with_suffix('.png'), dpi=220)
    plt.close(fig)

    with prefix.with_suffix('.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'n_flights', 'mean_all_marker_rmse_cm', 'sample_sd_cm', 'mean_euclidean_error_cm', 'tip_rmse_cm'])
        for model in models:
            row = report['aggregates'][model]
            writer.writerow([model, row['n_flights'], row['mean_all_marker_rmse_cm'],
                             row['sample_sd_all_marker_rmse_cm'], row['mean_distance_cm'], row['mean_marker_rmse_cm'][-1]])
    with prefix.with_name(prefix.name + '_markers.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['marker', 'distance_from_attachment_cm', 'M0_rmse_cm', 'M1_rmse_cm', 'M2_rmse_cm'])
        for marker in range(10):
            writer.writerow([marker + 1, 100 * report['marker_distances_m'][marker], *profiles[:, marker]])
    reduction = 100 * (1 - means[2] / means[0])
    lines = [
        '# Whole-cable prediction error: M0 to M2', '',
        'All three frozen models receive the same recorded commands and measured initialization information for five M0 and five M1 flights. The interval and masks match the existing paper tip-prediction comparison.', '',
        'The score includes ten physical cable markers, ordered from the attachment toward the tip. Marker 10 is the tip. The attachment site reconstructed from vehicle pose is excluded. Positions are compared in the world frame without spatial alignment or additional time shifts.', '',
        'For each flight, whole-cable RMSE is the square root of the mean squared 3D position-error norm over all valid marker–time pairs. Every valid observation receives equal weight; the tip receives no extra weight. The reported overall value is the arithmetic mean of the ten per-flight RMSEs. Error bars show sample standard deviation across flights, not confidence intervals.', '',
        'The marker curves show the arithmetic mean of each marker’s per-flight 3D RMSE. Their last points reproduce the paper tip-prediction values. Missing observations remain excluded; every model uses exactly the same masks.', '',
        '| Model | Whole-cable RMSE, mean ± SD (cm) | Tip RMSE (cm) |',
        '|---|---:|---:|',
        *[f'| {model} | {means[i]:.4f} ± {sd[i]:.4f} | {profiles[i, -1]:.4f} |' for i, model in enumerate(models)],
        '', f'The mean whole-cable RMSE changes by {reduction:.2f}% from M0 to M2 (positive denotes reduction).', '',
        '**Evidence scope:** These ten flights were used during model development. This is a retrospective comparison, not an independent generalization test. No model fitting or selection was performed for this analysis.', '',
        '**Verification:** All 30 newly replayed tip trajectories must match the archived paper predictions within 1e-8 m. Model identities, prepared-data hashes, initialization, timestamps, and observation masks are checked. Source data and model files are unchanged.', '',
        '## Suggested caption', '',
        'Command-driven cable-prediction error for M0–M2 on the same ten M0/M1 recordings over [0, 1.133) s. (a) Whole-cable 3D RMSE across all valid observations of ten cable markers, including the tip; bars show the mean and sample standard deviation of per-flight RMSEs. (b) Mean per-flight RMSE for each marker, ordered from the attachment toward the tip. All models use identical measurement masks and initialization information. These recordings were used during model development.', '',
        f'Full evaluation and per-flight data: `{folder}`',
    ]
    prefix.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    provenance = {'evaluation': str(folder), 'report_sha256': hashlib.sha256((folder / 'report.json').read_bytes()).hexdigest(),
                  'plot_script': str(Path(__file__).resolve()), 'plot_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    prefix.with_name(prefix.name + '_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(json.dumps({'mean_all_marker_rmse_cm': dict(zip(models, means.tolist())),
                      'M0_to_M2_reduction_percent': reduction, 'figure_pdf': str(prefix.with_suffix('.pdf'))}, indent=2))


if __name__ == '__main__':
    main()
