"""Reproduce figures and tables from the September 5 development audit.

This reads completed diagnostic outputs; it does not fit or apply a model.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experimental_data.force_dataset import _normalized_rotations
from experimental_data.io import atomic_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/calibration_audits'
OUT = BASE / '20260905_fitting_report'
RUNS = ['20260905_comprehensive_fit', '20260905_comprehensive_followup',
        '20260905_comprehensive_coupled', '20260905_drag_convergence']
COLORS = {'training': '#0072B2', 'validation': '#D55E00'}


def read(path):
    return json.loads(path.read_text())


def save(fig, name):
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(OUT / f'{name}.{suffix}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def compare(groups, names, labels, name, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, max(3.4, .49 * len(names))))
    for ax, metric, title_part in zip(axes, ['marker_rmse_m', 'tip_rmse_m'], ['All markers', 'Cable tip']):
        for role, dy, marker in [('training', -.12, 'o'), ('validation', .12, 's')]:
            values = [1000 * groups[(v, role)][metric] for v in names]
            ax.plot(values, np.arange(len(names)) + dy, marker, color=COLORS[role],
                    label='Fit takes' if role == 'training' else 'Development validation', markersize=6)
        ax.set_yticks(range(len(names)), labels if ax is axes[0] else [''] * len(names))
        ax.invert_yaxis()
        ax.set_xlim(left=0)
        ax.set_xlabel('Trajectory RMSE [mm]')
        ax.set_title(title_part)
        ax.grid(axis='x', alpha=.2)
    axes[0].legend(loc='lower right', fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    save(fig, name)


def geometry_report(geometry):
    source = ROOT / 'data/workflow_jobs/20260905-061423-970034-fit'
    candidate = np.array(geometry['candidate_offset_body_m'])
    rows = []
    for name, g in geometry['per_take'].items():
        with np.load(ROOT / 'data/processed_takes' / name / 'take.npz') as p, \
                np.load(source / 'force_takes' / name / 'take.npz') as f:
            rotation, rotation_valid = _normalized_rotations(p['uav_orientation_xyzw'])
            valid = f['state_valid'] & rotation_valid
            c1 = p['cable_marker_positions_m'][:, 0]
            root = p['uav_position_m'] + np.einsum('tij,j->ti', rotation, candidate)
            distance = np.linalg.norm(c1 - root, axis=1)
            row = dict(take=name, role=g['role'],
                       original_first_span_violation_percent=g['chord_exceeds_arc_plus_2mm_percent'][0],
                       candidate_first_span_violation_percent=float(100 * (distance[valid] > .065).mean()))
            for length in [.0315, .01575, .007875]:
                pinned_tip = root - length * rotation[:, :, 2]
                chord = np.linalg.norm(c1 - pinned_tip, axis=1)
                row[f'clamp_{length * 1000:g}mm_violation_percent'] = float(
                    100 * (chord[valid] > .063 - length + .002).mean())
            rows.append(row)
    atomic_json(OUT / 'attachment_consistency.json', rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), gridspec_kw={'width_ratios': [1.2, 1]})
    names = list(geometry['per_take'])
    for i, name in enumerate(names):
        g = geometry['per_take'][name]
        med, lo, hi = [1000 * g[k][0] for k in ['chord_median_m', 'chord_p05_m', 'chord_p95_m']]
        axes[0].errorbar(med, i, xerr=[[med-lo], [hi-med]], fmt='o', color=COLORS[g['role']], capsize=3)
    axes[0].axvline(63, color='black', linestyle='--', label='Configured arc length: 63 mm')
    axes[0].set_yticks(range(len(names)), names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Attachment-to-c1 chord [mm]')
    axes[0].set_title('Original attachment transform')
    axes[0].legend(fontsize=8, loc='lower right')
    y = np.arange(len(rows))
    axes[1].plot([r['original_first_span_violation_percent'] for r in rows], y-.1,
                 'o', color='#0072B2', label='Original transform')
    axes[1].plot([r['candidate_first_span_violation_percent'] for r in rows], y+.1,
                 's', color='#009E73', label='Tentative transform')
    axes[1].set_yticks(y, [''] * len(y))
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Valid frames with chord > 65 mm [%]')
    axes[1].set_title('Tentative offset does not resolve consistency')
    axes[1].legend(fontsize=8, loc='lower right')
    for ax in axes:
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Attachment geometry needs calibration before material identification')
    fig.text(.5, -.015, 'Whiskers: 5th–95th frame percentiles, not confidence intervals. Chord length cannot exceed an inextensible arc.',
             ha='center', fontsize=8)
    fig.tight_layout()
    save(fig, 'attachment_geometry')
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    groups = {}
    protocols = {}
    for name in RUNS:
        result = read(BASE / name / 'result.json')
        assert not result['failed'], result['failed']
        assert read(BASE / name / 'status.json')['status'] == 'COMPLETED'
        protocols[name] = read(BASE / name / 'protocol.json')
        for g in result['groups']:
            groups[(g['variant'], g['role'])] = g
    flat = [{k: v for k, v in g.items() if k != 'per_take'} for g in groups.values()]
    with (OUT / 'all_2s_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    names = ['previous_active_parameters', 'reference', 'previous_neural_residual',
             'offset_xyz_training', 'drag_0p5', 'offset_pivot_drag0p3', 'offset_clamp_drag0p3']
    labels = ['Active EI/Cb, current geometry', 'Prior fitted EI/Cb, current geometry',
              'Prior fit + frozen neural residual', 'Tentative attachment offset',
              'External drag only (0.5 /s)', 'Offset + pivot + drag (0.3 /s)*',
              'Offset + 31.5 mm clamp + drag*']
    compare(groups, names, labels, 'model_comparison_2s',
            'Two-second predictions · 70 windows · *12 substeps; other rows use 3')
    conv = ['reference', 'substeps_6', 'substeps_12', 'substeps_24']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric, title in zip(axes, ['marker_rmse_m', 'tip_rmse_m'], ['All markers', 'Cable tip']):
        for role in COLORS:
            ax.plot([3, 6, 12, 24], [1000 * groups[(v, role)][metric] for v in conv],
                    'o-', color=COLORS[role], label='Fit takes' if role == 'training' else 'Development validation')
        ax.set_xticks([3, 6, 12, 24])
        ax.set_xlabel('Physics substeps per 10 ms')
        ax.set_ylabel('Trajectory RMSE [mm]')
        ax.set_title(title)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Finer integration increases measured-data error at the same physical parameters')
    fig.tight_layout()
    save(fig, 'solver_sensitivity')
    geometry = read(BASE / RUNS[0] / 'geometry.json')
    consistency = geometry_report(geometry)
    long = read(BASE / '20260905_comprehensive_5s/result.json')
    assert not long['failed']
    assert read(BASE / '20260905_comprehensive_5s/status.json')['status'] == 'COMPLETED'
    long_groups = {(g['variant'], g['role']): g for g in long['groups']}
    long_names = ['reference', 'substeps_12', 'previous_neural_residual',
                  'offset_pivot_drag0p3', 'offset_clamp_drag0p3']
    long_labels = ['Prior fitted physics (3 substeps)', 'Same physics (12 substeps)',
                   'Prior fit + frozen neural residual', 'Offset + pivot + drag (12 substeps)',
                   'Offset + 31.5 mm clamp + drag (12 substeps)']
    compare(long_groups, long_names, long_labels, 'model_comparison_5s',
            'Five-second predictions · 56 common windows · no parameter re-fitting')
    per_take = []
    for g in long['groups']:
        for take, metrics in g['per_take'].items():
            per_take.append(dict(variant=g['variant'], role=g['role'], take=take, **metrics))
    with (OUT / 'per_take_5s_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_take[0]))
        writer.writeheader()
        writer.writerows(per_take)
    take_names = list(long_groups[('reference', 'training')]['per_take']) + list(
        long_groups[('reference', 'validation')]['per_take'])
    fig, ax = plt.subplots(figsize=(10, 4.7))
    for i, (variant, label, color) in enumerate([
        ('reference', 'Prior fitted physics', '#0072B2'),
        ('previous_neural_residual', 'Prior physics + neural residual', '#CC79A7'),
        ('offset_pivot_drag0p3', 'Tentative offset + pivot + drag', '#009E73')]):
        values = [next(r['marker_rmse_m'] * 1000 for r in per_take
                       if r['take'] == take and r['variant'] == variant) for take in take_names]
        ax.plot(np.arange(len(take_names)) + (i - 1) * .15, values, 'osD'[i], color=color,
                label=label, markersize=7)
    ax.axvline(4.5, linestyle='--', color='.5', linewidth=1)
    ax.set_xticks(range(len(take_names)), take_names, rotation=25, ha='right')
    ax.set_ylabel('All-marker trajectory RMSE [mm]')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=.2)
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, 1.18), ncol=3)
    ax.set_title('Five-second predictions by take · five fit takes | two development validation takes',
                 fontsize=10, pad=45)
    fig.tight_layout()
    save(fig, 'per_take_5s')
    atomic_json(OUT / 'evidence_index.json', dict(two_second_runs=RUNS,
        five_second_run='20260905_comprehensive_5s', variant_count=len(groups)//2,
        protected_test_used=False, active_model_changed=False))
    text = ['# Physical fitting audit — 5 September 2026', '',
        'The existing recordings are sufficient to diagnose important model mismatches. '
        'They do not currently identify unique EI and damping values. More optimization iterations '
        'on the original two-parameter model are unlikely to resolve the observed error.', '',
        '## Scope and protocol', '',
        f'Completed {len(groups)//2} model/solver sensitivity variants on the same 70 two-second windows '
        '(53 fit, 17 development validation), plus five models on 56 common five-second windows '
        '(43 fit, 13 development validation). All predictions were finite; no variant failed. '
        'Additional completed checks include three initialization methods and a 50-pair EI/Cb surface. '
        'The full software suite passed 119 tests, with four existing Torch JIT deprecation warnings.', '',
        'Five fit takes: fig8_001, fig8_002, fig8vertical_001, osc_001, osc_002. '
        'Development validation: fig8_003, osc_003. The protected fig8vertical_002 recording was not opened. '
        'Validation has been examined repeatedly, so these are development results, not final paper test results.', '',
        'Every model receives the measured attachment trajectory during prediction. This isolates conditional '
        'cable dynamics; it does not validate an open-loop force-controlled drone. Initial velocity uses only '
        'the previous 11 measured frames (including the current frame); no future cable feedback is used, '
        'except the separately labelled oracle-c1 diagnostic. Boundary, geometry and drag variants keep '
        'the prior fitted EI=2.83994e-8 N m² and Cb=3.75231e-5 N m² s fixed unless explicitly named otherwise. '
        'A gamma of 0.3 /s was selected from 0.1, 0.3 and 0.5 using training loss within each combined model family. '
        'This is a small sensitivity sweep, not a completed joint calibration.', '',
        'RMSE is the Euclidean 3D position error over prediction frames, averaged as RMSE within each take '
        'and then equally across takes. Time zero is excluded from trajectory RMSE. Overlapping windows are '
        'not independent experiments. Five-second and two-second cohorts differ; compare models within each cohort.', '',
        '## Measured results', '',
        '| Model | 2 s fit markers [mm] | 2 s validation markers [mm] | 2 s validation tip [mm] |',
        '| --- | ---: | ---: | ---: |']
    for v, label in zip(names, labels):
        t, val = groups[(v, 'training')], groups[(v, 'validation')]
        text.append(f"| {label} | {t['marker_rmse_m']*1000:.2f} | {val['marker_rmse_m']*1000:.2f} | {val['tip_rmse_m']*1000:.2f} |")
    text += ['', '*Combined offset/drag models use 12 substeps. The same original physics at 12 substeps '
        f"has validation marker RMSE {groups[('substeps_12','validation')]['marker_rmse_m']*1000:.2f} mm. "
        'Consequently, the combined improvement is not explained by using a coarser solver.', '',
        '| Model | 5 s fit markers [mm] | 5 s validation markers [mm] | 5 s validation tip [mm] |',
        '| --- | ---: | ---: | ---: |']
    for v, label in zip(long_names, long_labels):
        t, val = long_groups[(v, 'training')], long_groups[(v, 'validation')]
        text.append(f"| {label} | {t['marker_rmse_m']*1000:.2f} | {val['marker_rmse_m']*1000:.2f} | {val['tip_rmse_m']*1000:.2f} |")
    text += ['', 'The offset-plus-drag pivot hypothesis improves five-second marker RMSE on every '
        'development take relative to the original fitted physics. Errors still differ substantially: '
        'its marker RMSE is 29.77 mm on fig8_003 and 84.73 mm on osc_003. The physical hypotheses '
        'are promising but do not constitute an accurate, validated baseline yet. Per-take values '
        'are exported in `per_take_5s_results.csv` and plotted in `per_take_5s`.', '',
        '## What explains the mismatch?', '',
        '- **Initialization is not the main fix.** Centered offline differentiation and causal differentiation '
        'gave 62.06 and 62.74 mm validation marker RMSE. Optimizing the initial state against 0.2 seconds '
        'of preceding DER motion worsened it to 70.93 mm. Keep the simpler causal estimator; the current '
        'imperfect dynamics should not pull the observed starting cable shape away from measurements.',
        '- **EI/Cb are weakly identifiable under the current model.** In the 50-pair causal profile, '
        '36 pairs were within 1% of the minimum training loss; the selected pair improved training loss '
        'by only 0.032%. Extending EI to 0.001 and 0.004 N m² at finer integration also failed to improve '
        'prediction. This is evidence against simply adding fitting epochs, not proof of zero bending stiffness.',
        '- **Geometry has unresolved inconsistencies.** The original attachment-to-c1 chord exceeds '
        'the configured 63 mm arc plus a 2 mm tolerance in 29.9% of valid fig8_003 frames. '
        'Curvature alone cannot explain a chord longer than the arc. Possible causes include the attachment '
        'transform, first-span length/compliance, or measurement errors. Quiet training segments suggest '
        'a conditional body offset of approximately [7.63, -13.83, -50.41] mm, but that estimate assumes '
        'a straight first span aligned with c1–c2. It is not metrology. It actually increases the overall '
        'violation fraction in several takes. Preserve the user’s approximately 55 mm top-plane measurement '
        'as a prior, rather than silently replacing it.',
        '- **A long hard clamp can compensate for errors.** Fixing the first 31.5 mm along the body axis '
        'lowers RMSE but leaves insufficient free span to reach c1 in many frames. At the tentative offset, '
        'this geometric inconsistency occurs in about 62% of fig8_002 and 51% of fig8_003 valid frames. '
        'Shortening the fixed segment changes the result substantially. A hard-clamp model would also '
        'require an attachment-attitude prediction in the force-driven deployment model.',
        '- **External dissipation is a useful missing term.** The diagnostic applies exponential decay '
        'exp(-gamma dt) to world-frame nodal velocity, in addition to internal bending damping. '
        'The 0.3 /s candidate is an effective dissipation rate, not an identified air-drag coefficient '
        'or proof of downwash. Its benefit survives using 12 substeps and should be tested in a constrained '
        'joint fit with geometry and Cb.',
        '- **Numerical resolution must be fixed before interpreting material parameters.** Increasing '
        'substeps from 3 to 6, 12 and 24 changes original-model validation marker RMSE from 62.74 to '
        '66.97, 69.35 and 70.63 mm. Better data agreement at coarse resolution can reflect numerical '
        'damping. Constraint iterations, projection passes, reasonable differentiation choices, marker '
        'mass perturbations and small interval-length changes do not resolve the bulk discrepancy.', '',
        '### Combined-model integration check', '',
        '| Model | 12-substep validation markers [mm] | 24-substep validation markers [mm] |',
        '| --- | ---: | ---: |']
    for name in ['offset_pivot_drag0p3', 'offset_clamp_drag0p3']:
        text.append(f"| {name} | {groups[(name,'validation')]['marker_rmse_m']*1000:.2f} | {groups[(name+'_substeps24','validation')]['marker_rmse_m']*1000:.2f} |")
    text += ['', 'These data-error comparisons measure sensitivity; they are not a formal solver-error bound.', '',
        '## Recommended fitting procedure using the existing recordings', '',
        '1. **Calibrate measurement geometry first.** Use quiet training segments to jointly assess lateral '
        'attachment offset and first-span consistency, with ruler measurements as priors. Profile uncertainty '
        'in vertical offset and first-span length instead of freely fitting both to a single optimum. '
        'Do not replace arc lengths with median chords. Check residuals versus body attitude and speed; '
        'a single transform should explain all training takes without requiring systematic stretch.',
        '2. **Keep causal state initialization.** Reconstruct measured markers, use an 11-frame past-only '
        'velocity estimate, and apply minimal length/velocity constraint projection. Save the initial '
        'projection error as a separate metric. Avoid optimizing the starting state to conceal model error.',
        '3. **Use a physically defensible attachment model and adequate integration.** Start with the pivot '
        'model and compare a short compliant attachment only if geometry supports it. Use 12 substeps '
        'for development and check selected candidates at 24; choose the final resolution using a '
        'predeclared prediction-change tolerance. Match that model in subsequent policy compilation.',
        '4. **Fit EI, Cb and a nonnegative effective drag rate together after geometry is constrained.** '
        'Use bounded log-parameter multistart optimization and differentiable multi-step rollouts, '
        'robust marker-position loss, equal take weighting, and balanced motion intensities. '
        'Begin with shorter horizons and finish at two seconds; use five-second predictions as a '
        'drift check. Profile parameter sensitivity and leave-one-training-take-out stability. '
        'Keep weakly identified quantities fixed to defensible priors or report ranges; do not claim '
        'unique material constants from a flat loss surface.',
        '5. **Train the neural motion residual last.** Freeze the selected physical model first, then '
        'fit a small regularized correction. Require improvement across takes, speeds and long '
        'rollouts over the improved physics alone. The previous residual helps much less than the '
        'new physical hypotheses, so training it harder now risks learning geometry and numerical errors.',
        '6. **Freeze the complete protocol before opening the protected take.** Publish per-take errors, '
        'tip and all-marker error versus prediction horizon, initialization displacement, model ablations, '
        'and parameter sensitivity. Use take-level uncertainty when defensible; do not count overlapping '
        'windows as independent trials. With one protected recording, explicitly limit generalization claims.', '',
        '## Later real-flight updates', '',
        'Log complete successful and failed trajectories with timestamps, commanded force, initial state '
        'and recovery transition. Offline logging does not violate open-loop execution. Fit the physical '
        'model on the force-sequence portion with clear treatment of controller transitions, retaining '
        'preliminary data to prevent drift. Failure labels alone cannot identify dynamics. Update the '
        'next open-loop force plan through differentiable rollout optimization initialized by the policy; '
        'retrain or distill the policy periodically after enough validated updates. This audit only tests '
        'cable predictions under measured root motion, so command-to-drone/attachment dynamics need '
        'separate validation before claiming end-to-end sim-to-real performance.', '',
        '## Reproduction and files', '',
        'Run `.venv/Scripts/python.exe -m experimental_data.fitting_audit_report` from the repository. '
        'The audit runners are `experimental_data/comprehensive_fit_audit.py`, '
        '`initialization_benchmark.py`, and `initialization_parameter_audit.py`. Each experiment directory '
        'records its protocol, input hashes, source snapshot and per-window/per-take results. '
        'All 51 two-second variants are included in `all_2s_results.csv`; geometry consistency is in '
        '`attachment_consistency.json`. Figures are supplied as PDF, SVG and 200 dpi PNG. '
        'No active physical baseline or policy was changed by these sensitivity experiments.', '',
        'Figures: `model_comparison_2s`, `model_comparison_5s`, `solver_sensitivity`, '
        '`attachment_geometry`, `per_take_5s`. All are development-study figures; they must not be labelled as '
        'independent final-test results.']
    report = '\n'.join(text) + '\n'
    (OUT / 'README.md').write_text(report, encoding='utf-8')
    (ROOT / 'docs/FITTING_AUDIT_20260905.md').write_text(report, encoding='utf-8')
    print(f'Saved audit report and figures to {OUT}')


if __name__ == '__main__':
    main()
