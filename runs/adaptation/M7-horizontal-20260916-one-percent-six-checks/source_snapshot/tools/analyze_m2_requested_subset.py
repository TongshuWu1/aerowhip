"""Post-hoc sensitivity analysis; never replaces the five-take physical report."""
from pathlib import Path
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'runs/data_review/M2-paper-20260913'
SOURCE = REPORT / 'simple_performance.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protected = [SOURCE, REPORT / 'paper_metrics.json', REPORT / 'REPORT.md']
    before = {str(p): sha(p) for p in protected}
    original = json.loads(SOURCE.read_text(encoding='utf-8'))
    selected = ['M2_002', 'M2_004']
    excluded = ['M2_001', 'M2_003', 'M2_005']
    takes = original['results']['M2']['takes']
    assert set(selected + excluded) == set(takes)
    metrics = [
        ('fixed_time_error_m', 'Target error at original strike (cm)', 100),
        ('closest_error_m', 'Closest target distance (cm)', 100),
        ('tip_reference_rmse_m', 'Tip-reference RMSE (cm)', 100),
        ('quadrotor_reference_rmse_m', 'Quadrotor-reference RMSE (cm)', 100),
        ('fixed_time_tip_speed_m_s', 'Tip speed at original strike (m/s)', 1),
    ]
    subset = {}
    lines = [
        '# Exploratory M2 subset: takes 002 and 004', '',
        'This post-hoc subset was requested after reviewing the flight outcomes. '
        'Takes 001, 003 and 005 are omitted only from this sensitivity analysis. '
        'The operator suspects a hardware defect, but no independent take-specific '
        'fault evidence or outcome-independent exclusion criterion has been provided. '
        'All five takes remain in the primary report and all raw files are preserved.', '',
        '| Metric | M0, all 5 | M1, all 5 | M2, all 5 | M2, selected 2 |',
        '|---|---:|---:|---:|---:|',
    ]
    for key, label, scale in metrics:
        values = [takes[t][key] for t in selected]
        subset[key] = dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values),
                           n=len(values), per_take=values)
        columns = [original['results'][m]['aggregate'][key] for m in ('M0', 'M1', 'M2')]
        columns.append(subset[key])
        lines.append('| ' + label + ' | ' + ' | '.join(
            f"{scale*c['mean']:.2f} +/- {scale*c['sample_sd']:.2f}" for c in columns) + ' |')
    lines += ['', 'Values are equal-take means +/- sample standard deviation. '
              'Intervals, original reference, clock alignment and velocity estimation '
              'are unchanged from the primary analysis.', '',
              'The selected pair has lower mean target and trajectory errors than the '
              'five M1 flights. This describes these two executions; it does not '
              'establish improved typical performance or repeatability. Selection '
              'after observing outcomes can bias the comparison, and two repetitions '
              'do not characterize the full M2 variability.', '',
              'Take 002 retains the existing clock-alignment limitation: the two '
              'recording halves disagree by 9.46 ms. Its fixed-time target distance '
              'changes from 2.64 cm to 6.80--7.15 cm under +/-10 ms shifts. '
              'No timing shift was chosen to improve the subset results.', '',
              '[Primary five-take report](../REPORT.md).']
    out = REPORT / 'exploratory_subset_002_004'
    out.mkdir(exist_ok=True)
    payload = dict(analysis='post_hoc_exploratory_subset', source_sha256=before[str(SOURCE)],
                   retained_takes=selected, omitted_only_in_this_analysis=excluded,
                   independent_hardware_fault_evidence_provided=False,
                   primary_report_unchanged=True, aggregate=subset,
                   selected_take_metrics={t: takes[t] for t in selected},
                   source_intervals={k: original[k] for k in
                       ('reference_interval_s', 'closest_approach_interval_s', 'planned_strike_time_s')})
    (out / 'metrics.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    assert before == {str(p): sha(p) for p in protected}, 'Primary outputs changed'
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
