"""Brief measured M0/M1/M2 comparison against the unchanged M0 motion reference."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experimental_data.io import atomic_json,sha256_file
from experimental_data.whip_adaptation import verify_hashes
from experimental_data.adaptation_check import interpolate_positions
from experimental_data.flight_performance import local_velocity
from simulator.workflow import read_json


def stats(values):
    values=np.asarray(values,float)
    if not np.isfinite(values).all():raise ValueError('Missing metric; report explicitly instead of dropping a take')
    return dict(mean=float(values.mean()),sample_sd=float(values.std(ddof=1)),n=len(values),per_take=values.tolist())


def main():
    out=ROOT/'runs/data_review/M2-paper-20260913'
    batch=ROOT/'data/flight_batches/M2_paper_selected_correction_20260913'
    review=read_json(batch/'operator_review.json')
    assert review['no_cable_contact'] and review['no_abort'] and review['no_manual_intervention']
    verify_hashes(read_json(batch/'original_raw_hashes.json'))
    verify_hashes(read_json(out/'source_hashes.json'))
    refdir=ROOT/'runs/reference_tracking/M0-paper-fixed-reference';meta=read_json(refdir/'reference.json')
    assert sha256_file(refdir/'reference.npz')==meta['reference_sha256']
    with np.load(refdir/'reference.npz') as f:ref={k:f[k].copy() for k in f.files}
    rt=ref['time_s'];end=float(rt[-1]);strike=meta['planned_strike_time_s']
    hashes={str(refdir/'reference.npz'):sha256_file(refdir/'reference.npz'),
            str(batch/'operator_review.json'):sha256_file(batch/'operator_review.json')}
    results={}
    for label in ('M0','M1','M2'):
        folder=ROOT/f'runs/data_review/{label}-paper-20260913';path=folder/'paper_metrics.json'
        paper=read_json(path);hashes[str(path)]=sha256_file(path);takes={}
        for name,row in paper['takes'].items():
            path=folder/(name+'.npz');hashes[str(path)]=sha256_file(path)
            with np.load(path) as f:
                t=f['time_s'];mask=(t>=0)&(t<=end)
                tip=f['measured_cable'][:,-1];quad=f['measured_origin']
                desired_tip=interpolate_positions(rt,ref['tip_position_m'],t[mask])
                desired_quad=interpolate_positions(rt,ref['quadrotor_position_m'],t[mask])
                assert np.isfinite(tip[mask]).all() and np.isfinite(quad[mask]).all()
                tip_rmse=float(np.sqrt(np.mean(np.sum((tip[mask]-desired_tip)**2,axis=-1))))
                quad_rmse=float(np.sqrt(np.mean(np.sum((quad[mask]-desired_quad)**2,axis=-1))))
                v=interpolate_positions(t,local_velocity(t,tip),np.array([strike]))[0]
                speed=float(np.linalg.norm(v));assert abs(speed-row['fixed_time_tip_speed_m_s'])<1e-10
                takes[name]=dict(fixed_time_error_m=row['fixed_time_error_m'],
                    closest_error_m=row['closest']['nearest']['distance_m'],
                    closest_time_s=row['closest']['nearest']['time_s'],
                    tip_reference_rmse_m=tip_rmse,quadrotor_reference_rmse_m=quad_rmse,
                    fixed_time_tip_speed_m_s=speed,fixed_time_forward_velocity_m_s=float(v[0]),
                    fixed_time_strike_angle_deg=float(np.degrees(np.arccos(np.clip(v[0]/speed,-1,1)))))
        keys=list(next(iter(takes.values())))
        results[label]=dict(takes=takes,aggregate={k:stats([t[k] for t in takes.values()]) for k in keys})
    # Reproduce the previously reported M0/M1 reference-tracking values exactly.
    previous=read_json(ROOT/'runs/data_review/M1-paper-20260913/simple_performance.json')
    for label in ('M0','M1'):
        assert abs(results[label]['aggregate']['tip_reference_rmse_m']['mean']-
            previous['results'][label]['fixed_reference_tip_rmse_mean_m'])<1e-10
    verify_hashes(hashes)
    report=dict(results=results,reference_interval_s=[0,end],planned_strike_time_s=strike,
        closest_approach_interval_s=[0,1.5],source_hashes=hashes,operator_review=review,
        definition='Actual measured trajectories; equal-take averages; raw world coordinates and measured-stream clock alignment. No new simulation, fitting or tuning.',
        limitations=['Five repetitions per generation; not a randomized paired final comparison.',
            'Clock offsets include logging latency. M2_002 half-record offsets differ by 9.46 ms; fixed-time errors are timing-sensitive.',
            'Minimum distance is interpolated only along adjacent valid tracking samples; no physical impact is inferred.',
            'Tip velocity is estimated from measured positions using the same local derivative as M0/M1.'])
    atomic_json(out/'simple_performance.json',report)
    metrics=[('fixed_time_error_m','Target error at original strike','cm',100),
             ('closest_error_m','Closest target distance, 0-1.5 s','cm',100),
             ('tip_reference_rmse_m','Original tip-reference RMSE','cm',100),
             ('quadrotor_reference_rmse_m','Original quadrotor-reference RMSE','cm',100),
             ('fixed_time_tip_speed_m_s','Tip speed at original strike','m/s',1)]
    lines=['# M2 physical performance check','',
        'The operator confirmed that all five M2 takes had no cable contact, abort or manual intervention. This does not independently establish unchanged vehicle condition. Each controller recording matches all 124 dynamic packets of the selected M2 CSV. Tip and marker coverage is 100% over the 0-1.5 s strike window. Missing observations outside that window remain preserved.','',
        '**M2 follows the original M0 tip trajectory more closely on average, but target accuracy and repeatability did not improve over M1.** These are measured flight results, not model forecasts.','',
        '| Metric | M0 | M1 | M2 |','|---|---:|---:|---:|']
    for key,label,unit,scale in metrics:
        values=[results[n]['aggregate'][key] for n in ('M0','M1','M2')]
        lines.append('| '+label+' ('+unit+') | '+' | '.join(f"{scale*v['mean']:.2f} +/- {scale*v['sample_sd']:.2f}" for v in values)+' |')
    lines+=['','Entries are equal-take means +/- sample standard deviation, n=5 per generation. Reference tracking uses [0, 1.133333] s; the original strike time is 1.117248 s.','',
        '| M2 take | Error at original strike (cm) | Closest distance (cm) | Tip-reference RMSE (cm) | Tip speed (m/s) |',
        '|---|---:|---:|---:|---:|']
    for name,row in results['M2']['takes'].items():
        lines.append(f"| {name} | {100*row['fixed_time_error_m']:.2f} | {100*row['closest_error_m']:.2f} | {100*row['tip_reference_rmse_m']:.2f} | {row['fixed_time_tip_speed_m_s']:.2f} |")
    lines+=['','M2 tip-reference RMSE decreased by about 17% relative to M1. Mean fixed-time target error increased by 1.91 cm; closest distance increased by 0.77 cm. Fixed-time error standard deviation increased from 4.04 to 7.52 cm. Average measured tip speed is essentially unchanged (5.85 to 5.82 m/s).','',
        'M2_001 remained about 19.8 cm from the target at its closest approach. M2_005 had a 20.6 cm fixed-time error, but approached to 7.9 cm approximately 34 ms later. This identifies a timing difference in the measured trajectory; it does not establish a controller or model failure mechanism. All five takes remain included.','',
        'Clock alignment uses measured quadrotor streams, with no target-based time shift or spatial fit. M2_002 has about 9.46 ms disagreement between offsets from the two recording halves. Its minimum distance is about 2.63 cm, while fixed-time distance is sensitive to alignment (6.80-7.15 cm under +/-10 ms shifts, versus 2.64 cm nominal). The per-take timing sensitivity is in paper_metrics.json.','',
        'No model fitting, model selection or command changes were performed. The M2 batch is evaluation-only; no M3 training split was assigned. Raw hashes and frozen forecast/source hashes were verified. This is a brief five-take comparison, not evidence of statistically established generalization.','',
        '- Raw retained batch: `data/flight_batches/M2_paper_selected_correction_20260913`.',
        '- [Complete measured comparison](simple_performance.json).',
        '- [Per-take errors, coverage and timing sensitivity](paper_metrics.json).']
    conditions=batch/'reported_conditions.json'
    if conditions.exists():
        condition=read_json(conditions)
        lines += ['', '## Reported experimental conditions', '',
            'After reviewing the results, the operator reported slight quadrotor damage during M2 and suspected a battery-related change. The affected takes, damage onset and battery measurements were not provided. These factors may confound the comparison, but their contribution has not been established.', '',
            'All five takes remain included. No numerical adjustment, outcome-based exclusion or replacement of measured values was made. The results support improved average tip-reference tracking; they do not support improved target accuracy over M1 or a causal claim about battery state or damage.', '',
            '[Operator conditions note](CONDITIONS_NOTE.md).']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print({n:{k:r['aggregate'][k]['mean'] for k,_,_,_ in metrics} for n,r in results.items()})


if __name__=='__main__':main()
