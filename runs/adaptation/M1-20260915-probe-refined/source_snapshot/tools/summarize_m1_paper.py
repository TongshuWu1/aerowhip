"""Write a measured-data report after the frozen M1 comparison completes."""
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experimental_data.io import atomic_json,sha256_file
from experimental_data.whip_adaptation import verify_hashes
from simulator.workflow import read_json


def main():
    out=ROOT/'runs/evaluation/M1-paper-20260913-common-1p5s'
    report=read_json(out/'report.json');job=Path(report['source_fit'])
    fit=read_json(job/'fit/result.json');verify_hashes(fit['candidate_hashes'])
    selection=read_json(job/'fit/selection_frozen.json')
    labels={'drone':'Quadrotor','combined_tip':'Complete command-to-tip',
        'combined_markers':'Complete cable markers','conditional_tip':'Tip, measured attachment',
        'conditional_markers':'Cable markers, measured attachment'}
    lines=['# M1 model fit and frozen evaluation', '',
        '**M1 is fitted and registered as a candidate. M0 is preserved.**', '',
        'Training: M0_001, M0_002, M0_004 plus the retained preliminary training takes. Operational validation: M0_003 and M0_005. All five new takes remain in the study; only invalid observations are masked. No historical M1/M2 whip data enter this fit.', '',
        '## Held-out prediction comparison', '',
        'Both models use the same recorded commands, causal measured history, 150 Hz prediction grid and observation masks over 0–1.5 s. These are reinitialized postflight predictions, separate from the original M0 preflight forecast. Values below are equal-take mean RMSE in centimeters, with n=2 validation takes.', '',
        '| Quantity | M0 (cm) | M1 (cm) | Reduction (%) |', '|---|---:|---:|---:|']
    for key,label in labels.items():
        row=report['groups']['validation'][key]
        lines.append(f"| {label} | {row['M0_mean_m']*100:.2f} | {row['M1_mean_m']*100:.2f} | {row['reduction_percent']:.1f} |")
    lines += ['', 'A negative reduction means the error increased. The measured-attachment diagnostic isolates cable prediction from quadrotor-prediction error; complete command-to-tip error evaluates the combined model.', '',
        '## Per-take complete tip prediction', '', '| Take | Role | M0 RMSE (cm) | M1 RMSE (cm) |', '|---|---|---:|---:|']
    for role in ('adaptation','validation'):
        row=report['groups'][role]['combined_tip']
        for n,a,b in zip(row['takes'],row['M0_per_take_m'],row['M1_per_take_m']):
            lines.append(f'| {n} | {role} | {a*100:.2f} | {b*100:.2f} |')
    lines += ['', '## Fitting and numerical checks', '',
        '- Existing staged nominal quadrotor fit, quadrotor residual, attitude refinement, cable physics and cable residual. Training-only stopping and selection; validation never supplies gradients or checkpoint selection.',
        '- Whip/preliminary family weights are 2/3 and 1/3. Takes have equal weight within a family. There are 72 quadrotor windows and 135 cable windows (3 whip and 132 preliminary); six preliminary cable windows fail the existing history/coverage checks and remain documented.',
        '- Take 002 has four occluded preflight tip frames and invalid adjacent-marker geometry. The user authorized using the remaining observations. Per-node causal regression masks these observations; the minimum history coverage is 96.04%. The endpoint is observed, and no missing positions are synthesized.',
        '- The initially selected physical update failed the existing gradient check. A bounded audit of the seven saved fitting steps and the parent parameters retained update 6, the lowest-training-loss saved step with a passing gradient check. The rejected fit is preserved. This numerical recovery is disclosed; it is not a validation-driven retry.',
        '- Completed nominal and residual quadrotor stages were copied with identical model-content identities, rather than refitted. The cable residual continued under the original objective and architecture. The user stopped training to limit turnaround; the best numerically valid saved checkpoint was finalized without additional optimizer updates. This is a disclosed user stop, not a reached plateau.',
        f"- Quadrotor residual: {selection['drone_residual']['updates']} updates, selected update {selection['drone_residual']['selected_update']}. Cable residual: {selection['cable_residual']['updates']} updates, selected update {selection['cable_residual']['selected_update']}.",
        '- Both selected residual checkpoints passed the existing full-rollout directional gradient test (two stable finite-difference scales, less than 2% disagreement). This is a numerical check in an informative parameter direction, not proof of every Jacobian entry.',
        '- 47 relevant tests passed; six legacy integration tests were skipped because their old adp0 fixture is unavailable. Actual fitting and evaluation used Windows / RTX 4080.', '',
        '## Scope of the result', '',
        'These two held-out takes are repetitions of the M0 command at one target. They test prediction of this motion, not generalization to other targets or quadrotors. M0 has no cable residual; M1 enables one, so the comparison includes a model-capacity change. Timing is estimated from measured streams and includes logging latency. No spatial normalization or target-based alignment is applied.', '',
        '**M1 has not yet been flown.** Improved replay prediction does not establish improved physical target accuracy. The user chose command correction toward the fixed original M0 tip reference, with soft quadrotor-path tracking. A corrected command and prospective recordings are needed to test that result.', '',
        '## Files', '',
        f'- Candidate: [{job.name}/candidate/model.json]({(job/"candidate/model.json").as_posix()}).',
        '- [Numerical report](report.json), [evidence hashes](evidence_hashes.json).',
        '- [Held-out error curves (PDF)](M0_M1_validation.pdf) / [editable SVG](M0_M1_validation.svg).',
        '- [LaTeX prediction table](M0_M1_prediction_table.tex).',
        '- Original M0 physical metrics: `runs/data_review/M0-paper-20260913/REPORT.md`.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    tex=[r'\begin{table}[t]',r'\centering',r'\caption{Postflight model prediction on two held-out M0 executions over $[0,1.5]$~s. Entries are equal-take mean three-dimensional RMSE in centimeters. Both models use the same causal initialization procedure, commands and observation masks.}',
        r'\label{tab:m0_m1_prediction}',r'\begin{tabular}{lrr}',r'\hline',r'Prediction & M0 & M1 \\',r'\hline']
    for key in ('drone','combined_tip','conditional_tip'):
        row=report['groups']['validation'][key]
        tex.append(labels[key]+f" & {row['M0_mean_m']*100:.2f} & {row['M1_mean_m']*100:.2f} "+r'\\')
    tex += [r'\hline',r'\end{tabular}',r'\end{table}']
    (out/'M0_M1_prediction_table.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(9,5.8),layout='constrained',sharex=True)
    for col,n in enumerate(report['groups']['validation']['drone']['takes']):
        for model,color in [('M0','#0072B2'),('M1','#D55E00')]:
            with np.load(out/model/(n+'.npz')) as z:
                t=z['time_s'];q=z['coupled_cable'][:,-1];truth=z['measured_sites'][:,-1]
                mask=(t>=0)&(t<1.5);tip=np.linalg.norm(q-truth,axis=1);tip[~z['mask'][:,-1]]=np.nan
                drone=np.linalg.norm(z['predicted_origin']-z['measured_origin'],axis=1)
                axes[0,col].plot(t[mask],drone[mask]*100,label=model,color=color)
                axes[1,col].plot(t[mask],tip[mask]*100,label=model,color=color)
        axes[0,col].set_title(n+' (held out)')
        axes[0,col].set_ylabel('Quadrotor error (cm)');axes[1,col].set_ylabel('Combined tip error (cm)')
        axes[1,col].set_xlabel('Time from CSV onset (s)')
        for ax in axes[:,col]:ax.grid(alpha=.18);ax.set_xlim(0,1.5);ax.legend(frameon=False)
    for ext in ('pdf','svg','png'):fig.savefig(out/('M0_M1_validation.'+ext),dpi=180)
    plt.close(fig)
    atomic_json(out/'presentation_hashes.json',{str(f):sha256_file(f) for f in out.glob('*') if f.is_file() and f.name!='presentation_hashes.json'})
    print(out/'REPORT.md')


if __name__=='__main__':main()
