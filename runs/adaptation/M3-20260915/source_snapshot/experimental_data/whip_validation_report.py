"""Save tables and scientific plots for completed maneuver validation."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .historical_fit import read
from .io import atomic_json


def report(job):
    result=read(job/'validation/review.json');names=list(result['takes'])
    hanging=read(job/'validation_hanging/review.json')['takes']
    labels=list(result['takes'][names[0]]['metrics'])
    averages={label:{kind:float(np.mean([result['takes'][n]['metrics'][label][kind]['rmse_m'] for n in names]))
        for kind in ['attachment','cable','tip','maneuver_tip']} for label in labels}
    hanging_mean=float(np.mean([hanging[n]['metrics']['deployment_hanging_initial']['tip']['rmse_m'] for n in names]))
    atomic_json(job/'summary.json',dict(averages_m=averages,hanging_tip_rmse_m=hanging_mean,
        status='COMPLETED_PREDICTION_MISMATCH',independent_test=False,active_model_changed=False,ppo_started=False))
    fig=plt.figure(figsize=(16,13))
    for row,n in enumerate(names):
        d=np.load(job/'validation'/n/'trajectories.npz');h=np.load(job/'validation_hanging'/n/'trajectories.npz')
        ids=list(d['labels']);both=ids.index('both_residuals');phys=ids.index('physics_only');oracle=ids.index('measured_attachment_diagnostic')
        t=d['time_s'];valid=d['cable_valid']
        ax=fig.add_subplot(3,3,row*3+1,projection='3d')
        actual=d['measured_markers'][:,-1].copy();actual[~valid]=np.nan
        for positions,color,label in [(actual,'#222222','Measured tip'),(d['prediction'][both,:,-1],'#cc5d33','Predicted tip')]:
            ax.plot(*positions.T,color=color,label=label)
        for positions,color in [(d['measured_attachment'],'#222222'),(d['attachment_prediction'][both],'#cc5d33')]:
            ax.plot(*positions.T,color=color,linestyle='--',alpha=.7)
        target=np.array([1,0,1.4]);ax.scatter(*target,color='#aa0088',marker='x',s=40,label='Intended target')
        points=np.concatenate([actual,d['prediction'][both,:,-1],d['attachment_prediction'][both],target[None]])
        low=np.nanmin(points,axis=0);high=np.nanmax(points,axis=0);center=(low+high)/2;radius=max(high-low)*.55
        ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius),
            xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title=n+' — equal spatial scale')
        ax.set_box_aspect((1,1,1));ax.view_init(elev=22,azim=-65)
        if row==0:ax.legend(fontsize=7,loc='upper left')
        ax=fig.add_subplot(3,3,row*3+2)
        for idx,color,label in [(phys,'#888888','Nominal response'),(both,'#cc5d33','Response + NN')]:
            ax.plot(t,100*np.linalg.norm(d['attachment_prediction'][idx]-d['measured_attachment'],axis=1),color=color,label=label)
        ax.set(title='Attachment prediction error',xlabel='Time from maneuver onset (s)',ylabel='Error (cm)');ax.grid(alpha=.2)
        if row==0:ax.legend(fontsize=8)
        ax=fig.add_subplot(3,3,row*3+3)
        for pred,color,label,style in [(d['prediction'][phys],'#888888','Physics only','-'),(d['prediction'][both],'#cc5d33','Both residuals','-'),
            (d['prediction'][oracle],'#4477aa','Measured-attachment diagnostic',':'),(h['prediction'][0],'#228855','Straight initial cable','--')]:
            err=100*np.linalg.norm(pred[:,-1]-d['measured_markers'][:,-1],axis=1);err[~valid]=np.nan
            ax.plot(t,err,color=color,label=label,linestyle=style)
        ax.axvline(.67,color='#333333',linestyle=':',alpha=.5);ax.axvline(.78,color='#aa0088',linestyle=':',alpha=.6)
        ax.set(title='Cable-tip prediction error',xlabel='Time from maneuver onset (s)',ylabel='Error (cm)');ax.grid(alpha=.2)
        if row==0:ax.legend(fontsize=7)
    fig.suptitle('Held-out whip prediction — continuous 1.5 s, no state resets',fontsize=17)
    fig.tight_layout(rect=[0,.035,1,.965])
    fig.text(.02,.015,'Dashed 3D paths: attachment. Vertical lines: logged hold near 0.67 s; intended hit time 0.78 s. Gaps: masked observations. Development checks, not independent flight validation.',fontsize=9)
    fig.savefig(job/'validation_overview.png',dpi=160);plt.close(fig)
    lines=['# Combined held-out whip validation','',
        '**Outcome: the current combined model does not predict the complete whip accurately enough to treat it as a validated training model.**',
        '',f'Run: `{job.name}`. All three leave-one-whip-out predictions are complete. No active calibration, PPO, controller or logger was changed.',
        '', '## Protocol','',
        'Each whip is excluded from both component fits and cable geometry. Existing matching cable folds were reused; the drone response/NN were refitted on the other two whips with that fold’s geometry. These are held-out parameter fits, but still development checks: the recordings previously informed methodology and are not independent paper evidence.',
        '', 'Prediction begins at the first logged nonzero velocity/acceleration command and runs continuously for 1.5 s. The initial drone and cable state uses only past measurements. Thereafter the inputs are logged cmdFullState P/V/A, including the actual switch to hold near 0.67 s. No measured state is injected during the prediction. The separate measured-attachment diagnostic intentionally uses future measured attachment positions to isolate cable error; it is not an open-loop validation result.',
        '', 'All three recordings support this common horizon. Cable-quality masks exclude 9, 0 and 1 frames respectively without resetting the prediction or filling measurements. Commands and drone inputs are valid throughout the tested interval. A second run assumes a straight downward cable moving initially with the measured drone velocity, matching the planned drone-only initialization assumption.',
        '', '## Both residuals: per-take results','',
        '| Take | Attachment RMSE, 1.5 s | Tip RMSE, 1.5 s | Tip RMSE during 0–0.67 s | Tip position error at 0.78 s |',
        '|---|---:|---:|---:|---:|']
    for n in names:
        m=result['takes'][n]['metrics']['both_residuals']
        lines.append(f"| {n} | {100*m['attachment']['rmse_m']:.2f} cm | {100*m['tip']['rmse_m']:.2f} cm | {100*m['maneuver_tip']['rmse_m']:.2f} cm | {100*m['tip_error_at_planned_time_m']:.2f} cm |")
    lines+=['','## Component comparisons','', '| Model combination | Mean tip RMSE over 1.5 s |','|---|---:|']
    for label in ['physics_only','cable_nn_only','drone_nn_only','both_residuals','measured_attachment_diagnostic']:
        lines.append(f"| {label.replace('_',' ')} | {100*averages[label]['tip']:.2f} cm |")
    lines += [f'| Both residuals, assumed straight initial cable | {100*hanging_mean:.2f} cm |','',
        'The cable NN helps the average relative to the new physics-only chain, but substantial error remains. Adding the drone NN does not consistently improve this maneuver-wide prediction. Supplying the real attachment trajectory reduces error substantially, identifying drone-response error as a major contributor, while the remaining cable error shows that cable dynamics also need work. Similar measured-initial versus straight-initial results mean obtaining the initial cable state alone would not solve the mismatch in these trials.',
        '', 'The earlier few-centimeter averages used sampled windows with measured initialization at each start, many outside the complete maneuver. They understate the error of uninterrupted prediction from maneuver onset. This test does not tune models after examining these results.',
        '', '## Intended hit time and target','',
        'Target distances below use the historical intended target [1, 0, 1.4] m, not an independently surveyed physical object. The 0.78 s intended hit time occurs after the recorded switch to hold; these are not validations of an unexecuted 0.8 s command sequence. Clock alignment is approximate and includes logging latency.',
        '', '| Take | Predicted distance to target at 0.78 s | Observed distance to target at 0.78 s | Closest observed distance over 1.5 s |','|---|---:|---:|---:|']
    for n in names:
        r=result['takes'][n];p=r['metrics']['both_residuals']['target'];a=r['observed_target']
        lines.append(f"| {n} | {100*p['at_planned_time_m']:.2f} cm | {100*a['at_planned_time_m']:.2f} cm | {100*a['closest_m']:.2f} cm at {a['closest_time_s']:.2f} s |")
    lines+=['','## Next step','',
        'Verify command timing and fit drone transient response on continuous whip-centered rollouts, then address cable dynamics using measured attachment inputs and repeat the held-out checks. Do not train a new PPO assuming the present complete model is accurate. Preserve these negative results and all raw measurements; do not discard a take because it has large error.',
        '', 'Five targeted split/masking/command-history tests passed. Numerical validation ran on Windows/RTX 4080. Original source and snapshot checksums are preserved. No real flight was tested.',
        '', 'Artifacts: `validation/*/trajectories.npz`, `validation/*/metrics.json`, `validation_hanging/`, `summary.json`, and `validation_overview.png`.']
    (job/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);report(p.parse_args().job)
