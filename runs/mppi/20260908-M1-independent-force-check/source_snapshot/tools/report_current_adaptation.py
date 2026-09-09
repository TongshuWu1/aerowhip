"""Component diagnostics, figures and auditable report for completed adp0."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experimental_data.current_adaptation_validation import *
import argparse


def components(job):
    job=Path(job);out=job/'component_check';out.mkdir(exist_ok=True)
    if (out/'results.json').exists():return read(out/'results.json')
    model=read(job/'source_candidate/model.json');m0=read(ROOT/'runs/rehearsals/20260908-203914-039721/model.json')
    names=sorted(p.name for p in (job/'inputs').iterdir());trials=[Trial(job,n,model) for n in names]
    old=ResearchExecutionModel.from_mapping(m0,device='cuda');records,_=cable_windows(trials,old.physics,[0.,1.2,2.],1.02)
    data=join_windows(records)
    q,_=cable_forward(old,data,[m0['cable']['EI_n_m2'],m0['cable']['Cb_n_m2_s']])
    np.savez_compressed(out/'M0_measured_attachment.npz',positions=q.cpu().numpy())
    index=list(old.cable.marker_node_indices[1:]);rows=[]
    for t in trials:
        i=next(i for i,r in enumerate(records) if r['name']==t.name and r['cutoff']==0.)
        with np.load(job/'cable_batched'/('leave_out_'+t.name)/'predictions.npz') as z:new=z['positions'][i]
        original=q[i].cpu().numpy();times=records[i]['time']
        native=t.data['time'];native=native[(native>=0)&(native<=1.)]
        truth=t.measured(native)[2][:,1:]
        before=interpolate_positions(times,original[:,index],native)
        after=interpolate_positions(times,new[:,index],native)
        rows.append(dict(take=t.name,baseline_marker_rmse_m=float(np.sqrt(np.mean(np.sum((before-truth)**2,axis=-1)))),
            adapted_marker_rmse_m=float(np.sqrt(np.mean(np.sum((after-truth)**2,axis=-1)))),
            baseline_tip_rmse_m=float(np.sqrt(np.mean(np.sum((before[:,-1]-truth[:,-1])**2,axis=-1)))),
            adapted_tip_rmse_m=float(np.sqrt(np.mean(np.sum((after[:,-1]-truth[:,-1])**2,axis=-1))))))
    result=dict(assessment='held-out cable initialized from past; supplied measured rotated attachment; not a complete-model forecast',rows=rows,
        means={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='take'})
    save(out/'results.json',result);return result


def report(job=JOB):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    job=Path(job);result=read(job/'validation/results.json');rows=result['heldout_rows'];means=result['equal_flight_mean']
    component=read(job/'component_check/results.json')
    out=job/'report';out.mkdir(exist_ok=True)
    old_late=[];new_late=[];old_angle=[];new_angle=[];ablation={k:[] for k in ['drone_only','cable_only','both']}
    starts=[]
    for row in rows:
        n=row['take'];old=read(job/'validation/M0_initialized'/n/'metrics.json');new=result['per_fold']['leave_out_'+n][n]['both']
        old_late.append(old['origin']['late_recovery_hold']['rmse_m']);new_late.append(new['origin']['late_recovery_hold']['rmse_m'])
        old_angle.append(old['whip_attitude_rmse_deg']);new_angle.append(new['whip_attitude_rmse_deg'])
        for k in ablation:ablation[k].append(result['per_fold']['leave_out_'+n][n][k]['tip']['whip']['rmse_m'])
        with np.load(job/'inputs'/n/'data.npz') as z:
            ids=z['pre_indices'];starts.append(dict(take=n,mean_measured_hover_position_m=z['position'][ids].mean(0),
                hover_command_position_m=z['hover_commands'][0,:3],initial_position_m=z['position'][ids[-1]]))
    assessment=dict(whip_improved=bool(result['recommend_for_new_planning']),
        baseline_late_drone_rmse_m=float(np.mean(old_late)),adapted_late_drone_rmse_m=float(np.mean(new_late)),
        baseline_whip_attitude_rmse_deg=float(np.mean(old_angle)),adapted_whip_attitude_rmse_deg=float(np.mean(new_angle)),
        ablation_mean_tip_rmse_m={k:float(np.mean(v)) for k,v in ablation.items()},
        default_model_changed=False,policy_retrained=False,
        reason='Whip-focused candidate retained separately; late hover regression and prospective validation remain unresolved',
        initial_hover_positions=starts,neural_training_budget='80 drone updates and 24 cable updates; not a convergence claim')
    save(job/'assessment.json',assessment)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    colors=['#64748b','#087f8c']
    for ax,key,title in zip(axes,['drone_whip_rmse_m','tip_whip_rmse_m','tip_strike_error_m'],
        ['Drone position RMS','Cable-tip position RMS','Tip prediction error at 0.94 s']):
        x=np.arange(5)
        ax.bar(x-.18,[r['baseline_'+key]*100 for r in rows],.36,color=colors[0],label='M0')
        ax.bar(x+.18,[r['adapted_'+key]*100 for r in rows],.36,color=colors[1],label='Adapted, flight held out')
        ax.set_xticks(x,['001','002','003','004','005']);ax.set_xlabel('Flight');ax.set_ylabel('Error [cm]');ax.set_title(title)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylim(0,14);axes[0].legend(fontsize=8)
    fig.suptitle('First adaptation · fixed flown commands · native 100 Hz measurements',fontsize=13)
    fig.savefig(out/'heldout_errors.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,layout='constrained')
    for row in rows:
        name=row['take'];folder=job/'validation'/('leave_out_'+name)/name/'both'
        with np.load(folder/'prediction.npz') as z:
            t=z['time'];e1=np.linalg.norm(z['position']-z['measured_position'],axis=1);e2=np.linalg.norm(z['cable'][:,-1]-z['measured_sites'][:,-1],axis=1)
        for ax,e in zip(axes,[e1,e2]):ax.plot(t,100*e,label=name[-3:],lw=1.2)
    for ax,title in zip(axes,['Drone origin error','Cable-tip error']):
        ax.axvspan(0,1,color='#f0b450',alpha=.17);ax.axvline(.94,color='black',ls=':',lw=1)
        ax.set_ylabel(title+' [cm]');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False);ax.set_xlim(0,11.2)
    axes[0].legend(title='Held-out flight',ncol=5,fontsize=8);axes[1].set_xlabel('Time from CSV start [s]')
    fig.suptitle('Complete execution validation · shaded interval is the whip')
    fig.savefig(out/'complete_execution_errors.png',dpi=180);plt.close(fig)
    def cm(value):return f'{100*value:.2f}'
    lines=['# First adaptation from the five current adp0 flights','',
        'The model was fitted to the five current whip_adp_0 recordings, with each flight held out in turn. These are repeated executions of one command sequence, so this is local development validation. A new adp1 flight remains the prospective test.','',
        'The physical flight errors did not change. The numbers below measure how accurately the updated simulator predicts the already recorded flights.','',
        '| Metric during the one-second whip | M0 | Adapted model |','|---|---:|---:|']
    for title,key in [('Drone origin RMS [cm]','drone_whip_rmse_m'),('All cable markers RMS [cm]','marker_whip_rmse_m'),('Cable tip RMS [cm]','tip_whip_rmse_m'),('Tip prediction error at saved strike [cm]','tip_strike_error_m')]:
        lines.append(f'| {title} | {cm(means["baseline_"+key])} | {cm(means["adapted_"+key])} |')
    lines+=['','RMS is the square root of the mean squared 3D Euclidean position error, evaluated at native 100 Hz observation times. The table is the arithmetic mean of the five per-flight values. The prediction starts from causal measured pre-flight state for both models; the original saved deployment forecast is retained separately.','',
        '![Held-out errors](heldout_errors.png)','','## What was fitted','',
        '- Drone: six PD/feedforward response gains, a discrete effective delay, three attitude-response parameters, and the existing bounded 16-wide neural acceleration residual. Attitude was re-evaluated after fitting the translation residual.',
        '- Cable: EI/Cb within a conservative factor-of-four grid, followed by 24 BPTT updates of the 32-wide damping network and the new bounded acceleration head. The smooth curvature-frame setting stays fixed at 2e-7. The free-node correction is bounded to ±0.5 m/s² per axis.',
        '- Geometry, marker identities, attachment offset, mass distribution, command values, observation alignment and fixed zero external drag were preserved. No target-based loss or time alignment was used.',
        '- Drone fitting used measured drone pose. Cable fitting used measured rotated attachment motion. Combined validation used predicted attachment motion after initialization, with no future measured drone/cable inputs.',
        '- Each fold excluded its held-out flight from nominal fitting, NN losses, priors evaluated on trajectories, and checkpoint selection. An additional model uses all five flights for subsequent planning.','',
        '## Data handling','',
        'The complete CSV is 11.2 s at 30 Hz; the scored whip is 0–1 s. Nominal/drone-residual fitting assigns 80% weight to the whip and 20% to early recovery through 3 s. Cable BPTT uses 0.12 s windows and training-only checkpoint selection on 1.02 s rollouts from onset and usable early-recovery starts. Missing cable-history windows were rejected with reasons; raw recordings were not trimmed, filled or deleted.',
        '','The last 21 native samples before CSV onset initialize the complete forecast. Other short cable windows use 11 past samples. Projection onto the fixed cable lengths is recorded; it adjusts only the simulator initial state. File starts are not equated. Measured drone streams establish the fixed time alignment; matched packet receipts establish command timing.',
        '','## Cable-only diagnostic','',
        f'With measured attachment motion supplied, held-out cable-tip RMS is {cm(component["means"]["baseline_tip_rmse_m"])} → {cm(component["means"]["adapted_tip_rmse_m"])} cm. This isolates cable prediction and is not a complete-system result.','',
        '## Recovery and limitations','',
        f'Early-recovery drone RMS: {cm(means["baseline_drone_early_recovery_rmse_m"])} → {cm(means["adapted_drone_early_recovery_rmse_m"])} cm. Complete observed cable-tip RMS: {cm(means["baseline_tip_complete_rmse_m"])} → {cm(means["adapted_tip_complete_rmse_m"])} cm.',
        '',f'Late recovery/hold drone RMS worsened: {cm(assessment["baseline_late_drone_rmse_m"])} → {cm(assessment["adapted_late_drone_rmse_m"])} cm. Whip attitude RMS: {assessment["baseline_whip_attitude_rmse_deg"]:.2f} → {assessment["adapted_whip_attitude_rmse_deg"]:.2f} degrees. M1 is therefore saved as a separate whip-focused candidate; the active default was not replaced.',
        '', 'The short-flight model freezes effective hover compensation. Late controller-memory effects and NN behavior near hover are possible explanations for the late regression, not identified causes. This fit deliberately prioritizes the whip; it does not establish improved accuracy for every phase.',
        '','![Complete execution](complete_execution_errors.png)','',
        'Several fitted parameters reach the conservative search limits. They are effective local response parameters, not a globally identified vehicle or cable material. The 20–40 ms fitted delay is also confounded with the roughly 10 ms measurement-stream timing uncertainty. The extended NN can inject energy; its bound does not guarantee stability on other commands.','',
        'The NN updates used a fixed conservative budget; training loss was still decreasing, so this is not a convergence claim. Measured initial hover also differs from the nominal launch position. The per-flight offsets and attitude/ablation results are in `../assessment.json`; a future plan must distinguish commanded launch coordinates from the measured initial state.', '',
        'The first nominal/residual fitting helper and native executor differ by redundant subdivisions at floating-point time boundaries; observed translation differences are recorded per flight (under 20 micrometres). The final attitude cache uses the exact native subdivision rule and passed its rotation-parity check.','',
        'No PPO was resumed, no new command CSV was exported, and no flight was performed. The retained flown PPO/rehearsal and original M0 remain immutable.','',
        '## Artifacts','',
        '- All-five candidate: `../models/all_five/model.json` with its own assets.',
        '- Published copy: `data/model_candidates/20260908-adp0-M1/model.json` in the project root. This candidate is not selected as the active default.',
        '- Fold predictions and metrics: `../validation/`.',
        '- Prepared data, masks and provenance: `../inputs/` and `../preparation.json`.',
        '- Fit settings/history and excluded windows: `../protocol.json`, `../execution_notes.json`, `../drone/`, `../drone_attitude_refined/`, `../cable_batched/`.',
        '- Component diagnostics: `../component_check/results.json`.','']
    (out/'ADAPTATION_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    return out


def publish(job=JOB):
    job=Path(job);assessment=read(job/'assessment.json')
    result=read(job/'validation/results.json');checks=read(job/'verification/full_whip_gradient.json')
    if not checks['passed']:raise ValueError('Fitted model numerical check did not pass')
    folder=ROOT/'data/model_candidates/20260908-adp0-M1'
    if folder.exists():raise FileExistsError('Published model candidate already exists')
    model=read(job/'models/all_five/model.json');folder.mkdir(parents=True)
    model=snapshot_assets(model,folder)
    model['adaptation'].update(model_selected=False,
        validation='Five leave-one-flight-out fits assess the procedure; this all-five candidate has only in-sample current-data evaluation. New adp1 required.',
        status='fitted_whip_candidate_not_default',report=str((job/'report/ADAPTATION_REPORT.md').resolve()),
        limitation='Late hover drone prediction regresses; fixed-budget local adaptation, no convergence or new-flight improvement claim')
    model['force_accounting']['aerodynamic_drag']='Learned nonnegative damping plus bounded acceleration correction on free cable nodes; zero separate fixed drag'
    save(folder/'model.json',model)
    save(folder/'manifest.json',dict(schema='adp0_M1_candidate_v1',source_job=str(job.resolve()),model_sha256=sha256_file(folder/'model.json'),
        fitted_data='five current whip_adp_0 flights',assessment=assessment,heldout_summary=result['equal_flight_mean'],
        selected=False,policy_training_started=False,new_flight_performed=False,
        assets={p.name:sha256_file(p) for p in (folder/'assets').iterdir()}))
    ResearchExecutionModel.from_mapping(read(folder/'model.json'),root=folder,device='cpu')
    return folder


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['components','report','publish']);parser.add_argument('--job',type=Path,default=JOB)
    args=parser.parse_args();torch.set_num_threads(1)
    print({'components':components,'report':report,'publish':publish}[args.stage](args.job),flush=True)
