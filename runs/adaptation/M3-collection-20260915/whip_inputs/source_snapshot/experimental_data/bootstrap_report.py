"""Summarize the frozen bootstrap fit without changing active configuration."""
from pathlib import Path
import platform
import numpy as np
import torch
from .bootstrap_bundle import read, verify, VERSION, NOMINAL
from .bootstrap_combined import load_bundle
from .io import atomic_json, sha256_file


def audit(job):
    job=Path(job).resolve();root=Path(__file__).resolve().parents[1];p=read(job/'protocol.json');result=verify(job)
    for name,row in p['provenance'].items():
        assert sha256_file(job/'inputs'/(name+'_masks.npz'))==row['mask_sha256']
    for name,digest in p['nominal_hashes'].items():assert sha256_file(job/'nominal'/(name+'.json'))==digest
    pose_files=0
    for path in (job/'pose_inputs'/VERSION).rglob('*'):
        if path.is_file():
            original=root/'data/nominal_drone_runs'/NOMINAL/'inputs'/VERSION/path.relative_to(job/'pose_inputs'/VERSION)
            assert sha256_file(path)==sha256_file(original);pose_files+=1
    sources={}
    for stage in ['cable','drone','combined']:
        hashes=read(job/(stage+'_source_hashes.json'))
        for name,digest in hashes.items():
            assert sha256_file(job/'source_snapshot'/stage/name)==digest
            assert sha256_file(root/name)==digest, name
        sources[stage]=len(hashes)
    cable,drone=load_bundle(job/'bundle')
    from simulator.drone_pose_residual import load_residual
    from simulator.cable.residual import FrozenMotionResidual
    load_residual(drone['residual']['checkpoint'],drone['residual']['sha256'],'cuda')
    net=FrozenMotionResidual(cable['motion_residual']['checkpoint'],cable['motion_residual']['sha256'])
    q=torch.zeros(2,12,3,dtype=torch.float64,device='cuda');v=torch.ones_like(q)
    correction=net(q,v)
    assert torch.isfinite(correction).all() and (correction*v).sum()<=0
    result.update(platform=platform.platform(),gpu=torch.cuda.get_device_name(),
        masks_verified=len(p['provenance']),nominal_files_verified=4,pose_input_files_verified=pose_files,
        source_files_verified_by_stage=sources,bundle_loaded=True,active_model_changed=False,ppo_started=False,
        flight_ready=False,prospective_validation=False)
    atomic_json(job/'verification.json',result);return result


def run(job):
    job=Path(job).resolve();root=Path(__file__).resolve().parents[1]
    d=read(job/'drone/results.json');c=read(job/'cable/results.json');combined=read(job/'combined/results.json')
    cable,drone=load_bundle(job/'bundle');verification=audit(job)
    names=read(job/'protocol.json')['drone_response_takes']
    lines=['# Corrected preliminary drone/cable bundle — 8 September 2026','',
        'All four components are now fitted and saved as a separate development candidate: nominal drone response, bounded drone NN, cable physics and dissipative cable NN. The nominal drone parameters are the previously corrected attitude fit, frozen during this new residual fit.',
        '',f'Job: `{job.relative_to(root).as_posix()}`. Portable component bundle: `bundle/manifest.json`. This is an offline execution predictor; the GUI/PPO still uses the preserved selected model. No PPO/SAC training or flight was started.',
        '', '## Whole observed maneuver assessment','',
        'Predictions begin from past observations approximately 0.105 s before CSV onset and continue without measurement resets. Actual native delayed command events drive predicted drone pose; the rotated rigid offset drives the cable boundary. Only recorded CSV-phase targets are scored. Future measured cable/pose is never a predictor input in the combined cases.',
        '', '| Assessment | Take | Drone origin RMS (cm) | Attachment RMS (cm) | All cable markers RMS (cm) | Tip RMS (cm) | Tip max (cm) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for mode in ['final','left_out']:
        for n in names:
            label='final_all_three' if mode=='final' else 'leave_out_'+n
            row=combined[label][n]['cases']['both_residuals'];origin=d[label]['results'][n]['residual']['csv_maneuver']['origin_rmse_m']
            vals=[origin,row['attachment_rmse_m'],row['marker_rmse_m'],row['tip_rmse_m'],row['tip_max_m']]
            lines.append('| '+('All-data fit' if mode=='final' else 'Take left out')+' | '+n+' | '+' | '.join(f'{v*100:.2f}' for v in vals)+' |')
    lines+=['','The all-data rows are training-data performance. Each left-out row uses a drone fit excluding that whip and a cable fit excluding that whip plus its grouped preliminary takes. These similar legacy takes have already informed model design; the folds are development checks, not independent flight or paper evidence.',
        '', '## What the residuals change','',
        '| Take | Nominal drone origin RMS (cm) | With drone NN (cm) | Combined tip without either NN (cm) | Combined tip with both (cm) | Both, straight-down initialization (cm) |',
        '|---|---:|---:|---:|---:|---:|']
    for n in names:
        row=d['final_all_three']['results'][n];cases=combined['final_all_three'][n]['cases']
        vals=[row['nominal']['csv_maneuver']['origin_rmse_m'],row['residual']['csv_maneuver']['origin_rmse_m'],
            cases['nominal_drone_physics_cable']['tip_rmse_m'],cases['both_residuals']['tip_rmse_m'],cases['both_residuals_hanging_initialization']['tip_rmse_m']]
        lines.append('| '+n+' | '+' | '.join(f'{v*100:.2f}' for v in vals)+' |')
    lines+=['','The straight-down case uses the same measured drone pose/velocity but no measured cable shape or cable velocity. It is a sensitivity check at the recorded launch state, not proof that every ten-second hover produces the assumed state.',
        '', '### Separating cable error from drone-boundary error','',
        '| Take | Measured attachment, cable physics tip RMS (cm) | Measured attachment, cable NN tip RMS (cm) | Predicted attachment, both NN tip RMS (cm) |',
        '|---|---:|---:|---:|']
    for n in names:
        cases=combined['final_all_three'][n]['cases']
        vals=[cases[k]['tip_rmse_m'] for k in ['measured_attachment_physics_cable','measured_attachment_residual_cable','both_residuals']]
        lines.append('| '+n+' | '+' | '.join(f'{v*100:.2f}' for v in vals)+' |')
    lines+=['','Measured-attachment replay is an intentional cable-only diagnostic using the observed boundary, separately labelled from the complete prediction. Per-marker errors, all four NN ablations, masks, initial states and trajectories are saved in `combined/`.',
        '', '### Cable fitting-window results (all-data candidate)', '',
        '| Take | Physics-only marker RMS (cm) | With NN marker RMS (cm) | Physics-only tip RMS (cm) | With NN tip RMS (cm) |',
        '|---|---:|---:|---:|---:|']
    for n in c['final']['training']:
        r=c['final'];vals=[r[k][n][metric] for metric in ['marker_rmse_m','tip_rmse_m'] for k in ['physics','residual']]
        lines.append('| '+n+' | '+' | '.join(f'{v*100:.2f}' for v in vals)+' |')
    lines+=['','These are selected 0.65 s fitting windows, not the uninterrupted-maneuver scores above. All eleven takes contribute; this does not mean every raw frame enters the optimizer. The NN is selected on the pooled equal-take objective and can worsen an individual take or a particular whip-tip metric.',
        '', '## Fitted model and data contract','',
        '- Drone nominal: delayed PD plus acceleration feedforward and frozen causal pre-hover compensation at the OptiTrack origin. Independent positive acceleration-direction scales drive the second-order attitude response. Effective loaded-drone parameters are not firmware gains or motor constants.',
        '- Drone NN: 15 inputs (position error, velocity error, desired acceleration, predicted velocity, frozen hover compensation), two 16-unit tanh layers, three acceleration corrections bounded to ±0.5 m/s² per axis. No time/take identity or future measurement inputs. It changes realized translation; it does not directly add an attitude command. Fit only the three whip CSV maneuvers, with corresponding nominal fold parameters frozen.',
        '- Cable: fixed 0.9525 m measured geometry, 12 nodes, measured .157 kg drone plus .018 kg cable assembly. Tracking-origin-to-attachment offset remains the rotated [.006655, -.012874, -.055] m; attachment-to-C1 is a distinct flexible .063 m span. Mass distribution remains the documented proportional estimate.',
        f'- New cable physics: EI = {cable["cable"]["EI_n_m2"]:.8g} N·m² and Cb = {cable["cable"]["Cb_n_m2_s"]:.8g} N·m²·s. Selected by an explicit bounded grid/refinement with geometry/mass fixed. These are preliminary effective parameters, not an identifiability claim.',
        '- Cable NN: two 32-unit tanh layers, relative node positions/velocities and root velocity as input; state-dependent nonnegative per-axis damping on free nodes, bounded below 2 s⁻¹. Correction is −gamma(q,v)·v; it cannot inject kinetic energy directly or correct arbitrary conservative/elastic forces. Fixed external drag is exactly zero; no 0.3 s⁻¹ initialization and no separately learned scalar drag.',
        '- Cable fitting admits all eight preliminary and three whip takes. Whip scope is 1.3 s pre-hover plus the observed CSV, excluding post-hold/landing targets. Missing pose/markers, impossible jumps/chords, possible contact and original invalid intervals are recorded masks; raw data are preserved. No take was dropped because its fitted error was high.',
        '- Cable physics/NN selection uses equal-take measured-boundary 0.65 s windows spanning motion intensities. NN BPTT uses 0.05 s windows, 24 updates, and training-only 0.65 s selection including the zero residual; a saved dimensionless output-bias grid is also selected only on training windows. This is a bounded bootstrap optimization budget, not a claim of optimizer convergence or long-horizon accuracy.',
        '- Drone NN uses 100 Adam updates and the full observed maneuver; selection includes the zero-NN baseline and uses regularized training error, not held-out results.',
        '- Coupling: desired FullState → effective loaded drone → predicted pose/rigid attachment → cable. No extra cable reaction is applied to this fitted loaded-drone response. The virtual force generator is a distinct model stage and still needs the planned 30 Hz integration.',
        '', '## Remaining limitations','',
        'The old controller executed only the 20 observed maneuver rows (~0.66–0.67 s), not the later reference tail or planned ~0.78 s hit. These scores therefore do not establish hitting accuracy. Takeoff and post-hold were outside the CSV and are retained as separate phases, not silently appended to the maneuver loss.',
        '', '| Take | Nominal first-0.5-s post-hold origin RMS (cm) | With drone NN (cm) |', '|---|---:|---:|']
    for n in names:
        r=d['final_all_three']['results'][n]
        vals=[r[k]['post_maneuver_fullstate_hold']['origin_rmse_m'] for k in ['nominal','residual']]
        lines.append('| '+n+' | '+' | '.join(f'{v*100:.2f}' for v in vals)+' |')
    lines+=['','Post-hold was not fitted. Its logged hold target was selected using the real end position, so this is also not validation of future preplanned recovery. Longer maneuvers, recovery, different cable loads/shapes and thrust feasibility remain unestablished. The effective drone model can absorb cable-loading effects specific to these whips; it is not a generally identified two-way aircraft/cable plant.',
        '', 'The next integration step is to connect this reviewed candidate to a newly trained 30 Hz force policy and matching 30 Hz FullState path, with feasibility checks and consistent rehearsal/export. Do not relabel or retime the old 20 Hz checkpoint. New compatible recordings should assess the frozen candidate before adaptation; then use new recordings for subsequent fits while retaining legacy provenance as an archive.',
        '', '## Numerical sensitivity','',
        'A separate seven-case cable comparison did not pass strict batch-versus-single equality: maximum accumulated marker coordinate difference was 1.20 mm. Therefore the primary complete assessments run each trajectory separately. The original batched physical/residual fitting remains recorded as such; it is not claimed bit-identical to single-case deployment. See `numerical_batch_review.json` and both probe archives.',
        '', 'On take 1 with both residuals, increasing cable substeps from 12 to 24 per 10 ms step changed maximum marker coordinates by 3.70 mm, but tip RMS changed from 4.098 cm to 4.090 cm. This limited probe supports the scale of the reported error, not exact numerical equivalence or a general convergence guarantee. The fit and primary assessment keep the original 12 substeps; no simulator change was hidden inside fitting.',
        '', '## Verification and evidence','',
        f'Actual platform: {verification["platform"]}, {verification["gpu"]}. All 133 protected original measurement/model/config/policy files verified unchanged; all input/mask/nominal/source snapshots checked. Both packaged residuals load with SHA-256 verification. No automatic activation, source deletion, data retirement or flight.',
        '', '95 distinct targeted tests passed. See `test_verification.json` for exact test commands/results; `verification.json`, `protocol.json`, per-fold parameter grids/gradient checks/history, and `combined/results.json` for the numerical record.',
        '', '![Continuous maneuver errors](../data/bootstrap_model_runs/'+job.name+'/combined_errors.png)','']
    report=root/'docs/BOOTSTRAP_COMPLETE_MODEL_20260908.md';report.write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex='col',constrained_layout=True)
    for col,n in enumerate(names):
        for label,title,style in [('final_all_three','All-data fit','-'),('leave_out_'+n,'Take left out','--')]:
            with np.load(job/'combined'/label/(n+'.npz')) as a:
                t=a['time_s']-combined[label][n]['csv_start_s'];mask=a['marker_score_mask'][:,-1]
                e=np.linalg.norm(a['both_residuals_markers_m'][:,-1]-a['measured_markers_m'][:,-1],axis=-1)*100
                r=np.linalg.norm(a['both_residuals_attachment_m']-a['measured_attachment_m'],axis=-1)*100
                axes[0,col].plot(t[mask],r[mask],style,label=title);axes[1,col].plot(t[mask],e[mask],style,label=title)
        axes[0,col].set_title(n);axes[1,col].set_xlabel('Seconds since observed CSV onset')
        for ax in axes[:,col]:ax.grid(alpha=.25);ax.set_ylim(bottom=0)
    axes[0,0].set_ylabel('Attachment error (cm)');axes[1,0].set_ylabel('Cable tip error (cm)');axes[0,0].legend()
    fig.suptitle('Continuous prediction with both residuals — legacy development data')
    fig.savefig(job/'combined_errors.png',dpi=180);plt.close(fig)
    print(report,flush=True)
