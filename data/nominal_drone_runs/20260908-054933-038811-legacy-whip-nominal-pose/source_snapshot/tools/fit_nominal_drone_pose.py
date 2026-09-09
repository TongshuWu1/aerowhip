"""Prepare and fit a separate nominal drone candidate; never changes active models."""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experimental_data.nominal_pose_fit import (PreparedTrial,read_review,fit_translation,fit_attitude,
    score_pose,parameters,PRIOR,LOWER,UPPER,SCALES,PRIOR_WEIGHT,POSITION_SCALE_M,GAIN_NAMES,DELAYS,STARTS)
from experimental_data.io import sha256_file,atomic_json
from simulator.workflow import stamp

TAKES=('whip1_001','whip1_002','whip1_003')
VERSION='20260908-041031-260837'
SOURCES=('experimental_data/nominal_pose_fit.py','experimental_data/drone_pose_response_data.py',
    'experimental_data/drone_pose_initialization.py','simulator/drone_pose_response.py','simulator/geometry.py',
    'tools/fit_nominal_drone_pose.py','tests/calibration/test_nominal_pose_fit.py')


def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def load_trials(job):
    protocol=json.loads((job/'protocol.json').read_text(encoding='utf-8'))
    return [PreparedTrial(job/'inputs'/VERSION/name,review=protocol['reviews'][name]) for name in TAKES]


def prepare():
    job=ROOT/'data/nominal_drone_runs'/(stamp()+'-legacy-whip-nominal-pose');job.mkdir(parents=True)
    source=ROOT/'data/adaptation_rounds/adaptation0/processed'/VERSION
    destination=job/'inputs'/VERSION;destination.mkdir(parents=True)
    protected=json.loads((ROOT/'runs/audits/20260908-050940-048699-nominal-pose-implementation-review/protected-before.json').read_text(encoding='utf-8'))
    for path,digest in protected.items():assert sha256_file(path)==digest,path
    save(job/'protected-before.json',protected)
    copied={};reviews={};review_sources={}
    for name in ['processing.json','execution_profile.json','geometry_model.json']:
        shutil.copy2(source/name,destination/name);copied[str((destination/name).relative_to(job))]=sha256_file(source/name)
    for name in TAKES:
        folder=destination/name;folder.mkdir()
        for filename in ['dataset.npz','quality.json','execution_phases.json']:
            shutil.copy2(source/name/filename,folder/filename);copied[str((folder/filename).relative_to(job))]=sha256_file(source/name/filename)
        reviews[name],review_sources[name]=read_review(source.parent.parent/'reviews',VERSION,name)
    protocol=dict(schema='nominal_drone_pose_bootstrap_fit_v1',scope='Legacy whip-only nominal drone; no residual or cable fit',
        takes=list(TAKES),source_version=VERSION,input_hashes=copied,reviews=reviews,review_sources=review_sources,
        contact_review=dict(user_answer='no so the optitrack k data is trimmed by me, so the take start from its take off,in midair, the take end about the landing',
            interpretation='User answered no to contact/intervention during CSV maneuver; supplied files start during takeoff and end around landing.',
            outside_maneuver_contact_status='Not certified by this answer'),
        source_hashes={n:sha256_file(ROOT/n) for n in SOURCES},
        fitting_order='Position-only nominal gains/delay first; then one attitude timescale. Translation does not depend on attitude in this model.',
        translation_loss='Equal-take mean XYZ squared position error over valid CSV maneuver samples only, divided by 0.05m squared, plus fixed weak gain prior',
        gain_names=list(GAIN_NAMES),gain_lower=LOWER.tolist(),gain_upper=UPPER.tolist(),gain_prior=PRIOR.tolist(),
        gain_prior_scales=SCALES.tolist(),gain_prior_weight=PRIOR_WEIGHT,position_scale_m=POSITION_SCALE_M,
        gain_initializations=[x.tolist() for x in STARTS],delay_candidates_s=DELAYS.tolist(),
        attitude_bounds_s=[.02,.30],integration_maximum_step_s=.005,convergence_step_s=.0025,
        phase_scope='Entire observed CSV maneuver; pre-hover is initialization only; first 0.5s logged post-hold is diagnostic only',
        mask_scope='Saved valid measurements and manual interval plus jump/phase-boundary masks; commands/time remain complete',
        manual_exclusions=[],alignment='Per-take causal pre-hover effective alignment; not measured firmware mounting',
        initialization='Last measured pose before onset minus 0.1s; preceding 1s hover; compensation recomputed for each gain candidate',
        calibration='Frozen archived active geometry; no COM/mass/geometry fit or extra cable reaction',
        assessment='Three leave-one-whip-out development checks; final candidate on all three. Not independent paper evidence.',
        device='CPU NumPy analytic translational sensitivities / SciPy optimizer; full pose and attitude evaluation on CUDA',
        model_status='Candidate only; complete M0/PPO/export integration not finished')
    save(job/'protocol.json',protocol)
    for name in SOURCES:
        path=job/'source_snapshot'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,path)
    trials=load_trials(job);mask_report={}
    for trial in trials:
        folder=job/'masks'/trial.name;folder.mkdir(parents=True)
        np.savez_compressed(folder/'masks.npz',time_controller_s=trial.time,**trial.masks)
        mask_report[trial.name]=dict(initial_time_s=float(trial.time[0]),hover_start_s=float(trial.hover_time[0]),
            csv_start_s=trial.context['csv_onset_s'],csv_end_s=trial.context['csv_end_s'],
            csv_recorded_frames=int((trial.truth['execution_phase']=='csv_maneuver').sum()),
            counts={k:int(v.sum()) for k,v in trial.masks.items()},sha256=sha256_file(folder/'masks.npz'))
    save(job/'mask_report.json',mask_report)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    fig=Figure(figsize=(11,8));axes=fig.subplots(3,1)
    for ax,trial in zip(axes,trials):
        t=trial.time-trial.context['csv_onset_s'];ax.plot(t,trial.truth['position_origin_m'][:,2],label='Measured tracked-origin Z')
        m=trial.masks['fit_position'];ax.scatter(t[m],trial.truth['position_origin_m'][m,2],s=8,label='Maneuver fit samples')
        ax.axvspan(0,trial.context['csv_end_s']-trial.context['csv_onset_s'],alpha=.12,color='orange')
        ax.set_title(trial.name);ax.set_xlabel('Controller time from observed CSV onset (s)');ax.set_ylabel('Z (m)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Frozen fit masks: maneuver only; subsequent hold is diagnostic');fig.tight_layout();fig.savefig(job/'fit_masks.png',dpi=130)
    atomic_json(job/'status.json',dict(status='PREPARED',active_model_changed=False,training_started=False))
    print(job,flush=True)
    print(json.dumps(mask_report,indent=2),flush=True)
    return job


def run(job):
    if not torch.cuda.is_available():raise RuntimeError('Full pose assessment explicitly requires local CUDA')
    protocol=json.loads((job/'protocol.json').read_text(encoding='utf-8'))
    if tuple(protocol['takes'])!=TAKES:raise ValueError('This protocol fits only the three legacy whips')
    for name,digest in protocol['input_hashes'].items():assert sha256_file(job/name)==digest,name
    for name,digest in protocol['source_hashes'].items():assert sha256_file(ROOT/name)==digest,name
    started=time.perf_counter();trials=load_trials(job)
    def progress(*args,**kwargs):
        print(*args,**kwargs)
        atomic_json(job/'status.json',dict(status='FITTING_NOMINAL_ONLY',stage=' '.join(map(str,args)),elapsed_s=time.perf_counter()-started,
            active_model_changed=False,ppo_started=False))
    summaries={};saved={}
    for label,training in [('final_all_three',trials)]+[('leave_out_'+t.name,[x for x in trials if x.name!=t.name]) for t in trials]:
        folder=job/label;folder.mkdir()
        progress(label,'translation fit',flush=True)
        translation=fit_translation(training,progress=progress)
        save(folder/'translation_fit.json',translation)
        gains=np.array(translation['best']['gains']);delay=translation['best']['delay_s']
        progress(label,'attitude fit',flush=True)
        attitude=fit_attitude(training,gains,delay,progress=progress)
        save(folder/'attitude_fit.json',attitude);tau=attitude['tau_s']
        model=dict(schema='nominal_loaded_drone_pose_candidate_v1',parameters=asdict(parameters(gains,tau,delay)),
            training_takes=[t.name for t in training],alignment='prehover_effective_alignment',
            has_drone_residual=False,selected=False,reference_point='cf_7 tracking origin',
            geometry=trials[0].offset,fit_protocol_sha256=sha256_file(job/'protocol.json'))
        save(folder/'nominal_model.json',model)
        scores={};convergence={};parity={}
        for trial in trials:
            result=trial.pose(gains,tau,delay);arrays,metrics=score_pose(trial,result);scores[trial.name]=metrics
            p,v,_=trial.linear(gains,delay)
            dp=float(np.max(abs(p-arrays['position_origin_m'])));dv=float(np.max(abs(v-arrays['velocity_origin_m_s'])))
            if max(dp,dv)>1e-10:raise AssertionError('Fitting evaluator differs from full GPU pose engine')
            parity[trial.name]=dict(position_max_difference_m=dp,velocity_max_difference_m_s=dv)
            fine=trial.pose(gains,tau,delay,step=.0025)
            from scipy.spatial.transform import Rotation
            R=fine['rotation_tracking_to_world'][0].cpu().numpy()
            angles=Rotation.from_matrix(arrays['rotation_tracking_to_world'].transpose(0,2,1)@R).magnitude()
            convergence[trial.name]=dict(position_max_difference_m=float(np.max(np.linalg.norm(arrays['position_origin_m']-fine['position_origin_m'][0].cpu().numpy(),axis=1))),
                attachment_max_difference_m=float(np.max(np.linalg.norm(arrays['position_attachment_m']-fine['position_attachment_m'][0].cpu().numpy(),axis=1))),
                orientation_max_difference_deg=float(np.degrees(angles.max())))
            np.savez_compressed(folder/(trial.name+'.npz'),time_s=trial.time,**arrays,
                measured_position_origin_m=trial.truth['position_origin_m'],measured_position_attachment_m=trial.truth['position_attachment_m'],
                measured_rotation_tracking_to_world=trial.truth['rotation_tracking_to_world'],execution_phase=trial.truth['execution_phase'],
                fit_position=trial.masks['fit_position'],fit_orientation=trial.masks['fit_orientation'])
            saved[label,trial.name]=arrays
        summaries[label]=dict(training_takes=[t.name for t in training],held_out_take=label.removeprefix('leave_out_') if label.startswith('leave_out_') else None,
            parameters=model['parameters'],scores=scores,convergence=convergence,evaluator_parity=parity,
            near_gain_bounds=translation['bounds_close'],attitude_near_bound=attitude['near_bound'],delay_at_search_edge=delay in [float(DELAYS[0]),float(DELAYS[-1])],
            jacobian_condition=translation['scaled_data_jacobian_condition'])
        save(folder/'assessment.json',summaries[label]);progress(label,'assessment saved',flush=True)
    baseline={}
    for trial in trials:
        baseline[trial.name]=score_pose(trial,trial.pose(PRIOR,.08,.02))[1]
    save(job/'unfitted_example_comparison.json',baseline)
    save(job/'results.json',summaries)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    fig=Figure(figsize=(14,9));axes=fig.subplots(3,3)
    from scipy.spatial.transform import Rotation
    for i,trial in enumerate(trials):
        t=trial.time-trial.context['csv_onset_s']
        a=saved['final_all_three',trial.name];b=saved['leave_out_'+trial.name,trial.name]
        for j,(key,axis) in enumerate([('position_origin_m',0),('position_origin_m',2)]):
            ax=axes[i,j];ax.plot(t,trial.truth[key][:,axis],label='Measured');ax.plot(t,a[key][:,axis],label='All-three fit');ax.plot(t,b[key][:,axis],'--',label='Left-out take prediction')
            ax.set_ylabel(('X' if axis==0 else 'Z')+' at tracked origin (m)')
        ax=axes[i,2]
        for value,label,style in [(a,'All-three fit','-'),(b,'Left-out take prediction','--')]:
            error=Rotation.from_matrix(value['rotation_tracking_to_world'].transpose(0,2,1)@trial.truth['rotation_tracking_to_world']).magnitude()
            ax.plot(t,np.degrees(error),style,label=label)
        ax.set_ylabel('Orientation error (deg)')
        for ax in axes[i]:
            ax.axvspan(0,trial.context['csv_end_s']-trial.context['csv_onset_s'],alpha=.10,color='orange');ax.set_title(trial.name);ax.set_xlabel('Time from observed CSV onset (s)');ax.grid(alpha=.2)
        if i==0:
            for ax in axes[i]:ax.legend(fontsize=8)
    fig.suptitle('Nominal drone fit — development assessment; orange = CSV maneuver');fig.tight_layout();fig.savefig(job/'nominal_fit_review.png',dpi=140)
    protected=json.loads((job/'protected-before.json').read_text(encoding='utf-8'))
    for path,digest in protected.items():assert sha256_file(path)==digest,path
    verification=dict(platform=platform.platform(),gpu=torch.cuda.get_device_name(),elapsed_s=time.perf_counter()-started,
        protected_files_verified=len(protected),active_model_changed=False,ppo_started=False,drone_nn_fitted=False,cable_fitted=False,
        development_assessment_only=True,parameters_fitted=True)
    save(job/'verification.json',verification)
    atomic_json(job/'status.json',dict(status='NOMINAL_FITTED_REVIEW_REQUIRED',**verification))
    print('COMPLETE',job,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--job',type=Path)
    args=parser.parse_args();job=args.job or prepare()
    if not args.prepare_only:run(job.resolve())


if __name__=='__main__':main()
