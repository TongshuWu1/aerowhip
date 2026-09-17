"""Continuous native-command -> predicted pose -> attached cable assessment.

This is the effective loaded-drone execution model. Cable reaction is already
implicit in its fitted response; do not add a second explicit reaction force.
It is not the virtual force-to-reference generator or a flight sender.
"""
from pathlib import Path
from copy import deepcopy
import shutil
import numpy as np
import torch
from .bootstrap_bundle import read, VERSION, snapshot_stage, verify
from .bootstrap_drone import trial_args
from .nominal_pose_fit import PreparedTrial
from .force_dataset import reconstruct_dder_nodes
from .state_initialization import causal_state
from .cable_fit import PreparedTake
from .differentiable_fit import rollout
from .io import atomic_json, sha256_file
from .historical_fit import progress, check_stop
from simulator.geometry import normalized_rotations_xyzw
from simulator.drone_pose_response import PoseResponseParameters, predict_pose
from simulator.drone_pose_residual import load_residual
from simulator.cable import CableConfiguration, DderModel
from simulator.cable.residual import FrozenMotionResidual


def measured_history(dataset, start_time, cable, offset, *, samples=21):
    """Only past named markers and measured pose may initialize the cable."""
    times=dataset['controller_time_s']
    ids=np.flatnonzero(times<=start_time+1e-9)
    if len(ids)<samples or abs(times[ids[-1]]-start_time)>1e-8:
        raise ValueError('Missing cable initialization timestamp/history')
    ids=ids[-samples:]
    if not np.allclose(np.diff(times[ids]),.01,atol=1e-7):
        raise ValueError('Gapped cable initialization history')
    rotations, valid=normalized_rotations_xyzw(dataset['drone_quaternion_xyzw'][ids])
    root=dataset['drone_position_m'][ids]+np.einsum('tij,j->ti',rotations,offset)
    nodes, node_valid=reconstruct_dder_nodes(root,valid&dataset['drone_position_valid'][ids],
        dataset['cable_position_m'][ids],dataset['cable_valid'][ids],cable)
    if not node_valid.all():raise ValueError('Invalid measured cable initialization history')
    return nodes, int(ids[-1])


def marker_scores(predicted, measured, mask):
    if predicted.shape!=measured.shape or mask.shape!=measured.shape[:-1]:
        raise ValueError('Marker score shape mismatch')
    if not np.isfinite(predicted).all():raise ValueError('Nonfinite combined cable prediction')
    mask=mask&np.isfinite(measured).all(-1)
    squared=np.sum((predicted-measured)**2,axis=-1)
    def rms(values,valid):return float(np.sqrt(values[valid].mean())) if valid.any() else None
    return dict(marker_rmse_m=rms(squared,mask),tip_rmse_m=rms(squared[:,-1],mask[:,-1]),
        marker_max_m=float(np.sqrt(squared[mask].max())) if mask.any() else None,
        tip_max_m=float(np.sqrt(squared[:,-1][mask[:,-1]].max())) if mask[:,-1].any() else None,
        per_marker_rmse_m=[rms(squared[:,i],mask[:,i]) for i in range(mask.shape[1])],
        marker_samples=int(mask.sum()),tip_samples=int(mask[:,-1].sum()))


@torch.no_grad()
def assess_trial(trial, payload, drone_candidate, folder, *, device='cuda', case_filter=None):
    cable=CableConfiguration.from_mapping(payload['cable'])
    model=DderModel(cable.dder_parameters(EI=payload['cable']['EI_n_m2'],Cb=payload['cable']['Cb_n_m2_s']))
    params=PoseResponseParameters(**drone_candidate['nominal']['parameters'])
    spec=drone_candidate['residual'];drone_nn=load_residual(spec['checkpoint'],spec['sha256'],device)
    spec=payload['motion_residual'];cable_nn=FrozenMotionResidual(spec['checkpoint'],spec['sha256'])
    args=trial_args(trial,params,device=device,maneuver_only=True)
    times=args['output_time_s'];count=len(times)
    with np.load(trial.folder/'dataset.npz') as source:dataset={k:source[k] for k in source.files}
    offset=np.asarray(payload['recorded_data']['optitrack_to_attachment_offset_body_m'])
    if not np.allclose(offset,trial.offset,atol=1e-12,rtol=0):raise ValueError('Drone/cable attachment convention mismatch')
    history,start=measured_history(dataset,times[0],cable,offset)
    if not np.allclose(dataset['controller_time_s'][start:start+count],times,atol=1e-8,rtol=0):
        raise ValueError('Combined pose/marker timestamps differ')
    measured=dataset['cable_position_m'][start:start+count]
    mask=dataset['cable_valid'][start:start+count].copy()
    phase=trial.truth['execution_phase'][:count]=='csv_maneuver'
    mask&=phase[:,None]
    # No candidate-dependent exclusions; preserve phase-boundary samples for scoring.
    history=torch.as_tensor(history[None],dtype=torch.float64,device=device)
    state=causal_state(history,.01,model)
    projection=float(torch.linalg.vector_norm(state.positions_m-history[:,-1],dim=-1).max())
    pose={key:predict_pose(**args,residual=net) for key,net in [('nominal',None),('residual',drone_nn)]}
    measured_root=torch.as_tensor(trial.truth['position_attachment_m'][:count][None],dtype=torch.float64,device=device)
    if not torch.isfinite(measured_root).all():raise ValueError('Invalid measured attachment diagnostic')
    force_params=torch.tensor([payload['cable']['EI_n_m2'],payload['cable']['Cb_n_m2_s'],0.],dtype=torch.float64,device=device)
    initial_positions=state.positions_m;initial_velocities=state.velocities_m_s
    outputs={};arrays=dict(time_s=times,measured_markers_m=measured,marker_score_mask=mask,
        measured_attachment_m=measured_root[0].cpu().numpy(),initial_positions_m=initial_positions[0].cpu().numpy(),
        initial_velocities_m_s=initial_velocities[0].cpu().numpy())
    cases=[('nominal_drone_physics_cable','nominal',False,False),
        ('nominal_drone_residual_cable','nominal',True,False),
        ('residual_drone_physics_cable','residual',False,False),
        ('both_residuals','residual',True,False),
        ('measured_attachment_physics_cable','measured',False,False),
        ('measured_attachment_residual_cable','measured',True,False),
        ('both_residuals_hanging_initialization','residual',True,True)]
    if case_filter is not None:cases=[case for case in cases if case[0] in case_filter]
    for label,drone_mode,use_cable_nn,hanging in cases:
        root=measured_root if drone_mode=='measured' else pose[drone_mode]['position_attachment_m']
        if not torch.allclose(root[:,0],initial_positions[:,0],atol=1e-10,rtol=0):
            raise ValueError('Combined initial attachment mismatch')
        q=initial_positions;v=initial_velocities
        if hanging:
            q=root[:,:1].expand(-1,cable.node_count,-1).clone()
            q[:,:,2]-=q.new_tensor([0.,*np.cumsum(cable.rest_lengths_m)])
            v=pose[drone_mode]['velocity_attachment_m_s'][:,:1].expand_as(q).clone()
        take=PreparedTake(trial.name,'development',q,v,root,torch.as_tensor(measured[None],dtype=q.dtype,device=device),(start,),projection,.01)
        # Keep assessment cases separate: a bootstrap batch comparison showed
        # up to 1.2 mm accumulated cable differences despite identical inputs.
        # Do not call those numerically identical or select on the faster result.
        model.motion_residual=cable_nn if use_cable_nn else None
        prediction=rollout(take,model,cable,force_params)[0].cpu().numpy()
        scores=marker_scores(prediction,measured,mask)
        delta=(root-measured_root)[0].cpu().numpy()
        scores['attachment_rmse_m']=float(np.sqrt(np.sum(delta[phase]**2,axis=-1).mean()))
        outputs[label]=scores;arrays[label+'_markers_m']=prediction;arrays[label+'_attachment_m']=root[0].cpu().numpy()
    np.savez_compressed(folder/(trial.name+'.npz'),**arrays)
    return dict(initial_time_s=float(times[0]),final_observed_time_s=float(times[-1]),
        csv_start_s=trial.context['csv_onset_s'],csv_end_s=trial.context['csv_end_s'],
        initialization_max_projection_m=projection,continuous_no_resets=True,cases=outputs)


def save_bundle(job):
    bundle=job/'bundle';bundle.mkdir()
    cable=deepcopy(read(job/'cable/final/candidate_model.json'))
    drone=deepcopy(read(job/'drone/final_all_three/candidate.json'))
    for name,source in [('cable_residual.pt',cable['motion_residual']['checkpoint']),('drone_residual.pt',drone['residual']['checkpoint'])]:
        shutil.copy2(source,bundle/name)
    cable['motion_residual']['checkpoint']='cable_residual.pt';drone['residual']['checkpoint']='drone_residual.pt'
    # A cable component must not masquerade as the old force-driven point mass
    # model: the execution model supplies a kinematic boundary at the attachment.
    cable={key:cable[key] for key in ['cable','recorded_data','mass_measurement','motion_residual']}
    cable['schema']='bootstrap_cable_component_v1'
    cable['boundary']='Predicted rigid attachment position; freely pivoting first span'
    atomic_json(bundle/'cable_model.json',cable);atomic_json(bundle/'drone_model.json',drone)
    manifest=dict(schema='preliminary_execution_bundle_v1',status='DEVELOPMENT_CANDIDATE',
        files={p.name:sha256_file(p) for p in bundle.iterdir()},
        coupling='Delayed FullState -> effective loaded tracked-origin pose -> rigid attachment -> DDER cable; no added cable reaction on fitted drone',
        initialization='Past-only measured pose and cable for primary assessment; separate hanging-cable sensitivity',
        source_job=job.name,active_model_changed=False,flight_ready=False,ppo_integrated=False,
        limitations=['Legacy development data; prospective new-flight assessment required',
            'No demonstrated recovery or unflown hit-time prediction',
            'Virtual force-to-reference and 30 Hz PPO integration remain separate work'])
    atomic_json(bundle/'manifest.json',manifest)
    load_bundle(bundle)
    return manifest


def load_bundle(folder):
    """Resolve the portable component files and validate immutable hashes."""
    folder=Path(folder).resolve();manifest=read(folder/'manifest.json')
    if manifest['schema']!='preliminary_execution_bundle_v1':raise ValueError('Unknown bundle schema')
    for name,digest in manifest['files'].items():
        path=(folder/name).resolve()
        if path.parent!=folder or sha256_file(path)!=digest:raise ValueError('Bundle file hash/path mismatch')
    cable=read(folder/'cable_model.json');drone=read(folder/'drone_model.json')
    if cable['cable']['external_drag_s_inv']!=0:raise ValueError('Bootstrap cable requires zero fixed drag')
    for spec in (cable['motion_residual'],drone['residual']):
        path=(folder/spec['checkpoint']).resolve()
        if path.parent!=folder or sha256_file(path)!=spec['sha256']:raise ValueError('Residual hash/path mismatch')
        spec['checkpoint']=str(path)
    return cable,drone


def run(job):
    job=Path(job)
    # Require every fit to finish before packaging an apparently complete bundle.
    read(job/'cable/results.json');read(job/'drone/results.json')
    snapshot_stage(job,'combined',['experimental_data/bootstrap_combined.py','experimental_data/bootstrap_drone.py',
        'experimental_data/constrained_identification.py','experimental_data/bootstrap_report.py','tools/fit_bootstrap_bundle.py',
        'experimental_data/state_initialization.py','experimental_data/force_dataset.py','experimental_data/differentiable_fit.py',
        'simulator/drone_pose_response.py','simulator/drone_pose_residual.py','simulator/cable/dder.py','simulator/cable/residual.py'])
    output=job/'combined';output.mkdir();settings=read(job/'protocol.json');reviews=read(job/'nominal_source_protocol.json')['reviews']
    results={}
    for label in ['final_all_three']+['leave_out_'+n for n in settings['drone_response_takes']]:
        check_stop(job);folder=output/label;folder.mkdir()
        cable_label='final' if label=='final_all_three' else label
        payload=read(job/'cable'/cable_label/'candidate_model.json');drone=read(job/'drone'/label/'candidate.json')
        names=settings['drone_response_takes'] if label=='final_all_three' else [label.removeprefix('leave_out_')]
        results[label]={}
        for name in names:
            trial=PreparedTrial(job/'pose_inputs'/VERSION/name,review=reviews[name])
            results[label][name]=assess_trial(trial,payload,drone,folder)
            atomic_json(folder/'results.json',results[label]);progress(output,label+' '+name+' complete')
    atomic_json(output/'results.json',results);save_bundle(job)
    atomic_json(job/'combined_verification.json',verify(job))
    atomic_json(job/'status.json',dict(status='FIT_AND_COMBINED_ASSESSMENT_COMPLETE',active_model_changed=False,
        ppo_started=False,flight_ready=False,assessment='Legacy development folds; prospective assessment pending'))
