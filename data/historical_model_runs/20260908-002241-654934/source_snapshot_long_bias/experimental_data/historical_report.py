"""Review and, only after completed checks, install a versioned full-model baseline."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import numpy as np
from .historical_fit import read
from .io import atomic_json,sha256_file,canonical_json_hash
from simulator.workflow import stamp


def build_review(job):
    protocol=read(job/'protocol.json');cable_dir=job/read(job/'cable_result.json')['directory']
    records=[]
    for i,names in enumerate(protocol['folds'],1):
        c0=read(cable_dir/f'fold_{i}/physical_evaluation.json')
        c1=read(cable_dir/f'fold_{i}/residual_evaluation.json')
        d=read(job/f'drone_attachment/fold_{i}/evaluation.json')
        for name in names:
            records.append(dict(take=name,fold=i,cable_physics_marker_m=c0[name]['marker_rmse_m'],
                cable_residual_marker_m=c1[name]['marker_rmse_m'],cable_physics_tip_m=c0[name]['tip_rmse_m'],
                cable_residual_tip_m=c1[name]['tip_rmse_m'],drone_nominal_m=d['nominal'][name]['position_rmse_m'],
                drone_residual_m=d['residual'][name]['position_rmse_m']))
    averages={key:float(np.mean([r[key] for r in records])) for key in records[0] if key not in ('take','fold')}
    combined=read(job/'combined/review.json')['cross_fitted_equal_take']
    before,after=combined['nominal_drone_physics_cable'],combined['both_residuals']
    final=read(cable_dir/'final/review.json')
    old=read(job/'original_baseline_evaluation.json')['per_take']
    new=read(cable_dir/'final/residual_evaluation.json')
    previous_comparison={key:dict(previous=float(np.mean([r[key] for r in old.values()])),
        candidate=float(np.mean([r[key] for r in new.values()]))) for key in ('marker_rmse_m','tip_rmse_m')}
    checks=dict(cable_marker_improves=averages['cable_residual_marker_m']<averages['cable_physics_marker_m'],
        cable_tip_improves=averages['cable_residual_tip_m']<averages['cable_physics_tip_m'],
        drone_attachment_improves=averages['drone_residual_m']<averages['drone_nominal_m'],
        combined_marker_improves=after['marker_rmse_m']<before['marker_rmse_m'],
        combined_tip_improves=after['tip_rmse_m']<before['tip_rmse_m'],
        final_cable_nn_selected=final['selected_update']>0,
        improves_previous_cable_marker=previous_comparison['marker_rmse_m']['candidate']<previous_comparison['marker_rmse_m']['previous'],
        improves_previous_cable_tip=previous_comparison['tip_rmse_m']['candidate']<previous_comparison['tip_rmse_m']['previous'])
    result=dict(status='DEVELOPMENT_BASELINE_ACCEPTED' if all(checks.values()) else 'REVIEW_REQUIRED',
        checks=checks,per_take=records,equal_take=averages,combined=combined,previous_calibration_comparison=previous_comparison,
        independent_test=False,selection='Fit-only selection within each whole-take fold; final all-data refit',
        limitations=['Cable NN gains may be small; do not equate training fit with improved real hitting.',
            'Effective drone model includes fixed cable loading; one-way cable boundary, not separately identified motor physics.',
            'Cable stiffness is weakly identified: fold estimates span many orders of magnitude for very small objective differences. These are effective predictive parameters, not measured material constants.',
            'Recovery is exported as the existing gentle reference, not included in the learned strike objective.',
            'All historical takes are development data; next frozen-policy flights provide prospective assessment.'])
    atomic_json(job/'review.json',result)
    return result


def install(root,job):
    root,job=Path(root).resolve(),Path(job).resolve()
    review=build_review(job)
    if not all(review['checks'].values()):raise ValueError('Full model did not pass its development comparison; candidate preserved')
    runtime=read(job/'runtime_final.json')
    if not runtime.get('passed'):raise ValueError('Final model runtime verification is required')
    original=read(job/'original_model.json')
    if canonical_json_hash(read(root/'config/model.json'))!=canonical_json_hash(original):
        raise ValueError('Active model changed during fitting; preserving the newer selection')
    cable_dir=job/read(job/'cable_result.json')['directory']/'final'
    candidate=read(cable_dir/'candidate_model.json')
    drone_source=job/'drone_attachment/final/drone_residual.pt'
    version=stamp()+'-historical-full-model'
    dest=root/'data/baselines'/version;dest.mkdir(parents=True,exist_ok=False)
    for name,path in [('motion_residual.pt',Path(candidate['motion_residual']['checkpoint'])),
                      ('drone_attachment.pt',drone_source),('drone_origin.pt',job/'drone/final/drone_residual.pt')]:
        shutil.copy2(path,dest/name)
    candidate['motion_residual'].update(checkpoint=(dest/'motion_residual.pt').relative_to(root).as_posix(),
        sha256=sha256_file(dest/'motion_residual.pt'))
    candidate['fullstate_execution']=dict(enabled=True,schema='effective_attachment_execution_v1',
        checkpoint=(dest/'drone_attachment.pt').relative_to(root).as_posix(),sha256=sha256_file(dest/'drone_attachment.pt'),
        reference_rate_hz=30,force_policy_rate_hz=20,recovery_objective='excluded',
        coupling='Predicted attachment boundary; no second cable reaction',
        export_position='virtual_attachment_minus_initial_world_attachment_offset')
    candidate['cable']['previous_parameter_source']=candidate['cable'].get('parameter_source')
    candidate['cable']['parameter_source']=(dest/'manifest.json').relative_to(root).as_posix()
    from simulator.point_mass import ForceControlledPointCable
    from simulator.fullstate_execution import FullStateAttachmentModel
    ForceControlledPointCable.from_mapping(candidate,root=root)
    FullStateAttachmentModel(dest/'drone_attachment.pt',candidate['fullstate_execution']['sha256'])
    atomic_json(dest/'model.json',candidate)
    for name in ('review.json','protocol.json','audit.json','runtime_final.json'):
        shutil.copy2(job/name,dest/name)
    shutil.copy2(root/'config/ppo.json',dest/'previous_ppo_config.json')
    atomic_json(dest/'manifest.json',dict(version=version,source_job=str(job),provenance='All historical development data, quality masked, three whole-take folds and all-data final fit',
        model_sha256=canonical_json_hash(candidate),original_model_sha256=canonical_json_hash(original),
        independent_test=False,flight_validated=False,ppo_started=False))
    config=read(root/'config/ppo.json')
    config['deployment'].update(enabled=True,recovery_failure_penalty=0.,force_gain_fraction=0.,force_lag_max_s=0.)
    atomic_json(root/'config/ppo.json',config)
    manifest=read(root/'data/dataset_manifest.json')
    shutil.copy2(root/'data/dataset_manifest.json',dest/'previous_dataset_manifest.json')
    manifest['historical_fit_authorized']=True
    for name,row in manifest['takes'].items():
        if name in read(job/'protocol.json')['takes']:
            row.update(role='training',note='Authorized all-data historical baseline. Rotating development checks are recorded in '+str(job/'protocol.json'))
    atomic_json(root/'data/dataset_manifest.json',manifest)
    atomic_json(root/'config/model.json',candidate)
    atomic_json(root/'config/baseline.json',dict(version=version,model_sha256=canonical_json_hash(candidate)))
    atomic_json(job/'applied.json',dict(version=version,baseline=str(dest),model_sha256=canonical_json_hash(candidate),ppo_started=False))
    return dest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True);parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    print(install(Path(__file__).resolve().parents[1],args.job) if args.apply else build_review(args.job),flush=True)
