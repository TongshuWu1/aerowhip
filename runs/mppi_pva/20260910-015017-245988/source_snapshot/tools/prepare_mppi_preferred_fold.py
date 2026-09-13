"""One reviewed preferred-fold trial; fixed ranking and GPU agreement first."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from copy import deepcopy
import shutil
import json
import numpy as np
import torch
from experimental_data.io import atomic_json,sha256_file
from simulator.workflow import read_json
from planning.pva_job import prepare
from planning.mppi_trajectory import optimize,interpolation_matrix
from planning.whip_objective import WAVE_OBJECTIVE,PreferredFoldCapture,ENCOUNTER_FIELDS
from learning.pva_env import PVAEnvironment
from tools.check_preferred_fold_objective import ROOT,AUDIT,PREFERRED,CURRENT,ARCHIVE


@torch.no_grad()
def main():
    assert read_json(AUDIT/'ranking_check.json')['status']=='passed'
    if (AUDIT/'job.json').exists():raise FileExistsError('Prepared job already exists; inspect, do not duplicate')
    parent=ROOT/'runs/mppi_pva/20260910-011618-458510'
    assert read_json(parent/'status.json')['status'] in ('stopped','completed','failed')
    source=ROOT/'runs/audits/preliminary1-gradient-resolution/candidate/model.json'
    protected=read_json(ROOT/'runs/audits/preferred-whip-review-20260910/source_hashes.json')
    protected.update({str(p):sha256_file(p) for p in source.parent.iterdir() if p.is_file()})
    assert all(sha256_file(Path(p))==digest for p,digest in protected.items())
    cfg=read_json(parent/'settings.json');cfg['model_path']=str(source)
    cfg['trajectory_objective']=dict(WAVE_OBJECTIVE)
    cfg['mppi']['seed']=657
    review=read_json(parent/'identity.json')['development_review']+' User authorized preferred-fold objective, fixed ranking checks and one 512-sample 1.5 s trial.'
    job,command=prepare(ROOT,cfg,'M0 development | preferred travelling fold | 512 samples',development_review=review)
    with np.load(PREFERRED/'plan.npz') as z:old=z['normalized_jerk'].copy()
    assert old.shape==(39,3)
    native=np.concatenate((old,np.zeros((6,3))))
    np.savez_compressed(job/'initial_proposal.npz',normalized_jerk=native)
    shutil.copy2(AUDIT/'wave_reference.npz',job/'wave_reference.npz')
    atomic_json(job/'initial_proposal_provenance.json',dict(source=str(PREFERRED/'plan.npz'),
        source_sha256=sha256_file(PREFERRED/'plan.npz'),saved_sha256=sha256_file(job/'initial_proposal.npz'),
        transformation='39 exact accepted jerk commands plus six zero-jerk commands; no time compression. Control points perturb the exact latent baseline.',
        physics='Every candidate uses frozen development M0. No archived model or accepted forecast substitutes for prediction.'))
    atomic_json(job/'development_selection.json',dict(model_source=str(source),model_source_sha256=sha256_file(source),
        review=review,parent_run=str(parent),model_changed=False,task_or_bounds_changed=False,
        objective='preferred_fold_v1; strict historical wave success logged unchanged, new scalar reward selects candidates',
        reference=read_json(AUDIT/'ranking_check.json'),
        scoring_rate='First any contact or closest precontact approach at 150 Hz with interpolated event kinematics. Shape history at 30 Hz ending at exact event.',
        proposal_definition='Four proposals with ten smooth residual control points around exact old commands; 512 random samples, adaptive ESS, zero control prior'))
    atomic_json(AUDIT/'job.json',dict(job=str(job),command=command))
    model=read_json(job/'model.json');reference=torch.tensor(np.load(job/'wave_reference.npz')['shape'],device='cuda',dtype=torch.float64)
    with np.load(CURRENT/'plan.npz') as z:current=z['normalized_jerk'].copy()
    actions=torch.tensor(np.stack([native,current]),device='cuda',dtype=torch.float64)
    env=PVAEnvironment(model,cfg,root=job,batch_size=2,device='cuda');capture=PreferredFoldCapture(reference)
    result=env.rollout(actions=actions,observer=capture);scores,_=capture.score(env,result,actions,cfg['trajectory_objective'])
    reports=[]
    for i in range(2):
        check=PVAEnvironment(model,cfg,root=job,device='cuda');observer=PreferredFoldCapture(reference)
        output=check.rollout(actions=actions[i:i+1],trace=True,observer=observer)
        score,_=observer.score(check,output,actions[i:i+1],cfg['trajectory_objective'])
        errors={name:float((getattr(env,name)[i].double()-getattr(check,name)[0].double()).abs().max()) for name in ENCOUNTER_FIELDS}
        errors['score']=float(abs(scores[i]-score[0]))
        assert max(errors.values())<1e-7,errors
        reports.append(dict(case=('old_commands','current_contact')[i],errors=errors,contact=bool(check.contact[0]),
            time_s=float(check.encounter_time[0]),distance_m=float(check.encounter_distance[0])))
    assert reports[1]['contact'] and abs(reports[1]['time_s']-1.1248781338)<1e-7
    # Native pulse timing is preserved by the new residual parameterization.
    basis=interpolation_matrix(45,10,device='cuda')
    reconstructed=torch.tanh(torch.atanh(actions[0])+basis@torch.zeros(10,3,device='cuda',dtype=torch.float64))
    assert float((reconstructed-actions[0]).abs().max())<1e-14
    atomic_json(AUDIT/'gpu_agreement.json',reports)
    smoke=AUDIT/'smoke';smoke.mkdir();(smoke/'checkpoints').mkdir()
    for name in ('initial_proposal.npz','wave_reference.npz'):shutil.copy2(job/name,smoke/name)
    test_cfg=deepcopy(cfg);test_cfg['mppi'].update(samples=16,iterations=2,minimum_iterations=2,patience=1)
    atomic_json(smoke/'settings.json',test_cfg);atomic_json(smoke/'model.json',model)
    result=optimize(smoke,model,test_cfg)
    assert result['independent_score_difference']<1e-5
    assert all(sha256_file(Path(p))==digest for p,digest in protected.items())
    atomic_json(AUDIT/'preflight.json',dict(status='passed',independent_score_difference=result['independent_score_difference'],
        device='Windows / RTX 4080 / float64',protected_hashes=protected,gpu_agreement=reports,
        ranking_check='passed',unit_tests='25 passed',smoke=str(smoke)))
    print(json.dumps(dict(job=str(job),command=command)),flush=True)

if __name__=='__main__':main()
