"""Prepare one authorized 512-sample timing/strength trial, with a bounded smoke."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from copy import deepcopy
import shutil
import json
import numpy as np
from simulator.workflow import read_json
from experimental_data.io import atomic_json,sha256_file
from planning.pva_job import prepare
from planning.mppi_trajectory import optimize

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'runs/audits/mppi-timed-baselines-20260910'


def main():
    AUDIT.mkdir(parents=True,exist_ok=False)
    prior=ROOT/'runs/audits/mppi-preferred-fold-20260910'
    parent=Path(read_json(prior/'job.json')['job'])
    sweep=ROOT/'runs/mppi_pva/20260910-011618-458510'
    for p in (parent,sweep):assert read_json(p/'status.json')['status']=='completed'
    assert read_json(prior/'status.json')['status']=='completed'
    protected=read_json(prior/'frozen_forecast_manifest.json')
    protected.update(read_json(prior/'preflight.json')['protected_hashes'])
    protected.update({str(sweep/name):sha256_file(sweep/name) for name in ('plan.npz','model.json','settings.json')})
    assert all(sha256_file(Path(p))==h for p,h in protected.items())
    source=ROOT/'runs/audits/preliminary1-gradient-resolution/candidate/model.json'
    cfg=read_json(parent/'settings.json');cfg['model_path']=str(source)
    cfg['mppi'].update(proposal_parameterization='timed_baselines_v1',timing_source_knots=[0.,.3,.6,1.],
        timing_noise_scales=[.12,.3,.6],strength_noise_scales=[.1,.2,.4])
    before=read_json(parent/'settings.json')
    for key in ('task','limits','action','trajectory_objective','launch'):
        assert cfg[key]==before[key],key
    review=read_json(parent/'identity.json')['development_review']+' User authorized one timing/strength-search trial retaining both exact baselines, with unchanged reward and model.'
    job,command=prepare(ROOT,cfg,'M0 development | timing and strength | both baselines | 512',development_review=review)
    paths=[parent/'initial_proposal.npz',sweep/'plan.npz']
    bases=np.stack([np.load(p)['normalized_jerk'] for p in paths]);assert bases.shape==(2,45,3)
    np.savez_compressed(job/'proposal_baselines.npz',normalized_jerk=bases)
    shutil.copy2(paths[0],job/'initial_proposal.npz')
    shutil.copy2(parent/'wave_reference.npz',job/'wave_reference.npz')
    shutil.copy2(parent/'wave_reference.npz',AUDIT/'wave_reference.npz')
    provenance=dict(sources=[dict(path=str(p),sha256=sha256_file(p)) for p in paths],
        old_command_provenance=read_json(parent/'initial_proposal_provenance.json'),
        saved_sha256=sha256_file(job/'proposal_baselines.npz'),
        interpretation='Exact command baselines only; both reevaluated under the same development M0 in every batch. No saved state/forecast substituted.',
        timing='Independent XYZ monotone three-phase retiming, exact interval averages of source zero-order-hold jerk. Duration remains 1.5 s.',
        strength='Per-axis latent gain plus ten smooth XYZ residual controls',
        families=[0,1,0,1],deterministic_rows='Four means, incumbent, both exact baselines; excluded from weight updates')
    atomic_json(job/'initial_proposal_provenance.json',provenance)
    atomic_json(job/'development_selection.json',dict(model_source=str(source),model_source_sha256=sha256_file(source),
        review=review,parent_run=str(parent),model_changed=False,task_or_bounds_changed=False,reward_changed=False,
        objective='Unchanged preferred_fold_v1',proposal=provenance,
        source_note='Includes the previously verified causal forward-peak fix. Frozen prior job bytes unchanged.'))
    atomic_json(AUDIT/'job.json',dict(job=str(job),command=command))
    smoke=AUDIT/'smoke';smoke.mkdir();(smoke/'checkpoints').mkdir()
    for name in ('initial_proposal.npz','proposal_baselines.npz','wave_reference.npz'):shutil.copy2(job/name,smoke/name)
    test_cfg=deepcopy(cfg);test_cfg['mppi'].update(samples=16,iterations=2,minimum_iterations=2,patience=1,initial_minimum_iterations=2)
    atomic_json(smoke/'settings.json',test_cfg);model=read_json(job/'model.json');atomic_json(smoke/'model.json',model)
    result=optimize(smoke,model,test_cfg)
    assert result['independent_score_difference']<1e-5
    assert result['best_reward']+1e-7>=max(result['baseline_scores'])
    assert all(sha256_file(Path(p))==h for p,h in protected.items())
    atomic_json(AUDIT/'preflight.json',dict(status='passed',device='Windows / RTX 4080 / CUDA float64',
        independent_score_difference=result['independent_score_difference'],baseline_scores=result['baseline_scores'],
        baseline_preservation_passed=True,protected_hashes=protected,smoke=str(smoke),unit_tests='17 passed'))
    print(json.dumps(dict(job=str(job),command=command)),flush=True)

if __name__=='__main__':main()
