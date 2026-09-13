"""Prepare the user-authorized 512-sample, 1.5 s development-M0 trial."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from copy import deepcopy
import shutil
import json
import numpy as np
from experimental_data.io import atomic_json,sha256_file
from simulator.workflow import read_json
from planning.pva_job import prepare
from planning.mppi_trajectory import OBJECTIVE,optimize

ROOT=Path(__file__).resolve().parents[1]

def main():
    audit=ROOT/'runs/audits/mppi-whole-whip-20260910';audit.mkdir(parents=True,exist_ok=False)
    parent=ROOT/'runs/mppi_pva/20260910-004200-654955'
    assert read_json(parent/'status.json')['status'] in ('stopped','completed','failed')
    source=ROOT/'runs/audits/preliminary1-gradient-resolution/candidate/model.json'
    protected={str(p):sha256_file(p) for directory in (source.parent,parent/'assets') for p in directory.iterdir() if p.is_file()}
    cfg=read_json(parent/'settings.json');cfg['model_path']=str(source)
    cfg['task']['duration_s']=1.5
    cfg['mppi'].update(mode='open_loop',parameterization='control_points',samples=512,horizon_s=1.5,
        support_points=10,proposal_count=4,target_ess_fraction=.2,control_point_noise_scales=[.03,.09,.2],
        iterations=0,minimum_iterations=20,patience=12)
    cfg['trajectory_objective']=dict(OBJECTIVE)
    review=read_json(parent/'identity.json')['development_review']+' User authorized the full-trajectory MPPI trial; identical model, new planner/objective.'
    job,command=prepare(ROOT,cfg,'M0 development | whole whip | 512 samples | 1.5 s',development_review=review)
    with np.load(parent/'initial_proposal.npz') as data:native=data['normalized_jerk'][:45].copy()
    np.savez_compressed(job/'initial_proposal.npz',normalized_jerk=native)
    atomic_json(job/'initial_proposal_provenance.json',dict(source=str(parent/'initial_proposal.npz'),
        source_sha256=sha256_file(parent/'initial_proposal.npz'),transformation='First 45 of the original editable 60 jerk commands; no time compression or old accepted path',
        saved_sha256=sha256_file(job/'initial_proposal.npz')))
    atomic_json(job/'development_selection.json',dict(model_source=str(source),model_source_sha256=sha256_file(source),
        review=review,parent_run=str(parent),model_changed=False,
        objective='Separate trajectory_objective; legacy per-tick reward is logged only, strict success and existing bounds unchanged',
        scoring_rate='30 Hz continuous shape guidance; closest tip distance and strict strike checks at 150 Hz',
        proposal_definition='Four independent exponential-weight updates with adaptive temperature, zero control prior; deterministic baselines excluded from weights'))
    atomic_json(audit/'job.json',dict(job=str(job),command=command))
    # A bounded 2-update smoke check verifies the execution path, not another optimized trial.
    smoke=audit/'smoke';smoke.mkdir();(smoke/'checkpoints').mkdir()
    shutil.copy2(job/'initial_proposal.npz',smoke/'initial_proposal.npz')
    test_cfg=deepcopy(cfg);test_cfg['mppi'].update(samples=16,iterations=2,minimum_iterations=2,patience=1)
    atomic_json(smoke/'settings.json',test_cfg)
    atomic_json(smoke/'model.json',read_json(job/'model.json'))
    result=optimize(smoke,read_json(job/'model.json'),test_cfg)
    assert result['independent_score_difference']<1e-5
    assert all(sha256_file(Path(p))==digest for p,digest in protected.items())
    atomic_json(audit/'preflight.json',dict(status='passed',independent_score_difference=result['independent_score_difference'],
        device='Windows / RTX 4080 / float64',protected_hashes=protected,smoke=str(smoke)))
    print(json.dumps(dict(job=str(job),command=command)),flush=True)

if __name__=='__main__':main()
