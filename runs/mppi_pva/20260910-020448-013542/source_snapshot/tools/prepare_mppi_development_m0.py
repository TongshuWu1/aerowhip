"""Prepare one explicitly reviewed preliminary M0 development simulation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import shutil
import numpy as np
import torch
from experimental_data.io import atomic_json,sha256_file
from simulator.workflow import read_json
from planning.pva_job import prepare
from learning.pva_env import PVAEnvironment

ROOT=Path(__file__).resolve().parents[1]

def main():
    audit=ROOT/'runs/audits/mppi-development-M0-20260910'
    audit.mkdir(parents=True,exist_ok=False)
    source=ROOT/'runs/audits/preliminary1-gradient-resolution/candidate/model.json'
    parent=ROOT/'runs/mppi_pva/20260909-224132-314704'
    protected={str(p):sha256_file(p) for directory in (source.parent,parent/'assets') for p in directory.iterdir() if p.is_file()}
    cfg=read_json(parent/'settings.json');cfg['model_path']=str(source)
    cfg['visualization']['live_mppi']=True
    review=('User authorized one MPPI simulation on 10 September 2026. '
        'See docs/PRELIMINARY1_DAMPING_RESOLUTION.md: checked 1/2 s gradients, '
        'training-only scalar damping 0.4/s, curvature regularization 2e-5, cable NN disabled. '
        'Inherits preliminary M0 drone fit. Incomplete full-refit status is preserved; '
        'M0 development, not M1 or prospective flight validation.')
    job,command=prepare(ROOT,cfg,'M0 development | smooth damping | whip',development_review=review)
    for name in ('initial_proposal.npz','initial_proposal_provenance.json'):
        shutil.copy2(parent/name,job/name)
    atomic_json(job/'development_selection.json',dict(review=review,model_source=str(source),
        model_source_sha256=sha256_file(source),settings_parent=str(parent),
        seed_source=str(parent/'initial_proposal.npz'),seed_sha256=sha256_file(job/'initial_proposal.npz'),
        meaning='Original editable 60-action guess; every rollout uses this development model; no committed prefix reused'))
    atomic_json(audit/'job.json',dict(job=str(job),command=command))
    saved=read_json(job/'model.json')
    with np.load(job/'initial_proposal.npz') as z:actions=z['normalized_jerk'].copy()
    assert actions.shape==(60,3)
    predictions=[]
    with torch.no_grad():
        for fast in (True,False):
            env=PVAEnvironment(saved,cfg,root=job,batch_size=1,device='cuda',
                graph=fast,fused_ticks=fast,fast_solve=fast,fast_geometry=fast)
            assert env.engine.physics.motion_residual is None
            assert saved['cable']['external_drag_s_inv']==.4
            assert saved['cable']['curvature_frame_regularization']==2e-5
            result=env.rollout(actions=env.tensor(actions)[None],max_steps=60,trace=True)
            q=torch.stack([f['cable'] for f in env.frames],1).cpu().numpy()
            assert np.isfinite(q).all()
            predictions.append(q)
            atomic_json(audit/('fast_seed.json' if fast else 'reference_seed.json'),dict(
                success=bool(result['success'][0]),failed=bool(result['failed'][0]),
                distance_m=float(result['minimum_tip_distance_m'][0]),frames=q.shape[1]))
    delta=float(np.max(np.abs(predictions[0]-predictions[1])))
    assert delta<1e-7,delta
    assert all(sha256_file(Path(p))==digest for p,digest in protected.items())
    atomic_json(audit/'preflight.json',dict(status='passed',device=torch.cuda.get_device_name(),
        max_fast_reference_cable_difference_m=delta,protected_hashes=protected,
        cable_residual_enabled=False,external_drag_s_inv=.4,curvature_frame_regularization=2e-5))
    print(json.dumps(dict(job=str(job),command=command)),flush=True)

if __name__=='__main__':main()
