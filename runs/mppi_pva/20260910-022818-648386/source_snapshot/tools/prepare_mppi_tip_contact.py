"""Prepare the one authorized repeat with the simplified tip-contact gate."""
from pathlib import Path
import sys
import shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from simulator.workflow import read_json
from experimental_data.io import atomic_json,sha256_file
from planning.pva_job import prepare

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'runs/audits/mppi-tip-contact-20260910'

def main():
    AUDIT.mkdir(parents=True,exist_ok=False)
    prior=ROOT/'runs/audits/mppi-timed-baselines-20260910'
    parent=Path(read_json(prior/'job.json')['job'])
    assert read_json(parent/'status.json')['status']=='completed'
    assert read_json(prior/'status.json')['status']=='completed'
    assert read_json(ROOT/'runs/audits/tip-contact-gate-20260910/status.json')['status']=='completed'
    protected=read_json(prior/'frozen_forecast_manifest.json')
    protected.update(read_json(ROOT/'runs/audits/tip-contact-gate-20260910/protected_hashes.json'))
    # Replay selection will deliberately change only after a checked new export.
    protected.pop(str(ROOT/'config/pva/replay.json'),None)
    atomic_json(AUDIT/'previous_replay_selection.json',read_json(ROOT/'config/pva/replay.json'))
    assert all(sha256_file(Path(p))==h for p,h in protected.items())
    cfg=read_json(parent/'settings.json')
    cfg['task']['success_criterion']='tip_contact_v1'
    source=ROOT/'runs/audits/preliminary1-gradient-resolution/candidate/model.json'
    assert sha256_file(source)==read_json(parent/'identity.json')['model_source_sha256']
    review=read_json(parent/'identity.json')['development_review']+' User authorized one new MPPI run with tip_contact_v1; identical model, reward, search settings and baselines.'
    job,command=prepare(ROOT,cfg,'M0 development | tip contact | 512 samples | 1.5 s',development_review=review)
    for name in ('initial_proposal.npz','proposal_baselines.npz','wave_reference.npz'):
        shutil.copy2(parent/name,job/name)
    shutil.copy2(parent/'wave_reference.npz',AUDIT/'wave_reference.npz')
    provenance=dict(parent_run=str(parent),parent_provenance=read_json(parent/'initial_proposal_provenance.json'),
        sources={name:dict(path=str(parent/name),sha256=sha256_file(parent/name)) for name in ('initial_proposal.npz','proposal_baselines.npz','wave_reference.npz')},
        interpretation='Editable commands/style reference only; all dynamics and target predictions are newly evaluated under the same development M0.')
    atomic_json(job/'initial_proposal_provenance.json',provenance)
    atomic_json(job/'development_selection.json',dict(model_source=str(source),model_source_sha256=sha256_file(source),
        review=review,parent_run=str(parent),model_changed=False,reward_weights_changed=False,bounds_changed=False,
        changed_settings=['task.success_criterion'],success_criterion='tip_contact_v1',flight_ready=False,
        note='Hit termination and contact-dependent scores can change. Historical forecasts and flags remain unchanged.'))
    atomic_json(AUDIT/'job.json',dict(job=str(job),command=command))
    atomic_json(AUDIT/'preflight.json',dict(status='passed',protected_hashes=protected,
        gate_verification='runs/audits/tip-contact-gate-20260910/verification.json',tests='37 passed, 3 archived-fixture skips',
        settings_changes=['task.success_criterion'],additional_smoke_optimization=False))
    shutil.copy2(prior/'run_and_review.py',AUDIT/'run_and_review.py')
    print(str(job),flush=True)

if __name__=='__main__':main()
