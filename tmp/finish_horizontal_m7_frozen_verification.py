"""Finish post-fit checks using the run's exact snapshot after unrelated live edits."""
from pathlib import Path
import sys,time,json
ROOT=Path(__file__).resolve().parents[1]
JOB=ROOT/'runs/adaptation/M7-horizontal-20260916-one-percent-six-checks'
SNAPSHOT=JOB/'source_snapshot'
sys.path.insert(0,str(SNAPSHOT))
import torch
from experimental_data import whip_full_data as data,whip_full_fit as full
from experimental_data.whip_adaptation import verify_hashes
from experimental_data.io import atomic_json,sha256_file
from experimental_data.model_evaluation import model_identity
from simulator.workflow import read_json

def main():
    torch.set_num_threads(4)
    assert Path(full.__file__).is_relative_to(SNAPSHOT)
    status=read_json(JOB/'status.json')
    assert status['status']=='failed' and 'Source changed:' in status['error'],status
    assert 'tools' in status['error'] and 'build_fig1_photographic.py' in status['error'],status
    selection=read_json(JOB/'fit/selection_frozen.json')
    assert selection['stages_complete']
    assert sha256_file(JOB/'candidate/model.json')==selection['candidate_sha256']
    for stage in ('drone_residual','cable_residual'):
        assert selection[stage]['stop_reason']=='practical_plateau'
        assert selection[stage]['numerically_verified']
    code=read_json(JOB/'code_hashes.json')
    verify_hashes({str(SNAPSHOT/n):h for n,h in code.items()})
    # Algorithm modules must also still match the versions used at optimization start.
    verify_hashes({str(ROOT/n):h for n,h in code.items()
                   if Path(n).parts[0] in ('experimental_data','simulator','planning','learning') and 'gui' not in Path(n).parts})
    atomic_json(JOB/'verification_resume.json',dict(original_status=status,
        reason='An unrelated paper-figure builder changed in the live workspace. No optimizer state or numerical acceptance criterion changes; final verification uses the exact frozen source.',
        source=str(SNAPSHOT),source_hashes=code,optimization_repeated=False))
    c=read_json(JOB/'protocol.json')['full_update']
    full.validate_candidate(JOB/'candidate/model.json',c)
    models={'M6':JOB/'source_candidate/model.json','M7':JOB/'candidate/model.json'}
    output=ROOT/'runs/evaluation'/f'{JOB.name}-frozen-verification'
    full.progress(JOB,'combined_validation using frozen source snapshot')
    full.evaluate_pair(JOB,output,models)
    full.evaluate_preliminary_retention(JOB,models,c)
    full.evaluate_prior_retention(JOB,models)
    _,hashes=model_identity(JOB/'candidate/model.json')
    hashes.update(read_json(JOB/'before_update/evidence_hashes.json'))
    for p in (JOB/'before_update_ancestry.json',JOB/'fit/selection_frozen.json',output/'report.json',
              JOB/'preliminary_validation/report.json',JOB/'prior_validation/report.json',JOB/'verification_resume.json',Path(__file__)):
        hashes[str(p)]=sha256_file(p)
    result=dict(status='completed',schema=data.FULL_SCHEMA,training_takes=selection['training_takes'],candidate_hashes=hashes,
        stages=selection,comparison=str(output),elapsed_s=time.time()-(JOB/'protocol.json').stat().st_mtime,
        model_selected=False,completion_mode='Frozen-source post-fit verification after unrelated live workspace edit')
    atomic_json(JOB/'fit/result.json',result)
    data.ROOT=ROOT
    registration=full.register(JOB,output)
    atomic_json(JOB/'registration.json',registration)
    atomic_json(JOB/'status.json',dict(status='completed',stage='All adaptation stages and frozen-source validation complete',
        registration=registration,model_selected=False,comparison=str(output)))
    print('New M7 registered after frozen-source validation:',registration,flush=True)

if __name__=='__main__':main()
