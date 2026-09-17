"""Collect M6 horizontal recordings, then prepare a reviewed M7 full-model update."""
from pathlib import Path
from copy import deepcopy
import argparse
import shutil
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experimental_data import whip_adaptation as data, whip_full_data as full
from experimental_data.io import atomic_json,sha256_file
from experimental_data.model_evaluation import load_catalog,model_identity
from simulator.workflow import read_json

PACKAGE=ROOT/'runs/adaptation/M6-horizontal-collection-20260915'
JOB=ROOT/'runs/adaptation/M7-horizontal-20260915'
REHEARSAL=ROOT/'runs/rehearsals_pva/M6-Horizontal-Command-Correction'

def collect():
    parent=next(m for m in load_catalog(ROOT)['models'] if m['id']=='M6')
    assert not any(m['id']=='M7' for m in load_catalog(ROOT)['models'])
    assert not PACKAGE.exists() and not JOB.exists()
    assert model_identity(REHEARSAL/'model.json')[0]==parent['signature']
    data.verify_hashes(parent['hashes'])
    report_path=ROOT/'runs/data_review/M6-horizontal-complete-collection/report.json'
    report=read_json(report_path)
    data.verify_hashes(report['source_hashes'])
    assert report['processed_whips']==12 and report['all_one_second_initializations_passed']
    batch=PACKAGE/'batch'; comparison=PACKAGE/'saved_forecast_inputs'
    data.setup(ROOT,batch,rehearsal=REHEARSAL,command_csv=REHEARSAL/'fullstate_30hz.csv',
               take_count=12,take_prefix='M6_side',all_training=True)
    sources={str(report_path):sha256_file(report_path),**report['source_hashes']}
    alignment,groups={},{}
    for session in (1,2,3):
        source=ROOT/f'data/flight_batches/M6-horizontal-session{session}-four-whips'
        manifest=read_json(source/'split_manifest.json')
        data.verify_hashes(manifest['source_hashes'])
        alignment.update(read_json(source/'time_alignment.json'))
        for row in manifest['repetitions']:
            assert row['complete_moving_sequence'] and row['one_second_initialization_passed']
            groups[row['name']]=row['recording_group']
        for file in (source/'flight_take').iterdir():
            assert file.is_file()
            dest=batch/'flight_take'/file.name
            assert not dest.exists()
            shutil.copy2(file,dest);sources[str(file)]=sha256_file(file)
    assert len(alignment)==12
    atomic_json(batch/'time_alignment.json',alignment)
    p=read_json(batch/'protocol.json')
    p.update(model_id='M6',planned_roles={n:'adaptation' for n in groups},recording_groups=groups,
             actual_take_count=12,command_source='event_log',
             model_update='Full model warm start from M6 with preliminary training replay; no previous whipping batches replayed',
             evaluation_qualification='All twelve whips train M7; post-fit scores are training diagnostics.')
    atomic_json(batch/'protocol.json',p)
    data.compare(ROOT,batch,comparison)
    hashes=read_json(comparison/'source_hashes.json');hashes.update(sources)
    atomic_json(comparison/'source_hashes.json',hashes)
    print('Collected twelve M6 whips; no fitting or condition assumptions.',flush=True)

def prepare(conditions):
    conditions=Path(conditions).resolve();facts=read_json(conditions)
    assert facts['unchanged_setup'] is True
    assert facts['no_contact_or_intervention_during_whip'] is True
    assert facts['no_aborted_whips'] is True
    assert facts['source']=='user_confirmation'
    batch=PACKAGE/'batch';comparison=PACKAGE/'saved_forecast_inputs';prepared=PACKAGE/'whip_inputs'
    end=float(read_json(REHEARSAL/'rehearsal.json')['whip_end_s'])
    review=read_json(comparison/'review.template.json')
    for row in review['takes'].values():
        row.update(role='adaptation',accepted=True,
            reviewed_by='User flight-condition confirmation plus recorded-command and measured-stream checks',
            clock_reviewed=True,same_controller_and_hardware=True,no_intervention=True,
            physical_contact='none',free_motion_end_s=end,
            notes='Full event-log command sequence verified. Clock offset estimated from measured streams; clock_verified=false. Native missing-marker masks retained. No target-based alignment or spatial normalization.')
    review['operator_conditions']=facts
    atomic_json(comparison/'review.json',review)
    hashes=read_json(comparison/'source_hashes.json');hashes[str(conditions)]=sha256_file(conditions)
    atomic_json(comparison/'source_hashes.json',hashes)
    data.prepare(ROOT,batch,comparison,comparison/'review.json',prepared,full_model=True)
    parent=next(m for m in load_catalog(ROOT)['models'] if m['id']=='M6')
    previous=read_json(Path(parent['job'])/'protocol.json')['full_update']
    contract=full.default_contract()
    for key in contract:
        if key in previous:contract[key]=deepcopy(previous[key])
    contract['cable_residual_stopping']=deepcopy(previous['cable_residual_stopping'])
    contract.update(parent_id='M6',candidate_id='M7',evaluate_before_training=True,
        collection_recording_groups=read_json(batch/'protocol.json')['recording_groups'],
        runner='tools/fit_m7_horizontal.py',runner_sha256=sha256_file(ROOT/'tools/fit_m7_horizontal.py'))
    atomic_json(PACKAGE/'standard_update_contract.json',contract)
    full.prepare(JOB,prepared,ROOT/'runs/adaptation/20260909-preliminary1-M0-v2',contract)
    print('Prepared M7: M6 initialization, twelve new whips and preliminary training data.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['collect','prepare']);parser.add_argument('--conditions',type=Path)
    args=parser.parse_args();torch.set_num_threads(4)
    if args.stage=='collect':collect()
    else:
        if args.conditions is None:parser.error('--conditions is required for prepare')
        prepare(args.conditions)
