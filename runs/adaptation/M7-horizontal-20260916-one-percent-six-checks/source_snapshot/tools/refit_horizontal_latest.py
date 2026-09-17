"""Fit new M7 from the renamed M6 and its eight most recent physical whips."""
from pathlib import Path
from copy import deepcopy
import argparse,sys,shutil,json
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experimental_data import whip_adaptation as data,whip_full_data as full
from experimental_data.whip_full_fit import fit
from experimental_data.io import atomic_json,sha256_file
from experimental_data.model_evaluation import load_catalog,model_identity
from experimental_data.plateau import Plateau
from simulator.workflow import read_json

PACKAGE=ROOT/'runs/adaptation/M6-latest-eight-collection-20260916'
JOB=ROOT/'runs/adaptation/M7-horizontal-20260916-one-percent-six-checks'
REHEARSAL=ROOT/'runs/rehearsals_pva/M6-Horizontal-Command-Correction'

def stopping_check():
    p=Plateau(minimum=0,patience=6,relative=.01)
    p.observe(0,100.)
    for i in range(1,6):assert not p.observe(i,99.5)[1]
    assert p.observe(6,99.5)[1] and p.best==99.5
    p=Plateau(minimum=0,patience=6,relative=.01);p.observe(0,100.)
    for i in range(1,6):p.observe(i,99.5)
    assert not p.observe(6,98.9)[1] and p.stale==0
    return dict(passed=True,rule='Stop after six consecutive checks without improvement greater than 1% from the last meaningful-progress anchor; retain the absolute best weights.')

def prepare():
    torch.set_num_threads(4)
    parent=next(m for m in load_catalog(ROOT)['models'] if m['id']=='M6')
    assert not any(m['id']=='M7' for m in load_catalog(ROOT)['models'])
    assert not PACKAGE.exists() and not JOB.exists()
    assert model_identity(REHEARSAL/'model.json')[0]==parent['signature']
    data.verify_hashes(parent['hashes'])
    report_path=ROOT/'runs/data_review/M6-horizontal-complete-collection/report.json'
    report=read_json(report_path);data.verify_hashes(report['source_hashes'])
    assert report['processed_whips']==8 and report['all_one_second_initializations_passed']
    batch=PACKAGE/'batch';comparison=PACKAGE/'saved_forecast_inputs';prepared=PACKAGE/'whip_inputs'
    data.setup(ROOT,batch,rehearsal=REHEARSAL,command_csv=REHEARSAL/'fullstate_30hz.csv',
               take_count=8,take_prefix='M6_side',all_training=True)
    sources={str(report_path):sha256_file(report_path),**report['source_hashes']};alignment={};groups={}
    for session in (1,2):
        source=ROOT/f'data/flight_batches/M6-horizontal-session{session}-four-whips'
        manifest=read_json(source/'split_manifest.json');data.verify_hashes(manifest['source_hashes'])
        alignment.update(read_json(source/'time_alignment.json'))
        for row in manifest['repetitions']:
            assert row['complete_moving_sequence'] and row['one_second_initialization_passed']
            groups[row['name']]=row['recording_group']
        for file in (source/'flight_take').iterdir():
            if not file.is_file():continue
            dest=batch/'flight_take'/file.name;assert not dest.exists()
            shutil.copy2(file,dest);sources[str(file)]=sha256_file(file)
    assert len(alignment)==8
    atomic_json(batch/'time_alignment.json',alignment)
    p=read_json(batch/'protocol.json')
    p.update(model_id='M6',planned_roles={n:'adaptation' for n in groups},recording_groups=groups,
        actual_take_count=8,command_source='event_log',
        model_update='Warm start from renamed M6 (formerly M7); fit new M7 on these eight flights plus preliminary training replay only.',
        authorization='User explicitly confirmed this initialization and dataset; user states there is no contact or intervention.',
        evaluation_qualification='All eight whips are training data for new M7; post-fit scores are training diagnostics.')
    atomic_json(batch/'protocol.json',p)
    data.compare(ROOT,batch,comparison)
    hashes=read_json(comparison/'source_hashes.json');hashes.update(sources);atomic_json(comparison/'source_hashes.json',hashes)
    review=read_json(comparison/'review.template.json')
    end=float(read_json(REHEARSAL/'rehearsal.json')['whip_end_s'])
    for row in review['takes'].values():
        row.update(role='adaptation',accepted=True,reviewed_by='User flight-condition instructions and event-log/measurement checks',
            clock_reviewed=True,same_controller_and_hardware=True,no_intervention=True,
            physical_contact='none',free_motion_end_s=end,
            notes='Eight complete uploaded whips; original native missing-marker masks retained. Command onset from event logs; clock offset from measured streams, clock_verified=false. No target alignment or spatial normalization. One partial later recovery tail lies outside the scored/trained interval.')
    atomic_json(comparison/'review.json',review)
    data.prepare(ROOT,batch,comparison,comparison/'review.json',prepared,full_model=True)
    prior=read_json(Path(parent['job'])/'protocol.json')['full_update'];contract=full.default_contract()
    for key in contract:
        if key in prior:contract[key]=deepcopy(prior[key])
    contract.update(parent_id='M6',candidate_id='M7',evaluate_before_training=True,
        collection_recording_groups=groups,runner=str(Path(__file__)),runner_sha256=sha256_file(__file__),
        naming='Parent M6 is the former M7. This fit produces new M7. Historical generation index preserves the actual update count.',
        stopping_definition=stopping_check()['rule'])
    contract['nominal_stopping'].update(patience=6,relative=.01)
    contract['residual_stopping'].update(patience=6,relative=.01,ceiling=None)
    atomic_json(PACKAGE/'stopping_verification.json',stopping_check())
    atomic_json(PACKAGE/'standard_update_contract.json',contract)
    full.prepare(JOB,prepared,ROOT/'runs/adaptation/20260909-preliminary1-M0-v2',contract)
    assert not read_json(JOB/'protocol.json')['prior_whip_replay']
    print('Prepared:',JOB,flush=True)
    print(json.dumps({k:contract[k] for k in ('parent_id','candidate_id','nominal_stopping','residual_stopping','cable_log_difference_step')},indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','fit']);args=parser.parse_args()
    if args.stage=='prepare':prepare()
    else:
        assert read_json(JOB/'protocol.json')['full_update']['runner_sha256']==sha256_file(__file__)
        print(json.dumps(fit(JOB),indent=2),flush=True)
