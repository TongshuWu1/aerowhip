"""Prepare the user-confirmed eight M5 side whips for the M6 update."""
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experimental_data import whip_adaptation as data, whip_full_data as full
from experimental_data.adaptation_rounds import read_optitrack
from experimental_data.io import atomic_json, sha256_file
from experimental_data.model_evaluation import load_catalog, model_identity
from simulator.workflow import read_json


def main():
    torch.set_num_threads(4)
    package = ROOT / 'runs/adaptation/M5-side-collection-20260915'
    batch = package / 'batch'
    comparison = package / 'saved_forecast_inputs'
    prepared = package / 'whip_inputs'
    job = ROOT / 'runs/adaptation/M6-side-20260915'
    rehearsal = ROOT / 'runs/rehearsals_pva/M5-Curved-Side-Whip-1p3m'
    catalog = load_catalog(ROOT)
    parent = next(m for m in catalog['models'] if m['id'] == 'M5')
    assert not any(m['id'] == 'M6' for m in catalog['models'])
    assert not package.exists() and not job.exists()
    assert model_identity(rehearsal / 'model.json')[0] == parent['signature']
    data.verify_hashes(parent['hashes'])
    report_path = ROOT / 'runs/data_review/M5-curved-side-complete-collection/report.json'
    report = read_json(report_path)
    data.verify_hashes(report['source_hashes'])
    assert report['processed_whips'] == 8
    assert report['all_one_second_initializations_passed']
    end = float(read_json(rehearsal / 'rehearsal.json')['whip_end_s'])
    sources = {str(report_path): sha256_file(report_path)}
    sources.update(report['source_hashes'])
    alignment, groups = {}, {}
    data.setup(ROOT, batch, rehearsal=rehearsal,
               command_csv=rehearsal / 'fullstate_30hz.csv',
               take_count=8, take_prefix='M5_side', all_training=True)
    for session in (1, 2):
        source = ROOT / f'data/flight_batches/M5-curved-side-session{session}-four-whips'
        manifest = read_json(source / 'split_manifest.json')
        data.verify_hashes(manifest['source_hashes'])
        alignment.update(read_json(source / 'time_alignment.json'))
        for row in manifest['repetitions']:
            name = row['name']
            assert row['complete_moving_sequence'] and row['one_second_initialization_passed']
            m = read_optitrack(source / 'flight_take' / (name + '.csv'))
            relative = m['time'] + row['offset_s'] - row['onset_s']
            interval = m['time'][(relative >= 0) & (relative <= end)]
            assert len(interval) >= 153
            assert np.allclose(np.diff(interval), .01, atol=1e-7, rtol=0)
            groups[name] = row['recording_group']
        for file in (source / 'flight_take').iterdir():
            assert file.is_file()
            destination = batch / 'flight_take' / file.name
            assert not destination.exists()
            shutil.copy2(file, destination)
            sources[str(file)] = sha256_file(file)
    assert len(alignment) == 8
    atomic_json(batch / 'time_alignment.json', alignment)
    p = read_json(batch / 'protocol.json')
    p.update(model_id='M5', planned_roles={n: 'adaptation' for n in groups},
             recording_groups=groups, actual_take_count=8, command_source='event_log',
             model_update='Full model warm start from M5; equal-marker cable loss; preliminary training replay',
             authorization='User requested fitting and confirmed unchanged setup and no contact/intervention during all eight 0–1.533 s intervals.',
             evaluation_qualification='All eight whips train M6; post-fit scores are training diagnostics.')
    atomic_json(batch / 'protocol.json', p)
    data.compare(ROOT, batch, comparison)
    hashes = read_json(comparison / 'source_hashes.json')
    hashes.update(sources)
    atomic_json(comparison / 'source_hashes.json', hashes)
    review = read_json(comparison / 'review.template.json')
    for row in review['takes'].values():
        row.update(role='adaptation', accepted=True,
                   reviewed_by='User confirmation of flight conditions; Codex recording and alignment checks',
                   clock_reviewed=True, same_controller_and_hardware=True,
                   no_intervention=True, physical_contact='none', free_motion_end_s=end,
                   notes='User confirmed unchanged drone/controller/cable setup and no contact or intervention during the whip. Timing is estimated from measured streams, with clock_verified=false. Command-event logs supply inputs; native Motive supplies observations. Earlier/later physical contact and between-whip resets are not inferred.')
    atomic_json(comparison / 'review.json', review)
    data.prepare(ROOT, batch, comparison, comparison / 'review.json', prepared, full_model=True)
    contract = full.default_contract()
    contract.update(parent_id='M5', candidate_id='M6', evaluate_before_training=True,
                    collection_recording_groups=groups)
    previous = read_json(ROOT / 'runs/adaptation/M5-20260915/protocol.json')['full_update']
    for key in ('nominal_stopping', 'residual_stopping', 'replay_weight',
                'cable_residual', 'cable_tip_weight'):
        assert contract[key] == previous[key]
    atomic_json(package / 'standard_update_contract.json', contract)
    full.prepare(job, prepared, ROOT / 'runs/adaptation/20260909-preliminary1-M0-v2', contract)
    print('Prepared M6 from M5 with eight side whips and preliminary training replay.', flush=True)


if __name__ == '__main__':
    main()
