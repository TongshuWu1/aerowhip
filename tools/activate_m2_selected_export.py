"""Activate only the independently verified selected-M2 export; preserve history."""
from pathlib import Path
from datetime import datetime, timezone
import sys
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experimental_data.io import atomic_json, sha256_file
from experimental_data.model_evaluation import load_catalog, save_catalog, model_identity
from experimental_data.whip_adaptation import verify_hashes
from simulator.workflow import read_json


def main():
    job = ROOT / 'runs/adaptation/M2-selected-20260913'
    export = ROOT / 'exports/M2_selected_fixed_tip_reference'
    result = read_json(job / 'fit/result.json')
    comparison = read_json(job / 'comparison.json')
    summary = read_json(export / 'export.json')
    receipt = read_json(export / 'verification.json')
    verify_hashes(result['candidate_hashes'])
    verify_hashes(read_json(job / 'source_hashes.json'))
    assert receipt['csv_sha256'] == summary['csv_sha256'] == sha256_file(export / 'fullstate_30hz.csv')
    for key in ('selected_M2_identity_unchanged', 'original_reference_hash_verified',
                'independent_bspline_derivatives', 'recovery_continuous_PVA',
                'original_slower_brake_settings_unchanged', 'command_envelope_passed', 'offscreen_UI_loaded'):
        assert receipt[key], key
    signature, hashes = model_identity(job / 'candidate/model.json')
    assert signature == model_identity(ROOT / summary['rehearsal'] / 'model.json')[0]
    hashes.update(result['candidate_hashes'])
    hashes[str(job / 'fit/result.json')] = sha256_file(job / 'fit/result.json')
    catalog = load_catalog(ROOT)
    assert not any(m['id'] == 'M2-selected' for m in catalog['models'])
    previous = next(m for m in catalog['models'] if m['id'] == 'M2')
    source = ROOT / 'runs/adaptation/M2-paper-20260913'
    raw_hashes = read_json(source / 'source_hashes.json')
    selection_sources = set()
    for name in comparison['selection_takes']:
        matches = {h for path, h in raw_hashes.items() if Path(path).name == name + '.csv'}
        assert len(matches) == 1, name
        selection_sources.update(matches)
    catalog['models'].append(dict(id='M2-selected', parent='M1', generation_index=2,
        candidate_variant=comparison['selected'], model=str(job / 'candidate/model.json'),
        signature=signature, hashes=hashes, job=str(job),
        training_sources=sorted(set(previous['training_sources']) | selection_sources),
        gradient_training_sources=previous['training_sources'], selection_sources=sorted(selection_sources),
        ancestry_note='training_sources conservatively includes model-selection observations to prevent reuse as independent evidence.',
        status='Selected development model; prospective physical flight pending',
        created_at=datetime.now(timezone.utc).isoformat()))
    previous['status'] = 'Superseded for next flight by M2-selected; original full candidate preserved'
    save_catalog(ROOT, catalog)
    (export / 'flight_take').mkdir(exist_ok=True)
    atomic_json(export / 'model_selection.json', comparison)
    for name in ('reference.json', 'reference.npz'):
        shutil.copy2(ROOT / 'runs/reference_tracking/M0-paper-fixed-reference' / name, export / name)
    atomic_json(export / 'candidate_review.json', dict(model_id='M2-selected', selection_job=str(job),
        source_fit_job=str(source), selected_variant=comparison['selected'],
        evidence='Saved-model selection on existing recordings. No physical M2-selected flight yet.',
        data_use=comparison['data_use'], training_restarted=False,
        original_M0_reference_unchanged=True, direct_measured_bias_anchor=False,
        numerical_and_export_checks=receipt,
        physical_collision_clearance_certified=False))
    (export / 'README.md').write_text(
        '# Selected M2 command for the next physical trial\n\n'
        'Use `fullstate_30hz.csv` with the existing 30 Hz FullState flight program.\n'
        'Launch tracked origin: (0, 0, 1.4) m. Target: (1.25, 0, 1.25) m.\n'
        f'Duration: {summary["total_duration_s"]:.3f} s; {receipt["rows"]} rows. Execute the full recovery and final hold.\n'
        'Save the new raw OptiTrack/controller pairs in `flight_take/`.\n\n'
        'Selected model: updated nominal quadrotor response, retained M1 quadrotor residual,\n'
        'M2 cable physical parameters and numerically verified cable residual checkpoint 100.\n'
        'The regressing M2 quadrotor residual update is excluded. No further fitting ran.\n'
        'The corrected spline starts from the executed M1 command and tracks the same original\n'
        'M0 tip motion and timing, with the existing soft quadrotor-path penalty and slower brake.\n\n'
        'Model selection uses existing M1_003/005 recordings; they are now development data,\n'
        'not independent final evidence. The next physical flights test the corrected command.\n'
        'Offline model, PVA, continuity, complete replay and UI checks passed. These checks\n'
        'do not certify physical cable/propeller clearance or future tracking performance.\n\n'
        f'CSV SHA256: `{summary["csv_sha256"]}`\n', encoding='utf-8')
    replay = read_json(ROOT / 'config/pva/replay.json')
    replay.update(rehearsal=summary['rehearsal'], selected=True, time_s=0.,
        note='M2-selected: retains M1 quadrotor residual; saved-model selection complete. New physical flight pending.')
    atomic_json(ROOT / 'config/pva/replay.json', replay)
    atomic_json(ROOT / 'exports/CURRENT_FLIGHT.json', dict(model_id='M2-selected',
        csv='M2_selected_fixed_tip_reference/fullstate_30hz.csv',
        flight_take='M2_selected_fixed_tip_reference/flight_take', csv_sha256=summary['csv_sha256'],
        rehearsal=summary['rehearsal'], physical_flight_pending=True))
    old_readme = ROOT / 'exports/M2_local_fixed_tip_reference/README.md'
    old = old_readme.read_bytes()
    old_readme.write_bytes(b'> Superseded for the next trial: use `../M2_selected_fixed_tip_reference/fullstate_30hz.csv`. This original export is preserved.\n\n' + old)
    report = ('# M2 saved-component selection and replacement CSV\n\n'
        'No optimizer training updates were run. Five saved combinations were compared using\n'
        'identical causal initialization, masks and time grids on M1_003/005. The rule was\n'
        'lowest equal-take combined tip RMSE subject to quadrotor RMSE no worse than M1.\n'
        'These recordings are now development/model-selection data, not independent final evidence.\n\n'
        '| Saved model | Quadrotor prediction RMSE (cm) | Tip prediction RMSE (cm) |\n'
        '|---|---:|---:|\n')
    for name, row in comparison['metrics'].items():
        report += f'| {name} | {100*row["quadrotor_rmse_mean_m"]:.3f} | {100*row["tip_rmse_mean_m"]:.3f} |\n'
    report += ('\nSelected `retained_drone_M2_cable`: the M1 quadrotor residual tensor weights\n'
        'are exactly retained; nominal quadrotor parameters and the completed M2 cable update\n'
        'are used. Geometry and numerical settings match the original fit contract.\n'
        'This removes the regressing residual update, without claiming overfitting is permanently solved.\n\n'
        'The new command uses the same original M0 physical tip reference and timing, initialized\n'
        'from the executed M1 spline; no measured trajectory bias or new MPPI task was introduced.\n'
        f'Simulated tip-reference RMSE: {100*summary["original_reference_tip_rmse_m"]:.3f} -> '
        f'{100*summary["corrected_reference_tip_rmse_m"]:.3f} cm.\n'
        f'CSV: `exports/M2_selected_fixed_tip_reference/fullstate_30hz.csv`, {receipt["rows"]} rows, '
        f'{summary["total_duration_s"]:.3f} s at 30 Hz.\n'
        'Full independent replay, serialized PVA/joins, hashes and offscreen UI checked on Windows/RTX 4080.\n'
        'Sixteen targeted existing tests passed. No physical M2-selected measurements exist yet.\n'
        'Original M0/M1/M2 artifacts and raw recordings remain intact. Next fit must use\n'
        '`M2-selected` as its actual parent, not the superseded full `M2` candidate.\n')
    doc = ROOT / 'docs/data/M2_SELECTED_20260913.md'; doc.write_text(report, encoding='utf-8')
    print('Activated verified M2-selected CSV:', export / 'fullstate_30hz.csv')


if __name__ == '__main__':
    main()
