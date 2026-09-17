"""Activate the verified fixed-parameter cable-residual-only M2 trial."""
from pathlib import Path
from datetime import datetime,timezone
import sys
import shutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experimental_data.io import atomic_json,sha256_file
from experimental_data.whip_adaptation import verify_hashes
from experimental_data.model_evaluation import load_catalog,save_catalog,model_identity
from simulator.workflow import read_json


def main():
    job=ROOT/'runs/adaptation/M2-cable-only-20260913'
    if (job/'ARCHIVED').exists():
        raise ValueError('Cable-only activation was cancelled by the user; current flight retains the drone residual.')
    export=ROOT/'exports/M2_cable_only_fixed_tip_reference'
    summary=read_json(export/'export.json');receipt=read_json(export/'verification.json')
    comparison=read_json(job/'comparison.json');fit=read_json(job/'fit/result.json')
    verify_hashes(fit['candidate_hashes']);verify_hashes(read_json(job/'source_hashes.json'))
    assert receipt['csv_sha256']==summary['csv_sha256']==sha256_file(export/'fullstate_30hz.csv')
    for k in ('cable_only_M2_identity_unchanged','drone_residual_exactly_zero','cable_residual_enabled',
              'independent_bspline_derivatives','recovery_continuous_PVA','offscreen_UI_loaded'):
        assert receipt[k],k
    signature,hashes=model_identity(job/'candidate/model.json');hashes.update(fit['candidate_hashes'])
    hashes[str(job/'fit/result.json')]=sha256_file(job/'fit/result.json')
    assert signature==model_identity(ROOT/summary['rehearsal']/'model.json')[0]
    catalog=load_catalog(ROOT)
    assert not any(m['id']=='M2-cable-only' for m in catalog['models'])
    previous=next(m for m in catalog['models'] if m['id']=='M2-selected')
    catalog['models'].append(dict(id='M2-cable-only',parent='M1',generation_index=2,
        candidate_variant='cable_residual_only',model=str(job/'candidate/model.json'),signature=signature,
        hashes=hashes,job=str(job),training_sources=previous['training_sources'],
        gradient_training_sources=previous['gradient_training_sources'],selection_sources=previous['selection_sources'],
        ablation_source='M2-selected',status='User-selected cable-only residual ablation; physical flight pending',
        created_at=datetime.now(timezone.utc).isoformat()))
    previous['status']='Preserved comparison; next user-requested flight uses M2-cable-only'
    save_catalog(ROOT,catalog)
    (export/'flight_take').mkdir(exist_ok=True)
    atomic_json(export/'model_comparison.json',comparison)
    for name in ('reference.json','reference.npz'):
        shutil.copy2(ROOT/'runs/reference_tracking/M0-paper-fixed-reference'/name,export/name)
    atomic_json(export/'candidate_review.json',dict(model_id='M2-cable-only',source=str(job),
        drone_residual_enabled=False,cable_residual_enabled=True,nominal_parameters_refitted=False,
        training_restarted=False,physical_performance_pending=True,retrospective_comparison=comparison['cable_only'],
        comparison_note='Compared with M2-selected: lower quadrotor prediction error, higher tip prediction error. No overall improvement claim.',
        numerical_and_export_checks=receipt,physical_collision_clearance_certified=False))
    (export/'README.md').write_text(
        '# M2 with cable residual only\n\n'
        'Use `fullstate_30hz.csv` with the existing 30 Hz FullState flight program.\n'
        'Launch tracked origin: (0, 0, 1.4) m. Target: (1.25, 0, 1.25) m.\n'
        f'{receipt["rows"]} rows; {summary["total_duration_s"]:.3f} s. Execute the complete slower brake, return and final hold.\n'
        'Save raw OptiTrack/controller pairs in `flight_take/`.\n\n'
        'The quadrotor residual is disabled. Existing M2-selected nominal parameters,\n'
        'M2 cable parameters and cable residual checkpoint 100 are unchanged. No refitting ran.\n'
        'Compatibility packaging uses a zero-weight quadrotor checkpoint: exact equivalence\n'
        'to the nominal equations without a residual was verified on both comparison takes.\n\n'
        'The B-spline correction starts from executed M1 commands and targets the same original\n'
        'M0 physical tip trajectory and timing with the existing soft quadrotor-path penalty.\n'
        'Compared on M1_003/005, quadrotor prediction RMSE changes 6.72 to 5.79 cm;\n'
        'tip prediction RMSE changes 8.46 to 9.53 cm. These are development predictions\n'
        'against recorded flights, not performance of this new CSV. Physical testing is pending.\n'
        'The offline checks do not certify physical cable/propeller clearance.\n\n'
        f'CSV SHA256: `{summary["csv_sha256"]}`\n',encoding='utf-8')
    replay=read_json(ROOT/'config/pva/replay.json')
    replay.update(rehearsal=summary['rehearsal'],selected=True,time_s=0.,
        note='M2-cable-only: drone residual disabled; existing fitted parameters and cable residual retained. Physical test pending.')
    atomic_json(ROOT/'config/pva/replay.json',replay)
    atomic_json(ROOT/'exports/CURRENT_FLIGHT.json',dict(model_id='M2-cable-only',
        csv='M2_cable_only_fixed_tip_reference/fullstate_30hz.csv',
        flight_take='M2_cable_only_fixed_tip_reference/flight_take',csv_sha256=summary['csv_sha256'],
        rehearsal=summary['rehearsal'],physical_flight_pending=True))
    old=ROOT/'exports/M2_selected_fixed_tip_reference/README.md'
    old.write_bytes(b'> Superseded for the next user-requested trial: use `../M2_cable_only_fixed_tip_reference/fullstate_30hz.csv`. This comparison is preserved.\n\n'+old.read_bytes())
    atomic_json(ROOT/'docs/data/M2_CABLE_ONLY_20260913.json',dict(comparison=comparison,export=summary,verification=receipt,
        next_fit_parent='M2-cable-only',no_refitting=True,hardware='Windows / RTX 4080'))
    print('Activated:',export/'fullstate_30hz.csv')


if __name__=='__main__':main()
