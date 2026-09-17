"""Verify/export the saved horizontal correction with standard production physics."""
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from simulator.workflow import read_json
from experimental_data.io import atomic_json,sha256_file
from experimental_data.whip_adaptation import verify_hashes
from experimental_data.model_evaluation import load_catalog,model_identity
from planning.correction_job import run


def main():
    old=ROOT/'runs/reference_tracking/M6-horizontal-command-correction'
    reference=ROOT/'runs/reference_tracking/M5-horizontal-fixed-reference'
    job=ROOT/'runs/reference_tracking/M6-horizontal-command-correction-production'
    output=ROOT/'runs/rehearsals_pva/M6-Horizontal-Command-Correction'
    export=ROOT/'exports/M6_Horizontal_Command_Correction'
    model=next(m for m in load_catalog(ROOT)['models'] if m['id']=='M6')
    meta=read_json(reference/'reference.json')
    controls_hash=sha256_file(old/'current_best.npz')
    run(Path(model['job']),reference,job,output,export,iterations=30,
        strike_guard='reference',replay_controls_from=old)
    result=read_json(job/'result.json')
    assert sha256_file(old/'current_best.npz')==controls_hash
    with np.load(old/'current_best.npz') as a, np.load(job/'plan.npz') as b:
        np.testing.assert_array_equal(a['jerk_correction_coefficients_m_s3'],b['jerk_correction_coefficients_m_s3'])
    assert result['command_parameterization']=='additive_jerk_correction'
    assert result['strike_time_s']==meta['planned_strike_time_s']
    assert result['corrected_reference_tip_rmse_m']<result['original_reference_tip_rmse_m']
    assert result['optimization']['final_strike']['reference_error_m']<=result['optimization']['baseline_strike']['reference_error_m']+1e-10
    assert model_identity(output/'model.json')[0]==model['signature']
    verify_hashes({str(ROOT/k):h for k,h in meta['source_hashes'].items()})
    verify_hashes(read_json(job/'source_hashes.json'))
    csv=np.loadtxt(export/'fullstate_30hz.csv',delimiter=',',skiprows=1)
    assert np.isfinite(csv).all() and csv.shape[1]==12
    np.testing.assert_allclose(np.diff(csv[:,0]),1/30,rtol=0,atol=1e-10)
    np.testing.assert_allclose(csv[0,1:4],meta['launch_origin_m'],rtol=0,atol=1e-10)
    np.testing.assert_allclose(csv[-1,1:4],meta['launch_origin_m'],rtol=0,atol=1e-10)
    np.testing.assert_allclose(csv[-1,4:],0,rtol=0,atol=1e-10)
    assert sha256_file(export/'fullstate_30hz.csv')==result['csv_sha256']
    (export/'flight_take/M6').mkdir(parents=True)
    record=read_json(export/'export.json');record.update(model_id='M6',task_family='horizontal_curved_side',
        operation='fixed_reference_command_correction_not_mppi_replanning',source_reference=str(reference),
        recording_directory=str(export/'flight_take/M6'),active_flight_selection_changed=False,
        final_replay_backend='standard production',optimization_convergence_verified=False)
    atomic_json(export/'export.json',record)
    (export/'README.md').write_text(
        '# M6 horizontal whip — corrected command\n\n'
        '**Command correction only; no MPPI re-planning.** Uses M6 to track the unchanged saved M5 horizontal tip trajectory and timestamps.\n\n'
        f'- Required starting tracked-origin hover: {tuple(meta["launch_origin_m"])} m.\n'
        f'- Fixed reference strike point: {tuple(meta["physical_target_m"])} m; time {meta["planned_strike_time_s"]:.9f} s.\n'
        f'- Complete 30 Hz CSV: {len(csv)} rows, duration {csv[-1,0]:.3f} s.\n'
        '- Use the existing FullState controller and lab safety procedure. First row is the required hover, not a takeoff command.\n'
        '- Execute the complete CSV, including the model-checked recovery and final hold.\n'
        '- Record this horizontal task separately under `flight_take/M6`.\n'
        '- The original M5 export and active flight selection are unchanged.\n\n'
        f'Predicted tip-reference RMSE: {100*result["original_reference_tip_rmse_m"]:.3f} → {100*result["corrected_reference_tip_rmse_m"]:.3f} cm.\n'
        f'Predicted error at fixed strike time: {100*result["optimization"]["baseline_strike"]["reference_error_m"]:.3f} → {100*result["optimization"]["final_strike"]["reference_error_m"]:.3f} cm.\n\n'
        'The best accepted local correction is exported. The optimizer stopped after exhausting its trust radius; formal convergence is unverified. '
        'An accelerated-vs-standard long-horizon replay mismatch blocked the first export. This export retains identical corrected controls and uses standard production physics throughout; no tolerance was relaxed. '
        'Physical performance remains unvalidated.\n\n'
        f'CSV SHA-256: `{result["csv_sha256"]}`\n\nSource rehearsal: `{output}`\n',encoding='utf-8')
    atomic_json(export/'verification.json',dict(original_sources_verified=True,saved_controls_unchanged=True,
        csv_sha256=result['csv_sha256'],rows=len(csv),rate_hz=30,
        reference_sha256=result['reference_sha256'],model_signature=model['signature'],
        standard_complete_cable_max_difference_m=result['standard_complete_cable_max_difference_m'],
        standard_complete_pose_max_difference_m=result['standard_complete_pose_max_difference_m'],
        independent_cost_difference_m2=result['independent_cost_difference_m2'],
        original_failed_check=read_json(old/'status.json')))
    print('VERIFIED CORRECTED CSV',export/'fullstate_30hz.csv',flush=True)


if __name__=='__main__':main()
