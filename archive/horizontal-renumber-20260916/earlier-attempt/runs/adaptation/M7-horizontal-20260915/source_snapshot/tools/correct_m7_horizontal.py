"""Correct from the executed M6 command while preserving the M5 task trajectory."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experimental_data.io import atomic_json,sha256_file
from experimental_data.model_evaluation import load_catalog,model_identity
from experimental_data.whip_adaptation import verify_hashes
from simulator.workflow import read_json
from planning.correction_job import run

def main():
    source=ROOT/'runs/adaptation/M7-horizontal-20260915'
    reference=ROOT/'runs/reference_tracking/M5-horizontal-fixed-reference'
    previous=ROOT/'runs/rehearsals_pva/M6-Horizontal-Command-Correction'
    job=ROOT/'runs/reference_tracking/M7-horizontal-command-correction'
    output=ROOT/'runs/rehearsals_pva/M7-Horizontal-Command-Correction'
    export=ROOT/'exports/M7_Horizontal_Command_Correction'
    parent=next(m for m in load_catalog(ROOT)['models'] if m['id']=='M7')
    verify_hashes(parent['hashes'])
    run(source,reference,job,output,export,previous_rehearsal=previous,iterations=30,strike_guard='reference')
    verify(job,output,export,reference,parent)

def verify(job,output,export,reference,parent):
    meta=read_json(reference/'reference.json');result=read_json(job/'result.json')
    assert result['command_parameterization']=='additive_jerk_correction'
    assert result['strike_time_s']==meta['planned_strike_time_s']
    assert result['corrected_reference_tip_rmse_m']<result['original_reference_tip_rmse_m']
    assert result['optimization']['final_strike']['reference_error_m']<=result['optimization']['baseline_strike']['reference_error_m']+1e-10
    assert model_identity(output/'model.json')[0]==parent['signature']
    verify_hashes(read_json(job/'source_hashes.json'))
    csv=np.loadtxt(export/'fullstate_30hz.csv',delimiter=',',skiprows=1)
    assert np.isfinite(csv).all() and csv.shape[1]==12
    np.testing.assert_allclose(np.diff(csv[:,0]),1/30,rtol=0,atol=1e-10)
    np.testing.assert_allclose(csv[0,1:4],meta['launch_origin_m'],rtol=0,atol=1e-10)
    np.testing.assert_allclose(csv[-1,1:4],meta['launch_origin_m'],rtol=0,atol=1e-10)
    np.testing.assert_allclose(csv[-1,4:],0,rtol=0,atol=1e-10)
    assert sha256_file(export/'fullstate_30hz.csv')==result['csv_sha256']
    (export/'flight_take/M7').mkdir(parents=True,exist_ok=True)
    record=read_json(export/'export.json');record.update(model_id='M7',task_family='horizontal_curved_side',
        operation='fixed_reference_command_correction_not_mppi_replanning',source_reference=str(reference),
        previous_executed_command='M6-Horizontal-Command-Correction',
        recording_directory=str(export/'flight_take/M7'),active_flight_selection_changed=False)
    atomic_json(export/'export.json',record)
    atomic_json(export/'verification.json',dict(csv_sha256=result['csv_sha256'],rows=len(csv),rate_hz=30,
        reference_sha256=result['reference_sha256'],model_signature=parent['signature'],
        start_and_final_hold_verified=True,source_hashes_verified=True,
        independent_cost_difference_m2=result['independent_cost_difference_m2']))
    (export/'README.md').write_text(
        '# M7 horizontal command correction\n\n'
        'M7 is refitted from M6 using the twelve new horizontal whips plus preliminary training data. '
        'The correction starts from the executed M6 command and preserves the original M5 horizontal desired trajectory and timing.\n\n'
        f'- Required starting tracked-origin hover: {tuple(meta["launch_origin_m"])} m.\n'
        f'- Fixed planned strike time: {meta["planned_strike_time_s"]:.9f} s.\n'
        f'- Complete 30 Hz CSV: {len(csv)} rows, {csv[-1,0]:.3f} s, including recovery and final hold.\n'
        '- Use the existing FullState controller and established lab procedure; first row is the required hover, not takeoff.\n'
        '- Save new recordings under `flight_take/M7`. Active flight selection is unchanged.\n\n'
        f'Predicted tip-reference RMSE under M7: {100*result["original_reference_tip_rmse_m"]:.3f} -> {100*result["corrected_reference_tip_rmse_m"]:.3f} cm. '
        'These are simulation results; M7 physical performance awaits the next flights.\n\n'
        f'CSV SHA-256: `{result["csv_sha256"]}`\n',encoding='utf-8')
    print('VERIFIED M7 CSV',export/'fullstate_30hz.csv',flush=True)

if __name__=='__main__':main()
