"""Freeze the executed M5 horizontal motion and correct commands with M6 only."""
from pathlib import Path
import shutil
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experimental_data.io import atomic_json,sha256_file
from experimental_data.whip_adaptation import verify_hashes
from experimental_data.model_evaluation import load_catalog,model_identity
from simulator.workflow import read_json
from planning.jerk_reference_correction import JerkCorrection
from planning.correction_job import run


def main():
    previous=ROOT/'runs/rehearsals_pva/M5-Curved-Side-Whip-1p3m'
    reference=ROOT/'runs/reference_tracking/M5-horizontal-fixed-reference'
    job=ROOT/'runs/reference_tracking/M6-horizontal-command-correction'
    output=ROOT/'runs/rehearsals_pva/M6-Horizontal-Command-Correction'
    export=ROOT/'exports/M6_Horizontal_Command_Correction'
    if any(p.exists() for p in (reference,job,output,export)):
        raise FileExistsError('Use new artifact paths; do not overwrite previous work')
    model=next(m for m in load_catalog(ROOT)['models'] if m['id']=='M6')
    verify_hashes(model['hashes'])
    cfg=read_json(previous/'settings.json');meta=read_json(previous/'rehearsal.json')
    with np.load(previous/'rehearsal.npz') as z:a={k:z[k].copy() for k in z.files}
    with np.load(previous/'plan.npz') as z:jerk=z['normalized_jerk']*np.array(cfg['action']['jerk_limit_m_s3'])
    end=meta['whip_end_s'];strike=meta['strike_time_s'];count=round(end*30)+1
    adapter=JerkCorrection(a['commands'][0],jerk)
    packets,_=adapter.decode(torch.zeros((9,3),dtype=torch.float64),cfg['launch']['origin_m'])
    np.testing.assert_allclose(packets.numpy(),a['commands'][:count],rtol=0,atol=1e-10)
    csv=np.loadtxt(previous/'fullstate_30hz.csv',delimiter=',',skiprows=1)
    np.testing.assert_allclose(csv,np.c_[a['command_time_s'],a['commands']],rtol=0,atol=1e-12)
    keep=a['prediction_time_s']<=end+1e-10
    assert abs(a['prediction_time_s'][keep][-1]-end)<1e-10
    assert 0<strike<end and len(jerk)==count-1
    sources=[previous/n for n in ('model.json','plan.npz','rehearsal.npz','settings.json','rehearsal.json','fullstate_30hz.csv')]
    sources.append(Path(__file__).resolve())
    active=ROOT/'config/pva/flight_selection.json'
    if active.exists():sources.append(active)
    hashes={f.relative_to(ROOT).as_posix():sha256_file(f) for f in sources}
    reference.mkdir(parents=True)
    np.savez_compressed(reference/'reference.npz',time_s=a['prediction_time_s'][keep],
        tip_position_m=a['cable_positions_m'][keep,-1],quadrotor_position_m=a['origin_positions_m'][keep],
        cable_position_m=a['cable_positions_m'][keep],command_time_s=a['command_time_s'][:count],
        original_command_packets=a['commands'][:count],original_jerk_m_s3=jerk,
        target_position_m=a['target_position_m'])
    atomic_json(reference/'reference.json',dict(schema='frozen_physical_motion_reference_v1',reference_label='M5 horizontal',
        source_rehearsal=previous.relative_to(ROOT).as_posix(),source_hashes=hashes,
        reference_sha256=sha256_file(reference/'reference.npz'),
        physical_target_m=a['target_position_m'].tolist(),physical_target_present=False,
        launch_origin_m=cfg['launch']['origin_m'],interval_s=[0.,end],planned_strike_time_s=strike,
        reference_kind='Exact saved M5 horizontal forecast; no new simulation, time warping, or reference optimization.',
        task_family='horizontal_curved_side',release_height_m=float(a['target_position_m'][2])))
    # Existing local tracking optimizer, not MPPI. The strike guard also prevents
    # worsening the starting fixed-time reference error while reducing path cost.
    run(Path(model['job']),reference,job,output,export,iterations=30,strike_guard='reference')
    result=read_json(job/'result.json')
    assert result['command_parameterization']=='additive_jerk_correction'
    assert abs(result['strike_time_s']-strike)<1e-12
    assert result['corrected_reference_tip_rmse_m']<result['original_reference_tip_rmse_m']
    assert result['optimization']['final_strike']['reference_error_m']<=result['optimization']['baseline_strike']['reference_error_m']+1e-10
    assert model_identity(output/'model.json')[0]==model['signature']
    verify_hashes({str(ROOT/k):h for k,h in hashes.items()})
    csv=np.loadtxt(export/'fullstate_30hz.csv',delimiter=',',skiprows=1)
    assert np.isfinite(csv).all() and csv.shape[1]==12
    np.testing.assert_allclose(np.diff(csv[:,0]),1/30,rtol=0,atol=1e-10)
    np.testing.assert_allclose(csv[0,1:4],cfg['launch']['origin_m'],rtol=0,atol=1e-10)
    np.testing.assert_allclose(csv[-1,1:4],cfg['launch']['origin_m'],rtol=0,atol=1e-10)
    np.testing.assert_allclose(csv[-1,4:],0,rtol=0,atol=1e-10)
    assert sha256_file(export/'fullstate_30hz.csv')==result['csv_sha256']
    (export/'flight_take/M6').mkdir(parents=True)
    record=read_json(export/'export.json');record.update(model_id='M6',task_family='horizontal_curved_side',
        operation='fixed_reference_command_correction_not_mppi_replanning',source_reference=str(reference),
        recording_directory=str(export/'flight_take/M6'),active_flight_selection_changed=False)
    atomic_json(export/'export.json',record)
    (export/'README.md').write_text(
        '# M6 horizontal whip — fixed-reference command correction\n\n'
        '**This is command correction, not MPPI re-planning.** M6 is used to track the unchanged saved M5 horizontal tip trajectory at its original timestamps.\n\n'
        f'- Starting tracked-origin hover: {tuple(cfg["launch"]["origin_m"])} m.\n'
        f'- Original reference strike point: {tuple(a["target_position_m"])} m; fixed time {strike:.9f} s.\n'
        f'- Full 30 Hz CSV: {len(csv)} rows, duration {csv[-1,0]:.3f} s.\n'
        '- Use the existing FullState controller and lab safety procedure. The first row is the required hover, not a takeoff command.\n'
        '- Execute the complete CSV, including its checked recovery and final hold.\n'
        '- Save only horizontal-task recordings under `flight_take/M6`; keep vertical-whip data separate.\n'
        '- Frozen-model replay and command checks passed. Physical accuracy and safety are not established by simulation alone.\n'
        '- The active flight selection and original M5 export were not changed.\n\n'
        f'Predicted tip tracking RMSE: {100*result["original_reference_tip_rmse_m"]:.3f} → {100*result["corrected_reference_tip_rmse_m"]:.3f} cm.\n'
        f'CSV SHA-256: `{result["csv_sha256"]}`\n\nSource rehearsal: `{output}`\n',encoding='utf-8')
    atomic_json(export/'verification.json',dict(original_sources_verified=True,csv_sha256=result['csv_sha256'],
        rows=len(csv),rate_hz=30,reference_sha256=result['reference_sha256'],model_signature=model['signature'],
        standard_fast_cable_max_difference_m=result['standard_fast_cable_max_difference_m'],
        standard_fast_pose_max_difference_m=result['standard_fast_pose_max_difference_m'],
        independent_cost_difference_m2=result['independent_cost_difference_m2']))
    print('CORRECTED CSV',export/'fullstate_30hz.csv',flush=True)


if __name__=='__main__':main()
