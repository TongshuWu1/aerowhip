"""Locally invert finalized M1 toward the original M0 planned physical motion."""
from pathlib import Path
from datetime import datetime,timezone
import csv
import json
import shutil
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experimental_data.io import atomic_json,sha256_file
from experimental_data.whip_adaptation import verify_hashes
from experimental_data.model_evaluation import model_identity
from simulator.workflow import read_json
from simulator.research_execution import ResearchExecutionModel
from planning.position_spline import PositionSpline
from planning.local_reference_correction import optimize
from planning.pva_job import freeze_model_assets
from planning.reference_correction import CoupledRollout,tracking_cost,command_valid
from deployment.pva_rehearsal import complete_pva_packets
from deployment.research_rehearsal import FIELDS


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    source=ROOT/'runs/adaptation/M1-paper-20260913-verified-physical'
    result=read_json(source/'fit/result.json')
    if result['status']!='completed':raise ValueError('Complete M1 required')
    verify_hashes(result['candidate_hashes'])
    refdir=ROOT/'runs/reference_tracking/M0-paper-fixed-reference'
    meta=read_json(refdir/'reference.json')
    verify_hashes({str(ROOT/k):v for k,v in meta['source_hashes'].items()})
    if sha256_file(refdir/'reference.npz')!=meta['reference_sha256']:raise ValueError('Reference changed')
    with np.load(refdir/'reference.npz') as f:ref={k:f[k].copy() for k in f.files}
    original=ROOT/meta['source_rehearsal'];cfg=read_json(original/'settings.json')
    stamp=datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    job=ROOT/'runs/reference_tracking'/('M1-local-'+stamp)
    job.mkdir(parents=True,exist_ok=False)
    model=freeze_model_assets(read_json(source/'candidate/model.json'),job,source_root=source/'candidate',portable=True)
    atomic_json(job/'model.json',model)
    weights=dict(tip=1.,quadrotor=.1,command=.01)
    local_settings=dict(maximum_iterations=12,initial_trust_radius=.5,maximum_trust_radius=1.,
        finite_difference_step=.01,maximum_derivative_refinements=3,maximum_jacobian_relative_difference=.05,damping=1e-4,
        relative_improvement_stop=.001,backtracking_scales=[1.,.5,.25,.125,.0625,.03125,.015625])
    cfg.update(model_path='model.json',method='local_reference_correction',reward={},
        trajectory_objective=dict(schema='fixed_tip_reference_v1',weights=weights,
            interval_s=meta['interval_s'],reference_file='reference.npz'),
        correction=dict(original_spline_duration_s=cfg['task']['duration_s'],fixed_handover_s=meta['interval_s'][1],
            reference_sha256=meta['reference_sha256'],optimizer=local_settings,
            initialization='Original executed M0 B-spline controls; no new MPPI motion',
            measured_trajectory_anchor=False,model_fitting_changed=False,
            units='Tracking terms are mean squared 3D distances in m^2',physical_evidence=False))
    atomic_json(job/'settings.json',cfg)
    shutil.copy2(refdir/'reference.npz',job/'reference.npz')
    shutil.copy2(refdir/'reference.json',job/'reference.json')
    sources={str(f):sha256_file(f) for f in [Path(__file__),ROOT/'planning/reference_correction.py',
        ROOT/'planning/position_spline.py',ROOT/'planning/local_reference_correction.py',
        ROOT/'simulator/research_pose.py',ROOT/'simulator/research_physics.py',
        ROOT/'deployment/braking_recovery.py',source/'fit/selection_frozen.json',
        job/'settings.json',job/'reference.npz']}
    _,h=model_identity(job/'model.json');sources.update(h)
    atomic_json(job/'source_hashes.json',sources)
    print('JOB',job,flush=True)
    start=time.perf_counter()
    def status(stage,**values):
        atomic_json(job/'status.json',dict(status='running',stage=stage,**values))
        print(stage,values,flush=True)
    try:
        engine=ResearchExecutionModel.from_mapping(model,root=job,device='cuda')
        origin=cfg['launch']['origin_m'];limits=cfg['limits']
        tensor=lambda a:torch.as_tensor(a,device='cuda',dtype=torch.float64)
        spline=PositionSpline(cfg['correction']['original_spline_duration_s'],device='cuda')
        baseline=tensor(ref['original_position_control_points_m'])
        cutoff=len(ref['command_time_s'])
        base_packets,base_jerk=spline.decode(baseline,origin);base_packets=base_packets[:cutoff]
        if not np.allclose(base_packets.cpu(),ref['original_command_packets'],atol=1e-10,rtol=0):
            raise ValueError('Original command initialization differs')
        grid=ref['time_s'];times=ref['command_time_s']
        reference=dict(tip=tensor(ref['tip_position_m']),quadrotor=tensor(ref['quadrotor_position_m']),
            command=tensor(ref['original_command_packets'][:,:3]))
        initial_q=ref['cable_position_m'][0]
        single=CoupledRollout(engine,1,origin,initial_q,limits)
        def finish(free):
            packets,jerk=spline.decode(free,origin)
            packets=packets[:cutoff].cpu().numpy()
            t,complete,phases,recovery=complete_pva_packets(packets,np.asarray(origin),limits,
                recovery_settings=cfg['recovery'],jerk_limits=cfg['action']['jerk_limit_m_s3'])
            if not bool(command_valid(tensor(complete)[None],limits).all()):
                raise ValueError('Complete corrected command violates the saved envelope')
            fullgrid=np.arange(round(t[-1]/engine.dt_s)+1)*engine.dt_s
            prediction=single(tensor(complete)[None],t,fullgrid)
            if not bool(prediction['complete_valid'].all()):
                raise ValueError('Complete coupled recovery leaves the saved model envelope')
            return t,complete,phases,recovery,fullgrid,prediction
        status('Checking unchanged M0 command under M1')
        unchanged=single(base_packets[None],times,grid)
        base_cost,base_terms=tracking_cost(unchanged['cable_positions_m'][:,:,-1],
            unchanged['position_origin_m'],base_packets[None],reference,weights)
        atomic_json(job/'baseline.json',dict(cost_m2=float(base_cost[0]),
            terms_m2={k:float(v[0]) for k,v in base_terms.items()},valid=bool(unchanged['complete_valid'][0])))
        np.savez_compressed(job/'baseline_prediction.npz',time_s=grid,
            cable_positions_m=unchanged['cable_positions_m'][0].cpu().numpy(),
            origin_positions_m=unchanged['position_origin_m'][0].cpu().numpy())
        best,best_cost,best_terms,optimization=optimize(spline,baseline,origin,cutoff,
            engine,initial_q,times,grid,reference,weights,limits,cfg['action']['jerk_limit_m_s3'],
            finish,job,status,local_settings)
        if not np.isfinite(best_cost) or best_cost>=float(base_cost[0]):
            raise ValueError('No feasible correction improved the fixed-reference objective')
        status('Checking independent complete replay')
        t,packets,phases,recovery,fullgrid,prediction=finish(best)
        prefix=len(grid)
        actual,actual_terms=tracking_cost(prediction['cable_positions_m'][:,:prefix,-1],
            prediction['position_origin_m'][:,:prefix],tensor(packets[:cutoff])[None],reference,weights)
        if abs(float(actual[0])-best_cost)>1e-6:
            raise ValueError('Independent batch-one replay changed the tracking cost')
        # Verify the optimized fast propagation against the standard complete
        # production predictor, including every recovery command.
        standard=engine.predict(single.pose,single.state,tensor(packets)[None],t,fullgrid,
            graph=True,hover_command=single.hover)
        difference=float((standard['cable_positions_m']-prediction['cable_positions_m']).abs().max())
        pose_difference=float((standard['position_origin_m']-prediction['position_origin_m']).abs().max())
        if difference>1e-5 or pose_difference>1e-9:
            raise ValueError(f'Standard/fast complete replay differs: cable={difference}, pose={pose_difference}')
        if not np.allclose(packets[-1,:3],origin,atol=1e-10) or not np.allclose(packets[-1,3:],0,atol=1e-10):
            raise ValueError('Recovery does not finish at the original settled hover')
        verify_hashes(sources)
        output=ROOT/'runs/rehearsals_pva'/f'{stamp}-M1-local-fixed-tip-reference'
        output.mkdir(parents=True,exist_ok=False)
        saved=freeze_model_assets(model,output,source_root=job,portable=True)
        atomic_json(output/'model.json',saved);atomic_json(output/'settings.json',cfg)
        arrays=dict(command_time_s=t,commands=packets,command_phase=phases,prediction_time_s=fullgrid,
            cable_positions_m=prediction['cable_positions_m'][0].cpu().numpy(),
            cable_velocities_m_s=prediction['cable_velocities_m_s'][0].cpu().numpy(),
            origin_positions_m=prediction['position_origin_m'][0].cpu().numpy(),
            origin_velocities_m_s=prediction['velocity_origin_m_s'][0].cpu().numpy(),
            origin_rotations=prediction['rotation_tracking_to_world'][0].cpu().numpy(),
            target_position_m=cfg['launch']['target_m'],reference_time_s=grid,
            reference_tip_positions_m=ref['tip_position_m'],reference_origin_positions_m=ref['quadrotor_position_m'])
        np.savez_compressed(output/'rehearsal.npz',**arrays)
        with (output/'fullstate_30hz.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream);writer.writerow(FIELDS);writer.writerows(np.c_[t,packets])
        decoded,jerk=spline.decode(best,origin)
        np.savez_compressed(job/'plan.npz',position_control_points_m=best.cpu().numpy(),
            command_packets=decoded.cpu().numpy(),spline_knots_s=spline.knots,
            spline_degree=spline.degree,plan_complete=True,command_contract=cfg['command_contract'],
            fixed_handover_s=meta['interval_s'][1],planner_mode='fixed_reference_correction')
        shutil.copy2(job/'plan.npz',output/'plan.npz')
        strike=meta['planned_strike_time_s'];tip=arrays['cable_positions_m'][:,-1]
        tip_at=np.array([np.interp(strike,fullgrid,tip[:,j]) for j in range(3)])
        summary=dict(schema='pva_fullstate_30hz_v1',planner='Local M1 fixed-reference command correction',
            planner_mode='open_loop',command_contract=cfg['command_contract'],
            objective_schema='fixed_tip_reference_v1',job=str(job),checkpoint=None,
            initial_tracking_origin_m=origin,target_position_m=cfg['launch']['target_m'],
            whip_end_s=meta['interval_s'][1],total_duration_s=float(t[-1]),
            predicted_hit_time_s=None,predicted_valid_hit=False,
            strike_time_s=strike,strike_distance_m=float(np.linalg.norm(tip_at-np.asarray(cfg['launch']['target_m']))),
            minimum_tip_distance_m=float(np.linalg.norm(tip[:prefix]-np.asarray(cfg['launch']['target_m']),axis=-1).min()),
            original_reference_tip_rmse_m=float(base_terms['tip'][0].sqrt()),
            corrected_reference_tip_rmse_m=float(actual_terms['tip'][0].sqrt()),
            corrected_reference_quadrotor_rmse_m=float(actual_terms['quadrotor'][0].sqrt()),
            maximum_reference_quadrotor_deviation_m=float(np.linalg.norm(arrays['origin_positions_m'][:prefix]-ref['quadrotor_position_m'],axis=-1).max()),
            baseline_cost_m2=float(base_cost[0]),corrected_cost_m2=float(actual[0]),
            reference_sha256=meta['reference_sha256'],recovery=recovery,
            recovery_prediction_complete=True,prediction_valid_through_s=float(fullgrid[-1]),
            reference_feasible=True,predicted_attitude_checked=True,
            maximum_predicted_tilt_limit_deg=limits['maximum_tilt_deg'],
            minimum_predicted_cable_height_m=float(arrays['cable_positions_m'][...,2].min()),
            standard_fast_cable_max_difference_m=difference,standard_fast_pose_max_difference_m=pose_difference,
            independent_cost_difference_m2=abs(float(actual[0])-best_cost),
            csv_sha256=sha256_file(output/'fullstate_30hz.csv'),iterations=optimization['iterations'],
            optimization=optimization,elapsed_s=time.perf_counter()-start,
            command_semantics='Desired tracked-origin P/V/A, 30 Hz zero-order hold, kinematic acceleration, zero yaw',
            evidence='Simulation-only command correction toward the original M0 physical motion; M1 has not flown.',
            recovery_empirically_validated=False,success_meaning='No binary hit claim; fixed-time tracking objective')
        atomic_json(output/'rehearsal.json',summary)
        atomic_json(output/'task.json',dict(target_position_m=cfg['launch']['target_m'],
            desired_strike_direction_world=cfg['task']['strike_direction'],objective=cfg['trajectory_objective']))
        export=ROOT/'exports/M1_local_fixed_tip_reference';export.mkdir(exist_ok=False)
        shutil.copy2(output/'fullstate_30hz.csv',export/'fullstate_30hz.csv')
        atomic_json(export/'export.json',dict(rehearsal=output.relative_to(ROOT).as_posix(),
            correction_job=job.relative_to(ROOT).as_posix(),**summary))
        (export/'README.md').write_text(
            '# M1 local fixed-tip-reference command\n\n'
            'Use `fullstate_30hz.csv` with the existing 30 Hz FullState flight program.\n'
            'Launch tracked origin: (0, 0, 1.4) m. Target: (1.25, 0, 1.25) m.\n'
            'Hold at the launch before playback using the existing lab procedure.\n'
            'Execute the complete CSV, including slower braking, return and final hold.\n'
            f'Duration: {t[-1]:.3f} s. No new flight interface is implemented.\n\n'
            'Commands are corrected with M1 to track the original M0 predicted tip\n'
            'trajectory at the original times, with a smaller quadrotor tracking\n'
            'penalty. This is a simulated correction; physical performance is unmeasured.\n',encoding='utf-8')
        atomic_json(job/'result.json',dict(**summary,rehearsal=str(output),export=str(export)))
        atomic_json(job/'status.json',dict(status='completed',stage='Corrected CSV exported',export=str(export)))
        print(json.dumps(dict(export=str(export),rehearsal=str(output),**summary),indent=2),flush=True)
    except BaseException as exc:
        atomic_json(job/'status.json',dict(status='failed',error=str(exc)))
        raise


if __name__=='__main__':main()
