"""Translate a saved jerk plan in height and independently replay it."""
from pathlib import Path
import sys,json,shutil
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from planning.pva_job import prepare
from planning.strike_objective import StrikeCapture,fold_features
from planning.strike_mppi import check_recovery_reference
from learning.pva_env import PVAEnvironment
from experimental_data.io import atomic_json,sha256_file
parent=ROOT/'runs/mppi_pva/20260915-123600-984827'
original=ROOT/'runs/rehearsals_pva/M5-Curved-Side-Whip'
saved=json.loads((parent/'result.json').read_text())
cfg=json.loads((parent/'settings.json').read_text())
shift=1.3-saved['selected_strike_position_m'][2]
cfg['launch']['origin_m'][2]+=shift
cfg['launch']['target_m'][2]+=shift
job,_=prepare(ROOT,cfg,'M5 curved side whip - release at 1.3 m',
    development_review='User requested a lower maneuver with release around 1.3 m. Translate the existing command vertically, retain normalized jerk and timing, and independently replay all existing requirements and full recovery. No new MPPI search or target accuracy claim.')
shutil.copy2(parent/'proposal_baselines.npz',job/'proposal_baselines.npz')
shutil.copy2(__file__,job/'height_translation.py')
try:
    with np.load(parent/'plan.npz') as z:actions=z['normalized_jerk'].copy()
    model=json.loads((job/'model.json').read_text())
    env=PVAEnvironment(model,cfg,root=job,device='cuda')
    with torch.no_grad():
        result=env.rollout(actions=env.tensor(actions)[None],trace=True)
        score,terms=StrikeCapture().score(env,result,env.tensor(actions)[None],cfg['trajectory_objective'])
        assert torch.isfinite(score[0]),'Lowered command fails a saved motion or physical requirement'
        q=torch.stack([env.initial_state.positions_m[0]]+[f['cable'][0] for f in env.frames])
        v=torch.stack([env.initial_state.velocities_m_s[0]]+[f['cable_velocity'][0] for f in env.frames])
        times=np.r_[0.,[f['time_s'] for f in env.frames]]
        with np.load(original/'rehearsal.npz') as z:
            old_q=z['cable_positions_m'][:len(q)].copy();old_v=z['cable_velocities_m_s'][:len(v)].copy()
        expected=old_q+np.array([0.,0.,shift])
        qerror=float(np.max(abs(q.cpu().numpy()-expected)))
        verror=float(np.max(abs(v.cpu().numpy()-old_v)))
        assert qerror<1e-8 and verror<1e-7,(qerror,verror)
        assert abs(float(score[0])-saved['best_score'])<1e-8
        assert abs(float(env.strike_time[0])-saved['strike_time_s'])<1e-10
        assert abs(float(env.strike_position[0,2])-1.3)<1e-8
        check_recovery_reference(result['packets'][0],cfg,env)
        material=torch.cat((env.wave_material.new_zeros(1),env.wave_material,env.wave_material.new_ones(1)))
        opposition,turn,location,valid=fold_features(q,material,cfg['fold_constraint'])
    np.savez_compressed(job/'fold_diagnostics.npz',time_s=times,cable_positions_m=q.cpu().numpy(),
        cable_velocities_m_s=v.cpu().numpy(),fold_angle_rad=opposition.cpu().numpy(),
        local_turn_rad=turn.cpu().numpy(),bend_material_coordinate=location.cpu().numpy(),geometry_valid=valid.cpu().numpy())
    np.savez_compressed(job/'plan.npz',normalized_jerk=actions,plan_complete=True,committed_steps=len(actions),planner_mode=cfg['mppi']['parameterization'])
    summary=dict(saved)
    summary.update(iterations=0,random_candidate_budget=0,initialization='translated_saved_plan',
        source_job=str(parent),source_plan_sha256=sha256_file(parent/'plan.npz'),source_planner_iterations=saved['iterations'],
        best_score=float(score[0]),objective_components={k:float(x[0]) for k,x in terms.items()},
        independent_score_difference=abs(float(score[0])-saved['best_score']),
        selected_strike_position_m=env.strike_position[0].cpu().tolist(),
        stop_reason='vertical_translation_replay',height_translation_m=shift,
        translated_position_max_difference_m=qerror,velocity_max_difference_m_s=verror,
        evidence='Existing MPPI command lowered and independently replayed; no new optimization or physical flight',
        elapsed_s=None,complete_recovery_prediction_checked=False)
    atomic_json(job/'result.json',summary)
    atomic_json(job/'status.json',dict(summary,status='completed'))
    atomic_json(job/'height_translation.json',dict(parent_job=str(parent),parent_rehearsal=str(original),
        source_plan_sha256=sha256_file(parent/'plan.npz'),script_sha256=sha256_file(job/'height_translation.py'),
        height_shift_m=shift,desired_release_height_m=1.3,initial_hover_m=cfg['launch']['origin_m'],
        normalized_jerk_unchanged=True,timing_unchanged=True,position_max_difference_m=qerror,velocity_max_difference_m_s=verror))
    atomic_json(ROOT/'tmp/m5_side_whip_lowered.json',dict(job=str(job),output=str(ROOT/'runs/rehearsals_pva/M5-Curved-Side-Whip-1p3m')))
    print(json.dumps(dict(job=str(job),shift_m=shift,hover_m=cfg['launch']['origin_m'],release_m=summary['selected_strike_position_m'],position_error_m=qerror,velocity_error_m_s=verror),indent=2))
except Exception as exc:
    atomic_json(job/'status.json',dict(status='failed',stage=str(exc)));raise
