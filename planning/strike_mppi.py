"""Fixed-budget MPPI for a target strike with a required travelling fold."""
import math
import time

import numpy as np
import torch

from experimental_data.io import atomic_json, sha256_file
from learning.pva_env import PVAEnvironment
from planning.mppi_trajectory import interpolation_matrix, adaptive_weights
from planning.mppi_timing import TimedProposals
from planning.mppi_live import publish, series
from planning.strike_objective import StrikeCapture, fold_features, validate_settings
from simulator.artifact_io import replace_with_retry


def check_recovery_reference(packets, cfg):
    """Existing export constraints; no new reward or terminal-state target."""
    from deployment.pva_rehearsal import complete_pva_packets
    from simulator.research_reference import reference_packet_validity
    _, complete, _, _ = complete_pva_packets(
        packets.detach().cpu().numpy(), np.asarray(cfg['launch']['origin_m']), cfg['limits'])
    valid, _ = reference_packet_validity(torch.as_tensor(complete)[None], cfg['limits'])
    if not bool(valid.all()):
        raise ValueError('Complete reference recovery violates the saved command envelope')
    if complete[:, 2].min() < cfg['limits']['minimum_origin_z_m'] or complete[:, 2].max() > cfg['limits']['maximum_origin_z_m']:
        raise ValueError('Complete reference recovery violates the saved height envelope')


@torch.no_grad()
def optimize(job, model, cfg):
    from planning.pva_job import progress
    validate_settings(cfg)
    s=cfg['mppi']; samples=s['samples']; groups=s['proposal_count']; per=samples//groups
    steps=round(cfg['task']['duration_s']*30); start=time.perf_counter()
    actual=PVAEnvironment(model,cfg,root=job,device=cfg['device'])
    env=PVAEnvironment(model,cfg,root=job,batch_size=samples+groups+1,device=cfg['device'])
    with np.load(job/'proposal_baselines.npz',allow_pickle=False) as saved:
        baselines=env.tensor(saved['normalized_jerk'])
    if baselines.shape!=(2,steps,3) or not bool(torch.isfinite(baselines).all()) or bool((baselines.abs()>1).any()):
        raise ValueError('Two finite full-duration bounded command templates required')
    sampler=TimedProposals(baselines,interpolation_matrix(steps,s['support_points'],device=cfg['device']),s)
    means=baselines.new_zeros(groups,sampler.rows,3)
    rng=torch.Generator(device=env.device).manual_seed(s['seed'])
    best=-math.inf; best_actions=torch.zeros_like(baselines[0]); best_preview=None
    history=[]; best_metrics={}; best_terms={}
    for iteration in range(1,s['iterations']+1):
        progress(job,stage='Searching for a feasible travelling-fold strike',iteration=iteration,
                 planned_iterations=s['iterations'],best_score=best if math.isfinite(best) else None)
        controls=sampler.sample(means,rng)
        draws=sampler.decode(controls).reshape(samples,steps,3)
        actions=torch.cat((draws,sampler.decode(means),best_actions[None]))
        env.branch_from(actual); capture=StrikeCapture()
        result=env.rollout(actions=actions,observer=capture)
        score,terms=capture.score(env,result,actions,cfg['trajectory_objective'])
        # Lazy feasibility checks in descending score order: no new incumbent
        # may have an unrecoverable command exit. Lower-scoring samples remain
        # proposals, not accepted plans. Known failures receive zero weight.
        recovery_rejections=0
        for index in score.argsort(descending=True).tolist():
            if not math.isfinite(float(score[index])) or float(score[index])<=best:break
            try:check_recovery_reference(result['packets'][index],cfg)
            except ValueError:
                score[index]=-torch.inf;recovery_rejections+=1
                continue
            break
        weights,temperatures,ess=adaptive_weights(score[:samples].reshape(groups,per),s['target_ess_fraction'])
        updated=(weights[:,:,None,None]*controls).sum(1)
        means=torch.where((weights.sum(-1)>0)[:,None,None],updated,means)
        if bool(torch.isfinite(score).any()):
            index=int(score.argmax()); value=float(score[index])
            if value>best:
                best=value; best_actions=actions[index].clone()
                best_terms={key:float(val[index]) for key,val in terms.items()}
                best_metrics=dict(minimum_tip_distance_m=float(env.encounter_distance[index]),
                    strike_time_s=float(env.strike_time[index]),strike_distance_m=float(env.strike_distance[index]),
                    closest_approach_time_s=float(env.encounter_time[index]),
                    directed_tip_speed_m_s=float((env.strike_velocity[index]*env.direction).sum()),
                    tip_velocity_m_s=env.strike_velocity[index].cpu().tolist(),
                    fold_completed_time_s=float(env.fold_completed_time_s[index]),fold_valid=True)
                best_preview=series(capture,index,iteration,score,result,'Best verified-fold candidate')
        row=dict(iteration=iteration,best_score=best if math.isfinite(best) else None,
            physics_valid_fraction=float((~result['failed'][:samples]).double().mean()),
            accepted_fraction=float(torch.isfinite(score[:samples]).double().mean()),
            geometric_fold_fraction=float(env.fold_completed[:samples].double().mean()),
            recovery_reference_rejections=recovery_rejections,
            effective_samples=float(ess.sum()),elapsed_s=time.perf_counter()-start,
            objective_components=best_terms,**best_metrics)
        history.append(row); atomic_json(job/'history.json',history)
        if best_preview is not None and cfg.get('visualization',{}).get('live_mppi',True):
            publish(job,actual,capture,score,result,best_preview,iteration)
        torch.save(dict(iteration=iteration,means=means,rng=rng.get_state(),best_actions=best_actions,
            best_score=best,best_metrics=best_metrics),job/'checkpoints/search.tmp')
        replace_with_retry(job/'checkpoints/search.tmp',job/'checkpoints/search.pt')
    if not math.isfinite(best):
        raise ValueError('No candidate satisfied flight feasibility and the travelling-fold condition. No plan exported; constraints were not relaxed.')
    verification=actual.rollout(actions=best_actions[None],trace=True)
    score,terms=StrikeCapture().score(actual,verification,best_actions[None],cfg['trajectory_objective'])
    if not bool(torch.isfinite(score[0])) or not math.isclose(float(score[0]),best,rel_tol=0,abs_tol=1e-5):
        raise ValueError('Independent replay did not reproduce the accepted fold and strike score')
    frames=actual.frames
    q=torch.stack([actual.initial_state.positions_m[0]]+[f['cable'][0] for f in frames])
    material=torch.cat((actual.wave_material.new_zeros(1),actual.wave_material,actual.wave_material.new_ones(1)))
    opposition,turn,location,valid=fold_features(q,material,cfg['fold_constraint'])
    np.savez_compressed(job/'fold_diagnostics.npz',time_s=np.r_[0.,[f['time_s'] for f in frames]],
        cable_positions_m=q.cpu().numpy(),fold_angle_rad=opposition.cpu().numpy(),
        local_turn_rad=turn.cpu().numpy(),bend_material_coordinate=location.cpu().numpy(),geometry_valid=valid.cpu().numpy())
    np.savez_compressed(job/'plan.tmp.npz',normalized_jerk=best_actions.cpu().numpy(),
        proposal_mean=means.cpu().numpy(),plan_complete=True,committed_steps=steps,
        planner_mode='targeted_fold_strike_v1')
    replace_with_retry(job/'plan.tmp.npz',job/'plan.npz')
    summary=dict(iterations=s['iterations'],random_candidate_budget=samples*s['iterations'],
        proposal_baselines_sha256=sha256_file(job/'proposal_baselines.npz'),
        best_score=best,objective_components=best_terms,**best_metrics,
        independent_score_difference=abs(float(score[0])-best),command_steps=steps,
        maneuver_duration_s=cfg['task']['duration_s'],elapsed_s=time.perf_counter()-start,
        planner_mode='targeted_fold_strike_v1',stop_reason='fixed_budget',
        recovery_reference_checked=True,complete_recovery_prediction_checked=False,
        evidence='Simulation-only candidate; physical validation pending')
    atomic_json(job/'result.json',summary)
    return summary
