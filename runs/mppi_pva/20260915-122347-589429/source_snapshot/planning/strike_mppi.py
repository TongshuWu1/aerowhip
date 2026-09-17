"""Fixed-budget MPPI for a targeted strike with saved fold requirements."""
import math
import time

import numpy as np
import torch

from experimental_data.io import atomic_json, sha256_file
from learning.pva_env import PVAEnvironment
from planning.mppi_trajectory import interpolation_matrix, adaptive_weights
from planning.mppi_timing import TimedProposals
from planning.mppi_live import publish, series
from planning.strike_objective import StrikeCapture, fold_features, validate_settings, strike_angle_deg, uses_templates, rewarded_speed
from simulator.artifact_io import replace_with_retry


def check_recovery_reference(packets, cfg, prediction_env=None):
    """Existing export constraints; no new reward or terminal-state target."""
    from deployment.pva_rehearsal import complete_pva_packets
    from simulator.research_reference import reference_packet_validity
    times, complete, _, _ = complete_pva_packets(
        packets.detach().cpu().numpy(), np.asarray(cfg['launch']['origin_m']), cfg['limits'],
        recovery_settings=cfg.get('recovery'),jerk_limits=cfg['action']['jerk_limit_m_s3'])
    valid, _ = reference_packet_validity(torch.as_tensor(complete)[None], cfg['limits'])
    if not bool(valid.all()):
        raise ValueError('Complete reference recovery violates the saved command envelope')
    if complete[:, 2].min() < cfg['limits']['minimum_origin_z_m'] or complete[:, 2].max() > cfg['limits']['maximum_origin_z_m']:
        raise ValueError('Complete reference recovery violates the saved height envelope')
    if prediction_env is not None:
        env=prediction_env
        grid=np.arange(round(times[-1]/env.dt)+1)*env.dt
        prediction=env.engine.drone.predict(env.initial_pose,env.tensor(complete)[None],times,grid,
            env.engine.offset,hover_command=env.hover,maximum_tilt_deg=cfg['limits']['maximum_tilt_deg'])
        if not bool(prediction['valid'].all()):
            raise ValueError('Complete predicted attitude or model envelope violation')



def gain_guided_scores(scores, maximum_gain, failed, groups):
    """Guide an unqualified group toward the gain requirement; never accept it.

    Once a group has eligible task scores, its ordinary MPPI scores apply.
    This affects proposal updates only. Incumbent selection and export always
    use the original fully constrained task scores.
    """
    task=scores.reshape(groups,-1)
    guide=maximum_gain.masked_fill(failed,-torch.inf).reshape(groups,-1)
    missing=~torch.isfinite(task).any(-1)
    return torch.where(missing[:,None],guide,task).reshape_as(scores),missing


@torch.no_grad()
def optimize(job, model, cfg):
    from planning.pva_job import progress
    validate_settings(cfg)
    s=cfg['mppi']; samples=s['samples']; groups=s['proposal_count']; per=samples//groups
    steps=round(cfg['task']['duration_s']*30); start=time.perf_counter()
    actual=PVAEnvironment(model,cfg,root=job,device=cfg['device'])
    env=PVAEnvironment(model,cfg,root=job,batch_size=samples+groups+1,device=cfg['device'])
    templates=uses_templates(cfg)
    baselines=env.tensor(np.zeros((2,steps,3)))
    if templates:
        with np.load(job/'proposal_baselines.npz',allow_pickle=False) as saved:
            baselines=env.tensor(saved['normalized_jerk'])
        if baselines.shape!=(2,steps,3) or not bool(torch.isfinite(baselines).all()) or bool((baselines.abs()>1).any()):
            raise ValueError('Two finite full-duration bounded command templates required')
    from planning.position_spline import PositionSpline, SCHEMA as SPLINE_SCHEMA, replay as replay_spline, scratch_proposals
    spline_mode=cfg['command_contract']==SPLINE_SCHEMA
    if spline_mode:
        spline=PositionSpline(steps/30,device=env.device)
        if not templates:
            means=actual.origin0[0].expand(groups,9,3).clone()
            seeds=means.clone();noise_basis=spline.scratch_noise_basis()
        else:
            from simulator.pva_commands import jerk_packets
            spline=PositionSpline(steps/30,device=env.device)
            seed_packets=jerk_packets(actual.command.expand(2,-1),baselines*actual.limit)
            seeds=spline.fit_positions(seed_packets[...,:3],actual.origin0[0])
            # Contract the initialization shape per axis if fitting introduces
            # excessive jerk. This changes the seed curve, never clips PVA samples.
            derivative=torch.einsum('kc,bcd->bkd',spline.jerk_control_basis,spline.control_points(seeds,actual.origin0[0]))
            scale=(actual.limit/derivative.abs().amax(1).clamp_min(1e-12)).clamp(max=1.)
            seeds=actual.origin0[0]+(seeds-actual.origin0[0])*scale[:,None,:]
            family=torch.arange(groups,device=env.device)%2
            strengths=seeds.new_tensor(s['initial_strengths'])
            strength=strengths[(torch.arange(groups,device=env.device)//2)%len(strengths)]
            means=actual.origin0[0]+(seeds[family]-actual.origin0[0])*strength[:,None,None]
            coordinate=torch.arange(9,device=env.device,dtype=seeds.dtype)
            covariance=torch.exp(-.5*((coordinate[:,None]-coordinate[None,:])/s['position_correlation_length']).square())
            noise_basis=torch.linalg.cholesky(covariance+torch.eye(9,device=env.device,dtype=seeds.dtype)*1e-10)
            if s.get('position_noise_basis')=='jerk':noise_basis=spline.scratch_noise_basis()
        best_free=means[0].clone()
        np.savez_compressed(job/'spline_initialization.npz',control_points=seeds.cpu().numpy(),
            knots_s=spline.knots,degree=spline.degree,noise_basis=noise_basis.cpu().numpy(),source=('Least-squares fit to command positions only' if templates else 'Stationary launch; random smooth position proposals; no previous trajectories'))
    else:
        sampler=TimedProposals(baselines,interpolation_matrix(steps,s['support_points'],device=cfg['device']),s)
        means=baselines.new_zeros(groups,sampler.rows,3)
    rng=torch.Generator(device=env.device).manual_seed(s['seed'])
    best=-math.inf; best_actions=torch.zeros_like(baselines[0]); best_preview=None
    history=[]; best_metrics={}; best_terms={}
    for iteration in range(1,s['iterations']+1):
        progress(job,stage='Searching for a feasible targeted strike',iteration=iteration,
                 planned_iterations=s['iterations'],best_score=best if math.isfinite(best) else None)
        if spline_mode:
            if not templates:
                controls=scratch_proposals(means,actual.origin0[0],noise_basis,s,rng)
            else:
                noise=torch.randn(groups,per,9,3,device=env.device,dtype=means.dtype,generator=rng)
                noise=torch.einsum('ij,gpjc->gpic',noise_basis,noise)
                scales=means.new_tensor(s['position_noise_scales_m'])
                noise*=scales[torch.arange(per,device=env.device)%len(scales)][None,:,None,None]
                controls=means[:,None]+noise
            free=torch.cat((controls.reshape(samples,9,3),means,best_free[None]))
            _,jerk=spline.decode(free,actual.origin0[0]);actions=jerk/actual.limit
        else:
            controls=sampler.sample(means,rng)
            draws=sampler.decode(controls).reshape(samples,steps,3)
            actions=torch.cat((draws,sampler.decode(means),best_actions[None]))
        env.branch_from(actual); capture=StrikeCapture()
        result=replay_spline(env,free,observer=capture) if spline_mode else env.rollout(actions=actions,observer=capture)
        score,terms=capture.score(env,result,actions,cfg['trajectory_objective'])
        # Lazy feasibility checks in descending score order: no new incumbent
        # may have an unrecoverable command exit. Lower-scoring samples remain
        # proposals, not accepted plans. Known failures receive zero weight.
        recovery_rejections=0
        recovery_rejected=torch.zeros_like(score,dtype=torch.bool)
        for index in score.argsort(descending=True).tolist():
            if not math.isfinite(float(score[index])) or float(score[index])<=best:break
            try:check_recovery_reference(result['packets'][index],cfg,actual)
            except ValueError:
                score[index]=-torch.inf;recovery_rejected[index]=True;recovery_rejections+=1
                continue
            break
        search_score=score[:samples];gain_search_groups=0
        if 'minimum_tip_speed_gain_m_s' in cfg['trajectory_objective']:
            guide=env.whip_guidance if cfg['trajectory_objective'].get('free_target',False) else env.maximum_directional_speed_gain
            search_score,missing=gain_guided_scores(search_score,guide[:samples],
                result['failed'][:samples]|recovery_rejected[:samples],groups)
            gain_search_groups=int(missing.sum())
        weights,temperatures,ess=adaptive_weights(search_score.reshape(groups,per),s['target_ess_fraction'])
        updated=(weights[:,:,None,None]*controls).sum(1)
        means=torch.where((weights.sum(-1)>0)[:,None,None],updated,means)
        if bool(torch.isfinite(score).any()):
            index=int(score.argmax()); value=float(score[index])
            if value>best:
                best=value; best_actions=actions[index].clone()
                if spline_mode:best_free=free[index].clone()
                best_terms={key:float(val[index]) for key,val in terms.items()}
                best_metrics=dict(minimum_tip_distance_m=float(env.encounter_distance[index]),
                    strike_time_s=float(env.strike_time[index]),strike_distance_m=float(env.strike_distance[index]),
                    closest_approach_time_s=float(env.encounter_time[index]),
                    directed_tip_speed_m_s=float((env.strike_velocity[index]*env.direction).sum()),
                    strike_angle_deg=float(strike_angle_deg(env.strike_velocity[index],env.direction)),
                    root_velocity_m_s=env.strike_root_velocity[index].cpu().tolist(),
                    root_forward_speed_m_s=float((env.strike_root_velocity[index]*env.direction).sum()),
                    rewarded_tip_speed_m_s=float(rewarded_speed(env.strike_velocity[index],env.direction,cfg['trajectory_objective'],env.strike_root_velocity[index])),
                    tip_velocity_m_s=env.strike_velocity[index].cpu().tolist(),
                    fold_completed_time_s=float(env.fold_completed_time_s[index]) if bool(torch.isfinite(env.fold_completed_time_s[index])) else None,
                    fold_valid=bool(result['fold_valid'][index]))
                if cfg['trajectory_objective'].get('horizontal_cable_weight',0)>0:
                    best_metrics['horizontal_cable_rms_m']=float(env.strike_horizontal_error_m[index])
                if cfg['trajectory_objective'].get('free_target',False):
                    best_metrics.update(free_target=True,selected_strike_position_m=env.strike_position[index].cpu().tolist(),
                        backward_travel_m=float(env.strike_backward_travel[index]),
                        strike_distance_m=None,minimum_tip_distance_m=None,closest_approach_time_s=None,
                        target_note='Strike location selected from the motion; no target accuracy measured')
                    if cfg['trajectory_objective'].get('velocity_propagation'):
                        best_metrics.update(velocity_peak_times_s=env.strike_velocity_peak_times[index].cpu().tolist(),
                            velocity_peak_speeds_m_s=env.strike_velocity_peaks[index].cpu().tolist(),
                            propagation_note='Ordered proximal, middle, distal velocity peaks; kinematic proxy, not energy-flux measurement')
                best_preview=series(capture,index,iteration,score,result,'Best feasible strike candidate')
        peak_gain=env.maximum_directional_speed_gain[:samples].masked_fill(result['failed'][:samples],-torch.inf).max()
        row=dict(iteration=iteration,best_score=best if math.isfinite(best) else None,
            gain_search_groups=gain_search_groups,
            best_sampled_speed_gain_m_s=float(peak_gain) if bool(torch.isfinite(peak_gain)) else None,
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
            best_score=best,best_metrics=best_metrics,
            **(dict(best_control_points=best_free) if spline_mode else {})),job/'checkpoints/search.tmp')
        replace_with_retry(job/'checkpoints/search.tmp',job/'checkpoints/search.pt')
    if not math.isfinite(best):
        raise ValueError('No candidate satisfied flight feasibility and the saved strike requirements. No plan exported; constraints were not relaxed.')
    verification=replay_spline(actual,best_free[None],trace=True) if spline_mode else actual.rollout(actions=best_actions[None],trace=True)
    score,terms=StrikeCapture().score(actual,verification,best_actions[None],cfg['trajectory_objective'])
    if not bool(torch.isfinite(score[0])) or not math.isclose(float(score[0]),best,rel_tol=0,abs_tol=1e-5):
        raise ValueError('Independent replay did not reproduce the accepted strike score')
    frames=actual.frames
    q=torch.stack([actual.initial_state.positions_m[0]]+[f['cable'][0] for f in frames])
    material=torch.cat((actual.wave_material.new_zeros(1),actual.wave_material,actual.wave_material.new_ones(1)))
    opposition,turn,location,valid=fold_features(q,material,cfg['fold_constraint'])
    np.savez_compressed(job/'fold_diagnostics.npz',time_s=np.r_[0.,[f['time_s'] for f in frames]],
        cable_positions_m=q.cpu().numpy(),fold_angle_rad=opposition.cpu().numpy(),
        local_turn_rad=turn.cpu().numpy(),bend_material_coordinate=location.cpu().numpy(),geometry_valid=valid.cpu().numpy())
    plan_fields={}
    if spline_mode:
        packets,jerk=spline.decode(best_free,actual.origin0[0])
        plan_fields=dict(position_control_points_m=best_free.cpu().numpy(),
            command_packets=packets.cpu().numpy(),jerk_samples_m_s3=jerk.cpu().numpy(),
            spline_degree=spline.degree,spline_knots_s=spline.knots,
            command_contract=SPLINE_SCHEMA)
    else:plan_fields=dict(normalized_jerk=best_actions.cpu().numpy())
    np.savez_compressed(job/'plan.tmp.npz',**plan_fields,
        proposal_mean=means.cpu().numpy(),plan_complete=True,committed_steps=steps,
        planner_mode=cfg['mppi']['parameterization'])
    replace_with_retry(job/'plan.tmp.npz',job/'plan.npz')
    summary=dict(iterations=s['iterations'],random_candidate_budget=samples*s['iterations'],
        initialization=s.get('initialization','templates'),
        proposal_baselines_sha256=sha256_file(job/'proposal_baselines.npz') if templates else None,
        fold_requirement=cfg.get('fold_requirement','required'),
        best_score=best,objective_components=best_terms,**best_metrics,
        independent_score_difference=abs(float(score[0])-best),command_steps=steps,
        maneuver_duration_s=cfg['task']['duration_s'],elapsed_s=time.perf_counter()-start,
        planner_mode=cfg['mppi']['parameterization'],stop_reason='fixed_budget',
        recovery_reference_checked=True,complete_recovery_attitude_checked=True,complete_recovery_prediction_checked=False,
        evidence='Simulation-only candidate; physical validation pending')
    atomic_json(job/'result.json',summary)
    return summary
