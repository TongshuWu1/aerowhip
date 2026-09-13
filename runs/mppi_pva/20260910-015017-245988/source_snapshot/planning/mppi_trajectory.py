"""Offline, multiple-proposal MPPI in control-point space.

Zero control prior: adaptive-temperature exponential reward weighting, not a
claim of exact importance sampling against a fixed Gaussian control prior.
Every interpolated jerk sequence executes the original PVA/physics contract.
"""
import math
import time
import numpy as np
import torch
from experimental_data.io import atomic_json
from simulator.artifact_io import replace_with_retry
from learning.pva_env import PVAEnvironment
from planning.mppi_live import RolloutCapture,series,publish


OBJECTIVE=dict(distance=800.,joint=400.,pull=10.,release=10.,wave=15.,
    hit=4000.,vertical=60.,approach=30.,lateral=60.,jerk=.02,
    exit_speed=.5,exit_climb=5.,exit_acceleration=1.,proximity_scale_m=.35)


def interpolation_matrix(steps,points,*,device='cpu'):
    """Piecewise-linear latent interpolation, followed by tanh jerk bounds."""
    t=torch.linspace(0,points-1,steps,device=device,dtype=torch.float64)
    left=t.floor().long().clamp(max=points-2);fraction=t-left
    matrix=torch.zeros(steps,points,device=device,dtype=t.dtype)
    matrix.scatter_(1,left[:,None],(1-fraction)[:,None])
    matrix.scatter_add_(1,(left+1)[:,None],fraction[:,None])
    return matrix


def adaptive_weights(score,fraction):
    """Per-proposal ESS target among feasible random samples; deterministic rows excluded."""
    valid=torch.isfinite(score);count=valid.sum(-1)
    peak=score.masked_fill(~valid,-torch.inf).amax(-1)
    peak=torch.where(count>0,peak,torch.zeros_like(peak))
    delta=torch.where(valid,score-peak[:,None],torch.full_like(score,-torch.inf))
    target=(count*fraction).clamp_min(1.5).minimum(count)
    low=torch.full_like(peak,1e-3);high=torch.full_like(peak,1e5)
    def weights(temp):
        value=torch.exp(delta/temp[:,None]);return value/value.sum(-1,keepdim=True).clamp_min(1e-300)
    for _ in range(32):
        mid=(low*high).sqrt();w=weights(mid);ess=1/w.square().sum(-1).clamp_min(1e-300)
        low=torch.where(ess<target,mid,low);high=torch.where(ess>=target,mid,high)
    w=weights(high);ess=torch.where(count>0,1/w.square().sum(-1).clamp_min(1e-300),torch.zeros_like(peak))
    return w,high,ess


def joint_quality(distance,tip_forward,drone_forward,backward,reach,task,scale):
    """Continuous guidance; no wave/phase booleans gate approach credit."""
    proximity=torch.exp(-(distance/scale).square())
    speed=(tip_forward/task['minimum_directed_speed_m_s']).clamp(0,1)
    reverse=(-drone_forward/task['minimum_backward_speed_m_s']).clamp(0,1)
    travel=(backward/task['minimum_backward_distance_m']).clamp(0,1)
    return proximity*(.2+.8*speed)*(.2+.8*reverse)*(.2+.8*travel)*(.2+.8*reach)


class ObjectiveCapture(RolloutCapture):
    """30 Hz shaping and display. Strict hits/min distance stay at physics rate."""
    def __init__(self):super().__init__();self.metrics=[]
    def __call__(self,env):
        super().__call__(env)
        q=env.state.positions_m;v=env.state.velocities_m_s;p=env.pose.position
        forward=((p-env.origin0)*env.direction).sum(-1)
        reach=(((q[:,-1]-q[:,0])*env.direction).sum(-1)/env.cable_length).clamp(0,1)
        self.metrics.append(dict(distance=(q[:,-1]-env.target).norm(dim=-1),
            tip_forward=(v[:,-1]*env.direction).sum(-1),drone_forward=(env.pose.velocity*env.direction).sum(-1),
            backward=env.pull_peak-forward,reach=reach,
            vertical=(p[:,2]-env.origin0[:,2]).square(),approach=forward.clamp_min(0).square(),
            lateral=((p-env.origin0)-forward[:,None]*env.direction)[:,:2].square().sum(-1),
            time=env.index/30))

    def score(self,env,result,actions,weights):
        stack=lambda key:torch.stack([row[key] for row in self.metrics],-1)
        quality=joint_quality(*(stack(k) for k in ('distance','tip_forward','drone_forward','backward','reach')),
            env.settings['task'],weights['proximity_scale_m'])
        times=env.tensor([r['time'] for r in self.metrics])
        valid=times[None]<=result['duration_s'][:,None]+1e-9
        # Left rectangles stop at true termination; no repeated cost on frozen post-hit states.
        dt=(result['duration_s'][:,None]-times[None,:-1]).clamp(0,1/30)
        terms=dict(distance=-weights['distance']*(result['minimum_tip_distance_m']/env.cable_length).square(),
            joint=weights['joint']*quality.masked_fill(~valid,0).amax(-1),
            pull=weights['pull']*env.pull_credit,release=weights['release']*env.reverse_credit,
            wave=weights['wave']*env.wave_credit,hit=weights['hit']*result['success'].double())
        for key in ('vertical','approach','lateral'):
            terms[key]=-weights[key]*(stack(key)[:,:-1]*dt).sum(-1)
        action_times=torch.arange(actions.shape[1],device=env.device)/30
        active_dt=(result['duration_s'][:,None]-action_times[None]).clamp(0,1/30)
        terms['jerk']=-weights['jerk']*(actions.square().mean(-1)*active_dt).sum(-1)
        velocity,acceleration=env.command[:,3:6],env.command[:,6:9]
        along=((velocity*acceleration).sum(-1)/velocity.norm(dim=-1).clamp_min(1e-9)).clamp_min(0)
        terms['exit']=-(weights['exit_speed']*velocity.square().sum(-1)+
            weights['exit_climb']*velocity[:,2].clamp_min(0).square()+weights['exit_acceleration']*along.square())
        total=sum(terms.values())
        if not bool(torch.isfinite(total).all()):raise ValueError('Nonfinite trajectory objective')
        return total.masked_fill(result['failed'],-torch.inf),terms


@torch.no_grad()
def optimize(job,model,cfg):
    from planning.pva_job import progress
    from planning.mppi_receding import wave_seed_bank
    s=cfg['mppi'];samples=s['samples'];groups=s['proposal_count'];per=samples//groups
    points=s['support_points'];steps=round(cfg['task']['duration_s']*30)
    weights=cfg['trajectory_objective'];device=cfg['device'];start=time.perf_counter()
    preferred=weights.get('schema')=='preferred_fold_v1'
    capture_factory=ObjectiveCapture
    if preferred:
        from planning.whip_objective import PreferredFoldCapture
        with np.load(job/weights['reference_file']) as data:
            reference=torch.as_tensor(data['shape'],device=device,dtype=torch.float64)
        capture_factory=lambda:PreferredFoldCapture(reference)
    actual=PVAEnvironment(model,cfg,root=job,device=device)
    env=PVAEnvironment(model,cfg,root=job,batch_size=samples+groups+2,device=device)
    basis=interpolation_matrix(steps,points,device=device);inverse=torch.linalg.pinv(basis)
    rng=torch.Generator(device=device).manual_seed(s['seed'])
    with np.load(job/'initial_proposal.npz') as data:native=actual.tensor(data['normalized_jerk'][:steps])
    if native.shape!=(steps,3):raise ValueError('Native editable seed must span complete maneuver')
    # A residual parameterization preserves every native pulse at zero controls.
    # Smooth perturbations remain fully editable and all candidates use M0.
    native_latent=torch.atanh(native.clamp(-1+1e-12,1-1e-12))
    offset=native_latent if preferred else torch.zeros_like(native)
    bank=wave_seed_bank(actual,steps,s['seed'])
    ids=torch.linspace(0,len(bank)-1,samples,device=device).long()
    controls=torch.einsum('ph,bhc->bpc',inverse,torch.atanh(bank[ids].clamp(-.999,.999))-offset)
    controls[0]=inverse@((native_latent if preferred else torch.atanh(native.clamp(-.999,.999)))-offset)
    seed_actions=torch.tanh(offset+torch.einsum('hp,bpc->bhc',basis,controls))
    actions=torch.cat((seed_actions,seed_actions[:groups],native[None],native[None]))
    progress(job,stage='Screening diverse complete whip proposals',iteration=0)
    capture=capture_factory();env.branch_from(actual)
    result=env.rollout(actions=actions,observer=capture);score,terms=capture.score(env,result,actions,weights)
    if not bool(torch.isfinite(score[:samples]).any()):raise ValueError('No feasible control-point initialization')
    # Favor good but distinct initial controls; retain multiple basins throughout refinement.
    pool=score[:samples].clone();chosen=[]
    for _ in range(groups):
        if not bool(torch.isfinite(pool).any()):pool=score[:samples].clone()
        index=int(pool.argmax());chosen.append(index)
        distance=(controls-controls[index]).square().mean((1,2)).sqrt()
        pool=pool.masked_fill(distance<.08,-torch.inf)
    means=controls[chosen].clone()
    if preferred:means[0].zero_()  # retain a native-centered proposal basin
    best=-math.inf;best_hit=False;best_actions=native.clone();best_preview=None;best_terms={};best_metrics={}
    history=[];anchor=-math.inf;improved_at=0;iteration=0
    def record_candidate(i,iteration):
        nonlocal best,best_hit,best_actions,best_preview,best_terms,best_metrics
        hit=bool(result['success'][i]);value=float(score[i])
        if (value>best if preferred else (hit,value)>(best_hit,best)):
            best,best_hit=value,hit;best_actions=actions[i].clone()
            best_preview=series(capture,i,iteration,score,result,f'Best complete whip · iteration {iteration}')
            best_terms={k:float(v[i]) for k,v in terms.items()}
            best_metrics=dict(best_success=hit,best_failed=False,best_minimum_tip_distance_m=float(result['minimum_tip_distance_m'][i]),
                best_candidate_wave_stages=int(env.wave_stage[i]),cutoff=int(result['cutoffs'][i]),
                scored_duration_s=float(result['duration_s'][i]),legacy_reward=float(result['reward'][i]))
    def winner():
        if preferred:return int(score.argmax())
        valid_hits=result['success']&~result['failed']
        return int(score.masked_fill(~valid_hits,-torch.inf).argmax()) if bool(valid_hits.any()) else int(score.argmax())
    record_candidate(winner(),0)
    atomic_json(job/'initialization.json',dict(type='diverse_control_point_wave_screen',candidate_count=samples,
        selected_indices=chosen,objective_components=best_terms,**best_metrics))
    np.savez_compressed(job/'initial_screen.npz',scores=score.cpu().numpy(),legacy_reward=result['reward'].cpu().numpy(),
        distance_m=result['minimum_tip_distance_m'].cpu().numpy(),failed=result['failed'].cpu().numpy(),
        success=result['success'].cpu().numpy(),controls=controls.cpu().numpy(),**{k:v.cpu().numpy() for k,v in terms.items()})
    reason='iteration_ceiling'
    while not s['iterations'] or iteration<s['iterations']:
        iteration+=1;progress(job,stage='Optimizing complete 1.5 s whip',iteration=iteration)
        noise=torch.randn(groups,per,points,3,device=device,dtype=torch.float64,generator=rng)
        scales=env.tensor(s['control_point_noise_scales'])
        noise*=scales[torch.arange(per,device=device)%len(scales)][None,:,None,None]
        sampled=means[:,None]+noise
        interpolated=torch.tanh(offset+torch.einsum('hp,gbpc->gbhc',basis,sampled)).reshape(samples,steps,3)
        mean_actions=torch.tanh(offset+torch.einsum('hp,gpc->ghc',basis,means))
        actions=torch.cat((interpolated,mean_actions,best_actions[None],native[None]))
        env.branch_from(actual);capture=capture_factory()
        result=env.rollout(actions=actions,observer=capture);score,terms=capture.score(env,result,actions,weights)
        if not bool(torch.isfinite(score).any()):raise ValueError('All complete trajectories infeasible')
        importance,temperatures,ess=adaptive_weights(score[:samples].reshape(groups,per),s['target_ess_fraction'])
        updated=(importance[:,:,None,None]*sampled).sum(1)
        means=torch.where((importance.sum(-1)>0)[:,None,None],updated,means)
        record_candidate(winner(),iteration)
        if not math.isfinite(anchor) or best>anchor+max(.1,abs(anchor)*.005):anchor=best;improved_at=iteration
        row=dict(iteration=iteration,best_reward=best,elapsed_s=time.perf_counter()-start,
            success=float(result['success'][:samples].double().mean()),failures=float(result['failed'][:samples].double().mean()),
            effective_samples=float(ess.sum()),temperatures=temperatures.cpu().tolist(),
            effective_samples_per_proposal=ess.cpu().tolist(),objective_components=best_terms,**best_metrics)
        if cfg.get('visualization',{}).get('live_mppi',True):
            row['live_snapshot_seconds']=publish(job,actual,capture,score,result,best_preview,iteration)
        history.append(row);atomic_json(job/'history.json',history)
        temporary=job/'plan.tmp.npz'
        cutoff=best_metrics['cutoff'];selected=best_actions[:cutoff]
        np.savez_compressed(temporary,normalized_jerk=selected.cpu().numpy(),proposal_mean=means.cpu().numpy(),
            plan_complete=False,committed_steps=len(selected),planner_mode='offline_control_point_mppi_v1')
        replace_with_retry(temporary,job/'plan.npz')
        temp=job/'checkpoints/search.tmp.pt'
        torch.save(dict(iteration=iteration,means=means,rng=rng.get_state(),best_actions=best_actions,
            best=best,best_hit=best_hit,best_metrics=best_metrics,anchor=anchor,improved_at=improved_at),temp)
        replace_with_retry(temp,job/'checkpoints/search.pt')
        progress(job,stage='Complete whip update',**row)
        if iteration>=s['minimum_iterations'] and iteration-improved_at>=s['patience']:
            reason='reward_plateau';break
    # Independent batch-one replay before declaring a completed plan.
    check=capture_factory();verification=actual.rollout(actions=best_actions[None],observer=check,trace=True)
    verified,verified_terms=check.score(actual,verification,best_actions[None],weights)
    if bool(verification['failed'][0]) or bool(verification['success'][0])!=best_hit:
        raise ValueError('Selected trajectory fails independent outcome replay')
    if abs(float(verified[0])-best)>1e-5:raise ValueError('Batch and independent objective disagree')
    selected=best_actions[:int(verification['cutoffs'][0])]
    np.savez_compressed(job/'plan.tmp.npz',normalized_jerk=selected.cpu().numpy(),proposal_mean=means.cpu().numpy(),
        plan_complete=True,committed_steps=len(selected),planner_mode='offline_control_point_mppi_v1')
    replace_with_retry(job/'plan.tmp.npz',job/'plan.npz')
    summary=dict(iterations=iteration,stop_reason=reason,best_reward=best,**best_metrics,
        command_steps=len(selected),maneuver_duration_s=len(selected)/30,elapsed_s=time.perf_counter()-start,
        objective_components=best_terms,independent_score_difference=abs(float(verified[0])-best),
        planner_mode='offline_control_point_mppi_v1',evidence='Development M0 simulation only')
    atomic_json(job/'result.json',summary)
    return summary
