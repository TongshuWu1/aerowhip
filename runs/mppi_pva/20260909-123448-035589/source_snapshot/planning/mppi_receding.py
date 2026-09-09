"""Offline receding-horizon MPPI: predict, commit one packet, shift, repeat."""
import math
import time
import numpy as np
import torch
from experimental_data.io import atomic_json
from simulator.artifact_io import replace_with_retry
from learning.pva_env import PVAEnvironment


def terminal_value(env,settings):
    """Explicit lookahead guidance; never counted as an actual strike reward."""
    delta=env.state.positions_m[:,-1]-env.target
    desired=env.direction*env.settings['task']['minimum_directed_speed_m_s']
    velocity=env.state.velocities_m_s[:,-1]-desired
    cost=settings['terminal_distance']*delta.square().sum(-1)
    cost+=settings['terminal_velocity']*velocity.square().sum(-1)
    cost+=settings['terminal_command_speed']*env.command[:,3:6].square().sum(-1)
    cost+=settings['terminal_command_acceleration']*env.command[:,6:9].square().sum(-1)
    return torch.where(env.success|env.failed,torch.zeros_like(cost),-cost)


def shift_proposal(mean):
    return torch.cat((mean[1:],torch.zeros_like(mean[:1])),0)


def publish_plan(job,env,mean,complete):
    temporary=job/'plan.tmp.npz'
    np.savez_compressed(temporary,normalized_jerk=torch.stack(env.actions,1)[0].cpu().numpy(),
        proposal_mean=mean.cpu().numpy(),committed_steps=env.index,plan_complete=complete,
        planner_mode='receding_pva_v1')
    replace_with_retry(temporary,job/'plan.npz')


@torch.no_grad()
def optimize(job,model,cfg):
    from planning.pva_job import progress,whiten
    s=cfg['mppi'];horizon=round(s['horizon_s']*30);samples=s['samples']
    actual=PVAEnvironment(model,cfg,root=job,device=cfg['device'])
    candidates=PVAEnvironment(model,cfg,root=job,batch_size=samples+1,device=cfg['device'])
    rng=torch.Generator(device=actual.device).manual_seed(s['seed'])
    mean=actual.tensor(np.zeros((horizon,3)));history=[];windows=[];total_iterations=0;start=time.perf_counter()
    while bool(actual.active[0]):
        count=min(horizon,actual.steps-actual.index);best=-math.inf;best_latent=None
        anchor=-math.inf;last_improvement=0;iteration=0;reason='iteration_ceiling'
        baseline=float(actual.total[0])
        while not s['iterations'] or iteration<s['iterations']:
            iteration+=1;total_iterations+=1
            progress(job,stage='MPPI lookahead optimization',iteration=total_iterations,window_iteration=iteration,
                command_step=actual.index,maneuver_time_s=actual.index/30,lookahead_s=count/30)
            noise=torch.randn(samples,horizon,3,device=actual.device,dtype=torch.float64,generator=rng)*s['noise_std']
            rho=s['noise_correlation']
            for k in range(1,horizon):noise[:,k]=rho*noise[:,k-1]+math.sqrt(1-rho*rho)*noise[:,k]
            latent=mean[None]+noise;evaluated=torch.cat((latent,mean[None]),0);actions=torch.tanh(evaluated)
            candidates.branch_from(actual)
            result=candidates.rollout(actions=actions,max_steps=count)
            score=result['reward']-baseline+terminal_value(candidates,s)
            if not bool(torch.isfinite(score).all()):raise ValueError('Nonfinite MPPI lookahead scores')
            feasible=~result['failed']
            if not bool(feasible.any()):raise ValueError('No feasible MPPI lookahead candidate; partial committed plan retained, no export')
            score=score.masked_fill(~feasible,-torch.inf)
            prior=whiten(latent[:,:count],rho)/s['noise_std'];proposal=whiten(noise[:,:count],rho)/s['noise_std']
            ratio=-.5*(prior.square()-proposal.square()).sum((1,2))
            if bool(feasible[:-1].any()):
                weights=torch.softmax(score[:-1]/s['temperature']+s['control_prior']*ratio,0)
                mean=(weights[:,None,None]*latent).sum(0)
            else:weights=torch.zeros_like(score[:-1]);mean=evaluated[-1].clone()
            value,index=score.max(0);index=int(index)
            if float(value)>best:
                best=float(value);best_latent=evaluated[index].clone()
                best_metrics=dict(predicted_hit=bool(result['success'][index]),predicted_failed=bool(result['failed'][index]),
                    predicted_distance_m=float(result['minimum_tip_distance_m'][index]))
            if not math.isfinite(anchor) or best>anchor+max(.1,abs(anchor)*.005):anchor=best;last_improvement=iteration
            row=dict(iteration=total_iterations,command_step=actual.index,window_iteration=iteration,
                lookahead_score=best,effective_samples=float(1/weights.square().sum()) if bool(weights.any()) else 0.,
                success=float(result['success'][:-1].double().mean()),failures=float(result['failed'][:-1].double().mean()),
                elapsed_s=time.perf_counter()-start,**best_metrics)
            history.append(row);atomic_json(job/'history.json',history)
            if iteration>=s['minimum_iterations'] and iteration-last_improvement>=s['patience']:reason='reward_plateau';break
        # Candidate predictions never become state. Apply only the first selected
        # action to the independent actual simulator, retaining delay/history.
        progress(job,stage='Committing next 30 Hz command',iteration=total_iterations,command_step=actual.index)
        actual.step(torch.tanh(best_latent[:1]),trace=True)
        mean=shift_proposal(best_latent)
        complete=not bool(actual.active[0]);publish_plan(job,actual,mean,complete)
        window=dict(command_step=actual.index,time_s=actual.index/30,iterations=iteration,stop_reason=reason,
            actual_reward=float(actual.total[0]),actual_success=bool(actual.success[0]),actual_failed=bool(actual.failed[0]),
            actual_minimum_tip_distance_m=float(actual.minimum_distance[0]),lookahead_score=best,**best_metrics)
        windows.append(window);atomic_json(job/'windows.json',windows)
        progress(job,stage='MPPI command committed',iteration=total_iterations,**window)
    summary=dict(iterations=total_iterations,command_steps=actual.index,lookahead_s=s['horizon_s'],
        stop_reason='modeled_hit' if bool(actual.success[0]) else 'infeasible' if bool(actual.failed[0]) else 'maneuver_time_limit',
        best_reward=float(actual.total[0]),best_success=bool(actual.success[0]),best_failed=bool(actual.failed[0]),
        best_minimum_tip_distance_m=float(actual.minimum_distance[0]),
        maneuver_duration_s=actual.index/30,scored_duration_s=float(actual.termination_time[0]),
        elapsed_s=time.perf_counter()-start,plan=str(job/'plan.npz'),evidence='simulation only',planner_mode='receding_pva_v1')
    atomic_json(job/'result.json',summary);return summary
