"""Offline receding-horizon MPPI: predict, commit one packet, shift, repeat."""
import math
import time
from itertools import product
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
    # A short window cannot see the complete cable swing. A geometric carrier
    # staging value represents that continuation without extending the rollout.
    length=float(np.sum(env.engine.cable.rest_lengths_m))
    anchor=env.target+env.direction*(.6*length)+env.target.new_tensor([0.,0.,.7*length])
    cost+=settings.get('terminal_anchor',0.)*(env.pose.position-anchor).square().sum(-1)
    cost+=settings['terminal_velocity']*velocity.square().sum(-1)
    cost+=settings['terminal_command_speed']*env.command[:,3:6].square().sum(-1)
    cost+=settings['terminal_command_acceleration']*env.command[:,6:9].square().sum(-1)
    # A strike with an accelerating/climbing command near the speed limit can
    # leave no room for the separate return planner. Score that handover state
    # explicitly; do not change the measured hit or the accumulated task return.
    v,a=env.command[:,3:6],env.command[:,6:9]
    forward_acceleration=((v*a).sum(-1)/v.norm(dim=-1).clamp_min(1e-9)).clamp_min(0)
    exit_cost=settings.get('strike_exit_speed',0.)*v.square().sum(-1)
    exit_cost+=settings.get('strike_exit_climb',0.)*v[:,2].clamp_min(0).square()
    exit_cost+=settings.get('strike_exit_acceleration',0.)*forward_acceleration.square()
    return torch.where(env.failed,torch.zeros_like(cost),torch.where(env.success,-exit_cost,-cost))


def shift_proposal(mean,tail=None):
    """Append an editable latent guess; it is never a committed command."""
    return torch.cat((mean[1:],torch.zeros_like(mean[:1]) if tail is None else tail.reshape(1,3)),0)


def pullback_seed_bank(env,horizon):
    """Initialization guesses only: native 30 Hz jerk, freely refined by MPPI."""
    bank=[np.zeros((horizon,3))]
    direction=env.direction.cpu().numpy();limits=env.limit.cpu().numpy()
    for strength,lift,forward_hold,backward_hold in product((15.,20.,25.,30.),(10.,15.,20.,25.,30.),(0,3,6,9),(9,12,15,18)):
        jerk=np.zeros((horizon,3));start=0
        for frames,amount in ((6,strength),(forward_hold,0.),(6,-2*strength),(backward_hold,0.),(6,strength)):
            end=min(horizon,start+frames);jerk[start:end]+=direction*amount;start=end
        jerk[:min(6,horizon),2]+=lift
        jerk[6:min(18,horizon),2]-=lift
        jerk[18:min(24,horizon),2]+=lift
        bank.append(np.clip(jerk/limits,-.999,.999))
    return env.tensor(np.stack(bank))


def wave_seed_bank(env,horizon,seed):
    """Diversify pulse timing/amplitude/lift before local native-jerk MPPI."""
    bank=[x for x in pullback_seed_bank(env,horizon).cpu().numpy()]
    rng=np.random.default_rng(seed)
    direction=env.direction.cpu().numpy();limits=env.limit.cpu().numpy()
    for _ in range(704):
        jerk=np.zeros((horizon,3));start=0
        ramp=int(rng.integers(4,11));hold=int(rng.integers(0,13))
        strength=rng.uniform(12,38);reverse=rng.uniform(1.5,2.5)
        for frames,amount in ((ramp,strength),(hold,0.),(ramp,-reverse*strength),
                (int(rng.integers(5,22)),0.),(ramp,(reverse-1)*strength)):
            end=min(horizon,start+frames);jerk[start:end]+=direction*amount;start=end
        zstart=int(rng.integers(0,5));zramp=int(rng.integers(4,10));lift=rng.uniform(8,32)
        for frames,amount in ((zramp,lift),(2*zramp,-lift),(zramp,lift)):
            end=min(horizon,zstart+frames);jerk[zstart:end,2]+=amount;zstart=end
        bank.append(np.clip(jerk/limits,-.999,.999))
    return env.tensor(np.stack(bank))


def sample_noise(actual,settings,samples,horizon,rng):
    noise=torch.randn(samples,horizon,3,device=actual.device,dtype=torch.float64,generator=rng)
    scales=settings.get('noise_scales',[settings['noise_std']])
    scale=noise.new_tensor(scales)[torch.arange(samples,device=actual.device)%len(scales)]
    noise*=scale[:,None,None]
    rho=settings['noise_correlation']
    for k in range(1,horizon):noise[:,k]=rho*noise[:,k-1]+math.sqrt(1-rho*rho)*noise[:,k]
    return noise


@torch.no_grad()
def initialize_proposal(actual,candidates,settings,horizon):
    if settings.get('initialization','zero')=='zero':return actual.tensor(np.zeros((horizon,3))),None
    bank=(wave_seed_bank(actual,horizon,settings['seed']) if settings.get('initialization')=='wave'
        else pullback_seed_bank(actual,horizon))
    # One batched screen seeds a nominal proposal; this is not an elite update.
    indices=torch.linspace(0,len(bank)-1,min(len(bank),candidates.batch_size),device=actual.device).long()
    actions=bank[indices]
    if len(actions)<candidates.batch_size:actions=torch.cat((actions,actions[:1].expand(candidates.batch_size-len(actions),-1,-1)))
    candidates.branch_from(actual);r=candidates.rollout(actions=actions,max_steps=horizon)
    score=(r['reward']+terminal_value(candidates,settings)).masked_fill(r['failed'],-torch.inf)
    if not bool(torch.isfinite(score).any()):raise ValueError('No feasible initialization candidate')
    value,index=score.max(0);index=int(index)
    details=dict(type=settings.get('initialization')+'_jerk_guess',candidate_count=len(indices),score=float(value),
        predicted_hit=bool(r['success'][index]),predicted_distance_m=float(r['minimum_tip_distance_m'][index]))
    if actual.settings['task'].get('require_wave',False):
        details.update(wave_stages=int(candidates.wave_stage[index]),wave_credit=float(candidates.wave_credit[index]),
            feasible_candidates=int((~r['failed']).sum()),wave_complete_candidates=int(((candidates.wave_stage==3)&~r['failed']).sum()))
    return torch.atanh(actions[index]),details


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
    history=[];windows=[];total_iterations=0;start=time.perf_counter()
    from .mppi_live import RolloutCapture,series,publish as publish_live
    live_enabled=cfg.get('visualization',{}).get('live_mppi',True);seed_sequence=None
    progress(job,stage='Initializing MPPI proposal',command_step=0,maneuver_time_s=0.)
    if (job/'continuation.npz').exists():
        with np.load(job/'continuation.npz') as data:prefix=data['prefix'];proposal=data['proposal']
        if prefix.ndim!=2 or prefix.shape[1]!=3 or not 0<len(prefix)<actual.steps or proposal.shape!=(horizon,3):raise ValueError('Invalid continuation dimensions')
        if not np.isfinite(prefix).all() or not np.isfinite(proposal).all() or np.abs(prefix).max()>1 or np.abs(proposal).max()>=1:raise ValueError('Invalid continuation actions')
        for action in actual.tensor(prefix):
            progress(job,stage='Replaying preserved MPPI prefix',command_step=actual.index,maneuver_time_s=actual.index/30)
            actual.step(action[None],trace=True)
            if not bool(actual.active[0]) or bool(actual.contact[0]):raise ValueError('Continuation prefix ends or contacts the target under the current task')
            windows.append(dict(command_step=actual.index,time_s=actual.index/30,iterations=0,reused_prefix=True,
                actual_reward=float(actual.total[0]),actual_success=False,actual_failed=False,
                actual_minimum_tip_distance_m=float(actual.minimum_distance[0]),stop_reason='preserved_prefix'))
        mean=torch.atanh(actual.tensor(proposal));initialization=dict(type='preserved_prefix',command_steps=len(prefix))
        publish_plan(job,actual,mean,False);atomic_json(job/'windows.json',windows)
    elif (job/'initial_proposal.npz').exists():
        with np.load(job/'initial_proposal.npz') as data:proposal=data['normalized_jerk']
        if proposal.ndim!=2 or proposal.shape[1]!=3 or not horizon<=len(proposal)<=actual.steps or not np.isfinite(proposal).all() or np.abs(proposal).max()>=1:
            raise ValueError('Initial proposal must contain bounded jerk spanning the horizon, within the maneuver length')
        seed_sequence=torch.atanh(actual.tensor(proposal));mean=seed_sequence[:horizon].clone()
        initialization=dict(type='editable_saved_proposal',committed_prefix_steps=0,
            seed_command_steps=len(proposal),lookahead_command_steps=horizon,
            note='All actions remain free; future seed commands enter as editable tail guesses as the horizon advances; no historical command prefix is committed')
    else:mean,initialization=initialize_proposal(actual,candidates,s,horizon)
    if initialization:atomic_json(job/'initialization.json',initialization)
    while bool(actual.active[0]):
        count=min(horizon,actual.steps-actual.index);best=-math.inf;best_latent=None
        anchor=-math.inf;last_improvement=0;iteration=0;reason='iteration_ceiling';best_preview=None
        minimum=s.get('initial_minimum_iterations',s['minimum_iterations']) if actual.index==0 else s['minimum_iterations']
        patience=s.get('initial_patience',s['patience']) if actual.index==0 else s['patience']
        baseline=float(actual.total[0])
        while not s['iterations'] or iteration<s['iterations']:
            iteration+=1;total_iterations+=1
            progress(job,stage='MPPI lookahead optimization',iteration=total_iterations,window_iteration=iteration,
                command_step=actual.index,maneuver_time_s=actual.index/30,lookahead_s=count/30)
            noise=sample_noise(actual,s,samples,horizon,rng)
            rho=s['noise_correlation']
            latent=mean[None]+noise;evaluated=torch.cat((latent,mean[None]),0);actions=torch.tanh(evaluated)
            candidates.branch_from(actual)
            capture=RolloutCapture() if live_enabled else None
            result=candidates.rollout(actions=actions,max_steps=count,**({'observer':capture} if capture is not None else {}))
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
                if capture is not None:best_preview=series(capture,index,total_iterations,score,result,f'Best so far · iteration {total_iterations}')
                best_metrics=dict(predicted_hit=bool(result['success'][index]),predicted_failed=bool(result['failed'][index]),
                    predicted_distance_m=float(result['minimum_tip_distance_m'][index]))
            if not math.isfinite(anchor) or best>anchor+max(.1,abs(anchor)*.005):anchor=best;last_improvement=iteration
            row=dict(iteration=total_iterations,command_step=actual.index,window_iteration=iteration,
                lookahead_score=best,effective_samples=float(1/weights.square().sum()) if bool(weights.any()) else 0.,
                success=float(result['success'][:-1].double().mean()),failures=float(result['failed'][:-1].double().mean()),
                elapsed_s=time.perf_counter()-start,**best_metrics)
            if cfg['task'].get('require_wave',False):
                row.update(wave_complete_fraction=float((candidates.wave_stage[:-1]==3).double().mean()),
                    best_candidate_wave_stages=int(candidates.wave_stage[index]))
            if capture is not None:
                try:row['live_snapshot_seconds']=publish_live(job,actual,capture,score,result,best_preview,total_iterations)
                except OSError as exc:
                    # A display/file-sharing failure must not discard a valid plan.
                    print('Live 3D snapshot unavailable: '+str(exc),flush=True)
                del capture
            history.append(row);atomic_json(job/'history.json',history)
            if iteration>=minimum and iteration-last_improvement>=patience:reason='reward_plateau';break
        # Candidate predictions never become state. Apply only the first selected
        # action to the independent actual simulator, retaining delay/history.
        progress(job,stage='Committing next 30 Hz command',iteration=total_iterations,command_step=actual.index)
        actual.step(torch.tanh(best_latent[:1]),trace=True)
        tail_index=actual.index+horizon-1
        tail=seed_sequence[tail_index] if seed_sequence is not None and tail_index<len(seed_sequence) else None
        mean=shift_proposal(best_latent,tail)
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
    if cfg['task'].get('require_wave',False):
        summary.update(wave_stages=int(actual.wave_stage[0]),wave_credit=float(actual.wave_credit[0]),
            wave_completion_time_s=float(actual.wave_completion_time[0]) if bool(torch.isfinite(actual.wave_completion_time[0])) else None,
            wave_definition='Ordered persistent dominant bend; kinematic proxy, not measured energy transfer')
    atomic_json(job/'result.json',summary);return summary
