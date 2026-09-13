"""Independent immutable jobs for direct-PVA PPO and MPPI."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import json
import math
import os
import shutil
import sys
import time
import numpy as np
import torch
from experimental_data.io import atomic_json,sha256_file
from simulator.workflow import read_json
from simulator.artifact_io import replace_with_retry
from simulator.research_config import snapshot_assets
from simulator.pva_commands import SCHEMA
from learning.pva_env import PVAEnvironment,defaults
from learning.simple_ppo import SimplePPOAgent,PPORollout
from planning.ppo_progress import evaluation_better

ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT_SCHEMA='jerk_pva_ppo_checkpoint_v1'


def freeze_model_assets(model,directory):
    """Portable component paths, including historical absolute NN references."""
    frozen=snapshot_assets(model,directory)
    component=Path(frozen['fullstate_execution']['checkpoint'])
    payload=read_json(component);payload['residual']['checkpoint']='drone_residual.pt'
    atomic_json(component,payload)
    frozen['fullstate_execution']['sha256']=sha256_file(component)
    return frozen


def settings_path(root,method):return Path(root)/'config/pva'/f'{method}.json'


def load_settings(root,method):
    path=settings_path(root,method)
    if path.exists():return read_json(path)
    cfg=defaults(method)
    if method=='mppi':
        old=read_json(Path(root)/'config/mppi.json',{})
        launch=old.get('launch_setup',{})
        cfg['launch']['origin_m']=launch.get('initial_tracking_origin_m',[0,0,1.225])
        cfg['launch']['target_m']=launch.get('target_position_m',[1.5,0,1.1])
        cfg['launch'].update(start_radius_m=0.,target_radius_m=0.)
        if cfg['mppi'].get('mode')!='receding':cfg['task']['duration_s']=float(old.get('horizon_s',2.))
    return cfg


def validate_settings(cfg):
    if cfg['command_contract']!=SCHEMA or cfg['method'] not in ('ppo','mppi'):raise ValueError('Direct PVA settings required')
    if not math.isclose(cfg['task']['duration_s']*30,round(cfg['task']['duration_s']*30),abs_tol=1e-8):raise ValueError('Maneuver duration must contain whole 30 Hz intervals')
    def finite_tree(x):
        if isinstance(x,dict):return all(finite_tree(v) for v in x.values())
        if isinstance(x,list):return all(finite_tree(v) for v in x)
        if isinstance(x,(int,float)):return math.isfinite(x)
        return True
    if not finite_tree(cfg):raise ValueError('Settings must be finite')
    for k in ('duration_s','target_radius_m','minimum_directed_speed_m_s'):
        if cfg['task'][k]<=0:raise ValueError(k+' must be positive')
    for k in ('origin_m','target_m'):
        if len(cfg['launch'][k])!=3:raise ValueError('XYZ launch required')
    if any(v<0 for v in cfg['reward'].values()):raise ValueError('Reward weights/scales must be nonnegative')
    if cfg['reward']['proximity_scale_m']<=0:raise ValueError('Positive proximity scale required')
    if cfg['mppi']['samples']<2 or cfg['mppi']['temperature']<=0 or cfg['mppi']['noise_std']<=0 or not 0<=cfg['mppi']['noise_correlation']<1:raise ValueError('Invalid MPPI sampling parameters')
    if cfg['training']['batch_size']<2:raise ValueError('PPO needs at least two environments')
    if not 0<cfg['training'].get('gae_lambda',.95)<=1:raise ValueError('GAE lambda must be in (0,1]')
    if cfg['training'].get('reward_scale',1.)<=0:raise ValueError('PPO reward scale must be positive')
    if not -10<=cfg['training'].get('initial_log_std',-.5)<=1:raise ValueError('PPO initial log std must be in [-10,1]')
    if cfg['reward'].get('brake_progress',0.) and cfg.get('observation_contract')!='pva_whip_phase_v3':
        raise ValueError('Brake progress requires observable v3 brake credit')
    if cfg['reward'].get('joint_strike',0.) and (cfg['method']!='ppo' or not cfg['task'].get('require_wave') or cfg.get('observation_contract')!='pva_whip_phase_v3'):
        raise ValueError('Joint strike shaping requires PPO with observable v3 wave/pullback state')
    if cfg['method']=='ppo' and cfg['reward'].get('reach_progress',0.) and cfg.get('observation_contract') not in ('pva_whip_phase_v2','pva_whip_phase_v3'):
        raise ValueError('PPO reach progress requires observable whip phase history')
    for key in ('samples','minimum_iterations','patience'):
        value=cfg['mppi'][key]
        if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ValueError('MPPI '+key+' must be a positive integer')
    ceiling=cfg['mppi']['iterations']
    if isinstance(ceiling,bool) or not isinstance(ceiling,int) or ceiling<0:raise ValueError('MPPI iteration limit must be a nonnegative integer; zero means no limit')
    if ceiling and cfg['mppi']['minimum_iterations']>ceiling:raise ValueError('MPPI minimum iterations exceeds its ceiling')
    for key in ('initial_minimum_iterations','initial_patience'):
        if key in cfg['mppi']:
            value=cfg['mppi'][key]
            if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ValueError('MPPI '+key+' must be a positive integer')
    if ceiling and cfg['mppi'].get('initial_minimum_iterations',1)>ceiling:raise ValueError('MPPI initial minimum iterations exceeds its ceiling')
    direction=cfg['task']['strike_direction'];jerk=cfg['action']['jerk_limit_m_s3']
    if len(direction)!=3 or sum(v*v for v in direction)<=0:raise ValueError('Nonzero XYZ strike direction required')
    if len(jerk)!=3 or any(v<=0 for v in jerk):raise ValueError('Positive XYZ jerk bounds required')
    if cfg['limits']['minimum_origin_z_m']>=cfg['limits']['maximum_origin_z_m']:raise ValueError('Minimum height must be below maximum height')
    if cfg['task'].get('require_pullback',False):
        for key in ('minimum_pull_distance_m','minimum_pull_speed_m_s','minimum_backward_distance_m','minimum_backward_speed_m_s'):
            if cfg['task'][key]<=0:raise ValueError('Pullback criterion '+key+' must be positive')
    s=cfg['mppi']
    if cfg['task'].get('require_wave',False):
        if not cfg['task'].get('require_pullback',False):
            raise ValueError('Travelling-bend task requires pullback')
        if cfg['method']=='ppo' and cfg.get('observation_contract') not in ('pva_whip_phase_v2','pva_whip_phase_v3'):
            raise ValueError('PPO travelling-bend task requires observable whip phase history')
        task=cfg['task']
        if not 0<task['wave_proximal_end']<task['wave_distal_start']<1:raise ValueError('Ordered wave bands required')
        for key in ('wave_proximal_angle_rad','wave_middle_angle_rad','wave_distal_angle_rad','wave_dwell_s'):
            if task[key]<=0:raise ValueError('Positive wave thresholds required')
    if 'noise_scales' in s:
        if not s['noise_scales'] or any(v<=0 for v in s['noise_scales']):raise ValueError('Positive noise scales required')
        if s.get('control_prior',0)!=0:raise ValueError('Mixture sampling currently requires zero control prior')
    if s.get('mode','open_loop') not in ('open_loop','receding'):raise ValueError('Unknown MPPI mode')
    if s.get('parameterization')=='control_points':
        if cfg['method']!='mppi' or s.get('mode')!='open_loop' or s.get('control_prior',0)!=0:
            raise ValueError('Control-point search requires offline MPPI with zero control prior')
        if not 2<=s.get('support_points',0)<=round(cfg['task']['duration_s']*30):raise ValueError('Invalid control-point count')
        if s.get('proposal_count',0)<1 or s['samples']%s['proposal_count']:raise ValueError('Samples must divide between proposals')
        if not 0<s.get('target_ess_fraction',0)<1:raise ValueError('Invalid target effective sample fraction')
        if s.get('proposal_parameterization')=='timed_baselines_v1':
            if cfg.get('trajectory_objective',{}).get('schema')!='preferred_fold_v1':raise ValueError('Timed proposals require the preferred-fold objective')
            if s['proposal_count']<2:raise ValueError('Timed search must retain both baseline families')
            knots=s.get('timing_source_knots',[])
            if len(knots)!=4 or knots[0]!=0 or knots[-1]!=1 or any(b<=a for a,b in zip(knots,knots[1:])):raise ValueError('Three ordered source timing phases required')
            for key in ('timing_noise_scales','strength_noise_scales'):
                if len(s.get(key,[]))!=len(s['control_point_noise_scales']) or any(x<=0 for x in s[key]):raise ValueError('Positive matching timing/strength noise scales required')
    if s.get('initialization','zero') not in ('zero','pullback','wave'):raise ValueError('Unknown MPPI initialization')
    if s.get('mode')=='receding':
        h=s['horizon_s']
        if h<=0 or h>cfg['task']['duration_s'] or not math.isclose(h*30,round(h*30),abs_tol=1e-8):raise ValueError('MPPI lookahead must be whole 30 Hz intervals within the maneuver limit')
        if not 0<=s['control_prior']<=1:raise ValueError('MPPI control prior must be between zero and one')
        if any(s[k]<0 for k in ('terminal_distance','terminal_velocity','terminal_command_speed','terminal_command_acceleration')):raise ValueError('Terminal guidance weights must be nonnegative')
        if s.get('terminal_anchor',0)<0:raise ValueError('Terminal anchor guidance must be nonnegative')
        if any(s.get(k,0)<0 for k in ('strike_exit_speed','strike_exit_climb','strike_exit_acceleration')):raise ValueError('Strike exit costs must be nonnegative')


def prepare(root,settings,name,*,checkpoint=None,development_review=None):
    root=Path(root).resolve();cfg=deepcopy(settings);validate_settings(cfg)
    cfg.setdefault('performance',dict(fused_ticks=True,fast_solve=True,fast_geometry=True))
    source=Path(cfg['model_path']);source=source if source.is_absolute() else root/source
    model=read_json(source)
    if development_review is not None and (cfg['method']!='mppi' or not isinstance(development_review,str) or not development_review.strip()):
        raise ValueError('A documented development review is only supported for MPPI simulation')
    if model.get('provenance',{}).get('fit_complete') is False and not development_review:
        raise ValueError('Model fit checks have not completed; wait for its published candidate or provide an explicit MPPI development review')
    for key in ('motion_residual','fullstate_execution'):
        if model.get(key,{}).get('enabled') is False:continue
        p=Path(model[key]['checkpoint']);model[key]['checkpoint']=str(p if p.is_absolute() else source.parent/p)
    resume=None
    if checkpoint:
        checkpoint=Path(checkpoint).resolve();resume=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if resume.get('schema')!=CHECKPOINT_SCHEMA:raise ValueError('Historical force PPO cannot initialize a jerk/PVA policy')
        old=read_json(checkpoint.parent.parent/'settings.json')
        if old['command_contract']!=cfg['command_contract']:raise ValueError('Action semantics changed')
    stamp=datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    directory=root/'runs'/('ppo_pva' if cfg['method']=='ppo' else 'mppi_pva')/stamp
    directory.mkdir(parents=True);(directory/'checkpoints').mkdir()
    model=freeze_model_assets(model,directory);cfg['model_path']=str(source.resolve())
    atomic_json(directory/'model.json',model);atomic_json(directory/'settings.json',cfg)
    atomic_json(directory/'identity.json',dict(name=name.strip() or f'{cfg["method"].upper()} PVA',method=cfg['method'],
        model_source=str(source),model_source_sha256=sha256_file(source),command_contract=SCHEMA,
        source_checkpoint=str(checkpoint) if checkpoint else None,
        source_checkpoint_sha256=sha256_file(checkpoint) if checkpoint else None,evidence='simulation only',
        development_review=development_review))
    if resume is not None:shutil.copy2(checkpoint,directory/'resume.pt')
    snapshot=directory/'source_snapshot'
    for folder in ('learning','planning','simulator','deployment','experimental_data','tools'):
        for path in (root/folder).rglob('*.py'):
            if '__pycache__' not in path.parts:
                target=snapshot/path.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    atomic_json(directory/'source_manifest.json',{str(p.relative_to(snapshot)):sha256_file(p) for p in snapshot.rglob('*.py')})
    atomic_json(directory/'status.json',dict(status='prepared',stage='ready',attempts=0))
    return directory,[sys.executable,'-u',str(snapshot/'tools/run_pva.py'),'--job',str(directory)]


def make_agent(env,cfg):
    s=cfg['training']
    agent=SimplePPOAgent(env.observation_dim,3,device=env.device,hidden_dim=s['hidden_dim'],
        learning_rate=s['learning_rate'],entropy_coefficient=s['entropy_coefficient'],gamma=1.,gae_lambda=s.get('gae_lambda',.95),
        initial_log_std=s.get('initial_log_std',-.5))
    agent.fused_metrics=True
    return agent


def load_policy(checkpoint,env,cfg,*,optimizer=False,policy_only=False):
    payload=torch.load(checkpoint,map_location=env.device,weights_only=False)
    if payload.get('schema')!=CHECKPOINT_SCHEMA or payload.get('command_contract')!=SCHEMA:raise ValueError('Select a direct PVA PPO checkpoint')
    if payload['observation_dim']!=env.observation_dim:raise ValueError('Policy observation contract differs from this model/command delay')
    agent=make_agent(env,cfg);agent.policy.load_state_dict(payload['policy'])
    if not policy_only:agent.value.load_state_dict(payload['value'])
    if optimizer and not policy_only:
        agent.policy_optimizer.load_state_dict(payload['policy_optimizer']);agent.value_optimizer.load_state_dict(payload['value_optimizer'])
        for opt in (agent.policy_optimizer,agent.value_optimizer):
            for group in opt.param_groups:group['lr']=cfg['training']['learning_rate']
    return agent


def save_checkpoint(path,agent,env,attempts):
    payload=agent.checkpoint();payload.update(schema=CHECKPOINT_SCHEMA,command_contract=SCHEMA,
        observation_dim=env.observation_dim,observation_contract=env.settings.get('observation_contract','pva_v1'),
        attempts=attempts,action_units='normalized XYZ jerk; multiply by saved m/s^3 bounds')
    temp=path.with_suffix('.tmp');torch.save(payload,temp);temp.replace(path)


def progress(job,**state):
    if (job/'STOP').exists():raise InterruptedError('Stopped by request at a rollout/update boundary')
    atomic_json(job/'status.json',dict(status='running',pid=os.getpid(),**state));print(json.dumps(state),flush=True)


def ppo(job,model,cfg):
    s=cfg['training'];seed=s['seed'];torch.manual_seed(seed);torch.set_num_threads(4)
    env=PVAEnvironment(model,cfg,root=job,batch_size=s['batch_size'],device=cfg['device'])
    agent=load_policy(job/'resume.pt',env,cfg,optimizer=True,policy_only=s.get('reset_value_on_resume',False)) if (job/'resume.pt').exists() else make_agent(env,cfg)
    generator=torch.Generator(device=env.device).manual_seed(seed)
    evaluation=PVAEnvironment(model,cfg,root=job,batch_size=min(128,s['batch_size']),device=cfg['device'])
    rollout=PPORollout.allocate(env.steps,env.batch_size,env.observation_dim,3,device=env.device)
    history=[];episodes=[];best=-math.inf;anchor=-math.inf;last_improvement=0;reason='safety_ceiling';attempts=0
    best_success=anchor_success=-math.inf;success_priority=s.get('success_priority',False)
    start=time.perf_counter();update=0
    while attempts<s['maximum_attempts']:
        phases={k:history[-1][k] for k in ('pull_fraction','release_fraction','wave_complete_fraction','mean_release_reach_fraction') if history and k in history[-1]}
        progress(job,stage='PPO rollout',attempts=attempts,update=update,success=history[-1]['success'] if history else None,**phases)
        obs=env.reset(randomize=True,generator=generator)
        for k in range(env.steps):
            action,lp,val=agent.act(obs)
            following,reward,done,mask=env.step(action)
            rollout.observations[k]=obs;rollout.actions[k]=action;rollout.log_probabilities[k]=lp
            rollout.values[k]=val;rollout.rewards[k]=reward*s.get('reward_scale',1.);rollout.dones[k]=done;rollout.masks[k]=mask
            obs=following
            if not bool(env.active.any()):break
        # Completed episodes already have terminal flags. Omit the unused tail
        # instead of simulating quarantined rows to the maximum episode length.
        used=k+1
        collected=PPORollout(**{name:getattr(rollout,name)[:used] for name in rollout.__dataclass_fields__})
        metrics=agent.update(collected,minibatch_size=s['minibatch_size'],epochs=s['epochs'],generator=generator)
        attempts+=env.batch_size;update+=1
        row=dict(update=update,attempts=attempts,reward=float(env.total.mean()),success=float(env.success.double().mean()),
            failures=float(env.failed.double().mean()),minimum_tip_distance_m=float(env.minimum_distance.mean()),
            elapsed_s=time.perf_counter()-start,**asdict(metrics))
        row.update(tip_contact_fraction=float(env.tip_contact.double().mean()),
            invalid_contact_fraction=float((env.contact&~env.success).double().mean()))
        if bool(env.success.any()):
            # Preserve one successful sampled trajectory per update for an
            # independent replay. This is not the updated deterministic policy.
            index=int(torch.where(env.success,env.total,-torch.inf).argmax())
            count=int(env.cutoff[index]);samples=job/'sampled_hits';samples.mkdir(exist_ok=True)
            path=samples/f'update-{update:06d}.npz'
            np.savez_compressed(path,normalized_jerk=torch.stack(env.actions[:count],1)[index].cpu().numpy(),
                origin_m=env.origin0[index].cpu().numpy(),target_m=env.target[index].cpu().numpy())
            atomic_json(path.with_suffix('.json'),dict(update=update,attempts=attempts,row=index,
                reward=float(env.total[index]),contact_time_s=float(env.termination_time[index]),
                kind='Sampled training trajectory before this policy update; deterministic evaluation is separate'))
        if cfg['task'].get('require_wave',False):
            row.update(pull_fraction=float(env.pull_ready.double().mean()),
                release_fraction=float(env.reverse_ready.double().mean()),
                wave_complete_fraction=float((env.wave_stage>=3).double().mean()),
                mean_wave_progress=float(env.wave_credit.mean()),mean_release_reach_fraction=float(env.reach_credit.mean()))
            if cfg.get('observation_contract')=='pva_whip_phase_v3':row['mean_brake_progress']=float(env.brake_credit.mean())
        # One row per attempted episode is kept for the requested episode plots.
        with (job/'episodes.csv').open('a',encoding='utf-8') as stream:
            if attempts==env.batch_size:stream.write('attempt,reward,success,failed,duration_s,minimum_tip_distance_m\n')
            data=torch.stack((env.total,env.success.double(),env.failed.double(),env.termination_time,env.minimum_distance),1).cpu().numpy()
            for i,values in enumerate(data,attempts-env.batch_size+1):stream.write(str(i)+','+','.join(str(v) for v in values)+'\n')
        if update%s['evaluate_every_updates']==0 or attempts>=s['maximum_attempts']:
            evaluation.reset(randomize=True,generator=torch.Generator(device=env.device).manual_seed(seed+1000))
            result=evaluation.rollout(policy=agent.deterministic_action)
            score=float(result['reward'].mean());row.update(evaluation_reward=score,evaluation_success=float(result['success'].double().mean()))
            row.update(evaluation_failures=float(result['failed'].double().mean()),
                evaluation_minimum_tip_distance_m=float(result['minimum_tip_distance_m'].mean()),
                evaluation_duration_s=float(result['duration_s'].mean()),
                evaluation_tip_contact_fraction=float(evaluation.tip_contact.double().mean()),
                evaluation_invalid_contact_fraction=float((evaluation.contact&~evaluation.success).double().mean()))
            if cfg['task'].get('require_wave',False):
                row.update(evaluation_pull_fraction=float(evaluation.pull_ready.double().mean()),
                    evaluation_release_fraction=float(evaluation.reverse_ready.double().mean()),
                    evaluation_wave_complete_fraction=float((evaluation.wave_stage>=3).double().mean()),
                    evaluation_mean_release_reach_fraction=float(evaluation.reach_credit.mean()))
                if cfg.get('observation_contract')=='pva_whip_phase_v3':row['evaluation_mean_brake_progress']=float(evaluation.brake_credit.mean())
            rate=row['evaluation_success']
            if evaluation_better(score,rate,best,best_success,success_priority=success_priority):
                best=score;best_success=rate;save_checkpoint(job/'checkpoints/best.pt',agent,env,attempts)
                atomic_json(job/'checkpoints/best_evaluation.json',dict(attempts=attempts,reward=score,success=rate,
                    selection='hit rate then reward' if success_priority else 'reward'))
            if evaluation_better(score,rate,anchor,anchor_success,relative=s['relative_improvement'],success_priority=success_priority):
                anchor=score;anchor_success=rate;last_improvement=attempts
            if attempts>=s['minimum_attempts'] and attempts-last_improvement>=s['plateau_attempts']:reason='reward_plateau'
        history.append(row);atomic_json(job/'history.json',history)
        save_checkpoint(job/'checkpoints/latest.pt',agent,env,attempts)
        progress(job,stage='PPO update complete',**row)
        if reason=='reward_plateau':break
    if not (job/'checkpoints/best.pt').exists():save_checkpoint(job/'checkpoints/best.pt',agent,env,attempts)
    atomic_json(job/'result.json',dict(attempts=attempts,stop_reason=reason,best_evaluation_reward=best,best_evaluation_success=best_success,evidence='simulation development scenarios'))
    return dict(attempts=attempts,stop_reason=reason,checkpoint=str(job/'checkpoints/best.pt'))


def whiten(values,rho):
    return torch.cat((values[:,:1],(values[:,1:]-rho*values[:,:-1])/math.sqrt(1-rho*rho)),1)


def mppi(job,model,cfg):
    if cfg['mppi'].get('parameterization')=='control_points':
        from planning.mppi_trajectory import optimize
        return optimize(job,model,cfg)
    if cfg['mppi'].get('mode')=='receding':
        from planning.mppi_receding import optimize
        return optimize(job,model,cfg)
    # The extra mean trajectory is evaluated for plan selection only. It is not
    # a draw from q and must never enter the importance-weighted update.
    s=cfg['mppi'];env=PVAEnvironment(model,cfg,root=job,batch_size=s['samples']+1,device=cfg['device'])
    generator=torch.Generator(device=env.device).manual_seed(s['seed']);mean=torch.zeros(env.steps,3,device=env.device,dtype=torch.float64)
    best=-math.inf;best_actions=None;best_metrics={};history=[];last_improvement=0;anchor=-math.inf;reason='iteration_ceiling'
    start=time.perf_counter();iteration=0
    while not s['iterations'] or iteration<s['iterations']:
        iteration+=1
        progress(job,stage='MPPI sampled trajectories',iteration=iteration,iterations=s['iterations'] or None)
        noise=torch.randn(s['samples'],env.steps,3,device=env.device,dtype=torch.float64,generator=generator)*s['noise_std']
        rho=s['noise_correlation']
        for k in range(1,env.steps):noise[:,k]=rho*noise[:,k-1]+math.sqrt(1-rho*rho)*noise[:,k]
        latent=mean[None]+noise;actions=torch.tanh(torch.cat((latent,mean[None]),0))
        env.reset();result=env.rollout(actions=actions)
        if not bool(torch.isfinite(result['reward']).all()):raise ValueError('Nonfinite MPPI rewards; no proposal update accepted')
        # Change-of-measure ratio p(z)/q(z) under the SAME correlated Gaussian.
        # tanh Jacobians cancel. No elites or PPO seed enter this MPPI update.
        prior=whiten(latent,rho)/s['noise_std'];proposal=whiten(noise,rho)/s['noise_std']
        log_ratio=-.5*(prior.square()-proposal.square()).sum((1,2))
        weights=torch.softmax(result['reward'][:-1]/s['temperature']+log_ratio,0)
        mean=(weights[:,None,None]*latent).sum(0)
        score,index=result['reward'].max(0)
        if float(score)>best:
            best=float(score);best_actions=actions[int(index)].clone()
            best_metrics=dict(best_iteration=iteration,best_success=bool(result['success'][int(index)]),
                best_failed=bool(result['failed'][int(index)]),
                best_minimum_tip_distance_m=float(result['minimum_tip_distance_m'][int(index)]))
        if not math.isfinite(anchor) or best>anchor+max(.1,abs(anchor)*.005):anchor=best;last_improvement=iteration
        row=dict(iteration=iteration,best_reward=best,batch_reward=float(result['reward'][:-1].mean()),
            success=float(result['success'][:-1].double().mean()),effective_samples=float(1/weights.square().sum()),
            failures=float(result['failed'][:-1].double().mean()),mean_reward=float(result['reward'][-1]),
            mean_failed=bool(result['failed'][-1]),elapsed_s=time.perf_counter()-start,**best_metrics)
        history.append(row);atomic_json(job/'history.json',history)
        temporary=job/'plan.tmp.npz'
        np.savez_compressed(temporary,normalized_jerk=best_actions.cpu().numpy(),proposal_mean=mean.cpu().numpy())
        replace_with_retry(temporary,job/'plan.npz')
        progress(job,stage='MPPI update complete',**row)
        if iteration>=s['minimum_iterations'] and iteration-last_improvement>=s['patience']:reason='reward_plateau';break
    summary=dict(iterations=iteration,stop_reason=reason,best_reward=best,plan=str(job/'plan.npz'),
        elapsed_s=time.perf_counter()-start,evidence='simulation only',**best_metrics)
    atomic_json(job/'result.json',summary)
    return summary


def run(job):
    job=Path(job).resolve();lock=job/'worker.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    started=False
    try:
        if read_json(job/'status.json')['status']!='prepared':raise ValueError('Job already started; create an explicit continuation')
        started=True
        cfg=read_json(job/'settings.json');model=read_json(job/'model.json');validate_settings(cfg)
        result=ppo(job,model,cfg) if cfg['method']=='ppo' else mppi(job,model,cfg)
        atomic_json(job/'status.json',dict(status='completed',**result))
    except BaseException as exc:
        if started:
            state=read_json(job/'status.json',{});state.update(status='stopped' if isinstance(exc,InterruptedError) else 'failed',error=str(exc))
            if isinstance(exc,InterruptedError) and read_json(job/'settings.json',{}).get('method')=='ppo':
                history=read_json(job/'history.json',[])
                if history and history[-1].get('attempts',0)>state.get('attempts',0):
                    state.update(history[-1]);state['stage']='Stopped after completed update'
            atomic_json(job/'status.json',state)
        raise
    finally:lock.unlink(missing_ok=True)
