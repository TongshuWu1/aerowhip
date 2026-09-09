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
    s=cfg['mppi']
    if s.get('mode','open_loop') not in ('open_loop','receding'):raise ValueError('Unknown MPPI mode')
    if s.get('mode')=='receding':
        h=s['horizon_s']
        if h<=0 or h>cfg['task']['duration_s'] or not math.isclose(h*30,round(h*30),abs_tol=1e-8):raise ValueError('MPPI lookahead must be whole 30 Hz intervals within the maneuver limit')
        if not 0<=s['control_prior']<=1:raise ValueError('MPPI control prior must be between zero and one')
        if any(s[k]<0 for k in ('terminal_distance','terminal_velocity','terminal_command_speed','terminal_command_acceleration')):raise ValueError('Terminal guidance weights must be nonnegative')
        if s.get('terminal_anchor',0)<0:raise ValueError('Terminal anchor guidance must be nonnegative')
        if any(s.get(k,0)<0 for k in ('strike_exit_speed','strike_exit_climb','strike_exit_acceleration')):raise ValueError('Strike exit costs must be nonnegative')


def prepare(root,settings,name,*,checkpoint=None):
    root=Path(root).resolve();cfg=deepcopy(settings);validate_settings(cfg)
    cfg.setdefault('performance',dict(fused_ticks=True,fast_solve=True,fast_geometry=True))
    source=Path(cfg['model_path']);source=source if source.is_absolute() else root/source
    model=read_json(source)
    if model.get('provenance',{}).get('fit_complete') is False:raise ValueError('Model fit checks have not completed; wait for its published candidate')
    for key in ('motion_residual','fullstate_execution'):
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
        source_checkpoint=str(checkpoint) if checkpoint else None,evidence='simulation only'))
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
        learning_rate=s['learning_rate'],entropy_coefficient=s['entropy_coefficient'],gamma=1.,gae_lambda=.95)
    agent.fused_metrics=True
    return agent


def load_policy(checkpoint,env,cfg,*,optimizer=False):
    payload=torch.load(checkpoint,map_location=env.device,weights_only=False)
    if payload.get('schema')!=CHECKPOINT_SCHEMA or payload.get('command_contract')!=SCHEMA:raise ValueError('Select a direct PVA PPO checkpoint')
    if payload['observation_dim']!=env.observation_dim:raise ValueError('Policy observation contract differs from this model/command delay')
    agent=make_agent(env,cfg);agent.policy.load_state_dict(payload['policy']);agent.value.load_state_dict(payload['value'])
    if optimizer:
        agent.policy_optimizer.load_state_dict(payload['policy_optimizer']);agent.value_optimizer.load_state_dict(payload['value_optimizer'])
        for opt in (agent.policy_optimizer,agent.value_optimizer):
            for group in opt.param_groups:group['lr']=cfg['training']['learning_rate']
    return agent


def save_checkpoint(path,agent,env,attempts):
    payload=agent.checkpoint();payload.update(schema=CHECKPOINT_SCHEMA,command_contract=SCHEMA,
        observation_dim=env.observation_dim,attempts=attempts,action_units='normalized XYZ jerk; multiply by saved m/s^3 bounds')
    temp=path.with_suffix('.tmp');torch.save(payload,temp);temp.replace(path)


def progress(job,**state):
    if (job/'STOP').exists():raise InterruptedError('Stopped by request at a rollout/update boundary')
    atomic_json(job/'status.json',dict(status='running',pid=os.getpid(),**state));print(json.dumps(state),flush=True)


def ppo(job,model,cfg):
    s=cfg['training'];seed=s['seed'];torch.manual_seed(seed);torch.set_num_threads(4)
    env=PVAEnvironment(model,cfg,root=job,batch_size=s['batch_size'],device=cfg['device'])
    agent=load_policy(job/'resume.pt',env,cfg,optimizer=True) if (job/'resume.pt').exists() else make_agent(env,cfg)
    generator=torch.Generator(device=env.device).manual_seed(seed)
    evaluation=PVAEnvironment(model,cfg,root=job,batch_size=min(128,s['batch_size']),device=cfg['device'])
    rollout=PPORollout.allocate(env.steps,env.batch_size,env.observation_dim,3,device=env.device)
    history=[];episodes=[];best=-math.inf;anchor=-math.inf;last_improvement=0;reason='safety_ceiling';attempts=0
    start=time.perf_counter();update=0
    while attempts<s['maximum_attempts']:
        progress(job,stage='PPO rollout',attempts=attempts,update=update,success=history[-1]['success'] if history else None)
        obs=env.reset(randomize=True,generator=generator)
        for k in range(env.steps):
            action,lp,val=agent.act(obs)
            following,reward,done,mask=env.step(action)
            rollout.observations[k]=obs;rollout.actions[k]=action;rollout.log_probabilities[k]=lp
            rollout.values[k]=val;rollout.rewards[k]=reward;rollout.dones[k]=done;rollout.masks[k]=mask
            obs=following
        metrics=agent.update(rollout,minibatch_size=s['minibatch_size'],epochs=s['epochs'],generator=generator)
        attempts+=env.batch_size;update+=1
        row=dict(update=update,attempts=attempts,reward=float(env.total.mean()),success=float(env.success.double().mean()),
            failures=float(env.failed.double().mean()),minimum_tip_distance_m=float(env.minimum_distance.mean()),
            elapsed_s=time.perf_counter()-start,**asdict(metrics))
        # One row per attempted episode is kept for the requested episode plots.
        with (job/'episodes.csv').open('a',encoding='utf-8') as stream:
            if attempts==env.batch_size:stream.write('attempt,reward,success,failed,duration_s,minimum_tip_distance_m\n')
            data=torch.stack((env.total,env.success.double(),env.failed.double(),env.termination_time,env.minimum_distance),1).cpu().numpy()
            for i,values in enumerate(data,attempts-env.batch_size+1):stream.write(str(i)+','+','.join(str(v) for v in values)+'\n')
        if update%s['evaluate_every_updates']==0 or attempts>=s['maximum_attempts']:
            evaluation.reset(randomize=True,generator=torch.Generator(device=env.device).manual_seed(seed+1000))
            result=evaluation.rollout(policy=agent.deterministic_action)
            score=float(result['reward'].mean());row.update(evaluation_reward=score,evaluation_success=float(result['success'].double().mean()))
            if score>best:
                best=score;save_checkpoint(job/'checkpoints/best.pt',agent,env,attempts)
            threshold=max(abs(anchor)*s['relative_improvement'],.1) if math.isfinite(anchor) else 0
            if not math.isfinite(anchor) or score>anchor+threshold:anchor=score;last_improvement=attempts
            if attempts>=s['minimum_attempts'] and attempts-last_improvement>=s['plateau_attempts']:reason='reward_plateau'
        history.append(row);atomic_json(job/'history.json',history)
        save_checkpoint(job/'checkpoints/latest.pt',agent,env,attempts)
        progress(job,stage='PPO update complete',**row)
        if reason=='reward_plateau':break
    if not (job/'checkpoints/best.pt').exists():save_checkpoint(job/'checkpoints/best.pt',agent,env,attempts)
    atomic_json(job/'result.json',dict(attempts=attempts,stop_reason=reason,best_evaluation_reward=best,evidence='simulation development scenarios'))
    return dict(attempts=attempts,stop_reason=reason,checkpoint=str(job/'checkpoints/best.pt'))


def whiten(values,rho):
    return torch.cat((values[:,:1],(values[:,1:]-rho*values[:,:-1])/math.sqrt(1-rho*rho)),1)


def mppi(job,model,cfg):
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
            atomic_json(job/'status.json',state)
        raise
    finally:lock.unlink(missing_ok=True)
