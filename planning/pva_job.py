"""Immutable MPPI jobs for the fitted direct-PVA model."""
from copy import deepcopy
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

ROOT=Path(__file__).resolve().parents[1]


def freeze_model_assets(model,directory,*,source_root=None,portable=False):
    """Portable component paths, including historical absolute NN references."""
    frozen=snapshot_assets(model,directory,source_root=source_root)
    component=Path(frozen['fullstate_execution']['checkpoint'])
    payload=read_json(component);payload['residual']['checkpoint']='drone_residual.pt'
    atomic_json(component,payload)
    frozen['fullstate_execution']['sha256']=sha256_file(component)
    if portable:
        frozen['fullstate_execution']['checkpoint']='assets/drone_model.json'
        if frozen.get('motion_residual',{}).get('enabled'):
            frozen['motion_residual']['checkpoint']='assets/cable_residual.pt'
    return frozen


def settings_path(root,method):return Path(root)/'config/pva'/f'{method}.json'


def load_settings(root,method='mppi'):
    if method != 'mppi':raise ValueError('Only MPPI is supported; legacy policy training was retired')
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
    if cfg.get('method') != 'mppi':raise ValueError('Only MPPI is supported; legacy policy training was retired')
    if cfg.get('ppo_objective') or cfg.get('policy_timing') or cfg.get('reward',{}).get('joint_strike',0):
        raise ValueError('Legacy policy objectives and timing are not supported')
    from planning.position_spline import SCHEMA as SPLINE_SCHEMA
    if cfg.get('command_contract')==SPLINE_SCHEMA and cfg.get('trajectory_objective',{}).get('schema')=='preferred_fold_v1':
        old=deepcopy(cfg);old['command_contract']=SCHEMA
        validate_settings(old)
        if cfg.get('recovery'):
            from deployment.braking_recovery import validate as validate_recovery
            validate_recovery(cfg['recovery'])
        if cfg['mppi'].get('support_points')!=9:raise ValueError('Nine free B-spline position controls required')
        scales=cfg['mppi'].get('position_noise_scales_m',[])
        if not scales or any(not math.isfinite(v) or v<=0 for v in scales):raise ValueError('Positive finite position noise required')
        return
    from planning.strike_objective import enabled,validate_settings as validate_strike
    if enabled(cfg):return validate_strike(cfg)
    from learning.pva_success import criterion
    criterion(cfg['task'])
    from learning.two_target_whip import validate as validate_targets
    validate_targets(cfg)
    if cfg['command_contract']!=SCHEMA or cfg['method'] != 'mppi':raise ValueError('Direct PVA settings required')
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
    if cfg['reward'].get('early_hit_scale_s',1.)<=0:raise ValueError('Positive early-hit time scale required')
    if cfg['reward'].get('impact_scale_m_s',4.)<=0:raise ValueError('Positive impact speed scale required')
    if cfg['mppi']['samples']<2 or cfg['mppi']['temperature']<=0 or cfg['mppi']['noise_std']<=0 or not 0<=cfg['mppi']['noise_correlation']<1:raise ValueError('Invalid MPPI sampling parameters')
    if cfg['reward'].get('brake_progress',0.) and cfg.get('observation_contract')!='pva_whip_phase_v3':
        raise ValueError('Brake progress requires observable v3 brake credit')
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
    selection=s.get('selection_criterion','legacy_v1')
    if selection not in ('legacy_v1','tip_contact_then_score_v1'):raise ValueError('Unknown MPPI candidate selection criterion')
    if selection=='tip_contact_then_score_v1' and (cfg['method']!='mppi' or s.get('mode')!='open_loop'
            or s.get('parameterization')!='control_points' or cfg['task'].get('success_criterion') not in ('tip_contact_v1','ordered_two_target_v1')):
        raise ValueError('Contact-priority selection requires offline control-point MPPI with tip-contact success')
    if cfg['task'].get('require_wave',False):
        if not cfg['task'].get('require_pullback',False):
            raise ValueError('Travelling-bend task requires pullback')
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


def configured_development_review(cfg,source):
    review=cfg.get('development_model_review',{})
    if review and review.get('model_sha256')==sha256_file(source):
        return review.get('reason')
    return None


def prepare(root,settings,name,*,checkpoint=None,development_review=None):
    if checkpoint is not None:raise ValueError('Policy checkpoint continuation was retired; prepare a new MPPI job')
    root=Path(root).resolve();cfg=deepcopy(settings);validate_settings(cfg)
    cfg.setdefault('performance',dict(fused_ticks=True,fast_solve=True,fast_geometry=True))
    source=Path(cfg['model_path']);source=source if source.is_absolute() else root/source
    model=read_json(source)
    development_review=development_review or configured_development_review(cfg,source)
    if development_review is not None and (not isinstance(development_review,str) or not development_review.strip()):
        raise ValueError('A documented development review must identify the simulation-only use')
    if model.get('provenance',{}).get('fit_complete') is False and not development_review:
        raise ValueError('Model fit checks have not completed; wait for its published candidate or provide an explicit simulation development review')
    for key in ('motion_residual','fullstate_execution'):
        if model.get(key,{}).get('enabled') is False:continue
        p=Path(model[key]['checkpoint']);model[key]['checkpoint']=str(p if p.is_absolute() else source.parent/p)
    stamp=datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    directory=root/'runs'/'mppi_pva'/stamp
    directory.mkdir(parents=True)
    model=freeze_model_assets(model,directory);cfg['model_path']=str(source.resolve())
    atomic_json(directory/'model.json',model);atomic_json(directory/'settings.json',cfg)
    atomic_json(directory/'identity.json',dict(name=name.strip() or f'{cfg["method"].upper()} PVA',method=cfg['method'],
        success_criterion=cfg['task'].get('success_criterion','legacy_strike_v1'),
        model_source=str(source),model_source_sha256=sha256_file(source),command_contract=cfg['command_contract'],
        source_checkpoint=str(checkpoint) if checkpoint else None,
        source_checkpoint_sha256=sha256_file(checkpoint) if checkpoint else None,evidence='simulation only',
        development_review=development_review))
    if cfg.get('spline_seed_directory'):
        seed_root=Path(cfg['spline_seed_directory']);seed_root=seed_root if seed_root.is_absolute() else root/seed_root
        for filename in ('initial_proposal.npz','proposal_baselines.npz',cfg['trajectory_objective']['reference_file']):
            shutil.copy2(seed_root/filename,directory/filename)
    snapshot=directory/'source_snapshot'
    for folder in ('learning','planning','simulator','deployment','experimental_data','tools'):
        for path in (root/folder).rglob('*.py'):
            if '__pycache__' not in path.parts:
                target=snapshot/path.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    atomic_json(directory/'source_manifest.json',{p.relative_to(snapshot).as_posix():sha256_file(p) for p in snapshot.rglob('*.py')})
    atomic_json(directory/'status.json',dict(status='prepared',stage='ready',attempts=0))
    return directory,[sys.executable,'-u',str(snapshot/'tools/run_pva.py'),'--job',str(directory)]


def progress(job,**state):
    if (job/'STOP').exists():raise InterruptedError('Stopped by request at a rollout/update boundary')
    atomic_json(job/'status.json',dict(status='running',pid=os.getpid(),**state));print(json.dumps(state),flush=True)


def whiten(values,rho):
    return torch.cat((values[:,:1],(values[:,1:]-rho*values[:,:-1])/math.sqrt(1-rho*rho)),1)


def mppi(job,model,cfg):
    from planning.strike_objective import enabled
    if enabled(cfg):
        from planning.strike_mppi import optimize
        return optimize(job,model,cfg)
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
        result=mppi(job,model,cfg)
        atomic_json(job/'status.json',dict(status='completed',**result))
    except BaseException as exc:
        if started:
            state=read_json(job/'status.json',{});state.update(status='stopped' if isinstance(exc,InterruptedError) else 'failed',error=str(exc))
            atomic_json(job/'status.json',state)
        raise
    finally:lock.unlink(missing_ok=True)
