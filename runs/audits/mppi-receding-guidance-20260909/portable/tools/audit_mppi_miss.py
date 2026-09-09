"""Read-only short-horizon exploration and objective audit; no production fit/run."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import math
import numpy as np
import torch
from simulator.workflow import read_json
from experimental_data.io import atomic_json,sha256_file
from learning.pva_env import PVAEnvironment
from planning.pva_job import whiten

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    job=ROOT/'runs/mppi_pva/20260909-121945-649228';out=ROOT/'runs/audits/mppi-miss-20260909'
    out.mkdir(parents=True,exist_ok=False)
    cfg=read_json(job/'settings.json');scales=[.05,.2,.5,1.]
    size=1024;env=PVAEnvironment(read_json(job/'model.json'),cfg,root=job,batch_size=size*len(scales)+1)
    generator=torch.Generator(device=env.device).manual_seed(656)
    noise=torch.randn(size,env.steps,3,device=env.device,dtype=torch.float64,generator=generator)
    rho=cfg['mppi']['noise_correlation']
    for k in range(1,env.steps):noise[:,k]=rho*noise[:,k-1]+math.sqrt(1-rho*rho)*noise[:,k]
    with np.load(job/'plan.npz') as data:
        selected=data['normalized_jerk'].copy();mean=data['proposal_mean'].copy()
    actions=torch.cat([torch.tanh(noise*s) for s in scales]+[env.tensor(selected)[None]])
    result=env.rollout(actions=actions)
    report=dict(source_job=str(job),source_plan_sha256=sha256_file(job/'plan.npz'),horizon_s=cfg['task']['duration_s'],
        fitted_delay_s=env.delay,initial_tip_distance_m=float(env.initial_distance[0]),scales=[],
        evidence='One paired exploration batch per scale; no fit or production optimizer restart')
    for i,scale in enumerate(scales):
        r=slice(i*size,(i+1)*size);valid=~result['failed'][r];reward=result['reward'][r];index=int(reward.argmax())
        report['scales'].append(dict(noise_std=scale,candidates=size,feasible=int(valid.sum()),hits=int(result['success'][r].sum()),
            best_reward=float(reward.max()),best_failed=bool(result['failed'][r][index]),
            best_distance_m=float(result['minimum_tip_distance_m'][r][index]),
            closest_feasible_distance_m=float(result['minimum_tip_distance_m'][r][valid].min()) if bool(valid.any()) else None,
            command_max_displacement_m=float((result['packets'][r,:,:3]-env.origin0[r,None]).norm(dim=-1).max())))
    latent=torch.atanh(env.tensor(selected).clamp(-1+1e-12,1-1e-12))[None]
    prior_energy=.5*cfg['mppi']['temperature']*(whiten(latent,rho)/cfg['mppi']['noise_std']).square().sum()
    explicit_jerk=cfg['reward']['jerk']/30*env.tensor(selected).square().mean(-1).sum()
    packets=result['packets'][-1].cpu().numpy()
    report['saved_plan']=dict(reward=float(result['reward'][-1]),distance_m=float(result['minimum_tip_distance_m'][-1]),
        reference_displacement_at_end_m=(packets[-1,:3]-packets[0,:3]).tolist(),
        maximum_abs_jerk_m_s3=float(np.abs(selected*np.asarray(cfg['action']['jerk_limit_m_s3'])).max()),
        fixed_zero_prior_energy=float(prior_energy),explicit_jerk_reward_penalty=float(explicit_jerk),
        proposal_mean_abs_max=float(np.abs(mean).max()))
    # Inspect original saved PPO ghost at the same time; never regenerate/translate it.
    ppo=ROOT/'runs/rehearsals/20260909-030657-671710';meta=read_json(ppo/'rehearsal.json')
    with np.load(ppo/'rehearsal.npz') as data:
        mask=data['prediction_time_s']<=.4+1e-12
        distances=np.linalg.norm(data['cable_positions_m'][mask,-1]-data['target_position_m'],axis=-1)
        report['historical_ppo']=dict(rehearsal=str(ppo),whip_end_s=meta['whip_end_s'],hit_time_s=meta['predicted_hit_time_s'],
            closest_first_04s_distance_m=float(distances.min()),full_whip_distance_m=meta['minimum_tip_distance_m'],
            contract=meta['schema'],comparison='Historical force workflow, not same-contract PVA PPO')
    atomic_json(out/'result.json',report);print(report)
