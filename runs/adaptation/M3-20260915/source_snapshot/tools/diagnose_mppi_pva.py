"""Compare MPPI exploration scales in a single frozen-model GPU batch."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import math
import torch
from planning.pva_job import load_settings
from learning.pva_env import PVAEnvironment
from simulator.workflow import read_json
from experimental_data.io import atomic_json

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    cfg=load_settings(ROOT,'mppi');source=ROOT/cfg['model_path'];model=read_json(source)
    env=PVAEnvironment(model,cfg,root=source.parent,batch_size=1281)
    generator=torch.Generator(device=env.device).manual_seed(656)
    noise=torch.randn(256,env.steps,3,device=env.device,dtype=torch.float64,generator=generator)
    rho=cfg['mppi']['noise_correlation']
    for k in range(1,env.steps):noise[:,k]=rho*noise[:,k-1]+math.sqrt(1-rho*rho)*noise[:,k]
    scales=[.02,.05,.1,.2,.5]
    actions=torch.cat([noise.new_zeros(1,env.steps,3)]+[torch.tanh(noise*scale) for scale in scales])
    result=env.rollout(actions=actions)
    report=dict(hover={k:v[0].item() for k,v in result.items() if k in ('reward','success','failed','minimum_tip_distance_m','duration_s')},scales=[])
    for i,scale in enumerate(scales):
        rows=slice(1+i*256,1+(i+1)*256);valid=~result['failed'][rows];reward=result['reward'][rows]
        report['scales'].append(dict(scale=scale,feasible=int(valid.sum()),hits=int(result['success'][rows].sum()),
            best_reward=float(reward.max()),best_feasible_reward=float(reward[valid].max()) if bool(valid.any()) else None,
            closest_feasible_distance_m=float(result['minimum_tip_distance_m'][rows][valid].min()) if bool(valid.any()) else None))
    atomic_json(ROOT/'runs/audits/mppi-pva-continuation/sampling.json',report);print(report)
