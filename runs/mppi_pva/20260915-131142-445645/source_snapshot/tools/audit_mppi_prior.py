"""Paired, bounded diagnostic of prior strength; no active setting/model changes."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import math
import torch
from simulator.workflow import read_json
from experimental_data.io import atomic_json
from learning.pva_env import PVAEnvironment
from planning.pva_job import whiten

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    job=ROOT/'runs/mppi_pva/20260909-121945-649228';out=ROOT/'runs/audits/mppi-miss-20260909/prior-comparison.json'
    if out.exists():raise FileExistsError(out)
    cfg=read_json(job/'settings.json');samples=1024;groups=2;sigma=.3;temperature=1.;rho=.7
    env=PVAEnvironment(read_json(job/'model.json'),cfg,root=job,batch_size=groups*(samples+1))
    mean=env.tensor([0.]*groups*env.steps*3).reshape(groups,env.steps,3)
    rng=torch.Generator(device=env.device).manual_seed(656);best=[-math.inf]*groups;history=[]
    for iteration in range(1,41):
        noise=torch.randn(samples,env.steps,3,device=env.device,dtype=torch.float64,generator=rng)*sigma
        for k in range(1,env.steps):noise[:,k]=rho*noise[:,k-1]+math.sqrt(1-rho*rho)*noise[:,k]
        latent=mean[:,None]+noise[None];actions=torch.tanh(torch.cat([latent,mean[:,None]],1))
        env.reset();result=env.rollout(actions=actions.flatten(0,1));rows=[]
        for group,strength in enumerate([1.,0.]):
            sl=slice(group*(samples+1),(group+1)*(samples+1));reward=result['reward'][sl]
            ratio=-.5*((whiten(latent[group],rho)/sigma).square()-(whiten(noise,rho)/sigma).square()).sum((1,2))
            weights=torch.softmax(reward[:-1]/temperature+strength*ratio,0)
            mean[group]=(weights[:,None,None]*latent[group]).sum(0)
            index=int(reward.argmax());score=float(reward[index])
            if score>best[group]:
                best[group]=score
                metrics=dict(reward=score,distance_m=float(result['minimum_tip_distance_m'][sl][index]),
                    hit=bool(result['success'][sl][index]),failed=bool(result['failed'][sl][index]))
            else:metrics=history[-1]['conditions'][group]['best']
            rows.append(dict(prior_strength=strength,best=metrics,mean_abs_max=float(mean[group].abs().max()),
                ess=float(1/weights.square().sum()),hits=int(result['success'][sl].sum())))
        history.append(dict(iteration=iteration,conditions=rows))
        atomic_json(out,dict(experiment='Paired 40-iteration diagnostic only; current task, reward, physics and limits unchanged',
            noise_std=sigma,temperature=temperature,samples=samples,history=history))
        if iteration%10==0:print(history[-1],flush=True)
