"""Cold-start MPPI window diagnostics; candidate evidence, not executed plans."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from copy import deepcopy
import math,time,argparse
import numpy as np
import torch
from learning.pva_env import PVAEnvironment
from planning.pva_job import load_settings
from experimental_data.io import atomic_json,sha256_file
from simulator.workflow import read_json
from planning.mppi_receding import terminal_value

ROOT=Path(__file__).resolve().parents[1]

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--settings',type=Path);parser.add_argument('--horizons',type=float,nargs='+',default=[1.2,2.])
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    cfg=read_json(args.settings) if args.settings else load_settings(ROOT,'mppi');source=ROOT/cfg['model_path'];model=read_json(source)
    report=dict(model_sha256=sha256_file(source),evidence='Historical M1 candidate-window simulation only; no committed maneuver or flight',conditions=[])
    for horizon in args.horizons:
        setting=deepcopy(cfg);setting['mppi'].update(horizon_s=horizon,noise_std=.05,control_prior=0.)
        for key in setting['mppi']:
            if key.startswith('terminal_'):setting['mppi'][key]=0.
        atomic_json(out/f'settings-{horizon}.json',setting)
        env=PVAEnvironment(model,setting,root=source.parent,batch_size=1025)
        n=round(horizon*30);mean=env.tensor(np.zeros((n,3)));rng=torch.Generator(device=env.device).manual_seed(656)
        best=-math.inf;anchor=-math.inf;last=0;iteration=0;history=[];start=time.perf_counter()
        while True:
            if (out/'STOP').exists():raise InterruptedError('Diagnostic stopped')
            iteration+=1
            noise=torch.randn(1024,n,3,device=env.device,dtype=torch.float64,generator=rng)*.05
            for k in range(1,n):noise[:,k]=.7*noise[:,k-1]+math.sqrt(1-.7**2)*noise[:,k]
            latent=torch.cat((mean[None]+noise,mean[None]),0)
            env.reset();r=env.rollout(actions=torch.tanh(latent),max_steps=n)
            score=(r['reward']+terminal_value(env,setting['mppi'])).masked_fill(r['failed'],-torch.inf)
            if not bool(torch.isfinite(score).any()):raise ValueError('No feasible candidates')
            if bool(torch.isfinite(score[:-1]).any()):
                weights=torch.softmax(score[:-1],0);mean=(weights[:,None,None]*latent[:-1]).sum(0)
            value,index=score.max(0);index=int(index)
            if float(value)>best:
                best=float(value);metrics=dict(score=best,reward=float(r['reward'][index]),hit=bool(r['success'][index]),distance_m=float(r['minimum_tip_distance_m'][index]),duration_s=float(r['duration_s'][index]),
                    pull=bool(env.pull_ready[index]),reverse=bool(env.reverse_ready[index]))
                np.savez_compressed(out/f'candidate-{horizon}.npz',normalized_jerk=torch.tanh(latent[index]).cpu().numpy())
            if not math.isfinite(anchor) or best>anchor+max(.1,abs(anchor)*.005):anchor=best;last=iteration
            history.append(dict(iteration=iteration,elapsed_s=time.perf_counter()-start,best=metrics,hits=int(r['success'].sum()),feasible=int((~r['failed']).sum())))
            atomic_json(out/f'history-{horizon}.json',history)
            if iteration%5==0:print(horizon,history[-1],flush=True)
            if iteration>=20 and iteration-last>=15:break
        report['conditions'].append(dict(horizon_s=horizon,noise_std=.05,control_prior=0.,terminal_guidance=0.,iterations=iteration,elapsed_s=time.perf_counter()-start,best=metrics))
        atomic_json(out/'result.json',report)
        del env
    print(report,flush=True)

if __name__=='__main__':main()
