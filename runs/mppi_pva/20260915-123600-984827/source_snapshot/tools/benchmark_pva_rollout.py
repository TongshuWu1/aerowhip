"""Bounded identical-action comparison of complete PVA rollout backends."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import time
import torch
from learning.pva_env import PVAEnvironment,defaults
from experimental_data.current_adaptation import read


def benchmark(batch,seconds,fast_solve=False):
    source=Path('runs/adaptation/20260909-pva-M0-bootstrap/source_candidate/model.json')
    cfg=defaults();cfg['task']['duration_s']=seconds
    generator=torch.Generator(device='cuda').manual_seed(711)
    actions=torch.randn(batch,round(seconds*30),3,device='cuda',dtype=torch.float64,generator=generator).tanh()*.15
    results={};outputs={}
    for label,fast in [('reference',False),('fused',True)]:
        env=PVAEnvironment(read(source),cfg,root=source.parent,batch_size=batch,fused_ticks=fast,fast_geometry=fast,fast_solve=fast and fast_solve)
        times=[]
        for repetition in range(3):
            env.reset();torch.cuda.synchronize();start=time.perf_counter()
            result=env.rollout(actions=actions)
            torch.cuda.synchronize();times.append(time.perf_counter()-start)
            outputs[label]={**result,'q':env.state.positions_m.clone(),'v':env.state.velocities_m_s.clone(),
                'p':env.pose.position.clone(),'rotation':env.pose.rotation.clone(),'omega':env.pose.omega_tracking.clone()}
            print(json.dumps(dict(backend=label,batch=batch,repetition=repetition,seconds=times[-1])),flush=True)
        results[label]=times
        del env
    differences={}
    for name,a in outputs['reference'].items():
        b=outputs['fused'][name]
        if a.is_floating_point():
            differences[name]=float((a-b).abs().max())
            torch.testing.assert_close(a,b,atol=1e-8,rtol=1e-9)
        else:
            torch.testing.assert_close(a,b,atol=0,rtol=0)
    return dict(batch=batch,duration_s=seconds,timing=results,maximum_differences=differences,
        peak_allocated_bytes=torch.cuda.max_memory_allocated())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=64);p.add_argument('--seconds',type=float,default=.2)
    p.add_argument('--fast-solve',action='store_true')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();torch.set_num_threads(4)
    result=benchmark(args.batch,args.seconds,args.fast_solve);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
