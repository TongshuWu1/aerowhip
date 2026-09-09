"""Fault injection in an isolated process; never modifies the running PPO."""
from pathlib import Path
import json
import runpy
import torch
import learning.fullstate_rollout as full
from simulator.cable import DderState
from learning.deployment_rollout import execute_batch


def main():
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[1]
    out=Path((root/'runs/audits/current_theory_audit.txt').read_text())
    temporary=out/'failure_probe';temporary.mkdir(exist_ok=True)
    make=runpy.run_path(str(root/'tests/training/test_fullstate_execution.py'))['make_environment']
    original=full.CudaGraphCableBoundary;results={}
    try:
        for bad in (False,True):
            env,batch,forces,cutoffs=make(temporary,device='cuda')
            target=env.target.clone()
            class FakeBoundary:
                def __init__(self,*args):self.step=0
                def __call__(self,state,boundary):
                    self.step+=1
                    q=state.positions_m.clone();v=torch.zeros_like(q)
                    q[:,-1]=target;v[:,-1,0]=5
                    if bad and self.step>=3:q[:,-1]=float('nan')
                    return DderState(q,v)
            full.CudaGraphCableBoundary=FakeBoundary
            score=execute_batch(env,batch,forces,cutoffs,env.ppo_config['deployment'])
            results[str(bad)]=dict(success=score.episode_success.tolist(),failed=score.failed.tolist(),
                reward=score.episode_reward.tolist(),numerical_penalty=score.episode_component_sums['numerical_failure'].tolist())
    finally:
        full.CudaGraphCableBoundary=original
    path=out/'probe_evidence.json';record=json.loads(path.read_text())
    record['synthetic_post_hit_failure_probe']=results
    path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
