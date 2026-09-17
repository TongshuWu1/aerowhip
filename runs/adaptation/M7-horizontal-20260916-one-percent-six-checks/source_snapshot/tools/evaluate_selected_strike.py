"""Freeze and evaluate an explicit checkpoint, keeping development/holdout labels."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import io
import torch
import numpy as np
from run_ppo import build_agent,evaluate,_restore_agent
from simulator.workflow import read_json,atomic_json
from learning.experiment_records import finite_json,save_evaluation
from tools.source_snapshot import archive_sources

def main():
    parser=argparse.ArgumentParser();parser.add_argument('run')
    parser.add_argument('--checkpoint',default='checkpoints/latest.pt')
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--episodes',type=int,default=512)
    parser.add_argument('--recovery-duration',type=float)
    parser.add_argument('--output',required=True)
    parser.add_argument('--purpose',required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    source=Path(args.run);out=Path(args.output)
    if out.exists():raise ValueError('Choose a new output directory; evaluations are immutable.')
    raw=(source/args.checkpoint).read_bytes()
    saved=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    is_sac=saved.get('schema')=='force_sac_checkpoint_v1'
    m,t,p=[read_json(source/f'{name}.json') for name in ('model','task','ppo')]
    p['deployment']['validation_seed']=args.seed
    if args.recovery_duration is not None:p['deployment']['recovery_duration_s']=args.recovery_duration
    out.mkdir(parents=True);(out/'checkpoints').mkdir()
    (out/'checkpoints/latest.pt').write_bytes(raw)
    for name,data in [('model',m),('task',t),('ppo',p)]:atomic_json(out/f'{name}.json',data)
    atomic_json(out/'evaluation_protocol.json',dict(purpose=args.purpose,source=str(source.resolve()),
        checkpoint=args.checkpoint,checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
        episodes=args.episodes,seed=args.seed,recovery_duration_s=p['deployment']['recovery_duration_s'],
        training_episodes=saved['episodes'],selection_uses_this_evaluation=False))
    archive_sources(out)
    if is_sac:
        from simulator.rollout import _build_sac_agent
        from learning.deployment_rollout import evaluate_deployment
        algorithm=read_json(source/'sac.json')
        atomic_json(out/'sac.json',algorithm)
        agent=_build_sac_agent(source,torch.device('cuda'),checkpoint=saved)
        agent.actor.load_state_dict(saved['actor'])
        result=evaluate_deployment(m,t,p,agent,episodes=args.episodes,batch_size=args.episodes,device=torch.device('cuda'))
    else:
        agent=build_agent(p,torch.device('cuda'));_restore_agent(agent,saved,load_optimizer=False)
        result=evaluate(m,t,p,agent,episodes=args.episodes,batch_size=args.episodes,device=torch.device('cuda'))
    result['training_episodes']=saved['episodes'];result['evaluation_purpose']=args.purpose
    save_evaluation(out,result)
    summary=finite_json(dict(result))
    trials=read_json(next((out/'validation').glob('*-trials.json')))
    travel=[r['maximum_execution_drone_displacement_m'] for r in trials]
    varied=[r for r in trials if not r['nominal']]
    summary.update(peak_drone_travel_p95_m=float(np.percentile(travel,95)),
        peak_drone_travel_max_m=float(max(travel)),
        varied_trials=len(varied),varied_hits=sum(r['success'] for r in varied))
    atomic_json(out/'result.json',summary);print(summary,flush=True)

if __name__=='__main__':main()
