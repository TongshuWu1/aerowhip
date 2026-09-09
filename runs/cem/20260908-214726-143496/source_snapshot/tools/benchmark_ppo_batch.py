"""Time one complete collection/update/validation from a frozen checkpoint.

Uses the selected run's frozen implementation and model assets. Benchmark
optimizer updates and scene recordings stay in the audit directory; this never
starts a production trainer or writes an active-run pointer.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--batch',type=int,required=True)
    p.add_argument('--output',required=True);args=p.parse_args()
    run=Path(args.run).resolve();folder=Path(args.output).resolve();folder.mkdir(parents=True)
    sys.path.insert(0,str(run/'source_snapshot'))
    import torch
    import run_ppo
    import learning.deployment_rollout as deployment
    from learning.point_force_env import PointForceWhipEnvironment,POINT_FORCE_OBSERVATION_DIM
    from learning.simple_ppo import PPORollout
    from learning.live_scene import TrainingSceneRecorder
    from experimental_data.io import atomic_json,sha256_file
    from simulator.workflow import read_json
    manifest=read_json(run/'source_snapshot_manifest.json')['files']
    if not all(sha256_file(run/'source_snapshot'/name)==h for name,h in manifest.items()):
        raise ValueError('Frozen training source changed')
    model,task,config=[read_json(run/f'{n}.json') for n in ('model','task','ppo')]
    device=torch.device('cuda');run_ppo.configure_accelerator(device)
    seed=int(config['seed']);torch.manual_seed(seed+100);torch.cuda.manual_seed_all(seed+100)
    timings={};start=time.perf_counter()
    env=PointForceWhipEnvironment(model,task,config,batch_size=args.batch,device=device)
    agent=run_ppo.build_agent(config,device)
    checkpoint=run_ppo._load_checkpoint(agent,run/'checkpoints/latest.pt',load_optimizer=True)
    rollout=PPORollout.allocate(env.control_step_count,args.batch,POINT_FORCE_OBSERVATION_DIM,3,device=device)
    torch.cuda.synchronize();timings['setup_s']=time.perf_counter()-start
    torch.cuda.reset_peak_memory_stats();free_before,total=torch.cuda.mem_get_info()
    original_publish=TrainingSceneRecorder.publish
    def publish(*a,**kw):
        torch.cuda.synchronize();start=time.perf_counter();result=original_publish(*a,**kw)
        torch.cuda.synchronize();timings['scene_publication_s']=time.perf_counter()-start
        return result
    TrainingSceneRecorder.publish=publish
    originals={name:getattr(deployment,name) for name in ('plan_batch','execute_batch')}
    for name,original in originals.items():
        def timed(*a,_name=name,_original=original,**kw):
            torch.cuda.synchronize();start=time.perf_counter();result=_original(*a,**kw)
            torch.cuda.synchronize();timings[_name+'_s']=time.perf_counter()-start
            print(_name,'finished',timings[_name+'_s'],flush=True)
            return result
        setattr(deployment,name,timed)
    env._live_scene_context=dict(artifact=str(folder),episodes_before=checkpoint['episodes'],batch_index=0)
    try:
        started=time.perf_counter()
        score=deployment.collect_deployment_rollout(env,agent,rollout)
        torch.cuda.synchronize();timings['collection_total_s']=time.perf_counter()-started
        for name,original in originals.items():setattr(deployment,name,original)
        torch.testing.assert_close(rollout.rewards[:,:,0].sum(0).double(),score.episode_reward,rtol=1e-6,atol=1e-5)
        started=time.perf_counter()
        metrics=agent.update(rollout,minibatch_size=config['ppo']['minibatch_transitions'],
            epochs=config['ppo']['update_epochs'],generator=torch.Generator(device=device).manual_seed(seed+1))
        torch.cuda.synchronize();timings['optimizer_s']=time.perf_counter()-started
        print('optimizer finished',timings['optimizer_s'],flush=True)
        started=time.perf_counter()
        validation=deployment.evaluate_deployment(model,task,config,agent,
            episodes=config['validation']['episodes'],batch_size=config['validation']['episodes'],device=device)
        torch.cuda.synchronize();timings['validation_s']=time.perf_counter()-started
        elapsed=sum(timings[k] for k in ('collection_total_s','optimizer_s','validation_s'))
        result=dict(status='PASS',batch=args.batch,checkpoint_attempts=checkpoint['episodes'],
            checkpoint_sha256=sha256_file(run/'checkpoints/latest.pt'),device=torch.cuda.get_device_name(),
            timings=timings,cycle_s=elapsed,attempts_per_s=args.batch/elapsed,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            gpu_total_gib=total/2**30,gpu_free_before_gib=free_before/2**30,
            gpu_free_after_gib=torch.cuda.mem_get_info()[0]/2**30,
            mean_reward=float(score.episode_reward.mean()),failures=int(score.failed.sum()),
            finite_rewards=bool(torch.isfinite(score.episode_reward).all()),optimizer=asdict(metrics),
            validation_reward=validation['mean_episode_reward'],
            note='Single full five-second cycle, includes first shape-specific graph capture and live publication; not convergence speed.')
        atomic_json(folder/'benchmark.json',result)
        print(json.dumps(result),flush=True)
    except torch.OutOfMemoryError as error:
        atomic_json(folder/'benchmark.json',dict(status='OUT_OF_MEMORY',batch=args.batch,error=str(error)))
        raise


if __name__=='__main__':main()
