"""Replay a saved PPO policy and report motion and first-failure diagnostics."""
from pathlib import Path
import argparse
import sys
import shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from experimental_data.io import atomic_json,sha256_file
from simulator.workflow import read_json
from simulator.pva_commands import integrate_jerk
from simulator.research_reference import reference_packet_validity
from learning.pva_env import PVAEnvironment
from planning.pva_job import load_policy


@torch.no_grad()
def audit(job,output,checkpoint='best.pt',batch=128):
    output.mkdir(parents=True,exist_ok=False)
    # Training replaces best/latest atomically. Copy once so the reported hash
    # and the replay always refer to the same weights while a worker continues.
    path=output/'policy.pt';shutil.copy2(job/'checkpoints'/checkpoint,path)
    checkpoint_hash=sha256_file(path)
    cfg=read_json(job/'settings.json');model=read_json(job/'model.json')
    env=PVAEnvironment(model,cfg,root=job,batch_size=batch,device=cfg['device'])
    agent=load_policy(path,env,cfg)
    env.reset(randomize=True,generator=torch.Generator(device=env.device).manual_seed(cfg['training']['seed']+1000))
    counts={};rows=[]
    for step in range(env.steps):
        active=env.active.clone();failed=env.failed.clone()
        action=agent.deterministic_action(env.observation())
        command=integrate_jerk(env.command,action*env.limit,env.control_dt)
        valid,metrics=reference_packet_validity(command[:,None],cfg['limits'])
        reasons={key:value[:,0] for key,value in {
            'command_speed':metrics['speed']>cfg['limits']['maximum_speed_m_s'],
            'command_specific_force':metrics['norm']>cfg['limits']['maximum_specific_force_m_s2'],
            'command_tilt':metrics['tilt']>cfg['limits']['maximum_tilt_deg'],
            'command_vertical_acceleration':metrics['vertical']<cfg['limits']['minimum_specific_vertical_m_s2'],
        }.items()}
        reasons['command_height']=(command[:,2]<cfg['limits']['minimum_origin_z_m'])|(command[:,2]>cfg['limits']['maximum_origin_z_m'])
        _,reward,_,_=env.step(action)
        newly_failed=env.failed&~failed
        known=torch.zeros_like(newly_failed)
        for key,value in reasons.items():
            mask=newly_failed&value;counts[key]=counts.get(key,0)+int(mask.sum());known|=mask
        counts['modeled_motion_or_numerical']=counts.get('modeled_motion_or_numerical',0)+int((newly_failed&~known).sum())
        def mean(t):return float(t[active].double().mean()) if active.any() else None
        rows.append(dict(time_s=(step+1)/30,active=int(env.active.sum()),new_failures=int(newly_failed.sum()),
            reward=mean(reward),total_reward=float(env.total.mean()),
            origin_x=mean(env.pose.position[:,0]),origin_z=mean(env.pose.position[:,2]),
            drone_forward_speed=mean((env.pose.velocity*env.direction).sum(-1)),
            tip_forward_speed=mean((env.state.velocities_m_s[:,-1]*env.direction).sum(-1)),
            command_acceleration=[mean(command[:,i]) for i in range(6,9)],
            mean_action=[mean(action[:,i]) for i in range(3)],
            pull_credit=mean(env.pull_credit),reverse_credit=mean(env.reverse_credit),
            wave_credit=mean(env.wave_credit),minimum_distance=float(env.minimum_distance.mean())))
        if not env.active.any():break
    summary=dict(job=str(job),checkpoint=checkpoint,checkpoint_sha256=checkpoint_hash,batch=batch,
        failure_counts=counts,mean_return=float(env.total.mean()),success_fraction=float(env.success.double().mean()),
        failed_fraction=float(env.failed.double().mean()),pull_fraction=float(env.pull_ready.double().mean()),
        release_fraction=float(env.reverse_ready.double().mean()),wave_fraction=float((env.wave_stage>=3).double().mean()),
        mean_duration_s=float(env.termination_time.mean()),mean_minimum_distance_m=float(env.minimum_distance.mean()),
        latent_std=agent.policy.log_std.exp().cpu().tolist(),evidence='Repeated simulation development scenarios, not independent validation')
    atomic_json(output/'summary.json',summary);atomic_json(output/'steps.json',rows)
    print(summary,flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--checkpoint',default='best.pt')
    parser.add_argument('--batch',type=int,default=128);args=parser.parse_args()
    audit(args.job,args.output,args.checkpoint,args.batch)
