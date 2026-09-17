"""Numerical probe of a fitted full model; never updates the PPO or active config."""
import argparse
from pathlib import Path
import time
import torch
from .historical_report import candidate_model
from .io import atomic_json,canonical_json_hash,sha256_file
from run_ppo import load_configs,build_agent,_load_checkpoint
from learning.point_force_env import PointForceWhipEnvironment
from learning.deployment_rollout import sample_batch,plan_batch,execute_batch


def check(job,cable_directory,checkpoint,batch_size,output_name):
    torch.set_num_threads(1)
    model=candidate_model(job,cable_directory=cable_directory)
    _,task,config=load_configs()
    config['deployment'].update(enabled=True,recovery_failure_penalty=0.,force_gain_fraction=0.,force_lag_max_s=0.)
    before=sha256_file(checkpoint)
    device=torch.device('cuda')
    agent=build_agent(config,device)
    _load_checkpoint(agent,checkpoint,load_optimizer=False)
    updates=agent.gradient_updates
    started=time.perf_counter()
    env=PointForceWhipEnvironment(model,task,config,batch_size=batch_size,device=device)
    batch=sample_batch(env,config['deployment'],torch.Generator(device=device).manual_seed(1729))
    with torch.no_grad():
        forces,cutoffs=plan_batch(env,agent,batch)
        score=execute_batch(env,batch,forces,cutoffs,config['deployment'])
    torch.cuda.synchronize()
    state=score.execution_state
    finite=bool(torch.isfinite(state.positions_m).all() and torch.isfinite(state.velocities_m_s).all())
    unchanged=before==sha256_file(checkpoint) and agent.gradient_updates==updates
    result=dict(passed=finite and not bool(score.failed.any()) and unchanged,
        candidate_model_sha256=canonical_json_hash(model),cable_directory=cable_directory,
        selected_policy_sha256=before,policy_unchanged=unchanged,ppo_trained=False,
        scope='Old actor is a numerical probe only; no training or policy performance claim',
        hardware=torch.cuda.get_device_name(),os='Windows',batch=batch_size,
        elapsed_s=time.perf_counter()-started,numerical_failures=int(score.failed.sum()),
        predicted_hits=int(score.episode_success.sum()),maximum_position_m=float(state.positions_m.abs().max()),
        duration_range_s=[float(cutoffs.min())*env.physics_dt_s,float(cutoffs.max())*env.physics_dt_s])
    atomic_json(job/(output_name+'_config.json'),model)
    atomic_json(job/(output_name+'.json'),result)
    print(result,flush=True)
    if not result['passed']:raise ValueError('Full model numerical check failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--cable-directory',required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--output-name',default='runtime_candidate')
    args=parser.parse_args()
    check(args.job,args.cable_directory,args.checkpoint,args.batch_size,args.output_name)
