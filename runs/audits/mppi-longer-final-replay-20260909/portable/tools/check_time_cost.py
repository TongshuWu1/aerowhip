"""Bounded GPU audit of a saved reward continuation; never starts a trainer."""
import argparse
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
import time
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulator.workflow import read_json
from experimental_data.io import atomic_json, sha256_file
from learning.point_force_env import PointForceWhipEnvironment, POINT_FORCE_OBSERVATION_DIM
from learning.deployment_rollout import sample_batch, plan_batch, collect_deployment_rollout
from learning.research_rollout import execute_research_batch
from learning.simple_ppo import PPORollout
from run_ppo import build_agent, _load_checkpoint, configure_accelerator, validate_contract


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True)
    root=Path(__file__).resolve().parents[1]
    model,task,config=[read_json(args.config/f'{n}.json') for n in ('model','task','ppo')]
    previous=read_json(args.parent/'ppo.json');validate_contract(model,task,config)
    assert task==read_json(args.parent/'task.json')
    assert config['ppo']==previous['ppo']
    assert config['reward']['time_to_success_weight_per_s']==10.
    # All execution/planning dependencies except the intentional time finalizer
    # and launch workflow must be byte-identical to the parent's frozen source.
    manifest=read_json(args.parent/'source_snapshot_manifest.json')['files']
    changed=[]
    for name,digest in manifest.items():
        if name.endswith('.py'):
            assert sha256_file(args.parent/'source_snapshot'/name)==digest
            if sha256_file(root/name)!=digest:changed.append(name)
    assert set(changed)<= {'learning/research_rollout.py','simulator/workflow.py','simulator/gui/reward_explanations.py'},changed
    assert {'learning/research_rollout.py','simulator/workflow.py'} <= set(changed)
    spec=importlib.util.spec_from_file_location('learning._time_audit_parent',
        args.parent/'source_snapshot/learning/research_rollout.py')
    old_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(old_module)
    device=torch.device('cuda');configure_accelerator(device)
    torch.manual_seed(656);torch.cuda.manual_seed_all(656)
    agent=build_agent(config,device)
    checkpoint=args.parent/'checkpoints/latest.pt';digest=sha256_file(checkpoint)
    _load_checkpoint(agent,checkpoint,load_optimizer=True)
    old=PointForceWhipEnvironment(model,task,previous,batch_size=16,device=device)
    new=PointForceWhipEnvironment(model,task,config,batch_size=16,device=device)
    batch=sample_batch(old,previous['deployment'],torch.Generator(device=device).manual_seed(90651))
    started=time.monotonic()
    old_forces,old_cutoffs=plan_batch(old,agent,batch)
    print('Parent deterministic planning completed.',flush=True)
    new_forces,new_cutoffs=plan_batch(new,agent,batch)
    torch.testing.assert_close(old_forces,new_forces,rtol=0,atol=0)
    torch.testing.assert_close(old_cutoffs,new_cutoffs,rtol=0,atol=0)
    print('Amended deterministic planning exactly matches the parent.',flush=True)
    before=old_module.execute_research_batch(old,batch,old_forces,old_cutoffs,previous['deployment'])
    print('Parent complete-model execution completed.',flush=True)
    after=execute_research_batch(new,batch,new_forces,new_cutoffs,config['deployment'])
    for key in before.predicted_pose:
        if isinstance(before.predicted_pose[key],torch.Tensor):
            torch.testing.assert_close(before.predicted_pose[key],after.predicted_pose[key],rtol=0,atol=0,equal_nan=True)
    torch.testing.assert_close(before.execution_state.positions_m,after.execution_state.positions_m,rtol=0,atol=0)
    torch.testing.assert_close(before.episode_success,after.episode_success,rtol=0,atol=0)
    for key in before.episode_component_sums:
        if key!='time':torch.testing.assert_close(before.episode_component_sums[key],after.episode_component_sums[key],rtol=0,atol=0)
    torch.testing.assert_close(after.episode_reward-before.episode_reward,
        after.episode_component_sums['time']-before.episode_component_sums['time'],rtol=0,atol=1e-9)
    deployed=after.deployment['planned']
    torch.testing.assert_close(after.episode_component_sums['time'][deployed],
        -10.*after.deployment['duration_s'][deployed].to(after.episode_reward),rtol=0,atol=0)
    print('Same checkpoint: exact forces, cutoffs, predicted pose, cable endpoint state and hit outcomes; only time reward changed.',flush=True)
    new._live_scene_context=dict(artifact=str(args.output.resolve()),episodes_before=39936,batch_index=0)
    rollout=PPORollout.allocate(new.control_step_count,16,POINT_FORCE_OBSERVATION_DIM,3,device=device)
    score=collect_deployment_rollout(new,agent,rollout,generator=torch.Generator(device=device).manual_seed(90652))
    torch.testing.assert_close(rollout.rewards[:,:,0].sum(0).double(),score.episode_reward,rtol=1e-6,atol=1e-5)
    torch.testing.assert_close(sum(score.episode_component_sums.values()),score.episode_reward,rtol=0,atol=1e-9)
    assert torch.isfinite(score.episode_reward).all()
    weights={k:v.clone() for k,v in agent.policy.state_dict().items()}
    metrics=agent.update(rollout,minibatch_size=config['ppo']['minibatch_transitions'],
        epochs=config['ppo']['update_epochs'],generator=torch.Generator(device=device).manual_seed(991))
    assert all(torch.isfinite(v).all() for v in agent.policy.state_dict().values())
    assert any(not torch.equal(v,weights[k]) for k,v in agent.policy.state_dict().items())
    assert sha256_file(checkpoint)==digest
    from tools.multidrone_data import read_live_batch
    arrays,meta,_=read_live_batch(args.output)
    import numpy as np
    np.testing.assert_array_equal(arrays['reward'],score.episode_reward.cpu().numpy())
    report=dict(os=sys.platform,device=torch.cuda.get_device_name(),parent_checkpoint_sha256=digest,
        config_sha256={n:sha256_file(args.config/f'{n}.json') for n in ('model','task','ppo')},
        modified_parent_sources=changed,deterministic_cases=16,stochastic_cases=16,
        full_horizon_s=task['episode_duration_s'],same_actions_pose_cable_and_outcomes=True,
        only_time_component_changed=True,full_planned_time_charged=True,scene_reward_parity=True,
        mean_old_reward=float(before.episode_reward.mean()),mean_new_reward=float(after.episode_reward.mean()),
        deterministic_successes=int(after.episode_success.sum()),
        stochastic_failures=int(score.failed.sum()),finite_optimizer_update=asdict(metrics),elapsed_s=time.monotonic()-started)
    atomic_json(args.output/'verification.json',report)
    print(report,flush=True)


if __name__=='__main__':main()
