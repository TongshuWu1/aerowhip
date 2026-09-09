"""Bounded native GPU check of terminal reward, PPO credit, scene and export."""
import argparse
from dataclasses import asdict
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from simulator.workflow import read_json
from experimental_data.io import atomic_json,sha256_file
from learning.point_force_env import PointForceWhipEnvironment,POINT_FORCE_OBSERVATION_DIM
from learning.deployment_rollout import collect_deployment_rollout
from learning.simple_ppo import PPORollout
from run_ppo import build_agent,_load_checkpoint,configure_accelerator,validate_contract


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True)
    model,task,config=[read_json(args.config/f'{n}.json') for n in ('model','task','ppo')]
    validate_contract(model,task,config);assert config['deployment']['termination']=='execution_success_or_timeout'
    source=args.parent/'checkpoints/latest.pt';source_hash=sha256_file(source)
    device=torch.device('cuda');configure_accelerator(device);torch.manual_seed(656);torch.cuda.manual_seed_all(656)
    agent=build_agent(config,device);_load_checkpoint(agent,source,load_optimizer=True)
    env=PointForceWhipEnvironment(model,task,config,batch_size=16,device=device)
    started=time.monotonic()
    score=collect_deployment_rollout(env,agent,generator=torch.Generator(device=device).manual_seed(90651))
    hits=score.episode_success;misses=~hits & ~score.failed
    torch.testing.assert_close(score.deployment['termination_time_s'][hits],score.episode_hit_time_s[hits],atol=1e-12,rtol=0)
    torch.testing.assert_close(score.deployment['termination_time_s'][misses],score.episode_reward.new_full((int(misses.sum()),),5.))
    torch.testing.assert_close(score.episode_component_sums['time'],-10.*score.deployment['termination_time_s'],rtol=0,atol=0)
    assert (score.terminal_steps<=750).all()
    deterministic=dict(cases=16,hits=int(hits.sum()),failures=int(score.failed.sum()),
        hit_times_s=score.episode_hit_time_s[hits].cpu().tolist(),terminal_steps=score.terminal_steps.cpu().tolist(),
        reward_time=score.episode_component_sums['time'].cpu().tolist())
    print('Deterministic hit/timeout durations and time costs passed.',flush=True)
    env._live_scene_context=dict(artifact=str(args.output.resolve()),episodes_before=44032,batch_index=0)
    rollout=PPORollout.allocate(env.control_step_count,16,POINT_FORCE_OBSERVATION_DIM,3,device=device)
    score=collect_deployment_rollout(env,agent,rollout,generator=torch.Generator(device=device).manual_seed(90652))
    expected=(score.terminal_steps+4)//5
    torch.testing.assert_close(rollout.masks[:,:,0].sum(0).long(),expected)
    torch.testing.assert_close(rollout.rewards[:,:,0].sum(0).double(),score.episode_reward,rtol=1e-6,atol=1e-5)
    torch.testing.assert_close(sum(score.episode_component_sums.values()),score.episode_reward,rtol=0,atol=1e-9)
    from tools.multidrone_data import read_live_batch
    arrays,meta,_=read_live_batch(args.output)
    np.testing.assert_array_equal(arrays['reward'],score.episode_reward.cpu().numpy())
    np.testing.assert_array_equal(arrays['cutoff'],score.terminal_steps.cpu().numpy())
    for row,end in enumerate(arrays['cutoff']):
        np.testing.assert_array_equal(arrays['cable'][end:,row],np.broadcast_to(arrays['cable'][end,row],arrays['cable'][end:,row].shape))
        np.testing.assert_array_equal(arrays['origin'][end:,row],np.broadcast_to(arrays['origin'][end,row],arrays['origin'][end:,row].shape))
    weights={k:v.clone() for k,v in agent.policy.state_dict().items()}
    metrics=agent.update(rollout,minibatch_size=config['ppo']['minibatch_transitions'],epochs=config['ppo']['update_epochs'],
        generator=torch.Generator(device=device).manual_seed(991))
    assert all(torch.isfinite(v).all() for v in agent.policy.state_dict().values())
    assert any(not torch.equal(weights[k],v) for k,v in agent.policy.state_dict().items())
    print('Stochastic terminal credit, frozen scene and PPO update passed.',flush=True)
    # Single nominal full CSV rehearsal from the original, unmodified checkpoint.
    trial=args.output/'export_source';(trial/'checkpoints').mkdir(parents=True)
    for n,v in zip(('model','task','ppo'),(model,task,config)):atomic_json(trial/f'{n}.json',v)
    shutil.copy2(source,trial/'checkpoints/policy.pt')
    from deployment.research_rehearsal import generate
    origin=np.asarray(task['initial_root_position_m'])-np.asarray(model['recorded_data']['optitrack_to_attachment_offset_body_m'])
    export=generate(trial/'checkpoints/policy.pt',args.output/'rehearsal',origin,task['target_position_m'],device='cuda')
    assert export['training_export_prefix_max_difference_m']<1e-8
    assert export['whip_end_s']<=5.+1e-9
    if export['predicted_valid_hit']:
        assert -1e-9<=export['whip_end_s']-export['predicted_hit_time_s']<1/30+1e-9
    assert sha256_file(source)==source_hash
    report=dict(device=torch.cuda.get_device_name(),os=sys.platform,source_checkpoint_sha256=source_hash,
        deterministic=deterministic,stochastic_cases=16,finite_optimizer_update=asdict(metrics),
        reward_component_and_mc_parity=True,scene_terminal_parity=True,post_terminal_credit_excluded=True,
        export=export,elapsed_s=time.monotonic()-started)
    atomic_json(args.output/'verification.json',report)
    print(report,flush=True)


if __name__=='__main__':main()
