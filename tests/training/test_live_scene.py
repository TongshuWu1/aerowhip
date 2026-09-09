"""Live visualization observes the same stochastic PPO collection and return."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Calibrated CUDA model parity check')
def test_live_recording_preserves_actions_rewards_rng_and_update(tmp_path):
    from simulator.research_config import workspace_configs
    from learning.point_force_env import PointForceWhipEnvironment, POINT_FORCE_OBSERVATION_DIM
    from learning.deployment_rollout import collect_deployment_rollout
    from learning.simple_ppo import PPORollout
    from run_ppo import build_agent, _load_checkpoint, configure_accelerator
    from tools.multidrone_data import read_live_batch
    root=Path(__file__).resolve().parents[2]
    model,task,config=deepcopy(workspace_configs(root))
    task['episode_duration_s']=.2  # Six native action packets; same production equations.
    device=torch.device('cuda');configure_accelerator(device)
    checkpoint=root/'runs/ppo/20260908-142124-902520-seed655/checkpoints/best_validation.pt'
    if not checkpoint.exists():pytest.skip('Research checkpoint required')
    results=[]
    for enabled in (False,True):
        torch.manual_seed(671);torch.cuda.manual_seed_all(671)
        env=PointForceWhipEnvironment(model,task,config,batch_size=4,device=device)
        agent=build_agent(config,device);_load_checkpoint(agent,checkpoint,load_optimizer=False)
        rollout=PPORollout.allocate(env.control_step_count,4,POINT_FORCE_OBSERVATION_DIM,3,device=device)
        if enabled:env._live_scene_context=dict(artifact=str(tmp_path),episodes_before=100,batch_index=0)
        score=collect_deployment_rollout(env,agent,rollout)
        from dataclasses import fields
        data={field.name:getattr(rollout,field.name).clone() for field in fields(rollout)}
        data['rng']=torch.cuda.get_rng_state().clone()
        data['reward']=score.episode_reward.clone()
        agent.update(rollout,minibatch_size=32,epochs=1,generator=torch.Generator(device=device).manual_seed(909))
        data['weights']={key:v.clone() for key,v in agent.policy.state_dict().items()}
        results.append(data)
    for key in results[0]:
        if key=='weights':
            for name in results[0][key]:torch.testing.assert_close(results[0][key][name],results[1][key][name],rtol=0,atol=0)
        else:torch.testing.assert_close(results[0][key],results[1][key],rtol=0,atol=0)
    arrays,meta,folder=read_live_batch(tmp_path)
    np.testing.assert_array_equal(arrays['reward'],results[1]['reward'].cpu().numpy())
    np.testing.assert_array_equal(arrays['policy_actions'],results[1]['actions'].cpu().numpy())
    assert meta['batch_start_attempt']==101 and meta['batch_end_attempt']==104
    assert meta['mean_episode_reward']==float(arrays['reward'].mean())
    assert read_live_batch(tmp_path,meta['generation']) is None
    # A slot replaced before the pointer commit must never mix two generations.
    altered=dict(meta,generation='uncommitted')
    (folder/'replay.json').write_text(json.dumps(altered))
    assert read_live_batch(tmp_path) is None


def test_live_batch_reader_rejects_cross_run_cache(tmp_path):
    from tools.multidrone_data import read_live_batch
    folder=tmp_path/'live_scene';folder.mkdir()
    (folder/'latest.json').write_text(json.dumps(dict(slot='../elsewhere',generation='x')))
    assert read_live_batch(tmp_path) is None
