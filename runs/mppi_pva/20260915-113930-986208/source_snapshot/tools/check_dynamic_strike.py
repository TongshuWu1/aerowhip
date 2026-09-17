"""Five-second GPU reward/optimizer and approach-feasibility audit; no worker launch."""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experimental_data.io import atomic_json, sha256_file
from simulator.workflow import read_json
from learning.point_force_env import PointForceWhipEnvironment, POINT_FORCE_OBSERVATION_DIM
from learning.simple_ppo import PPORollout
from learning.deployment_rollout import collect_deployment_rollout, sample_batch
from learning.research_rollout import execute_research_batch
from run_ppo import build_agent, configure_accelerator, validate_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); folder = Path(args.output).resolve(); folder.mkdir(parents=True)
    source = Path(args.config).resolve()
    model, task, config = [read_json(source/f'{n}.json') for n in ('model','task','ppo')]
    validate_contract(model,task,config)
    assert task['episode_duration_s'] == 5. and config['reward']['source']=='dynamic_strike_v1_20260908'
    device = torch.device('cuda'); configure_accelerator(device)
    torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
    env = PointForceWhipEnvironment(model,task,config,batch_size=16,device=device)
    agent = build_agent(config,device)
    rollout = PPORollout.allocate(env.control_step_count,16,POINT_FORCE_OBSERVATION_DIM,3,device=device)
    env._live_scene_context = dict(artifact=str(folder), episodes_before=0, batch_index=0)
    started = time.monotonic()
    score = collect_deployment_rollout(env,agent,rollout)
    torch.testing.assert_close(rollout.rewards[:,:,0].sum(0).double(),score.episode_reward,rtol=1e-6,atol=1e-5)
    torch.testing.assert_close(sum(score.episode_component_sums.values()),score.episode_reward,rtol=1e-10,atol=1e-9)
    assert torch.isfinite(score.episode_reward).all()
    weights = {k:v.clone() for k,v in agent.policy.state_dict().items()}
    metrics = agent.update(rollout, minibatch_size=config['ppo']['minibatch_transitions'],
        epochs=config['ppo']['update_epochs'], generator=torch.Generator(device=device).manual_seed(991))
    assert all(torch.isfinite(v).all() for v in agent.policy.state_dict().values())
    assert any(not torch.equal(v,weights[k]) for k,v in agent.policy.state_dict().items())
    print('Full five-second stochastic collection and PPO update passed.',flush=True)
    training = dict(attempts=16, action_packets=150, physics_steps=750,
        elapsed_s=time.monotonic()-started, mean_reward=float(score.episode_reward.mean()),
        failed=int(score.failed.sum()), metrics=asdict(metrics))

    # A modest feedforward preparation pulse through the SAME force -> reference
    # -> empirical drone -> cable path. A feasibility probe, never a PPO prior.
    env = PointForceWhipEnvironment(model,task,config,batch_size=4,device=device)
    settings=dict(config['deployment'], nominal_fraction=1.)
    batch=sample_batch(env,settings,torch.Generator(device=device).manual_seed(992))
    env.target=batch.target_position_m.clone();env.reset(batch.estimate)
    packet_t=torch.arange(150,dtype=env.dtype,device=device)/30
    u=(packet_t/3.).clamp(0,1)
    acceleration=.75/3.**2*(60*u-180*u*u+120*u*u*u)
    packet_force=env.hover_force_world_n.expand(150,3).clone()
    packet_force[:,0] += model['mass_measurement']['total_mass_kg']*acceleration
    forces=packet_force.repeat_interleave(5,dim=0)[:,None].expand(-1,4,-1).clone()
    cutoffs=torch.full((4,),750,dtype=torch.long,device=device)
    score=execute_research_batch(env,batch,forces,cutoffs,settings)
    pose=score.predicted_pose
    displacement=pose['position_attachment_m'][:,-1]-pose['position_attachment_m'][:,0]
    assert not bool(score.failed.any()), 'Preparation probe violates the existing simulation feasibility envelope'
    assert bool((displacement[:,0] > .5).all()), 'Probe did not demonstrate the required approach distance'
    probe=dict(meaning='Feasible preparation only, not a successful whip or hardware validation',
        final_attachment_displacement_m=displacement.cpu().tolist(),
        maximum_reference_tilt_deg=score.deployment['maximum_reference_tilt_deg'].cpu().tolist(),
        maximum_reference_speed_m_s=score.deployment['maximum_reference_speed_m_s'].cpu().tolist(),
        failed=int(score.failed.sum()),successes=int(score.episode_success.sum()))
    np.savez_compressed(folder/'approach_probe.npz', time_s=np.arange(751)/150,
        force_n=forces.cpu().numpy(),origin_m=pose['position_origin_m'].cpu().numpy(),
        attachment_m=pose['position_attachment_m'].cpu().numpy())
    report=dict(device=torch.cuda.get_device_name(),os=sys.platform,training=training,approach_probe=probe,
        config_sha256={n:sha256_file(source/f'{n}.json') for n in ('model','task','ppo')},
        reward_accounting_passed=True, optimizer_updated=True)
    atomic_json(folder/'verification.json',report)
    print(report,flush=True)


if __name__=='__main__': main()
