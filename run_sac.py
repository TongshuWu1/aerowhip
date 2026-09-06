"""Train SAC as an initial-state-only force-sequence planner."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from learning import POINT_FORCE_OBSERVATION_DIM, PointForceWhipEnvironment, ReplayBuffer, SimpleSACAgent
from learning.deployment_rollout import evaluate_deployment
from learning.sac_deployment import collect_sac, critic_context_dimension
from learning.training_control import stoppable_training, TrainingStopped
from simulator.rollout import load_json, resolve_device
from run_ppo import _atomic_checkpoint, _atomic_json, validate_contract

ROOT = Path(__file__).resolve().parent
EXECUTION_MODE = 'initial_state_only_open_loop_once_with_pid_recovery'
AGENT_KEYS = ('hidden_dim', 'actor_learning_rate', 'critic_learning_rate',
              'temperature_learning_rate', 'gamma', 'target_smoothing_tau',
              'initial_temperature', 'stochastic_action_indices', 'maximum_gradient_norm',
              'initial_log_std','target_entropy')


def configs(config_directory=None):
    return tuple(load_json((Path(config_directory) if config_directory else ROOT / 'config') / name)
                 for name in ('model.json', 'task.json', 'ppo.json', 'sac.json'))


def build_agent(config, device, task=None):
    if task is None:
        task = load_json(ROOT / 'config/task.json')
    source = config['sac']
    return SimpleSACAgent(POINT_FORCE_OBSERVATION_DIM, 3, device=device,
                          critic_context_dim=critic_context_dimension(task),
                          action_prior=config.get('bootstrap'),
                          **{key: source[key] for key in AGENT_KEYS if key in source})


def checkpoint(agent, episodes, *, total_transitions=0, generator=None):
    result = {
        'schema': 'force_sac_checkpoint_v1', 'observation_dim': POINT_FORCE_OBSERVATION_DIM,
        'action_dim': 3, 'episodes': episodes, 'total_transitions': total_transitions,
        'gradient_updates': agent.gradient_updates, 'agent_config': agent.agent_config,
        'training_execution_mode': EXECUTION_MODE,
        'log_temperature': agent.log_temperature.detach().cpu(),
        'torch_rng_state': torch.random.get_rng_state(),
        'replay_buffer_saved': False,
    }
    for name in ('actor', 'critic1', 'critic2', 'target1', 'target2',
                 'actor_optimizer', 'critic_optimizer', 'temperature_optimizer'):
        result[name] = getattr(agent, name).state_dict()
    if generator is not None:
        result['sample_generator_state'] = generator.get_state()
    if agent.device.type == 'cuda':
        result['cuda_rng_state'] = torch.cuda.get_rng_state(agent.device)
    return result


def restore(agent, path, generator=None):
    payload = torch.load(path, map_location=agent.device, weights_only=False)
    if (payload.get('schema') != 'force_sac_checkpoint_v1'
            or payload.get('training_execution_mode') != EXECUTION_MODE
            or payload.get('agent_config', {}).get('critic_context_dim') != agent.critic_context_dim):
        raise ValueError('Continuation requires a SAC checkpoint from initial-state-only training.')
    if payload['agent_config'] != agent.agent_config:
        raise ValueError('SAC continuation settings differ from the saved checkpoint.')
    for name in ('actor', 'critic1', 'critic2', 'target1', 'target2',
                 'actor_optimizer', 'critic_optimizer', 'temperature_optimizer'):
        getattr(agent, name).load_state_dict(payload[name])
    with torch.no_grad():
        agent.log_temperature.copy_(payload['log_temperature'])
    agent.gradient_updates = int(payload['gradient_updates'])
    if 'torch_rng_state' in payload:
        torch.random.set_rng_state(payload['torch_rng_state'].cpu())
    if agent.device.type == 'cuda' and 'cuda_rng_state' in payload:
        torch.cuda.set_rng_state(payload['cuda_rng_state'].cpu(), agent.device)
    if generator is not None and 'sample_generator_state' in payload:
        generator.set_state(payload['sample_generator_state'].cpu())
    return payload


def validation_rank(result):
    return (result['hit_and_recovery_rate'], result['success_rate'],
            result['plan_success_rate'], result['mean_episode_reward'])


@stoppable_training(lambda: ROOT / 'runs/sac' / datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%S.%fZ'), return_status=True)
def run(artifact: Path | None, episodes_target: int | None, preflight: bool = False,
        device_name: str | None = None, batch_override: int | None = None,
        resume_checkpoint: Path | None = None, config_directory: Path | None = None):
    model, task, shared, algorithm = configs(config_directory)
    # The shared task/reward settings do not select SAC's actor initialization.
    validate_contract(model, task, {**shared,'bootstrap':algorithm.get('bootstrap')})
    if algorithm.get('schema') != 'point_force_sac_algorithm_v1':
        raise ValueError('Unsupported SAC configuration schema.')
    if not shared.get('deployment', {}).get('enabled') or algorithm['sac']['gamma'] != 1.0:
        raise ValueError('SAC uses initial-state-only training with undiscounted plan rewards.')
    device = resolve_device(device_name or algorithm['training']['device'])
    torch.set_num_threads(1)
    torch.manual_seed(int(algorithm['seed']))
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('highest')
    batch = 4 if preflight else int(batch_override or algorithm['training']['collection_batch'])
    target = batch if preflight else int(episodes_target or algorithm['training']['requested_episodes'])
    if batch < 1 or target < 1:
        raise ValueError('SAC episodes and parallel batch must be positive.')
    if preflight:
        task = {**task, 'episode_duration_s': .2}
    source = algorithm['sac']
    environment = PointForceWhipEnvironment(model, task, shared, batch_size=batch, device=device)
    agent = build_agent(algorithm, device, task)
    replay = ReplayBuffer(1024 if preflight else int(source['replay_capacity']),
                           POINT_FORCE_OBSERVATION_DIM, 3, device,
                           critic_context_dim=agent.critic_context_dim)
    generator = torch.Generator(device=device).manual_seed(int(algorithm['seed']) + 1)
    episodes = total_transitions = 0
    if resume_checkpoint is not None:
        parent = restore(agent, resume_checkpoint, generator)
        episodes, total_transitions = int(parent['episodes']), int(parent['total_transitions'])
        if target <= episodes:
            raise ValueError('Continuation target must exceed the checkpoint episode count.')
    if not preflight and artifact is None:
        artifact = ROOT / 'runs/sac' / datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%S.%fZ')
    if artifact is not None:
        artifact = artifact.resolve()
        if (artifact / 'checkpoints/latest.pt').exists():
            raise ValueError('Use a new run directory; existing checkpoints are preserved.')
        artifact.mkdir(parents=True, exist_ok=True)
        (artifact / 'checkpoints').mkdir(exist_ok=True)
        try:
            metadata = load_json(artifact / 'run.json')
        except (OSError, ValueError):
            metadata = {}
        metadata.update(schema='point_force_training_run_v1', algorithm='SAC',
                        display_name=metadata.get('display_name', artifact.name),
                        training_execution_mode=EXECUTION_MODE,
                        resumed_from=str(resume_checkpoint) if resume_checkpoint else None,
                        initial_episodes=episodes, parent_training_episodes=episodes, target_episodes=target,
                        replay_on_continuation='refill; model and optimizer states are restored')
        _atomic_json(artifact / 'run.json', metadata)
        for name, data in zip(('model.json', 'task.json', 'ppo.json', 'sac.json'),
                              (model, task, shared, algorithm)):
            _atomic_json(artifact / name, data)
        _atomic_json(artifact / 'config_snapshot.json', dict(model=model, task=task,
                     shared_experiment=shared, algorithm=algorithm,
                     effective_run=dict(device=str(device), collection_batch=batch, episodes_target=target)))
        files = ['run_sac.py', 'learning/simple_sac.py', 'learning/sac_deployment.py',
                 'learning/training_control.py',
                 'learning/deployment_rollout.py', 'learning/point_force_env.py',
                 'simulator/live_flight.py', 'simulator/cable/dder.py', 'simulator/point_mass.py',
                 'simulator/cuda_graph_physics.py']
        _atomic_json(artifact / 'source_hashes.json',
                     {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files})
        if not preflight:
            pointer = ROOT / 'runs/sac/ACTIVE_RUN.txt'
            pointer.parent.mkdir(parents=True, exist_ok=True)
            pointer.write_text(str(artifact) + '\n', encoding='utf-8')
    started = time.perf_counter()
    initial_episodes = episodes
    status = dict(schema='point_force_training_status_v1', algorithm='SAC', status='STARTING',
                  pid=os.getpid(), episodes=episodes, episodes_target=target, target_episodes=target, collection_batch=batch,
                  device=str(device), training_execution_mode=EXECUTION_MODE)
    validation = algorithm['validation']
    best_rank = (-1., -1., -1., -float('inf'))
    best_validation = None
    history_fields = ['episodes', 'success_rate', 'mean_episode_reward', 'mean_minimum_tip_distance_m',
                      'actor_loss', 'critic_loss', 'temperature', 'transitions_per_second',
                      'plan_success_rate', 'hit_and_recovery_rate', 'numerical_failure_rate', 'replay_size',
                      'nonfinite_rate', 'position_limit_rate', 'speed_limit_rate']

    last_progress_time = 0.
    last_progress_stage = None
    def progress(stage, step=0, total=0):
        nonlocal last_progress_time, last_progress_stage
        now=time.perf_counter()
        if stage==last_progress_stage and now-last_progress_time<2 and step!=total:
            return
        last_progress_time,last_progress_stage=now,stage
        status.update(status='RUNNING',stage=stage,episodes=episodes,collection_batch=batch,
                      phase_step=step,phase_total=total,progress_updated_at=time.time())
        if artifact is not None:
            _atomic_json(artifact/'status.json',status)

    def publish():
        if artifact is not None:
            _atomic_json(artifact / 'status.json', status)
        print(json.dumps(status), flush=True)

    def save(name):
        if artifact is not None:
            _atomic_checkpoint(artifact / 'checkpoints' / name,
                               checkpoint(agent, episodes, total_transitions=total_transitions, generator=generator))

    def validate():
        nonlocal best_rank, best_validation
        progress('Validating policy')
        result = evaluate_deployment(model, task, shared, agent,
                                      episodes=int(validation['episodes']),
                                      batch_size=int(validation['episodes']), device=device)
        result['training_episodes'] = episodes
        if artifact is not None:
            _atomic_json(artifact / 'validation_latest.json', result)
        rank = validation_rank(result)
        if rank > best_rank:
            best_rank, best_validation = rank, result
            save('best_validation.pt')
            if artifact is not None:
                _atomic_json(artifact / 'best_validation.json', result)
        status.update(validation_success_rate=result['success_rate'],
                      validation_hit_and_recovery_rate=result['hit_and_recovery_rate'])

    publish()
    save('latest.pt')
    try:
        if not preflight:
            validate()
        while episodes < target:
            if artifact is not None and (artifact / 'STOP_REQUESTED').exists():
                break
            if target-episodes < batch:
                batch=target-episodes
                environment=PointForceWhipEnvironment(model,task,shared,batch_size=batch,device=device)
            progress('Preparing training batch')
            score, count = collect_sac(environment, agent, replay,
                                       warmup=resume_checkpoint is None and total_transitions < int(source['warmup_transitions']),
                                       generator=generator,progress=progress)
            episodes += batch
            total_transitions += count
            metrics = None
            minibatch = min(128, replay.size) if preflight else int(source['minibatch_transitions'])
            if replay.size >= max(2, minibatch):
                updates=1 if preflight else int(source['updates_per_collection'])
                if not preflight and source.get('scale_updates_with_actual_batch',False):
                    updates=max(1,round(updates*batch/int(algorithm['training']['collection_batch'])))
                warmup_updates=int(source.get('critic_warmup_updates',0))
                progress('Learning SAC critics' if agent.gradient_updates < warmup_updates else 'Updating SAC',0,updates)
                for update_index in range(updates):
                    sample=replay.sample(minibatch, device, generator)
                    critics_only=agent.gradient_updates < warmup_updates
                    if critics_only:
                        metrics = agent.update(sample,update_actor=False)
                    else:
                        metrics = agent.update(sample)
                    progress('Learning SAC critics' if critics_only else 'Updating SAC',update_index+1,updates)
            elapsed = time.perf_counter() - started
            status.update(episodes=episodes, stage='Update complete',phase_step=0,phase_total=0, gradient_updates=agent.gradient_updates,
                          success_rate=float(score.episode_success.float().mean()),
                          mean_episode_reward=float(score.episode_reward.mean()),
                          mean_minimum_tip_distance_m=float(score.episode_minimum_tip_distance.mean()),
                          plan_success_rate=float(score.deployment['predicted_valid_hit'].float().mean()),
                          execution_rate=float(score.deployment['planned'].float().mean()),
                          hit_and_recovery_rate=float(score.deployment['joint_success'].float().mean()),
                          numerical_failure_rate=float(score.failed.float().mean()),
                          nonfinite_rate=float(score.episode_nonfinite.float().mean()),
                          position_limit_rate=float(score.episode_position_limit.float().mean()),
                          speed_limit_rate=float(score.episode_speed_limit.float().mean()),
                          replay_size=replay.size, total_transitions=total_transitions,
                          transitions_per_second=(total_transitions - (int(parent['total_transitions']) if resume_checkpoint else 0)) / max(elapsed, 1e-6),
                          episodes_per_second=(episodes-initial_episodes) / max(elapsed, 1e-6), elapsed_s=elapsed)
            if metrics:
                status.update(actor_loss=metrics.actor_loss, critic_loss=metrics.critic_loss, temperature=metrics.temperature)
            save('latest.pt')
            if algorithm.get('logging',{}).get('per_attempt',False) and artifact is not None:
                from learning.attempt_records import save_attempts
                save_attempts(artifact,score,episodes,elapsed)
            if artifact is not None:
                log = artifact / 'training_log.csv'
                new_file = not log.exists()
                with log.open('a', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=history_fields)
                    if new_file:
                        writer.writeheader()
                    writer.writerow({name: status.get(name, '') for name in history_fields})
            if not preflight and (episodes // int(validation['every_episodes']) > (episodes-batch) // int(validation['every_episodes'])):
                validate()
            publish()
        if not preflight and episodes >= target and (best_validation is None or best_validation['training_episodes'] != episodes):
            validate()
        save('terminal.pt')
        status['status'] = 'PREFLIGHT_PASS' if preflight else ('STOPPED' if episodes < target else 'COMPLETED')
        status['stage'] = 'finished'
        publish()
        return status
    except TrainingStopped:
        raise
    except BaseException as error:
        status.update(status='FAILED', error=f'{type(error).__name__}: {error}')
        publish()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--train', action='store_true')
    parser.add_argument('--artifact', type=Path)
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--device')
    parser.add_argument('--batch', type=int)
    parser.add_argument('--resume-checkpoint', type=Path)
    parser.add_argument('--config-directory', type=Path)
    args = parser.parse_args()
    run(args.artifact, args.episodes, args.preflight, args.device, args.batch, args.resume_checkpoint,
        args.config_directory)


if __name__ == '__main__':
    main()
