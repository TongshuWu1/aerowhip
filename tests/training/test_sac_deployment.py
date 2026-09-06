from copy import deepcopy

import torch

from learning import ReplayBuffer
from learning.sac_deployment import add_plan_to_replay, critic_context_dimension
from learning.simple_ppo import PPORollout
from learning.simple_sac import SACActor, SimpleSACAgent
from run_sac import build_agent, checkpoint, configs, restore
from simulator.live_flight import LiveFlight
from simulator.rollout import simulate_policy


def test_sac_exact_budget_progress_and_validation_replay(tmp_path, monkeypatch):
    import csv
    import shutil
    import run_sac
    from simulator.workflow import atomic_json,read_json
    model,task,shared,algorithm=deepcopy(configs())
    algorithm['bootstrap']=None
    algorithm['sac']['critic_warmup_updates']=0
    task['episode_duration_s']=.2
    shared['deployment']['recovery_duration_s']=.5
    algorithm['sac'].update(hidden_dim=32,replay_capacity=64,minibatch_transitions=2,updates_per_collection=2)
    algorithm['validation'].update(episodes=2,every_episodes=3)
    for name,data in [('model',model),('task',task),('ppo',shared),('sac',algorithm)]:
        atomic_json(tmp_path/'config'/f'{name}.json',data)
    for name in ['run_sac.py','learning/simple_sac.py','learning/sac_deployment.py',
                 'learning/training_control.py',
                 'learning/deployment_rollout.py','learning/point_force_env.py',
                 'simulator/live_flight.py','simulator/cable/dder.py','simulator/point_mass.py',
                 'simulator/cuda_graph_physics.py']:
        destination=tmp_path/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(run_sac.ROOT/name,destination)
    monkeypatch.setattr(run_sac,'ROOT',tmp_path)
    updates=[]
    original=run_sac._atomic_json
    def capture(path,data):
        if path.name=='status.json':updates.append(dict(data))
        original(path,data)
    monkeypatch.setattr(run_sac,'_atomic_json',capture)
    output=tmp_path/'run'
    status=run_sac.run(output,5,device_name='cpu',batch_override=3,config_directory=tmp_path/'config')
    assert status['status']=='COMPLETED' and status['episodes']==5
    assert all(key in status for key in ['nonfinite_rate','position_limit_rate','speed_limit_rate'])
    assert any(row.get('phase_step',0)>0 and row['episodes']==0 for row in updates)
    assert any(row.get('stage')=='Updating SAC' for row in updates)
    with (output/'training_log.csv').open() as stream:
        assert [int(row['episodes']) for row in csv.DictReader(stream)]==[3,5]
    latest=read_json(output/'validation/latest.json')
    assert latest['training_episodes']==5 and (output/latest['replay']).exists()
    assert abs(sum(latest['reward_components'].values())-latest['mean_episode_reward'])<1e-8
    assert latest['median_planning_minimum_tip_distance_m']>=0
    assert read_json(output/'sac.json')['sac']['stochastic_action_indices']==[0,1,2]


def test_sac_replay_uses_terminal_plan_reward_and_only_past_commands():
    rollout = PPORollout.allocate(3, 2, 79, 3, device=torch.device('cpu'))
    for name in ('observations', 'actions', 'rewards', 'dones', 'masks', 'log_probabilities', 'values'):
        getattr(rollout, name).zero_()
    rollout.observations[0, 0] = .1
    rollout.observations[0, 1] = .2
    rollout.observations[1, 0] = .3
    rollout.actions[0] = .4
    rollout.actions[1] = .5
    rollout.actions[2] = .9  # Padding/future commands must never enter earlier context.
    rollout.masks[0] = 1
    rollout.masks[1, 0] = 1
    rollout.dones[0, 1] = 1
    rollout.dones[1, 0] = 1
    rollout.rewards[0, 1] = 50
    rollout.rewards[1, 0] = 120
    replay = ReplayBuffer(8, 79, 3, critic_context_dim=88)
    assert add_plan_to_replay(rollout, replay) == 3
    observation, action, reward, next_observation, done, context, next_context = [v[:3] for v in replay.tensors]
    assert reward[:, 0].tolist() == [0, 50, 120]
    assert done[:, 0].tolist() == [0, 1, 1]
    torch.testing.assert_close(next_observation[0], observation[2])
    assert not context[:2, 79:].any()
    torch.testing.assert_close(context[2, :79], observation[0])
    torch.testing.assert_close(context[2, 79:82], action[0])
    assert not context[2, 82:].any()
    torch.testing.assert_close(next_context[2, 82:85], action[2])
    assert not next_context[:, 85:].any()


def test_sac_critics_use_history_but_actor_still_only_receives_79_inputs():
    torch.set_num_threads(1)
    agent = SimpleSACAgent(79, device=torch.device('cpu'), hidden_dim=32, critic_context_dim=88, gamma=1.)
    seen = []
    handle = agent.actor.network.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].shape[-1]))
    observation = torch.randn(8, 79)
    batch = (observation, agent.act(observation), torch.randn(8, 1), torch.randn(8, 79),
             torch.zeros(8, 1), torch.randn(8, 88), torch.randn(8, 88))
    result = agent.update(batch)
    handle.remove()
    assert seen and set(seen) == {79}
    assert all(torch.isfinite(torch.tensor(value)) for value in vars(result).values())
    assert all(parameter.grad is None for parameter in agent.target1.parameters())
    assert all(parameter.requires_grad for parameter in agent.critic1.parameters())
    assert torch.count_nonzero(agent.act(observation)[:, 1]) == 0


def test_saturated_sac_actions_keep_finite_log_probabilities():
    actor = SACActor(79, 3, 16)
    with torch.no_grad():
        actor.network[-1].weight.zero_()
        actor.network[-1].bias[:2] = torch.tensor([30., -30.])
    actions, logp = actor.sample(torch.zeros(4, 79))
    assert torch.isfinite(logp).all()
    assert torch.count_nonzero(actions[:, 1]) == 0


def test_sac_checkpoint_restores_weights_optimizers_temperature_and_generator(tmp_path):
    _, task, _, algorithm = configs()
    task = {**task, 'episode_duration_s': .2}
    algorithm = deepcopy(algorithm)
    algorithm['sac']['hidden_dim'] = 32
    agent = build_agent(algorithm, torch.device('cpu'), task)
    context_dim = critic_context_dimension(task)
    observation = torch.randn(8, 79)
    agent.update((observation, agent.act(observation), torch.ones(8, 1), observation,
                  torch.ones(8, 1), torch.zeros(8, context_dim), torch.zeros(8, context_dim)))
    generator = torch.Generator().manual_seed(81)
    path = tmp_path / 'portable.pt'
    torch.save(checkpoint(agent, 8, total_transitions=16, generator=generator), path)
    fresh = build_agent(algorithm, torch.device('cpu'), task)
    restored_generator = torch.Generator().manual_seed(9)
    payload = restore(fresh, path, restored_generator)
    torch.testing.assert_close(agent.deterministic_action(observation), fresh.deterministic_action(observation))
    torch.testing.assert_close(agent.log_temperature, fresh.log_temperature)
    assert fresh.actor_optimizer.state and fresh.critic_optimizer.state
    assert fresh.gradient_updates == 1 and payload['episodes'] == 8
    assert torch.equal(generator.get_state(), restored_generator.get_state())
    assert payload['replay_buffer_saved'] is False


def test_portable_sac_checkpoint_works_in_live_one_shot_and_offline_replay(tmp_path):
    model, task, shared, algorithm = configs()
    shared=deepcopy(shared)
    shared['bootstrap']=None
    shared['deployment']['strike_followthrough_s']=0.
    task = deepcopy(task)
    task['episode_duration_s'] = .2
    task['success'].update(minimum_directed_tip_speed_m_s=0.,
                           maximum_tip_velocity_to_desired_direction_error_deg=180.)
    # This test's easy contact makes the inference/handoff check independent of learning.
    initial = LiveFlight(model, task, shared).state
    task['target_position_m'] = initial.positions_m[0, -1].tolist()
    algorithm = deepcopy(algorithm)
    algorithm['sac']['hidden_dim'] = 32
    agent = build_agent(algorithm, torch.device('cpu'), task)
    with torch.no_grad():
        agent.actor.network[-1].weight.zero_()
        agent.actor.network[-1].bias.zero_()
        agent.actor.network[-1].bias[0] = .1
    path = tmp_path / 'relocated-sac.pt'
    torch.save(checkpoint(agent, 8), path)
    flight = LiveFlight.from_checkpoint(model, task, shared, path)
    plan = flight.plan_strike(flight.state)
    assert len(plan.forces_world_n) == 1
    def forbidden(_):
        raise AssertionError('Execution must never query SAC.')
    flight.policy = forbidden
    flight.settled_s = .5
    flight.start_strike(plan)
    flight.step()
    assert flight.phase == flight.RECOVER and flight.algorithm == 'SAC'
    assert flight.recording()[1]['algorithm'] == 'SAC'
    arrays, summary = simulate_policy(model, task, shared, checkpoint_path=path)
    assert summary['algorithm'] == 'SAC' and summary['finite']
    assert len(arrays['commanded_force_world_n']) == 1
