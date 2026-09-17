from copy import deepcopy

import pytest
import torch

from learning.deployment_rollout import DeploymentBatch, execute_batch, plan_batch, sample_batch
from learning.point_force_env import PointForceWhipEnvironment
from simulator.cable import DderState
from simulator.cable.dder import DderModel
from simulator.live_flight import HoverPID
from tests.force_config import load_configs


@pytest.fixture
def env():
    torch.set_num_threads(1)
    model, task, config = load_configs()
    task["episode_duration_s"] = .2
    return PointForceWhipEnvironment(model, task, config, batch_size=2, device=torch.device("cpu"))


def canonical_batch(env):
    settings = {**env.ppo_config["deployment"], "nominal_fraction": 1.}
    return sample_batch(env, settings, torch.Generator().manual_seed(8))


def test_initial_randomization_preserves_lengths_and_is_repeatable_without_touching_exploration(env):
    rng_before = torch.random.get_rng_state().clone()
    first = sample_batch(env, env.ppo_config["deployment"], torch.Generator().manual_seed(5))
    second = sample_batch(env, env.ppo_config["deployment"], torch.Generator().manual_seed(5))
    torch.testing.assert_close(first.truth.positions_m, second.truth.positions_m)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    for state in (first.truth, first.estimate):
        lengths = (state.positions_m[:, 1:] - state.positions_m[:, :-1]).norm(dim=-1)
        expected = lengths.new_tensor(env.model.cable_configuration.rest_lengths_m).expand_as(lengths)
        torch.testing.assert_close(lengths, expected)


def test_planner_never_observes_true_plant_and_truncates_inside_action_hold(env):
    batch = canonical_batch(env)
    q = batch.estimate.positions_m.clone()
    q[:, -1] = env.target + q.new_tensor([-.1, 0., 0.])
    batch.estimate = DderState(q, torch.zeros_like(q))
    observations = []
    class Agent:
        def deterministic_action(self, observation):
            observations.append(observation.clone())
            return torch.zeros((2, 3))
    def transition(previous, force, dt):
        q = previous.positions_m.clone()
        q[:, -1, 0] += .1
        v = torch.zeros_like(q)
        v[:, -1, 0] = 5.
        return env.model._result(previous, DderState(q, v), force, dt)
    env.model.step_runtime = transition
    first_forces, first_cutoffs = plan_batch(env, Agent(), batch)
    batch.truth = DderState(batch.truth.positions_m + 9, batch.truth.velocities_m_s + 50)
    second_forces, second_cutoffs = plan_batch(env, Agent(), batch)
    torch.testing.assert_close(first_forces, second_forces)
    torch.testing.assert_close(first_cutoffs, second_cutoffs)
    torch.testing.assert_close(observations[0], observations[1])
    cutoff = 1 + round(env.ppo_config['deployment']['strike_followthrough_s'] / env.physics_dt_s)
    assert len(first_forces) == cutoff and first_cutoffs.tolist() == [cutoff, cutoff]


def test_execution_keeps_frozen_commands_after_early_hit_and_after_miss(env, monkeypatch):
    batch = canonical_batch(env)
    q = batch.truth.positions_m.clone()
    q[0, -1] = env.target[0] + q.new_tensor([-.1, 0., 0.])
    batch.truth = DderState(q, torch.zeros_like(q))
    def dynamics(_self, state, *args, **kwargs):
        q = state.positions_m.clone()
        q[:, 0, 0] += .001
        q[0, -1, 0] += .2
        v = torch.zeros_like(q)
        v[0, -1, 0] = 5.
        return DderState(q, v)
    monkeypatch.setattr(DderModel, "step_runtime", dynamics)
    forces = env.hover_force_world_n.repeat(3, 2, 1)
    forces[:, :, 0] = torch.tensor([.2, -.3, .4])[:, None]
    trace = []
    result = execute_batch(env, batch, forces, torch.tensor([3, 2]),
                           {**env.ppo_config["deployment"], "recovery_duration_s": .03},
                           trace=lambda *row: trace.append(row))
    assert trace[0][4].tolist() == [True, False]
    for index in range(3):
        torch.testing.assert_close(trace[index][1][0], forces[index, 0])
    for index in range(2):
        torch.testing.assert_close(trace[index][1][1], forces[index, 1])
    assert trace[2][3].tolist() == [True, False]
    assert trace[3][3].tolist() == [False, False]
    assert result.episode_success.tolist() == [True, False]
    assert result.episode_hit_time_s[0] == .01
    # Scoring freezes at first success; actual motion continues through recovery.
    assert result.execution_state.positions_m[0, -1, 0] > result.state.positions_m[0, -1, 0]


def test_refused_plan_never_executes_and_is_not_a_success(env, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A refused plan must stay in hover, not execute a failed trajectory")
    monkeypatch.setattr(DderModel, "step_runtime", forbidden)
    batch = canonical_batch(env)
    result = execute_batch(env, batch, torch.ones((20, 2, 3)), torch.zeros(2, dtype=torch.long),
                           env.ppo_config["deployment"])
    assert not result.episode_success.any()
    assert not result.deployment["planned"].any()
    torch.testing.assert_close(result.execution_state.positions_m, batch.truth.positions_m)


def test_batched_recovery_pid_matches_independent_live_controllers(env):
    target = env.initial_root_position
    batch_pid = HoverPID(env.model, target, 3.2)
    individual = [HoverPID(env.model, target[i:i+1], 3.2) for i in range(2)]
    q = env.state.positions_m.clone()
    q[0, :, 0] += .1
    q[1, :, 0] += 10  # Only this row saturates; it must not block the other integral.
    state = DderState(q, torch.zeros_like(q))
    for _ in range(4):
        commands = batch_pid.command(state, .01)
        expected = torch.cat([pid.command(DderState(q[i:i+1], state.velocities_m_s[i:i+1]), .01)
                              for i, pid in enumerate(individual)])
        torch.testing.assert_close(commands, expected)
    assert batch_pid.integral[0, 0] != 0 and batch_pid.integral[1, 0] == 0
