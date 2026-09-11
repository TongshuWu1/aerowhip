from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from simulator.cable import DderState
from simulator.live_flight import LiveFlight, prepare_live_physics
from simulator.replay import save_replay, load_replay
from simulator.strike_plan import StrikePlan, compile_strike_plan

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def configs():
    torch.set_num_threads(1)
    return tuple(json.loads((ROOT / "config" / name).read_text(encoding="utf-8"))
                 for name in ("model.json", "task.json", "ppo.json"))


@pytest.fixture(scope="module")
def physics(configs):
    return prepare_live_physics(json.dumps(configs[0], sort_keys=True))


def test_compiled_live_dynamics_match_reference_during_a_bent_3d_trajectory(configs, physics):
    flight = LiveFlight(*configs)
    reference = compiled = flight.state
    for index in range(60):
        force = torch.tensor([[1.2 if index < 30 else -1., .4, 2.1]], dtype=torch.float64)
        reference = flight.model.step_runtime(reference, force, flight.dt_s).state
        q, v = physics(compiled.positions_m, compiled.velocities_m_s, force)
        compiled = DderState(q, v)
        torch.testing.assert_close(compiled.positions_m, reference.positions_m, rtol=1e-8, atol=1e-9)
        torch.testing.assert_close(compiled.velocities_m_s, reference.velocities_m_s, rtol=1e-7, atol=1e-8)
    assert float(compiled.positions_m[0, 0, 1]) > .01


def _plan(flight, steps=3):
    state = DderState(flight.state.positions_m.clone(), flight.state.velocities_m_s.clone())
    forces = flight.last_command.repeat(steps, 1)
    forces[:, 0] = .3
    return StrikePlan(forces, flight.dt_s, state, torch.zeros((1, 79)))


def test_open_loop_handoff_preserves_state_and_never_queries_policy(configs, physics):
    def forbidden(_observation):
        pytest.fail("Execution must not query the policy or motion state")
    flight = LiveFlight(*configs, policy=forbidden, physics=physics)
    with pytest.raises(ValueError, match="settle"):
        flight.start_strike(_plan(flight))
    for _ in range(60):
        flight.step()
    assert flight.ready
    plan = _plan(flight)
    before = flight.state
    flight.start_strike(plan)
    assert flight.state is before
    for command in plan.forces_world_n:
        frame = flight.step()
        torch.testing.assert_close(torch.from_numpy(frame["command"]), command)
    assert flight.phase == flight.RECOVER
    elapsed = flight.strike_steps
    for _ in range(20):
        flight.step()
    assert flight.strike_steps == elapsed
    assert flight.force_sequence is None


@pytest.mark.parametrize('tip_speed',[1.,5.])
def test_compile_uses_initial_state_and_cuts_the_sequence_at_predicted_first_hit(configs,tip_speed):
    from copy import deepcopy
    model, task, ppo = deepcopy(configs)
    flight = LiveFlight(model, task, ppo)
    q = flight.state.positions_m + .01
    v = torch.full_like(q, .02)
    initial = DderState(q, v)
    task["target_position_m"] = (q[0, -1] + q.new_tensor([.1, 0., 0.])).tolist()
    next_q, next_v = q.clone(), v.clone()
    next_q[0, -1, 0] += .2
    next_v[0, -1] = q.new_tensor([tip_speed, 0., 0.])
    observations = []
    def policy(observation):
        observations.append(observation.clone())
        return torch.zeros((1, 3))
    plan = compile_strike_plan(model, task, ppo, initial, policy,
                               physics=lambda *_: (next_q, next_v))
    # One contact step plus the configured frozen follow-through; no new actor query.
    expected_steps = 1 + round(ppo['deployment']['strike_followthrough_s']/model['simulation']['dt_s'])
    assert len(plan.forces_world_n) == expected_steps
    assert plan.duration_s == expected_steps * model['simulation']['dt_s']
    assert len(observations) == 1
    torch.testing.assert_close(plan.initial_observation, observations[0])
    torch.testing.assert_close(observations[0][0, 36:72], (v / 5.).float().flatten())
    torch.testing.assert_close(plan.initial_state.positions_m, q)
    torch.testing.assert_close(plan.initial_state.velocities_m_s, v)


def test_no_predicted_hit_executes_once_then_returns_to_pid(configs, physics):
    from copy import deepcopy
    model, task, ppo = deepcopy(configs)
    task["episode_duration_s"] = .1
    flight = LiveFlight(model, task, ppo, policy=lambda obs: torch.zeros((1, 3)), physics=physics)
    before = flight.state
    plan=flight.plan_strike(before)
    assert len(plan.forces_world_n)==10
    assert flight.state is before and flight.phase == flight.HOVER
    flight.settled_s=1.
    flight.start_strike(plan)
    for _ in range(11):
        flight.step()
    assert flight.phase==flight.RECOVER
    assert flight.attempt==1
    assert not flight.success


@pytest.mark.parametrize("velocity,other_first,expected_hit", [
    ([5., 0., 0.], False, True), ([3., 0., 0.], False, False),
    ([5., 0., 6.], False, False), ([5., 0., 0.], True, False),
])
def test_simulated_hit_is_diagnostic_and_does_not_control_sequence_length(configs, velocity, other_first, expected_hit):
    flight = LiveFlight(*configs, policy=lambda obs: torch.zeros((1, 3)))
    flight.settled_s = .5
    flight.start_strike(_plan(flight))
    q = flight.state.positions_m.clone()
    target = flight.environment.target[0]
    q[0, -1] = target + q.new_tensor([-.1, 0., 0.])
    if other_first:
        q[0, -2] = q[0, -1]
    flight.state = DderState(q, torch.zeros_like(q))
    next_q, next_v = q.clone(), torch.zeros_like(q)
    next_q[0, -1] = target + q.new_tensor([.1, 0., 0.])
    next_v[0, -1] = q.new_tensor(velocity)
    if other_first:
        next_q[0, -2] = next_q[0, -1]
    flight.physics = lambda *_: (next_q, next_v)
    frame = flight.step()
    assert frame["hit"] == expected_hit
    assert flight.phase == flight.POLICY  # Neither a hit nor a miss changes the fixed sequence.
    torch.testing.assert_close(flight.state.positions_m, next_q)
    torch.testing.assert_close(flight.state.velocities_m_s, next_v)
    flight.step()
    flight.step()
    assert flight.phase == flight.RECOVER
    assert flight.strike_steps == 3


def test_sequence_end_returns_to_pid_even_without_hit_feedback_and_recording_keeps_history(configs, physics, tmp_path):
    flight = LiveFlight(*configs, policy=lambda obs: torch.zeros((1, 3)), physics=physics)
    flight.settled_s = .5
    flight.start_strike(_plan(flight))
    for _ in range(3):
        flight.step()
    assert flight.phase == flight.RECOVER
    assert not flight.success
    assert "Single strike finished" in flight.message
    flight.step()
    arrays, summary = flight.recording()
    directory = save_replay(tmp_path, arrays, summary)
    restored, metadata = load_replay(directory)
    np.testing.assert_array_equal(restored["controller_phase"], arrays["controller_phase"])
    np.testing.assert_array_equal(restored["valid_hit"], arrays["valid_hit"])
    assert metadata["events"][-1]["event"] == "Sequence complete"


def test_bad_physics_stops_without_installing_a_corrupt_state(configs):
    flight = LiveFlight(*configs)
    previous = flight.state
    flight.physics = lambda q, v, f: (torch.full_like(q, torch.nan), v)
    with pytest.raises(RuntimeError, match="numerical limits"):
        flight.step()
    assert flight.state is previous
