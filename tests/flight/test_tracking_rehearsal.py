import json
from pathlib import Path

import numpy as np
import pytest
import torch

from deployment.tracking_rehearsal import PositionTracking, TrackingRehearsalFlight
from simulator.strike_plan import StrikePlan

ROOT = Path(__file__).resolve().parents[2]


def test_position_packets_estimate_causal_velocity_and_hold_duplicate_timestamps():
    tracker = PositionTracking()
    first = tracker.sample(0.,[0,0,1.5],[1,0,1.4])
    np.testing.assert_array_equal(first.velocity_m_s,[0,0,0])
    next_packet = tracker.sample(.01,[.001,0,1.5],[1.01,0,1.4])
    np.testing.assert_allclose(next_packet.velocity_m_s,[.1,0,0])
    assert tracker.sample(.01,[9,9,9],[9,9,9]) is next_packet
    np.testing.assert_array_equal(next_packet.target_m,[1.01,0,1.4])
    with pytest.raises(ValueError):
        tracker.sample(.005,[0,0,0],[0,0,0])


def test_hundred_hz_tracking_and_twenty_hz_controller_with_partial_cutoff():
    configs = [json.loads((ROOT/'config'/f'{name}.json').read_text()) for name in ('model','task','ppo')]
    flight = TrackingRehearsalFlight(*configs,policy=lambda _:pytest.fail('Actor queried during execution'),
        physics=lambda q,v,f:(q.clone(),v.clone()))
    calls = []
    original = flight.pid.command
    def command(state,dt):
        calls.append((flight.time_s,dt))
        return original(state,dt)
    flight.pid.command = command
    for _ in range(15):
        flight.step()
    assert calls == [(0.,.05),(.05,.05),(.1,.05)]
    # Even absurd true velocities cannot leak into measured drone initialization.
    flight.state.velocities_m_s.fill_(50.)
    assumed = flight.launch_state()
    assert not assumed.velocities_m_s.any()
    flight.settled_s = 10.
    forces = flight.last_command.repeat(8,1)
    forces[:5,0] = .1
    forces[5:,0] = -.1
    flight.start_strike(StrikePlan(forces,.01,assumed,torch.zeros(1,79)))
    for force in forces:
        frame = flight.step()
        np.testing.assert_array_equal(frame['command'],force.numpy())
    assert len(calls) == 3
    assert flight.phase == flight.RECOVER
    flight.step()
    assert calls[-1] == (.23,.05)
    arrays,summary = flight.recording()
    np.testing.assert_allclose(np.diff(arrays['tracking_time_s']),.01,atol=1e-14)
    assert summary['controller_hz'] == 20 and summary['tracking_hz'] == 100
