"""Ideal 100 Hz position packets and a 20 Hz outer force controller.

This simulates the data interface, not an OptiTrack network connection. No
measurement noise/latency is invented; timestamped positions yield causal velocity.
"""
from dataclasses import dataclass
import numpy as np
import torch

from deployment.rehearsal import RehearsalFlight, assumed_hanging_state
from simulator.cable import DderState


@dataclass(frozen=True)
class TrackingPacket:
    time_s: float
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    target_m: np.ndarray


class PositionTracking:
    def __init__(self):
        self.packet = None

    def sample(self, time_s, position, target):
        if self.packet is not None and time_s == self.packet.time_s:
            return self.packet
        if self.packet is not None and time_s < self.packet.time_s:
            raise ValueError('Tracking timestamps must increase.')
        position, target = np.array(position, dtype=float, copy=True), np.array(target, dtype=float, copy=True)
        if (position.shape != (3,) or target.shape != (3,)
                or not np.isfinite(position).all() or not np.isfinite(target).all() or not np.isfinite(time_s)):
            raise ValueError('Tracking packet requires finite time and XYZ positions.')
        velocity = (np.zeros(3) if self.packet is None else
                    (position - self.packet.position_m) / (time_s - self.packet.time_s))
        for array in (position, target, velocity):
            array.flags.writeable = False
        self.packet = TrackingPacket(float(time_s), position, velocity, target)
        return self.packet


class TrackingRehearsalFlight(RehearsalFlight):
    def __init__(self, *args, **kwargs):
        self.tracking = PositionTracking()
        self.held_hover_force = None
        self.next_hover_update = 0
        self.controller_updates = []
        super().__init__(*args, **kwargs)
        if abs(self.dt_s-.01)>1e-12 or abs(float(self.environment.task_config['control_dt_s'])-.05)>1e-12:
            raise ValueError('Tracking rehearsal requires 100 Hz physics and 20 Hz control.')

    def sample_tracking(self):
        return self.tracking.sample(self.time_s, self.state.positions_m[0,0].numpy(),
                                    self.environment.target[0].numpy())

    def is_settled(self):
        packet = self.tracking.packet or self.sample_tracking()
        return (np.linalg.norm(packet.position_m-self.hover_position[0].numpy()) < .02
                and np.linalg.norm(packet.velocity_m_s) < .01)

    def launch_state(self):
        packet = self.sample_tracking()
        return assumed_hanging_state(self.environment.model_config,
                                     packet.position_m.copy(), packet.velocity_m_s.copy())

    def normal_command(self):
        if self.held_hover_force is None or self.steps >= self.next_hover_update:
            packet = self.tracking.packet or self.sample_tracking()
            measured = DderState(torch.tensor(packet.position_m.copy())[None,None],
                                 torch.tensor(packet.velocity_m_s.copy())[None,None])
            self.held_hover_force = self.pid.command(measured, .05).clone()
            self.next_hover_update = self.steps + 5
            self.controller_updates.append(self.time_s)
        return self.held_hover_force

    def return_to_hover(self, reason='Manual return'):
        super().return_to_hover(reason)
        self.held_hover_force = None

    def step(self):
        packet = self.sample_tracking()
        if self.phase == self.POLICY and self.strike_steps % 5 == 0:
            self.controller_updates.append(self.time_s)
        frame = super().step()
        frame.update(tracking_time_s=packet.time_s, tracking_position_m=packet.position_m.copy(),
                     tracking_velocity_m_s=packet.velocity_m_s.copy(), tracking_target_m=packet.target_m.copy())
        return frame

    def recording(self):
        arrays, summary = super().recording()
        frames = [f for f in self.history if 'tracking_time_s' in f]
        for key in ('tracking_time_s','tracking_position_m','tracking_velocity_m_s','tracking_target_m'):
            arrays[key] = np.asarray([f[key] for f in frames])
        if frames and all('tracking_wall_time_s' in f for f in frames):
            arrays['tracking_wall_time_s'] = np.asarray([f['tracking_wall_time_s'] for f in frames])
        arrays['controller_update_time_s'] = np.asarray(self.controller_updates)
        summary.update(tracking_hz=100., controller_hz=20.,
                       tracking_model='ideal sampled positions; backward-difference velocity; no noise or transport delay')
        return arrays, summary
