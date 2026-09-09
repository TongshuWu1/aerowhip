"""Drone-only initialization and a simulated hover/plan/execute rehearsal."""
import json
from pathlib import Path

import numpy as np
import torch

from simulator.live_flight import LiveFlight
from simulator.point_mass import ForceControlledPointCable


def assumed_hanging_state(model_config, attachment_position_m, attachment_velocity_m_s):
    """Inputs use the attachment reference in the common world frame, not body COM."""
    values = []
    for value in (attachment_position_m, attachment_velocity_m_s):
        value = torch.as_tensor(value, dtype=torch.float64)
        if value.shape != (3,) or not bool(torch.isfinite(value).all()):
            raise ValueError('Expected finite attachment XYZ position and velocity.')
        values.append(value)
    return ForceControlledPointCable.from_mapping(model_config).hanging_state(*values)


class RehearsalFlight(LiveFlight):
    settling_duration_s = 10.

    @property
    def ready(self):
        return (self.phase == self.HOVER and self.settled_s + 1e-9 >= self.settling_duration_s
                and self.policy is not None)

    def is_settled(self):
        # Only the drone point is observed by preparation and launch checks.
        return (float((self.state.positions_m[:, 0] - self.hover_position).norm()) < .02
                and float(self.state.velocities_m_s[:, 0].norm()) < .01)

    def launch_state(self):
        return assumed_hanging_state(self.environment.model_config,
            self.state.positions_m[0, 0].detach().clone(),
            self.state.velocities_m_s[0, 0].detach().clone())

    def begin_approach(self):
        # No ground/contact/takeoff model exists: begin suspended 10 cm below hover.
        root = self.hover_position.clone()
        root[:, 2] -= .1
        length = sum(self.model.cable_configuration.rest_lengths_m)
        if float(root[0, 2]) <= length + .05:
            raise ValueError('Raise the initial attachment so the suspended cable clears the ground.')
        self.state = self.model.hanging_state(root)
        self.settled_s = 0.
        self.history.clear()
        self.history.append(self.snapshot())

    def set_target(self, target):
        target = np.asarray(target, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError('Expected a finite target XYZ.')
        self.environment.task_config['target_position_m'] = target.tolist()
        self.environment.target = torch.as_tensor(target, dtype=torch.float64)[None].clone()


def save_rehearsal(directory, flight, **extra):
    arrays, summary = flight.recording()
    summary.update(source='drone_only_rehearsal', simulation_only=True,
                   initialization='assumed_vertical_cable', settling_duration_s=10., **extra)
    directory = Path(directory)
    np.savez_compressed(directory / 'flight.npz', **arrays)
    (directory / 'flight.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
