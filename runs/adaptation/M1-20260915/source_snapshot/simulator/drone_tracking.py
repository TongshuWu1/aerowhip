"""Effective full-state tracking surrogate; no firmware or motor identification.

The residual models the fixed drone/cable setup's observed tracking response.
Do not add a second cable reaction to this effective model without re-identifying
the nominal response. This module sends no commands and does not alter PPO.
"""
import torch
from torch import nn


class DroneTrackingResidual(nn.Module):
    def __init__(self, hidden=32, acceleration_limit=2.0):
        super().__init__()
        self.hidden, self.acceleration_limit = hidden, acceleration_limit
        self.net = nn.Sequential(nn.Linear(21, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, position, velocity, command, past_command):
        features = torch.cat((command[..., :3]-position, (command[..., 3:6]-velocity)/2,
            command[..., 6:9]/10, velocity/2, past_command[..., :3]-position,
            past_command[..., 3:6]/2, past_command[..., 6:9]/10), dim=-1)
        return self.acceleration_limit*torch.tanh(self.net(features))

    def specification(self):
        return dict(hidden=self.hidden, acceleration_limit=self.acceleration_limit)


def predict_trajectory(initial_position, initial_velocity, commands, past_commands,
                       dt_s, gains, residual=None):
    """Batched recursive P/V/A prediction from initial state and known commands.

commands are causal-held, already delay-adjusted P/V/A for each integration
interval. No measured positions are consumed after initialization.
"""
    position, velocity = initial_position, initial_velocity
    positions, velocities, accelerations = [position], [velocity], []
    kp, kd, feedforward = gains[:3], gains[3:6], gains[6:9]
    for index in range(commands.shape[1]):
        command = commands[:, index]
        acceleration = (kp*(command[:,:3]-position) + kd*(command[:,3:6]-velocity)
                        + feedforward*command[:,6:9])
        if residual is not None:
            acceleration = acceleration + residual(position, velocity, command, past_commands[:,index])
        step = dt_s[:,index,None]
        position = position + step*velocity + .5*step.square()*acceleration
        velocity = velocity + step*acceleration
        positions.append(position); velocities.append(velocity); accelerations.append(acceleration)
    return torch.stack(positions,1), torch.stack(velocities,1), torch.stack(accelerations,1)


def load_tracking_model(checkpoint, *, device='cpu'):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload.get('schema') != 'effective_fullstate_drone_residual_v1':
        raise ValueError('Expected a full-state drone tracking residual checkpoint.')
    network = DroneTrackingResidual(**payload['specification']).to(device=device, dtype=torch.float64)
    network.load_state_dict(payload['state_dict'])
    network.eval().requires_grad_(False)
    gains = torch.tensor(payload['nominal']['gains'], dtype=torch.float64, device=device)
    return network, gains, payload['nominal']['delay_s'], payload['history_s']
