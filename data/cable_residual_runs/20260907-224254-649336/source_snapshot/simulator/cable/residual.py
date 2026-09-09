"""Bounded learned acceleration correction, evaluated from predicted state only.

This is a model discrepancy term, not an identified material parameter. It can
represent external effects and does not claim conservation of momentum.
"""
from pathlib import Path
import hashlib
import torch
from torch import nn


class MotionResidual(nn.Module):
    def __init__(self, node_count, *, hidden=48, acceleration_limit=2.0):
        super().__init__()
        self.node_count = int(node_count)
        self.hidden = int(hidden)
        self.acceleration_limit = float(acceleration_limit)
        self.net = nn.Sequential(nn.Linear(6 * node_count + 3, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 3 * (node_count - 1)))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, positions, velocities):
        # Translation invariant; absolute velocity is retained for external drag.
        relative_q = positions - positions[:, :1]
        relative_v = (velocities - velocities[:, :1]) / 2.0
        features = torch.cat((relative_q.flatten(1), relative_v.flatten(1),
                              velocities[:, 0] / 2.0), dim=1)
        correction = self.acceleration_limit * torch.tanh(self.net(features))
        correction = correction.reshape(-1, self.node_count - 1, 3)
        # No learned drone force: only cable nodes receive discrepancy forces.
        return torch.cat((torch.zeros_like(positions[:, :1]), correction), dim=1)

    def specification(self):
        return dict(node_count=self.node_count, hidden=self.hidden,
                    acceleration_limit=self.acceleration_limit)


class FrozenMotionResidual:
    """Load immutable weights once per runtime dtype/device."""
    def __init__(self, path, sha256):
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
            raise ValueError('Residual checkpoint hash mismatch.')
        self.payload = torch.load(path, map_location='cpu', weights_only=True)
        self.cache = {}

    def __call__(self, positions, velocities):
        key = (positions.device, positions.dtype)
        if key not in self.cache:
            network = MotionResidual(**self.payload['specification']).to(
                device=positions.device, dtype=positions.dtype)
            network.load_state_dict(self.payload['state_dict'])
            network.eval()
            network.requires_grad_(False)
            self.cache[key] = network
        return self.cache[key](positions, velocities)
