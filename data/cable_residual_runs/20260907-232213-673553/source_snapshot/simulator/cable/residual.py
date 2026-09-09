"""Bounded learned acceleration correction, evaluated from predicted state only.

This is a model discrepancy term, not an identified material parameter. It can
represent external effects and does not claim conservation of momentum.
"""
from pathlib import Path
import hashlib
import math
import torch
from torch import nn


def cable_drag_description(model):
    residual = model.get('motion_residual', {})
    if residual.get('enabled') and residual.get('specification', {}).get('learn_drag'):
        return 'cable drag learned in residual (s⁻¹); additional bounded NN correction'
    value = model.get('cable', {}).get('external_drag_s_inv', 0)
    return f'fixed cable damping {value:g} s⁻¹ in this model; drag learning is not active'


class MotionResidual(nn.Module):
    def __init__(self, node_count, *, hidden=48, acceleration_limit=2.0,
                 learn_drag=False, initial_drag_s_inv=0.0):
        super().__init__()
        self.node_count = int(node_count)
        self.hidden = int(hidden)
        self.acceleration_limit = float(acceleration_limit)
        self.learn_drag = bool(learn_drag)
        self.initial_drag_s_inv = float(initial_drag_s_inv)
        if not math.isfinite(self.initial_drag_s_inv) or self.initial_drag_s_inv < 0:
            raise ValueError('Initial residual drag must be finite and nonnegative.')
        if self.learn_drag:
            # Positive damping is dissipative. The saved calibration is only an
            # initialization, not a fixed coefficient. Float64 avoids rounding it.
            value = max(self.initial_drag_s_inv, 1e-8)
            self.raw_drag = nn.Parameter(torch.tensor(value + math.log(-math.expm1(-value)), dtype=torch.float64))
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

    def drag_coefficient(self):
        return torch.nn.functional.softplus(self.raw_drag) if self.learn_drag else None

    def drag_rates(self, reference):
        """Learned cable damping, integrated by the solver's exponential step.

        This is the structured part of the residual; forward() supplies only the
        additional NN acceleration. Never apply this damping to the drone node.
        """
        weights = reference.new_ones(self.node_count)
        weights[0] = 0
        return weights * self.drag_coefficient() if self.learn_drag else weights * 0

    def specification(self):
        result = dict(node_count=self.node_count, hidden=self.hidden,
                      acceleration_limit=self.acceleration_limit)
        if self.learn_drag:
            result.update(learn_drag=True, initial_drag_s_inv=self.initial_drag_s_inv)
        return result


class FrozenMotionResidual:
    """Load immutable weights once per runtime dtype/device."""
    def __init__(self, path, sha256):
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
            raise ValueError('Residual checkpoint hash mismatch.')
        self.payload = torch.load(path, map_location='cpu', weights_only=True)
        self.cache = {}

    def _network(self, positions):
        key = (positions.device, positions.dtype)
        if key not in self.cache:
            network = MotionResidual(**self.payload['specification']).to(
                device=positions.device, dtype=positions.dtype)
            network.load_state_dict(self.payload['state_dict'])
            network.eval()
            network.requires_grad_(False)
            self.cache[key] = network
        return self.cache[key]

    def __call__(self, positions, velocities):
        return self._network(positions)(positions, velocities)

    def drag_rates(self, reference):
        return self._network(reference).drag_rates(reference)
