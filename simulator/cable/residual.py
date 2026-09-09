"""Bounded NN acceleration plus optional learned nonnegative cable damping.

This is a model discrepancy term, not an identified material parameter. It can
represent external effects and does not claim conservation of momentum.
"""
from pathlib import Path
from copy import deepcopy
import hashlib
import math
import torch
from torch import nn


def cable_drag_description(model):
    residual = model.get('motion_residual', {})
    if residual.get('enabled') and residual.get('specification', {}).get('mode') == 'dissipative_plus_acceleration':
        return 'learned cable damping + bounded acceleration correction; no separate drag coefficient'
    if residual.get('enabled') and residual.get('drag_mode') == 'nn_only':
        return 'cable effects learned by NN residual; no separate drag coefficient'
    if residual.get('enabled') and residual.get('specification', {}).get('learn_drag'):
        return 'cable drag learned in residual (s⁻¹); additional bounded NN correction'
    value = model.get('cable', {}).get('external_drag_s_inv', 0)
    return f'fixed cable damping {value:g} s⁻¹ in this model; drag learning is not active'


class MotionResidual(nn.Module):
    def __init__(self, node_count, *, hidden=48, acceleration_limit=2.0,
                 learn_drag=False, initial_drag_s_inv=0.0, mode='acceleration', damping_limit_s_inv=2.0):
        super().__init__()
        self.node_count = int(node_count)
        self.hidden = int(hidden)
        self.acceleration_limit = float(acceleration_limit)
        self.learn_drag = bool(learn_drag)
        self.initial_drag_s_inv = float(initial_drag_s_inv)
        if mode not in ('acceleration','dissipative','dissipative_plus_acceleration'):raise ValueError('Unknown cable residual mode')
        self.mode=mode;self.damping_limit_s_inv=float(damping_limit_s_inv)
        if mode in ('dissipative','dissipative_plus_acceleration') and (learn_drag or not math.isfinite(self.damping_limit_s_inv) or self.damping_limit_s_inv<=0):
            raise ValueError('Dissipative NN requires a positive bound and no separate scalar drag')
        if mode=='dissipative_plus_acceleration' and (not math.isfinite(self.acceleration_limit) or self.acceleration_limit<=0):
            raise ValueError('Additional acceleration requires a finite positive bound')
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
        if self.mode=='dissipative_plus_acceleration':
            self.correction_head=nn.Linear(hidden,3*(node_count-1))
            nn.init.zeros_(self.correction_head.weight)
            nn.init.zeros_(self.correction_head.bias)

    def forward(self, positions, velocities, *, output_bias_shift=None):
        return self._evaluate(positions,velocities,output_bias_shift=output_bias_shift,split=False)

    def components(self, positions, velocities, *, output_bias_shift=None):
        """Return legacy correction and optional extra acceleration separately.

        The added term is an effective model discrepancy, not a conservative
        elastic force. Each free-node axis is bounded by acceleration_limit.
        Exposing it separately permits a magnitude penalty during future fitting.
        """
        return self._evaluate(positions,velocities,output_bias_shift=output_bias_shift,split=True)

    def _evaluate(self,positions,velocities,*,output_bias_shift,split):
        # Translation invariant; absolute velocity is retained for external drag.
        relative_q = positions - positions[:, :1]
        relative_v = (velocities - velocities[:, :1]) / 2.0
        features = torch.cat((relative_q.flatten(1), relative_v.flatten(1),
                              velocities[:, 0] / 2.0), dim=1)
        if self.mode=='dissipative_plus_acceleration':
            hidden=features
            for layer in list(self.net.children())[:-1]:hidden=layer(hidden)
            output=self.net[-1](hidden)
            extra=self.acceleration_limit*torch.tanh(self.correction_head(hidden).reshape(-1,self.node_count-1,3))
        else:
            output=self.net(features)
            extra=None
        if output_bias_shift is not None:output=output+output_bias_shift
        raw=output.reshape(-1,self.node_count-1,3)
        if self.mode in ('dissipative','dissipative_plus_acceleration'):
            # State-dependent NN damping, initialized to exactly zero. No scalar
            # drag parameter or calibrated-drag initialization is present.
            # abs uses the symmetric subgradient at zero, matching a centered
            # finite-difference check at this nonnegative boundary.
            value=torch.nn.functional.softplus(raw)-math.log(2.)
            positive=.5*(value+value.abs())
            coefficient=self.damping_limit_s_inv*torch.tanh(positive)
            correction=-coefficient*velocities[:,1:]
        else:
            correction=self.acceleration_limit*torch.tanh(raw)
        # No learned drone force: only cable nodes receive discrepancy forces.
        zero=torch.zeros_like(positions[:,:1])
        legacy=torch.cat((zero,correction),dim=1)
        if extra is None:
            return (legacy,torch.zeros_like(positions)) if split else legacy
        additional=torch.cat((zero,extra),dim=1)
        return (legacy,additional) if split else legacy+additional

    def drag_coefficient(self):
        return torch.nn.functional.softplus(self.raw_drag) if self.learn_drag else None

    def drag_rates(self, reference):
        """Learned cable damping, integrated by the solver's exponential step.

        This is the structured part of the residual; forward() supplies only the
        additional NN acceleration. Never apply this damping to the drone node.
        """
        weights = reference.new_ones(self.node_count)
        weights[:1].zero_()  # Device operation; scalar assignment copies from CPU during CUDA capture.
        return weights * self.drag_coefficient() if self.learn_drag else weights * 0

    def specification(self):
        result = dict(node_count=self.node_count, hidden=self.hidden,
                      acceleration_limit=self.acceleration_limit)
        if self.learn_drag:
            result.update(learn_drag=True, initial_drag_s_inv=self.initial_drag_s_inv)
        if self.mode!='acceleration':
            result.update(mode=self.mode,damping_limit_s_inv=self.damping_limit_s_inv)
        return result


def with_acceleration_correction(network, *, acceleration_limit):
    """Create a zero-extension candidate without modifying the fitted source.

    Only dissipative checkpoints can be extended this way. Their damping
    weights and outputs are preserved; the caller must explicitly choose the
    new per-axis acceleration bound and save/select a separate candidate.
    """
    if network.mode!='dissipative' or network.learn_drag:
        raise ValueError('Only a dissipative residual can be extended without changing its meaning')
    if not math.isfinite(acceleration_limit) or acceleration_limit<=0:
        raise ValueError('Additional acceleration requires a finite positive bound')
    result=deepcopy(network)
    result.mode='dissipative_plus_acceleration'
    result.acceleration_limit=float(acceleration_limit)
    reference=result.net[-1].weight
    result.correction_head=nn.Linear(result.hidden,3*(result.node_count-1),
        device=reference.device,dtype=reference.dtype)
    nn.init.zeros_(result.correction_head.weight)
    nn.init.zeros_(result.correction_head.bias)
    result.correction_head.train(network.training)
    result.correction_head.requires_grad_(reference.requires_grad)
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
