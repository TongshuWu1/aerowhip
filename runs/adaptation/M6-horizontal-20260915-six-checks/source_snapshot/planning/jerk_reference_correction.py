"""Affine, derivative-consistent corrections to an existing 30 Hz jerk command.

Nine smooth jerk-offset coefficients per axis; zero coefficients reproduce the
executed command exactly. This does not fit or replace it with a position spline.
"""
import numpy as np
import torch
from scipy.interpolate import BSpline
from simulator.pva_commands import jerk_packets


class JerkCorrection:
    control_artifact_key = 'jerk_correction_coefficients_m_s3'

    def __init__(self, initial_packet, original_jerk, *, device='cpu', rate=30):
        self.device = device
        self.original_jerk = torch.as_tensor(original_jerk, dtype=torch.float64, device=device)
        initial = torch.as_tensor(initial_packet, dtype=torch.float64, device=device)
        if initial.shape != (11,) or self.original_jerk.ndim != 2 or self.original_jerk.shape[1] != 3:
            raise ValueError('Expected one initial PVA packet and N XYZ jerk intervals')
        if not bool(torch.isfinite(initial).all() and torch.isfinite(self.original_jerk).all()):
            raise ValueError('Finite command required')
        self.steps = len(original_jerk)
        self.time = np.arange(self.steps+1)/rate
        self.duration = self.steps/rate
        self.original_packets = jerk_packets(initial[None], self.original_jerk[None], dt=1/rate)[0]
        knots = np.r_[np.zeros(4), np.arange(1,6)/6, np.ones(4)]
        b = BSpline(knots, np.eye(9), 3)((np.arange(self.steps)+.5)/self.steps)
        self.offset_basis = torch.as_tensor(b, dtype=torch.float64, device=device)
        # Affine P/V/A Jacobians from exactly the same jerk integration equations.
        impulses = torch.zeros((9,self.steps,3), dtype=torch.float64, device=device)
        impulses[:,:,0] = self.offset_basis.T
        response = jerk_packets(torch.zeros((9,11),dtype=torch.float64,device=device),impulses,dt=1/rate)
        pad = lambda x: torch.cat((x.new_zeros((len(x),3)),x),dim=1)
        self.matrices = [pad(response[:,:,i].T) for i in (0,3,6)]
        self.jerk_control_basis = pad(self.offset_basis)

    def decode(self, free, origin):
        if free.shape[-2:] != (9,3):
            raise ValueError('Expected nine XYZ jerk-offset coefficients')
        delta = [torch.einsum('tc,...cd->...td',b[:,3:],free) for b in self.matrices]
        offset = torch.cat((*delta,torch.zeros_like(delta[0][...,:2])),dim=-1)
        return self.original_packets+offset, self.jerk_values(free,origin)

    def jerk_values(self, free, origin):
        return self.original_jerk+torch.einsum('tc,...cd->...td',self.offset_basis,free)

    def jerk_valid(self, free, origin, limits):
        jerk = self.jerk_values(free,origin)
        return torch.isfinite(jerk).all((-1,-2)) & (jerk.abs()<=limits+1e-8).all((-1,-2))
