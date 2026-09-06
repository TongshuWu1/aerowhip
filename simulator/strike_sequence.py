"""Freeze a bounded single-strike follow-through before plant execution."""
import math
import torch


def freeze_followthrough(forces, cutoffs, *, dt_s, maximum_steps, duration_s=0.):
    duration_s=float(duration_s)
    if not math.isfinite(duration_s) or duration_s<0:
        raise ValueError('Strike follow-through must be finite and nonnegative.')
    steps=round(duration_s/dt_s)
    if not math.isclose(steps*dt_s,duration_s,abs_tol=1e-9):
        raise ValueError('Strike follow-through must align with the physics time step.')
    if not steps or not bool((cutoffs>0).any()):
        return forces,cutoffs
    extended=torch.where(cutoffs>0,(cutoffs+steps).clamp_max(maximum_steps),cutoffs)
    indices=torch.arange(int(extended.max()),device=forces.device)[:,None]
    # Hold the last force of this strike; never append another policy query or
    # use any state/contact information from the execution plant.
    indices=torch.minimum(indices,(cutoffs-1).clamp_min(0)[None]).clamp_max(len(forces)-1)
    frozen=forces[indices,torch.arange(len(cutoffs),device=forces.device)[None]]
    return frozen,extended
