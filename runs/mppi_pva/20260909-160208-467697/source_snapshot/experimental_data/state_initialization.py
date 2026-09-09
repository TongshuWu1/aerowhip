"""Causal cable state estimation from a supplied pre-handover position history.

The functions accept history only, never a future prediction target. The optional
DER fit adjusts state, not material parameters or neural weights.
"""
from dataclasses import dataclass
import time

import torch

from simulator.cable import DderState, START_PINNED_FREE_END


@dataclass(frozen=True)
class HistoryFitSettings:
    derivative_samples: int = 11
    history_steps: int = 20
    updates: int = 15
    learning_rate: float = .05
    position_adjustment_m: float = .01
    velocity_adjustment_m_s: float = .3
    position_prior: float = .1
    velocity_prior_s2: float = .001


def endpoint_velocity(history, dt_s, samples=11):
    """Quadratic least-squares endpoint derivative, using only the supplied past."""
    if history.ndim != 4 or history.shape[1] < samples or samples < 3 or dt_s <= 0:
        raise ValueError('Need BxTxNx3 history, positive dt, and at least three past samples.')
    if not bool(torch.isfinite(history[:, -samples:]).all()):
        raise ValueError('Initializer history contains invalid observations.')
    # Dimensionless time improves conditioning; the endpoint is exactly time zero.
    time_axis = torch.linspace(-1, 0, samples, dtype=history.dtype, device=history.device)
    design = torch.stack((torch.ones_like(time_axis), time_axis, time_axis.square()), -1)
    weights = torch.linalg.pinv(design)[1] / ((samples - 1) * dt_s)
    return torch.einsum('t,btnc->bnc', weights, history[:, -samples:])


def project_state(model, positions, velocities, *, passes=8):
    boundary = positions[:, :1].clone()
    root_velocity = velocities[:, :1].clone()
    for _ in range(passes):
        positions = model.project_lengths(positions, boundary, pinned_endpoints=START_PINNED_FREE_END)
    velocities = model.project_velocities(positions, velocities, root_velocity,
                                         pinned_endpoints=START_PINNED_FREE_END)
    return DderState(positions, velocities)


def causal_state(history, dt_s, model, *, samples=11):
    return project_state(model, history[:, -1], endpoint_velocity(history, dt_s, samples))


def history_rollout(model, state, boundary, dt_s, *, gradients=False):
    """Return full node states through an already-observed attachment history."""
    constants = model.runtime_constants(state.positions_m)
    dt = state.positions_m.new_full((len(state.positions_m),), dt_s)
    positions = [state.positions_m]
    for i in range(1, boundary.shape[1]):
        state = model.step_runtime(state, boundary[:, i, None], dt, constants,
            iterative_damping=False, pinned_endpoints=START_PINNED_FREE_END,
            create_graph=gradients, dense_constraint_solve=True, analytic_bending=True)
        positions.append(state.positions_m)
    return torch.stack(positions, 1), state


def physics_assisted_state(history, dt_s, model, marker_nodes, settings=HistoryFitSettings()):
    """Fit a bounded initial-state correction against past observations only.

    Per-window selection includes the unadjusted history rollout. A lower history
    loss does not imply a better future prediction; the benchmark must test that.
    """
    if model.motion_residual is not None:
        raise ValueError('Initialization comparison requires a physical-only model.')
    needed = settings.history_steps + settings.derivative_samples
    if history.shape[1] < needed:
        raise ValueError(f'Physics initialization needs at least {needed} history samples.')
    if not bool(torch.isfinite(history).all()):
        raise ValueError('Initializer history contains invalid observations.')
    started = time.perf_counter()
    seed_history = history[:, :-settings.history_steps]
    with torch.no_grad():
        seed = causal_state(seed_history, dt_s, model, samples=settings.derivative_samples)
    observations = history[:, -(settings.history_steps + 1):].detach()
    raw = torch.nn.Parameter(history.new_zeros((2, len(history), history.shape[2] - 1, 3)))
    optimizer = torch.optim.Adam([raw], lr=settings.learning_rate)
    marker_nodes = torch.as_tensor(marker_nodes, device=history.device)

    def evaluate():
        dq = settings.position_adjustment_m * raw[0].tanh()
        dv = settings.velocity_adjustment_m_s * raw[1].tanh()
        zero = torch.zeros_like(seed.positions_m[:, :1])
        initial = project_state(model, seed.positions_m + torch.cat((zero, dq), 1),
            seed.velocities_m_s + torch.cat((zero, dv), 1), passes=2)
        predicted, final = history_rollout(model, initial, observations[:, :, 0], dt_s,
                                          gradients=torch.is_grad_enabled())
        errors = predicted.index_select(2, marker_nodes) - observations.index_select(2, marker_nodes)
        mse = errors.square().sum(-1).mean((1, 2))
        penalty = settings.position_prior * dq.square().sum(-1).mean(1)
        penalty = penalty + settings.velocity_prior_s2 * dv.square().sum(-1).mean(1)
        return mse + penalty, mse, final

    with torch.no_grad():
        best_loss, initial_mse, initial_final = evaluate()
        best_loss = best_loss.clone()
        best_mse = initial_mse.clone()
        best_q, best_v = initial_final.positions_m.clone(), initial_final.velocities_m_s.clone()
    selected = torch.zeros(len(history), dtype=torch.long, device=history.device)
    max_gradient, completed, nonfinite = 0., 0, False
    for update in range(1, settings.updates + 1):
        optimizer.zero_grad()
        losses, _, _ = evaluate()
        if not bool(torch.isfinite(losses).all()):
            nonfinite = True
            break
        losses.mean().backward()
        if raw.grad is None or not bool(torch.isfinite(raw.grad).all()):
            nonfinite = True
            break
        norm = torch.nn.utils.clip_grad_norm_([raw], 1.)
        max_gradient = max(max_gradient, float(norm))
        optimizer.step()
        with torch.no_grad():
            candidate, mse, final = evaluate()
            good = torch.isfinite(candidate) & (candidate < best_loss)
            good &= torch.isfinite(final.positions_m).all((1, 2)) & torch.isfinite(final.velocities_m_s).all((1, 2))
            best_loss[good] = candidate[good]
            best_mse[good] = mse[good]
            best_q[good] = final.positions_m[good]
            best_v[good] = final.velocities_m_s[good]
            selected[good] = update
        completed = update
    diagnostics = dict(elapsed_s=time.perf_counter()-started, batch_size=len(history),
        updates=completed, max_gradient_norm_before_clipping=max_gradient,
        stopped_for_nonfinite=nonfinite, selected_update=selected.tolist(),
        initial_history_rmse_m=initial_mse.sqrt().tolist(),
        selected_history_rmse_m=best_mse.sqrt().tolist(),
        selection='per-window minimum past-history MSE plus state prior; no future targets')
    return DderState(best_q.detach(), best_v.detach()), diagnostics
