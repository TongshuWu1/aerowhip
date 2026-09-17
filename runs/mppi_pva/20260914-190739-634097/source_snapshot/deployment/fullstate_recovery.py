"""Historical v2 recovery prototype, superseded by export-only recorded v3.

Not used by the UI or rehearsal worker. Retained to interpret existing v2 artifacts.
"""
import numpy as np
import torch

DEFAULT_RECOVERY = dict(brake_s=.5, return_s=3., hold_s=3.)


def quintic(p0, v0, a0, p1, duration):
    """Physical-time coefficients, with zero terminal velocity/acceleration."""
    p0, v0, a0, p1 = map(lambda x: np.asarray(x, dtype=float), (p0, v0, a0, p1))
    T = float(duration)
    c = np.zeros((6, 3))
    c[:3] = p0, v0, a0/2
    matrix = np.array([[T**3,T**4,T**5],[3*T*T,4*T**3,5*T**4],[6*T,12*T*T,20*T**3]])
    c[3:] = np.linalg.solve(matrix, np.stack((p1-p0-v0*T-a0*T*T/2, -v0-a0*T, -a0)))
    return c


def evaluate(c, time):
    t = np.asarray(time, dtype=float).reshape(-1, 1)
    p = sum(c[k]*t**k for k in range(6))
    v = sum(k*c[k]*t**(k-1) for k in range(1,6))
    a = sum(k*(k-1)*c[k]*t**(k-2) for k in range(2,6))
    return p, v, a


def recovery_path(position, velocity, acceleration, hover_position, settings=None):
    options = dict(DEFAULT_RECOVERY, **(settings or {}))
    durations = np.array([options[k] for k in DEFAULT_RECOVERY], dtype=float)
    hover = np.asarray(hover_position, dtype=float)
    if (not np.isfinite(durations).all() or np.any(durations <= 0)
            or np.any(durations > 60) or hover.shape != (3,) or not np.isfinite(hover).all()):
        raise ValueError('Recovery durations must be finite, positive and <=60 s; hover must be finite XYZ')
    brake, back, hold = durations
    # A free braking endpoint avoids forcing an instant reversal toward hover.
    stop = np.asarray(position) + np.asarray(velocity)*brake/2 + np.asarray(acceleration)*brake**2/12
    coefficients = [quintic(position, velocity, acceleration, stop, brake),
                    quintic(stop, np.zeros(3), np.zeros(3), hover, back)]
    def sample(time):
        t = np.asarray(time, dtype=float)
        if np.any(t < -1e-10) or np.any(t > sum(durations)+1e-10):
            raise ValueError('Recovery time outside its duration')
        p = np.broadcast_to(hover, (len(t),3)).copy()
        v, a = np.zeros_like(p), np.zeros_like(p)
        for mask, c, local in [(t < brake, coefficients[0], t),
                                ((t >= brake) & (t < brake+back), coefficients[1], t-brake)]:
            if mask.any():
                p[mask], v[mask], a[mask] = evaluate(c, local[mask])
        return p, v, a
    return sample, dict(**options, hover_position_m=hover.tolist(), brake_position_m=stop.tolist(),
                       coefficients=[c.tolist() for c in coefficients])


def complete_reference(trajectory, hover_position, settings=None):
    from .fullstate import sample_fullstate
    t, p, v, a = sample_fullstate(trajectory['time_s'], trajectory['positions_m'][:,0],
                                 trajectory['velocities_m_s'][:,0])
    cutoff = float(t[-1])
    sample, details = recovery_path(p[-1], v[-1], a[-1], hover_position, settings)
    brake_end = cutoff+details['brake_s']
    return_end = brake_end+details['return_s']
    end = return_end+details['hold_s']
    # Keep every original whip row; add exact phase boundaries to the global 30 Hz grid.
    extra = np.arange(int(np.ceil(end*30))+1)/30
    extra = extra[(extra > cutoff+1e-10) & (extra < end-1e-10)]
    boundaries = np.array([brake_end,return_end,end])
    extra = extra[np.all(abs(extra[:,None]-boundaries)>1e-10,axis=1)]
    extra = np.sort(np.r_[extra,boundaries])
    rp, rv, ra = sample(extra-cutoff)
    times = np.r_[t,extra]
    phases = np.full(len(times),'whip',dtype='<U16')
    phases[times > cutoff+1e-10] = 'recovery_brake'
    phases[times >= brake_end-1e-10] = 'recovery_return'
    phases[times >= return_end-1e-10] = 'hover_hold'
    return dict(time_s=times, position_m=np.vstack((p,rp)), velocity_m_s=np.vstack((v,rv)),
                acceleration_m_s2=np.vstack((a,ra)), phase=phases), sample, dict(
                    **details, whip_end_s=cutoff, brake_end_s=brake_end,
                    return_end_s=return_end, total_duration_s=end)


@torch.no_grad()
def prepare_complete_trajectory(trajectory, model_config, hover_position, *, settings=None,
                                device='cpu', cancel_requested=None):
    """Preserve the source strike, then drive the cable pivot along recovery.

    This predicts cable motion under ideal attachment tracking, not motor/firmware response.
    """
    from simulator.point_mass import ForceControlledPointCable
    from simulator.cable import DderState, START_PINNED_FREE_END
    reference, sample, details = complete_reference(trajectory, hover_position, settings)
    model = ForceControlledPointCable.from_mapping(model_config).dder
    q = torch.tensor(trajectory['positions_m'][-1:]).to(device)
    v = torch.tensor(trajectory['velocities_m_s'][-1:]).to(device)
    constants = model.runtime_constants(q)
    duration = details['total_duration_s']-details['whip_end_s']
    local = np.arange(1, int(np.ceil(duration/.01)))*.01
    local = np.r_[local[local < duration-1e-10],duration]
    roots, root_v, _ = sample(local)
    positions, velocities = [], []
    previous = 0.
    for t, root, velocity in zip(local, roots, root_v):
        if cancel_requested and cancel_requested():
            raise InterruptedError('Full trajectory preparation cancelled')
        state = model.step_runtime(DderState(q,v),q.new_tensor(root)[None,None],
            q.new_tensor([t-previous]),constants,pinned_endpoints=START_PINNED_FREE_END,
            iterative_damping=False,dense_constraint_solve=True,analytic_bending=True)
        q,v = state.positions_m,state.velocities_m_s
        v[:,0] = v.new_tensor(velocity)
        positions.append(q[0].cpu().numpy().copy())
        velocities.append(v[0].cpu().numpy().copy())
        previous = t
    preview = dict(time_s=np.r_[trajectory['time_s'],details['whip_end_s']+local],
                   positions_m=np.concatenate((trajectory['positions_m'],positions)),
                   velocities_m_s=np.concatenate((trajectory['velocities_m_s'],velocities)))
    if not all(np.isfinite(x).all() for x in preview.values()):
        raise ValueError('Nonfinite full trajectory: export refused')
    lengths = np.linalg.norm(np.diff(preview['positions_m'],axis=1),axis=2)
    rest = np.asarray(model.parameters.rest_lengths_m)
    details['max_segment_length_error_m'] = float(abs(lengths-rest).max())
    details['final_max_cable_speed_m_s'] = float(np.linalg.norm(preview['velocities_m_s'][-1,1:],axis=1).max())
    details['minimum_cable_height_m'] = float(preview['positions_m'][:,:,2].min())
    if details['max_segment_length_error_m'] > .02:
        raise ValueError('Cable constraint error exceeds 2 cm during full reference preview')
    return dict(trajectory, reference=reference, recovery=details, preview=preview)
