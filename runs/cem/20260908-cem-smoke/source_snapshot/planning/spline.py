"""Clamped quintic position splines with analytic P/V/A in world coordinates."""
import numpy as np
from scipy.interpolate import BSpline


def knots(count):
    if count < 6:
        raise ValueError('At least six control points are required.')
    return np.r_[np.zeros(6), np.arange(1, count-5)/(count-5), np.ones(6)]


def duration_frames(seconds):
    return max(1, int(round(float(seconds)*30)))


def sample(coefficients, duration, times):
    """First three identical coefficients enforce the settled P/V/A boundary."""
    c = np.asarray(coefficients, float)
    if c.ndim != 2 or c.shape[1] != 3 or not np.isfinite(c).all() or duration <= 0:
        raise ValueError('Invalid spline coefficients or duration.')
    spline = BSpline(knots(len(c)), c, 5)
    u = np.clip(np.asarray(times)/duration, 0, 1)
    return np.concatenate([spline(u, nu=j)/duration**j for j in range(3)] +
                          [np.zeros((len(u), 2))], axis=1)


def fit_seed(packets, duration, count=12):
    """Approximate the seed without translating it; fitting is subsequently simulated."""
    p = np.asarray(packets, float)
    u = np.linspace(0, 1, len(p))
    basis = BSpline(knots(count), np.eye(count), 5)
    fixed = np.repeat(p[0:1, :3], 3, axis=0)
    # Fit positions and derivatives together in dimensionally scaled units.
    matrices = [basis(u, nu=j) for j in range(3)]
    weights = [1., .12, .015]
    a = np.concatenate([w*m[:, 3:] for w, m in zip(weights, matrices)])
    b = np.concatenate([w*(p[:, j*3:j*3+3]*duration**j-m[:, :3]@fixed)
                        for j, (w, m) in enumerate(zip(weights, matrices))])
    return np.r_[fixed, np.linalg.lstsq(a, b, rcond=None)[0]]


def decode(vector, origin, count, duration_bounds):
    frames = np.clip(duration_frames(vector[-1]),
                     int(np.ceil(duration_bounds[0]*30-1e-9)),
                     int(np.floor(duration_bounds[1]*30+1e-9)))
    c = np.r_[np.repeat(np.asarray(origin)[None], 3, axis=0),
              np.asarray(vector[:-1]).reshape(count-3, 3)]
    return c, float(frames/30)
