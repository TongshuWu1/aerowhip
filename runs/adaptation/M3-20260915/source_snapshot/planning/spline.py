"""Clamped quintic position splines with analytic P/V/A in world coordinates."""
import numpy as np
from scipy.interpolate import BSpline, PPoly


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


def initialize_at_origin(packets, duration, count, origin):
    """Reanchor a seed guess before optimization, never a saved/exported result."""
    coefficients = fit_seed(packets, duration, count)
    coefficients += np.asarray(origin, float)-coefficients[0].copy()
    coefficients[:3] = origin
    return coefficients


def polynomial_feasible(c, duration, limits, height_bounds):
    """Check continuous polynomial extrema, including between CSV samples."""
    from numpy.polynomial import polynomial as poly
    from deployment.curved_recovery import metrics, extrema, squared_norm
    m=metrics(c,duration)
    a=poly.polyder(c,m=2,axis=0)/duration**2
    vertical=a[:,2].copy();vertical[0]+=9.80665
    tilt=poly.polysub(squared_norm(a[:,:2]),
        np.tan(np.deg2rad(limits['maximum_tilt_deg']))**2*poly.polymul(vertical,vertical))
    return (m['minimum_height_m']>=height_bounds[0]-1e-9 and m['peak_height_m']<=height_bounds[1]+1e-9
        and m['peak_speed_m_s']<=limits['maximum_speed_m_s']+1e-8
        and m['peak_specific_force_m_s2']<=limits['maximum_specific_force_m_s2']+1e-8
        and m['minimum_specific_vertical_m_s2']>=limits['minimum_specific_vertical_m_s2']-1e-8
        and extrema(tilt)[1]<=1e-7)


def spline_feasible(c,duration,limits,height_bounds):
    """Convert each quintic span to normalized polynomial and bound it exactly."""
    pp=[PPoly.from_spline((knots(len(c)),c[:,j],5)) for j in range(3)]
    for i,(left,right) in enumerate(zip(pp[0].x[:-1],pp[0].x[1:])):
        h=right-left
        if h<=0:continue
        coefficients=np.column_stack([p.c[::-1,i] for p in pp])*h**np.arange(6)[:,None]
        if not polynomial_feasible(coefficients,duration*h,limits,height_bounds):return False
    return True
