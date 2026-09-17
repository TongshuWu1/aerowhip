"""Smooth position references for offline coupled-model trajectory search.

Clamped quintic B-spline: C4 at simple interior knots. The first three control
points fix a settled start; the remaining nine XYZ points are optimized. No
terminal rest condition is imposed on the scored whip. SciPy builds small basis
matrices once on CPU; candidate evaluation is batched Torch on CPU or CUDA.
"""
import numpy as np
import torch
from scipy.interpolate import BSpline

SCHEMA = 'position_spline_pva_30hz_v1'


class PositionSpline:
    degree = 5
    controls = 12
    free_controls = 9

    def __init__(self, duration, *, device='cpu', rate=30):
        self.duration=float(duration)
        if not np.isfinite(duration) or duration <= 0 or not np.isclose(duration*rate, round(duration*rate)):
            raise ValueError('Spline duration must contain whole command intervals')
        self.steps=round(duration*rate)
        self.knots=np.r_[np.zeros(6),np.arange(1,7)/7,np.ones(6)]*duration
        self.basis=BSpline(self.knots,np.eye(12),5)
        self.device=device
        self.time=np.arange(self.steps+1)/rate
        self.matrices=[torch.tensor(self.basis(self.time,nu=k),dtype=torch.float64,device=device) for k in range(3)]
        self.jerk_basis=torch.tensor(self.basis((self.time[:-1]+self.time[1:])/2,nu=3),dtype=torch.float64,device=device)
        # Derivative control coefficients bound jerk throughout every span by
        # the B-spline convex-hull property, including between command samples.
        derivative=self.basis.derivative(3)
        self.jerk_control_basis=torch.tensor(derivative.c[:len(derivative.t)-derivative.k-1],dtype=torch.float64,device=device)

    def control_points(self, free, origin):
        origin=torch.as_tensor(origin,dtype=free.dtype,device=free.device)
        fixed=origin.expand(*free.shape[:-2],3,3)
        return torch.cat((fixed,free),-2)

    def decode(self, free, origin):
        if free.shape[-2:]!=(9,3):raise ValueError('Nine XYZ spline control points required')
        points=self.control_points(free,origin)
        pva=[torch.einsum('tc,...cd->...td',b,points) for b in self.matrices]
        packets=torch.cat((*pva,torch.zeros_like(pva[0][...,:2])),-1)
        jerk=torch.einsum('tc,...cd->...td',self.jerk_basis,points)
        return packets,jerk

    def jerk_valid(self, free, origin, limits):
        points=self.control_points(free,origin)
        controls=torch.einsum('kc,...cd->...kd',self.jerk_control_basis,points)
        return torch.isfinite(controls).all((-1,-2)) & (controls.abs()<=limits+1e-8).all((-1,-2))

    def scratch_noise_basis(self):
        """Full-rank position covariance from spline geometry, not saved motion.

        Inverting the free-point third-derivative map favors coherent smooth
        excursions. Normalize maximum control-point standard deviation to 1 m.
        These are position proposals; PVA still comes from the position spline.
        """
        basis=torch.linalg.inv(self.jerk_control_basis[:,3:])
        return basis/basis.square().sum(-1).sqrt().max()

    def fit_positions(self, positions, origin):
        """Initialize from command positions only; no cable forecasts are used."""
        origin=torch.as_tensor(origin,dtype=positions.dtype,device=positions.device)
        rhs=positions-torch.einsum('tc,cd->td',self.matrices[0][:,:3],origin.expand(3,3))
        return torch.einsum('ct,...td->...cd',torch.linalg.pinv(self.matrices[0][:,3:]),rhs)


def replay(env, free, *, trace=False, observer=None):
    spline=PositionSpline(env.steps/30,device=env.device)
    packets,jerk=spline.decode(free,env.origin0[0])
    admissible=spline.jerk_valid(free,env.origin0[0],env.limit)
    result=env.rollout(actions=jerk/env.limit,packets=packets,trace=trace,observer=observer)
    result['failed'] |= ~admissible
    result['success'] &= admissible
    if 'fold_valid' in result:result['fold_valid'] &= admissible
    return result


def scratch_proposals(means, origin, basis, settings, rng):
    """Mix local updates with fresh zero-mean launch-centered exploration."""
    groups=len(means);per=settings['samples']//groups
    noise=torch.randn(groups,per,9,3,device=means.device,dtype=means.dtype,generator=rng)
    noise=torch.einsum('ij,gpjc->gpic',basis,noise)
    scales=means.new_tensor(settings['position_noise_scales_m'])
    noise*=scales[torch.arange(per,device=means.device)%len(scales)][None,:,None,None]
    centers=means[:,None].expand(-1,per,-1,-1).clone()
    fresh=round(per*settings['fresh_sample_fraction'])
    if fresh:centers[:,:fresh]=origin
    return centers+noise


class RetainedM0SplineProposals:
    """Residual position controls around the user's selected saved motion."""
    rows=9
    def __init__(self,baseline,settings,noise_basis):
        self.baseline=baseline;self.settings=settings;self.noise_basis=noise_basis

    def sample(self,means,rng):
        per=self.settings['samples']//len(means)
        noise=torch.randn(len(means),per,9,3,device=means.device,dtype=means.dtype,generator=rng)
        noise=torch.einsum('ij,gpjc->gpic',self.noise_basis,noise)
        scales=means.new_tensor(self.settings['position_noise_scales_m'])
        return means[:,None]+noise*scales[torch.arange(per,device=means.device)%len(scales)][None,:,None,None]

    def decode(self,controls):return self.baseline+controls


def fit_bounded_positions(spline,positions,origin,jerk_limits):
    """Least-squares conversion inside the saved continuous jerk envelope."""
    from scipy.optimize import minimize,LinearConstraint
    original=positions.shape
    values=positions.detach().cpu().numpy().reshape(-1,original[-2],3)
    start=np.asarray(torch.as_tensor(origin).cpu(),dtype=float)
    limits=np.asarray(torch.as_tensor(jerk_limits).cpu(),dtype=float)
    basis=spline.matrices[0].cpu().numpy()[:,3:]
    derivative=spline.jerk_control_basis.cpu().numpy()[:,3:]
    output=[]
    for reference in values:
        axes=[]
        for axis in range(3):
            rhs=reference[:,axis]-start[axis]
            def objective(x):
                error=basis@x-rhs
                return .5*error@error,basis.T@error
            fit=minimize(objective,np.zeros(9),jac=True,method='SLSQP',
                constraints=[LinearConstraint(derivative,-limits[axis]+1e-6,limits[axis]-1e-6)],
                options={'ftol':1e-12,'maxiter':500})
            if not fit.success or np.max(np.abs(derivative@fit.x))>limits[axis]+1e-8:
                raise ValueError('Could not convert saved positions within the jerk envelope')
            axes.append(fit.x+start[axis])
        output.append(np.stack(axes,axis=-1))
    return torch.as_tensor(np.stack(output).reshape(*original[:-2],9,3),dtype=positions.dtype,device=positions.device)
