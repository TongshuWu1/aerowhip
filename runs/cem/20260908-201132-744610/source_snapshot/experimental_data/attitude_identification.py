"""Identify an independent nominal attitude mapping with translation frozen.

The two positive direction scales are empirical, not vehicle thrust constants.
Only maneuver rotations enter the loss. Post-hold commands/measurements do not.
"""
import itertools
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .nominal_pose_fit import integration_steps

LOWER=np.array([.1,.05,.02])
UPPER=np.array([3.,1.5,.30])
DOMAIN_ERROR='Attitude error is too close to 180 degrees for this local response model'


def command_frames(a,yaw):
    specific=a+np.array([0,0,9.80665]);norm=np.linalg.norm(specific,axis=-1,keepdims=True)
    if (norm<1e-7).any():raise ValueError('Desired specific-force direction is undefined')
    z=specific/norm;heading=np.stack([np.cos(yaw),np.sin(yaw),np.zeros_like(yaw)],axis=-1)
    y=np.cross(z,heading);norm=np.linalg.norm(y,axis=-1,keepdims=True)
    if (norm<1e-7).any():raise ValueError('Desired heading and specific-force direction are singular')
    y/=norm
    return np.stack([np.cross(y,z),y,z],axis=-1)


def local_log(r):
    cosine=(np.trace(r)-1)*.5
    if cosine<-.9999:raise ValueError(DOMAIN_ERROR)
    v=np.array([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])*.5
    sine=np.linalg.norm(v)
    factor=1+sine*sine/6 if sine<1e-6 and cosine>0 else np.arctan2(sine,cosine)/max(sine,1e-12)
    return v*factor


class AttitudeTrial:
    """Cache the fixed translational recurrence, not measured future motion."""
    def __init__(self,trial,gains,delay):
        self.name=trial.name
        self.time=trial.time[trial.time<=trial.context['csv_end_s']]
        self.mask=trial.masks['fit_orientation'][:len(self.time)].copy()
        self.truth=trial.truth['rotation_tracking_to_world'][:len(self.time)].copy()
        self.dt,cmd,self.indices=integration_steps(trial.schedule,self.time,delay)
        g=np.asarray(gains);kp=g[:2][[0,0,1]];kd=g[2:4][[0,0,1]];ff=g[4:][[0,0,1]]
        p=trial.p0.copy();v=trial.v0.copy();b=trial.b0+trial.basis@g
        aa=[];am=[]
        for h,c in zip(self.dt,cmd):
            a=kp*(c[:3]-p)+kd*(c[3:6]-v)+ff*c[6:9]+b
            pm=p+.5*h*v;vm=v+.5*h*a
            mid=kp*(c[:3]-pm)+kd*(c[3:6]-vm)+ff*c[6:9]+b
            aa.append(a);am.append(mid);p=p+h*vm;v=v+h*mid
        self.a=np.asarray(aa);self.am=np.asarray(am);self.yaw=cmd[:,9]
        state=trial.state(g,'cpu')
        self.r0=state.rotation[0].numpy();self.w0=state.omega_tracking[0].numpy()
        self.mean_a=trial.b0.copy();self.hover_yaw=float(trial.hover[2][0,0,9])
        u,_,vh=np.linalg.svd(trial.hover[1][0].mean(axis=0))
        self.mean_rotation=u@np.diag([1,1,np.linalg.det(u@vh)])@vh

    def predict(self,values):
        sx,sz,tau=values;scale=np.array([sx,sx,sz])
        align=command_frames(self.mean_a*scale,np.asarray(self.hover_yaw)).T@self.mean_rotation
        desired=command_frames(self.a*scale,self.yaw)@align
        desired_mid=command_frames(self.am*scale,self.yaw)@align
        r=self.r0.copy();w=self.w0.copy();out=[r.copy()]
        for h,rd,rmid in zip(self.dt,desired,desired_mid):
            alpha=local_log(r.T@rd)/tau**2-2*w/tau
            rm=r@Rotation.from_rotvec(.5*h*w).as_matrix();wm=w+.5*h*alpha
            alpham=local_log(rm.T@rmid)/tau**2-2*wm/tau
            r=r@Rotation.from_rotvec(h*wm).as_matrix();w=w+h*alpham
            out.append(r.copy())
        return np.asarray(out)[self.indices]


def fit_attitude_mapping(trials,gains,delay,*,progress=print):
    cached=[AttitudeTrial(t,gains,delay) for t in trials];trace=[]
    count=sum(int(t.mask.sum())*3 for t in cached)
    def residual(log_values):
        values=np.exp(log_values);rows=[]
        try:
            for t in cached:
                predicted=t.predict(values)[t.mask]
                error=Rotation.from_matrix(predicted.transpose(0,2,1)@t.truth[t.mask]).as_rotvec()
                rows.append(error.ravel()/np.sqrt(t.mask.sum()*len(cached)))
            out=np.concatenate(rows);loss=float(out@out)
            trace.append(dict(values=values.tolist(),valid=True,mean_squared_angle_rad2=loss))
        except ValueError as exc:
            if str(exc) not in (DOMAIN_ERROR,'Desired heading and specific-force direction are singular','Desired specific-force direction is undefined'):raise
            trace.append(dict(values=values.tolist(),valid=False,mean_squared_angle_rad2=None,reason=str(exc),take=t.name))
            return np.full(count,100/np.sqrt(count))
        if not np.isfinite(loss):raise FloatingPointError('Non-finite attitude loss')
        if len(trace)%50==0:progress(f'Attitude mapping evaluation {len(trace)}: RMS {np.degrees(np.sqrt(loss)):.2f} deg',flush=True)
        return out
    # Fixed coarse starts and refinement use training rotations only.
    grid=list(itertools.product([.5,1.,1.5],[.25,.5,.75],[.03,.07,.15]))
    evaluated=[]
    for row in grid:
        residual(np.log(row))
        if trace[-1]['valid']:evaluated.append(trace[-1])
    if not evaluated:
        return dict(valid=False,tau_s=None,scales=None,near_bound=None,trace=trace,reason='No valid attitude mapping on fixed grid')
    starts=sorted(evaluated,key=lambda x:x['mean_squared_angle_rad2'])[:3];optimizers=[]
    for start in starts:
        result=least_squares(residual,np.log(start['values']),bounds=(np.log(LOWER),np.log(UPPER)),
            max_nfev=65,ftol=2e-6,xtol=2e-6,gtol=2e-6,diff_step=1e-4)
        optimizers.append(dict(values=np.exp(result.x).tolist(),success=bool(result.success),nfev=result.nfev,message=result.message))
    best=min((row for row in trace if row['valid']),key=lambda row:row['mean_squared_angle_rad2'])
    sx,sz,tau=best['values'];values=np.array(best['values'])
    return dict(valid=True,tau_s=tau,scales=[sx,sz],mean_squared_angle_rad2=best['mean_squared_angle_rad2'],
        near_bound=bool((np.minimum(values-LOWER,UPPER-values)<.01*(UPPER-LOWER)).any()),
        bound_flags={k:bool(min(values[i]-LOWER[i],UPPER[i]-values[i])<.01*(UPPER[i]-LOWER[i])) for i,k in enumerate(['scale_xy','scale_z','tau_s'])},
        bounds=[LOWER.tolist(),UPPER.tolist()],trace=trace,optimizers=optimizers,
        model='independent_scale_v3',position_parameters_frozen=True)
