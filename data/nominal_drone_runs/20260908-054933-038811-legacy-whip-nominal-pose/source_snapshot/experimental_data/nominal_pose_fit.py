"""Nominal-only bootstrap identification; explicit trials/masks, no active selection.

Translation is independent of attitude in the adopted effective model. Fit its
six shared-XY/separate-Z gains and discrete delay to position, then fit the one
attitude timescale to orientation. No residual, cable fit or PPO is launched.
"""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.spatial.transform import Rotation
import torch

from .drone_pose_response_data import load_nominal_pose_trial
from .io import sha256_file
from simulator.drone_pose_response import (PoseResponseParameters, initialize_from_hover,
    CommandSchedule, predict_pose, rotation_log)
from simulator.geometry import normalized_rotations_xyzw

GAIN_NAMES=('kp_xy','kp_z','kd_xy','kd_z','feedforward_xy','feedforward_z')
LOWER=np.array([.1,.1,.1,.1,0.,0.])
UPPER=np.array([80.,80.,20.,20.,2.,2.])
PRIOR=np.array([4.,4.,3.,3.,1.,1.])
SCALES=np.array([20.,20.,10.,10.,1.,1.])
POSITION_SCALE_M=.05
PRIOR_WEIGHT=.01
DELAYS=np.arange(7)*.02
STARTS=(PRIOR,np.array([16.,16.,8.,8.,1.,1.]),np.array([40.,40.,12.,12.,.7,.7]))


def parameters(gains=PRIOR,tau=.08,delay=.02):
    return PoseResponseParameters(*map(float,gains),float(tau),float(delay))


def read_review(directory,version,name):
    for path in sorted(Path(directory).glob('*.json'),reverse=True):
        review=json.loads(path.read_text(encoding='utf-8'))
        if review.get('processed_version')==version and review.get('trial_id')==name:
            return review,dict(path=str(path),sha256=sha256_file(path))
    return {},None


def fitting_masks(time,truth,*,review=None,exclusions=()):
    """Materialize saved reviews/exclusions; never remove/concatenate raw time.

    Explicit exclusions use [start_s,end_s] on the controller time axis and
    component drone/cable/both. The current fit is drone-only. A caller must
    separately delimit a rollout before physical contact; missing observations
    may be masked without resetting the predicted state.
    """
    t=np.asarray(time);review=review or {}
    allowed=np.ones(len(t),bool)
    lo,hi=review.get('precontact_start_s'),review.get('precontact_end_s')
    if (lo is None)!=(hi is None):raise ValueError('Review needs both interval endpoints')
    if lo is not None:
        if not np.isfinite([lo,hi]).all() or lo>=hi:raise ValueError('Invalid review interval')
        allowed&=(t>=lo)&(t<=hi)
    manual=np.zeros(len(t),bool)
    for row in exclusions:
        if row['component'] not in ('drone','cable','both'):raise ValueError('Unknown exclusion component')
        if not row.get('reason'):raise ValueError('Exclusions require a reason')
        a,b=row['start_s'],row['end_s']
        if not np.isfinite([a,b]).all() or a>b:raise ValueError('Invalid exclusion interval')
        if row['component'] in ('drone','both'):manual|=(t>=a)&(t<=b)
    # Conservative measurement-jump flag, not an aircraft speed limit.
    jump=np.zeros(len(t),bool)
    edges=np.linalg.norm(np.diff(truth['position_origin_m'],axis=0),axis=1)>.1
    jump[:-1]|=edges;jump[1:]|=edges
    measurement_ok=allowed&~manual&~jump
    fit_phase=(truth['execution_phase']=='csv_maneuver')&truth['reference_valid']&~truth['phase_boundary_uncertain']
    masks=dict(manual_allowed=allowed,manual_excluded=manual,position_jump=jump,
        phase_boundary_uncertain=truth['phase_boundary_uncertain'],
        fit_position=fit_phase&measurement_ok&truth['position_valid'],
        fit_orientation=fit_phase&measurement_ok&truth['orientation_valid'],
        score_position=measurement_ok&truth['position_valid'],
        score_orientation=measurement_ok&truth['orientation_valid'],
        score_attachment=measurement_ok&truth['attachment_valid'])
    if masks['fit_position'].sum()<20 or masks['fit_orientation'].sum()<20:
        raise ValueError('Insufficient valid maneuver samples after explicit masks')
    return masks


def integration_steps(schedule,times,delay,maximum_step_s=.005):
    """Same event partition as predict_pose, reused by the linear fit evaluator."""
    times=np.asarray(times,dtype=float)
    events=np.unique(np.r_[times,schedule.time+delay,schedule.valid_until.ravel()+delay])
    events=events[(events>=times[0])&(events<=times[-1])]
    for left,right in zip(events[:-1],events[1:]):
        if right-left>1e-12:schedule.sample((left+right)*.5-delay)
    for t in times:schedule.sample(t-delay)
    steps=[];commands=[];indices=[0]
    for start,end in zip(times[:-1],times[1:]):
        local=np.r_[start,events[(events>start+1e-12)&(events<end-1e-12)],end]
        for a,b in zip(local[:-1],local[1:]):
            count=max(1,int(np.ceil((b-a)/maximum_step_s)))
            h=(b-a)/count;cmd=schedule.sample((a+b)*.5-delay)[0].detach().cpu().numpy()
            steps.extend([h]*count);commands.extend([cmd]*count)
        indices.append(len(steps))
    return np.asarray(steps),np.asarray(commands),np.asarray(indices)


def linear_prediction(gains,p0,v0,b0,basis,plan):
    """Exact algebra of the engine's translational Lie-midpoint substep.

    Analytic gain sensitivities include gain-dependent pre-hover compensation.
    CPU NumPy handles these small recurrences; full pose assessment uses Torch
    on the GPU. This evaluator is verified against the original engine.
    """
    gains=np.asarray(gains);dt,commands,indices=plan
    pick=np.array([0,0,1]);kp=gains[:2][pick];kd=gains[2:4][pick];ff=gains[4:][pick]
    dkp=np.zeros((3,6));dkd=np.zeros((3,6));dff=np.zeros((3,6))
    dkp[np.arange(3),pick]=1;dkd[np.arange(3),pick+2]=1;dff[np.arange(3),pick+4]=1
    b=b0+basis@gains
    p=np.asarray(p0).copy();v=np.asarray(v0).copy()
    sp=np.zeros((3,6));sv=np.zeros((3,6))
    ps=[p.copy()];vs=[v.copy()];jac=[sp.copy()]
    for h,cmd in zip(dt,commands):
        ep,ev=cmd[:3]-p,cmd[3:6]-v
        a=kp*ep+kd*ev+ff*cmd[6:9]+b
        sa=dkp*ep[:,None]-kp[:,None]*sp+dkd*ev[:,None]-kd[:,None]*sv+dff*cmd[6:9,None]+basis
        pm,vm=p+.5*h*v,v+.5*h*a
        smp,smv=sp+.5*h*sv,sv+.5*h*sa
        ep,ev=cmd[:3]-pm,cmd[3:6]-vm
        am=kp*ep+kd*ev+ff*cmd[6:9]+b
        sam=dkp*ep[:,None]-kp[:,None]*smp+dkd*ev[:,None]-kd[:,None]*smv+dff*cmd[6:9,None]+basis
        p,v=p+h*vm,v+h*am
        sp,sv=sp+h*smv,sv+h*sam
        ps.append(p.copy());vs.append(v.copy());jac.append(sp.copy())
    return np.asarray(ps)[indices],np.asarray(vs)[indices],np.asarray(jac)[indices]


class PreparedTrial:
    def __init__(self,folder,*,review=None,exclusions=()):
        self.folder=Path(folder);self.name=self.folder.name
        args,self.truth,self.context=load_nominal_pose_trial(folder,parameters(),device='cpu',alignment_mode='prehover_effective_alignment')
        self.time=args['output_time_s'];self.schedule=args['schedule'];self.offset=args['offset_tracking_m']
        self.masks=fitting_masks(self.time,self.truth,review=review,exclusions=exclusions)
        with np.load(self.folder/'dataset.npz') as d:
            ids=np.flatnonzero((d['controller_time_s']>=self.time[0]-1.)&(d['controller_time_s']<=self.time[0]))
            self.hover_time=d['controller_time_s'][ids]
            rotations,valid=normalized_rotations_xyzw(d['drone_quaternion_xyzw'][ids]);assert valid.all()
            self.hover=(d['drone_position_m'][ids][None],rotations[None],d['reference_fullstate'][ids][None])
        if review and review.get('precontact_start_s') is not None:
            if review['precontact_start_s']>self.hover_time[0] or review['precontact_end_s']<self.time[0]:
                raise ValueError('Reviewed contact-free interval must include the complete initialization history')
        self.p0=args['initial'].position[0].numpy();self.v0=args['initial'].velocity[0].numpy()
        zero=np.zeros(6);self.b0=self.state(zero,'cpu').compensation[0].numpy()
        self.basis=np.zeros((3,6))
        for i in range(4):
            x=zero.copy();x[i]=1
            self.basis[:,i]=self.state(x,'cpu').compensation[0].numpy()-self.b0
        self.plans={};self.gpu_schedule=None

    def state(self,gains,device,tau=.08,delay=.02):
        h=[torch.as_tensor(a,dtype=torch.float64,device=device) for a in self.hover]
        return initialize_from_hover(self.hover_time,*h,parameters(gains,tau,delay),alignment_mode='prehover_effective_alignment')[0]

    def linear(self,gains,delay,step=.005):
        key=(float(delay),float(step))
        if key not in self.plans:self.plans[key]=integration_steps(self.schedule,self.time,delay,step)
        return linear_prediction(gains,self.p0,self.v0,self.b0,self.basis,self.plans[key])

    @torch.no_grad()
    def pose(self,gains,tau,delay,*,device='cuda',step=.005):
        if device=='cuda':
            if self.gpu_schedule is None:
                s=self.schedule
                self.gpu_schedule=CommandSchedule(s.time,s.values.to('cuda'),coverage_end_s=s.coverage_end,valid=s.valid,valid_until_s=s.valid_until)
            stream=self.gpu_schedule
        else:stream=self.schedule
        return predict_pose(self.state(gains,device,tau,delay),stream,self.time,parameters(gains,tau,delay),offset_tracking_m=self.offset,maximum_step_s=step)


def translation_objective(trials,gains,delay,*,regularize=True,prior_weight=PRIOR_WEIGHT):
    residuals=[];jacobians=[]
    for trial in trials:
        p,_,jac=trial.linear(gains,delay);mask=trial.masks['fit_position']
        scale=POSITION_SCALE_M*np.sqrt(mask.sum()*3*len(trials))
        residuals.append(((p[mask]-trial.truth['position_origin_m'][mask])/scale).ravel())
        jacobians.append(jac[mask].reshape(-1,6)/scale)
    if regularize:
        residuals.append(np.sqrt(prior_weight)*(gains-PRIOR)/SCALES)
        jacobians.append(np.diag(np.sqrt(prior_weight)/SCALES))
    return np.concatenate(residuals),np.vstack(jacobians)


def fit_translation(trials,*,delays=DELAYS,starts=STARTS,prior_weight=PRIOR_WEIGHT,progress=print):
    choices=[]
    for delay in delays:
        candidates=[]
        for start in starts:
            cache={}
            def objective(x):
                if 'x' not in cache or not np.array_equal(cache['x'],x):
                    cache['value']=translation_objective(trials,x,float(delay),prior_weight=prior_weight);cache['x']=x.copy()
                return cache['value']
            result=least_squares(lambda x:objective(x)[0],start,jac=lambda x:objective(x)[1],bounds=(LOWER,UPPER),
                x_scale=SCALES,max_nfev=120,ftol=1e-9,xtol=1e-9,gtol=1e-8)
            candidates.append(dict(gains=result.x.tolist(),objective=float(2*result.cost),nfev=result.nfev,
                optimizer_success=bool(result.success),message=result.message,start=np.asarray(start).tolist()))
        winner=min(candidates,key=lambda row:row['objective'])
        choices.append(dict(delay_s=float(delay),**winner,starts=candidates))
        progress(f'Translation delay {delay:.3f} s: objective {winner["objective"]:.6f}',flush=True)
    best=min(choices,key=lambda row:row['objective'])
    gains=np.array(best['gains'])
    _,jac=translation_objective(trials,gains,best['delay_s'],regularize=False)
    singular=np.linalg.svd(jac*SCALES[None],compute_uv=False)
    return dict(best=best,delay_profile=choices,scaled_data_jacobian_singular_values=singular.tolist(),
        scaled_data_jacobian_condition=float(singular[0]/max(singular[-1],1e-15)),
        bounds_close={name:bool(min(gains[i]-LOWER[i],UPPER[i]-gains[i])<.01*(UPPER[i]-LOWER[i])) for i,name in enumerate(GAIN_NAMES)})


def fit_attitude(trials,gains,delay,*,device='cuda',progress=print):
    trace=[]
    def objective(log_tau):
        tau=float(np.exp(log_tau));loss=0.
        for trial in trials:
            out=trial.pose(gains,tau,delay,device=device)
            mask=trial.masks['fit_orientation']
            predicted=out['rotation_tracking_to_world'][0,mask]
            actual=torch.as_tensor(trial.truth['rotation_tracking_to_world'][mask],dtype=predicted.dtype,device=predicted.device)
            loss+=float(rotation_log(predicted.transpose(-1,-2)@actual).square().sum(-1).mean().cpu())/len(trials)
        trace.append(dict(tau_s=tau,mean_squared_angle_rad2=loss))
        if len(trace)%5==0:progress(f'Attitude evaluation {len(trace)}: tau {tau:.4f} s, RMS {np.degrees(np.sqrt(loss)):.2f} deg',flush=True)
        return loss
    result=minimize_scalar(objective,bounds=(np.log(.02),np.log(.30)),method='bounded',options=dict(xatol=.002,maxiter=30))
    # Evaluate boundaries explicitly rather than concealing a boundary optimum.
    objective(np.log(.02));objective(np.log(.30))
    best=min(trace,key=lambda row:row['mean_squared_angle_rad2'])
    return dict(**best,trace=trace,optimizer_success=bool(result.success),bounds_s=[.02,.30],
                near_bound=bool(best['tau_s']<.021 or best['tau_s']>.29))


def score_pose(trial,result):
    arrays={key:value[0].detach().cpu().numpy() for key,value in result.items()}
    angle=np.full(len(trial.time),np.nan);valid=trial.masks['score_orientation']
    angle[valid]=np.linalg.norm(Rotation.from_matrix(
        arrays['rotation_tracking_to_world'][valid].transpose(0,2,1)@trial.truth['rotation_tracking_to_world'][valid]).as_rotvec(),axis=1)
    scores={}
    for phase in ['csv_maneuver','post_maneuver_fullstate_hold']:
        active=trial.truth['execution_phase']==phase;scores[phase]={}
        for point in ['origin','attachment']:
            mask=active&trial.masks['score_position' if point=='origin' else 'score_attachment']
            error=np.linalg.norm(arrays[f'position_{point}_m']-trial.truth[f'position_{point}_m'],axis=1)
            scores[phase][point+'_rmse_m']=float(np.sqrt(np.mean(error[mask]**2))) if mask.any() else None
            scores[phase][point+'_max_m']=float(error[mask].max()) if mask.any() else None
            scores[phase][point+'_samples']=int(mask.sum())
        mask=active&trial.masks['score_orientation']
        scores[phase]['orientation_rmse_deg']=float(np.degrees(np.sqrt(np.mean(angle[mask]**2)))) if mask.any() else None
    return arrays,scores
