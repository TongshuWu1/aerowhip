"""Nominal loaded-drone FullState response at the OptiTrack tracking origin.

Standalone, differentiable, batched Torch model. No default fitted parameters,
NN, actuator identification, cable reaction injection, exporter or flight sender.
The conventional command attitude frame C is NOT implicitly a firmware frame.
R_WT = R_WC R_CT; every state carries an explicit R_CT and its provenance.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch


def skew(v):
    x,y,z=v.unbind(-1);o=torch.zeros_like(x)
    return torch.stack((o,-z,y,z,o,-x,-y,x,o),-1).reshape(*v.shape[:-1],3,3)


def rotation_exp(vector):
    angle=torch.linalg.vector_norm(vector,dim=-1)
    k=skew(vector)
    eye=torch.eye(3,dtype=vector.dtype,device=vector.device)
    a=torch.sinc(angle/math.pi)[...,None,None]
    b=(.5*torch.sinc(angle/(2*math.pi)).square())[...,None,None]
    return eye+a*k+b*(k@k)


def rotation_log(rotation):
    """Principal SO(3) log; reject the ambiguous near-180-degree branch."""
    v=torch.stack((rotation[...,2,1]-rotation[...,1,2],
                   rotation[...,0,2]-rotation[...,2,0],
                   rotation[...,1,0]-rotation[...,0,1]),-1)*.5
    sine=torch.linalg.vector_norm(v,dim=-1)
    cosine=(rotation.diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5
    if bool((cosine.detach() < -.9999).any()):
        raise ValueError('Attitude error is too close to 180 degrees for this local response model')
    angle=torch.atan2(sine,cosine)
    factor=torch.where((sine<1e-6)&(cosine>0),1+sine.square()/6,
                       angle/sine.clamp_min(1e-12))
    return v*factor[...,None]


def _check_rotation(rotation):
    if rotation.shape[-2:]!=(3,3) or not bool(torch.isfinite(rotation).all()):
        raise ValueError('Expected finite 3x3 rotation matrices')
    eye=torch.eye(3,dtype=rotation.dtype,device=rotation.device)
    if (not torch.allclose(rotation.transpose(-1,-2)@rotation,eye.expand_as(rotation),atol=2e-5,rtol=2e-5)
            or not torch.allclose(torch.linalg.det(rotation),torch.ones_like(rotation[...,0,0]),atol=2e-5,rtol=2e-5)):
        raise ValueError('Orientation/alignment must be proper rotations')


@dataclass(frozen=True)
class PoseResponseParameters:
    kp_xy: object
    kp_z: object
    kd_xy: object
    kd_z: object
    feedforward_xy: object
    feedforward_z: object
    attitude_time_constant_s: object
    delay_s: float
    gravity_m_s2: float = 9.80665
    attitude_drive_model: str = 'kinematic_v1'
    attitude_acceleration_scale_xy: object = 1.
    attitude_acceleration_scale_z: object = 1.

    def validate(self):
        for key in ('kp_xy','kp_z','kd_xy','kd_z','feedforward_xy','feedforward_z','attitude_time_constant_s'):
            value=torch.as_tensor(getattr(self,key),dtype=torch.float64)
            if value.numel()!=1 or not bool(torch.isfinite(value).all()) or float(value.detach())<0:
                raise ValueError(f'{key} must be a finite nonnegative scalar')
        if float(torch.as_tensor(self.attitude_time_constant_s,dtype=torch.float64).detach())<=0:
            raise ValueError('Attitude time constant must be positive')
        if not math.isfinite(self.delay_s) or self.delay_s<0:
            raise ValueError('Effective command delay must be finite and nonnegative')
        if not math.isfinite(self.gravity_m_s2) or self.gravity_m_s2<=0:
            raise ValueError('Gravity magnitude must be finite and positive')
        if self.attitude_drive_model not in ('kinematic_v1','normalized_response_v2','independent_scale_v3'):
            raise ValueError('Unknown attitude drive model; choose an explicit supported convention')
        if self.attitude_drive_model=='normalized_response_v2':
            for key in ('feedforward_xy','feedforward_z'):
                if float(torch.as_tensor(getattr(self,key)).detach())<=1e-6:
                    raise ValueError('Normalized attitude drive requires strictly positive response gains above 1e-6')
        for key in ('attitude_acceleration_scale_xy','attitude_acceleration_scale_z'):
            value=torch.as_tensor(getattr(self,key),dtype=torch.float64)
            if value.numel()!=1 or not bool(torch.isfinite(value)) or float(value.detach())<=0:
                raise ValueError('Attitude acceleration scales must be finite positive scalars')

    def tensors(self,reference):
        def xyz(horizontal,vertical):
            h=torch.as_tensor(horizontal,dtype=reference.dtype,device=reference.device).reshape(())
            z=torch.as_tensor(vertical,dtype=reference.dtype,device=reference.device).reshape(())
            return torch.stack((h,h,z))
        return (xyz(self.kp_xy,self.kp_z),xyz(self.kd_xy,self.kd_z),
                xyz(self.feedforward_xy,self.feedforward_z),
                torch.as_tensor(self.attitude_time_constant_s,dtype=reference.dtype,device=reference.device).reshape(()))


@dataclass(frozen=True)
class PoseResponseState:
    position: torch.Tensor
    velocity: torch.Tensor
    rotation: torch.Tensor
    omega_tracking: torch.Tensor
    compensation: torch.Tensor
    rotation_command_from_tracking: torch.Tensor
    alignment_provenance: str
    time_s: float

    def validate(self):
        if not math.isfinite(self.time_s):raise ValueError('Initial state requires its measured timestamp')
        if self.position.ndim!=2 or self.position.shape[-1]!=3:
            raise ValueError('Expected Bx3 tracked-origin state')
        for value in (self.velocity,self.omega_tracking,self.compensation):
            if value.shape!=self.position.shape:
                raise ValueError('All vector states must have the same Bx3 shape')
        for value in (self.rotation,self.rotation_command_from_tracking):
            if value.shape!=self.position.shape[:1]+(3,3):
                raise ValueError('Expected Bx3x3 orientation and explicit alignment')
            _check_rotation(value)
        for value in (self.position,self.velocity,self.omega_tracking,self.compensation,self.rotation,self.rotation_command_from_tracking):
            if value.device!=self.position.device or value.dtype!=self.position.dtype or not bool(torch.isfinite(value).all()):
                raise ValueError('State must be finite, with one dtype/device')
        if self.position.dtype not in (torch.float32,torch.float64):
            raise ValueError('Use float32 or float64 states')
        if self.alignment_provenance not in ('explicit_calibration','prehover_effective_alignment'):
            raise ValueError('Alignment provenance must be explicit; firmware axes are not assumed')


def command_attitude(acceleration,yaw,gravity_m_s2):
    """Conventional desired thrust-direction/heading frame C, in world axes.

    Acceleration remains kinematic. Gravity appears ONLY in this orientation
    construction, not as another acceleration in the effective O dynamics.
    """
    specific=acceleration+acceleration.new_tensor([0.,0.,gravity_m_s2])
    norm=torch.linalg.vector_norm(specific,dim=-1,keepdim=True)
    if bool((norm.detach()<1e-7).any()):
        raise ValueError('Desired specific-force direction is undefined')
    z=specific/norm
    heading=torch.stack((torch.cos(yaw),torch.sin(yaw),torch.zeros_like(yaw)),-1)
    y=torch.linalg.cross(z,heading,dim=-1)
    ynorm=torch.linalg.vector_norm(y,dim=-1,keepdim=True)
    if bool((ynorm.detach()<1e-7).any()):
        raise ValueError('Desired heading and specific-force direction are singular')
    y=y/ynorm;x=torch.linalg.cross(y,z,dim=-1)
    return torch.stack((x,y,z),-1)


def attitude_drive_acceleration(acceleration,parameters):
    """Separate inferred control acceleration u from effective response a.

    In v2, a = Ga*u and u has unit acceleration feedforward. Thus
    u = (Kp/Ga)*ep + (Kd/Ga)*ev + ad + b/Ga. This is an explicit
    effective-model factorization, not measured firmware thrust or COM a.
    v1 is retained solely for named historical comparisons.
    """
    if parameters.attitude_drive_model=='kinematic_v1':return acceleration
    if parameters.attitude_drive_model=='independent_scale_v3':
        xy=torch.as_tensor(parameters.attitude_acceleration_scale_xy,dtype=acceleration.dtype,device=acceleration.device).reshape(())
        z=torch.as_tensor(parameters.attitude_acceleration_scale_z,dtype=acceleration.dtype,device=acceleration.device).reshape(())
        return acceleration*torch.stack((xy,xy,z))
    if parameters.attitude_drive_model!='normalized_response_v2':
        raise ValueError('Unknown attitude drive model')
    _,_,gain,_=parameters.tensors(acceleration)
    if bool((gain.detach()<=1e-6).any()):
        raise ValueError('Normalized attitude drive requires strictly positive response gains above 1e-6')
    return acceleration/gain


def _endpoint_velocity(time,positions,*,end,samples):
    tt=time[-samples:] if end else time[:samples]
    pp=positions[:,-samples:] if end else positions[:,:samples]
    duration=tt[-1]-tt[0];origin=tt[-1] if end else tt[0]
    x=(tt-origin)/duration
    design=torch.stack((torch.ones_like(x),x,x.square()),-1)
    weights=torch.linalg.pinv(design)[1]/duration
    return torch.einsum('t,btc->bc',weights,pp),weights


def initialize_from_hover(time_s,positions,rotations,commands,parameters,*,
                          alignment_mode,rotation_command_from_tracking=None,derivative_samples=11):
    """Estimate an effective frozen compensation from already observed hover.

    All history must precede the prediction start (which is the last timestamp).
    Commands must be constant position/yaw holds with zero V/A and yaw rate.
    This estimates initial state/compensation, not firmware integral gains.
    """
    parameters.validate()
    t=torch.as_tensor(time_s,dtype=positions.dtype,device=positions.device)
    if positions.ndim!=3 or positions.shape[-1]!=3 or len(t)<derivative_samples or derivative_samples<3:
        raise ValueError('Need BxHx3 pre-hover positions and sufficient derivative history')
    if rotations.shape!=positions.shape[:2]+(3,3) or commands.shape!=positions.shape[:2]+(11,) or len(t)!=positions.shape[1]:
        raise ValueError('Pre-hover pose/command history shapes differ')
    if not bool(torch.isfinite(t).all()) or not bool((t[1:]>t[:-1]).all()):
        raise ValueError('Pre-hover timestamps must be finite and increasing')
    gaps=t[1:]-t[:-1]
    if bool((gaps>1.5*gaps.median()).any()):
        raise ValueError('Pre-hover history has a timestamp gap')
    if not bool(torch.isfinite(positions).all()) or not bool(torch.isfinite(commands).all()):
        raise ValueError('Missing pre-hover observations/commands cannot initialize memory')
    _check_rotation(rotations)
    if (not torch.allclose(commands,commands[:,:1].expand_as(commands),atol=1e-10,rtol=0)
            or bool((commands[...,3:9].abs()>1e-10).any()) or bool((commands[...,10].abs()>1e-10).any())):
        raise ValueError('Hover initializer requires observed constant position/yaw and zero V/A/rate commands')
    velocity,weights=_endpoint_velocity(t,positions,end=True,samples=derivative_samples)
    first_velocity,_=_endpoint_velocity(t,positions,end=False,samples=derivative_samples)
    duration=t[-1]-t[0]
    mean_p=(((positions[:,:-1]+positions[:,1:])*.5)*gaps[None,:,None]).sum(1)/duration
    mean_v=(positions[:,-1]-positions[:,0])/duration
    mean_a=(velocity-first_velocity)/duration
    kp,kd,_,_=parameters.tensors(positions)
    compensation=mean_a-kp*(commands[:,0,:3]-mean_p)+kd*mean_v
    relative=rotations[:,-1,None].transpose(-1,-2)@rotations[:,-derivative_samples:]
    omega=torch.einsum('t,btc->bc',weights,rotation_log(relative))
    if alignment_mode=='prehover_effective_alignment':
        if rotation_command_from_tracking is not None:
            raise ValueError('Do not supply a calibration and request effective alignment together')
        # Closest proper mean rotation; observations are fixed, not trainable.
        u,_,vh=torch.linalg.svd(rotations.mean(1).detach())
        fix=torch.ones_like(positions[:,0]);fix[:,-1]=torch.linalg.det(u@vh)
        mean_rotation=u@torch.diag_embed(fix)@vh
        command_rotation=command_attitude(attitude_drive_acceleration(mean_a,parameters),commands[:,0,9],parameters.gravity_m_s2)
        alignment=command_rotation.transpose(-1,-2)@mean_rotation
    elif alignment_mode=='explicit_calibration':
        if rotation_command_from_tracking is None:
            raise ValueError('Explicit alignment calibration is required')
        alignment=rotation_command_from_tracking
    else:
        raise ValueError('Choose explicit_calibration or prehover_effective_alignment')
    state=PoseResponseState(positions[:,-1],velocity,rotations[:,-1],omega,compensation,alignment,alignment_mode,float(t[-1].detach()))
    state.validate()
    return state,dict(initial_time_s=float(t[-1].detach()),history_duration_s=float(duration.detach()),
        compensation_estimated_from='Mean pre-hover acceleration minus nominal PD response; frozen during short rollout',
        controller_integral_observed=False,alignment_provenance=alignment_mode,
        hardware_mounting_verified_by_this_code=False)


class CommandSchedule:
    """Causal zero-order hold with explicit valid intervals and support end.

    For sampled logs, valid_until_s also bounds the age of the cached command.
    Invalid rows may contain NaN and are never replaced by hover/zero commands.
    Times are a shared host-side schedule; values are BxCx11 Torch tensors.
    """
    def __init__(self,time_s,values,*,coverage_end_s,valid=None,valid_until_s=None):
        self.time=np.asarray(time_s,dtype=float)
        if self.time.ndim!=1 or len(self.time)<1 or not np.isfinite(self.time).all() or np.any(np.diff(self.time)<=0):
            raise ValueError('Command timestamps must be finite and strictly increasing')
        if values.ndim!=3 or values.shape[1:]!=(len(self.time),11):
            raise ValueError('Expected BxCx11 P/V/A/yaw/yaw_rate commands')
        self.values=values
        self.coverage_end=float(coverage_end_s)
        if not math.isfinite(self.coverage_end) or self.coverage_end<=self.time[-1]:
            raise ValueError('Command support must end strictly after its last timestamp')
        self.valid=np.ones((len(values),len(self.time)),bool) if valid is None else np.broadcast_to(np.asarray(valid,dtype=bool),(len(values),len(self.time)))
        ends=np.r_[self.time[1:],self.coverage_end]
        self.valid_until=np.broadcast_to(ends,(len(values),len(self.time))).copy()
        if valid_until_s is not None:self.valid_until=np.minimum(self.valid_until,np.broadcast_to(np.asarray(valid_until_s,dtype=float),self.valid_until.shape))
        finite=torch.isfinite(values).all(-1).detach().cpu().numpy()
        if np.any(self.valid&(~finite|~np.isfinite(self.valid_until)|(self.valid_until<=self.time[None]))):
            raise ValueError('Valid command rows need finite values and nonempty support')
        if bool((values[...,10][torch.as_tensor(self.valid.copy(),device=values.device)].abs()>1e-10).any()):
            raise ValueError('This nominal model supports zero body-rate feedforward; nonzero yaw_rate needs an explicit interface model')

    def sample(self,time_s):
        time_s=float(time_s)
        if not math.isfinite(time_s):
            raise ValueError('Command query time must be finite')
        # Round-off tolerance applies only to coincident event boundaries.
        i=int(np.searchsorted(self.time,float(time_s)+1e-12,side='right')-1)
        if i<0 or time_s>=self.coverage_end or not self.valid[:,i].all() or np.any(time_s>=self.valid_until[:,i]):
            raise ValueError(f'No fresh observed command at time {time_s:.9f}; cannot infer missing input')
        return self.values[:,i]


def derivatives(state,command,parameters):
    kp,kd,ff,tau=parameters.tensors(state.position)
    acceleration=kp*(command[:,:3]-state.position)+kd*(command[:,3:6]-state.velocity)+ff*command[:,6:9]+state.compensation
    desired=command_attitude(attitude_drive_acceleration(acceleration,parameters),command[:,9],parameters.gravity_m_s2)@state.rotation_command_from_tracking
    error=rotation_log(state.rotation.transpose(-1,-2)@desired)
    alpha=error/tau.square()-2*state.omega_tracking/tau
    return acceleration,alpha


def midpoint_step(state,command,dt_s,parameters):
    """Second-order Lie midpoint; p/v at O, R/omega in tracked coordinates."""
    h=float(dt_s);a,alpha=derivatives(state,command,parameters)
    midpoint=PoseResponseState(state.position+.5*h*state.velocity,state.velocity+.5*h*a,
        state.rotation@rotation_exp(.5*h*state.omega_tracking),state.omega_tracking+.5*h*alpha,
        state.compensation,state.rotation_command_from_tracking,state.alignment_provenance,state.time_s+.5*h)
    am,alpham=derivatives(midpoint,command,parameters)
    return PoseResponseState(state.position+h*midpoint.velocity,state.velocity+h*am,
        state.rotation@rotation_exp(h*midpoint.omega_tracking),state.omega_tracking+h*alpham,
        state.compensation,state.rotation_command_from_tracking,state.alignment_provenance,state.time_s+h)


def attachment_pva(state,acceleration,alpha,offset_tracking_m):
    r=torch.as_tensor(offset_tracking_m,dtype=state.position.dtype,device=state.position.device)
    if r.shape==(3,):r=r.expand_as(state.position)
    if r.shape!=state.position.shape or not bool(torch.isfinite(r).all()):
        raise ValueError('Supply one finite tracked-frame attachment offset per batch or one shared XYZ offset')
    rotate=lambda v:torch.einsum('bij,bj->bi',state.rotation,v)
    cross=lambda a,b:torch.linalg.cross(a,b,dim=-1)
    return (state.position+rotate(r),state.velocity+rotate(cross(state.omega_tracking,r)),
            acceleration+rotate(cross(alpha,r)+cross(state.omega_tracking,cross(state.omega_tracking,r))))


def predict_pose(initial,schedule,output_time_s,parameters,*,offset_tracking_m,maximum_step_s=.005):
    """Recursive prediction; no measured future pose/cable arrays are accepted.

    Integration lands on every delayed command event and requested output time.
    P/V/A outputs are at O and A; acceleration uses the command at the output
    timestamp (right limit), so it can jump at a command event.
    """
    initial.validate();parameters.validate()
    times=np.asarray(output_time_s,dtype=float)
    if times.ndim!=1 or len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Output timestamps must be finite and strictly increasing; first is initial state time')
    if abs(initial.time_s-times[0])>1e-10:
        raise ValueError('Prediction must start at the timestamp of the measured initial state')
    if not math.isfinite(maximum_step_s) or maximum_step_s<=0:
        raise ValueError('Maximum integration step must be positive and finite')
    if maximum_step_s>float(torch.as_tensor(parameters.attitude_time_constant_s,dtype=torch.float64).detach())/4:
        raise ValueError('Reduce integration step below one quarter of attitude time constant')
    kp,kd,_,_=parameters.tensors(initial.position)
    # Resolve translation too: a fitted Kd can be much faster than attitude.
    # This is a conservative resolution check, not proof of error/stability
    # bounds (in particular for undamped motion). Verify step convergence.
    fastest_translation=float((kd+torch.sqrt(kp)).max().detach())
    if maximum_step_s*fastest_translation>.25:
        raise ValueError('Reduce integration step to resolve translational gains: dt * max(Kd + sqrt(Kp)) <= 0.25')
    if schedule.values.shape[0]!=initial.position.shape[0] or schedule.values.dtype!=initial.position.dtype or schedule.values.device!=initial.position.device:
        raise ValueError('Commands and initial state must share batch, dtype and device')
    # Preflight coverage prevents stepping across short missing/stale intervals.
    events=np.unique(np.r_[times,schedule.time+parameters.delay_s,schedule.valid_until.ravel()+parameters.delay_s])
    events=events[(events>=times[0])&(events<=times[-1])]
    for left,right in zip(events[:-1],events[1:]):
        if right-left>1e-12:schedule.sample((left+right)*.5-parameters.delay_s)
    for t in times:schedule.sample(t-parameters.delay_s)
    outputs={key:[] for key in ('position_origin_m','velocity_origin_m_s','acceleration_origin_m_s2',
        'rotation_tracking_to_world','omega_tracking_rad_s','alpha_tracking_rad_s2',
        'position_attachment_m','velocity_attachment_m_s','acceleration_attachment_m_s2')}
    def record(state,time):
        a,alpha=derivatives(state,schedule.sample(time-parameters.delay_s),parameters)
        pa,va,aa=attachment_pva(state,a,alpha,offset_tracking_m)
        for key,value in zip(outputs,(state.position,state.velocity,a,state.rotation,state.omega_tracking,alpha,pa,va,aa)):
            if not bool(torch.isfinite(value).all()):raise ValueError('Nonfinite nominal pose prediction')
            outputs[key].append(value)
    state=initial;record(state,times[0])
    for start,end in zip(times[:-1],times[1:]):
        local=np.r_[start,events[(events>start+1e-12)&(events<end-1e-12)],end]
        for a,b in zip(local[:-1],local[1:]):
            count=max(1,int(np.ceil((b-a)/maximum_step_s)))
            h=(b-a)/count
            command=schedule.sample((a+b)*.5-parameters.delay_s)
            for _ in range(count):state=midpoint_step(state,command,h,parameters)
        record(state,end)
    return {key:torch.stack(value,1) for key,value in outputs.items()}
