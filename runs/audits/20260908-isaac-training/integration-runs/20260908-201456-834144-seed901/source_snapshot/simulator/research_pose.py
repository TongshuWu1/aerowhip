"""GPU execution of the fitted pose equations with per-trajectory domain flags.

The checked fitting implementation remains the reference. Invalid rows are
reported to the caller; a singular row never supplies a replacement valid pose.
"""
from pathlib import Path
from dataclasses import replace
import math
import numpy as np
import torch
from .drone_pose_response import PoseResponseParameters,PoseResponseState,rotation_exp,nominal_acceleration,attitude_drive_acceleration
from .drone_pose_residual import load_residual
from .workflow import read_json
from experimental_data.io import sha256_file


def tensor_derivatives(state,command,parameters,residual):
    a=nominal_acceleration(state.position,state.velocity,state.compensation,command,parameters)
    gravity=torch.stack((torch.zeros_like(a[:,0]),torch.zeros_like(a[:,0]),torch.full_like(a[:,0],parameters.gravity_m_s2)),-1)
    u=attitude_drive_acceleration(a,parameters)+gravity
    norm=u.norm(dim=-1,keepdim=True);z=u/norm.clamp_min(1e-12)
    yaw=command[:,9];heading=torch.stack((yaw.cos(),yaw.sin(),torch.zeros_like(yaw)),-1)
    y=torch.linalg.cross(z,heading,dim=-1);yn=y.norm(dim=-1,keepdim=True);y=y/yn.clamp_min(1e-12)
    desired=torch.stack((torch.linalg.cross(y,z,dim=-1),y,z),-1)@state.rotation_command_from_tracking
    error=state.rotation.transpose(-1,-2)@desired
    vee=torch.stack((error[:,2,1]-error[:,1,2],error[:,0,2]-error[:,2,0],error[:,1,0]-error[:,0,1]),-1)*.5
    sine=vee.norm(dim=-1);cosine=(error.diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5
    angle=torch.atan2(sine,cosine)
    factor=torch.where((sine<1e-6)&(cosine>0),1+sine.square()/6,angle/sine.clamp_min(1e-12))
    tau=parameters.tensors(a)[3];alpha=vee*factor[:,None]/tau.square()-2*state.omega_tracking/tau
    if residual is not None:a=a+residual(state.position,state.velocity,state.compensation,command)
    valid=(norm[:,0]>=1e-7)&(yn[:,0]>=1e-7)&(cosine>=-.9999)&torch.isfinite(a).all(-1)&torch.isfinite(alpha).all(-1)
    return a,alpha,valid


def tensor_midpoint(state,command,h,parameters,residual):
    a,alpha,ok=tensor_derivatives(state,command,parameters,residual)
    mid=PoseResponseState(state.position+.5*h*state.velocity,state.velocity+.5*h*a,
        state.rotation@rotation_exp(.5*h*state.omega_tracking),state.omega_tracking+.5*h*alpha,
        state.compensation,state.rotation_command_from_tracking,state.alignment_provenance,0.)
    am,alpham,valid=tensor_derivatives(mid,command,parameters,residual)
    result=PoseResponseState(state.position+h*mid.velocity,state.velocity+h*am,
        state.rotation@rotation_exp(h*mid.omega_tracking),state.omega_tracking+h*alpham,
        state.compensation,state.rotation_command_from_tracking,state.alignment_provenance,0.)
    return result,ok&valid


class PoseStepper:
    def __init__(self,initial,parameters,residual,*,graph=True):
        self.parameters=replace(parameters,**{name:torch.as_tensor(getattr(parameters,name),device=initial.position.device,dtype=initial.position.dtype)
            for name in ['kp_xy','kp_z','kd_xy','kd_z','feedforward_xy','feedforward_z','attitude_time_constant_s',
                'attitude_acceleration_scale_xy','attitude_acceleration_scale_z']});self.residual=residual
        self.state=PoseResponseState(*(getattr(initial,n).clone() for n in ['position','velocity','rotation','omega_tracking','compensation','rotation_command_from_tracking']),initial.alignment_provenance,0.)
        self.command=initial.position.new_zeros(len(initial.position),11);self.command[:,:3]=initial.position
        self.h=initial.position.new_tensor(1/300)
        self.graph=None
        if graph and initial.position.is_cuda:
            stream=torch.cuda.Stream(device=initial.position.device);stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):self._step()
            torch.cuda.current_stream().wait_stream(stream)
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):self.output=self._step()

    def _step(self):return tensor_midpoint(self.state,self.command,self.h,self.parameters,self.residual)

    def __call__(self,state,command,h):
        for name in ['position','velocity','rotation','omega_tracking','compensation','rotation_command_from_tracking']:
            getattr(self.state,name).copy_(getattr(state,name))
        self.command.copy_(command);self.h.fill_(h)
        if self.graph is not None:self.graph.replay();out,ok=self.output
        else:out,ok=self._step()
        return PoseResponseState(*(getattr(out,n).clone() for n in ['position','velocity','rotation','omega_tracking']),
            state.compensation,state.rotation_command_from_tracking,state.alignment_provenance,0.),ok.clone()


class ResearchPoseModel:
    def __init__(self,path,digest,device='cuda'):
        path=Path(path)
        if sha256_file(path)!=digest:raise ValueError('Drone component hash mismatch')
        payload=read_json(path);self.parameters=PoseResponseParameters(**payload['nominal']['parameters']);self.parameters.validate()
        spec=payload['residual'];self.residual=load_residual(path.parent/spec['checkpoint'],spec['sha256'],device)

    @torch.no_grad()
    def predict(self,initial,packets,packet_times,output_times,offset,*,graph=True,hover_command=None):
        """Fixed packets, exact delayed events, and a causal supplied initial state."""
        initial.validate();p=self.parameters
        packet_times=np.asarray(packet_times,dtype=float);times=np.asarray(output_times,dtype=float)
        if abs(times[0]-initial.time_s)>1e-9 or len(times)<2 or np.any(np.diff(times)<=0):raise ValueError('Invalid pose output times')
        if packets.shape!=(len(initial.position),len(packet_times),11):raise ValueError('Packet dimensions differ')
        if len(packet_times)<1 or np.any(np.diff(packet_times)<=0) or not torch.isfinite(packets).all():raise ValueError('Invalid FullState packets')
        if torch.any(packets[:,:,10]!=0):raise ValueError('Nonzero body-rate feedforward is not supported')
        maximum=min(.005,float(p.attitude_time_constant_s)/4,.25/float((p.tensors(initial.position)[1]+p.tensors(initial.position)[0].sqrt()).max()))
        if hover_command is None:
            hover_command=packets[:,0].clone();hover_command[:,3:9]=0.
        steps=PoseStepper(initial,p,self.residual,graph=graph)
        state=initial;valid=torch.ones(len(packets),device=packets.device,dtype=torch.bool)
        r=initial.position.new_tensor(offset);roots=[];positions=[];velocities=[];rotations=[];omegas=[];valids=[]
        def record():
            roots.append(state.position+torch.einsum('bij,j->bi',state.rotation,r));positions.append(state.position)
            velocities.append(state.velocity);rotations.append(state.rotation);omegas.append(state.omega_tracking);valids.append(valid.clone())
        events=packet_times+p.delay_s;record()
        for left,right in zip(times[:-1],times[1:]):
            boundaries=np.r_[left,events[(events>left+1e-12)&(events<right-1e-12)],right]
            for a,b in zip(boundaries[:-1],boundaries[1:]):
                index=np.searchsorted(packet_times,(a+b)/2-p.delay_s+1e-12,side='right')-1
                command=hover_command if index<0 else packets[:,index]
                count=max(1,int(math.ceil((b-a)/maximum-1e-10)));h=(b-a)/count
                for _ in range(count):
                    proposed,ok=steps(state,command,h);valid&=ok
                    state=PoseResponseState(*(torch.where(valid.reshape((-1,)+(1,)*(getattr(state,n).ndim-1)),getattr(proposed,n),getattr(state,n))
                        for n in ['position','velocity','rotation','omega_tracking']),state.compensation,state.rotation_command_from_tracking,state.alignment_provenance,0.)
            record()
        return dict(position_origin_m=torch.stack(positions,1),velocity_origin_m_s=torch.stack(velocities,1),
            rotation_tracking_to_world=torch.stack(rotations,1),omega_tracking_rad_s=torch.stack(omegas,1),
            position_attachment_m=torch.stack(roots,1),valid=torch.stack(valids,1))


def settled_initial(root,velocity,offset):
    """Ideal settled-hover initial condition for offline scenario simulation."""
    eye=torch.eye(3,device=root.device,dtype=root.dtype)[None].expand(len(root),-1,-1).clone()
    zero=torch.zeros_like(root)
    return PoseResponseState(root-root.new_tensor(offset),velocity,eye,zero,zero,eye,
        'prehover_effective_alignment',0.)
