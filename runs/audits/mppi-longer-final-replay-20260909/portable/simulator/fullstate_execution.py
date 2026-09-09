"""Effective loaded-drone attachment response for offline full-state prediction.

Commands in this interface use desired attachment P/V/A. To obtain vehicle
cmdFullState positions, subtract the attachment offset rotated by the measured
initial orientation. The fitted residual includes subsequent offset motion.
This is an empirical one-way response model: do not add cable reaction again.
"""
from pathlib import Path
import hashlib
import torch
from .drone_tracking import load_tracking_model, predict_trajectory
from .cable import DderState, START_PINNED_FREE_END


class FullStateAttachmentModel:
    def __init__(self, checkpoint, sha256, *, device='cpu'):
        path=Path(checkpoint)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=sha256:
            raise ValueError('Attachment tracking checkpoint hash mismatch')
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload.get('reference_point')!='cable_attachment':
            raise ValueError('Combined cable prediction requires an attachment tracking model')
        if payload.get('command_position_transform')!='logged_cf7_command_plus_initial_world_attachment_offset':
            raise ValueError('Unrecognized full-state/attachment command transform')
        self.network,self.gains,self.delay_s,self.history_s=load_tracking_model(path,device=device)
        self.offset_body_m=payload['attachment_offset_body_m']

    @torch.no_grad()
    def predict(self,position,velocity,commands,past_commands,dt,*,residual=True):
        """Commands are already causal-held at t-delay; history at t-delay-50 ms."""
        return predict_trajectory(position,velocity,commands,past_commands,dt,self.gains,
                                  self.network if residual else None)


def attachment_to_vehicle_commands(commands, initial_world_offset):
    """Constant initial offset changes position only, preserving PVA derivatives."""
    output=commands.clone()
    output[...,:3]=commands[...,:3]-initial_world_offset[...,None,:]
    return output


def sample_kinematic_reference(positions, velocities, dt_s, query_times):
    """Batched cubic Hermite PVA matching deployment.fullstate.sample_fullstate.

Positions are B x time x 3; queries B x samples. The final boundary uses its
left derivative, so recovery acceleration cannot contaminate the whip cutoff.
"""
    if positions.shape!=velocities.shape or positions.ndim!=3 or positions.shape[-1]!=3:
        raise ValueError('Expected matching BxTx3 position and velocity')
    if query_times.ndim!=2 or query_times.shape[0]!=positions.shape[0] or dt_s<=0:
        raise ValueError('Expected BxS queries and positive dt')
    last=(positions.shape[1]-1)*dt_s
    if not bool(torch.isfinite(query_times).all()) or bool(((query_times<0)|(query_times>last+1e-10)).any()):
        raise ValueError('Query outside the planned trajectory')
    knots=torch.arange(positions.shape[1],device=positions.device,dtype=query_times.dtype)*dt_s
    i=(torch.searchsorted(knots,query_times.contiguous(),right=True)-1).clamp(0,positions.shape[1]-2)
    batch=torch.arange(len(positions),device=positions.device)[:,None]
    p0,p1=positions[batch,i],positions[batch,i+1]
    v0,v1=velocities[batch,i],velocities[batch,i+1]
    x=(query_times-knots[i])[...,None]
    h=(knots[i+1]-knots[i])[...,None]
    c2=3*(p1-p0)/h**2-(2*v0+v1)/h
    c3=-2*(p1-p0)/h**3+(v0+v1)/h**2
    return torch.cat((p0+v0*x+c2*x*x+c3*x**3,v0+2*c2*x+3*c3*x*x,2*c2+6*c3*x),dim=-1)


class CudaGraphCableBoundary:
    """Fixed-batch cable execution driven by an attachment path, with no drone force."""
    def __init__(self,model,state,dt,constants):
        if not state.positions_m.is_cuda:raise ValueError('CUDA boundary graph requires CUDA state')
        self.model=model;self.q=state.positions_m.detach().clone();self.v=state.velocities_m_s.detach().clone()
        self.boundary=self.q[:,:1].clone();self.dt=self.q.new_full((len(self.q),),dt);self.constants=constants
        stream=torch.cuda.Stream(device=self.q.device)
        stream.wait_stream(torch.cuda.current_stream(self.q.device))
        with torch.cuda.stream(stream):
            for _ in range(2):self._transition()
        torch.cuda.current_stream(self.q.device).wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.output=self._transition()

    def _transition(self):
        return self.model.step_runtime(DderState(self.q,self.v),self.boundary,self.dt,self.constants,
            pinned_endpoints=START_PINNED_FREE_END,create_graph=False,iterative_damping=True,
            damping_backend='pcg32_experimental',analytic_bending=True)

    @torch.no_grad()
    def __call__(self,state,boundary):
        if state.positions_m.shape!=self.q.shape:raise ValueError('Boundary graph batch shape changed')
        self.q.copy_(state.positions_m);self.v.copy_(state.velocities_m_s);self.boundary.copy_(boundary[:,None])
        self.graph.replay()
        return DderState(self.output.positions_m.clone(),self.output.velocities_m_s.clone())
