"""Bounded, translation-invariant acceleration residual for nominal pose v3."""
import hashlib
from pathlib import Path
import torch
from torch import nn


class DronePoseResidual(nn.Module):
    def __init__(self,hidden=16,acceleration_limit=.5,hover_gate=False):
        super().__init__();self.hidden=int(hidden);self.acceleration_limit=float(acceleration_limit)
        self.hover_gate=bool(hover_gate)
        if self.hidden<1 or not 0<self.acceleration_limit<=10:raise ValueError('Invalid residual specification')
        self.net=nn.Sequential(nn.Linear(15,self.hidden),nn.Tanh(),nn.Linear(self.hidden,self.hidden),nn.Tanh(),nn.Linear(self.hidden,3))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,p,v,b,command):
        x=torch.cat(((command[:,:3]-p)/.5,(command[:,3:6]-v)/2,command[:,6:9]/10,v/2,b/3),dim=-1)
        correction=self.acceleration_limit*torch.tanh(self.net(x))
        if self.hover_gate:
            # New cold-start models keep the equilibrium represented by the
            # nominal hover compensation. No arbitrary NN kick at exact rest.
            # Historical checkpoints omit this option and retain their outputs.
            motion=(v/0.5).square().sum(-1,keepdim=True)
            motion=motion+(command[:,3:6]/0.5).square().sum(-1,keepdim=True)
            motion=motion+(command[:,6:9]/2.).square().sum(-1,keepdim=True)
            correction=correction*(-torch.expm1(-motion))
        return correction
    def specification(self):
        result=dict(hidden=self.hidden,acceleration_limit=self.acceleration_limit)
        if self.hover_gate:result['hover_gate']=True
        return result


def save_residual(path,network):
    torch.save(dict(schema='drone_pose_residual_v1',specification=network.specification(),
        state_dict={k:v.detach().cpu() for k,v in network.state_dict().items()}),path)


def load_residual(path,sha256,device='cuda'):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=sha256:raise ValueError('Drone residual checkpoint hash mismatch')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['schema']!='drone_pose_residual_v1':raise ValueError('Wrong drone residual family')
    model=DronePoseResidual(**payload['specification']).to(device=device,dtype=torch.float64)
    model.load_state_dict(payload['state_dict']);model.eval();model.requires_grad_(False)
    return model
