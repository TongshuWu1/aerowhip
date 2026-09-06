"""Check usable identification gradients on actual recorded starting states."""
import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from torch import nn

from simulator.cable.residual import MotionResidual
from .constrained_identification import load_takes, read
from .differentiable_fit import rollout, subset
from .io import atomic_json, sha256_file


class ProbeResidual(nn.Module):
    def __init__(self, nodes, reference):
        super().__init__()
        self.network=MotionResidual(nodes,hidden=32,acceleration_limit=.5).to(reference)
        with torch.no_grad():
            self.network.net[-1].weight.normal_(std=.005)
            self.network.net[-1].bias.fill_(.005)
        self.network.requires_grad_(False)
        self.gain=reference.new_zeros(())

    def forward(self,q,v):
        return self.gain.exp()*self.network(q,v)


def run(output,source,benchmark,horizons,names):
    torch.set_num_threads(1);torch.manual_seed(1729)
    payload=read(output/'geometry_model.json')
    cable,model,takes,_=load_takes(source,benchmark,payload,device='cuda' if torch.cuda.is_available() else 'cpu')
    rows=[]
    for name in names:
        original=subset(next(t for t in takes if t.take_id==name),[0])
        for horizon in horizons:
            frames=round(horizon/.01)+1
            t=replace(original,root_positions_m=original.root_positions_m[:,:frames],
                      measured_marker_positions_m=original.measured_marker_positions_m[:,:frames])
            reference=t.initial_positions_m
            x=reference.new_tensor([np.log(.001),np.log(.0001),np.log(.3),0.],requires_grad=True)
            probe=ProbeResidual(cable.node_count,reference);model.motion_residual=probe
            def objective(z,gradients=False):
                probe.gain=z[3]
                y=rollout(t,model,cable,z[:3].exp(),gradients=gradients)
                sq=(y[:,1:]-t.measured_marker_positions_m[:,1:]).square().sum(-1)
                return (.002*(torch.sqrt(1+sq/.002**2)-1)).mean()
            value=objective(x,True);g=torch.autograd.grad(value,x)[0].detach().cpu().numpy()
            finite={}
            for eps in (1e-4,1e-5):
                fd=[]
                with torch.no_grad():
                    for j in range(4):
                        step=torch.zeros_like(x);step[j]=eps
                        fd.append(float((objective(x+step)-objective(x-step))/(2*eps)))
                finite[str(eps)]=fd
            passed=all(np.all(np.abs(np.array(fd)-g)<=1e-7+.05*np.maximum(np.abs(fd),np.abs(g))) for fd in finite.values())
            rows.append(dict(take=name,start_frame=t.starts[0],horizon_s=horizon,gradient=g.tolist(),
                finite_differences=finite,passed=bool(passed),probe='log EI, log Cb, log drag, learned residual log gain'))
            atomic_json(output/'short_gradient_checks.json',dict(rows=rows,source_sha256=sha256_file(Path(__file__)),
                interpretation='Two real initial windows, two perturbation sizes; local checks do not prove global smoothness.'))
            print(f'{name} {horizon}s gradient check: {passed}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--horizons',type=float,nargs='+',default=[.05,.1])
    p.add_argument('--takes',nargs='+',default=['fig8_002','osc_001'])
    a=p.parse_args();run(a.output.resolve(),a.source.resolve(),a.benchmark.resolve(),a.horizons,a.takes)
