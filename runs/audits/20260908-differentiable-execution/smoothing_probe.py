"""Isolated investigation of the legacy curvature-damping branch; no model files changed."""
from pathlib import Path
import inspect,json,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from simulator.cable import dder,DderState
from simulator.drone_pose_response import PoseResponseState
from simulator.research_execution import ResearchExecutionModel
torch.set_num_threads(1)
source=inspect.getsource(dder._curvature_rate_jacobian_impl)
begin=source.index('        unit_normal =')
end=source.index('\n    zero_block =',begin)
replacement='''        normal_squared=normal.square().sum(-1)
        regularization=2e-7
        denominator=normal_squared+regularization
        coefficient=-2.0/(1.0+(previous*following).sum(-1)).clamp_min(128*torch.finfo(positions.dtype).eps)
        bent_previous=coefficient[...,None,None]*normal[..., :,None]*torch.linalg.cross(normal,previous,dim=-1)[...,None,:]/denominator[...,None,None]
        bent_following=coefficient[...,None,None]*normal[..., :,None]*torch.linalg.cross(following,normal,dim=-1)[...,None,:]/denominator[...,None,None]
        straight_inverse=0.5*(identity_batch-previous[..., :,None]*previous[...,None,:])
        corotation=_skew_matrix(curvature)
        w=(regularization/denominator)[...,None,None]
        return (bent_previous+w*(derivative_previous+corotation@straight_inverse@_skew_matrix(previous)),
                bent_following+w*(derivative_following+corotation@straight_inverse@_skew_matrix(following)))
'''
namespace=dict(vars(dder));exec(source[:begin]+replacement+source[end:],namespace)
dder._curvature_rate_jacobian_impl=namespace['_curvature_rate_jacobian_impl']
folder=ROOT/'runs/rehearsals/20260908-203914-039721'
path=ROOT/'data/model_candidates/20260908-differentiable-zero-extension/model.json'
with np.load(folder/'rehearsal.npz') as z:data={k:z[k] for k in z.files}
t=lambda a:torch.as_tensor(a,dtype=torch.float64,device='cpu')
p=t(data['origin_positions_m'][:1]);r=t(data['origin_rotations'][:1]);zero=torch.zeros_like(p)
initial=PoseResponseState(p,zero,r,zero,zero,torch.eye(3,dtype=p.dtype)[None],'prehover_effective_alignment',0.)
q=t(data['cable_positions_m'][:1]);state=DderState(q,torch.zeros_like(q))
packets=t(data['commands'][None]);pt=data['command_time_s'];times=data['prediction_time_s'][:151]
hover=packets[:,0].clone();hover[:,3:9]=0
model=ResearchExecutionModel.from_mapping(json.loads(path.read_text()),device='cpu')
s=np.clip(pt,0,1);active=pt<=1;direction=torch.zeros_like(packets)
for col,values in [(0,64*s**3*(1-s)**3),(3,64*(3*s**2-12*s**3+15*s**4-6*s**5)),(6,64*(6*s-36*s**2+60*s**3-30*s**4))]:
 direction[0,:,col]=t(np.where(active,values,0))
weight=t([.7,-.2,.5]);amplitude=t(0.).requires_grad_()
def forward(a,grad=False):
 return model.predict(initial,state,packets+a*direction,pt,times,hover_command=hover,
   gradients=grad,graph=False,checkpoint_steps=25)
start=time.perf_counter();out=forward(amplitude,True)
score=(out['cable_positions_m'][0,-1,-1]*weight).sum()
g=torch.autograd.grad(score,amplitude)[0]
report={'gradient':float(g),'forward_backward_s':time.perf_counter()-start,
 'max_difference_from_M0_m':float((out['cable_positions_m'][0].detach()-t(data['cable_positions_m'][:151])).abs().max()),
 'tip_end_difference_from_M0_m':float((out['cable_positions_m'][0,-1,-1].detach()-t(data['cable_positions_m'][150,-1])).norm())}
print(report,flush=True)
for epsilon in [1e-4,1e-5,1e-6,1e-7]:
 a=forward(epsilon);b=forward(-epsilon)
 fd=((a['cable_positions_m'][0,-1,-1]-b['cable_positions_m'][0,-1,-1])*weight).sum()/(2*epsilon)
 report[str(epsilon)]=float(fd);print('FD',epsilon,float(fd),flush=True)
 (Path(__file__).parent/'smoothing_probe.json').write_text(json.dumps(report,indent=2))
