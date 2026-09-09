from pathlib import Path
import sys,json,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from simulator.research_execution import ResearchExecutionModel
from simulator.drone_pose_response import PoseResponseState
from simulator.cable import DderState
torch.set_num_threads(1)
folder=ROOT/'runs/rehearsals/20260908-203914-039721'
candidate=ROOT/'data/model_candidates/20260908-differentiable-zero-extension/model.json'
with np.load(folder/'rehearsal.npz') as z:data={k:z[k] for k in z.files}
t=lambda a:torch.as_tensor(a,dtype=torch.float64,device='cuda')
p=t(data['origin_positions_m'][:1]);r=t(data['origin_rotations'][:1]);zero=torch.zeros_like(p)
initial=PoseResponseState(p,zero,r,zero,zero,torch.eye(3,device='cuda',dtype=p.dtype)[None],'prehover_effective_alignment',0.)
q=t(data['cable_positions_m'][:1]);state=DderState(q,torch.zeros_like(q))
packets=t(data['commands'][None]);pt=data['command_time_s'];times=data['prediction_time_s'][:151]
hover=packets[:,0].clone();hover[:,3:9]=0
model=ResearchExecutionModel.from_mapping(json.loads(candidate.read_text()),device='cuda')
s=np.clip(pt,0,1);active=pt<=1
direction=torch.zeros_like(packets)
for col,values in [(0,64*s**3*(1-s)**3),(3,64*(3*s**2-12*s**3+15*s**4-6*s**5)),(6,64*(6*s-36*s**2+60*s**3-30*s**4))]:
    direction[0,:,col]=t(np.where(active,values,0.))
weight=t([.7,-.2,.5])
def forward(amplitude,grad=False):
    result=model.predict(initial,state,packets+amplitude*direction,pt,times,hover_command=hover,
        gradients=grad,graph=not grad,checkpoint_steps=0)
    return (result['cable_positions_m'][0,:,-1]*weight).sum(-1)
report={'checkpointed_reverse_final':20729.4621326192}
base=forward(0.)
report['base']=base.detach().cpu().tolist()
for epsilon in [1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]:
    fd=(forward(epsilon)-forward(-epsilon))/(2*epsilon)
    report[str(epsilon)]=fd.detach().cpu().tolist()
    print('FD',epsilon,[float(fd[i]) for i in [15,37,75,112,141,150]],flush=True)
    (Path(__file__).parent/'gradient_probe.json').write_text(json.dumps(report,indent=2))
try:
    value,derivative=torch.func.jvp(lambda a:forward(a,True),(t(0.),),(t(1.),))
    report['jvp']=derivative.detach().cpu().tolist()
    print('JVP',[float(derivative[i]) for i in [15,37,75,112,141,150]],flush=True)
except Exception as exc:
    report['jvp_error']=str(exc);print('JVP error',repr(exc),flush=True)
(Path(__file__).parent/'gradient_probe.json').write_text(json.dumps(report,indent=2))
