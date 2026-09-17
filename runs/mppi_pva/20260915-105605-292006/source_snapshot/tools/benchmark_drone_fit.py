from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from copy import deepcopy
import json,time
import torch
from experimental_data.current_adaptation import Trial,read
from experimental_data.current_adaptation_fit import WeightedTranslationBatch
from experimental_data.cuda_drone_fit import CudaDroneFit
from simulator.drone_pose_response import PoseResponseParameters
from simulator.drone_pose_residual import load_residual
from experimental_data.io import sha256_file

torch.set_num_threads(4)
job=Path('runs/adaptation/20260909-pva-M0-bootstrap')
model=read(job/'source_candidate/model.json')
trials=[Trial(job,p.name,model,end=1.) for p in sorted((job/'inputs').iterdir())]
for trial in trials:
    valid=trial.masks['fit_position']&(trial.time<=1.)
    trial.weights=valid.astype(float)/valid.sum()
params=PoseResponseParameters(**read(job/'drone/drone_model.json')['nominal']['parameters'])
path=job/'drone/drone_residual.pt';net=load_residual(path,sha256_file(path),'cpu').requires_grad_(True)
batch=WeightedTranslationBatch(trials,params,device='cpu')
start=time.perf_counter();loss=batch.loss(net);grad=torch.autograd.grad(loss,tuple(net.parameters()))
base=time.perf_counter()-start;score=float(loss.detach());del loss
print(json.dumps(dict(stage='cpu',seconds=base,loss=score)),flush=True)
gpu=deepcopy(net).cuda();start=time.perf_counter();fast=CudaDroneFit(trials,params,gpu)
torch.cuda.synchronize();capture=time.perf_counter()-start;times=[]
for _ in range(4):
    torch.cuda.synchronize();start=time.perf_counter();loss=fast();g=torch.autograd.grad(loss,tuple(gpu.parameters()))
    torch.cuda.synchronize();times.append(time.perf_counter()-start)
    torch.testing.assert_close(loss.cpu(),torch.tensor(score,dtype=torch.float64),atol=1e-11,rtol=1e-10)
    for a,b in zip(g,grad):torch.testing.assert_close(a.cpu(),b,atol=1e-10,rtol=1e-8)
result=dict(cpu_seconds=base,gpu_seconds=times,capture_seconds=capture,loss=score,
    maximum_gradient_difference=max(float((a.cpu()-b).abs().max()) for a,b in zip(g,grad)))
out=Path('runs/audits/pva-performance/drone.json');out.parent.mkdir(exist_ok=True,parents=True)
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
