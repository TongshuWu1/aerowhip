"""Training-only derivative diagnosis, without fitting or selecting a model."""
from pathlib import Path
import argparse,copy,time,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experimental_data.current_adaptation import read,save
from experimental_data.preliminary_prepare import PreliminaryTrial
from experimental_data.current_adaptation_fit import cable_windows,join_windows,cable_objectives
from experimental_data.cuda_cable_fit import CudaCableFit
from simulator.research_execution import ResearchExecutionModel
from experimental_data.io import sha256_file


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',required=True)
    parser.add_argument('--regularization',type=float,default=2e-7)
    parser.add_argument('--dense',action='store_true')
    parser.add_argument('--steps',type=int,default=150)
    args=parser.parse_args()
    torch.set_num_threads(4)
    source=ROOT/'runs/adaptation/20260909-preliminary1-M0-v2'
    out=ROOT/'runs/audits/preliminary1-gradient-resolution'/args.name
    out.mkdir(parents=True,exist_ok=False)
    import shutil
    shutil.copy2(__file__,out/'source.py')
    model=read(source/'candidate/model.json');base=read(source/'source_candidate/model.json')
    protected={str(p):sha256_file(p) for p in (source/'candidate').glob('*') if p.is_file()}
    model['cable']['curvature_frame_regularization']=args.regularization
    e=ResearchExecutionModel.from_mapping(model,root=source/'candidate',device='cuda',trainable_residuals=True)
    trials=[]
    for name in ('figure8_001-00641','figure8_001-02441'):
        t=PreliminaryTrial(source,next(w for w in read(source/'windows.json') if w['name']==name),base)
        assert t.role=='training'
        t.cable_history_s=1.;t.cable_velocity_weight_tau_s=.02;trials.append(t)
    records,rejected=cable_windows(trials,e.physics,[0.],args.steps/150)
    assert len(records)==2 and not rejected
    data=join_windows(records);net=e.physics.motion_residual
    params=data['q'].new_tensor([model['cable']['EI_n_m2'],model['cable']['Cb_n_m2_s']])
    save(out/'protocol.json',dict(settings=vars(args),model_sha256=protected[str(source/'candidate/model.json')],
        windows=[t.name for t in trials],scope='Frozen residual and physical parameters; no optimization'))
    start=time.perf_counter()
    accelerator=CudaCableFit(e,data,params,block_steps=3,fast_solve=not args.dense)
    ids=list(e.cable.marker_node_indices[1:]);bias=net.correction_head.bias;original=bias.detach().clone()
    net.zero_grad(set_to_none=True)
    q,v=accelerator(data,gradients=True);loss=cable_objectives(q,data['truth'],ids).mean();loss.backward()
    index=int(bias.grad.abs().argmax());ad=float(bias.grad[index]);norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),float('inf')))
    arrays=q.detach().cpu().numpy()
    grad_vector=torch.cat([p.grad.flatten() for p in net.parameters()]).cpu().numpy()
    with torch.no_grad():q0,_=accelerator(data,gradients=False)
    row=dict(autograd=ad,bias_index=index,loss=float(loss),gradient_norm=norm,
        differentiable_vs_inference_position_max_m=float((q.detach()-q0).abs().max()),finite_differences=[])
    del q,v,loss
    for eps in (1e-4,1e-6,1e-8,1e-9,1e-10,1e-11):
        values=[]
        for sign in (-1,1):
            with torch.no_grad():
                bias.copy_(original);bias[index]+=sign*eps
                q,_=accelerator(data,gradients=False)
                values.append(float(cable_objectives(q,data['truth'],ids).mean()))
        fd=(values[1]-values[0])/(2*eps)
        row['finite_differences'].append(dict(epsilon=eps,derivative=fd,
            relative_error=abs(fd-ad)/max(abs(fd),abs(ad),1e-10)))
        save(out/'result.json',row)
        print(args.name,row['finite_differences'][-1],flush=True)
    with torch.no_grad():bias.copy_(original)
    row['elapsed_s']=time.perf_counter()-start
    row['protected_unchanged']=all(sha256_file(p)==h for p,h in protected.items())
    save(out/'result.json',row)
    np.savez_compressed(out/'baseline.npz',q=arrays,gradients=grad_vector)
    print(args.name,row,flush=True)


if __name__=='__main__':main()
