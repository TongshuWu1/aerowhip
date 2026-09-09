"""Audit the new execution modes against the exact retained flight rehearsal.

No optimizer, training worker, active-model selection, CSV export or flight API.
Writes audit outputs only. Run after preparing a separate unfitted candidate.
"""
from pathlib import Path
import argparse
import json
import platform
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from experimental_data.io import atomic_json, sha256_file
from simulator.cable import DderState
from simulator.drone_pose_response import PoseResponseState
from simulator.research_execution import ResearchExecutionModel


def audit(rehearsal,candidate,directory,device='cuda',smooth_candidate=None):
    rehearsal=Path(rehearsal).resolve();candidate=Path(candidate).resolve();directory=Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    with np.load(rehearsal/'rehearsal.npz') as z:data={k:z[k] for k in z.files}
    csv=np.loadtxt(rehearsal/'fullstate_30hz.csv',delimiter=',',skiprows=1)
    np.testing.assert_allclose(csv[:,0],data['command_time_s'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(csv[:,1:],data['commands'],atol=1e-12,rtol=0)
    tensor=lambda x:torch.as_tensor(x,dtype=torch.float64,device=device)
    p=tensor(data['origin_positions_m'][:1]);r=tensor(data['origin_rotations'][:1])
    zero=torch.zeros_like(p);eye=torch.eye(3,dtype=p.dtype,device=device)[None]
    pose=PoseResponseState(p,zero,r,zero,zero,eye,'prehover_effective_alignment',0.)
    q=tensor(data['cable_positions_m'][:1]);cable=DderState(q,torch.zeros_like(q))
    packets=tensor(data['commands'][None]);pt=data['command_time_s'];times=data['prediction_time_s']
    hover=packets[:,0].clone();hover[:,3:9]=0.
    old=ResearchExecutionModel.from_mapping(json.loads((rehearsal/'model.json').read_text()),root=rehearsal,device=device)
    new=ResearchExecutionModel.from_mapping(json.loads(candidate.read_text()),root=candidate.parent,device=device)
    report=dict(schema='differentiable_execution_audit_v1',platform=platform.platform(),
        python=platform.python_version(),torch=torch.__version__,device=str(device),
        gpu=torch.cuda.get_device_name(0) if str(device).startswith('cuda') else None,
        rehearsal=str(rehearsal),candidate_model=str(candidate),candidate_sha256=sha256_file(candidate),
        csv_sha256=sha256_file(rehearsal/'fullstate_30hz.csv'),
        trained=False,model_selected=False,flight_performed=False)

    def forward(model,values,output_times,**kwargs):
        return model.predict(pose,cable,values,pt,output_times,hover_command=hover,**kwargs)

    start=time.perf_counter()
    print('Checking complete 11.2 s M0 inference against saved prediction',flush=True)
    baseline=forward(old,packets,times,graph=str(device).startswith('cuda'))
    report['baseline_forward_s']=time.perf_counter()-start
    mapping={'position_origin_m':'origin_positions_m','rotation_tracking_to_world':'origin_rotations',
        'cable_positions_m':'cable_positions_m'}
    report['saved_prediction_max_abs_difference']={}
    for key,saved in mapping.items():
        expected=tensor(data[saved][None])
        difference=float((baseline[key]-expected).abs().max())
        report['saved_prediction_max_abs_difference'][key]=difference
        torch.testing.assert_close(baseline[key],expected,atol=2e-7,rtol=1e-8)
    print('Checking complete zero-extension candidate against M0',flush=True)
    start=time.perf_counter()
    expanded=forward(new,packets,times,graph=str(device).startswith('cuda'))
    report['candidate_forward_s']=time.perf_counter()-start
    report['zero_extension_max_abs_difference']={}
    for key in baseline:
        if baseline[key].dtype==torch.bool:
            assert torch.equal(baseline[key],expanded[key]);continue
        report['zero_extension_max_abs_difference'][key]=float((expanded[key]-baseline[key]).abs().max())
        torch.testing.assert_close(expanded[key],baseline[key],atol=1e-10,rtol=1e-10)

    if smooth_candidate is not None:
        path=Path(smooth_candidate).resolve()
        new=ResearchExecutionModel.from_mapping(json.loads(path.read_text()),root=path.parent,device=device)
        print('Checking separately versioned smooth-damping candidate',flush=True)
        expanded=forward(new,packets,times,graph=str(device).startswith('cuda'))
        report['gradient_candidate']=str(path)
        report['gradient_candidate_sha256']=sha256_file(path)
        report['damping_frame_regularization']=new.physics.parameters.curvature_frame_regularization
        report['smooth_candidate_change_from_M0_m']=float((expanded['cable_positions_m']-baseline['cable_positions_m']).abs().max())

    meta=json.loads((rehearsal/'rehearsal.json').read_text())
    end=int(round(meta['whip_end_s']/new.dt_s))+1;whip_times=times[:end]
    print('Checking full-whip BPTT and a smooth PVA perturbation gradient',flush=True)
    values=packets.clone().requires_grad_()
    start=time.perf_counter()
    learned=forward(new,values,whip_times,gradients=True,checkpoint_steps=25)
    for key in expanded:
        torch.testing.assert_close(learned[key],expanded[key][:,:end],atol=1e-9,rtol=1e-9)
    weight=packets.new_tensor([.7,-.2,.5])
    objective=(learned['cable_positions_m'][0,-1,-1]*weight).sum()
    gradient=torch.autograd.grad(objective,values)[0]
    assert bool(torch.isfinite(gradient).all())
    report['whip_forward_backward_s']=time.perf_counter()-start
    report['command_gradient_norm']=float(gradient.norm())
    report['whip_forward_mode_max_abs_difference_m']=float((learned['cable_positions_m']-expanded['cable_positions_m'][:,:end]).abs().max().detach())
    # One metre amplitude times a C2 bump over the whip. P, V and A change
    # consistently; +/-epsilon are numerical tests, never exported commands.
    duration=float(meta['whip_end_s']);s=pt/duration;active=(s>=0)&(s<=1)
    s=np.clip(s,0,1)
    bump=64*s**3*(1-s)**3
    velocity=64*(3*s**2-12*s**3+15*s**4-6*s**5)/duration
    acceleration=64*(6*s-36*s**2+60*s**3-30*s**4)/duration**2
    direction=torch.zeros_like(values)
    for col,array in [(0,bump),(3,velocity),(6,acceleration)]:direction[0,:,col]=tensor(np.where(active,array,0.))
    analytical=(gradient*direction).sum()
    def objective_forward(amplitude):
        result=forward(new,packets+amplitude*direction,whip_times,graph=str(device).startswith('cuda'))
        return (result['cable_positions_m'][0,-1,-1]*weight).sum()
    report['smooth_pva_gradient']=[]
    for epsilon in [1e-5,1e-6]:
        finite_difference=(objective_forward(epsilon)-objective_forward(-epsilon))/(2*epsilon)
        item=dict(autograd=float(analytical),finite_difference=float(finite_difference),
            perturbation_amplitude_m=epsilon,absolute_difference=float((analytical-finite_difference).abs()))
        report['smooth_pva_gradient'].append(item)
        atomic_json(directory/'verification_in_progress.json',report)
        torch.testing.assert_close(analytical,finite_difference,atol=2e-5,rtol=3e-3)
    if str(device).startswith('cuda'):
        report['peak_allocated_cuda_bytes']=torch.cuda.max_memory_allocated()
    integrity=directory/'preserved_before.json'
    if integrity.exists():
        original=json.loads(integrity.read_text())
        changed=[name for name,digest in original.items() if not (ROOT/name).is_file() or sha256_file(ROOT/name)!=digest]
        report['preserved_files_checked']=len(original);report['preserved_files_changed']=changed
        if changed:raise AssertionError(f'Preserved artifacts changed: {changed}')
    report['passed']=True
    atomic_json(directory/'verification.json',report)
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rehearsal',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--smooth-candidate',type=Path)
    args=parser.parse_args()
    audit(args.rehearsal,args.candidate,args.output,args.device,args.smooth_candidate)
