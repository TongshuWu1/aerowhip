"""Post-selection ablation: does M1's cable residual improve held-out prediction?"""
from pathlib import Path
from dataclasses import asdict
import gc
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experimental_data import whip_adaptation_fit as evaluation
from experimental_data.io import atomic_json,sha256_file
from experimental_data.whip_adaptation import verify_hashes
from experimental_data.model_evaluation import model_identity
from experimental_data.whip_full_fit import immutable_identity
from simulator.research_execution import ResearchExecutionModel
from simulator.workflow import read_json


def main():
    torch.set_num_threads(4)
    common=ROOT/'runs/evaluation/M1-paper-20260913-common-1p5s'
    report=read_json(common/'report.json');job=Path(report['source_fit'])
    verify_hashes(read_json(common/'evidence_hashes.json'))
    verify_hashes(read_json(job/'fit/result.json')['candidate_hashes'])
    out=common/'cable_residual_effect';out.mkdir(exist_ok=False)
    path=job/'stages/cable_physics/model.json'
    a=read_json(path);b=read_json(job/'candidate/model.json')
    if a.get('motion_residual',{}).get('enabled'):
        raise ValueError('Expected the preserved physical-only M1 stage without cable residual')
    if a['cable']!=b['cable'] or immutable_identity(path)!=immutable_identity(job/'candidate/model.json'):
        raise ValueError('Ablation changes cable physics or unreviewed settings')
    physical=ResearchExecutionModel.from_mapping(a,root=path.parent,device='cuda')
    full=ResearchExecutionModel.from_mapping(b,root=job/'candidate',device='cuda')
    if asdict(physical.drone.parameters)!=asdict(full.drone.parameters):
        raise ValueError('Ablation changes the fitted quadrotor parameters')
    if any(not torch.equal(v,full.drone.residual.state_dict()[k]) for k,v in physical.drone.residual.state_dict().items()):
        raise ValueError('Ablation changes the quadrotor residual')
    del full;gc.collect();torch.cuda.empty_cache()
    diag=common/'inputs';protocol=read_json(diag/'protocol.json')
    baseline_path=job/'source_candidate/model.json';baseline_model=read_json(baseline_path)
    baseline=ResearchExecutionModel.from_mapping(baseline_model,root=baseline_path.parent,device='cuda')
    rows=evaluation.records(diag,list(protocol['takes']),baseline_model,baseline,'cuda')
    print('Evaluating M1 with fitted physics and without cable residual',flush=True)
    no_residual=evaluation.evaluate(rows,physical,a['cable']['external_drag_s_inv'],out/'physical_only')
    groups={}
    for role in ('adaptation','validation'):
        names=[n for n,v in protocol['takes'].items() if v['role']==role]
        groups[role]={}
        for route in ('command_driven','conditional_cable'):
            x=np.array([no_residual[n][route]['tip']['rmse_m'] for n in names])
            y=np.array([report['models']['M1'][n][route]['tip']['rmse_m'] for n in names])
            groups[role][route]=dict(takes=names,without_residual_per_take_m=x.tolist(),
                with_residual_per_take_m=y.tolist(),without_residual_mean_m=float(x.mean()),
                with_residual_mean_m=float(y.mean()),reduction_percent=float(100*(1-y.mean()/x.mean())))
    result=dict(interval_s=[0.,1.5],groups=groups,physical_only=no_residual,
        interpretation='Post-selection diagnostic only. Identical fitted quadrotor and cable physics; only the cable residual differs. No tuning or candidate reselection from this result.',
        source_full_report=str(common/'report.json'),source_physical_model=str(path))
    atomic_json(out/'report.json',result)
    _,hashes=model_identity(path)
    for f in [common/'report.json',Path(__file__),*out.rglob('*')]:
        if f.is_file():hashes[str(f)]=sha256_file(f)
    atomic_json(out/'evidence_hashes.json',hashes);verify_hashes(hashes)
    print(groups,flush=True)


if __name__=='__main__':main()
