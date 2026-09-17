"""Post-selection M0/M1 comparison on a shared, reviewed 0--1.5 s interval.

This diagnostic never changes a fit job, its selected checkpoint or its masks.
The copied protocol only extends evaluation to the operator-reviewed interval.
"""
from pathlib import Path
import sys
import argparse
import shutil
import json
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experimental_data.io import atomic_json,sha256_file
from experimental_data.whip_adaptation import verify_hashes
from experimental_data import whip_adaptation_fit as evaluation
from experimental_data.whip_full_fit import immutable_identity
from experimental_data.model_evaluation import model_identity
from simulator.research_execution import ResearchExecutionModel
from simulator.workflow import read_json


def main():
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',type=Path,default=ROOT/'runs/adaptation/M1-paper-20260913')
    job=parser.parse_args().job.resolve()
    result=read_json(job/'fit/result.json')
    if result['status']!='completed':raise ValueError('Complete and freeze selection before this comparison')
    verify_hashes(result['candidate_hashes'])
    verify_hashes(read_json(job/'prepared_hashes.json'))
    verify_hashes(read_json(job/'source_hashes.json'))
    out=ROOT/'runs/evaluation/M1-paper-20260913-common-1p5s'
    out.mkdir(parents=True,exist_ok=False)
    diag=out/'inputs';diag.mkdir()
    shutil.copytree(job/'inputs',diag/'inputs')
    p=read_json(job/'protocol.json')
    for row in p['takes'].values():
        if row['review']['physical_contact']!='none' or row['review']['free_motion_end_s']<1.5:
            raise ValueError('Full shared interval was not reviewed')
        row['end_s']=1.5
    p.update(diagnostics_only=True,evaluation_interval_s=[0.,1.5],source_frozen_fit=str(job))
    atomic_json(diag/'protocol.json',p)
    models={'M0':job/'source_candidate/model.json','M1':job/'candidate/model.json'}
    baseline=read_json(models['M0'])
    engine=ResearchExecutionModel.from_mapping(baseline,root=models['M0'].parent,device='cuda')
    rows=evaluation.records(diag,list(p['takes']),baseline,engine,'cuda')
    identity=immutable_identity(models['M0'])
    reports={};hashes={str(job/'fit/selection_frozen.json'):sha256_file(job/'fit/selection_frozen.json')}
    for name,path in models.items():
        print('Common 0–1.5 s comparison:',name,flush=True)
        if immutable_identity(path)!=identity:raise ValueError('Model geometry or numerical settings differ')
        model=read_json(path)
        e=ResearchExecutionModel.from_mapping(model,root=path.parent,device='cuda')
        reports[name]=evaluation.evaluate(rows,e,model['cable']['external_drag_s_inv'],out/name)
        _,h=model_identity(path);hashes.update(h)
    def get(row,metric):
        group,key=metric
        return row[group]['rmse_m'] if key is None else row[group][key]['rmse_m']
    metrics={'drone':('drone',None),'combined_tip':('command_driven','tip'),
        'combined_markers':('command_driven','markers'),'conditional_tip':('conditional_cable','tip'),
        'conditional_markers':('conditional_cable','markers')}
    groups={}
    for role in ('adaptation','validation'):
        names=[n for n,v in p['takes'].items() if v['role']==role]
        groups[role]={}
        for metric,route in metrics.items():
            a=np.array([get(reports['M0'][n],route) for n in names]);b=np.array([get(reports['M1'][n],route) for n in names])
            groups[role][metric]=dict(takes=names,M0_mean_m=float(a.mean()),M1_mean_m=float(b.mean()),
                M0_sample_sd_m=float(a.std(ddof=1)),M1_sample_sd_m=float(b.std(ddof=1)),
                reduction_percent=float(100*(1-b.mean()/a.mean())),M0_per_take_m=a.tolist(),M1_per_take_m=b.tolist())
    atomic_json(out/'report.json',dict(schema='paper_same_flight_common_interval_v1',
        interval_s=[0.,1.5],groups=groups,models=reports,source_fit=str(job),
        initialization='Same observed causal history and cable state; model-dependent drone compensation estimated by the same procedure.',
        interpretation='Postflight prediction, not physical M1 target performance. Validation 003/005 never enters optimization or selection.',
        timing='Estimated measured-stream clock alignment; no spatial normalization; identical schedules, time grids and masks.'))
    for f in out.rglob('*'):
        if f.is_file():hashes[str(f)]=sha256_file(f)
    hashes[str(Path(__file__))]=sha256_file(__file__)
    atomic_json(out/'evidence_hashes.json',hashes)
    verify_hashes(hashes)
    print(json.dumps(groups,indent=2),flush=True)


if __name__=='__main__':main()
