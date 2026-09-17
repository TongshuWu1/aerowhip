"""Post-selection diagnosis only: identical held-out takes across saved stages."""
from pathlib import Path
import sys,gc
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experimental_data import whip_full_data as data
from experimental_data.whip_full_fit import compare_drone_runtime
from experimental_data.io import atomic_json,sha256_file
from simulator.research_execution import ResearchExecutionModel
from simulator.workflow import read_json


def main():
    torch.set_num_threads(4)
    job=ROOT/'runs/adaptation/M2-paper-20260913'
    assert read_json(job/'status.json')['status']=='completed'
    out=ROOT/'runs/audits/M2-drone-stage-diagnosis-20260913';out.mkdir(parents=True,exist_ok=False)
    p=read_json(job/'protocol.json');report={}
    stages={'M1_parent':job/'source_candidate/model.json',
        'M2_nominal':job/'stages/drone_nominal/model.json',
        'M2_drone_residual':job/'stages/drone_residual/model.json'}
    for name,path in stages.items():
        model=read_json(path);engine=ResearchExecutionModel.from_mapping(model,root=path.parent,device='cuda')
        trials=data.drone_trials(job,model,validation=True)
        rows=compare_drone_runtime(trials,engine,p['full_update'],out/name)
        report[name]=dict(position_rmse_mean_m=float(np.mean([r['position_rms_m'] for r in rows.values()])),
            per_take_m={n:r['position_rms_m'] for n,r in rows.items()},model=str(path))
        del trials,engine;gc.collect();torch.cuda.empty_cache()
    for name,folder in [('training_M1',job/'baseline_drone'),('training_M2',job/'adapted_drone')]:
        rows=read_json(folder/'report.json');groups={}
        for category in sorted(set(r['category'] for r in rows.values())):
            group={}
            for row in rows.values():
                if row['category']==category:group.setdefault(row['take'],[]).append(row['position_rms_m'])
            groups[category]=dict(per_take_m={k:float(np.mean(v)) for k,v in group.items()},
                mean_m=float(np.mean([np.mean(v) for v in group.values()])))
        report[name]=groups
    atomic_json(out/'report.json',dict(stages=report,validation_used_for_selection=False,
        candidate_changed=False,training_restarted=False,script_sha256=sha256_file(__file__),
        interpretation='Post-selection saved-stage diagnostic; native recorded timestamps and identical held-out take membership.'))
    print(report,flush=True)


if __name__=='__main__':main()
