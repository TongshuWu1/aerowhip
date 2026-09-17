"""Bounded training-only numerical audit of the already saved physical iterates."""
from pathlib import Path
import sys,gc
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experimental_data import whip_full_data as data
from experimental_data.whip_full_cable import make_residual,residual_objective,gradient_check
from experimental_data.io import atomic_json,sha256_file
from simulator.research_execution import ResearchExecutionModel
from simulator.workflow import read_json


def main():
    torch.set_num_threads(4)
    job=ROOT/'runs/adaptation/M1-paper-20260913'
    out=ROOT/'runs/audits/M1-paper-20260913-gradients'
    out.mkdir(parents=True,exist_ok=False)
    model,p,engine=data.load(job)
    c=p['full_update'];trials=data.drone_trials(job,model)
    rows=data.cable_rows(job,model,engine,trials,review_path=out/'windows.json')
    cd=data.cable_data(rows,c['replay_weight'])
    history=read_json(job/'cable_physics/history.json')
    initial=[model['cable'][k] for k in ('EI_n_m2','Cb_n_m2_s','external_drag_s_inv')]
    candidates=[dict(update=0,values=initial,loss=history[0]['baseline_loss'],best_loss=history[0]['baseline_loss']),*history[:-1]]
    # Final failed iterate is retained from its original audit; no need to retry it.
    review={'7':dict(update=7,parameters=history[-1]['values'],training_loss=history[-1]['best_loss'],
        gradient=read_json(job/'cable_residual_gradient_check.json'))}
    for row in candidates:
        del engine;gc.collect();torch.cuda.empty_cache()
        m=read_json(job/'stages/drone_residual/model.json')
        m['cable'].update(dict(zip(('EI_n_m2','Cb_n_m2_s','external_drag_s_inv'),row['values'])))
        engine=ResearchExecutionModel.from_mapping(m,root=job/'stages/drone_residual',device='cuda')
        net=make_residual(engine,c);objective,accelerator=residual_objective(engine,cd,np.array(row['values']),c)
        dest=out/f"update-{row['update']:03d}.json"
        try:
            gradient_check(net,objective,dest)
        except ValueError:
            if not dest.exists():raise
        result=read_json(dest)
        review[str(row['update'])]=dict(update=row['update'],parameters=row['values'],training_loss=row['loss'],gradient=result)
        atomic_json(out/'review.json',review)
        print('Saved iterate',row['update'],'gradient passed:',result['passed'],'relative error:',result['relative_error'],flush=True)
        del objective,accelerator,net
    atomic_json(out/'provenance.json',dict(source=str(job),validation_read=False,
        selection='No new physical optimization; audit all retained iterates and parent parameters under unchanged gradient tolerance.',
        source_history_sha256=sha256_file(job/'cable_physics/history.json'),script_sha256=sha256_file(__file__)))


if __name__=='__main__':main()
