"""Bounded diagnostic of smooth damping and one missing dissipative term."""
from pathlib import Path
from dataclasses import replace
import sys,copy,shutil,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools.diagnose_preliminary_cable import load,engine,METHODS,metrics,select_data
from experimental_data.current_adaptation import read,save
from experimental_data.current_adaptation_fit import cable_objectives
from experimental_data.io import sha256_file
from simulator.research_physics import ResearchPhysics
from simulator.cable import DderState


def main():
    torch.set_num_threads(4)
    pilot=ROOT/'runs/audits/preliminary1-cable-only-pilot'
    out=ROOT/'runs/audits/preliminary1-gradient-resolution/dynamic-mismatch'
    out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,out/'source.py')
    model,selected,states=load(pilot)
    model['cable']['curvature_frame_regularization']=2e-5
    e=engine(model)
    data=states[METHODS[1]];ids=list(e.cable.marker_node_indices[1:])
    train=[i for i,w in enumerate(selected) if w['role']=='training']
    assert len(train)==12 and all(selected[i]['take']!='figure8_002' for i in train)
    rates=np.array([0.,.1,.2,.4,.8,1.2,1.6,2.])
    save(out/'protocol.json',dict(scope='One finite diagnostic; no new M0 or neural training',
        source_pilot=str(pilot),regularization=2e-5,drag_rates_s_inv=rates,
        physical_pair='Published M0 EI/Cb fixed; cable residual disabled',
        selection='Equal training-take weight, equal robust losses at 1 and 2 seconds',
        training_windows=[selected[i]['name'] for i in train],
        separate_take='Only evaluated after freezing selected scalar; already inspected in earlier pilots',
        interpretation='Effective nonnegative linear velocity damping, not calibrated aerodynamics'))
    def forward(data,rates):
        expanded={k:v.repeat_interleave(len(rates),0) for k,v in data.items()}
        q,v=expanded['q'],expanded['v']
        constants=replace(e.physics.runtime_constants(q),external_drag_s_inv=q.new_tensor(np.tile(rates,len(data['q']))))
        step=ResearchPhysics(e.physics,DderState(q,v),e.dt_s,constants=constants,graph=True,fast_solve=True,fast_geometry=True)
        qs=[q]
        for root in expanded['roots'][:,1:].unbind(1):
            state=step(DderState(q,v),root);q,v=state.positions_m,state.velocities_m_s;qs.append(q)
        q=torch.stack(qs,1)
        assert torch.isfinite(q).all()
        return q,expanded
    started=time.perf_counter()
    q,expanded=forward(select_data(data,train),rates)
    losses={}
    for steps in (150,300):
        losses[str(steps/150)]=cable_objectives(q[:,:steps+1],expanded['truth'][:,:steps+1],ids).reshape(12,len(rates)).mean(0).cpu().numpy()
    scores=.5*(losses['1.0']+losses['2.0']);best=int(scores.argmin())
    save(out/'selection.json',dict(rates=rates,scores=scores,per_horizon=losses,selected=float(rates[best]),
        stop_reason='Finite diagnostic grid complete, no convergence claim',search_s=time.perf_counter()-started))
    report={}
    for rate in np.unique([0.,rates[best]]):
        q,_=forward(data,[rate])
        report[str(rate)]=metrics(q,data,selected,ids)
        np.savez_compressed(out/f'predictions-drag-{rate}.npz',q=q.cpu().numpy())
    save(out/'evaluation.json',report)
    before=read(pilot/'protected_before.json')
    changed=[p for p,h in before.items() if sha256_file(p)!=h]
    save(out/'integrity.json',dict(protected_files=len(before),changed=changed))
    assert not changed
    print('SELECTION',read(out/'selection.json'),flush=True)
    for rate,rows in report.items():
        print(rate,{role:{h:float(np.sqrt(np.mean([r['horizons'][h]['tip_rmse_m']**2 for r in rows if r['role']==role]))) for h in ('1.0','2.0')} for role in ('training','validation')},flush=True)


if __name__=='__main__':main()
