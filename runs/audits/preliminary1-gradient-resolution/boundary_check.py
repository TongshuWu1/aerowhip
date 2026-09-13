from pathlib import Path
import sys
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
import torch,numpy as np
from experimental_data.current_adaptation import read,save
from experimental_data.current_adaptation_fit import cable_windows,join_windows
from experimental_data.preliminary_prepare import PreliminaryTrial
from experimental_data.preliminary_fit import CableForward,rms
from simulator.research_execution import ResearchExecutionModel
torch.set_num_threads(4)
out=Path(__file__).parent;source=ROOT/'runs/adaptation/20260909-preliminary1-M0-v2'
coupled=read(out/'coupled_diagnostics.json')['takes'];windows=read(source/'windows.json');base=read(source/'source_candidate/model.json')
trials=[]
for take,row in coupled.items():
    t=PreliminaryTrial(source,next(w for w in windows if w['name']==row['window']),base)
    t.cable_history_s=1.;t.cable_velocity_weight_tau_s=.02;trials.append(t)
model=read(out/'candidate/model.json');e=ResearchExecutionModel.from_mapping(model,root=out/'candidate',device='cuda')
records,rejected=cable_windows(trials,e.physics,[0.],2.)
save(out/'boundary_strict_eligibility.json',dict(rejected=rejected))
# Diagnostic comparison, not fitting: preserve missing marker masks exactly as
# in coupled RMS evaluation; only the driving attachment must be fully observed.
records=[]
for t in trials:
    state,start,projection=t.cable_state(e.physics)
    _,_,sites=t.measured(start+np.arange(301)*e.dt_s)
    assert np.isfinite(sites[:,0]).all()
    records.append(dict(q=state.positions_m,v=state.velocities_m_s,
        roots=torch.tensor(sites[None,:,0],device='cuda',dtype=torch.float64),
        truth=torch.tensor(sites[None,:,1:],device='cuda',dtype=torch.float64)))
data=join_windows(records);q,_=CableForward(e,data)([model['cable']['EI_n_m2'],model['cable']['Cb_n_m2_s']])
report={}
for t,arr,truth in zip(trials,q.cpu().numpy(),data['truth'].cpu().numpy()):
    report[t.take]=dict(window=t.name,measured_attachment_tip=rms(arr[1:,-1],truth[1:,-1]),
        command_driven_tip=coupled[t.take]['models']['development_M0_weighted']['tip'],
        drone=coupled[t.take]['models']['development_M0_weighted']['drone'])
    np.savez_compressed(out/f'{t.take}-development-measured-attachment.npz',q=arr)
save(out/'boundary_comparison.json',dict(scope='Same candidate, exact same windows, initial state and 2 s duration; only boundary prediction source changes',takes=report))
print(report)
