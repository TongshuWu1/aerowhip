from pathlib import Path
import sys,json,numpy as np,torch
from dataclasses import replace
root=Path.cwd();sys.path.insert(0,str(root));torch.set_num_threads(1)
from experimental_data.current_adaptation import Trial,read,save
from experimental_data.current_adaptation_fit import pose_prediction
from simulator.research_execution import ResearchExecutionModel
from experimental_data.adaptation_check import load_comparison,interpolate_positions
job=root/'runs/audits/adp1-model-isolation';model=read(root/'data/model_candidates/20260908-normalized-M1/model.json');rows=[]
for folder in sorted((job/'inputs').iterdir()):
 trial=Trial(job,folder.name,model,end=1.);times=trial.grid(1.);native=trial.data['time'];native=native[(native>=0)&(native<=1)];truth=trial.measured(native)[0]
 for mode in ['M1_without_drone_NN','M1_hover_NN_consistent_compensation']:
  eng=ResearchExecutionModel.from_mapping(model,device='cuda');init=trial.initial_pose(eng.drone.parameters);d=trial.data
  if mode=='M1_without_drone_NN':
   with torch.no_grad():
    for p in eng.drone.residual.parameters():p.zero_()
  else:
   b0=init.compensation;bias=b0.clone();p=torch.tensor(trial.hover[0][0],device='cuda',dtype=torch.float64);v=init.velocity.expand_as(p);commands=torch.tensor(d['hover_commands'],device='cuda',dtype=torch.float64)
   with torch.no_grad():
    for _ in range(30):bias=b0-eng.drone.residual(p,v,bias.expand_as(p),commands).mean(0,keepdim=True)
   init=replace(init,compensation=bias)
  result=eng.drone.predict(init,torch.tensor(d['packets'][None],device='cuda',dtype=torch.float64),d['packet_time'],times,trial.offset,graph=True,hover_command=torch.tensor(d['hover_commands'][-1:],device='cuda',dtype=torch.float64))
  pred=interpolate_positions(times,result['position_origin_m'][0].cpu().numpy(),native);rms=float(np.sqrt(np.mean(np.sum((pred-truth)**2,axis=1))))
  rows.append(dict(take=trial.name,mode=mode,drone_rms_m=rms));print(rows[-1],flush=True)
save(job/'drone_sensitivity.json',dict(rows=rows,means={m:float(np.mean([r['drone_rms_m'] for r in rows if r['mode']==m])) for m in {r['mode'] for r in rows}},note='Diagnostics only; compensation sensitivity approximates mean hover residual using final velocity. No deployed changes or parameter training.'))
