from pathlib import Path
import sys,json,numpy as np,torch
root=Path.cwd();sys.path.insert(0,str(root))
from experimental_data.current_adaptation import prepare,Trial,read,save
from experimental_data.current_adaptation_fit import cable_windows,join_windows,cable_forward,pose_prediction
from experimental_data.adaptation_check import interpolate_positions,load_comparison
from simulator.research_execution import ResearchExecutionModel
from dataclasses import replace
job=root/'runs/audits/adp1-model-isolation'
batch=root/'rehearsal_csv_and_result_in_real_flight/20260908-195207-486249-seed655_best_validation/adp1'
source=root/'data/model_candidates/20260908-normalized-M1/model.json'
if not job.exists():
 prepare(job,batch,source)
 protocol=read(job/'protocol.json');protocol.update(schema='adp1_read_only_isolation_v1',training='NONE: four recorded flights, no weights updated',heldout='Diagnostic only; no fitting',contact='No contact assumption required for comparing signals; user excluded005',vertical_convention='User-reviewed per-take constant Z normalization',source_model='Frozen M1 and M0 compared; original ghost never replaced');save(job/'protocol.json',protocol)
torch.set_num_threads(1);m1=read(source);m0=read(root/'runs/rehearsals/20260908-203914-039721/model.json');models={'M0':m0,'M1':m1};results=[]
for label,model in models.items():
 engine=ResearchExecutionModel.from_mapping(model,device='cuda');names=sorted(p.name for p in (job/'inputs').iterdir());trials=[Trial(job,n,model,end=1.) for n in names]
 records,rejected=cable_windows(trials,engine.physics,[0.],1.02)
 save(job/(label+'_rejected.json'),rejected)
 if len(records)!=4:raise ValueError('Need four complete measured-boundary windows; inspect rejection reasons')
 data=join_windows(records);q,_=cable_forward(engine,data,[model['cable']['EI_n_m2'],model['cable']['Cb_n_m2_s']]);q=q.cpu().numpy();indices=list(engine.cable.marker_node_indices[1:])
 for tr,rec,measured_boundary_q in zip(trials,records,q):
  times=tr.grid(1.);pose=pose_prediction(tr,engine.drone,times);state,_,proj=tr.cable_state(engine.physics)
  coupled,_=cable_forward(engine,dict(q=state.positions_m,v=state.velocities_m_s,roots=pose['position_attachment_m']),[model['cable']['EI_n_m2'],model['cable']['Cb_n_m2_s']]);coupled=coupled[0].cpu().numpy()
  native=tr.data['time'];native=native[(native>=0)&(native<=1.)];truthp,truthr,truths=tr.measured(native)
  pp=interpolate_positions(times,pose['position_origin_m'][0].cpu().numpy(),native);qc=interpolate_positions(times,coupled,native);qm=interpolate_positions(rec['time'],measured_boundary_q,native)
  def rms(x,y):return float(np.sqrt(np.nanmean(np.sum((x-y)**2,axis=-1))))
  init=tr.initial_pose(engine.drone.parameters);hover=torch.tensor(tr.data['hover_commands'][-1:],device='cuda',dtype=torch.float64)
  residual=engine.drone.residual(init.position,init.velocity,init.compensation,hover).detach().cpu().numpy()[0]
  row=dict(model=label,take=tr.name,causal_drone_rms_m=rms(pp,truthp),coupled_tip_rms_m=rms(qc[:,-1],truths[:,-1]),measured_attachment_tip_rms_m=rms(qm[:,-1],truths[:,-1]),measured_attachment_marker_rms_m=rms(qm[:,indices],truths[:,1:]),initial_projection_max_coordinate_m=proj,initial_compensation_m_s2=init.compensation.cpu().numpy()[0],hover_nn_acceleration_m_s2=residual,drone_axis_rms_m=np.sqrt(np.nanmean((pp-truthp)**2,axis=0)))
  results.append(row);save(job/'results_in_progress.json',results);print(json.dumps({k:v for k,v in row.items() if isinstance(v,(str,float))}),flush=True)
  np.savez_compressed(job/(label+'_'+tr.name+'.npz'),time=native,measured_position=truthp,measured_sites=truths,causal_drone=pp,coupled_cable=qc,measured_boundary_cable=qm,marker_indices=indices)
save(job/'results.json',dict(rows=results,means={label:{k:float(np.mean([r[k] for r in results if r['model']==label])) for k in ['causal_drone_rms_m','coupled_tip_rms_m','measured_attachment_tip_rms_m','measured_attachment_marker_rms_m']} for label in models},parameters_fitted=False,original_ghost_replaced=False))
print(read(job/'results.json')['means'],flush=True)
