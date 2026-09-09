from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
root=Path.cwd();sys.path.insert(0,str(root))
from simulator.research_config import snapshot_assets
from simulator.research_execution import ResearchExecutionModel
from experimental_data.io import atomic_json,sha256_file
job=root/'runs/adaptation/20260908-normalized-M1'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
model=read(job/'models/all_five/model.json');protocol=read(job/'protocol.json')
rows=[]
for n in sorted(p.name for p in (job/'inputs').iterdir()):
 a=read(job/'validation/M0_initialized'/n/'metrics.json');b=read(job/'validation'/('leave_out_'+n)/n/'both/metrics.json')
 row={'take':n}
 for prefix,m in [('baseline',a),('adapted',b)]:
  for key,field in [('drone_whip_rmse_m','origin'),('tip_whip_rmse_m','tip'),('marker_whip_rmse_m','all_cable_markers')]:row[prefix+'_'+key]=m[field]['whip']['rmse_m']
  row[prefix+'_tip_strike_error_m']=m['tip_prediction_error_at_saved_strike_m']
 rows.append(row)
means={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='take'}
results=dict(heldout_rows=rows,equal_flight_mean=means,validation_stopped_by_user=True,remaining_all_five_replays_not_completed=True,evaluation_frame='hover_normalized_z',independent_future_flight_test=False,limitation=protocol['heldout'])
atomic_json(job/'validation/results.json',results)
report=job/'report/ADAPTATION_REPORT.md';report.parent.mkdir(exist_ok=True)
report.write_text('# Fresh normalized M1\n\nFitted from M0 and five cf7 flights; no retired M1 assets used. Drone nominal model, attitude response and residual; cable EI/Cb and damping plus bounded acceleration residual fitted. Source geometry/mass unchanged (157 g drone, 18 g cable assembly).\n\nAll errors use Z normalized by subtracting 0.05070575 m once. Each maneuver is excluded from its fit, but pre/post hover calibration uses all five flights. Conditional retrospective diagnostics, not independent or prospective flight evidence.\n\n| Whip prediction RMS | M0 | M1 |\n|---|---:|---:|\n'+''.join(f'| {label} | {100*means["baseline_"+key]:.2f} cm | {100*means["adapted_"+key]:.2f} cm |\n' for label,key in [('Drone','drone_whip_rmse_m'),('Tip','tip_whip_rmse_m')])+'\nTip improves in all five comparisons. Late recovery/hold drone RMS worsens from 9.56 to 15.44 cm. Whip is the priority.\n\nFitting completed. User stopped remaining all-five validation; no further ablations or gradient experiment were run. Existing completed full-sequence rollouts were finite. All-five candidate is for planning, and its fit evidence is in-sample. New real flights must establish policy improvement.\n',encoding='utf-8')
folder=root/'data/model_candidates/20260908-normalized-M1';folder.mkdir(exist_ok=False)
model=snapshot_assets(model,folder)
model['adaptation'].update(report=str(report.resolve()),evaluation_frame='hover_normalized_z',vertical_calibration=protocol['vertical_calibration'],validation=protocol['heldout'],status='fresh_normalized_whip_candidate',model_selected=False,old_M1_used=False,fitted_vehicle='cf7')
atomic_json(folder/'model.json',model)
engine=ResearchExecutionModel.from_mapping(model,root=folder,device='cpu')
assert model['motion_residual']['enabled'] and model['fullstate_execution']['enabled']
protected=read(job/'protected_before.json');changed=[p for p,h in protected.items() if not Path(p).exists() or sha256_file(p)!=h];assert not changed
atomic_json(job/'verification/final_checks.json',dict(passed=True,model_load_cpu=True,protected_files=len(protected),changed=changed,validation_stopped_by_user=True,gradient_check_run=False))
atomic_json(folder/'manifest.json',dict(source_job=str(job),model_sha256=sha256_file(folder/'model.json'),old_M1_used=False,policy_training_started=False,calibration=protocol['vertical_calibration'],completed_comparison_means=means))
atomic_json(job/'status.json',dict(status='FIT_COMPLETED_VALIDATION_STOPPED_BY_USER',published_model=str(folder/'model.json')))
print(folder/'model.json');print(means)
