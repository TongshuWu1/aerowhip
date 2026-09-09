import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from pathlib import Path
import json,numpy as np
from experimental_data.adaptation_check import load_comparison,flight_names,sha256
from experimental_data.adaptation_progress import measured_flight_metrics
from experimental_data.adaptation_rounds import read_optitrack
from experimental_data.hover_calibration import load_calibration,corrected_tracking
from experimental_data.io import atomic_json
root=Path.cwd();b=root/'rehearsal_csv_and_result_in_real_flight/20260908-195207-486249-seed655_best_validation/adp1';out=b/'processed/hover_normalized';out.mkdir(parents=True,exist_ok=True);rows=[]
for n in flight_names(b):
 c=load_calibration(b,n);raw=read_optitrack(b/'flight_take'/f'{n}.csv');m=corrected_tracking(raw,c['bias_z_m']);d=load_comparison(root,b,n);metrics=measured_flight_metrics(d);rows.append(metrics)
 np.testing.assert_allclose(m['drone'][:,:2],raw['drone'][:,:2],equal_nan=True)
 np.testing.assert_allclose(m['cable']-m['drone'][:,None],raw['cable']-raw['drone'][:,None],equal_nan=True,atol=1e-14)
 np.savez_compressed(out/(n+'.npz'),optitrack_time_s=m['time'],position_origin_m=m['drone'],cable_marker_positions_m=m['cable'],quaternion_xyzw=m['quaternion'],bias_z_m=c['bias_z_m'],controller_time_offset_s=d['alignment']['offset_s'])
 atomic_json(out/(n+'.json'),dict(take=n,normalization=c,metrics=metrics,warning='Derived OptiTrack arrays already normalized; do not apply bias a second time'))
 print(n,'upward_cm',-100*c['bias_z_m'],'complete_whip',metrics['complete_whip'],flush=True)
atomic_json(root/'runs/audits/adp1-normalization/results.json',dict(rows=rows,normalization='user-reviewed separate constant per take; drift remains',excluded=['whip_adp_1_005']))
cal=load_calibration(b);assert all(sha256(p)==h for p,h in cal['source_hashes'].items());print('All source hashes unchanged')

