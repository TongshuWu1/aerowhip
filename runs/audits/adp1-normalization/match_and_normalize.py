from pathlib import Path
import sys,json,importlib.util
import numpy as np
root=Path.cwd();sys.path.insert(0,str(root))
from experimental_data.adaptation_check import flight_names,sha256,find_rehearsal
from experimental_data.adaptation_rounds import read_optitrack,read_controller
from experimental_data.hover_calibration import calibrate_batch
from experimental_data.io import atomic_json
spec=importlib.util.spec_from_file_location('timing_audit',root/'runs/audits/20260909-adp0-hover-height/check_pairing.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
b=root/'rehearsal_csv_and_result_in_real_flight/20260908-195207-486249-seed655_best_validation/adp1';audit=root/'runs/audits/adp1-normalization'
entries={};rows=[]
for name in flight_names(b):
 mp=b/'flight_take'/f'{name}.csv';cp=b/'flight_take'/f'experiment_{name}.csv';m=read_optitrack(mp);c=read_controller(cp)
 xyz=module.match(m['time'],m['drone'],c);xy=module.match(m['time'],m['drone'],c,(0,1));z=module.match(m['time'],m['drone'],c,(2,))
 row=dict(take=name,drone=m['drone_label'],xyz=xyz,xy=xy,z=z);rows.append(row);print(json.dumps(row),flush=True)
 if xyz['rms_m']>.05 or abs(xyz['offset_s']-xy['offset_s'])>.03 or abs(xyz['offset_s']-z['offset_s'])>.03:raise ValueError('Review inconsistent measured-stream timing')
 entries[name]=dict(offset_s=xyz['offset_s'],source='User-authorized measured-stream XYZ time matching only; OptiTrack supplies all evaluation and normalization state. Includes logging latency; no spatial shift or command-motion alignment.',clock_verified=False,optitrack_sha256=sha256(mp),controller_sha256=sha256(cp))
atomic_json(audit/'timing_checks.json',dict(rows=rows));atomic_json(b/'time_alignment.json',entries)
cal=calibrate_batch(root,b);print(json.dumps(cal,indent=2),flush=True)
