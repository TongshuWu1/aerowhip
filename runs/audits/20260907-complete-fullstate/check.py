"""Offline Windows/GPU check of complete export using the preserved selected plan."""
from pathlib import Path
import sys, json, shutil, time, hashlib, os
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from deployment.fullstate_recovery import prepare_complete_trajectory
from deployment.fullstate import write_fullstate
from deployment.fullstate_playback import load_reference

HERE = Path(__file__).resolve().parent
SOURCE = ROOT/'runs/rehearsals/20260906-222405-804438'
target = HERE/'plan_001'
target.mkdir(exist_ok=False)
torch.set_num_threads(1)
model = json.loads((SOURCE/'package/policy/model.json').read_text(encoding='utf-8'))
task = json.loads((SOURCE/'rehearsal_task.json').read_text(encoding='utf-8'))
metadata = json.loads((SOURCE/'plan_001/plan.json').read_text(encoding='utf-8'))
with np.load(SOURCE/'plan_001/fullstate_source.npz') as data:
    trajectory={k:data[k] for k in data.files}
print('Preparing unchanged selected strike + recovery on',torch.cuda.get_device_name(),flush=True)
started=time.perf_counter()
complete=prepare_complete_trajectory(trajectory,model,task['initial_root_position_m'],device='cuda')
print('Cable preview completed',time.perf_counter()-started,flush=True)
for name in ('plan.npz','plan.json','commands.csv','controller_force.csv','controller_acceleration.csv'):
    shutil.copyfile(SOURCE/'plan_001'/name,target/name)
meta=write_fullstate(complete,target,metadata)
rows,_=load_reference(target)
original=np.genfromtxt(SOURCE/'plan_001/fullstate_30hz.csv',delimiter=',',names=True)
fields=original.dtype.names
actual=np.array([[row[k] for k in fields] for row in rows[:len(original)]])
np.testing.assert_array_equal(actual,np.column_stack([original[k] for k in fields]))
record=dict(device=torch.cuda.get_device_name(),os=sys.platform,wall_seconds=time.perf_counter()-started,
            sample_count=len(rows),total_duration_s=meta['total_duration_s'],recovery=meta['recovery'],
            peak_speed_m_s=meta['peak_speed_m_s'],peak_acceleration_m_s2=meta['peak_acceleration_m_s2'],
            position_min_m=meta['position_min_m'],position_max_m=meta['position_max_m'],
            unchanged_whip_rows=len(original),simulation_only=True)
(HERE/'result.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2),flush=True)
