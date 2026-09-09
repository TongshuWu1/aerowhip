"""Build a separate corrected copy; preserve the supplied CSV and old bundle."""
from pathlib import Path
import sys
import csv
import json
import hashlib
import shutil
import zipfile
import time
import numpy as np

root=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root))
from deployment.gentle_recovery import replace_recorded_recovery
from deployment.fullstate_playback import load_reference

source=root/'runs/rehearsals/20260907-221528-727626/plan_001'
supplied=Path('C:/Users/wts28/Downloads/fullstate_30hz.csv')
assert source.joinpath('fullstate_30hz.csv').read_bytes()==supplied.read_bytes()
destination=root/'policies/PPO-fullstate-gentle-recovery-20260907'
shutil.copytree(source,destination)  # Refuses an existing destination.
started=time.perf_counter()
meta=replace_recorded_recovery(destination)
elapsed=time.perf_counter()-started
shutil.copy2(root/'deployment/fullstate_playback.py',destination/'fullstate_playback.py')
rows,_=load_reference(destination)
old=supplied.read_bytes().splitlines()
new=(destination/'fullstate_30hz.csv').read_bytes().splitlines()
assert old[:26]==new[:26]
assert (destination/'recorded_pid_reference.csv').read_bytes()==supplied.read_bytes()
t=np.array([r['time_s'] for r in rows])
a=np.array([[r[k] for k in ('ax_m_s2','ay_m_s2','az_m_s2')] for r in rows])
direction=a+[0,0,9.80665]
direction/=np.linalg.norm(direction,axis=1)[:,None]
jumps=np.rad2deg(np.arccos(np.clip((direction[1:]*direction[:-1]).sum(1),-1,1)))
recovery=t[1:]>.8+1e-10
result=dict(csv=str(destination/'fullstate_30hz.csv'),source_unchanged=True,
    whip_rows_byte_identical=True,whip_rows=25,rows=len(rows),generation_wall_s=elapsed,
    first_recovery_direction_jump_deg=float(jumps[24]),
    max_recovery_sample_direction_jump_deg=float(jumps[recovery].max()),
    duration_s=float(t[-1]),recovery=meta['recovery'],
    selected_ppo_sha256=hashlib.sha256((root/'runs/ppo/20260906-201957-294110-measured-mass-seed653/checkpoints/best_validation.pt').read_bytes()).hexdigest())
(Path(__file__).parent/'gentle_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
(destination/'READ_GENTLE_RECOVERY.md').write_text(
    '# Gentle recovery reference\n\nUse fullstate_30hz.csv for the new reference. '
    'recorded_pid_reference.csv is provenance only: it contains the OLD abrupt recovery.\n\n'
    'All 25 whip rows are byte-identical to the supplied export. Smooth braking is followed '
    'by a 10-second return and a 3-second final hold. Keep full-state feedback enabled. '
    'Position, velocity and acceleration are generated together; no acceleration clipping.\n\n'
    'This fixes the reference discontinuity. It is not vehicle validation. Initial braking '
    'still requires tilt. Inspect recovery_reference.png and the recorded bounds in fullstate.json. '
    'The new recovery has no cable or attitude tracking simulation; the source rehearsal '
    'in fullstate_source.npz retains the original PID recovery.\n',encoding='utf-8')
with zipfile.ZipFile(destination.with_suffix('.zip'),'x',compression=zipfile.ZIP_DEFLATED) as archive:
    for path in destination.iterdir():
        if path.is_file():archive.write(path,path.name)
print(json.dumps(result,indent=2))
