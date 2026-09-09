"""Run the unchanged GPU rehearsal once and check its export-only full-state path."""
import sys,json,time,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from PySide6.QtCore import QCoreApplication
from simulator.gui.rehearsal_worker import RehearsalWorker
from deployment.fullstate_playback import load_reference
import numpy as np
app=QCoreApplication([])
source=ROOT/'runs/ppo/20260906-201957-294110-measured-mass-seed653'
directory=ROOT/'runs/audits/20260907-export-only-rehearsal'
directory.mkdir(exist_ok=False)
configs=[json.loads((source/f'{name}.json').read_text(encoding='utf-8')) for name in ('model','task','ppo')]
worker=RehearsalWorker(configs,source/'checkpoints/best_validation.pt',directory,fullstate_mode=True)
started=time.perf_counter()
events=[]
def ready(path,meta):
    events.append(dict(wall_seconds=time.perf_counter()-started,schema=meta['schema']))
    print(events[-1],flush=True)
    if meta['schema']=='frozen_open_loop_force_plan_v1':
        worker.command('execute')
worker.plan_ready.connect(ready)
worker.status.connect(lambda message:print(message,flush=True))
worker.failed.connect(lambda message:print('ERROR:',message,flush=True))
timer=threading.Timer(90,lambda:worker.command('stop'))
timer.start()
try:
    worker.run()
finally:
    timer.cancel()
rows,meta=load_reference(directory/'plan_001')
full=np.load(directory/'flight.npz')
exported=np.load(directory/'plan_001/fullstate_source.npz')
start=np.flatnonzero(np.isclose(full['time_s'],meta['recording_start_time_s'],atol=1e-9,rtol=0))[0]
np.testing.assert_array_equal(full['cable_node_position_world_m'][start:],exported['positions_m'])
np.testing.assert_array_equal(full['cable_node_velocity_world_m_s'][start:],exported['velocities_m_s'])
result=dict(events=events,wall_seconds=time.perf_counter()-started,sample_count=len(rows),
            total_duration_s=meta['total_duration_s'],source_arrays_exact=True)
(directory/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2),flush=True)
