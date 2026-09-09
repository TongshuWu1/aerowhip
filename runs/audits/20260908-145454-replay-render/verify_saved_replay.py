from pathlib import Path
import hashlib
import json
import os
import sys

os.environ['QT_QPA_PLATFORM']='windows'
os.environ['QT_API']='pyside6'
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
from PySide6.QtWidgets import QApplication
from simulator.gui.rehearsal_workspace import RehearsalWorkspace
from simulator.gui.viewer_3d import PointCableViewer3D

out=Path(__file__).resolve().parent
before=json.loads((out/'before.json').read_text())
run=Path(before['rehearsal'])
app=QApplication([])
page=RehearsalWorkspace(ROOT)
page.resize(1400,900)
page.load_result(run)
page.set_page_active(True)
page.show()
app.processEvents()
v=page.viewer
assert isinstance(v,PointCableViewer3D)
a=page.arrays
frames=[]
render=v.plotter.render
def observe(*args,**kwargs):
    m=v._drone_actor.GetMatrix()
    frames.append(np.array([[m.GetElement(i,j) for j in range(4)] for i in range(4)]))
    return render(*args,**kwargs)
v.plotter.render=observe
indices=list(range(0,len(a['prediction_time_s']),15))+[len(a['prediction_time_s'])-1,0]
checked=0
max_error=0.
for _ in range(2):
    for index in indices:
        frames.clear()
        page.draw_frame(index)
        assert len(frames)==1,len(frames)
        expected=np.eye(4)
        expected[:3,:3]=a['origin_rotations'][index]
        expected[:3,3]=a['origin_positions_m'][index]
        error=float(np.max(np.abs(frames[0]-expected)))
        assert error<1e-12,error
        max_error=max(error,max_error)
        np.testing.assert_allclose(v._cable_mesh.points,a['cable_positions_m'][index],atol=1e-12)
        checked+=1
page.draw_frame(110)
v.plotter.screenshot(str(out/'native-replay.png'))
page.shutdown();page.close();app.processEvents()
model=json.loads((run/'model.json').read_text())
offset=np.array(model['recorded_data']['optitrack_to_attachment_offset_body_m'])
expected_attachment=a['origin_positions_m']+np.einsum('nij,j->ni',a['origin_rotations'],offset)
attachment_error=float(np.max(np.linalg.norm(expected_attachment-a['cable_positions_m'][:,0],axis=-1)))
changed=[n for n,h in before['files'].items() if hashlib.sha256((run/n).read_bytes()).hexdigest()!=h]
assert not changed,changed
assert attachment_error<1e-10,attachment_error
report=dict(renderer=v.backend_name,checked_rendered_frames=checked,maximum_actor_matrix_error=max_error,
    complete_replay_passes=2,source_rehearsal=str(run),source_rehearsal_changes=changed,
    origin_attachment_geometry_max_error_m=attachment_error,
    old_bug_reproduction_false_jump_m=1.5005332385522154,
    test_result='4 targeted tests passed in 13.31 s, including native launcher and replay/scrubbing',
    scope='Display-only change; stored predictions, commands, policy, physics and training are unchanged.')
(out/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report),flush=True)
