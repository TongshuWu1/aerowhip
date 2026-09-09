"""Observe native VTK frame presentation, including replay rewinds."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.skipif(sys.platform != 'win32' or os.environ.get('WHIP_TEST_NATIVE_GUI') != '1',
                    reason='Requires native Windows Qt/VTK')
def test_rehearsal_presents_one_coherent_tracked_pose_per_frame(tmp_path):
    root = Path(__file__).resolve().parents[2]
    config = tmp_path/'config'; config.mkdir()
    for name in ('model', 'task', 'ppo'):
        shutil.copy2(root/f'config/research_30hz/{name}.json', config/f'{name}.json')
    folder = tmp_path/'rehearsal'; folder.mkdir()
    t = np.arange(5)/30
    origin = np.array([[.1+i*.03, -.2+i*.02, 1.55+i*.01] for i in range(5)])
    angles = np.linspace(-.3, .6, 5)
    rotation = np.array([[[np.cos(a), 0, np.sin(a)], [0, 1, 0],
                         [-np.sin(a), 0, np.cos(a)]] for a in angles])
    attachment = origin+np.einsum('nij,j->ni', rotation, [.006, -.012, -.055])
    cable = np.repeat(attachment[:,None], 12, axis=1)
    cable[:,:,2] -= np.linspace(0, .9525, 12)
    commands = np.zeros((5,11)); commands[:,:3] = origin
    target = [.94,.02,1.41]
    np.savez(folder/'rehearsal.npz', command_time_s=t, commands=commands,
        command_phase=np.ones(5), prediction_time_s=t, cable_positions_m=cable,
        origin_positions_m=origin, origin_rotations=rotation, force_time_s=t,
        virtual_force_n=np.zeros((5,3)), target_position_m=target)
    (folder/'task.json').write_text(json.dumps(dict(target_position_m=[1,0,1.4],
        desired_strike_direction_world=[.8,.6,0], success=dict(tip_target_distance_m=.05))))
    (folder/'rehearsal.json').write_text(json.dumps(dict(schema='research_fullstate_30hz_v1',
        checkpoint='saved.pt', checkpoint_sha256='a'*64, initial_tracking_origin_m=origin[0].tolist(),
        target_position_m=target, whip_end_s=.0667, total_duration_s=t[-1],
        predicted_valid_hit=False, minimum_tip_distance_m=1, recovery_prediction_complete=True,
        prediction_valid_through_s=t[-1])))
    script = r'''
import sys
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import QApplication
from simulator.gui.rehearsal_workspace import RehearsalWorkspace
from simulator.gui.viewer_3d import PointCableViewer3D
app = QApplication([])
page = RehearsalWorkspace(Path(sys.argv[1]))
page.resize(1200,800)
page.load_result(Path(sys.argv[2]))
page.active = True
page.ensure_viewer()
viewer = page.viewer
assert isinstance(viewer, PointCableViewer3D)
a = page.arrays
np.testing.assert_allclose(viewer._target, a['target_position_m'])
np.testing.assert_allclose(viewer._desired_direction, [.8,.6,0])
frames = []
native_render = viewer.plotter.render
def record_render(*args, **kwargs):
    matrix = viewer._drone_actor.GetMatrix()
    frames.append(np.array([[matrix.GetElement(i,j) for j in range(4)] for i in range(4)]))
    return native_render(*args, **kwargs)
viewer.plotter.render = record_render
page.draw_frame(0)
assert len(frames) == 1, len(frames)
actor = viewer.plotter.renderer.actors['TrackedOrigin']
def camera_values():
    c = viewer.plotter.camera
    return np.array([c.GetPosition(),c.GetFocalPoint(),c.GetViewUp()])
camera = camera_values()
for index in [0,1,3,4,0,4,2,0]*3:
    frames.clear()
    page.draw_frame(index)
    assert len(frames) == 1, len(frames)
    expected = np.eye(4)
    expected[:3,:3] = a['origin_rotations'][index]
    expected[:3,3] = a['origin_positions_m'][index]
    np.testing.assert_allclose(frames[0], expected, atol=1e-12)
    np.testing.assert_allclose(viewer._cable_mesh.points, a['cable_positions_m'][index], atol=1e-12)
    np.testing.assert_allclose(page._origin_mesh.points[0], a['origin_positions_m'][index], atol=1e-12)
    np.testing.assert_allclose(camera_values(), camera, atol=1e-12)
    assert viewer.plotter.renderer.actors['TrackedOrigin'] is actor
# Switching back to the legacy view cannot retain the tracked user matrix.
q = a['cable_positions_m'][2]
frames.clear()
viewer.update_state(q,np.zeros(3),q[:1],q[-1:])
assert len(frames) == 1
np.testing.assert_allclose(frames[0][:3,:3], np.eye(3), atol=1e-12)
np.testing.assert_allclose(frames[0][:3,3], q[0], atol=1e-12)
page.shutdown();page.close();app.processEvents()
print('24 replay/scrub frames: one correct pose per draw; persistent actors/camera; legacy reset passed.')
'''
    result = subprocess.run([sys.executable,'-c',script,str(tmp_path),str(folder)],cwd=root,
        env=dict(os.environ,QT_QPA_PLATFORM='windows',QT_API='pyside6'),
        capture_output=True,text=True,timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr
