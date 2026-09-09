"""Opt-in desktop check; use a fresh process to avoid offscreen test settings."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(
    sys.platform != 'win32' or os.environ.get('WHIP_TEST_NATIVE_GUI') != '1',
    reason='Requires an interactive Windows desktop and WHIP_TEST_NATIVE_GUI=1',
)
def test_normal_launcher_displays_native_ppo_without_sac():
    script = r'''
import sys
import traceback
import run_simulation
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from simulator.gui import app
from simulator.gui.viewer_3d import PointCableViewer3D
import numpy as np

visited = []
errors = []
def fail(kind, error, tb):
    traceback.print_exception(kind, error, tb)
    errors.append(str(error))
    QApplication.instance().exit(1)
sys.excepthook = fail

class CheckedWindow(app.SimulatorMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps = iter([2, 1, 3, 4, 0, 2])
        assert self.main_tabs.count() == 5
        QTimer.singleShot(250, self.check_next)

    def check_next(self):
        index = next(self.steps, None)
        if index is None:
            self.close()
            return
        self.navigation_buttons[index].click()
        assert self.main_tabs.currentIndex() == index
        if index == 2:
            page = self.training_page
            viewer = page.viewport.viewer
            assert isinstance(viewer, PointCableViewer3D), type(viewer)
            assert viewer.plotter.ren_win.GetInteractor() is not None
            cable = np.asarray(viewer._cable_mesh.points).copy()
            for camera in ('Side XZ', 'Top XY', 'Front YZ', 'Perspective'):
                page.viewport.camera.setCurrentText(camera)
                viewer.update_state(cable, np.zeros(3), cable[:1], cable[-1:])
            print(f'Page {index}: {viewer.backend_name}', flush=True)
        if index == 4:
            assert self.fullstate_page.viewer is None
            assert not self.fullstate_page.job.running
            assert not self.fullstate_page.play.isEnabled()
        visited.append(index)
        QTimer.singleShot(200, self.check_next)

app.SimulatorMainWindow = CheckedWindow
sys.argv = ['run_simulation.py']
assert run_simulation.main() == 0
assert not errors, errors
assert visited == [2, 1, 3, 4, 0, 2], visited
print('Native launcher, five pages without SAC, camera updates and shutdown passed.', flush=True)
'''
    environment = dict(os.environ, QT_QPA_PLATFORM='windows', QT_API='pyside6')
    result = subprocess.run(
        [sys.executable, '-c', script],
        cwd=Path(__file__).resolve().parents[2],
        env=environment, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count('PYVISTA / VTK') == 2, result.stdout
