"""The PPO button launches one Isaac-hosted trainer, without a replay viewer."""
import os
from pathlib import Path
import shutil
import sys


def test_training_button_launches_hosted_backend(tmp_path, monkeypatch):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    import simulator.gui.training_workspace as module
    from simulator.workflow import atomic_json
    root = Path(__file__).resolve().parents[2]
    shutil.copytree(root/'config', tmp_path/'config')
    atomic_json(tmp_path/'config/multidrone_viewer.json', {'isaac_python': sys.executable})
    app = QApplication.instance() or QApplication([])
    page = module.AlgorithmTrainingPage(tmp_path, 'PPO')
    page.setup_button.setChecked(True)
    prepared = []; launched = []; replay = []
    page.live_scene_requested.connect(replay.append)
    def prepare(*args, **kwargs):
        prepared.append(kwargs)
        return tmp_path/'run', ['isaac-python', 'train_ppo_isaaclab.py']
    monkeypatch.setattr(module, 'prepare_training', prepare)
    monkeypatch.setattr(page.job, 'start', lambda *args: launched.append(args))
    monkeypatch.setattr(page, 'refresh', lambda: None)
    try:
        assert page.live_scene.isChecked()
        page.start_training(False)
        assert len(launched) == 1 and not replay
        assert prepared[0]['training_backend']['type'] == 'isaaclab_model'
        assert not prepared[0]['live_scene']
        page.live_scene.setChecked(False)
        page.start_training(False)
        assert prepared[1]['training_backend'] == {'type': 'project'}
    finally:
        page.shutdown(); page.close(); app.processEvents()
