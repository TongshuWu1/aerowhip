import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PySide6.QtWidgets import QApplication
from simulator.workflow import prepare_training, read_json
from simulator.policy_library import list_policies, set_deleted
from simulator.gui.training_workspace import AlgorithmTrainingPage

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def library_root(tmp_path):
    shutil.copytree(ROOT/'config', tmp_path/'config')
    run = tmp_path/'runs/ppo/example'
    (run/'checkpoints').mkdir(parents=True)
    for name in ('model', 'task', 'ppo'):
        shutil.copy2(tmp_path/f'config/{name}.json', run/f'{name}.json')
    (run/'run.json').write_text(json.dumps(dict(display_name='Original policy')))
    (run/'status.json').write_text(json.dumps(dict(status='STOPPED', episodes=500)))
    import torch
    for name in ('best_validation', 'latest'):
        torch.save(dict(episodes=100 if name == 'best_validation' else 500), run/f'checkpoints/{name}.pt')
    return tmp_path, run


def test_named_new_and_continued_runs_preserve_source(library_root):
    root, source = library_root
    checkpoint = source/'checkpoints/best_validation.pt'
    original = checkpoint.read_bytes()
    new, command = prepare_training(root, 'PPO', seed=17, episodes=600, batch=8,
                                   device='cpu', run_name='Adaptation / first try')
    assert read_json(new/'run.json')['display_name'] == 'Adaptation / first try'
    assert '--resume-checkpoint' not in command
    continued, command = prepare_training(root, 'PPO', seed=18, episodes=600, batch=8,
        device='cpu', resume=checkpoint, run_name='Continue best')
    assert command[command.index('--resume-checkpoint')+1] == str(checkpoint)
    assert read_json(continued/'run.json')['display_name'] == 'Continue best'
    assert read_json(continued/'launch_config/model.json') == read_json(source/'model.json')
    assert checkpoint.read_bytes() == original


def test_library_delete_restore_and_exact_resume_without_launch(library_root, monkeypatch):
    app = QApplication.instance() or QApplication([])
    root, source = library_root
    page = AlgorithmTrainingPage(root, 'PPO')
    checkpoint = source/'checkpoints/best_validation.pt'
    original = checkpoint.read_bytes()
    page.library.table.selectRow(0)
    selected = page.library.selected()['path']
    emitted = []
    page.checkpoint_requested.connect(emitted.append)
    page.library.use.click()
    assert emitted == [selected]
    page.library.resume.click()
    assert page.resume_checkpoint == checkpoint
    assert page.tabs.currentIndex() == 0
    assert not page.job.running
    page.run_name.setText('Continue chosen best')
    page.live_scene.setChecked(False)  # Fixture is a legacy checkpoint, not an Isaac Lab native model.
    page.episodes.setValue(200)  # Above best checkpoint count, below latest count.
    launches = []
    monkeypatch.setattr(page.job, 'start', lambda directory, command: launches.append((directory, command)))
    page.resume_button.click()
    assert len(launches) == 1
    directory, command = launches[0]
    assert command[command.index('--resume-checkpoint')+1] == str(checkpoint)
    assert read_json(directory/'run.json')['display_name'] == 'Continue chosen best'
    assert checkpoint.read_bytes() == original
    page.library.protected_paths = lambda: [checkpoint]
    page.library.refresh()
    page.library.table.selectRow(0)
    assert not page.library.delete.isEnabled()
    page.library.protected_paths = lambda: []
    page.library.update_actions()
    page.library.delete.click()
    assert checkpoint.read_bytes() == original
    assert len(list_policies(root)) == 1
    page.library.show_deleted.setChecked(True)
    page.library.table.selectRow(0)
    assert page.library.restore.isEnabled()
    page.library.restore.click()
    assert len(list_policies(root)) == 2
    assert page.shutdown()
    page.close()
    app.processEvents()


def test_delete_rejects_outside_project(library_root):
    root, source = library_root
    with pytest.raises(ValueError):
        set_deleted(root, root/'unrelated.pt', True)


def test_fullstate_uses_exact_checkpoint_and_filters_deleted(library_root):
    from types import SimpleNamespace
    from PySide6.QtWidgets import QTabWidget
    from simulator.gui.rehearsal_workspace import RehearsalWorkspace
    from simulator.gui.main_window import SimulatorMainWindow
    app = QApplication.instance() or QApplication([])
    root, source = library_root
    training = AlgorithmTrainingPage(root, 'PPO')
    fullstate = RehearsalWorkspace(root)
    tabs = QTabWidget()
    tabs.addTab(training, 'PPO')
    tabs.addTab(fullstate, 'Full-state')
    window = SimpleNamespace(fullstate_page=fullstate, training_page=training, main_tabs=tabs)
    checkpoint = str((source/'checkpoints/latest.pt').resolve())
    SimulatorMainWindow.use_fullstate_checkpoint(window, checkpoint)
    assert fullstate.tabs.currentIndex()==1
    legacy=fullstate.legacy
    assert legacy.checkpoints.currentData() == checkpoint
    assert tabs.currentWidget() is fullstate
    assert not legacy.worker
    legacy.thread = object()
    other = str((source/'checkpoints/best_validation.pt').resolve())
    SimulatorMainWindow.use_fullstate_checkpoint(window, other)
    assert legacy.checkpoints.currentData() == checkpoint
    legacy.thread = None
    set_deleted(root, other, True)
    legacy.refresh_checkpoints()
    assert legacy.checkpoints.findData(other) == -1
    assert legacy.checkpoints.currentData() == checkpoint
    assert fullstate.shutdown() and training.shutdown()
    tabs.close()
    app.processEvents()
