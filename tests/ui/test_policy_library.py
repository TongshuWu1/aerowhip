import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PySide6.QtWidgets import QApplication
from simulator.workflow import prepare_training, read_json
from simulator.policy_library import list_policies, set_deleted

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def library_root(tmp_path):
    shutil.copytree(ROOT/'config', tmp_path/'config')
    (tmp_path/'config/research_workspace.json').write_text(json.dumps(dict(config_directory='config')))
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




def test_delete_rejects_outside_project(library_root):
    root, source = library_root
    with pytest.raises(ValueError):
        set_deleted(root, root/'unrelated.pt', True)
