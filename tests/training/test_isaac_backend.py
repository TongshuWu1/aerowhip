"""Isaac launcher isolation and event-loop cancellation contracts."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest
from simulator.workflow import prepare_training, read_json
from simulator.research_config import workspace_configs
from experimental_data.io import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]


def workspace(tmp_path):
    configs = deepcopy(workspace_configs(ROOT))
    for name, value in zip(('model', 'task', 'ppo'), configs):
        atomic_json(tmp_path/'config'/f'{name}.json', value)
    return configs


def test_isaac_launcher_freezes_same_model_and_disables_replay(tmp_path):
    model, task, config = workspace(tmp_path)
    backend = dict(type='isaaclab_model', python=sys.executable, headless=True, render_stride=5)
    run, command = prepare_training(tmp_path, 'ppo', seed=7, episodes=8, batch=4,
                                    device='cuda', training_backend=backend)
    assert command[0] == str(Path(sys.executable).resolve())
    assert Path(command[2]) == run/'source_snapshot/tools/train_ppo_isaaclab.py'
    assert '--headless' in command and command[command.index('--batch-size')+1] == '4'
    saved = read_json(run/'launch_config/ppo.json')
    assert saved['training_backend'] == backend and not saved['live_scene']['enabled']
    assert saved['reward'] == config['reward'] and saved['ppo'] == config['ppo']
    assert read_json(run/'launch_config/task.json') == task
    actual = read_json(run/'launch_config/model.json')
    for key in ('cable', 'point_mass', 'recorded_data', 'simulation'):
        assert actual[key] == model[key]
    assert actual['motion_residual']['sha256'] == model['motion_residual']['sha256']
    for path, digest in read_json(run/'source_snapshot_manifest.json')['files'].items():
        assert sha256_file(run/'source_snapshot'/path) == digest
    assert read_json(run/'launch.json')['command'] == command


@pytest.mark.parametrize('backend,device,legacy', [
    ({'type': 'unknown'}, 'cuda', False),
    ({'type': 'isaaclab_model', 'python': sys.executable}, 'cpu', False),
    ({'type': 'isaaclab_model', 'python': sys.executable}, 'cuda', True),
    ({'type': 'isaaclab_model', 'python': 'missing-python.exe'}, 'cuda', False),
])
def test_invalid_backend_fails_before_creating_run(tmp_path, backend, device, legacy):
    model, _, _ = workspace(tmp_path)
    if legacy:
        model.pop('fullstate_execution')
        atomic_json(tmp_path/'config/model.json', model)
    with pytest.raises(ValueError):
        prepare_training(tmp_path, 'ppo', seed=7, episodes=8, batch=4,
                         device=device, training_backend=backend)
    assert not (tmp_path/'runs').exists()


def test_project_training_keeps_original_entrypoint(tmp_path):
    workspace(tmp_path)
    run, command = prepare_training(tmp_path, 'ppo', seed=7, episodes=8, batch=4,
                                    device='cuda', live_scene=False)
    assert Path(command[2]) == run/'source_snapshot/run_ppo.py'
    assert '--headless' not in command
    assert read_json(run/'launch_config/ppo.json')['training_backend']['type'] == 'project'


def test_event_pump_is_scoped_and_can_stop_training():
    from learning.training_control import set_runtime_pump, reset_runtime_pump, check_training_stop, TrainingStopped
    calls = []
    def stop():
        calls.append('stop')
        raise TrainingStopped()
    token = set_runtime_pump(stop)
    try:
        with pytest.raises(TrainingStopped): check_training_stop()
    finally:
        reset_runtime_pump(token)
    check_training_stop()
    assert calls == ['stop']


def test_pause_resumes_without_advancing_model_and_stop_is_cooperative(tmp_path):
    from types import SimpleNamespace
    from simulator.isaac_training import IsaacLabTrainingSession
    from learning.training_control import TrainingStopped
    session = IsaacLabTrainingSession.__new__(IsaacLabTrainingSession)
    session.artifact = tmp_path
    session.paused = True; session.stopped = False
    session.next_pump = 0.; session.next_status = float('inf')
    calls = []
    def update():
        calls.append('event')
        if len(calls) == 3: session.paused = False
    session.app = SimpleNamespace(is_running=lambda: True, update=update)
    session.pump()
    assert calls == ['event']*3
    session.request_stop()
    assert (tmp_path/'STOP_REQUESTED').exists()
    with pytest.raises(TrainingStopped): session.pump()


def test_display_freezes_drone_pose_with_terminal_and_failed_cable_rows():
    from types import SimpleNamespace
    import torch
    from simulator.isaac_training import IsaacLabTrainingSession
    session = IsaacLabTrainingSession.__new__(IsaacLabTrainingSession)
    session.in_collection = True; session.render_stride = 5
    session.execution_frames = 0; session.metrics = None
    seen = []
    session.present = lambda q, origin, *args: seen.append((q.clone(), origin.clone()))
    pose = {'position_origin_m': torch.arange(11.)[None, :, None].expand(3, 11, 3),
            'rotation_tracking_to_world': torch.eye(3).expand(3, 11, 3, 3)}
    score = SimpleNamespace(physics_dt_s=1/150, target=torch.zeros(3, 3),
                            episode_success=torch.tensor([True, False, False]), target_radius_m=.05)
    cutoffs = torch.tensor([2, 10, 10])
    for step in range(1, 11):
        # Row 0 hits at 2; row 1 fails at 3 and retains its state at 2.
        times = torch.tensor([min(step, 2), min(step, 2), step], dtype=torch.float32)
        state = SimpleNamespace(positions_m=times[:, None, None].expand(3, 2, 3))
        failed = torch.tensor([False, step >= 3, False])
        session.execution_step(score, state, pose, step, cutoffs, failed)
    assert session.execution_frames == 2
    for q, origin in seen:
        torch.testing.assert_close(q[:, 0], origin)
