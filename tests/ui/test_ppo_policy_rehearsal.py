import os
from pathlib import Path
import numpy as np
import torch
from experimental_data.io import atomic_json,sha256_file
from simulator.workflow import read_json
from planning.pva_job import CHECKPOINT_SCHEMA,load_settings
from deployment.ppo_snapshot import checkpoint_reader,prepare_snapshot


def fake_run(root,monkeypatch):
    import deployment.ppo_snapshot as snapshots
    monkeypatch.setattr(snapshots,'freeze_model_assets',lambda model,directory:model)
    run=root/'runs/ppo_pva/live'
    atomic_json(run/'settings.json',load_settings(root,'ppo'))
    atomic_json(run/'identity.json',dict(name='Live policy'))
    atomic_json(run/'model.json',dict(motion_residual=dict(checkpoint='cable.pt'),fullstate_execution=dict(checkpoint='drone.json')))
    atomic_json(run/'status.json',dict(status='running',attempts=9999))
    (run/'source_snapshot/learning').mkdir(parents=True)
    (run/'source_snapshot/learning/pva_env.py').write_text('# original environment')
    (run/'checkpoints').mkdir()
    torch.save(dict(schema=CHECKPOINT_SCHEMA,attempts=2048),run/'checkpoints/latest.pt')
    for relative in ('deployment/pva_policy_preview.py','tools/rehearse_ppo_snapshot.py'):
        p=root/relative;p.parent.mkdir(exist_ok=True);p.write_text('# adapter')
    return run


def test_checkpoint_reader_allows_atomic_replacement_and_keeps_original_bytes(tmp_path):
    checkpoint=tmp_path/'latest.pt';checkpoint.write_bytes(b'old policy')
    replacement=tmp_path/'latest.tmp';replacement.write_bytes(b'new policy')
    with checkpoint_reader(checkpoint) as reader:
        replacement.replace(checkpoint)
        assert reader.read()==b'old policy'
    assert checkpoint.read_bytes()==b'new policy'


def test_running_policy_snapshot_freezes_checkpoint_and_original_source(tmp_path,monkeypatch):
    run=fake_run(tmp_path,monkeypatch);destination=tmp_path/'snapshot'
    before=(run/'status.json').read_bytes()
    provenance=prepare_snapshot(tmp_path,run,destination)
    assert provenance['checkpoint_attempts']==2048  # Saved weights, not the live status counter.
    torch.save(dict(schema=CHECKPOINT_SCHEMA,attempts=4096),run/'checkpoints/latest.pt')
    assert sha256_file(destination/'checkpoints/policy.pt')==provenance['checkpoint_sha256']
    assert provenance['checkpoint_sha256']!=sha256_file(run/'checkpoints/latest.pt')
    assert (destination/'source_snapshot/learning/pva_env.py').read_bytes()==(run/'source_snapshot/learning/pva_env.py').read_bytes()
    assert read_json(destination/'status.json')['status']=='snapshot'
    assert (run/'status.json').read_bytes()==before and not (run/'STOP').exists()


def test_progress_rehearsal_and_refresh_work_on_running_ppo_without_stopping(tmp_path,monkeypatch):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    from simulator.gui.pva_workspace import PVAPlannerPage
    app=QApplication.instance() or QApplication([])
    run=fake_run(tmp_path,monkeypatch);page=PVAPlannerPage(tmp_path,'ppo')
    calls=[];monkeypatch.setattr(page.rehearsal_job,'start',lambda directory,command:calls.append((directory,command)))
    page.rehearse_current()
    first=calls[-1][0]/'snapshot'
    assert page.preview_complete.isChecked()
    assert page.tabs.currentIndex()==3 and '--complete' in calls[-1][1]
    assert read_json(first/'policy_snapshot.json')['checkpoint_choice']=='latest.pt'
    torch.save(dict(schema=CHECKPOINT_SCHEMA,attempts=4096),run/'checkpoints/latest.pt')
    page.refresh_policy_rehearsal()
    second=calls[-1][0]/'snapshot'
    assert first!=second and read_json(first/'policy_snapshot.json')['checkpoint_attempts']==2048
    assert read_json(second/'policy_snapshot.json')['checkpoint_attempts']==4096
    assert not (run/'STOP').exists() and read_json(run/'status.json')['status']=='running'
    assert '--complete' in calls[-1][1]  # Refresh retains complete recovery mode.
    page.preview_complete.setChecked(False);page.rehearse_current()
    assert '--complete' not in calls[-1][1]  # Explicit diagnostic preview remains available.
    page.shutdown();page.close();app.processEvents()


def test_missing_live_checkpoint_has_useful_message_and_does_not_launch(tmp_path,monkeypatch):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    from simulator.gui.pva_workspace import PVAPlannerPage
    app=QApplication.instance() or QApplication([])
    run=fake_run(tmp_path,monkeypatch);page=PVAPlannerPage(tmp_path,'ppo')
    page.preview_checkpoint.setCurrentIndex(1);page.rehearse_current()
    assert 'Wait for a completed training update' in page.policy_preview_note.text()
    assert not page.rehearsal_job.running and not (run/'STOP').exists()
    page.shutdown();page.close();app.processEvents()


def test_immediate_failure_preview_is_viewable_but_cannot_export(tmp_path):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    from simulator.gui.rehearsal_workspace import RehearsalWorkspace
    app=QApplication.instance() or QApplication([])
    atomic_json(tmp_path/'rehearsal.json',dict(schema='pva_policy_preview_v1',preview_only=True,planner='PPO policy preview',
        initial_tracking_origin_m=[0,0,1],target_position_m=[1,0,1],whip_end_s=0,total_duration_s=0,
        predicted_valid_hit=False,minimum_tip_distance_m=1,outcome='infeasible attempt',
        recovery_prediction_complete=False,policy_snapshot=dict(source_run_name='Live',checkpoint_choice='latest.pt',checkpoint_attempts=2048)))
    np.savez(tmp_path/'rehearsal.npz',commands=np.zeros((1,11)),command_time_s=[0.],prediction_time_s=[0.],
        origin_positions_m=[[0,0,1]],origin_velocities_m_s=np.zeros((1,3)),cable_velocities_m_s=np.zeros((1,3,3)))
    page=RehearsalWorkspace(tmp_path,inspection_only=True);page.load_result(tmp_path)
    assert page.play.isEnabled() and not page.save.isEnabled() and not page.package.isEnabled()
    assert not page.views.isTabVisible(3) and 'infeasible attempt' in page.status.text()
    page.save_csv();page.export_package()  # No dialog or legacy export fallback.
    page.shutdown();page.close();app.processEvents()
