from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from experimental_data.io import atomic_json, sha256_file
from simulator.workflow import read_json, prepare_training, apply_baseline

ROOT = Path(__file__).resolve().parents[2]


def test_recommended_fit_dispatch_and_complete_application(workspace):
    from experimental_data.io import canonical_json_hash
    from simulator.workflow import prepare_data_job, apply_recommended_baseline
    model=read_json(workspace/'config/model.json')
    directory,_=prepare_data_job(workspace,'fit',model)
    assert read_json(directory/'fit_config.json')['method']=='constrained_geometry_drag'
    candidate=deepcopy(model)
    candidate['recorded_data']['optitrack_to_attachment_offset_body_m']=[.006,-.012,-.055]
    candidate['cable'].update(external_drag_s_inv=.3,substeps=12)
    digest=canonical_json_hash(candidate)
    atomic_json(directory/'recommended_model.json',candidate)
    atomic_json(directory/'candidate_review.json',{'recommended':'recommended','model_sha256':digest})
    with pytest.raises(ValueError,match='changed since'):
        apply_recommended_baseline(workspace,directory,'stale')
    assert read_json(workspace/'config/model.json')==model
    version=apply_recommended_baseline(workspace,directory,digest)
    actual=read_json(workspace/'config/model.json')
    assert actual['recorded_data']==candidate['recorded_data']
    assert actual['cable']['external_drag_s_inv']==.3
    assert actual['cable']['substeps']==12
    assert read_json(workspace/'data/baselines'/version/'manifest.json')['reviewed_candidate_sha256']==digest


def test_calibration_review_controls_keep_saved_candidate(workspace,monkeypatch,tmp_path):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication,QFileDialog
    from simulator.gui.calibration_page import BaselinePage
    source=ROOT/'data/calibration_audits/20260905_constrained_fit'
    directory=workspace/'data/calibration_audits/review'
    directory.mkdir(parents=True)
    for name in ['recommended_model.json','candidate_review.json','evaluation.json','geometry_drag_seed_2s_predictions.npz']:
        shutil.copy2(source/name,directory/name)
    app=QApplication.instance() or QApplication([])
    page=BaselinePage(workspace);page.show();app.processEvents()
    assert page.steps.count()==3 and page.candidate==directory
    initial=read_json(workspace/'config/model.json')
    page.change_role('fig8_001',enabled=False)
    assert page.candidate==directory and page.apply_button.isEnabled()
    page.use_candidate_inputs()
    assert page.model_values()['cable']['external_drag_s_inv']==.3
    assert read_json(workspace/'config/model.json')==initial
    page.view.setCurrentIndex(1);app.processEvents()
    assert len(page.figure.axes)==3 and len(page.figure.axes[0].lines)==2
    destination=tmp_path/'export';destination.mkdir()
    monkeypatch.setattr(QFileDialog,'getExistingDirectory',lambda *a,**kw:str(destination))
    page.export_fit()
    assert (destination/'calibration_1.pdf').exists()
    assert (destination/'geometry_drag_seed_2s_predictions.npz').exists()
    launched=[]
    monkeypatch.setattr(page.job,'start',lambda directory,command:launched.append(directory))
    page.start_job('fit')
    assert read_json(launched[0]/'fit_config.json')['method']=='constrained_geometry_drag'
    page.steps.setEnabled(True)
    page.close();app.processEvents()


def test_fit_window_sampling_excludes_disabled_and_protected(tmp_path):
    from experimental_data.recommended_fit import select_windows
    import csv
    manifest={'takes':{name:{'role':role,'enabled':enabled} for name,role,enabled in
        [('fit','training',True),('val','validation',True),('disabled','training',False),('protected','untouched_test',True)]}}
    atomic_json(tmp_path/'dataset_manifest.json',manifest)
    for name in ['fit','val']:
        dest=tmp_path/'force_takes'/name;dest.mkdir(parents=True)
        velocity=np.zeros((1000,12,3));velocity[:,-1,0]=np.linspace(0,5,1000)
        np.savez(dest/'take.npz',state_valid=np.ones(1000,dtype=bool),cable_node_velocity_world_m_s=velocity)
    benchmark=select_windows(tmp_path)
    with (benchmark/'windows.csv').open() as file:rows=list(csv.DictReader(file))
    assert {row['take'] for row in rows}=={'fit','val'}
    assert all(int(row['start_frame'])>=20 for row in rows)
    assert all(row['method']=='causal_polynomial' for row in rows)


@pytest.fixture
def workspace(tmp_path):
    shutil.copytree(ROOT / 'config', tmp_path / 'config')
    (tmp_path / 'data').mkdir()
    shutil.copy2(ROOT / 'data/dataset_manifest.json', tmp_path / 'data/dataset_manifest.json')
    return tmp_path


def test_launch_snapshots_isolate_algorithms_and_future_edits(workspace):
    before = sha256_file(workspace / 'config/ppo.json')
    ppo, command = prepare_training(workspace, 'PPO', seed=17, episodes=16, batch=8, device='cpu',
                                    overrides={'learning_rate': .0002})
    sac, _ = prepare_training(workspace, 'SAC', seed=19, episodes=16, batch=4, device='cpu')
    assert read_json(ppo / 'launch_config/ppo.json')['seed'] == 17
    assert read_json(sac / 'launch_config/sac.json')['seed'] == 19
    assert sha256_file(workspace / 'config/ppo.json') == before
    assert read_json(ppo / 'launch_config/ppo.json')['validation']['every_episodes'] == 8
    pointer=read_json(workspace/'config/research_workspace.json',{})
    task_path=workspace/pointer.get('config_directory','config')/'task.json'
    task = read_json(task_path)
    original_target_x=task['target_position_m'][0]
    task['target_position_m'][0] = 2
    atomic_json(task_path, task)
    assert read_json(ppo / 'launch_config/task.json')['target_position_m'][0] == original_target_x
    assert '--config-directory' in command


def test_ui_selects_an_externally_started_run(workspace):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    from simulator.gui.training_workspace import AlgorithmTrainingPage
    directory,_=prepare_training(workspace,'PPO',seed=7,episodes=512,batch=512,device='cpu')
    atomic_json(directory/'status.json',{'status':'STARTING','pid':os.getpid(),'episodes':0,'target_episodes':512})
    app=QApplication.instance() or QApplication([])
    page=AlgorithmTrainingPage(workspace,'PPO')
    assert page.directory==directory
    # These controls configure a NEW run; selecting an old run must not silently
    # replace the current workspace's new-training defaults.
    from simulator.research_config import workspace_configs
    defaults=workspace_configs(workspace)[2]
    assert page.batch.value()==defaults['training']['collection_batch']
    assert page.episodes.value()==defaults['training']['requested_episodes']
    assert page.seed.value()==defaults['seed']
    assert not page.start_button.isEnabled()
    assert page.stop_button.isEnabled()
    assert page.progress.format().startswith('Starting')
    metadata=read_json(directory/'run.json')
    metadata['parent_training_episodes']=128
    atomic_json(directory/'run.json',metadata)
    atomic_json(directory/'validation_latest.json',{'training_episodes':128})
    page.refresh_results()
    assert page.progress.format().startswith('Running · 128 / 512')
    assert 'Collecting training batch' in page.progress.format()
    atomic_json(directory/'status.json',dict(status='RUNNING',pid=os.getpid(),episodes=128,
        target_episodes=512,collection_batch=256,stage='Planning force sequences',phase_step=23,phase_total=70))
    page.refresh_results()
    assert not page.batch_progress.isHidden()
    assert page.batch_progress.value()==328
    assert '23/70 steps' in page.batch_progress.format()
    assert '128 / 512 completed attempts' in page.progress.format()
    page.stop_training()
    assert (directory/'STOP_REQUESTED').exists()
    page.refresh_results()
    assert not page.stop_button.isEnabled()
    assert page.progress.format().startswith('Stopping')
    page.shutdown();page.close();app.processEvents()


def test_apply_baseline_versions_model_without_touching_raw_data(workspace):
    raw = workspace / 'data/raw_takes/take/recording.csv'
    raw.parent.mkdir(parents=True)
    raw.write_text('preserved raw samples')
    model = read_json(workspace / 'config/model.json')
    model['point_mass']['mass_kg'] = .17
    model['fullstate_execution'] = {'enabled': True, 'checkpoint': 'previous_geometry.pt'}
    version = apply_baseline(workspace, model)
    assert read_json(workspace / 'config/model.json')['point_mass']['mass_kg'] == .17
    assert 'fullstate_execution' not in read_json(workspace / 'config/model.json')
    assert (workspace / 'data/baselines' / version / 'model.json').exists()
    assert raw.read_text() == 'preserved raw samples'
    fit = workspace / 'data/job'
    atomic_json(fit / 'model.json', model)
    atomic_json(fit / 'fit_result.json', {'fitted_parameters': {'EI_n_m2': 4e-5, 'Cb_n_m2_s': .001}})
    changed = deepcopy(model)
    changed['point_mass']['mass_kg'] = .2
    with pytest.raises(ValueError, match='changed since fitting'):
        apply_baseline(workspace, changed, fit_directory=fit)
    apply_baseline(workspace, model, fit_directory=fit)
    assert read_json(workspace / 'config/model.json')['cable']['EI_n_m2'] == 4e-5
    assert 'fullstate_execution' not in read_json(workspace / 'config/model.json')


def test_validation_journal_keeps_actual_execution_and_reused_policy_identity(tmp_path, monkeypatch):
    from learning.deployment_rollout import evaluate_deployment
    from simulator.cable import DderState
    from simulator.cable.dder import DderModel
    from run_ppo import load_configs, _atomic_json, _atomic_checkpoint
    torch.set_num_threads(1)
    model, task, shared = load_configs()
    task['episode_duration_s'] = .2
    shared['deployment']['nominal_fraction'] = 1.
    shared['deployment']['recovery_duration_s'] = .03
    task['target_position_m'] = [.2, 0, task['initial_root_position_m'][2] - .9525]
    def dynamics(_self, state, *args, **kwargs):
        q = state.positions_m.clone()
        q[:, -1, 0] += .1
        v = torch.zeros_like(q)
        v[:, -1, 0] = 10
        return DderState(q, v)
    monkeypatch.setattr(DderModel, 'step_runtime', dynamics)
    calls = []
    class Agent:
        def deterministic_action(self, observation):
            calls.append(observation.clone())
            return torch.zeros((len(observation), 3))
    result = evaluate_deployment(model, task, shared, Agent(), episodes=2, batch_size=2, device=torch.device('cpu'))
    assert len(calls) == 1
    assert result['success_rate'] == 1
    assert result.recording[0]['positions_m'][-1, 0, -1, 0] > .2  # PID motion after contact
    assert len(result.trials) == 2
    for name, value in zip(('model', 'task', 'ppo'), (model, task, shared)):
        atomic_json(tmp_path / f'{name}.json', value)
    result['training_episodes'] = 2
    checkpoint = tmp_path / 'checkpoints/latest.pt'
    _atomic_checkpoint(checkpoint, {'episodes': 2, 'actor': {'weights': 1}})
    _atomic_json(tmp_path / 'validation_latest.json', result)
    first = read_json(tmp_path / 'validation/latest.json')
    frozen_hash = sha256_file(tmp_path / first['checkpoint'])
    _atomic_json(tmp_path / 'validation_latest.json', result)
    assert len((tmp_path / 'validation_history.jsonl').read_text().splitlines()) == 1
    reused = deepcopy(result)
    reused['training_episodes'] = 4
    _atomic_checkpoint(checkpoint, {'episodes': 4, 'actor': {'weights': 1}})
    _atomic_json(tmp_path / 'validation_latest.json', reused)
    second = read_json(tmp_path / 'validation/latest.json')
    assert first['evaluation_id'] == second['evaluation_id']
    assert second['evaluation_reused']
    assert second['evaluated_at_training_episodes'] == 2
    assert first['scenario_id'] == second['scenario_id']
    assert first['checkpoint_sha256'] != second['checkpoint_sha256']
    assert sha256_file(tmp_path / first['checkpoint']) == frozen_hash
    assert len((tmp_path / 'validation_history.jsonl').read_text().splitlines()) == 2
    from simulator.gui.research_widgets import export_learning
    export_learning(tmp_path, tmp_path / 'figures', 'PPO')
    assert (tmp_path / 'figures/ppo_learning.svg').stat().st_size > 1000
    assert (tmp_path / 'figures/source_validation_history.csv').exists()
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    from simulator.gui.training_workspace import PolicyViewport
    app = QApplication.instance() or QApplication([])
    viewport = PolicyViewport(ROOT, 'PPO')
    viewport.set_active(True)
    assert viewport.load_trial(tmp_path / first['replay'], first)
    np.testing.assert_array_equal(viewport.arrays['positions_m'], result.recording[0]['positions_m'])
    viewport.clear_trial()
    assert viewport.arrays is None and not viewport.play.isEnabled()
    viewport.timer.stop()
    viewport.viewer.close()
    viewport.close()
    app.processEvents()


@pytest.mark.parametrize('algorithm', ['ppo', 'sac'])
def test_real_short_training_writes_batch_validation_and_run_snapshot(workspace, monkeypatch, algorithm):
    import run_ppo
    import run_sac
    torch.set_num_threads(1)
    model = read_json(workspace / 'config/model.json')
    task = read_json(workspace / 'config/task.json')
    shared = read_json(workspace / 'config/ppo.json')
    task['episode_duration_s'] = .1
    # This short synthetic task cannot use the production one-second prior.
    shared['bootstrap']['enabled'] = False
    shared['deployment']['recovery_duration_s'] = .5
    shared['ppo'].update(hidden_dim=8, minibatch_transitions=2, update_epochs=1)
    shared['validation'].update(episodes=2, every_episodes=2)
    shared['update_guard'].update(validation_episodes=2)
    shared['deployment']['holdout_episodes'] = 2
    for name, value in (('model', model), ('task', task), ('ppo', shared)):
        atomic_json(workspace / 'config' / f'{name}.json', value)
    sac = read_json(workspace / 'config/sac.json')
    sac['bootstrap']['enabled'] = False
    sac['sac'].update(hidden_dim=8, replay_capacity=32, minibatch_transitions=2, updates_per_collection=1)
    sac['validation']['episodes'] = 2
    atomic_json(workspace / 'config/sac.json', sac)
    directory, command = prepare_training(workspace, algorithm, seed=23, episodes=2, batch=2, device='cpu')
    # Keep smoke artifacts and active-run pointers entirely in pytest's workspace.
    monkeypatch.setattr(run_ppo, 'write_active_run', lambda root, artifact: None)
    if algorithm == 'ppo':
        run_ppo.train(device_name='cpu', requested_episodes=2, batch_size=2, artifact=directory,
                      resume_checkpoint=None, config_directory=directory / 'launch_config')
    else:
        # Source hashes need the real source root, so intercept only the active pointer.
        original_write = Path.write_text
        def write(path, data, *args, **kwargs):
            if str(path).endswith('ACTIVE_RUN.txt') and ROOT in path.parents:
                return len(data)
            return original_write(path, data, *args, **kwargs)
        monkeypatch.setattr(Path, 'write_text', write)
        run_sac.run(directory, 2, device_name='cpu', batch_override=2,
                    config_directory=directory / 'launch_config')
    assert read_json(directory / f'{algorithm}.json')['seed'] == 23
    latest = read_json(directory / 'validation/latest.json')
    assert latest['training_episodes'] == 2
    assert len((directory / 'validation_history.jsonl').read_text().splitlines()) == 2
    with np.load(directory / latest['replay']) as data:
        assert data['positions_m'].shape[1:] == (1, 12, 3)


def test_historical_research_pages_and_parallel_plot_viewport(workspace):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    from simulator.gui.main_window import HistoricalResearchWindow as SimulatorMainWindow, PAGE_DEFINITIONS
    app = QApplication.instance() or QApplication([])
    window = SimulatorMainWindow(workspace, *[read_json(workspace / 'config' / name)
                                               for name in ('model.json', 'task.json', 'ppo.json')])
    window.show()
    assert window.main_tabs.count() == 7
    assert all('MPCC' not in title for title, _ in PAGE_DEFINITIONS)
    assert not hasattr(window, "sac_page")
    for index, page in ((2, window.training_page),):
        window.main_tabs.setCurrentIndex(index)
        app.processEvents()
        assert page.canvas.isVisible()
        assert page.viewport.viewer.isVisible()
        assert not page.run_latest_button.isEnabled()
        assert not page.export_button.isEnabled()
        assert len(page.figure.axes[0].lines) == 0
    assert window.model_page.tabs.tabText(0)=='Model library'
    assert not window.model_page.prepare.isEnabled()
    assert not window.model_page.job.running
    assert hasattr(window.recordings_page,'current')
    assert window.fullstate_page.controls_scroll.widgetResizable()
    window.close()
    app.processEvents()


def test_fitting_emits_measured_and_predicted_validation_window(workspace):
    from experimental_data.cable_fit import fit_pivot_cable
    torch.set_num_threads(1)
    data_root = workspace / 'fit_data'
    manifest = {'takes': {}, 'excluded_takes': {'fig8vertical_002': 'protected_untouched_test'}}
    for name, role in (('fig8_001', 'training'), ('fig8_003', 'validation')):
        destination = data_root / name
        destination.mkdir(parents=True)
        with np.load(ROOT / 'data/force_takes' / name / 'take.npz') as data:
            np.savez_compressed(destination / 'take.npz', **{key: data[key][:85] for key in data.files})
        manifest['takes'][name] = {'role': role, 'path': f'{name}/take.npz'}
    atomic_json(data_root / 'manifest.json', manifest)
    model = read_json(workspace / 'config/model.json')
    fit = read_json(workspace / 'config/cable_fit.json')
    fit.update(device='cpu', dtype='float64', validation_lead_times_s=[.1])
    fit['objective'].update(horizon_s=.1, stride_s=.1)
    fit['search'].update(grid_size_per_axis=2, passes=1,
                         EI_n_m2=[3e-5, 5e-5], Cb_n_m2_s=[.0008, .0012])
    fit['comparison_baseline'] = dict(boundary='current_one_position_node_pivot',
        EI_n_m2=model['cable']['EI_n_m2'], Cb_n_m2_s=model['cable']['Cb_n_m2_s'])
    atomic_json(workspace / 'config/cable_fit.json', fit)
    output = workspace / 'fit_result'
    result = fit_pivot_cable(model_path=workspace / 'config/model.json',
        fit_config_path=workspace / 'config/cable_fit.json', data_root=data_root,
        output_root=output, progress=lambda text: None)
    assert result['selection']['validation_used_for_selection'] is False
    with np.load(output / 'validation_window.npz') as data:
        assert data['measured_markers_m'].shape == (11, 10, 3)
        assert data['candidate_markers_m'].shape == (11, 10, 3)
        assert np.isfinite(data['candidate_markers_m']).all()
        metadata = json.loads(str(data['metadata']))
        assert metadata['role'] == 'validation'
        assert metadata['take'] == 'fig8_003'
