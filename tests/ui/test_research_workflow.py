from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from experimental_data.io import atomic_json, sha256_file
from simulator.workflow import read_json, apply_baseline

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
    # These independent force-backend checks must not inherit the live PVA pointer.
    atomic_json(tmp_path/'config/research_workspace.json',dict(config_directory='config'))
    (tmp_path / 'data').mkdir()
    shutil.copy2(ROOT / 'data/dataset_manifest.json', tmp_path / 'data/dataset_manifest.json')
    return tmp_path


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
