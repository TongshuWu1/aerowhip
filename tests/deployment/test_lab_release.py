"""Clean application release and private seed boundaries, using synthetic assets."""
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

from deployment.lab_seed import BASE, export_seed, import_seed, load_baseline, sha256, write
from tools.build_lab_release import (ROOT_REQUIRED, REQUIRED_IMPLEMENTATION, build,
                                     verify_release)


def application(root):
    root.mkdir()
    for name in ROOT_REQUIRED + REQUIRED_IMPLEMENTATION:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic application source\n', encoding='utf-8')
    write(root / 'config/pva/mppi.json', dict(model_path='C:/old/workstation/model.json', device='cuda'))
    write(root / 'config/experiment.json', dict(fit_job='C:/old/fit'))
    write(root / 'config/evaluation/campaign.json', dict(models=['old private model'], flights=['old trial']))
    write(root / 'data/dataset_manifest.json', dict(takes={'private': 'source data'}))
    return root


def add_baseline(root, tmp_path):
    # Reuse the synthetic source factory; do not depend on local research assets.
    helper = Path(__file__).with_name('test_lab_seed.py')
    spec = importlib.util.spec_from_file_location('lab_seed_test_data', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / 'seed-source'
    module.seed_source(source)
    seed = tmp_path / 'seed'
    export_seed(source, seed, expected_signature=None)
    return import_seed(seed, root)


def test_source_release_is_clean_and_clears_local_selection(tmp_path):
    root = application(tmp_path / 'app')
    for name in ('experiments/private/study.json', 'exports/private/fullstate.csv',
                 'runs/failed/failed.log', '.git/private', '.venv/private',
                 'private_bundles/private.zip', 'workspace/current_study.json',
                 'deployment/__pycache__/private.pyc'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'must not be released')
    result = build(root, tmp_path / 'source-release')
    output = Path(result['directory'])
    manifest = verify_release(output)
    assert manifest['includes_private_baseline'] is False
    assert not (output / BASE).exists()
    assert json.loads((output / 'config/pva/mppi.json').read_text())['model_path'] == ''
    assert json.loads((output / 'config/experiment.json').read_text())['fit_job'] is None
    assert json.loads((output / 'config/evaluation/campaign.json').read_text())['flights'] == []
    assert json.loads((output / 'data/dataset_manifest.json').read_text())['takes'] == {}
    with zipfile.ZipFile(result['archive']) as archive:
        assert 'run_lab.py' in archive.namelist()
        assert 'tools/lab.py' in archive.namelist()
        assert not any(name.startswith(('runs/', 'experiments/', 'exports/', '.git/', '.venv/', 'private_bundles/'))
                       for name in archive.namelist())
        assert archive.getinfo('start_lab.sh').external_attr >> 16 & 0o111 == 0o111
    assert json.loads(Path(result['checksum']).read_text())['sha256'] == sha256(result['archive'])
    with pytest.raises(FileExistsError, match='immutable'):
        build(root, output)


def test_private_release_retains_verified_relocatable_seed(tmp_path):
    root = application(tmp_path / 'app')
    original = add_baseline(root, tmp_path)
    result = build(root, tmp_path / 'private-release', include_baseline=True)
    extracted = tmp_path / 'extracted elsewhere'
    with zipfile.ZipFile(result['archive']) as archive:
        archive.extractall(extracted)
    manifest = verify_release(extracted)
    assert manifest['includes_private_baseline'] is True
    assert load_baseline(extracted)['m0_signature'] == original['m0_signature']
    for name, digest in original['files'].items():
        assert sha256(extracted / BASE / name) == digest
    assert (extracted / BASE / 'planner/initial_proposal.npz').is_file()
    assert not (extracted / 'experiments').exists()


def test_release_rejects_absolute_runtime_configuration(tmp_path):
    root = application(tmp_path / 'app')
    write(root / 'config/model.json', dict(fullstate_execution=dict(checkpoint='/outside/model.json')))
    with pytest.raises(ValueError, match='Absolute configuration path'):
        build(root, tmp_path / 'bad-release')
    assert not (tmp_path / 'bad-release').exists()


def test_release_rejects_missing_or_modified_baseline(tmp_path):
    root = application(tmp_path / 'app')
    with pytest.raises(FileNotFoundError):
        build(root, tmp_path / 'missing-release', include_baseline=True)
    add_baseline(root, tmp_path)
    (root / BASE / 'model/assets/drone_residual.pt').write_bytes(b'modified')
    with pytest.raises(ValueError, match='checksum'):
        build(root, tmp_path / 'corrupt-release', include_baseline=True)


def test_release_verification_detects_modified_application(tmp_path):
    root = application(tmp_path / 'app')
    result = build(root, tmp_path / 'source-release')
    output = Path(result['directory'])
    (output / 'run_lab.py').write_text('changed code')
    with pytest.raises(ValueError, match='Release checksum mismatch'):
        verify_release(output)


def test_protected_paths_are_excluded_even_when_named_like_source(tmp_path):
    root = application(tmp_path / 'app')
    (root / 'tools/fig8vertical_002.py').write_text('# must remain excluded')
    with pytest.raises(ValueError, match='Protected recording'):
        build(root, tmp_path / 'bad-release')
