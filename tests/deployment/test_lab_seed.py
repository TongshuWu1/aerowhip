"""Transfer checks use synthetic byte assets, never laboratory recordings."""
import json
from pathlib import Path
import shutil
import zipfile

import pytest

from deployment.lab_seed import (BASE, M0_REHEARSAL, PRELIMINARY, export_seed,
    import_seed, load_baseline, model_identity, portable_model, read,
    resolve_baseline, safe_relative, sha256, write)


def seed_source(root):
    rehearsal = root / M0_REHEARSAL
    assets = rehearsal / 'assets'
    assets.mkdir(parents=True)
    (assets / 'drone_residual.pt').write_bytes(b'synthetic opaque checkpoint')
    write(assets / 'drone_model.json', dict(nominal=dict(parameters=dict(kp=2.)),
        residual=dict(checkpoint='drone_residual.pt', sha256=sha256(assets / 'drone_residual.pt'))))
    write(rehearsal / 'model.json', dict(cable=dict(length=1.), motion_residual=dict(enabled=False),
        fullstate_execution=dict(enabled=True, checkpoint=str(assets / 'drone_model.json'),
                                 sha256=sha256(assets / 'drone_model.json')),
        provenance=dict(note='Original synthetic provenance')))
    write(rehearsal / 'settings.json', dict(method='mppi', model_path=str(rehearsal / 'model.json'),
                                          device='cuda', reward=dict(keep=3.)))
    for name in ('fullstate_30hz.csv', 'rehearsal.npz', 'plan.npz', 'wave_reference.npz', 'proposal_baselines.npz'):
        (rehearsal / name).write_bytes(('original ' + name).encode())
    job = root / 'runs/mppi_pva/synthetic'
    write(rehearsal / 'rehearsal.json', dict(job=str(job), csv_sha256=sha256(rehearsal / 'fullstate_30hz.csv')))
    (job / 'source_snapshot').mkdir(parents=True)
    (job / 'initial_proposal.npz').write_bytes(b'original complete-horizon proposal')
    (job / 'source_snapshot/example.py').write_text('# preserved source\n')
    pre = root / PRELIMINARY
    raw = root / 'raw'
    raw.mkdir()
    hashes = {}
    for take in ('training', 'validation'):
        original = raw / (take + '.csv')
        original.write_bytes(b'synthetic original data; no numerical interpretation')
        hashes[str(original)] = sha256(original)
        folder = pre / 'inputs' / take
        write(folder / 'provenance.json', dict(source_hashes={original.name: sha256(original)}))
        (folder / 'data.npz').write_bytes(b'synthetic prepared data')
    write(pre / 'protocol.json', dict(normalization_applied=False, source_batch=str(raw),
        training_takes=['training'], validation_takes=['validation']))
    write(pre / 'windows.json', [dict(take='training', role='training')])
    write(pre / 'preparation.json', dict(note='immutable source preparation'))
    write(pre / 'protected_before.json', hashes)
    return rehearsal, pre


def test_export_import_relocates_without_modifying_originals(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    rehearsal, pre = seed_source(source)
    before = {str(p): sha256(p) for p in source.rglob('*') if p.is_file()}
    exported = tmp_path / 'seed'
    manifest = export_seed(source, exported, expected_signature=None)
    assert manifest['m0_signature'] == model_identity(rehearsal / 'model.json')
    repository = tmp_path / 'first-repository'
    import_seed(Path(str(exported) + '.zip'), repository)
    moved = tmp_path / 'relocated-repository'
    shutil.copytree(repository, moved)
    monkeypatch.chdir(moved)
    resolved = resolve_baseline(moved)
    assert resolved['m0_model'].is_relative_to(moved)
    assert model_identity(resolved['m0_model']) == manifest['m0_signature']
    preliminary = resolved['preliminary']
    protocol = read(preliminary / 'protocol.json')
    assert Path(protocol['source_batch']).is_dir()
    for name, digest in read(preliminary / 'protected_before.json').items():
        assert not Path(name).is_absolute()
        assert sha256(name) == digest
    assert (moved / BASE / 'provenance/preliminary/protocol.json').read_bytes() == (pre / 'protocol.json').read_bytes()
    assert resolved['m0_rehearsal'].joinpath('fullstate_30hz.csv').read_bytes() == rehearsal.joinpath('fullstate_30hz.csv').read_bytes()
    assert before == {str(p): sha256(p) for p in source.rglob('*') if p.is_file()}
    assert read(moved / BASE / 'planner/settings.json')['reward'] == {'keep': 3.}
    with pytest.raises(FileExistsError):
        import_seed(exported, moved)


def test_seed_rejects_mutation_before_installation(tmp_path):
    source = tmp_path / 'source'
    seed_source(source)
    exported = tmp_path / 'seed'
    export_seed(source, exported, expected_signature=None)
    (exported / 'rehearsal/fullstate_30hz.csv').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        import_seed(exported, tmp_path / 'repo')
    assert not (tmp_path / 'repo' / BASE).exists()


@pytest.mark.parametrize('path', ['../outside', '/absolute', 'C:/outside', 'C:\\outside',
                                   'raw/fig8vertical_002.csv'])
def test_seed_rejects_nonportable_and_protected_paths(path):
    with pytest.raises(ValueError):
        safe_relative(path)


def test_zip_traversal_is_rejected_without_writes_outside_staging(tmp_path):
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr('../escaped.txt', b'bad')
    with pytest.raises(ValueError, match='Unsafe'):
        import_seed(archive, tmp_path / 'repo')
    assert not (tmp_path / 'escaped.txt').exists()


def test_portable_model_and_freeze_resolve_the_owner_directory(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    rehearsal, _ = seed_source(source)
    portable = tmp_path / 'portable/model.json'
    identity = portable_model(rehearsal / 'model.json', portable)
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    from planning.pva_job import freeze_model_assets
    output = tmp_path / 'frozen'
    output.mkdir()
    frozen = freeze_model_assets(read(portable), output, source_root=portable.parent, portable=True)
    write(output / 'model.json', frozen)
    assert frozen['fullstate_execution']['checkpoint'] == 'assets/drone_model.json'
    assert model_identity(output / 'model.json') == identity


def test_expected_m0_identity_is_required_for_real_default(tmp_path):
    source = tmp_path / 'source'
    seed_source(source)
    with pytest.raises(ValueError, match='M0 identity'):
        export_seed(source, tmp_path / 'seed')
    assert not (tmp_path / 'seed').exists()


def test_full_prepare_accepts_portable_preliminary_provenance_keys(tmp_path, monkeypatch):
    """Exercise the native preliminary source-map join used before full fitting."""
    source = tmp_path / 'source'
    seed_source(source)
    exported = tmp_path / 'seed'
    export_seed(source, exported, expected_signature=None)
    repo = tmp_path / 'relocated repository'
    import_seed(exported, repo)
    monkeypatch.chdir(repo)
    from experimental_data import whip_full_data
    from experimental_data.whip_adaptation import SCHEMA
    # Source preparation only: no model rollout or optimizer is constructed.
    monkeypatch.setattr(whip_full_data, 'ROOT', repo)
    source_code=repo/'experimental_data/nested/module.py'
    source_code.parent.mkdir(parents=True)
    source_code.write_text('# portable source index\n',encoding='utf-8')
    native = repo / 'whip-prepared'
    portable_model(repo / BASE / 'model/model.json', native / 'source_candidate/model.json')
    write(native / 'protocol.json', dict(schema=SCHEMA, parent_generation=0,
          takes={'whip_001': {'role': 'adaptation'}}))
    write(native / 'review.json', {'review': 'synthetic'})
    write(native / 'source_hashes.json', {})
    write(native / 'prepared_hashes.json', {})
    (native / 'inputs/whip_001').mkdir(parents=True)
    (native / 'inputs/whip_001/data.npz').write_bytes(b'synthetic prepared whip')
    (native / 'source_snapshot').mkdir()
    (native / 'source_snapshot/example.py').write_text('# exact source')
    write(native / 'code_hashes.json', {'example.py': sha256(native / 'source_snapshot/example.py')})
    job = repo / 'full-prepared'
    baseline = resolve_baseline(repo)
    whip_full_data.prepare(job, native, baseline['preliminary'])
    assert read(job / 'status.json')['status'] == 'prepared'
    assert read(job / 'protocol.json')['full_update']['schema'] == whip_full_data.FULL_SCHEMA
    code=read(job/'code_hashes.json')
    assert code=={'experimental_data/nested/module.py':sha256(source_code)}
    assert all('\\' not in key for key in code)
    assert sha256(job/'source_snapshot/experimental_data/nested/module.py')==code['experimental_data/nested/module.py']
    sources = read(job / 'source_hashes.json')
    for name, digest in read(baseline['preliminary'] / 'protected_before.json').items():
        assert sources[name] == digest
        assert sha256(repo / name) == digest
