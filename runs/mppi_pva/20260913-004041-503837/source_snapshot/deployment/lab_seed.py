"""Verified private M0/preliminary transfer; never fit, plan, or contact hardware.

The seed retains original provenance separately and derives only portable paths.
All runtime paths in its manifest are relative to the deployment repository.
"""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import tempfile
import zipfile

SCHEMA = 'deployment_lab_seed_v1'
BASE = 'workspace/baseline'
M0_REHEARSAL = 'runs/rehearsals_pva/20260910-022818-648386-M0-development-whip'
PRELIMINARY = 'runs/adaptation/20260909-preliminary1-M0-v2'
M0_SIGNATURE = 'fc854eae6a37bd8cd3457f1eb7eeb3ba52ae8e0b1315b8fdcc6534a5f94b2858'
PATH_KEYS = ('m0_model', 'preliminary', 'm0_rehearsal', 'mppi_settings')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(value):
    """Reject traversal and both Windows/POSIX absolute archive paths."""
    if not isinstance(value, str) or '\\' in value:
        raise ValueError('Expected a portable relative path')
    path = PurePosixPath(value)
    if not value or path.is_absolute() or PureWindowsPath(value).drive or '..' in path.parts:
        raise ValueError('Unsafe relative path: ' + value)
    if any('fig8vertical_002' in part.lower() for part in path.parts):
        raise ValueError('Protected recording is excluded from deployment seeds')
    return Path(*path.parts)


def model_identity(path):
    """Same content identity as model_evaluation, without importing numerical code."""
    def walk(value, base):
        if isinstance(value, list):
            return [walk(item, base) for item in value]
        if not isinstance(value, dict):
            return value
        result = {}
        for key, item in value.items():
            if key in ('provenance', 'sha256'):
                continue
            if key == 'checkpoint' and isinstance(item, str) and item:
                asset = Path(item)
                asset = asset if asset.is_absolute() else base / asset
                result[key] = walk(read(asset), asset.parent) if asset.suffix == '.json' else sha256(asset)
            else:
                result[key] = walk(item, base)
        return result
    path = Path(path)
    return hashlib.sha256(json.dumps(walk(read(path), path.parent), sort_keys=True,
                                      allow_nan=False).encode()).hexdigest()


def portable_model(source, destination):
    """Copy model assets with local paths and verify exact model identity."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Choose a new model destination')
    model = deepcopy(read(source))
    assets = destination.parent / 'assets'
    assets.mkdir(parents=True, exist_ok=False)
    for key, filename in (('motion_residual', 'cable_residual.pt'),
                          ('fullstate_execution', 'drone_model.json')):
        spec = model.get(key, {})
        if not spec.get('enabled'):
            continue
        original = Path(spec['checkpoint'])
        original = original if original.is_absolute() else source.parent / original
        if sha256(original) != spec['sha256']:
            raise ValueError('Model component checksum mismatch: ' + key)
        target = assets / filename
        if key == 'fullstate_execution':
            drone = read(original)
            residual = Path(drone['residual']['checkpoint'])
            residual = residual if residual.is_absolute() else original.parent / residual
            if sha256(residual) != drone['residual']['sha256']:
                raise ValueError('Drone residual checksum mismatch')
            shutil.copy2(residual, assets / 'drone_residual.pt')
            drone['residual']['checkpoint'] = 'drone_residual.pt'
            write(target, drone)
        else:
            shutil.copy2(original, target)
        spec.update(checkpoint='assets/' + filename, sha256=sha256(target))
    write(destination, model)
    signature = model_identity(source)
    if model_identity(destination) != signature:
        raise ValueError('Portable copy changed model identity')
    return signature


def _verify_files(folder, files):
    folder = Path(folder).resolve()
    for name, expected in files.items():
        path = folder / safe_relative(name)
        if path.is_symlink() or not path.resolve().is_relative_to(folder):
            raise ValueError('Linked or escaped seed asset: ' + name)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError('Seed checksum mismatch: ' + name)


def verify_seed(folder):
    folder = Path(folder).resolve()
    manifest = read(folder / 'manifest.json')
    if manifest.get('schema') != SCHEMA:
        raise ValueError('Unsupported deployment seed')
    for key in PATH_KEYS:
        name = safe_relative(manifest[key]).as_posix()
        if not name.startswith(BASE + '/'):
            raise ValueError('Seed runtime path must be inside ' + BASE)
    _verify_files(folder, manifest['files'])
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    if actual != set(manifest['files']) | {'manifest.json'}:
        raise ValueError('Unexpected or missing seed files')
    if model_identity(folder / 'model/model.json') != manifest['m0_signature']:
        raise ValueError('M0 identity differs from the seed manifest')
    for field, name in (('command_sha256', 'rehearsal/fullstate_30hz.csv'),
                        ('forecast_sha256', 'rehearsal/rehearsal.npz')):
        if sha256(folder / name) != manifest[field]:
            raise ValueError('Seed ' + field + ' disagrees with the retained artifact')
    protocol = read(folder / 'preliminary/protocol.json')
    if protocol['source_batch'] != BASE + '/preliminary/raw':
        raise ValueError('Preliminary sources must remain local to the retained baseline')
    for name, expected in read(folder / 'preliminary/protected_before.json').items():
        relative = safe_relative(name).as_posix()
        if not relative.startswith(BASE + '/preliminary/raw/'):
            raise ValueError('Preliminary hash map references a nonlocal source')
        if sha256(folder / relative[len(BASE) + 1:]) != expected:
            raise ValueError('Preliminary raw source checksum mismatch')
    return manifest


def export_seed(source_root, destination, *, m0_rehearsal=M0_REHEARSAL,
                preliminary=PRELIMINARY, expected_signature=M0_SIGNATURE, archive=True):
    """Make a new private seed using allowlisted M0/preliminary sources only."""
    root, output = Path(source_root).resolve(), Path(destination).resolve()
    if output.exists() or Path(str(output) + '.zip').exists():
        raise FileExistsError('Existing seeds are immutable; choose a new destination')
    rehearsal, pre = root / safe_relative(m0_rehearsal), root / safe_relative(preliminary)
    protocol = read(pre / 'protocol.json')
    if protocol.get('normalization_applied') is not False:
        raise ValueError('Expected the retained raw-coordinate preliminary preparation')
    original_model = rehearsal / 'model.json'
    signature = model_identity(original_model)
    if expected_signature is not None and signature != expected_signature:
        raise ValueError('Retained M0 identity does not match the explicitly selected baseline')
    output.mkdir(parents=True)
    sources = {}

    def copy(source, relative):
        source = Path(source)
        if source.is_symlink():
            raise ValueError('Linked seed sources are not copied')
        source = source.resolve()
        if not source.is_relative_to(root):
            raise ValueError('Seed source must be an ordinary file inside the source repository')
        safe_relative(source.relative_to(root).as_posix())
        target = output / safe_relative(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha256(source)
        shutil.copy2(source, target)
        if sha256(target) != before or sha256(source) != before:
            raise ValueError('Source changed during transfer')
        sources[source.relative_to(root).as_posix()] = before

    portable_model(original_model, output / 'model/model.json')
    portable_model(original_model, output / 'rehearsal/model.json')
    portable_model(original_model, output / 'planner/model.json')
    for name in ('model.json', 'settings.json', 'rehearsal.json'):
        copy(rehearsal / name, 'provenance/rehearsal/' + name)
    for file in (rehearsal / 'assets').glob('*'):
        if file.is_file():
            copy(file, 'provenance/rehearsal/assets/' + file.name)
    for name in ('fullstate_30hz.csv', 'rehearsal.npz', 'task.json', 'jerk_30hz.csv',
                 'plan.npz', 'proposal_baselines.npz', 'wave_reference.npz'):
        if (rehearsal / name).is_file():
            copy(rehearsal / name, 'rehearsal/' + name)
    metadata = read(rehearsal / 'rehearsal.json')
    if sha256(rehearsal / 'fullstate_30hz.csv') != metadata['csv_sha256']:
        raise ValueError('M0 frozen CSV checksum mismatch')
    metadata.update(job=BASE + '/planner', checkpoint=None,
                    deployment_provenance='Original metadata retained in ../provenance/rehearsal')
    write(output / 'rehearsal/rehearsal.json', metadata)
    settings = read(rehearsal / 'settings.json')
    settings.update(model_path=BASE + '/model/model.json', device='auto')
    write(output / 'rehearsal/settings.json', settings)
    write(output / 'planner/settings.json', settings)
    for name in ('plan.npz', 'proposal_baselines.npz', 'wave_reference.npz'):
        if (rehearsal / name).is_file():
            copy(rehearsal / name, 'planner/' + name)
    # Exact originating implementation, never a new rollout or current-source substitution.
    original_job = Path(read(rehearsal / 'rehearsal.json')['job'])
    if not original_job.is_absolute():
        original_job = root / original_job
    # Rehearsals retain the winning plan, while the optimizer additionally needs
    # its complete-horizon editable incumbent from the originating job.
    if not (original_job / 'initial_proposal.npz').is_file():
        raise ValueError('The original complete-horizon M0 planning incumbent is unavailable')
    copy(original_job / 'initial_proposal.npz', 'planner/initial_proposal.npz')
    for name in ('initial_proposal_provenance.json', 'source_manifest.json'):
        if (original_job / name).is_file():
            copy(original_job / name, 'provenance/planner/' + name)
    snapshot = original_job / 'source_snapshot'
    if not snapshot.is_dir() or not snapshot.resolve().is_relative_to(root):
        raise ValueError('The originating M0 source snapshot is unavailable')
    for file in sorted(snapshot.rglob('*.py')):
        copy(file, 'planner/source_snapshot/' + file.relative_to(snapshot).as_posix())
    takes = protocol['training_takes'] + protocol['validation_takes']
    if len(set(takes)) != len(takes):
        raise ValueError('Preliminary roles overlap')
    raw_source = Path(protocol['source_batch'])
    if not raw_source.is_absolute():
        raw_source = root / raw_source
    original_hashes = read(pre / 'protected_before.json')
    derived_hashes = {}
    for name in ('protocol.json', 'windows.json', 'preparation.json', 'protected_before.json'):
        copy(pre / name, 'provenance/preliminary/' + name)
    for name in ('windows.json', 'preparation.json'):
        copy(pre / name, 'preliminary/' + name)
    for take in takes:
        safe_relative(take)
        provenance = read(pre / 'inputs' / take / 'provenance.json')
        for name in ('data.npz', 'provenance.json'):
            copy(pre / 'inputs' / take / name, 'preliminary/inputs/' + take + '/' + name)
        for filename, expected in provenance['source_hashes'].items():
            safe_relative(filename)
            if len(PurePosixPath(filename).parts) != 1:
                raise ValueError('Preliminary source filenames must be basenames')
            source = raw_source / filename
            if original_hashes.get(str(source)) != expected or sha256(source) != expected:
                raise ValueError('Preliminary raw provenance mismatch: ' + filename)
            relative = 'preliminary/raw/' + filename
            copy(source, relative)
            derived_hashes[BASE + '/' + relative] = expected
    protocol['source_batch'] = BASE + '/preliminary/raw'
    write(output / 'preliminary/protocol.json', protocol)
    write(output / 'preliminary/protected_before.json', derived_hashes)
    write(output / 'provenance/source_files.json', sources)
    manifest = dict(schema=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
                    scope='Private retained M0 and preliminary seed; no old M1/M2 flights',
                    m0_model=BASE + '/model/model.json', preliminary=BASE + '/preliminary',
                    m0_rehearsal=BASE + '/rehearsal', mppi_settings=BASE + '/planner/settings.json',
                    m0_signature=signature, command_sha256=sha256(output / 'rehearsal/fullstate_30hz.csv'),
                    forecast_sha256=sha256(output / 'rehearsal/rehearsal.npz'),
                    preliminary_training_takes=protocol['training_takes'],
                    preliminary_validation_takes=protocol['validation_takes'],
                    retained_m0_refitted=False, flight_ready=False,
                    transformations=['Model component paths made relative; model identity verified unchanged',
                        'Runtime preliminary source paths relocated; original metadata preserved under provenance',
                        'Planner recipe retained from frozen M0; only device and model location changed'],
                    files={p.relative_to(output).as_posix(): sha256(p)
                           for p in sorted(output.rglob('*')) if p.is_file()})
    write(output / 'manifest.json', manifest)
    verify_seed(output)
    if archive:
        archive_path = Path(str(output) + '.zip')
        with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED) as stream:
            for file in sorted(output.rglob('*')):
                if file.is_file():
                    stream.write(file, file.relative_to(output).as_posix())
        write(Path(str(archive_path) + '.sha256.json'), dict(sha256=sha256(archive_path), file=archive_path.name))
    return manifest


def import_seed(seed, repository):
    """Verify before installation; never overwrite an existing retained baseline."""
    seed, root = Path(seed).resolve(), Path(repository).resolve()
    destination = root / BASE
    if destination.exists():
        raise FileExistsError('A retained baseline already exists; it will not be overwritten')
    if seed.is_dir():
        verify_seed(seed)
        shutil.copytree(seed, destination)
    else:
        with tempfile.TemporaryDirectory(prefix='whip-seed-') as tmp:
            folder = Path(tmp)
            with zipfile.ZipFile(seed) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)):
                    raise ValueError('Duplicate seed archive entries')
                for member in archive.infolist():
                    relative = safe_relative(member.filename)
                    if member.is_dir():
                        continue
                    if (member.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError('Linked seed archive entry')
                    target = folder / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, target.open('xb') as out:
                        shutil.copyfileobj(source, out)
            verify_seed(folder)
            shutil.copytree(folder, destination)
    return load_baseline(root)


def load_baseline(repository):
    """Return the verified manifest; missing seed raises FileNotFoundError."""
    return verify_seed(Path(repository).resolve() / BASE)


def resolve_baseline(repository):
    root = Path(repository).resolve()
    manifest = load_baseline(root)
    return {**manifest, **{key: root / safe_relative(manifest[key]) for key in PATH_KEYS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export')
    export.add_argument('--source', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    load = commands.add_parser('import')
    load.add_argument('seed', type=Path)
    load.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    verify = commands.add_parser('verify')
    verify.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = (export_seed(args.source, args.output) if args.command == 'export' else
              import_seed(args.seed, args.root) if args.command == 'import' else load_baseline(args.root))
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
