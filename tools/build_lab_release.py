"""Build a clean offline lab application archive, optionally with the private M0 seed.

No experiment execution, historical trials, Git metadata, or local environments
are copied. Existing output directories and archives are never overwritten.
"""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path, PureWindowsPath
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deployment.lab_seed import BASE, load_baseline, read, safe_relative, sha256, write
from tools.build_source_release import portable_guide_links

PACKAGES = ('deployment', 'simulator', 'planning', 'learning', 'experimental_data', 'tools', 'tests')
SOURCE_SUFFIXES = {'.py', '.svg', '.cu', '.cuh', '.cpp', '.h'}
ROOT_REQUIRED = ('README.md', 'run_lab.py', 'setup_lab.py', 'start_lab.cmd', 'start_lab.sh',
                 'requirements.txt', 'pytest.ini')
ROOT_OPTIONAL = ('AGENTS.md', 'HANDOFF.md', 'SOURCE_SNAPSHOT.json', 'SOURCE_INTEGRATION.json', '.editorconfig',
                 '.gitattributes', '.github/workflows/smoke.yml',
                 'run_simulation.py', 'run_ppo.py', 'run_sac.py', 'run_tests.py')
CONFIGS = ('model.json', 'task.json', 'ppo.json', 'sac.json', 'cable_fit.json', 'baseline.json',
           'current_vehicle.json', 'research_workspace.json', 'experiment.json',
           'research_30hz/model.json', 'research_30hz/task.json', 'research_30hz/ppo.json',
           'pva/mppi.json', 'pva/ppo.json')
REQUIRED_IMPLEMENTATION = ('deployment/lab_gui.py', 'deployment/lab_seed.py', 'tools/check_lab.py',
                           'deployment/lab_workflow.py', 'tools/lab.py')
RELEASE_SCHEMA = 'deployment_lab_release_v1'


def _git_revision(root):
    if not (root / '.git').exists():
        return None
    try:
        result = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _local_configuration(value, route='config'):
    """A source release must not depend on any developer workstation path."""
    if isinstance(value, dict):
        for key, item in value.items():
            _local_configuration(item, route + '.' + key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _local_configuration(item, route + '[' + str(index) + ']')
    elif isinstance(value, str) and value:
        # Full narrative strings and URLs are not filesystem references.
        if value.startswith('/') or PureWindowsPath(value).drive:
            raise ValueError('Absolute configuration path at ' + route)


def _clean_configs(root):
    configurations = {}
    for name in CONFIGS:
        path = root / 'config' / name
        if path.is_file():
            configurations[name] = deepcopy(read(path))
    # No selected model or populated campaign is inherited from the developer.
    configurations['experiment.json'] = dict(schema='unseen_system_experiment_v1',
        preliminary_batch=None, fit_job=None, status='Open the lab application to create a study from the retained baseline.')
    configurations['evaluation/campaign.json'] = dict(schema='model_evolution_v1', models=[], flights=[])
    configurations['research_workspace.json'] = dict(schema='research_workspace_v1',
        config_directory='config/research_30hz', bundle=None,
        model_label='Unfitted structural template', selected_by_user=False)
    for method in ('mppi', 'ppo'):
        settings = configurations.get('pva/' + method + '.json')
        if settings is not None:
            settings['model_path'] = ''
            settings['device'] = 'auto'
            settings.pop('development_model_review', None)
    for name, value in configurations.items():
        _local_configuration(value, 'config/' + name)
    return configurations


def _validate_baseline_runtime(folder):
    """Preserved provenance may name old sources; active asset paths may not."""
    baseline = load_baseline(folder)
    root = folder / BASE
    for name in ('model/model.json', 'rehearsal/model.json', 'planner/model.json'):
        model = read(root / name)
        owner = (root / name).parent
        for field in ('motion_residual', 'fullstate_execution'):
            spec = model.get(field, {})
            if not spec.get('enabled'):
                continue
            relative = safe_relative(spec['checkpoint'])
            component = owner / relative
            if not component.resolve().is_relative_to(root.resolve()):
                raise ValueError('Model component escapes the retained baseline')
            if field == 'fullstate_execution':
                drone = read(component)
                child = component.parent / safe_relative(drone['residual']['checkpoint'])
                if not child.resolve().is_relative_to(root.resolve()) or not child.is_file():
                    raise ValueError('Drone residual is not a local retained asset')
    for name in ('rehearsal/settings.json', 'planner/settings.json'):
        settings = read(root / name)
        model_path = safe_relative(settings['model_path'])
        if not (folder / model_path).is_file():
            raise ValueError('Baseline planner model is missing')
        reference = settings.get('trajectory_objective', {}).get('reference_file')
        if reference and not ((root / name).parent / safe_relative(reference)).is_file():
            raise ValueError('Baseline planner reference is missing')
    for name in ('planner/initial_proposal.npz', 'planner/proposal_baselines.npz'):
        if not (root / name).is_file():
            raise ValueError('Baseline planning input is missing: ' + name)
    return baseline


def verify_release(folder):
    folder = Path(folder).resolve()
    manifest = read(folder / 'LAB_RELEASE_MANIFEST.json')
    if manifest.get('schema') != RELEASE_SCHEMA:
        raise ValueError('Unsupported lab release manifest')
    for name, expected in manifest['files'].items():
        file = folder / safe_relative(name)
        if file.is_symlink() or not file.resolve().is_relative_to(folder):
            raise ValueError('Linked release file')
        if not file.is_file() or file.stat().st_size != expected['bytes'] or sha256(file) != expected['sha256']:
            raise ValueError('Release checksum mismatch: ' + name)
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    if actual != set(manifest['files']) | {'LAB_RELEASE_MANIFEST.json'}:
        raise ValueError('Release contains unexpected or missing files')
    for file in (folder / 'config').rglob('*.json'):
        _local_configuration(read(file), file.relative_to(folder).as_posix())
    if manifest['includes_private_baseline']:
        baseline = _validate_baseline_runtime(folder)
        if baseline['m0_signature'] != manifest['m0_signature']:
            raise ValueError('Release M0 identity differs from retained seed')
    elif (folder / BASE).exists():
        raise ValueError('Source-only release unexpectedly includes private baseline')
    return manifest


def build(root, output, *, include_baseline=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    archive = Path(str(output) + '.zip')
    checksum = Path(str(archive) + '.sha256.json')
    if output.exists() or archive.exists() or checksum.exists():
        raise FileExistsError('Choose a new release directory; existing releases are immutable')
    baseline = _validate_baseline_runtime(root) if include_baseline else None
    configs = _clean_configs(root)
    selected = set()
    for name in ROOT_REQUIRED + REQUIRED_IMPLEMENTATION:
        if not (root / name).is_file():
            raise FileNotFoundError('Required lab application file is missing: ' + name)
        selected.add(name)
    for name in ROOT_OPTIONAL:
        if (root / name).is_file():
            selected.add(name)
    for package in PACKAGES:
        for file in (root / package).rglob('*'):
            if file.is_file() and file.suffix in SOURCE_SUFFIXES and '__pycache__' not in file.parts:
                selected.add(file.relative_to(root).as_posix())
    for file in (root / 'requirements').glob('*'):
        if file.is_file() and file.suffix in ('.txt', '.json'):
            selected.add(file.relative_to(root).as_posix())
    if (root / 'experimental_data/default_processing.json').is_file():
        selected.add('experimental_data/default_processing.json')
    # Include the organized guides used by both application entry points.
    for file in (root / 'docs').rglob('*'):
        if file.is_file() and file.suffix in ('.md', '.bib'):
            selected.add(file.relative_to(root).as_posix())
    for name in ('workspace', 'experiments', 'exports', 'runs', 'data', 'tests', 'deployment', 'tools'):
        if (root / name / 'README.md').is_file():
            selected.add(name + '/README.md')
    if include_baseline:
        selected.update(BASE + '/' + name for name in baseline['files'])
        selected.add(BASE + '/manifest.json')
    for name in selected:
        relative = safe_relative(name)
        source = root / relative
        if source.is_symlink() or not source.resolve().is_relative_to(root):
            raise ValueError('Linked or escaped release source: ' + name)
    output.mkdir(parents=True)
    original_hashes = {}
    for name in sorted(selected):
        source, target = root / name, output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        original_hashes[name] = sha256(source)
        shutil.copy2(source, target)
        if sha256(target) != original_hashes[name]:
            raise ValueError('Source changed while packaging: ' + name)
    for name, value in configs.items():
        write(output / 'config' / name, value)
    write(output / 'data/dataset_manifest.json', dict(schema='aerial_cable_dataset_manifest_v1', takes={}))
    (output / '.gitignore').write_text('__pycache__/\n*.py[cod]\n.venv/\n.pytest_cache/\n.test_artifacts/\n'
        '/workspace/**\n!/workspace/README.md\n/experiments/**\n!/experiments/README.md\n'
        '/exports/**\n!/exports/README.md\n/runs/**\n!/runs/README.md\n/private_bundles/\n'
        '/tmp/\n/dist/\n/data/**\n!/data/README.md\n!/data/dataset_manifest.json\n', encoding='utf-8')
    # Guides may cite research assets deliberately absent from the lab archive.
    # Rewrite only documentation, never baseline files or numerical inputs.
    documents = {p.relative_to(output).as_posix(): b''
                 for p in output.rglob('*') if p.is_file()}
    for name in documents:
        if name.endswith('.md') and not name.startswith(BASE + '/'):
            documents[name] = (output / name).read_bytes()
    portable_guide_links(root, documents)
    for name, raw in documents.items():
        if name.endswith('.md') and not name.startswith(BASE + '/'):
            (output / name).write_bytes(raw)
    manifest = dict(schema=RELEASE_SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
        includes_private_baseline=bool(include_baseline),
        scope='Private colleague application with retained M0/preliminary seed' if include_baseline else
              'Source application; import a separately provided baseline before creating a study',
        m0_signature=baseline['m0_signature'] if baseline else None,
        entrypoint='run_lab.py', setup='setup_lab.py', offline_only=True,
        source_commit=_git_revision(root),
        source_snapshot_commit=read(root / 'SOURCE_SNAPSHOT.json').get('source_commit') if (root / 'SOURCE_SNAPSHOT.json').is_file() else None,
        transformations=['Empty study, flight selection and recording catalogs',
                        'Documentation links to absent research evidence labeled separately held',
                        'Source planner selections cleared; no job starts automatically',
                        'Retained baseline copied byte-for-byte only when explicitly requested'],
        limitations=['No environment installation or Ubuntu/GPU validation is implied by packaging',
                     'No vehicle sender is included; software produces offline CSV artifacts'],
        files={p.relative_to(output).as_posix(): dict(sha256=sha256(p), bytes=p.stat().st_size)
               for p in sorted(output.rglob('*')) if p.is_file()})
    write(output / 'LAB_RELEASE_MANIFEST.json', manifest)
    verify_release(output)
    # A concurrent source edit must not quietly produce a mixed implementation.
    for name, expected in original_hashes.items():
        if sha256(root / name) != expected:
            raise ValueError('Source changed during release; retain this attempt and build a new release: ' + name)
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as stream:
        for file in sorted(output.rglob('*')):
            if file.is_file():
                name = file.relative_to(output).as_posix()
                item = zipfile.ZipInfo.from_file(file, name)
                item.compress_type = zipfile.ZIP_DEFLATED
                item.external_attr = (0o100755 if name.endswith('.sh') else 0o100644) << 16
                stream.writestr(item, file.read_bytes())
    write(checksum, dict(file=archive.name, sha256=sha256(archive), bytes=archive.stat().st_size))
    return dict(directory=str(output), archive=str(archive), checksum=str(checksum),
                files=len(manifest['files']), includes_private_baseline=bool(include_baseline),
                sha256=sha256(archive))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-baseline', action='store_true',
                        help='Private colleague transfer: include the verified retained M0/preliminary baseline.')
    args = parser.parse_args(argv)
    print(json.dumps(build(args.root, args.output, include_baseline=args.include_baseline), indent=2))


if __name__ == '__main__':
    main()
