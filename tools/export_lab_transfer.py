"""Export the current research workspace and a selected, offline PPO planner.

No recordings are interpreted, checkpoints selected heuristically, or vehicles contacted.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SELECTED_RUN = 'runs/ppo/20260905-235530-202591-seed651'
SELECTED_CHECKPOINT = 'manual_000229376_20260906T021921_current.pt'
SELECTED_SHA256 = '3c470ec4ec7b7cd3bec4312a507b7bca275c8c05eb87401287cff5c37a2f932d'
SKIP = {'.git', '.venv', '.idea', '__pycache__', '.pytest_cache', '.ruff_cache', 'archive', 'dist'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def copy_tree(source, destination, *, code_only=False, excluded=()):
    source, destination = Path(source), Path(destination)
    for path in sorted(source.rglob('*')):
        relative = path.relative_to(source)
        if any(part in SKIP or part in excluded for part in relative.parts) or path.suffix in {'.pyc', '.pyo'}:
            continue
        if path.is_symlink():
            raise ValueError(f'Linked source is not exported: {path}')
        if path.is_file() and (not code_only or path.suffix in {'.py', '.svg'}):
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def seal(folder):
    files = {p.relative_to(folder).as_posix(): sha256(p)
             for p in sorted(folder.rglob('*')) if p.is_file()}
    write_json(folder / 'TRANSFER_MANIFEST.json', {
        'schema': 'lab_transfer_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'private lab transfer; no flight authorization or public-release license',
        'files': files})
    archive = folder.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                z.write(path, arcname=(Path(folder.name) / path.relative_to(folder)).as_posix())
    return {'folder': str(folder), 'archive': str(archive), 'sha256': sha256(archive),
            'archive_bytes': archive.stat().st_size, 'files': len(files)}


def build(output, root=ROOT):
    output, root = Path(output).resolve(), Path(root).resolve()
    if output.is_relative_to(root) or output.exists():
        raise ValueError('Choose a NEW output directory outside the project')
    checkpoint = root / SELECTED_RUN / 'checkpoints' / SELECTED_CHECKPOINT
    if sha256(checkpoint) != SELECTED_SHA256:
        raise ValueError('Selected PPO has changed; review selection before exporting')
    output.mkdir(parents=True)
    full, small = output / 'Whip-Development', output / 'Whip-Deployment'
    full.mkdir(); small.mkdir()
    for name in ('simulator', 'learning', 'experimental_data', 'deployment', 'tools', 'tests',
                 'config', 'requirements', 'docs', 'data', 'runs', 'results', '.run', '.github'):
        if (root/name).exists():
            copy_tree(root/name, full/name)
    for name in ('README.md', 'HANDOFF.md', 'AGENTS.md', 'requirements.txt', 'pytest.ini',
                 '.editorconfig', '.gitattributes', '.gitignore', 'run_simulation.py',
                 'run_ppo.py', 'run_sac.py', 'run_tests.py'):
        shutil.copy2(root/name, full/name)
    # Relocatable UI selectors only; immutable run snapshots retain provenance.
    for name in ('ppo', 'sac'):
        pointer = full/'runs'/name/'ACTIVE_RUN.txt'
        if pointer.exists():
            run_name = pointer.read_text(encoding='utf-8').strip().replace('\\', '/').split('/')[-1]
            if not (full/'runs'/name/run_name).is_dir():
                raise ValueError(f'Active {name} run is absent from export')
            pointer.write_text(f'runs/{name}/{run_name}\n', encoding='utf-8')
    # Headless dependencies only. No GUI, recordings, experiment results or SAC weights.
    for name in ('simulator', 'learning', 'experimental_data', 'deployment'):
        copy_tree(root/name, small/name, code_only=True, excluded=('gui',))
    # simulator.gui is never imported by the planner, and is deliberately omitted.
    # Build its subset by excluding it before copying, rather than deleting afterward.
    policy = small/'policy'
    (policy/'checkpoints').mkdir(parents=True)
    for name in ('model', 'task', 'ppo'):
        shutil.copy2(root/SELECTED_RUN/f'{name}.json', policy/f'{name}.json')
    shutil.copy2(checkpoint, policy/'checkpoints/policy.pt')
    write_json(policy/'policy_manifest.json', {
        'schema': 'selected_ppo_package_v1', 'source_run': SELECTED_RUN,
        'source_checkpoint': SELECTED_CHECKPOINT, 'episodes': 229376,
        'flight_ready': False,
        'files': {p.relative_to(policy).as_posix(): sha256(p)
                  for p in sorted(policy.rglob('*')) if p.is_file()}})
    shutil.copy2(root/'requirements/headless.txt', small/'requirements.txt')
    shutil.copy2(root/'deployment/README.md', small/'README.md')
    shutil.copy2(root/'deployment/README.md', small/'deployment/README.md')
    shutil.copy2(root/'HANDOFF.md', small/'HANDOFF.md')
    shutil.copy2(root/'AGENTS.md', small/'AGENTS.md')
    (small/'docs').mkdir()
    for name in ('CONTROLLER_INTERFACE_REVIEW.md', 'INITIAL_STATE_OPEN_LOOP.md',
                 'FLIGHT_ADAPTATION_QUICKSTART.md', 'LAB_SETUP.md', 'LAB_VALIDATION.json'):
        shutil.copy2(root/'docs'/name, small/'docs'/name)
    result = {'development': seal(full), 'deployment': seal(small)}
    write_json(output/'BUNDLES.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    print(json.dumps(build(parser.parse_args().output), indent=2))
