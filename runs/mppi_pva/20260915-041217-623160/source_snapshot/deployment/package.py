"""Export an explicitly chosen saved PPO with its configurations and headless planner."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_policy(root, checkpoint, destination, *, experiment_setup=None):
    root, checkpoint, destination = map(Path, (root, checkpoint, destination))
    run = checkpoint.parent.parent
    if destination.exists():
        raise ValueError('Choose a new export directory.')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload.get('schema') != 'force_ppo_checkpoint_v1':
        raise ValueError('Select a point-force PPO checkpoint.')
    for name in ('model', 'task', 'ppo'):
        if not (run / f'{name}.json').is_file():
            raise ValueError('Checkpoint must have its saved model/task/PPO configurations.')
    saved_model = json.loads((run/'model.json').read_text(encoding='utf-8'))
    residual = saved_model.get('motion_residual', {})
    execution = saved_model.get('fullstate_execution', {})
    execution_path = None
    if execution.get('enabled'):
        execution_path = Path(execution['checkpoint'])
        if execution_path.is_absolute() or '..' in execution_path.parts or execution_path.parts[:2] != ('data','baselines'):
            raise ValueError('Export requires an applied full-state execution baseline.')
        if digest(root/execution_path) != execution['sha256']:
            raise ValueError('Full-state execution checkpoint changed since calibration.')
    residual_path = None
    if residual.get('enabled'):
        residual_path = Path(residual['checkpoint'])
        if residual_path.is_absolute() or '..' in residual_path.parts or residual_path.parts[:2] != ('data', 'baselines'):
            raise ValueError('Export requires a cable residual applied as a project baseline.')
        if digest(root/residual_path) != residual['sha256']:
            raise ValueError('Cable residual checkpoint changed since calibration.')
    policy = destination / 'policy'
    (policy / 'checkpoints').mkdir(parents=True)
    for name in ('model', 'task', 'ppo'):
        shutil.copy2(run / f'{name}.json', policy / f'{name}.json')
    if residual_path is not None:
        target = destination/residual_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/residual_path, target)
    if execution_path is not None:
        target = destination/execution_path
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/execution_path,target)
    shutil.copy2(checkpoint, policy / 'checkpoints/policy.pt')
    # Validate the copied bytes as well: latest.pt may be replaced atomically by training.
    copied = torch.load(policy / 'checkpoints/policy.pt', map_location='cpu', weights_only=False)
    if copied.get('schema') != 'force_ppo_checkpoint_v1':
        raise ValueError('Exported checkpoint is not PPO.')
    for name in ('simulator', 'learning', 'experimental_data', 'deployment'):
        for source in (root / name).rglob('*'):
            relative = source.relative_to(root)
            if (source.is_file() and not source.is_symlink()
                    and not {'gui', '__pycache__'}.intersection(relative.parts)
                    and source.suffix in {'.py', '.cu', '.cuh', '.cpp', '.h'}):
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    if (run / 'source_snapshot_manifest.json').is_file():
        shutil.copy2(run / 'source_snapshot_manifest.json', destination / 'training_source_manifest.json')
    shutil.copy2(root / 'requirements/headless.txt', destination / 'requirements.txt')
    shutil.copy2(root / 'deployment/TESTING_README.md', destination / 'README.md')
    if experiment_setup is not None:
        (destination / 'experiment_setup.json').write_text(
            json.dumps(experiment_setup, indent=2)+'\n', encoding='utf-8')
    manifest = dict(schema='selected_ppo_package_v1', source_run=run.name,
        source_checkpoint=checkpoint.name, episodes=int(copied['episodes']), flight_ready=False,
        runtime_source='export-time workspace; separately hashed from training source',
        files={p.relative_to(policy).as_posix(): digest(p) for p in policy.rglob('*') if p.is_file()})
    (policy / 'policy_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    files = {p.relative_to(destination).as_posix(): digest(p)
             for p in destination.rglob('*') if p.is_file()}
    (destination / 'TRANSFER_MANIFEST.json').write_text(json.dumps(dict(files=files), indent=2)+'\n', encoding='utf-8')
    archive = destination.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as stream:
        for path in destination.rglob('*'):
            if path.is_file():
                stream.write(path, (Path(destination.name) / path.relative_to(destination)).as_posix())
    return destination
