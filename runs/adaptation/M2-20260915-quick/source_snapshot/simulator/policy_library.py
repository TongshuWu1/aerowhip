"""Recoverable library removal without moving research artifacts or breaking provenance."""
from pathlib import Path
from experimental_data.io import atomic_json
from simulator.workflow import read_json


def deleted_checkpoints(root):
    return set(read_json(Path(root)/'config/policy_library.json', {}).get('deleted', []))


def checkpoint_key(root, checkpoint):
    root, checkpoint = Path(root).resolve(), Path(checkpoint).resolve()
    parent = root/'runs/ppo'
    if not checkpoint.is_relative_to(parent) or checkpoint.suffix != '.pt':
        raise ValueError('Choose a checkpoint inside this project’s PPO runs.')
    return checkpoint.relative_to(root).as_posix()


def set_deleted(root, checkpoint, deleted):
    key = checkpoint_key(root, checkpoint)
    if not Path(checkpoint).is_file():
        raise ValueError('The checkpoint no longer exists.')
    values = deleted_checkpoints(root)
    if deleted:
        values.add(key)
    else:
        values.discard(key)
    atomic_json(Path(root)/'config/policy_library.json', dict(deleted=sorted(values)))


def list_policies(root, include_deleted=False):
    root = Path(root).resolve()
    deleted = deleted_checkpoints(root)
    rows = []
    for run in sorted((root/'runs/ppo').glob('*'), reverse=True):
        if not run.is_dir():
            continue
        metadata = read_json(run/'run.json', {})
        status = read_json(run/'status.json', {})
        for checkpoint in sorted((run/'checkpoints').glob('*.pt')):
            removed = checkpoint_key(root, checkpoint) in deleted
            if removed and not include_deleted:
                continue
            rows.append(dict(path=str(checkpoint.resolve()), run=str(run),
                name=metadata.get('display_name', run.name), checkpoint=checkpoint.name,
                status=status.get('status', 'Saved'), deleted=removed))
    return rows
