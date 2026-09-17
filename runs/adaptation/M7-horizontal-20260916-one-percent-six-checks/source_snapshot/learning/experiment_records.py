"""Append-only evaluation records, with a replay of the actual first trial."""
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np

from experimental_data.io import atomic_json, sha256_file


class EvaluationResult(dict):
    """JSON-compatible metrics carrying non-JSON trial data through PPO rollback."""


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def save_evaluation(directory: Path, result: EvaluationResult):
    """Called only after latest.pt contains the evaluated (accepted) weights.

    PPO may reuse an accepted evaluation after rejecting an update. Keep its
    original evaluation ID so reuse is explicit, while associating it with the
    new training-attempt count and checkpoint bytes.
    """
    checkpoint = directory / 'checkpoints/latest.pt'
    if not checkpoint.is_file() or not hasattr(result, 'recording'):
        return
    checkpoint_hash = sha256_file(checkpoint)
    step = int(result['training_episodes'])
    folder = directory / 'validation'
    folder.mkdir(exist_ok=True)
    record_id = f'{step:010d}-{checkpoint_hash[:12]}'
    entry = folder / f'{record_id}.json'
    if entry.exists():
        return
    arrays, metadata = result.recording
    scenarios = result.scenarios
    digest = hashlib.sha256()
    for key, values in sorted(scenarios.items()):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(values).tobytes())
    scenario_id = digest.hexdigest()
    config_hashes = {name: sha256_file(directory / name) for name in
                     ('model.json', 'task.json', 'ppo.json', 'sac.json') if (directory / name).is_file()}
    try:
        previous = json.loads((folder / 'latest.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        previous = {}
    reused = previous.get('evaluation_id') == result.evaluation_id
    record = finite_json(dict(result, record_id=record_id, checkpoint_sha256=checkpoint_hash,
                             checkpoint=f'validation/{record_id}.pt', scenario_id=scenario_id,
                             config_sha256=config_hashes, evaluation_id=result.evaluation_id,
                             evaluation_reused=reused,
                             evaluated_at_training_episodes=previous.get('evaluated_at_training_episodes', step) if reused else step,
                             recorded_at_utc=datetime.now(timezone.utc).isoformat()))
    # A completed JSON entry is the commit marker. Readers ignore partial work.
    from simulator.validation_preview import write_recording
    policy_path = folder / f'{record_id}.pt'
    if not policy_path.exists():
        try:
            os.link(checkpoint, policy_path)
        except OSError:
            shutil.copy2(checkpoint, policy_path)
    replay_path = folder / f'{record_id}.npz'
    write_recording(replay_path, arrays, dict(metadata, **record,
                    schema='training_validation_trial_v1', trials=1))
    scenario_path = folder / f'scenarios-{scenario_id[:16]}.npz'
    if not scenario_path.exists():
        write_recording(scenario_path, scenarios, {'scenario_id': scenario_id})
    trials = finite_json(result.trials)
    atomic_json(folder / f'{record_id}-trials.json', trials)
    record['replay'] = str(replay_path.relative_to(directory)).replace('\\', '/')
    record['trial_outcomes'] = f'validation/{record_id}-trials.json'
    atomic_json(entry, record)
    with (directory / 'validation_history.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')
    fields = ['training_episodes', 'success_rate', 'hit_and_recovery_rate',
              'mean_episode_reward', 'episodes', 'record_id', 'evaluation_id',
              'scenario_id', 'checkpoint_sha256']
    csv_path = directory / 'validation_history.csv'
    exists = csv_path.exists()
    with csv_path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: record[key] for key in fields})
    atomic_json(folder / 'latest.json', record)
