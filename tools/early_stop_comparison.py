"""Attach validation-reward early stopping to an already running comparison.

The frozen training workers are unchanged. Their existing cooperative stop
interface is used, and every original checkpoint and validation point is kept.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulator.workflow import atomic_json, read_json

DEFAULTS = dict(minimum_episodes=0, patience=5, minimum_improvement=2.0)
TERMINAL = {'COMPLETED', 'STOPPED', 'NEEDS_ATTENTION', 'FAILED'}


def assess(records, settings):
    """Patience uses significant improvements; selection uses the actual maximum."""
    best = None
    anchor = None
    stale = 0
    seen = set()
    last = None
    for record in sorted(records, key=lambda r: r['training_episodes']):
        identity = record.get('evaluation_id', record['record_id'])
        reward = record.get('mean_episode_reward')
        if identity in seen or record.get('evaluation_reused') or reward is None or not math.isfinite(reward):
            continue
        seen.add(identity)
        last = record
        if best is None or reward > best['mean_episode_reward']:
            best = record
        if anchor is None or reward >= anchor + settings['minimum_improvement']:
            anchor = reward
            stale = 0
        else:
            stale += 1
    return dict(best=best, stale_checks=stale,
                latest_validation_episodes=last['training_episodes'] if last else 0,
                should_stop=bool(last and last['training_episodes'] >= settings['minimum_episodes']
                                 and stale >= settings['patience']))


def records_for(directory):
    # Immutable individual JSON records are commit markers. Never read a
    # partially appended journal or count the latest.json pointer twice.
    return [read_json(path) for path in (directory/'validation').glob('[0-9]*.json')
            if not path.name.endswith('-trials.json')]


def check_run(directory, settings):
    decision = assess(records_for(directory), settings)
    best = decision.pop('best')
    old = read_json(directory/'early_stopping.json', {})
    if best:
        source = (directory/best['checkpoint']).resolve()
        if not source.is_relative_to(directory.resolve()):
            raise ValueError('Validation checkpoint outside run')
        selected = directory/'checkpoints/best_reward.pt'
        if old.get('selected_record_id') != best['record_id'] or not selected.exists():
            if hashlib.sha256(source.read_bytes()).hexdigest() != best['checkpoint_sha256']:
                raise ValueError('Validation checkpoint checksum mismatch')
            temporary = selected.with_suffix('.tmp')
            shutil.copyfile(source, temporary)
            os.replace(temporary, selected)
        decision.update(selected_record_id=best['record_id'],
                        selected_checkpoint=str(selected),
                        selected_checkpoint_sha256=best['checkpoint_sha256'],
                        best_reward=best['mean_episode_reward'],
                        best_reward_episodes=best['training_episodes'])
    decision['stop_requested'] = old.get('stop_requested', False)
    status = read_json(directory/'status.json', {})
    stop = directory/'STOP_REQUESTED'
    if decision['should_stop'] and status.get('status') == 'RUNNING' and not stop.exists():
        # This is a per-run stop. The study queue remains free to launch SAC.
        stop.write_text('Validation reward plateau; see early_stopping.json\n')
        decision.update(stop_requested=True, stop_requested_at=datetime.now(timezone.utc).isoformat())
    elif old.get('stop_requested_at'):
        decision['stop_requested_at'] = old['stop_requested_at']
    atomic_json(directory/'early_stopping.json', decision)
    return decision


def evaluation_checkpoint(directory, decision, status):
    """Honor an explicit user-selected completion without treating a crash as one."""
    manual = read_json(directory/'manual_completion.json', {})
    if status.get('status') == 'STOPPED' and manual.get('final_evaluation_authorized'):
        path = (directory/manual['checkpoint']).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError('Manual checkpoint outside run')
        if hashlib.sha256(path.read_bytes()).hexdigest() != manual['checkpoint_sha256']:
            raise ValueError('Manual completion checkpoint checksum mismatch')
        return manual['checkpoint'], 'User-selected PPO completion; reserved evaluation does not select weights'
    if status.get('status') == 'COMPLETED' or (status.get('status') == 'STOPPED' and decision['stop_requested']):
        if 'selected_checkpoint' in decision:
            return 'checkpoints/best_reward.pt', 'Validation-reward selected checkpoint under documented early-stopping amendment; reserved evaluation does not select weights'
    return None


def monitor(study):
    protocol = read_json(study/'protocol.json')
    amendment = read_json(study/'early_stopping_protocol.json')
    settings = amendment['settings']
    state = dict(status='MONITORING', pid=os.getpid(), runs={})
    state_path = study/'early_stopping_status.json'
    try:
        while True:
            # A whole-study cancellation wins over this supervisor.
            if (study/'STOP_REQUESTED').exists():
                state['status'] = 'STOPPED'
                atomic_json(state_path, state)
                return
            for label, item in protocol['runs'].items():
                state['runs'][label] = check_run(Path(item['directory']), settings)
            state['checked_at'] = datetime.now(timezone.utc).isoformat()
            atomic_json(state_path, state)
            queue = read_json(study/'status.json', {})
            if queue.get('status') in TERMINAL:
                break
            time.sleep(5)
        # The original queue evaluates completed terminal checkpoints. Wait
        # until it releases the GPU before evaluating our reward-selected ones.
        for label, item in protocol['runs'].items():
            if (study/'STOP_REQUESTED').exists():
                state['status'] = 'STOPPED'
                atomic_json(state_path, state)
                return
            directory = Path(item['directory'])
            decision = state['runs'][label]
            status = read_json(directory/'status.json', {})
            selection = evaluation_checkpoint(directory, decision, status)
            if selection is None:
                decision['evaluation_skipped'] = 'Run failed or stopped outside plateau rule'
                continue
            output = study/'selected_reward_evaluation'/label.lower()
            state.update(status='EVALUATING', active_algorithm=label)
            atomic_json(state_path, state)
            if output.exists():
                raise ValueError(f'Evaluation already exists; inspect before restarting: {output}')
            command = [sys.executable, str(study/'code/tools/evaluate_selected_strike.py'), str(directory),
                       '--checkpoint', selection[0],
                       '--seed', str(protocol['final_evaluation_seed']),
                       '--episodes', str(protocol['final_evaluation_episodes']), '--output', str(output),
                       '--purpose', selection[1]]
            decision['evaluation_checkpoint'] = selection[0]
            with (study/f'{label.lower()}_selected_reward_evaluation.log').open('wb') as log:
                result = subprocess.run(command, cwd=study/'code', stdout=log, stderr=subprocess.STDOUT,
                                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            decision['evaluation_exit_code'] = result.returncode
        state.update(status='COMPLETED' if all(r.get('evaluation_exit_code') == 0 for r in state['runs'].values())
                     else 'NEEDS_ATTENTION', active_algorithm=None)
        atomic_json(state_path, state)
        # Preserve original queue outcome labels; annotate why a worker stopped.
        queue['early_stopping_summary'] = str(state_path)
        atomic_json(study/'status.json', queue)
    except BaseException as error:
        state.update(status='FAILED', error=f'{type(error).__name__}: {error}')
        atomic_json(state_path, state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    parser.add_argument('--once', action='store_true', help='Read-only plateau assessment')
    args = parser.parse_args()
    study = args.study.resolve()
    if args.once:
        protocol = read_json(study/'protocol.json')
        settings = read_json(study/'early_stopping_protocol.json')['settings']
        for label, item in protocol['runs'].items():
            print(label, json.dumps(assess(records_for(Path(item['directory'])), settings)))
    else:
        monitor(study)
