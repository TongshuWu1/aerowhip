"""Apply a recorded stopping-rule amendment to a frozen, already-running PPO worker."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experimental_data.io import atomic_json
from learning.reward_plateau import RewardPlateau
from simulator.gui.process_status import process_is_running


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def assess_run(run):
    settings = dict(read(run/'ppo.json')['early_stopping'])
    amendment = read(run/'early_stopping_override.json')
    settings.update(amendment['settings'])
    start = int(read(run/'run.json')['parent_training_episodes'])
    monitor = RewardPlateau(settings, start_episodes=start)
    paths = sorted((run/'validation').glob('[0-9]*.json'))
    for path in paths:
        if not path.name.endswith('-trials.json'):
            monitor.observe(read(path))
    return dict(monitor.decision, amendment=amendment)


def request_stop(run, decision):
    if not decision['should_stop'] or read(run/'status.json')['status'] != 'RUNNING':
        return False
    payload = dict(source='reward_plateau_override', decision=decision,
                   requested_at=datetime.now(timezone.utc).isoformat())
    try:
        with (run/'STOP_REQUESTED').open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(payload, indent=2)+'\n')
    except FileExistsError:
        return False  # Never replace an existing user stop.
    atomic_json(run/'reward_plateau_stop_request.json', payload)
    return True


def watch(run, expected_pid):
    state_path = run/'reward_stop_monitor.json'
    state = dict(status='MONITORING', pid=os.getpid(), worker_pid=expected_pid,
                 stop_requested=False)
    stop_decision = None
    lock = run/'reward_stop_monitor.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        while True:
            status = read(run/'status.json')
            if int(status.get('pid', 0)) != expected_pid:
                raise RuntimeError('Worker identity changed; monitor will not follow another worker.')
            alive = process_is_running(expected_pid)
            if status['status'] not in ('RUNNING', 'STARTING') or not alive:
                if alive:  # Wait for the worker to finish its own final status write.
                    time.sleep(2)
                    continue
                if status['status'] == 'STOPPED' and state['stop_requested']:
                    atomic_json(run/'status_before_stop_annotation.json', status)
                    patience = stop_decision['settings']['patience_episodes']
                    status.update(stop_reason=f'Validation reward plateau ({patience:,}-attempt rule)',
                                  stage='Stopped · validation reward plateau',
                                  reward_convergence=stop_decision)
                    atomic_json(run/'status.json', status)
                state.update(status='FINISHED', worker_status=status['status'])
                atomic_json(state_path, state)
                return
            decision = assess_run(run)
            atomic_json(run/'reward_convergence_override.json', decision)
            if not state['stop_requested'] and request_stop(run, decision):
                state['stop_requested'] = True
                stop_decision = decision
            state.update(checked_at=datetime.now(timezone.utc).isoformat(),
                         latest_validation_episodes=decision.get('latest_episodes'),
                         attempts_without_improvement=decision.get('attempts_without_improvement'))
            atomic_json(state_path, state)
            time.sleep(5)
    except BaseException as error:
        state.update(status='FAILED', error=f'{type(error).__name__}: {error}')
        atomic_json(state_path, state)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--worker-pid', type=int, required=True)
    args = parser.parse_args()
    watch(args.run.resolve(), args.worker_pid)
