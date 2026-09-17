"""Cooperative cancellation with rollback to the last durable checkpoint."""
from contextvars import ContextVar
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import shutil
import time

import torch

from simulator.artifact_io import replace_with_retry

_stop_context = ContextVar('training_stop_context', default=None)
_runtime_pump = ContextVar('training_runtime_pump', default=None)


def set_runtime_pump(callback):
    """Optional in-process application event pump; ordinary training has none."""
    return _runtime_pump.set(callback)


def reset_runtime_pump(token):
    _runtime_pump.reset(token)


class TrainingStopped(Exception):
    pass


def check_training_stop():
    pump=_runtime_pump.get()
    if pump is not None:pump()
    context = _stop_context.get()
    if context is None:
        return
    now = time.monotonic()
    if now < context['next_check']:
        return
    context['next_check'] = now + .1
    if (context['artifact'] / 'STOP_REQUESTED').exists():
        raise TrainingStopped()


def finish_stopped(artifact):
    latest = artifact / 'checkpoints/latest.pt'
    checkpoint = torch.load(latest, map_location='cpu', weights_only=False)
    temporary = artifact / 'checkpoints/terminal.pt.tmp'
    shutil.copyfile(latest, temporary)
    replace_with_retry(temporary, artifact / 'checkpoints/terminal.pt')
    path = artifact / 'status.json'
    status = json.loads(path.read_text(encoding='utf-8'))
    # In-flight metrics may refer to an unsaved update; keep only run identity.
    status = {key: value for key, value in status.items() if key in (
        'schema','algorithm','pid','device','target_episodes','episodes_target',
        'collection_batch','artifact','training_execution_mode')}
    status.update(status='STOPPED', episodes=int(checkpoint['episodes']),
                  stage='Stopped · last saved update preserved', phase_step=0, phase_total=0,
                  stop_reason='User request; unfinished work discarded',
                  resume_checkpoint=str(latest.resolve()))
    if 'successes' in checkpoint:
        status['successes'] = int(checkpoint['successes'])
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(status, indent=2)+'\n', encoding='utf-8')
    replace_with_retry(temporary,path)
    return status


def stoppable_training(default_artifact, *, return_status=False):
    def decorate(function):
        signature = inspect.signature(function)
        @wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            artifact = bound.arguments.get('artifact')
            if artifact is None and not bound.arguments.get('preflight',False):
                artifact = default_artifact()
                bound.arguments['artifact'] = artifact
            token = _stop_context.set(None if artifact is None else
                dict(artifact=Path(artifact).resolve(),next_check=0.))
            try:
                return function(*bound.args,**bound.kwargs)
            except TrainingStopped:
                status = finish_stopped(Path(artifact).resolve())
                print(json.dumps(status),flush=True)
                return status if return_status else Path(artifact).resolve()
            finally:
                _stop_context.reset(token)
        return wrapped
    return decorate
