"""Freeze a PPO checkpoint for rehearsal without stopping its writer."""
from contextlib import contextmanager
from pathlib import Path
import os
import shutil
import torch
from simulator.workflow import read_json
from experimental_data.io import atomic_json,sha256_file
from planning.pva_job import freeze_model_assets,CHECKPOINT_SCHEMA


@contextmanager
def checkpoint_reader(path,*,directory=None):
    if os.name!='nt':
        with Path(path).open('rb') as stream:yield stream
        return
    # MoveFileEx can refuse to replace an open destination even with DELETE
    # sharing. Pin a separate hard link first, then read that name with all
    # sharing enabled. The trainer only writes temporary files and replaces
    # best/latest; its original path remains closed throughout this copy.
    import ctypes
    from ctypes import wintypes
    import msvcrt
    from uuid import uuid4
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    create=kernel.CreateFileW
    create.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    create.restype=wintypes.HANDLE
    link=Path(directory or Path(path).parent)/('.snapshot-'+uuid4().hex+'.pt')
    os.link(path,link)
    try:
        handle=create(str(link.resolve()),0x80000000,7,None,3,0x08000000,None)
        if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
        try:fd=msvcrt.open_osfhandle(handle,os.O_RDONLY|os.O_BINARY)
        except BaseException:
            kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle(handle);raise
        with os.fdopen(fd,'rb') as stream:yield stream
    finally:link.unlink()


def prepare_snapshot(root,run,directory,choice='latest.pt'):
    root=Path(root);run=Path(run).resolve();directory=Path(directory).resolve()
    cfg=read_json(run/'settings.json')
    if cfg['method']!='ppo':raise ValueError('Checkpoint snapshots are for PPO policies only')
    if choice not in ('latest.pt','best.pt'):raise ValueError('Choose latest.pt or best.pt')
    source=run/'checkpoints'/choice
    if not source.is_file():raise ValueError('This checkpoint has not been saved yet. Wait for a completed training update.')
    if not (run/'source_snapshot').is_dir():raise ValueError('The original run source snapshot is missing')
    directory.mkdir(parents=True,exist_ok=False);(directory/'checkpoints').mkdir()
    checkpoint=directory/'checkpoints/policy.pt'
    with checkpoint_reader(source,directory=checkpoint.parent) as reader,checkpoint.open('xb') as writer:shutil.copyfileobj(reader,writer)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if payload.get('schema')!=CHECKPOINT_SCHEMA:raise ValueError('Select a direct PVA PPO checkpoint')
    provenance=dict(source_run=str(run),source_run_name=read_json(run/'identity.json',{}).get('name',run.name),
        checkpoint_choice=choice,checkpoint_attempts=payload.get('attempts'),checkpoint_sha256=sha256_file(checkpoint),
        source_status=read_json(run/'status.json',{}).get('status','unknown'))
    model=read_json(run/'model.json')
    for key in ('motion_residual','fullstate_execution'):
        path=Path(model[key]['checkpoint']);model[key]['checkpoint']=str(path if path.is_absolute() else run/path)
    frozen=freeze_model_assets(model,directory)
    atomic_json(directory/'model.json',frozen);atomic_json(directory/'settings.json',cfg)
    atomic_json(directory/'policy_snapshot.json',provenance)
    atomic_json(directory/'status.json',dict(status='snapshot',attempts=payload.get('attempts')))
    shutil.copytree(run/'source_snapshot',directory/'source_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
    if (run/'source_manifest.json').exists():shutil.copy2(run/'source_manifest.json',directory/'original_source_manifest.json')
    # Add only the preview adapter. Original environment, policy and recovery
    # implementations remain the exact versions frozen by the training run.
    additions={}
    for relative in ('deployment/pva_policy_preview.py','tools/rehearse_ppo_snapshot.py'):
        destination=directory/'source_snapshot'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/relative,destination);additions[relative]=sha256_file(destination)
    atomic_json(directory/'preview_adapter_manifest.json',additions)
    return provenance
