"""Explicit, bounded M1/M0 continuation study. No automatic retry or flight sender."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from experimental_data.io import atomic_json,sha256_file
from simulator.workflow import read_json,prepare_training,stamp
from simulator.research_config import snapshot_assets

PARENT='runs/ppo/20260908-195207-486249-seed655/checkpoints/best_validation.pt'
PARENT_SHA='d10657f471deb8b22cafbee9008792e8378c8c764ca97039d6dbe7afcc260eea'
M1='data/model_candidates/20260908-adp0-M1/model.json'
M1_SHA='ee99478f670ae36cd21ca15095b52251d394a761ac9b044369845c6e0cb02d5a'
ORIGIN=[-2.,0.,1.255]
TARGET=[-1.,0.,1.1]


def prepare(root):
    root=Path(root).resolve();checkpoint=root/PARENT;model_path=root/M1
    if sha256_file(checkpoint)!=PARENT_SHA or sha256_file(model_path)!=M1_SHA:
        raise ValueError('Selected policy or M1 differs from the authorized immutable input.')
    name=stamp()+'-M1-policy-adaptation';study=root/'runs/policy_adaptation'/name
    study.mkdir(parents=True)
    protected=[checkpoint.parent.parent,root/'runs/rehearsals/20260908-203914-039721',model_path.parent,
        root/'rehearsal_csv_and_result_in_real_flight/20260908-195207-486249-seed655_best_validation/adp0']
    hashes={str(p):sha256_file(p) for folder in protected for p in folder.rglob('*') if p.is_file()}
    for p in (root/'config/research_30hz').glob('*.json'):hashes[str(p)]=sha256_file(p)
    atomic_json(study/'protected_inputs.json',hashes)
    parent=study/'parent_policy';parent.mkdir();(parent/'checkpoints').mkdir()
    frozen=parent/'checkpoints/best_validation.pt';shutil.copy2(checkpoint,frozen)
    model=read_json(checkpoint.parent.parent/'model.json')
    atomic_json(parent/'model.json',snapshot_assets(model,parent))
    for name in ('task','ppo'):shutil.copy2(checkpoint.parent.parent/f'{name}.json',parent/f'{name}.json')
    start=int(torch.load(frozen,map_location='cpu',weights_only=False)['episodes']);additional=20480;target=start+additional
    offset=np.array(model['recorded_data']['optitrack_to_attachment_offset_body_m'])
    task_overrides=dict(initial_root_position_m=(np.array(ORIGIN)+offset).tolist(),target_position_m=TARGET)
    backend=dict(type='isaaclab_model',python=str(Path.home()/'env_isaaclab/Scripts/python.exe'),headless=False,render_stride=5)
    if not Path(backend['python']).is_file():raise ValueError('Configured Isaac Lab runtime is unavailable.')
    jobs={}
    for label,path in [('M1',model_path),('M0',parent/'model.json')]:
        directory,command=prepare_training(root,'ppo',seed=655,episodes=target,batch=2048,device='cuda',
            resume=frozen,model_path=path,keep_optimizer_state=False,keep_stopping_history=False,
            task_overrides=task_overrides,early_stopping=dict(enabled=False,
                reason='Fixed equal additional-attempt budget for the authorized M1/M0 comparison'),
            live_scene=False,training_backend=backend,
            run_name=f'{label} PPO - '+('adaptation' if label=='M1' else 'extra-training control')+' - 20480 attempts')
        atomic_json(directory/'policy_study.json',dict(study=str(study),condition=label,parent_policy=str(checkpoint),
            parent_policy_sha256=PARENT_SHA,additional_attempts=additional,real_flight_performance='UNKNOWN'))
        atomic_json(directory/'status.json',dict(status='QUEUED',episodes=start,target_episodes=target,
            collection_batch=2048,stage='Prepared; not started',pid=0))
        jobs[label]=dict(directory=str(directory),command=command)
    configs=[read_json(Path(j['directory'])/'launch_config/ppo.json') for j in jobs.values()]
    tasks=[read_json(Path(j['directory'])/'launch_config/task.json') for j in jobs.values()]
    assert configs[0]==configs[1] and tasks[0]==tasks[1]
    assert configs[0]['reward']==read_json(parent/'ppo.json')['reward']
    assert tasks[0]['episode_duration_s']==1. and tasks[0]['control_dt_s']==1/30
    assert configs[0]['deployment']['reset_optimizer_on_resume']
    assert np.allclose(np.array(tasks[0]['initial_root_position_m'])-offset,ORIGIN,atol=1e-12)
    baseline_source=study/'frozen_policy_M1';baseline_source.mkdir();(baseline_source/'checkpoints').mkdir()
    shutil.copy2(frozen,baseline_source/'checkpoints/policy.pt')
    for name in ('model','task','ppo'):
        shutil.copy2(Path(jobs['M1']['directory'])/'launch_config'/f'{name}.json',baseline_source/f'{name}.json')
    rehearsal=root/'runs/rehearsals'/(study.name+'-unchanged-PPO-M1')
    baseline_command=[sys.executable,'-u',str(Path(jobs['M1']['directory'])/'source_snapshot/tools/rehearse_research.py'),
        '--checkpoint',str(baseline_source/'checkpoints/policy.pt'),'--output',str(rehearsal),
        '--origin',*map(str,ORIGIN),'--target',*map(str,TARGET),'--device','cuda']
    atomic_json(study/'study.json',dict(schema='m1_policy_adaptation_study_v1',root=str(root),jobs=jobs,
        original_policy=str(checkpoint),original_policy_sha256=PARENT_SHA,M1_source=str(model_path),M1_sha256=M1_SHA,
        additional_attempts=additional,parent_attempts=start,target_attempts=target,
        baseline_command=baseline_command,baseline_rehearsal=str(rehearsal),backend=backend,
        launch=dict(initial_tracking_origin_m=ORIGIN,target_position_m=TARGET),
        setup_amendment='Both training arms use the flown tracked-origin Y=0, correcting the old training center at Y=+0.012874 m.',
        inference='Simulation-only prediction. Policy performance in real flight is unknown until adp1.',
        checkpoint_selection='Existing validation_rank: success rate, reward, displacement; same 256 fixed development scenarios per arm.',
        stopping='Exactly 20480 additional attempts per arm; no automatic extension. Stop/error prevents the next arm starting.'))
    atomic_json(study/'status.json',dict(status='PREPARED',training_started=False,real_flights_collected=False))
    return study


def launch(command,root,log):
    env=os.environ.copy();env['PYTHONUTF8']='1'
    for key in ('PYTHONPATH','QT_QPA_PLATFORM_PLUGIN_PATH','QT_PLUGIN_PATH','QT_QPA_PLATFORM'):env.pop(key,None)
    with Path(log).open('ab') as output:
        return subprocess.Popen(command,cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)


def run(study):
    study=Path(study).resolve();spec=read_json(study/'study.json');root=Path(spec['root'])
    if read_json(study/'status.json')['status']!='PREPARED':raise ValueError('This study has already started; never restart automatically.')
    with (study/'worker.lock').open('x') as f:f.write(str(os.getpid()))
    started=time.perf_counter()
    try:
        atomic_json(study/'status.json',dict(status='BASELINE',pid=os.getpid(),training_started=False,real_flights_collected=False))
        baseline=launch(spec['baseline_command'],root,study/'baseline.log');code=baseline.wait()
        meta=read_json(Path(spec['baseline_rehearsal'])/'rehearsal.json',{})
        atomic_json(study/'baseline_result.json',dict(returncode=code,rehearsal=spec['baseline_rehearsal'],
            simulation_only=True,metadata=meta,real_flight_performance='UNKNOWN'))
        if code:raise RuntimeError('Unchanged-policy M1 rehearsal failed; inspect baseline.log before training.')
        for label,job in spec['jobs'].items():
            if (study/'STOP_REQUESTED').exists():raise InterruptedError('Study stopped before next condition.')
            directory=Path(job['directory'])
            if (directory/'STOP_REQUESTED').exists():raise InterruptedError('Queued condition was stopped by user.')
            process=launch(job['command'],root,directory/'console.log')
            atomic_json(study/'status.json',dict(status='TRAINING',condition=label,pid=os.getpid(),
                worker_pid=process.pid,directory=str(directory),training_started=True,real_flights_collected=False))
            while process.poll() is None:
                if (study/'STOP_REQUESTED').exists():(directory/'STOP_REQUESTED').touch()
                time.sleep(1)
            status=read_json(directory/'status.json',{})
            if process.returncode or status.get('status')!='COMPLETED':
                raise InterruptedError(f'{label} ended as {status.get("status","unknown")}; next condition will not launch.')
        results={label:dict(run=job['directory'],best_simulation_validation=read_json(Path(job['directory'])/'best_validation.json'),
            status=read_json(Path(job['directory'])/'status.json')) for label,job in spec['jobs'].items()}
        atomic_json(study/'simulation_results.json',dict(conditions=results,simulation_only=True,
            real_flight_performance='UNKNOWN: evaluate on new adp1 flights before any M2 fitting.'))
        atomic_json(study/'status.json',dict(status='COMPLETED',elapsed_s=time.perf_counter()-started,
            training_started=True,real_flights_collected=False))
    except Exception as error:
        atomic_json(study/'status.json',dict(status='STOPPED' if isinstance(error,InterruptedError) else 'FAILED',message=str(error)))
        raise
    finally:
        before=read_json(study/'protected_inputs.json');changed=[p for p,h in before.items() if not Path(p).is_file() or sha256_file(p)!=h]
        atomic_json(study/'preservation_check.json',dict(files=len(before),changed=changed,passed=not changed))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);parser.add_argument('--study')
    args=parser.parse_args()
    if args.action=='prepare':print(prepare(Path(__file__).resolve().parents[1]))
    else:run(args.study)
