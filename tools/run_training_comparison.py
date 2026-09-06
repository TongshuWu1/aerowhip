"""Freeze, launch and automatically export a full-budget PPO/SAC comparison."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import os
import subprocess
import time
import zipfile
from simulator.workflow import read_json,atomic_json,canonical_json_hash,prepare_training,stamp
from tools.source_snapshot import archive_sources
from tools.plot_training_comparison import export

ROOT=Path(__file__).resolve().parents[1]
HIDDEN=getattr(subprocess,'CREATE_NO_WINDOW',0)


def prepare(batch):
    task=read_json(ROOT/'config/task.json')
    frequency=1/task['control_dt_s'];steps=round(task['episode_duration_s']/task['control_dt_s'])
    study=ROOT/'results'/f'{stamp()}-ppo-sac-500k'
    study.mkdir(parents=True)
    archive_sources(study)
    code=study/'code';code.mkdir()
    with zipfile.ZipFile(study/'training_source.zip') as archive:
        # Archive entries are generated solely from our project-relative sources.
        for entry in archive.infolist():
            if not (code/entry.filename).resolve().is_relative_to(code.resolve()):raise ValueError('Invalid source archive path')
        archive.extractall(code)
    for name in ('model','task','ppo','sac','baseline'):
        atomic_json(study/'configuration_before'/f'{name}.json',read_json(ROOT/f'config/{name}.json'))
    runs={}
    for algorithm in ('ppo','sac'):
        directory,command=prepare_training(ROOT,algorithm,seed=651,episodes=500000,batch=batch,device='cuda')
        shared=read_json(directory/'launch_config/ppo.json')
        shared['update_guard']['enabled']=False
        shared['deployment']['evaluate_final_holdout']=False
        shared['validation'].update(episodes=256,every_episodes=batch)
        shared['logging']['per_attempt']=True
        shared['status']=f'frozen_{frequency:g}hz_500k_comparison'
        atomic_json(directory/'launch_config/ppo.json',shared)
        if algorithm=='sac':
            sac=read_json(directory/'launch_config/sac.json')
            sac['validation'].update(episodes=256,every_episodes=batch)
            sac['logging']={'per_attempt':True}
            sac['sac']['updates_per_collection']=256*batch//1024
            sac['sac']['scale_updates_with_actual_batch']=True
            sac['sac']['replay_capacity']=max(262144,4*steps*batch)
            atomic_json(directory/'launch_config/sac.json',sac)
        configs={p.stem:read_json(p) for p in (directory/'launch_config').glob('*.json')}
        meta=read_json(directory/'run.json')
        meta.update(display_name=f'{algorithm.upper()} · 500,000 attempts · {frequency:g} Hz comparison',
            comparison_directory=str(study),config_sha256={k:canonical_json_hash(v) for k,v in configs.items()})
        atomic_json(directory/'run.json',meta)
        atomic_json(directory/'status.json',dict(status='QUEUED',algorithm=algorithm.upper(),episodes=0,
            target_episodes=500000,episodes_target=500000,collection_batch=batch,pid=0,stage='Waiting for full GPU'))
        command[2]=str(code/f'run_{algorithm}.py')
        runs[algorithm.upper()]=dict(directory=str(directory),command=command)
        (ROOT/f'runs/{algorithm}/ACTIVE_RUN.txt').write_text(str(directory)+'\n')
        # Snapshot root defaults too, so subsequent manual runs use the same experiment settings.
        if algorithm=='ppo':atomic_json(ROOT/'config/ppo.json',shared)
        else:atomic_json(ROOT/'config/sac.json',sac)
    for name in ('model','task','ppo','sac','baseline'):
        atomic_json(code/'config'/f'{name}.json',read_json(ROOT/f'config/{name}.json'))
    protocol=dict(schema='ppo_sac_comparison_v1',runs=runs,attempts_per_algorithm=500000,
        seed=651,collection_batch=batch,policy_frequency_hz=frequency,validation_episodes=256,validation_seed=90651,
        final_evaluation_seed=290653,final_evaluation_episodes=512,smoothing_attempts=5000,
        schedule='PPO then SAC; one GPU worker at a time',
        initialization='Same original CEM force prior; no trained actor weights; preceding search cost 8192 attempts',
        caveats=['One training seed; not a replicated algorithm ranking',
            'Parallel rows within a batch have arbitrary order, not successive policy updates',
            'Validation rollback disabled for PPO; validation not used for optimizer acceptance',
            'SAC updates scale with collection size to preserve 256 updates per 1024 attempts; replay holds four full plans batches',
            'Large collection batches change update cadence; 500000 attempts are not 500000 gradient steps',
            'Last collection uses exact remainder; published metrics include failures'],
        shared_model_sha256=canonical_json_hash(read_json(ROOT/'config/model.json')),
        shared_task_sha256=canonical_json_hash(read_json(ROOT/'config/task.json')),
        shared_reward_sha256=canonical_json_hash(read_json(ROOT/'config/ppo.json')['reward']))
    atomic_json(study/'protocol.json',protocol)
    export(study)
    (study/'README.md').write_text(
        '# PPO and SAC: 500,000 attempts each\n\n'
        f'Both start fresh from the same searched force sequence, at {frequency:g} Hz, with the same physics and reward. '
        'See protocol.json and each launch_config for exact settings.\n\n'
        'The queue gives PPO and then SAC the full GPU. plots/ contains PNG and vector PDF exports, refreshed '
        'after new completed batches or validation results. raw/ contains per-attempt CSV and validation JSON. '
        'A trailing 5,000-attempt mean is used for training curves, with shorter windows at the beginning. '
        'Validation points are unsmoothed. Within-batch attempt order is arbitrary.\n\n'
        'Final-checkpoint evaluation on 512 reserved scenarios runs automatically for both algorithms. '
        'All collapses are retained. This is one seed, not a publication-ready replicated ranking.\n\n'
        'To cancel the whole queue, create STOP_REQUESTED in this folder. The current worker will stop cooperatively '
        'and the next worker will not start. Using a training page Stop button stops that worker only.\n',encoding='utf-8')
    with (study/'queue.log').open('wb') as log:
        worker=subprocess.Popen([sys.executable,str(code/'tools/run_training_comparison.py'),'--run',str(study)],
            cwd=code,stdout=log,stderr=subprocess.STDOUT,creationflags=HIDDEN,
            env={**os.environ,'WHIP_PROJECT_ROOT':str(ROOT)})
    atomic_json(study/'launch.json',dict(pid=worker.pid,directory=str(study)))
    print(study,flush=True)


def run(study):
    study=study.resolve();protocol=read_json(study/'protocol.json')
    project=Path(os.environ.get('WHIP_PROJECT_ROOT',ROOT))
    state=dict(status='RUNNING',pid=os.getpid(),outcomes={},study=str(study))
    atomic_json(study/'status.json',state)
    signature=None
    for label,item in protocol['runs'].items():
        if (study/'STOP_REQUESTED').exists():break
        directory=Path(item['directory'])
        (project/f'runs/{label.lower()}/ACTIVE_RUN.txt').write_text(str(directory)+'\n')
        with (directory/'console.log').open('wb') as log:
            worker=subprocess.Popen(item['command'],cwd=study/'code',stdout=log,stderr=subprocess.STDOUT,creationflags=HIDDEN)
        state.update(active_algorithm=label,worker_pid=worker.pid)
        atomic_json(study/'status.json',state)
        while worker.poll() is None:
            if (study/'STOP_REQUESTED').exists():(directory/'STOP_REQUESTED').touch()
            current=tuple((p.name,p.stat().st_size) for p in directory.glob('validation_history.jsonl'))+tuple(p.name for p in (directory/'attempts').glob('*.npz'))
            if current!=signature:
                try:
                    export(study);signature=current
                except Exception as error:
                    print(f'Plot export will retry: {error}',flush=True)
            time.sleep(10)
        status=read_json(directory/'status.json',{})
        state['outcomes'][label]=dict(exit_code=worker.returncode,status=status.get('status'),episodes=status.get('episodes'))
        atomic_json(study/'status.json',state);export(study)
    if not (study/'STOP_REQUESTED').exists():
        for label,item in protocol['runs'].items():
            if state['outcomes'].get(label,{}).get('status')!='COMPLETED':continue
            output=study/'final_evaluation'/label.lower()
            command=[sys.executable,str(study/'code/tools/evaluate_selected_strike.py'),item['directory'],
                '--seed',str(protocol['final_evaluation_seed']),'--episodes','512','--output',str(output),
                '--purpose','Predeclared final-checkpoint evaluation; not used for tuning or selection']
            state.update(active_algorithm=f'{label} final evaluation');atomic_json(study/'status.json',state)
            with (study/f'{label.lower()}_final_evaluation.log').open('wb') as log:
                result=subprocess.run(command,cwd=study/'code',stdout=log,stderr=subprocess.STDOUT,creationflags=HIDDEN)
            state['outcomes'][label]['evaluation_exit_code']=result.returncode
    state['status']='STOPPED' if (study/'STOP_REQUESTED').exists() else (
        'COMPLETED' if len(state['outcomes'])==2 and all(x.get('status')=='COMPLETED' and x.get('evaluation_exit_code')==0 for x in state['outcomes'].values()) else 'NEEDS_ATTENTION')
    state['active_algorithm']=None
    atomic_json(study/'status.json',state);export(study)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path);parser.add_argument('--batch',type=int,default=32768)
    args=parser.parse_args()
    if args.run:run(args.run)
    else:prepare(args.batch)
