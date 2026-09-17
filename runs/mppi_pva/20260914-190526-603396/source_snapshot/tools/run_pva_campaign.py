"""Explicit overnight fit -> fresh PPO -> rehearsal, with durable stage state.

No automatic worker restart. A failed/stopped fit, non-plateau NN fit, failed
training or export stops the campaign for review. Existing flight assets and
historical workers are never touched.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from experimental_data.io import atomic_json
from simulator.workflow import read_json
from planning.pva_job import prepare,load_settings

ROOT=Path(__file__).resolve().parents[1]


def prepare_campaign(folder,fit):
    folder=Path(folder).resolve();fit=Path(fit).resolve();folder.mkdir(parents=True,exist_ok=False)
    cfg=load_settings(ROOT,'ppo');cfg['model_path']=str(fit/'candidate/model.json')
    atomic_json(folder/'ppo_settings.json',cfg)
    atomic_json(folder/'protocol.json',dict(fit=str(fit),created=datetime.now().isoformat(),
        authorized='Fresh direct-PVA PPO after completed normalized-adp0 bootstrap; user requested autonomous overnight completion',
        model_contract='loaded_drone_cable_pva_model_v1',historical_checkpoint_resume=False))
    atomic_json(folder/'status.json',dict(status='prepared',stage='awaiting fit'))


def run(folder):
    folder=Path(folder).resolve();lock=folder/'worker.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    child=None
    def status(stage,**more):
        atomic_json(folder/'status.json',dict(status='running',stage=stage,pid=os.getpid(),**more));print(stage,flush=True)
    def stopped():
        if (folder/'STOP').exists():raise InterruptedError('Campaign stopped by request')
    def execute(command,stage,job=None):
        nonlocal child
        stopped()
        with (folder/'stages.log').open('ab') as output:
            child=subprocess.Popen(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            status(stage,child_pid=child.pid,training_job=str(job) if job else None)
            while child.poll() is None:
                if (folder/'STOP').exists():
                    if job:(job/'STOP').touch()
                    else:child.terminate()
                time.sleep(5)
            stopped()
            if child.returncode:raise RuntimeError(f'{stage} failed with exit code {child.returncode}; inspect stages.log')
    try:
        if read_json(folder/'status.json')['status']!='prepared':raise ValueError('Campaign already started; inspect it rather than duplicate')
        protocol=read_json(folder/'protocol.json');fit=Path(protocol['fit'])
        status('Waiting for complete-whip model fit')
        while True:
            stopped();state=read_json(fit/'status.json',{})
            if state.get('status')=='completed':break
            if state.get('status') in ('failed','stopped'):raise RuntimeError('Model fit needs review: '+state.get('error',state['status']))
            time.sleep(15)
        for path in (fit/'drone/stopping.json',fit/'cable/residual_full_whip/stopping.json'):
            if not read_json(path).get('converged'):raise RuntimeError('Model reached a safety ceiling rather than plateau: '+str(path))
        candidate=fit/'candidate/model.json';model=read_json(candidate)
        if model.get('schema')!=protocol['model_contract'] or not model.get('provenance',{}).get('fit_complete'):raise ValueError('Expected completed cold PVA model')
        diagnostics=read_json(candidate.parent/'training_diagnostics.json')
        if len(diagnostics.get('takes',{}))!=5:raise ValueError('Five normalized training-flight diagnostics required')
        for take in diagnostics['takes'].values():
            for key in ('drone','tip'):
                value=take[key].get('rmse_m')
                if value is None or not __import__('math').isfinite(value):raise ValueError('Incomplete coupled model diagnostic')
        cfg=read_json(folder/'ppo_settings.json')
        job,command=prepare(ROOT,cfg,'M0 PVA · normalized adp0 · fresh PPO')
        atomic_json(folder/'training_job.json',dict(job=str(job),command=command))
        execute(command,'Fresh PVA PPO training',job)
        result=read_json(job/'result.json')
        if result.get('stop_reason')!='reward_plateau':raise RuntimeError('PPO reached its review ceiling; inspect improvement before extending')
        output=ROOT/'runs/rehearsals_pva'/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-M0-PPO')
        atomic_json(folder/'rehearsal_job.json',dict(output=str(output)))
        command=[sys.executable,'-u',str(job/'source_snapshot/tools/rehearse_pva.py'),'--job',str(job),
            '--checkpoint',str(job/'checkpoints/best.pt'),'--output',str(output)]
        execute(command,'Best PVA policy rehearsal')
        from deployment.pva_rehearsal import export_package
        destination=ROOT/'exports'/('M0-PVA-'+job.name+'.zip');destination.parent.mkdir(exist_ok=True)
        export_package(output,destination)
        atomic_json(folder/'status.json',dict(status='completed',training_job=str(job),rehearsal=str(output),package=str(destination),
            evidence='Simulation only; review in UI before new flight collection'))
    except BaseException as error:
        atomic_json(folder/'status.json',dict(status='stopped' if isinstance(error,InterruptedError) else 'needs_attention',error=str(error)))
        raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--fit',type=Path)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.prepare:
        if not args.fit:parser.error('--fit is required when preparing')
        prepare_campaign(args.campaign,args.fit)
    if args.run:run(args.campaign)
