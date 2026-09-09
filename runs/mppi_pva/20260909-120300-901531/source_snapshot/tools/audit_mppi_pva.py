"""Bounded MPPI simulation and portable replay check; never fits or trains."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import subprocess
import shutil
import zipfile
import numpy as np
import torch
from planning.pva_job import load_settings,prepare
from deployment.pva_rehearsal import export_package
from experimental_data.io import atomic_json,sha256_file
from simulator.workflow import read_json

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--replay-job',type=Path,help='Freeze current recovery code around an unchanged completed plan; no optimizer restart')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    cfg=load_settings(ROOT,'mppi')
    cfg['model_path']='data/model_candidates/20260908-normalized-M1/model.json'
    if args.replay_job:
        parent=args.replay_job.resolve()
        if read_json(parent/'status.json')['status'] not in ('completed','stopped'):raise ValueError('Parent plan must be stopped/completed')
        cfg=read_json(parent/'settings.json');cfg['model_path']=str(parent/'model.json')
    job,command=prepare(ROOT,cfg,'MPPI PVA · historical M1 · simulation diagnostic')
    report=dict(job=str(job),model=cfg['model_path'],model_sha256=sha256_file(ROOT/cfg['model_path']),
        evidence='Historical M1 simulation only; no fitting, PPO or flight',gpu=torch.cuda.get_device_name(),torch=torch.__version__)
    atomic_json(out/'result.json',report)
    if args.replay_job:
        shutil.copy2(parent/'plan.npz',job/'plan.npz')
        report.update(parent_job=str(parent),optimizer_rerun=False,plan_sha256=sha256_file(job/'plan.npz'))
        assert report['plan_sha256']==sha256_file(parent/'plan.npz')
        identity=read_json(job/'identity.json');identity.update(parent_job=str(parent),parent_plan_sha256=report['plan_sha256'],
            name='MPPI PVA · unchanged strike · updated PVA recovery',operation='replay verification only; optimizer not rerun',
            optimization_history_source=str(parent))
        atomic_json(job/'identity.json',identity)
        if (parent/'history.json').exists():shutil.copy2(parent/'history.json',job/'history.json')
        result=read_json(parent/'result.json');result.update(plan=str(job/'plan.npz'),optimizer_rerun=False,parent_job=str(parent))
        atomic_json(job/'result.json',result);atomic_json(job/'status.json',dict(result,status='completed',stage='Unchanged saved strike; updated recovery'))
    else:
        with (out/'optimization.log').open('w',encoding='utf-8') as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    report['optimization']=read_json(job/'result.json');atomic_json(out/'result.json',report)
    rehearsal=ROOT/'runs/rehearsals_pva'/(job.name+'-mppi-diagnostic')
    command=[sys.executable,str(job/'source_snapshot/tools/rehearse_pva.py'),'--job',str(job),'--output',str(rehearsal)]
    with (out/'rehearsal.log').open('w',encoding='utf-8') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    if result.returncode:
        report['rehearsal_error']=(out/'rehearsal.log').read_text(encoding='utf-8')
        atomic_json(out/'result.json',report);return
    report['rehearsal']=str(rehearsal);report['metadata']=read_json(rehearsal/'rehearsal.json')
    package=out/'MPPI-PVA-simulation-diagnostic.zip';export_package(rehearsal,package)
    extracted=out/'portable';extracted.mkdir()
    with zipfile.ZipFile(package) as archive:archive.extractall(extracted)
    regenerated=out/'regenerated'
    with (out/'portable.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,str(extracted/'tools/rehearse_pva.py'),'--job',str(extracted),'--output',str(regenerated)],
            check=True,stdout=log,stderr=subprocess.STDOUT,cwd=extracted)
    report['portable_csv_identical']=sha256_file(rehearsal/'fullstate_30hz.csv')==sha256_file(regenerated/'fullstate_30hz.csv')
    with np.load(rehearsal/'rehearsal.npz') as original,np.load(regenerated/'rehearsal.npz') as replay:
        report['portable_max_difference']={key:float(np.max(np.abs(original[key]-replay[key]))) for key in original.files}
    atomic_json(out/'result.json',report)
    assert report['portable_csv_identical']
    assert max(report['portable_max_difference'].values())<1e-8

if __name__=='__main__':main()
