"""Bounded MPPI simulation and portable replay check; never fits or trains."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import subprocess
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
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    cfg=load_settings(ROOT,'mppi')
    cfg['model_path']='data/model_candidates/20260908-normalized-M1/model.json'
    job,command=prepare(ROOT,cfg,'MPPI PVA · historical M1 · simulation diagnostic')
    report=dict(job=str(job),model=cfg['model_path'],model_sha256=sha256_file(ROOT/cfg['model_path']),
        evidence='Historical M1 simulation only; no fitting, PPO or flight',gpu=torch.cuda.get_device_name(),torch=torch.__version__)
    atomic_json(out/'result.json',report)
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
