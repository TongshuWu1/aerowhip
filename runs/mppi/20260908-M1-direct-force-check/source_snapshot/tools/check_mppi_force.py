"""Bounded direct-force MPPI integration audit, never a flight sender."""
from pathlib import Path
import sys
import subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planning.mppi_force_run import prepare_job,DEFAULTS
from planning.cem_run import export_package
from simulator.workflow import read_json
from experimental_data.io import sha256_file,atomic_json
import numpy as np
import zipfile

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1];seed=root/'runs/rehearsals/20260908-203914-039721'
    output=root/'runs/mppi/20260908-M1-direct-force-check'
    folders=[seed,root/'runs/ppo/20260908-195207-486249-seed655',root/'data/model_candidates/20260908-adp0-M1',
        root/'rehearsal_csv_and_result_in_real_flight/20260908-195207-486249-seed655_best_validation/adp0']
    before={str(p):sha256_file(p) for folder in folders for p in folder.rglob('*') if p.is_file()}
    settings=dict(DEFAULTS,**read_json(root/'config/mppi.json'));settings.update(population=16,iterations=2,display_name='M1 MPPI direct-force integration check')
    command=prepare_job(root,seed,output,settings)
    with (output/'console.log').open('w',encoding='utf-8') as log:
        process=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    report=dict(returncode=process.returncode,protected_files=len(before),protected_unchanged=all(sha256_file(p)==h for p,h in before.items()),
        no_training=True,no_flight=True)
    if process.returncode==0:
        meta=read_json(output/'rehearsal.json')
        with np.load(output/'rehearsal.npz') as data:
            csv=np.loadtxt(output/'fullstate_30hz.csv',delimiter=',',skiprows=1)
            np.testing.assert_array_equal(csv[:,1:],data['commands'])
            force=np.loadtxt(output/'virtual_force_30hz.csv',delimiter=',',skiprows=1)
            np.testing.assert_array_equal(force[:,1:],data['virtual_force_n'][::5])
        package=export_package(output,output.parent/(output.name+'.zip'))
        with zipfile.ZipFile(package) as archive:assert archive.read('fullstate_30hz.csv')==(output/'fullstate_30hz.csv').read_bytes()
        report.update(metadata=meta,exact_csv_and_zip=True)
    atomic_json(output/'verification.json',report);print(report)
    assert report['protected_unchanged']
    sys.exit(process.returncode)
