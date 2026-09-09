"""Bounded native M1 MPPI integration check. No training or aircraft interface."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experimental_data.io import atomic_json,sha256_file
from planning.cem_run import MPPI_DEFAULTS,prepare_job,run_job,export_package
from simulator.workflow import read_json


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--population',type=int,default=128)
    parser.add_argument('--iterations',type=int,default=6)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    output=Path(args.output).resolve()
    seed=root/'runs/rehearsals/20260908-203914-039721'
    protected=[root/'runs/ppo/20260908-195207-486249-seed655',seed,
        root/'data/model_candidates/20260908-adp0-M1',
        root/'rehearsal_csv_and_result_in_real_flight/20260908-195207-486249-seed655_best_validation/adp0']
    before={str(p):sha256_file(p) for folder in protected for p in folder.rglob('*') if p.is_file()}
    for p in (root/'config/cem.json',root/'config/launch_setup.json',root/'config/research_30hz/model.json'):
        before[str(p)]=sha256_file(p)
    settings=dict(MPPI_DEFAULTS,population=args.population,iterations=args.iterations,
        display_name='M1 MPPI bounded integration check',
        launch_setup=dict(initial_tracking_origin_m=[-2,0,1.255],target_position_m=[-1,0,1.1]))
    command=prepare_job(root,seed,output,settings)
    # Execute the immutable worker just as the GUI does.
    import subprocess
    with (output/'console.log').open('w',encoding='utf-8') as log:
        result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    unchanged=all(sha256_file(p)==h for p,h in before.items())
    check=dict(worker_returncode=result.returncode,protected_files=len(before),protected_unchanged=unchanged,
        model_source_sha256=sha256_file(root/MPPI_DEFAULTS['model_path']))
    if result.returncode==0 and (output/'rehearsal.json').exists():
        import numpy as np
        import zipfile
        meta=read_json(output/'rehearsal.json')
        with np.load(output/'rehearsal.npz') as arrays:
            csv=np.loadtxt(output/'fullstate_30hz.csv',delimiter=',',skiprows=1)
            np.testing.assert_allclose(csv[:,1:],arrays['commands'],rtol=0,atol=0)
            np.testing.assert_allclose(np.diff(csv[:,0]),1/30,rtol=0,atol=2e-15)
            assert np.isfinite(arrays['cable_positions_m']).all()
        assert meta['model_provenance']['source_sha256']==check['model_source_sha256']
        package=export_package(output,output.parent/(output.name+'.zip'))
        with zipfile.ZipFile(package) as archive:
            assert archive.read('fullstate_30hz.csv')==(output/'fullstate_30hz.csv').read_bytes()
        check.update(exact_csv_and_zip=True,metadata=meta,package=str(package))
    atomic_json(output/'integration_check.json',check)
    print(check,flush=True)
    assert unchanged,'Protected input changed during the check; inspect before continuing.'
    sys.exit(result.returncode)
