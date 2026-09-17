"""Run the full reviewed fit, preserving M6's stage-specific plateau settings."""
from pathlib import Path
from copy import deepcopy
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experimental_data import whip_full_continuation as continuation
from experimental_data.whip_full_fit import fit
from experimental_data.io import sha256_file
from simulator.workflow import read_json

if __name__=='__main__':
    job=ROOT/'runs/adaptation/M7-horizontal-20260915'
    contract=read_json(job/'protocol.json')['full_update']
    assert contract['runner_sha256']==sha256_file(Path(__file__))
    original=continuation.train_to_plateau
    def stage_plateau(net,objective,settings,folder,job,label,**kwargs):
        effective=deepcopy(settings)
        key=label+'_stopping'
        if key in effective:effective['residual_stopping']=deepcopy(effective[key])
        return original(net,objective,effective,folder,job,label,**kwargs)
    continuation.train_to_plateau=stage_plateau
    result=fit(job)
    print('M7 full fit completed:',result['comparison'],flush=True)
