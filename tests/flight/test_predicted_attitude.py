"""Regression for a command-feasible trajectory that predicted a full roll."""
import numpy as np
import pytest
import torch
from simulator.drone_pose_response import rotation_exp
from simulator.predicted_envelope import attitude_valid


def test_tilt_is_geometric_and_respects_tracking_alignment():
    rotation=rotation_exp(torch.tensor([[0.,0.,0.],[np.pi/3,0.,0.],[np.pi*2/3,0.,0.],
                                        [0.,0.,np.pi]],dtype=torch.float64))
    eye=torch.eye(3,dtype=torch.float64)
    assert attitude_valid(rotation,eye,60).tolist()==[True,True,False,True]
    alignment=rotation_exp(torch.tensor([.7,-.2,.3],dtype=torch.float64))
    assert attitude_valid(rotation@alignment,alignment,60).tolist()==[True,True,False,True]
    rotation[0,0,0]=float('nan')
    assert not attitude_valid(rotation,eye,60)[0]


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_saved_full_roll_is_rejected_during_whip(device):
    from pathlib import Path
    import json
    from learning.pva_env import PVAEnvironment
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    root=Path(__file__).resolve().parents[2]
    job=root/'runs/mppi_pva/20260912-212707-357664'
    if not job.exists():pytest.skip('Local development counterexample is not distributed')
    cfg=json.loads((job/'settings.json').read_text(encoding='utf-8'))
    model=json.loads((job/'model.json').read_text(encoding='utf-8'))
    env=PVAEnvironment(model,cfg,root=job,device=device)
    with np.load(job/'plan.npz') as data:actions=env.tensor(data['normalized_jerk'])[None]
    result=env.rollout(actions=actions)
    assert bool(result['failed'][0])
    assert not bool(result['fold_valid'][0])
    assert float(env.termination_time[0])<.8
