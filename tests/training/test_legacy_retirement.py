"""The current pipeline must not load or launch retired policy trainers."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import pytest
import torch

from learning.pva_env import PVAEnvironment, defaults
from planning.pva_job import load_settings, prepare, validate_settings


@pytest.mark.parametrize('method', ['ppo', 'sac'])
def test_retired_methods_rejected_before_creating_jobs(tmp_path, method):
    with pytest.raises(ValueError, match='Only MPPI'):
        defaults(method)
    with pytest.raises(ValueError, match='Only MPPI'):
        load_settings(tmp_path, method)
    cfg=defaults('mppi');cfg['method']=method
    with pytest.raises(ValueError, match='Only MPPI'):
        prepare(tmp_path,cfg,'retired')
    with pytest.raises(ValueError, match='Only MPPI'):
        PVAEnvironment({},cfg,device='cpu')
    assert not list(tmp_path.iterdir())


def test_policy_checkpoint_and_objective_cannot_be_silently_reinterpreted(tmp_path):
    cfg=defaults('mppi')
    with pytest.raises(ValueError, match='checkpoint continuation was retired'):
        prepare(tmp_path,cfg,'retired',checkpoint=tmp_path/'old.pt')
    for key,value in [('ppo_objective',{'schema':'old'}),('policy_timing',{'decision_steps':3})]:
        invalid=deepcopy(cfg);invalid[key]=value
        with pytest.raises(ValueError,match='Legacy policy'):
            validate_settings(invalid)
    assert not list(tmp_path.iterdir())


def test_frozen_action_rollout_retains_observer_order_and_terminal_cutoff():
    env=PVAEnvironment.__new__(PVAEnvironment)
    env.steps=5;env.index=0;env.active=torch.tensor([True]);seen=[];received=[]
    def step(action,**kwargs):
        received.append(action.clone());env.index+=1
        if env.index==3:env.active.zero_()
    env.step=step;env.result=lambda: {'count':env.index}
    actions=torch.arange(15).reshape(1,5,3)
    assert env.rollout(actions=actions,observer=lambda e:seen.append(e.index))=={'count':3}
    assert seen==[0,1,2,3]
    torch.testing.assert_close(torch.stack(received,1),actions[:,:3])
    with pytest.raises(ValueError,match='Policy replay was retired'):
        env.rollout(policy=lambda _:None)


def test_active_pipeline_imports_no_policy_trainers():
    root=Path(__file__).resolve().parents[2]
    script='''import sys
import planning.pva_job, planning.correction_job, deployment.pva_rehearsal
import simulator.gui.pva_main_window
assert not any(n in sys.modules for n in ('run_ppo','run_sac','learning.simple_ppo','learning.simple_sac'))
'''
    result=subprocess.run([sys.executable,'-c',script],cwd=root,capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stdout+result.stderr
    for name in ('run_ppo.py','run_sac.py','learning/simple_ppo.py','learning/simple_sac.py'):
        assert not (root/name).exists()
