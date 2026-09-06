from copy import deepcopy
import shutil
import pytest
import torch
from simulator.workflow import atomic_json
from tools.plot_training_comparison import read_attempts
import run_ppo
import run_sac


@pytest.mark.parametrize('algorithm',['ppo','sac'])
def test_small_comparison_logs_every_attempt_after_checkpoint(tmp_path,monkeypatch,algorithm):
    torch.set_num_threads(1)
    for package in ('learning','simulator'):
        shutil.copytree(run_sac.ROOT/package,tmp_path/package,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('run_sac.py','run_ppo.py'):shutil.copy2(run_sac.ROOT/name,tmp_path/name)
    model,task,shared,sac=deepcopy(run_sac.configs())
    task['episode_duration_s']=.1;task['control_dt_s']=.05
    model['simulation']['dt_s']=.01
    shared['bootstrap']=None;sac['bootstrap']=None
    shared['logging']['per_attempt']=True;sac['logging']={'per_attempt':True}
    shared['deployment'].update(recovery_duration_s=.5,evaluate_final_holdout=False)
    shared['ppo'].update(hidden_dim=8,minibatch_transitions=2,update_epochs=1)
    shared['update_guard']['enabled']=False
    shared['validation'].update(episodes=2,every_episodes=3)
    sac['sac'].update(hidden_dim=8,replay_capacity=64,minibatch_transitions=2,updates_per_collection=1,critic_warmup_updates=0)
    sac['validation'].update(episodes=2,every_episodes=3)
    for name,value in [('model',model),('task',task),('ppo',shared),('sac',sac)]:atomic_json(tmp_path/'config'/f'{name}.json',value)
    monkeypatch.setattr(run_sac,'ROOT',tmp_path);monkeypatch.setattr(run_ppo,'ROOT',tmp_path)
    output=tmp_path/'run'
    if algorithm=='ppo':
        run_ppo.train(device_name='cpu',requested_episodes=5,batch_size=3,artifact=output,resume_checkpoint=None,config_directory=tmp_path/'config')
    else:
        run_sac.run(output,5,device_name='cpu',batch_override=3,config_directory=tmp_path/'config')
    data=read_attempts(output)
    assert data['episode'].tolist()==[1,2,3,4,5]
    assert data['batch_end'].tolist()==[3,3,3,5,5]
    saved=torch.load(output/'checkpoints/latest.pt',weights_only=False)
    assert saved['episodes']==5
