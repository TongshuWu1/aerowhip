from copy import deepcopy
from pathlib import Path
import json
import pytest
import torch
from learning.reward_plateau import RewardPlateau


def test_task_setup_amendment_preserves_source_and_resets_history(tmp_path):
    import run_ppo
    from simulator.workflow import prepare_training,read_json
    from experimental_data.io import atomic_json
    model,task,config=deepcopy(run_ppo.load_configs());model.pop('fullstate_execution',None)
    parent=tmp_path/'parent';cp=parent/'checkpoints/best_validation.pt';cp.parent.mkdir(parents=True)
    torch.save(dict(episodes=293888),cp)
    for name,value in zip(('model','task','ppo'),(model,task,config)):atomic_json(parent/f'{name}.json',value)
    atomic_json(tmp_path/'config/ppo.json',config)
    originals={p:p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    options=dict(seed=655,episodes=314368,batch=2048,device='cpu',resume=cp,keep_optimizer_state=True,
        task_overrides=dict(initial_root_position_m=[-1.993345,0,1.2],target_position_m=[-1,0,1.1]))
    with pytest.raises(ValueError,match='task setup'):
        prepare_training(tmp_path,'ppo',keep_stopping_history=True,**options)
    run,_=prepare_training(tmp_path,'ppo',**options)
    actual=read_json(run/'launch_config/task.json');saved=read_json(run/'launch_config/ppo.json')
    assert actual['initial_root_position_m']==[-1.993345,0,1.2] and actual['target_position_m']==[-1,0,1.1]
    assert actual['success']==task['success'] and actual['episode_duration_s']==task['episode_duration_s']
    assert saved['reward']==config['reward'] and saved['ppo']==config['ppo']
    assert not saved['deployment']['reset_optimizer_on_resume'] and 'reward_plateau_resume' not in saved
    assert read_json(run/'run.json')['task_amendment']['stopping_history_reset']
    assert all(p.read_bytes()==value for p,value in originals.items())
    for bad in [dict(episode_duration_s=2),dict(target_position_m=[0,0]),dict(target_position_m=[0,0,float('nan')])]:
        with pytest.raises(ValueError):prepare_training(tmp_path,'ppo',**dict(options,task_overrides=bad))


def test_stopping_history_matches_uninterrupted_with_larger_next_batch():
    settings=dict(enabled=True,window_evaluations=4,patience_episodes=80,
        minimum_additional_episodes=100,minimum_reward_improvement=2.,minimum_relative_improvement=.01)
    records=[dict(training_episodes=s,mean_episode_reward=200.,evaluation_id=str(s)) for s in (0,20,40,60)]
    uninterrupted=RewardPlateau(settings,start_episodes=0)
    for record in records:uninterrupted.observe(record)
    history=dict(schema='reward_plateau_resume_v1',original_start_episodes=0,
        checkpoint_episodes=60,records=records)
    resumed=RewardPlateau.from_history(settings,start_episodes=60,history=history)
    before=list(resumed.values)
    resumed.observe(dict(training_episodes=60,mean_episode_reward=200.,evaluation_id='repeated-start'))
    assert list(resumed.values)==before
    for s in (100,140):  # Twice as many attempts between optimizer/evaluation cycles.
        record=dict(training_episodes=s,mean_episode_reward=200.,evaluation_id=str(s))
        actual=resumed.observe(record);expected=uninterrupted.observe(record)
        for key in ('should_stop','rolling_mean_reward','last_improvement_episodes','additional_episodes'):
            assert actual[key]==expected[key]
    assert actual['should_stop'] and actual['start_episodes']==0


def test_stopping_history_rejects_wrong_checkpoint_or_future_validation():
    history=dict(schema='reward_plateau_resume_v1',original_start_episodes=0,
        checkpoint_episodes=20,records=[dict(training_episodes=40,mean_episode_reward=2.)])
    with pytest.raises(ValueError,match='checkpoint'):
        RewardPlateau.from_history({},start_episodes=30,history=history)
    with pytest.raises(ValueError,match='beyond'):
        RewardPlateau.from_history({},start_episodes=20,history=history)


def test_prepare_larger_batch_snapshots_history_and_preserves_training_semantics(tmp_path):
    import run_ppo
    from simulator.workflow import prepare_training,read_json
    from experimental_data.io import atomic_json,sha256_file
    model,task,config=deepcopy(run_ppo.load_configs());model.pop('fullstate_execution',None)
    config['early_stopping']=dict(enabled=True,patience_episodes=20000,window_evaluations=20)
    parent=tmp_path/'runs/ppo/parent'
    for n,v in zip(('model','task','ppo'),(model,task,config)):atomic_json(parent/f'{n}.json',v)
    atomic_json(tmp_path/'config/ppo.json',config)
    atomic_json(parent/'reward_convergence.json',dict(start_episodes=0))
    cp=parent/'checkpoints/latest.pt';cp.parent.mkdir();torch.save(dict(episodes=7168),cp)
    records=[dict(training_episodes=s,mean_episode_reward=float(s),evaluation_id=str(s)) for s in (0,6144,7168,8192)]
    (parent/'validation_history.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n')
    originals={p:p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    run,command=prepare_training(tmp_path,'ppo',seed=config['seed'],episodes=500000,batch=2048,device='cpu',
        resume=cp,keep_optimizer_state=True,keep_stopping_history=True)
    saved=read_json(run/'launch_config/ppo.json')
    assert saved['reward']==config['reward'] and saved['ppo']==config['ppo']
    assert saved['training']['collection_batch']==2048
    assert saved['validation']['every_episodes']==2048
    assert not saved['deployment']['reset_optimizer_on_resume']
    history=saved['reward_plateau_resume']
    assert history['original_start_episodes']==0
    assert history['source_checkpoint_sha256']==sha256_file(cp)
    assert [r['training_episodes'] for r in history['records']]==[0,6144,7168]
    assert all(p.read_bytes()==v for p,v in originals.items())
    monitor=RewardPlateau.from_history(saved['early_stopping'],start_episodes=7168,history=history)
    assert monitor.start==0 and monitor.last_step==7168


def test_actual_resume_keeps_optimizer_and_stopping_clock(tmp_path,monkeypatch):
    import run_ppo
    from simulator.workflow import prepare_training,read_json
    from experimental_data.io import atomic_json
    torch.set_num_threads(1)
    model,task,config=deepcopy(run_ppo.load_configs());model.pop('fullstate_execution',None)
    task['episode_duration_s']=.1;config['bootstrap']=None
    config['deployment']['enabled']=False;config['ppo'].update(hidden_dim=16,update_epochs=1)
    config['validation'].update(enabled=True,episodes=2,every_episodes=2)
    config['update_guard']['enabled']=False
    config['early_stopping']=dict(enabled=True,window_evaluations=4,patience_episodes=8,minimum_additional_episodes=10)
    for n,v in zip(('model','task','ppo'),(model,task,config)):atomic_json(tmp_path/'config'/f'{n}.json',v)
    monkeypatch.setattr(run_ppo,'write_active_run',lambda *args:None)
    monkeypatch.setattr(run_ppo,'evaluate',lambda *args,**kw:dict(
        success_rate=.5,mean_episode_reward=200.,mean_point_displacement_cost_integral_s=1.))
    parent=run_ppo.train(device_name='cpu',requested_episodes=4,batch_size=2,
        artifact=tmp_path/'parent',resume_checkpoint=None,config_directory=tmp_path/'config')
    # The constant evaluator above bypasses the physical replay recorder. Supply
    # its three actual observed results in that recorder's normal history form.
    (parent/'validation_history.jsonl').write_text('\n'.join(json.dumps(dict(
        training_episodes=s,mean_episode_reward=200.,evaluation_id=str(s))) for s in (0,2,4))+'\n')
    source=parent/'checkpoints/latest.pt'
    child,command=prepare_training(tmp_path,'ppo',seed=config['seed'],episodes=100,batch=4,device='cpu',
        resume=source,keep_optimizer_state=True,keep_stopping_history=True)
    run_ppo.train(device_name='cpu',requested_episodes=100,batch_size=4,artifact=child,
        resume_checkpoint=source,config_directory=child/'launch_config')
    status=read_json(child/'status.json')
    assert status['episodes']==16 and status['stop_reason']=='Validation reward plateau'
    assert status['reward_convergence']['start_episodes']==0
    a=torch.load(source,map_location='cpu',weights_only=False)
    b=torch.load(child/'checkpoints/best_reward.pt',map_location='cpu',weights_only=False)
    def same(a,b):
        if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:same(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for x,y in zip(a,b):same(x,y)
        else:assert a==b
    for key in ('policy','value','policy_optimizer','value_optimizer','gradient_updates','episodes'):
        same(a[key],b[key])


def test_reward_amendment_preserves_parent_and_resets_stopping_history(tmp_path):
    import run_ppo
    from simulator.workflow import prepare_training,read_json
    from experimental_data.io import atomic_json,sha256_file
    model,task,config=deepcopy(run_ppo.load_configs());model.pop('fullstate_execution',None)
    config['reward']['time_to_success_weight_per_s']=.2
    config['early_stopping']=dict(enabled=True,patience_episodes=20000,minimum_additional_episodes=20000)
    config['reward_plateau_resume']=dict(schema='reward_plateau_resume_v1',original_start_episodes=0,
        checkpoint_episodes=39936,records=[dict(training_episodes=39936,mean_episode_reward=144.)])
    parent=tmp_path/'runs/ppo/parent'
    for n,v in zip(('model','task','ppo'),(model,task,config)):atomic_json(parent/f'{n}.json',v)
    atomic_json(tmp_path/'config/ppo.json',config)
    cp=parent/'checkpoints/latest.pt';cp.parent.mkdir();torch.save(dict(episodes=39936),cp)
    originals={p:p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    options=dict(seed=config['seed'],episodes=500000,batch=2048,device='cpu',resume=cp,
        keep_optimizer_state=True,reward_overrides=dict(time_to_success_weight_per_s=10.))
    with pytest.raises(ValueError,match='changed reward'):
        prepare_training(tmp_path,'ppo',keep_stopping_history=True,**options)
    run,_=prepare_training(tmp_path,'ppo',**options)
    saved=read_json(run/'launch_config/ppo.json')
    expected=deepcopy(config['reward']);expected['time_to_success_weight_per_s']=10.
    assert saved['reward']==expected and saved['ppo']==config['ppo']
    assert 'reward_plateau_resume' not in saved
    assert not saved['deployment']['reset_optimizer_on_resume']
    monitor=RewardPlateau.from_history(saved['early_stopping'],start_episodes=39936,
        history=saved.get('reward_plateau_resume'))
    assert monitor.start==39936 and len(monitor.values)==0
    amendment=read_json(run/'run.json')['reward_amendment']
    assert amendment['changes']==dict(time_to_success_weight_per_s=dict(before=.2,after=10.))
    assert amendment['parent_checkpoint_sha256']==sha256_file(cp)
    assert all(p.read_bytes()==v for p,v in originals.items())


def test_legacy_sac_preparation_does_not_require_ppo_reward_fields(tmp_path):
    from simulator.workflow import prepare_training,read_json
    from experimental_data.io import atomic_json
    root=Path(__file__).resolve().parents[2]
    for name in ('model','task','ppo','sac'):
        payload=read_json(root/'config'/f'{name}.json')
        if name=='model':payload.pop('fullstate_execution',None)
        atomic_json(tmp_path/'config'/f'{name}.json',payload)
    run,_=prepare_training(tmp_path,'sac',seed=123,episodes=8,batch=4,device='cpu')
    assert 'reward' not in read_json(run/'launch_config/sac.json')


def test_execution_termination_amendment_is_frozen_and_cannot_mix_history(tmp_path):
    from simulator.workflow import prepare_training,read_json
    from experimental_data.io import atomic_json
    root=Path(__file__).resolve().parents[2];parent=tmp_path/'parent'
    for n in ('model','task','ppo'):
        payload=read_json(root/'config'/f'{n}.json')
        if n=='model':payload.pop('fullstate_execution',None)
        atomic_json(parent/f'{n}.json',payload)
    atomic_json(tmp_path/'config/ppo.json',read_json(parent/'ppo.json'))
    cp=parent/'checkpoints/latest.pt';cp.parent.mkdir();torch.save(dict(episodes=44032),cp)
    options=dict(seed=656,episodes=500000,batch=2048,device='cpu',resume=cp,keep_optimizer_state=True,
        deployment_overrides=dict(termination='execution_success_or_timeout'))
    with pytest.raises(ValueError,match='deployment semantics'):
        prepare_training(tmp_path,'ppo',keep_stopping_history=True,**options)
    run,_=prepare_training(tmp_path,'ppo',**options)
    assert read_json(run/'launch_config/ppo.json')['deployment']['termination']=='execution_success_or_timeout'
    assert read_json(run/'run.json')['deployment_amendment']['termination']['after']=='execution_success_or_timeout'
    assert 'termination' not in read_json(parent/'ppo.json')['deployment']
