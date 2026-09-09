from copy import deepcopy
import json
import numpy as np
import pytest
from planning.cem_objective import REWARD_DEFAULTS,reward_components,resolve_reward,task_with_settings


def result():
    return {k:np.asarray(v) for k,v in dict(success=[True,False],hit_time=[.8,np.nan],
        distance=[.03,.4],closest_speed=[5.,2.],peak=[2.7,2.5],invalid_contact=[False,True]).items()}


def test_default_weights_reproduce_previous_formula():
    r=result();duration=[1,2];jerk=np.array([100.,20000.])
    expected=(1000*r['success']-100*np.clip(r['distance'],0,10)
              +10*np.clip(r['closest_speed']/4,0,1)-25*np.array([.8,2])
              -.001*np.minimum(jerk,10000)-100*r['invalid_contact']-20*np.maximum(r['peak']-2.6,0))
    np.testing.assert_allclose(sum(reward_components(r,duration,jerk,4).values()),expected,atol=1e-12)


@pytest.mark.parametrize('weight,component',[('success_bonus','success'),('distance_weight','distance'),
    ('speed_weight','speed'),('time_weight_per_s','time'),('jerk_weight','jerk'),
    ('invalid_contact_penalty','invalid_contact'),('height_weight','height')])
def test_each_reward_weight_controls_only_its_term(weight,component):
    base=reward_components(result(),[1,2],[100,200],4)
    changed=reward_components(result(),[1,2],[100,200],4,{weight:0})
    np.testing.assert_array_equal(changed[component],[0,0])
    for key in base:
        if key!=component:np.testing.assert_array_equal(changed[key],base[key])


@pytest.mark.parametrize('bad',[{'time_weight_per_s':-1},{'jerk_weight':float('nan')},
    {'distance_cap_m':0},{'surprise':2}])
def test_bad_reward_settings_fail_explicitly(bad):
    with pytest.raises(ValueError):resolve_reward(bad)


def test_hit_criteria_are_copied_and_direction_normalized():
    original=dict(success=dict(tip_target_distance_m=.05,minimum_directed_tip_speed_m_s=4,
        maximum_tip_velocity_to_desired_direction_error_deg=45,first_contact_only=True,
        tip_must_enter_before_other_markers=True),desired_strike_direction_world=[1,0,0])
    before=deepcopy(original)
    changed=task_with_settings(original,dict(success=dict(tip_target_distance_m=.08),desired_strike_direction_world=[0,2,0]))
    assert original==before and changed['success']['tip_target_distance_m']==.08
    assert changed['desired_strike_direction_world']==[0,1,0]
    with pytest.raises(ValueError):task_with_settings(original,{'desired_strike_direction_world':[0,0,0]})


@pytest.mark.parametrize('launch_source',['saved','explicit','arguments'])
def test_prepare_snapshots_effective_reward_and_task_without_editing_seed(tmp_path,monkeypatch,launch_source):
    from planning.cem_run import prepare_job
    from experimental_data.io import atomic_json
    import planning.cem_run as run
    import simulator.research_config as config
    monkeypatch.setattr(run,'snapshot_assets',lambda model,output:deepcopy(model))
    monkeypatch.setattr(config,'validate_research_contract',lambda *args:None)
    root=tmp_path/'project';root.mkdir();seed=root/'seed';seed.mkdir()
    for name in ('run_ppo.py','requirements.txt'):(root/name).write_text('')
    model=dict(recorded_data={'optitrack_to_attachment_offset_body_m':[0,0,-.055]},
        motion_residual={'checkpoint':'cable.pt'},fullstate_execution={'checkpoint':'drone.json'})
    task=dict(success=dict(tip_target_distance_m=.05,minimum_directed_tip_speed_m_s=4,
        maximum_tip_velocity_to_desired_direction_error_deg=45),desired_strike_direction_world=[1,0,0])
    for name,value in [('model',model),('task',task),('ppo',{}),('rehearsal',dict(schema='research_fullstate_30hz_v1',
        whip_end_s=1,initial_tracking_origin_m=[-2,0,1.255],target_position_m=[-1,0,1.1]))]:atomic_json(seed/(name+'.json'),value)
    np.savez(seed/'rehearsal.npz',commands=np.zeros((31,11)))
    launch=dict(initial_tracking_origin_m=[-2.5,.3,1.4],target_position_m=[-.8,.2,1.15])
    atomic_json(root/'config/launch_setup.json',dict(initial_tracking_origin_m=[8,8,8],target_position_m=[9,9,9]))
    atomic_json(root/'config/ppo.json',dict(preserved=True))
    atomic_json(root/'config/cem.json',dict(launch_setup=launch if launch_source=='saved' else
        dict(initial_tracking_origin_m=[-2,0,1.255],target_position_m=[-1,0,1.1])))
    before={p:p.read_bytes() for p in [*seed.iterdir(),*(root/'config').iterdir()]}
    output=root/'new'
    settings=dict(reward={'time_weight_per_s':75},success={'tip_target_distance_m':.07},desired_strike_direction_world=[0,3,0])
    if launch_source=='explicit':settings['launch_setup']=launch
    positions=dict(origin=launch['initial_tracking_origin_m'],target=launch['target_position_m']) if launch_source=='arguments' else {}
    prepare_job(root,seed,output,settings,**positions)
    saved=json.loads((output/'cem.json').read_text());frozen_task=json.loads((output/'task.json').read_text())
    assert saved['reward']['time_weight_per_s']==75 and saved['reward']['success_bonus']==1000
    assert frozen_task['success']['tip_target_distance_m']==.07 and saved['success']==frozen_task['success']
    assert frozen_task['desired_strike_direction_world']==[0,1,0]
    assert saved['launch_setup']==launch and frozen_task['target_position_m']==launch['target_position_m']
    np.testing.assert_allclose(frozen_task['initial_root_position_m'],[-2.5,.3,1.345],atol=1e-12)
    assert all(p.read_bytes()==value for p,value in before.items())
    assert json.loads((seed/'task.json').read_text())==task
