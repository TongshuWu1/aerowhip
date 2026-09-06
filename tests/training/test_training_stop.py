from copy import deepcopy
import shutil
import pytest
import torch
import run_ppo
import run_sac
from learning import deployment_rollout
from learning import training_control
from simulator.workflow import atomic_json,read_json


@pytest.mark.parametrize('algorithm',['ppo','sac'])
@pytest.mark.parametrize('stage',['validation','planning','recovery','optimizer'])
def test_stop_preserves_durable_checkpoint(tmp_path,monkeypatch,algorithm,stage):
    torch.set_num_threads(1)
    model,task,shared,sac=deepcopy(run_sac.configs())
    shared['bootstrap']=None
    sac['bootstrap']=None
    task['episode_duration_s']=.2
    # Guaranteed first-step contact exercises execution/recovery, not refusal.
    from simulator.live_flight import LiveFlight
    initial=LiveFlight(model,task,shared).state
    task['target_position_m']=initial.positions_m[0,-1].tolist()
    task['success'].update(minimum_directed_tip_speed_m_s=0.,
        maximum_tip_velocity_to_desired_direction_error_deg=180.)
    shared['deployment'].update(nominal_fraction=1.,recovery_duration_s=.5)
    shared['validation'].update(enabled=True,episodes=2)
    shared['ppo'].update(minibatch_transitions=2,update_epochs=2)
    sac['validation'].update(episodes=2,every_episodes=4)
    # Keep multiple updates so the stop interrupts an unfinished collection.
    # Workstation batch scaling would reduce this tiny fixture to one update.
    sac['sac'].update(hidden_dim=32,replay_capacity=64,minibatch_transitions=2,
                      updates_per_collection=3,scale_updates_with_actual_batch=False)
    for name,data in [('model',model),('task',task),('ppo',shared),('sac',sac)]:
        atomic_json(tmp_path/'config'/f'{name}.json',data)
    output=tmp_path/'run'
    requested=[]
    def request():
        (output/'STOP_REQUESTED').touch()
        training_control._stop_context.get()['next_check']=0.
        requested.append(True)
    if stage=='optimizer':
        original=torch.optim.Adam.step
        def step(*args,**kwargs):
            result=original(*args,**kwargs)
            request()  # Stop after a real partial optimizer update.
            return result
        monkeypatch.setattr(torch.optim.Adam,'step',step)
    else:
        name={'validation':'evaluate_deployment','planning':'plan_batch','recovery':'execute_batch'}[stage]
        original=getattr(deployment_rollout,name)
        def interrupt(*args,**kwargs):
            request()
            return original(*args,**kwargs)
        monkeypatch.setattr(deployment_rollout,name,interrupt)
        if stage=='validation':monkeypatch.setattr(run_sac,name,interrupt)
    if algorithm=='ppo':
        write_active=run_ppo.write_active_run
        monkeypatch.setattr(run_ppo,'write_active_run',lambda _root,path:write_active(tmp_path,path))
        run_ppo.train(device_name='cpu',requested_episodes=8,batch_size=4,
            artifact=output,resume_checkpoint=None,config_directory=tmp_path/'config')
    else:
        # Keep the test's ACTIVE_RUN pointer and all artifacts isolated.
        for name in ['run_sac.py','learning/simple_sac.py','learning/sac_deployment.py',
                     'learning/training_control.py','learning/deployment_rollout.py',
                     'learning/point_force_env.py','simulator/live_flight.py',
                     'simulator/cable/dder.py','simulator/point_mass.py','simulator/cuda_graph_physics.py']:
            destination=tmp_path/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(run_sac.ROOT/name,destination)
        monkeypatch.setattr(run_sac,'ROOT',tmp_path)
        run_sac.run(output,8,device_name='cpu',batch_override=4,config_directory=tmp_path/'config')
    assert requested
    status=read_json(output/'status.json')
    assert status['status']=='STOPPED' and status['episodes']==0
    assert (output/'checkpoints/terminal.pt').read_bytes()==(output/'checkpoints/latest.pt').read_bytes()
    assert training_control._stop_context.get() is None
