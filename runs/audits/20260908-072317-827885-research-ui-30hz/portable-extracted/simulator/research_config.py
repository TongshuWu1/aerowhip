"""Explicit new-workspace configuration; archived 20 Hz runs stay immutable."""
from copy import deepcopy
from pathlib import Path
import shutil
from experimental_data.io import atomic_json, sha256_file
from .workflow import read_json

BOOTSTRAP_JOB='20260908-064040-669809'
EXECUTION_SCHEMA='tracked_pose_execution_v1'


def prepare_workspace(root):
    root=Path(root).resolve();folder=root/'config/research_30hz'
    if (folder/'model.json').exists():return folder
    bundle=root/'data/bootstrap_model_runs'/BOOTSTRAP_JOB/'bundle'
    from experimental_data.bootstrap_combined import load_bundle
    cable,drone=load_bundle(bundle)
    model=deepcopy(read_json(root/'config/model.json'))
    model['cable']=deepcopy(cable['cable']);model['motion_residual']=deepcopy(cable['motion_residual'])
    model['recorded_data']=deepcopy(cable['recorded_data'])
    model['mass_measurement']=deepcopy(cable['mass_measurement'])
    model['point_mass']['mass_kg']=cable['mass_measurement']['drone_mass_kg']
    # 150 Hz resolves both 30 Hz packets and the 20 ms follow-through exactly.
    # Eight internal steps preserve the fitted 1200 Hz cable integration rate.
    model['simulation']['dt_s']=1/150;model['cable']['substeps']=8
    model['fullstate_execution']=dict(enabled=True,schema=EXECUTION_SCHEMA,
        checkpoint=str(bundle/'drone_model.json'),sha256=sha256_file(bundle/'drone_model.json'),
        source_job=BOOTSTRAP_JOB,command_rate_hz=30.,recovery_objective='excluded',
        initialization='settled_hover_pose_and_hanging_planning_cable',
        reference='integrated_piecewise_linear_virtual_velocity_v1',
        feasibility=dict(minimum_specific_vertical_m_s2=2.,maximum_tilt_deg=60.,
            maximum_specific_force_m_s2=3.2/.175,maximum_speed_m_s=5.,
            source='Provisional simulation envelope, not measured hardware limits'))
    task=deepcopy(read_json(root/'config/task.json'));task['control_dt_s']=1/30
    ppo=deepcopy(read_json(root/'config/ppo.json'));ppo['seed']=655
    ppo['status']='research_30hz_both_residuals'
    # This is a fresh actor, not a retimed checkpoint. A separately searched
    # 30 Hz force prior can be saved here before starting the new run.
    ppo['bootstrap']={'enabled':False}
    ppo['ppo']['initial_log_std']=[-1.5,-2.3,-1.5]
    ppo['deployment'].update(enabled=True,initial_position_radius_m=.05,target_position_radius_m=.05,
        planning_cable_state='hanging',force_gain_fraction=0.,force_lag_max_s=0.,
        stiffness_fraction=0.,damping_fraction=0.,recovery_failure_penalty=0.,
        require_predicted_success=False,evaluate_final_holdout=False)
    ppo['validation'].update(episodes=256,every_episodes=1024)
    ppo['training'].update(collection_batch=1024,device='cuda',requested_episodes=500000)
    folder.mkdir(parents=True)
    for name,value in [('model',model),('task',task),('ppo',ppo)]:atomic_json(folder/(name+'.json'),value)
    atomic_json(root/'config/research_workspace.json',dict(schema='research_workspace_v1',
        config_directory='config/research_30hz',bundle=str(bundle.relative_to(root)),
        model_label='Bootstrap M0 · both residuals',selected_by_user=True))
    return folder


def workspace_configs(root):
    root=Path(root);pointer=read_json(root/'config/research_workspace.json',{})
    folder=root/pointer.get('config_directory','config')
    return tuple(read_json(folder/(name+'.json')) for name in ('model','task','ppo'))


def snapshot_assets(model, directory):
    """A training run owns its model weights, independently of active pointers."""
    model=deepcopy(model);directory=Path(directory).resolve();assets=directory/'assets';assets.mkdir()
    residual=model['motion_residual'];source=Path(residual['checkpoint'])
    if sha256_file(source)!=residual['sha256']:raise ValueError('Cable residual changed')
    shutil.copy2(source,assets/'cable_residual.pt');residual['checkpoint']=str(assets/'cable_residual.pt')
    execution=model['fullstate_execution'];source=Path(execution['checkpoint'])
    if sha256_file(source)!=execution['sha256']:raise ValueError('Drone model changed')
    payload=read_json(source);weight=source.parent/payload['residual']['checkpoint']
    if sha256_file(weight)!=payload['residual']['sha256']:raise ValueError('Drone residual changed')
    shutil.copy2(source,assets/'drone_model.json');shutil.copy2(weight,assets/'drone_residual.pt')
    execution['checkpoint']=str(assets/'drone_model.json')
    return model


def freeze_training_source(root,directory,command):
    """Run a new native-30-Hz job from its own source, even while the UI evolves."""
    root=Path(root);directory=Path(directory);snapshot=directory/'source_snapshot';snapshot.mkdir()
    if not (root/'run_ppo.py').is_file():root=Path(__file__).resolve().parents[1]
    for folder in ['learning','simulator','experimental_data','tools','deployment']:
        for path in (root/folder).rglob('*'):
            if path.is_file() and path.suffix in ('.py','.cu','.cuh','.h') and '__pycache__' not in path.parts:
                target=snapshot/path.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    shutil.copy2(root/'run_ppo.py',snapshot/'run_ppo.py');shutil.copytree(directory/'launch_config',snapshot/'config')
    atomic_json(directory/'source_snapshot_manifest.json',{'files':{p.relative_to(snapshot).as_posix():sha256_file(p) for p in snapshot.rglob('*') if p.is_file()}})
    command=list(command);command[2]=str(snapshot/'run_ppo.py');return command
