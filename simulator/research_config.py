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
    # A clean checkout starts from portable, unfitted configuration.
    folder.mkdir(parents=True,exist_ok=True)
    for name in ('model','task','ppo'):
        shutil.copy2(root/'config'/(name+'.json'),folder/(name+'.json'))
    atomic_json(root/'config/research_workspace.json',dict(schema='research_workspace_v1',
        config_directory='config/research_30hz',model_label='Unfitted simulator baseline'))
    return folder



def workspace_configs(root):
    root=Path(root);pointer=read_json(root/'config/research_workspace.json',{})
    folder=root/pointer.get('config_directory','config')
    return tuple(read_json(folder/(name+'.json')) for name in ('model','task','ppo'))


def validate_research_contract(model,task,config):
    """Reject settings this native execution path does not implement."""
    import math
    execution=model.get('fullstate_execution',{})
    termination=config.get('deployment',{}).get('termination')
    if termination not in (None,'execution_success_or_timeout'):
        raise ValueError('Unsupported execution termination rule.')
    if termination and execution.get('schema')!=EXECUTION_SCHEMA:
        raise ValueError('Execution-success termination requires the native 30 Hz model.')
    if execution.get('schema')!=EXECUTION_SCHEMA:return
    if termination and config['deployment'].get('require_predicted_success',True):
        raise ValueError('Execution-success termination must allow plans without a virtual-model hit.')
    if not execution.get('enabled') or not config.get('deployment',{}).get('enabled'):
        raise ValueError('The native 30 Hz model requires FullState deployment scoring.')
    if not math.isclose(float(task['control_dt_s']),1/30,rel_tol=0,abs_tol=1e-12) or execution.get('command_rate_hz')!=30:
        raise ValueError('This model requires native 30 Hz force actions and 30 Hz FullState packets.')
    duration=float(task['episode_duration_s'])*30
    if not math.isfinite(duration) or duration<1 or not math.isclose(duration,round(duration),abs_tol=1e-9):
        raise ValueError('The maximum plan duration must be a whole number of 30 Hz packets.')
    if model.get('motion_residual',{}).get('enabled') and model['cable'].get('external_drag_s_inv',0)!=0:
        raise ValueError('Learned cable damping requires zero separate fixed external drag.')
    if config['deployment'].get('planning_cable_state')!='hanging':raise ValueError('This workspace plans from a hanging cable.')
    for key in ['force_gain_fraction','force_lag_max_s','stiffness_fraction','damping_fraction']:
        if config['deployment'].get(key,0)!=0:raise ValueError(f'{key} is not modeled by this fitted FullState execution path.')


def snapshot_assets(model, directory):
    """A training run owns its model weights, independently of active pointers."""
    model=deepcopy(model);directory=Path(directory).resolve();assets=directory/'assets';assets.mkdir()
    root=Path(__file__).resolve().parents[1]
    def resolve(value):
        path=Path(value)
        return path if path.is_absolute() else root/path
    residual=model.get('motion_residual',{})
    if residual.get('enabled'):
        source=resolve(residual['checkpoint'])
        if sha256_file(source)!=residual['sha256']:raise ValueError('Cable residual changed')
        shutil.copy2(source,assets/'cable_residual.pt');residual['checkpoint']=str(assets/'cable_residual.pt')
    execution=model['fullstate_execution'];source=resolve(execution['checkpoint'])
    if sha256_file(source)!=execution['sha256']:raise ValueError('Drone model changed')
    payload=read_json(source);spec=payload.get('residual',{})
    if spec.get('enabled',bool(spec.get('checkpoint'))):
        weight=source.parent/spec['checkpoint']
        if sha256_file(weight)!=spec['sha256']:raise ValueError('Drone residual changed')
        shutil.copy2(weight,assets/'drone_residual.pt')
        spec['checkpoint']='drone_residual.pt'
    atomic_json(assets/'drone_model.json',payload)
    execution['checkpoint']=str(assets/'drone_model.json')
    execution['sha256']=sha256_file(assets/'drone_model.json')

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
