"""Explicit new-workspace configuration; archived 20 Hz runs stay immutable."""
from copy import deepcopy
from pathlib import Path
import shutil
from experimental_data.io import sha256_file
from .workflow import read_json

EXECUTION_SCHEMA='tracked_pose_execution_v1'


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
    if not model.get('motion_residual',{}).get('enabled') or model['cable'].get('external_drag_s_inv',0)!=0:
        raise ValueError('This selected model uses the cable NN with zero fixed external drag.')
    if config['deployment'].get('planning_cable_state')!='hanging':raise ValueError('This workspace plans from a hanging cable.')
    for key in ['force_gain_fraction','force_lag_max_s','stiffness_fraction','damping_fraction']:
        if config['deployment'].get(key,0)!=0:raise ValueError(f'{key} is not modeled by this fitted FullState execution path.')


def snapshot_assets(model, directory, *, source_root=None):
    """A saved job owns its model weights, independently of active pointers."""
    model=deepcopy(model);directory=Path(directory).resolve();assets=directory/'assets';assets.mkdir()
    base=Path(source_root).resolve() if source_root is not None else Path.cwd()
    def source_path(value):
        path=Path(value)
        return path if path.is_absolute() else base/path
    residual=model['motion_residual']
    if residual.get('enabled') is not False:
        source=source_path(residual['checkpoint'])
        if sha256_file(source)!=residual['sha256']:raise ValueError('Cable residual changed')
        shutil.copy2(source,assets/'cable_residual.pt');residual['checkpoint']=str(assets/'cable_residual.pt')
    execution=model['fullstate_execution'];source=source_path(execution['checkpoint'])
    if sha256_file(source)!=execution['sha256']:raise ValueError('Drone model changed')
    payload=read_json(source);weight=source.parent/payload['residual']['checkpoint']
    if sha256_file(weight)!=payload['residual']['sha256']:raise ValueError('Drone residual changed')
    shutil.copy2(source,assets/'drone_model.json');shutil.copy2(weight,assets/'drone_residual.pt')
    execution['checkpoint']=str(assets/'drone_model.json')
    return model
