"""Research workflow operations independent of Qt."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys

from experimental_data.io import atomic_json, canonical_json_hash, resolve_raw_pair, sha256_file


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        if default is not None:
            return deepcopy(default)
        raise


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')


def import_take(root, source):
    source = Path(source).resolve()
    name = source.name
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
        raise ValueError('Use a folder name containing letters, digits, underscores or hyphens.')
    pair = resolve_raw_pair(source)
    destination = Path(root) / 'data/raw_takes' / name
    if destination.exists():
        raise ValueError(f'{name} already exists. Existing raw recordings are preserved.')
    destination.mkdir(parents=True)
    # Import only the validated logger/Motive pair, never arbitrary directory trees.
    paths = pair.values() if isinstance(pair, dict) else pair
    for path in paths:
        shutil.copy2(path, destination / Path(path).name)
    manifest_path = Path(root) / 'data/dataset_manifest.json'
    manifest = read_json(manifest_path)
    manifest['takes'][name] = dict(enabled=True, role='training', note='Imported preliminary recording.',
                                 segments=[], episode_breaks_s=[])
    atomic_json(manifest_path, manifest)
    return name


def apply_baseline(root, model, *, fit_directory=None, include_residual=False):
    root = Path(root)
    from simulator.point_mass import ForceControlledPointCable
    ForceControlledPointCable.from_mapping(model, root=root)
    version = stamp()
    destination = root / 'data/baselines' / version
    model = deepcopy(model)
    if fit_directory is not None:
        fit_directory = Path(fit_directory)
        fit = read_json(fit_directory / 'fit_result.json')
        source_model = read_json(fit_directory / 'model.json')
        if canonical_json_hash(source_model) != canonical_json_hash(model):
            raise ValueError('Model inputs changed since fitting. Fit the current inputs before applying this candidate.')
        model['cable'].update(fit['fitted_parameters'])
        # A newly fitted physical model invalidates an older neural correction.
        model.pop('motion_residual', None)
        model.pop('fullstate_execution', None)
        if include_residual:
            residual = fit.get('residual', {})
            if not residual.get('validation_improved_at_all_horizons', False):
                raise ValueError('The neural candidate has not improved validation at every evaluated horizon. Apply physics only.')
            source = fit_directory / residual['checkpoint']
            if sha256_file(source) != residual['sha256']:
                raise ValueError('Residual checkpoint changed since validation.')
            destination.mkdir(parents=True)
            shutil.copy2(source, destination / 'motion_residual.pt')
            model['motion_residual'] = dict(enabled=True,
                checkpoint=str((destination / 'motion_residual.pt').relative_to(root)).replace('\\', '/'),
                sha256=residual['sha256'], specification=residual['specification'])
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fit_directory / 'fit_result.json', destination / 'fit_result.json')
        model['cable']['parameter_source'] = str((destination / 'fit_result.json').relative_to(root)).replace('\\', '/')
    else:
        # Manual changes cannot silently retain a correction trained for another model.
        model.pop('motion_residual', None)
        model.pop('fullstate_execution', None)
        model['cable']['previous_parameter_source'] = model['cable'].get('parameter_source')
        model['cable']['parameter_source'] = str((destination / 'manifest.json').relative_to(root)).replace('\\', '/')
    atomic_json(destination / 'model.json', model)
    atomic_json(destination / 'manifest.json', dict(version=version,
        model_sha256=canonical_json_hash(model), fit_directory=str(fit_directory) if fit_directory else None,
        provenance='fitted' if fit_directory else 'manually configured'))
    atomic_json(root / 'config/model.json', model)
    atomic_json(root / 'config/baseline.json', dict(version=version, model_sha256=canonical_json_hash(model)))
    return version


def prepare_training(root, algorithm, *, seed, episodes, batch, device, overrides=None, resume=None, run_name=None,
                     keep_optimizer_state=None, early_stopping=None, live_scene=None, keep_stopping_history=False,
                     reward_overrides=None, deployment_overrides=None, task_overrides=None, training_backend=None):
    root = Path(root)
    algorithm = algorithm.lower()
    if algorithm not in ('ppo', 'sac'):
        raise ValueError('Choose PPO or SAC.')
    if resume:
        source=Path(resume).parent.parent
    else:
        pointer=read_json(root/'config/research_workspace.json',{}) if algorithm=='ppo' else {}
        source=root/pointer.get('config_directory','config')
    configs = {name: read_json(source / f'{name}.json') for name in ('model', 'task', 'ppo')}
    # Explicit setup adaptation must use a new task snapshot, never edit the
    # selected checkpoint's source task or inherit its convergence history.
    import math
    task_changes={}
    for key,value in (task_overrides or {}).items():
        if key not in ('initial_root_position_m','target_position_m') or not isinstance(value,(list,tuple)) or len(value)!=3 or not all(math.isfinite(float(v)) for v in value):
            raise ValueError('Task overrides require finite XYZ initial_root_position_m or target_position_m.')
        value=[float(v) for v in value]
        if value!=configs['task'].get(key):task_changes[key]=dict(before=deepcopy(configs['task'].get(key)),after=value)
    if task_changes and keep_stopping_history:
        raise ValueError('A changed task setup requires new stopping history.')
    for key,change in task_changes.items():configs['task'][key]=change['after']
    # Validated execution acceleration may change on continuation; physics and task stay saved.
    runtime = read_json(root/'config/ppo.json')
    configs['ppo']['cuda_graph_physics'] = bool(runtime.get('cuda_graph_physics', False))
    if algorithm == 'sac':
        configs['sac'] = read_json(source / 'sac.json')
    config = configs[algorithm]
    if reward_overrides and algorithm != 'ppo':
        raise ValueError('Reward overrides are supported for PPO only.')
    deployment_changes = {key: dict(before=deepcopy(config.get('deployment',{}).get(key)),after=deepcopy(value))
        for key,value in (deployment_overrides or {}).items() if config.get('deployment',{}).get(key)!=value}
    if deployment_changes and keep_stopping_history:
        raise ValueError('Changed deployment semantics require new stopping history.')
    if deployment_overrides:config.setdefault('deployment',{}).update(deepcopy(deployment_overrides))
    reward_changes = {key: dict(before=deepcopy(config['reward'].get(key)), after=deepcopy(value))
                      for key, value in (reward_overrides or {}).items() if config['reward'].get(key) != value}
    if reward_changes and keep_stopping_history:
        raise ValueError('A changed reward requires new stopping history; old reward totals are not comparable.')
    if reward_overrides:
        config['reward'].update(deepcopy(reward_overrides))
    if resume and (source/'early_stopping_override.json').is_file():
        # Live stopping amendments carry forward without rewriting the parent snapshot.
        amendment = read_json(source/'early_stopping_override.json')
        config.setdefault('early_stopping', {}).update(deepcopy(amendment['settings']))
    if configs['model'].get('fullstate_execution',{}).get('enabled'):
        if algorithm!='ppo':raise ValueError('This full-state execution baseline is implemented for PPO only.')
        execution=configs['model']['fullstate_execution']
        if execution.get('schema')=='tracked_pose_execution_v1':
            from simulator.research_pose import ResearchPoseModel
            ResearchPoseModel(root/execution['checkpoint'],execution['sha256'],device='cpu')
        else:
            from simulator.fullstate_execution import FullStateAttachmentModel
            FullStateAttachmentModel(root/execution['checkpoint'],execution['sha256'])
        config['deployment'].update(enabled=True,recovery_failure_penalty=0.,force_gain_fraction=0.,force_lag_max_s=0.)
    if not resume:
        config['seed'] = int(seed)
        config[algorithm].update(overrides or {})
    elif keep_optimizer_state is not None:
        config['deployment']['reset_optimizer_on_resume'] = not bool(keep_optimizer_state)
    if early_stopping is not None:
        config['early_stopping'] = deepcopy(early_stopping)
    inherited_history = config.pop('reward_plateau_resume', None)
    if keep_stopping_history:
        if not resume or algorithm != 'ppo' or not config.get('early_stopping', {}).get('enabled'):
            raise ValueError('Preserving stopping history requires a PPO continuation with plateau stopping enabled.')
        import torch
        checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
        cutoff = int(checkpoint['episodes'])
        records = list((inherited_history or {}).get('records', []))
        history_path = source / 'validation_history.jsonl'
        if history_path.is_file():
            records.extend(json.loads(line) for line in history_path.read_text(encoding='utf-8').splitlines() if line.strip())
        convergence = read_json(source/'reward_convergence.json', {})
        if not records and convergence.get('window_count', 0):
            raise ValueError('The parent has stopping statistics but its validation history is unavailable.')
        config['reward_plateau_resume'] = dict(schema='reward_plateau_resume_v1',
            original_start_episodes=int((inherited_history or {}).get('original_start_episodes',
                convergence.get('start_episodes', cutoff))), checkpoint_episodes=cutoff,
            source_run=str(source.resolve()), source_checkpoint_sha256=sha256_file(Path(resume)),
            records=[row for row in records if int(row['training_episodes']) <= cutoff])
    native = algorithm == 'ppo' and configs['model'].get('fullstate_execution',{}).get('schema')=='tracked_pose_execution_v1'
    backend=training_backend or {'type':'project'}
    if backend.get('type') not in ('project','isaaclab_model'):raise ValueError('Unknown PPO training backend.')
    if backend['type']=='isaaclab_model':
        if not native or device not in ('cuda','cuda:0'):raise ValueError('Isaac Lab model training requires native 30 Hz PPO on CUDA device 0.')
        if not Path(backend.get('python','')).is_file():raise ValueError('Select the installed Isaac Lab Python executable.')
        live_scene=False
    config['training_backend']=backend.copy()
    if live_scene is True and not native:
        raise ValueError('Live Isaac visualization requires a native 30 Hz policy. Legacy training retains its original workflow.')
    config['live_scene'] = dict(enabled=native if live_scene is None else bool(live_scene),
        source='actual_training_collection', retention='last_two_completed_batches')
    config['training'].update(requested_episodes=int(episodes), collection_batch=int(batch), device=device)
    # The UI publishes one evaluation of the current policy after every batch.
    config['validation'].update(enabled=True, every_episodes=int(batch))
    if algorithm == 'ppo' and config.get('update_guard', {}).get('enabled'):
        config['update_guard']['validation_episodes'] = config['validation']['episodes']
    name = f'{stamp()}-seed{config["seed"]}'
    directory = root / 'runs' / algorithm / name
    launch = directory / 'launch_config'
    launch.mkdir(parents=True)
    if configs['model'].get('fullstate_execution',{}).get('schema')=='tracked_pose_execution_v1':
        from simulator.research_config import snapshot_assets
        configs['model']=snapshot_assets(configs['model'],directory)
    for key, value in configs.items():
        atomic_json(launch / f'{key}.json', value)
    atomic_json(directory / 'run.json', dict(schema='point_force_training_run_v1', algorithm=algorithm.upper(),
        display_name=(run_name or '').strip() or name, seed=config['seed'], resumed_from=str(resume) if resume else None,
        task_amendment=(dict(changes=task_changes,stopping_history_reset=True,
            parent_checkpoint_sha256=sha256_file(Path(resume)) if resume else None) if task_changes else None),
        reward_amendment=(dict(changes=reward_changes, stopping_history_reset=True,
            parent_checkpoint_sha256=sha256_file(Path(resume)) if resume else None) if reward_changes else None),
        deployment_amendment=deployment_changes or None,
        baseline=({'source_job':configs['model']['fullstate_execution']['source_job'], 'schema':'tracked_pose_execution_v1'}
                  if configs['model'].get('fullstate_execution',{}).get('schema')=='tracked_pose_execution_v1'
                  else read_json(root / 'config/baseline.json', {})) if not resume else {'inherited_from': str(source)},
        config_sha256={key: canonical_json_hash(value) for key, value in configs.items()}))
    command = [sys.executable, '-u', f'run_{algorithm}.py', '--train',
               '--config-directory', str(launch), '--device', device, '--episodes', str(episodes)]
    command += ['--artifact-directory' if algorithm == 'ppo' else '--artifact', str(directory),
                '--batch-size' if algorithm == 'ppo' else '--batch', str(batch)]
    if resume:
        command += ['--resume-checkpoint', str(resume)]
    if algorithm=='ppo' and configs['model'].get('fullstate_execution',{}).get('schema')=='tracked_pose_execution_v1':
        from simulator.research_config import freeze_training_source
        command=freeze_training_source(root,directory,command)
        if backend['type']=='isaaclab_model':
            command[0]=str(Path(backend['python']).resolve())
            command[2]=str(directory/'source_snapshot/tools/train_ppo_isaaclab.py')
            if backend.get('headless',False):command.append('--headless')
    atomic_json(directory/'launch.json',dict(command=command,working_directory=str(root)))
    return directory, command


def prepare_data_job(root, kind, model):
    root = Path(root)
    directory = root / 'data/workflow_jobs' / f'{stamp()}-{kind}'
    directory.mkdir(parents=True)
    atomic_json(directory / 'model.json', model)
    manifest = read_json(root / 'data/dataset_manifest.json')
    atomic_json(directory / 'dataset_manifest.json', manifest)
    fit = read_json(root / 'config/cable_fit.json')
    if kind == 'fit':
        fit['method'] = 'constrained_geometry_drag'
    fit['comparison_baseline'] = dict(boundary='current_one_position_node_pivot',
        **{key: model['cable'][key] for key in ('EI_n_m2', 'Cb_n_m2_s')})
    atomic_json(directory / 'fit_config.json', fit)
    return directory, [sys.executable, '-u', '-m', 'simulator.workflow',
                        kind, '--root', str(root), '--job', str(directory)]


def run_data_job(root, directory, kind):
    from experimental_data.processing import process_take
    from experimental_data.force_dataset import build_force_dataset
    from experimental_data.cable_fit import fit_pivot_cable
    root, directory = Path(root).resolve(), Path(directory).resolve()
    manifest = read_json(directory / 'dataset_manifest.json')
    atomic_json(directory / 'status.json', {'status': 'RUNNING'})
    try:
        if kind == 'process':
            for name, row in manifest['takes'].items():
                if row.get('enabled', True) and row['role'] != 'untouched_test':
                    print(f'Processing {name}', flush=True)
                    result = process_take(root / 'data/raw_takes' / name,
                                          processed_root=root / 'data/processed_takes', force=True)
                    if result.get('quality_status') == 'PROCESSING_FAILED':
                        raise ValueError(f'Processing failed: {name}')
        print('Reconstructing cable states and estimated forces…', flush=True)
        # Fitting always consumes its own immutable derived-data snapshot.
        output = directory / 'force_takes'
        build_force_dataset(model_path=directory / 'model.json',
            manifest_path=directory / 'dataset_manifest.json', processed_root=root / 'data/processed_takes',
            output_root=output)
        if kind == 'fit':
            if read_json(directory / 'fit_config.json').get('method') == 'constrained_geometry_drag':
                from experimental_data.recommended_fit import fit_recommended
                fit_recommended(directory, root)
            elif read_json(directory / 'fit_config.json').get('method') == 'differentiable_physics_residual':
                from experimental_data.differentiable_fit import fit_physics_and_residual
                fit_physics_and_residual(directory)
            else:
                fit_pivot_cable(model_path=directory / 'model.json', fit_config_path=directory / 'fit_config.json',
                                data_root=output, output_root=directory, progress=lambda text: print(text, flush=True))
        else:
            shutil.copytree(output, root / 'data/force_takes', dirs_exist_ok=True)
        atomic_json(directory / 'status.json', {'status': 'COMPLETED'})
    except BaseException as error:
        atomic_json(directory / 'status.json', {'status': 'FAILED', 'error': str(error)})
        raise


def apply_recommended_baseline(root, directory, expected_hash):
    """Apply the entire reviewed model, including geometry, drag and solver settings."""
    from simulator.point_mass import ForceControlledPointCable
    root,directory=Path(root),Path(directory)
    model=read_json(directory/'recommended_model.json')
    review=read_json(directory/'candidate_review.json')
    digest=canonical_json_hash(model)
    if digest!=expected_hash or review.get('model_sha256',digest)!=digest:
        raise ValueError('Candidate changed since it was selected. Select it again before applying.')
    ForceControlledPointCable.from_mapping(model,root=root)
    if model.get('motion_residual',{}).get('enabled'):
        raise ValueError('This workflow applies the physical candidate only.')
    version=stamp();destination=root/'data/baselines'/version
    model=deepcopy(model)
    model['cable']['previous_parameter_source']=model['cable'].get('parameter_source')
    model['cable']['parameter_source']=str((destination/'manifest.json').relative_to(root)).replace('\\','/')
    for name in ('candidate_review.json','evaluation.json','geometry_fit.json'):
        if (directory/name).exists():
            destination.mkdir(parents=True,exist_ok=True);shutil.copy2(directory/name,destination/name)
    atomic_json(destination/'model.json',model)
    atomic_json(destination/'manifest.json',dict(version=version,model_sha256=canonical_json_hash(model),
        reviewed_candidate_sha256=digest,fit_directory=str(directory),provenance='constrained_geometry_drag'))
    atomic_json(root/'config/model.json',model)
    atomic_json(root/'config/baseline.json',dict(version=version,model_sha256=canonical_json_hash(model)))
    return version


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['process', 'fit'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    run_data_job(args.root, args.job, args.kind)
