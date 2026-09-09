"""Immutable CEM jobs, complete reference validation and portable CSV bundles."""
from pathlib import Path
import csv
import json
import shutil
import sys
import time
import zipfile
import numpy as np
from experimental_data.io import atomic_json, sha256_file
from simulator.workflow import read_json
from simulator.research_config import snapshot_assets
from .spline import fit_seed, sample
from .cem import optimize


DEFAULTS = dict(population=64, iterations=12, control_points=12, elite_fraction=.125,
                position_std_m=.025, duration_std_s=.08, minimum_duration_s=.8,
                maximum_duration_s=2., maximum_height_m=2.8, minimum_height_m=.08,
                random_seed=655, device='cuda', display_name='CEM spline')


def validate_settings(settings):
    for key in ('population', 'iterations', 'control_points', 'random_seed'):
        if int(settings[key]) != settings[key]:
            raise ValueError(f'{key} must be an integer.')
    if not 4 <= settings['population'] <= 2048 or not 1 <= settings['iterations'] <= 1000 or not 6 <= settings['control_points'] <= 30:
        raise ValueError('Population 4–2048, iterations 1–1000 and control points 6–30 are supported.')
    if not 0 < settings['elite_fraction'] <= .5 or settings['position_std_m'] <= 0 or settings['duration_std_s'] <= 0:
        raise ValueError('Exploration and elite fraction must be positive.')
    lo, hi = settings['minimum_duration_s'], settings['maximum_duration_s']
    if not .2 <= lo <= hi <= 5 or np.ceil(lo*30-1e-9) > np.floor(hi*30+1e-9):
        raise ValueError('Duration range must contain a 30 Hz boundary and lie in 0.2–5 s.')
    if not 0 <= settings['minimum_height_m'] < settings['maximum_height_m']:
        raise ValueError('Invalid height interval.')
    if any(not np.isfinite(v) for v in settings.values() if isinstance(v, (float, int))):
        raise ValueError('Settings must be finite.')


def prepare_job(root, seed_directory, output, settings, origin=None, target=None):
    root, seed_directory, output = map(lambda p: Path(p).resolve(), (root, seed_directory, output))
    settings = dict(DEFAULTS, **settings); validate_settings(settings)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a new empty CEM run folder.')
    meta = read_json(seed_directory/'rehearsal.json')
    if meta.get('schema') not in ('research_fullstate_30hz_v1', 'cem_fullstate_30hz_v1'):
        raise ValueError('Choose a completed native 30 Hz rehearsal as the seed.')
    with np.load(seed_directory/'rehearsal.npz') as data:
        seed = data['commands'][:int(round(meta['whip_end_s']*30))+1].copy()
    model, task, config = [read_json(seed_directory/f'{name}.json') for name in ('model','task','ppo')]
    for name in ('motion_residual','fullstate_execution'):
        path = Path(model[name]['checkpoint'])
        if not path.is_absolute():
            model[name]['checkpoint'] = str((seed_directory/path).resolve())
    from simulator.research_config import validate_research_contract
    validate_research_contract(model, task, config)
    origin = np.asarray(meta['initial_tracking_origin_m'] if origin is None else origin, float)
    target = np.asarray(meta['target_position_m'] if target is None else target, float)
    if origin.shape != (3,) or target.shape != (3,) or not np.isfinite([origin, target]).all():
        raise ValueError('Finite XYZ positions are required.')
    task['initial_root_position_m'] = (origin+np.asarray(model['recorded_data']['optitrack_to_attachment_offset_body_m'])).tolist()
    task['initial_root_velocity_m_s'] = [0.,0.,0.]; task['target_position_m'] = target.tolist()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/'model.json', snapshot_assets(model, output))
    for name, value in [('task',task),('ppo',config),('cem',settings)]:
        atomic_json(output/f'{name}.json', value)
    np.savez_compressed(output/'seed.npz', commands=seed, duration_s=meta['whip_end_s'])
    atomic_json(output/'seed_provenance.json', dict(source=str(seed_directory), metadata=meta,
        rehearsal_sha256=sha256_file(seed_directory/'rehearsal.npz'), usage='Spline initialization only; no PPO inference or post-export translation'))
    snapshot = output/'source_snapshot'
    for folder in ('planning','simulator','learning','deployment','experimental_data','tools'):
        for path in (root/folder).rglob('*'):
            if path.is_file() and path.suffix in ('.py','.cu','.cuh','.h') and '__pycache__' not in path.parts:
                dest = snapshot/path.relative_to(root); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path,dest)
    for name in ('run_ppo.py','requirements.txt'):
        shutil.copy2(root/name,snapshot/name)
    atomic_json(output/'source_manifest.json', {p.relative_to(snapshot).as_posix():sha256_file(p) for p in snapshot.rglob('*') if p.is_file()})
    atomic_json(output/'run.json', dict(status='PREPARED', display_name=settings['display_name']))
    return [sys.executable,'-u',str(snapshot/'tools/plan_cem.py'),'--job',str(output)]


def run_job(output):
    import torch
    from .cem_execution import SplineEvaluator, Cancelled
    from deployment.research_rehearsal import complete_packets, FIELDS
    from simulator.research_reference import reference_packet_validity
    output = Path(output).resolve()
    settings = read_json(output/'cem.json'); validate_settings(settings)
    model, task, config = [read_json(output/f'{n}.json') for n in ('model','task','ppo')]
    # Assets remain portable if a run directory is moved.
    for name, file in [('motion_residual','cable_residual.pt'),('fullstate_execution','drone_model.json')]:
        model[name]['checkpoint'] = str(output/'assets'/file)
    cancelled = lambda: (output/'STOP_REQUESTED').exists()
    start = time.perf_counter()
    atomic_json(output/'run.json', dict(status='RUNNING', display_name=settings['display_name']))
    try:
        if settings['device'] == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable. Select CPU explicitly to run on CPU.')
        evaluator = SplineEvaluator(model, task, config, settings, settings['device'], cancelled)
        with np.load(output/'seed.npz') as data:
            duration = float(data['duration_s']); coefficients = fit_seed(data['commands'], duration, settings['control_points'])
        coefficients[:3] = evaluator.origin  # Actual boundary, not translation of the reference.
        duration = np.clip(duration, settings['minimum_duration_s'], settings['maximum_duration_s'])
        mean = np.r_[coefficients[3:].ravel(), duration]
        std = np.r_[np.full(len(mean)-1,settings['position_std_m']),settings['duration_std_s']]
        history = []
        def progress(row, best, bank):
            history.append(row)
            atomic_json(output/'history.json', history)
            np.savez_compressed(output/'candidates.npz', scores=[s for s,_ in bank], vectors=[v for _,v in bank])
            atomic_json(output/'progress.json', dict(label=f'CEM iteration {row["iteration"]}/{settings["iterations"]} · best {row["best_score"]:.2f} · population hits {row["success_fraction"]:.0%}',step=row['iteration'],total=settings['iterations'],**row))
            print(json.dumps(row), flush=True)
        _, _, bank = optimize(mean,std,evaluator,population=settings['population'],iterations=settings['iterations'],
            elite_fraction=settings['elite_fraction'],seed=settings['random_seed'],std_floor=np.r_[np.full(len(mean)-1,.001),1/60],
            progress=progress,cancelled=cancelled)
        evaluator.check_cancelled()
        rejected, seen = [], set()
        selected = None
        # Final verification includes the entire curved recovery and hold. It is
        # a feasibility filter, not a recovery term in the whip objective.
        for score, vector in bank:
            key = vector.tobytes()
            if key in seen:
                continue
            seen.add(key)
            c, duration = evaluator.splines([vector])[0]
            whip = sample(c,duration,np.arange(int(round(duration*30))+1)/30)
            hit = evaluator.rollout([whip],[duration])
            if hit['failed'][0]:
                rejected.append('Single-candidate prediction failed');continue
            if hit['success'][0]:
                end = min(len(whip),int(np.ceil(hit['hit_time'][0]*30-1e-9))+1)
                whip = whip[:max(2,end)]
            try:
                times, packets, phases, recovery = complete_packets(whip,evaluator.origin)
            except ValueError as error:
                rejected.append(str(error));continue
            feasible,_ = reference_packet_validity(torch.as_tensor(packets)[None],model['fullstate_execution']['feasibility'])
            if not bool(feasible.all()) or packets[:,2].max() > settings['maximum_height_m'] or packets[:,2].min() < settings['minimum_height_m']:
                rejected.append('Complete command exceeds envelope/height');continue
            atomic_json(output/'progress.json',dict(label='Checking complete drone/cable recovery and hold',step=settings['iterations'],total=settings['iterations']))
            prediction = evaluator.rollout([packets],[times[-1]],record=True)
            if prediction['failed'][0]:
                rejected.append('Complete predicted drone/cable exceeds limits');continue
            selected = score,vector,c,duration,whip,times,packets,phases,recovery,hit,prediction
            break
        atomic_json(output/'recovery_rejections.json',rejected)
        if selected is None:
            raise ValueError('No retained candidate passed complete recovery prediction and height limits. Optimization saved; no CSV exported.')
        score,vector,c,duration,whip,times,packets,phases,recovery,hit,prediction = selected
        np.savez_compressed(output/'spline.npz',coefficients=c,duration_s=duration,vector=vector)
        np.savez_compressed(output/'rehearsal.npz',command_time_s=times,commands=packets,command_phase=phases,
            prediction_time_s=np.arange(len(prediction['cable']))/150,cable_positions_m=prediction['cable'],
            origin_positions_m=prediction['origin'],origin_rotations=prediction['rotations'],target_position_m=task['target_position_m'])
        with (output/'fullstate_30hz.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream);writer.writerow(FIELDS);writer.writerows(np.c_[times,packets])
        meta=dict(schema='cem_fullstate_30hz_v1',planner='CEM quintic position spline',display_name=settings['display_name'],
            initial_tracking_origin_m=evaluator.origin.tolist(),target_position_m=task['target_position_m'],
            frame='World XYZ, metres; unshifted OptiTrack tracked origin',command_rate_hz=30,
            initial_state='Settled level hover; zero velocity and acceleration; hanging cable',preflight_hold_s=10,
            whip_end_s=(len(whip)-1)/30,total_duration_s=float(times[-1]),optimized_duration_s=duration,
            predicted_valid_hit=bool(hit['success'][0]),predicted_hit_time_s=float(hit['hit_time'][0]) if hit['success'][0] else None,
            minimum_tip_distance_m=float(hit['distance'][0]),objective_score=score,
            recovery=recovery,recovery_prediction_complete=True,recovery_empirically_validated=False,
            prediction_valid_through_s=float(times[-1]),reference_feasible=True,
            maximum_command_height_m=float(packets[:,2].max()),maximum_predicted_drone_height_m=float(prediction['origin'][:,2].max()),
            maximum_predicted_cable_height_m=float(prediction['cable'][:,:,2].max()),
            source_job=model['fullstate_execution']['source_job'],elapsed_s=time.perf_counter()-start,
            execution='Take off; hold 10 s at initial tracked origin; execute complete CSV once at its timestamps; land. Offline artifact only.')
        atomic_json(output/'rehearsal.json',meta)
        atomic_json(output/'run.json',dict(status='COMPLETED',display_name=settings['display_name'],elapsed_s=meta['elapsed_s']))
        return meta
    except Cancelled as error:
        atomic_json(output/'run.json',dict(status='STOPPED',display_name=settings['display_name'],message=str(error)))
        return None
    except Exception as error:
        atomic_json(output/'run.json',dict(status='FAILED',display_name=settings['display_name'],message=str(error)))
        raise


def export_package(directory,destination):
    directory,destination=Path(directory),Path(destination)
    if destination.resolve().is_relative_to(directory.resolve()):
        raise ValueError('Save the ZIP outside the CEM run folder.')
    if read_json(directory/'rehearsal.json',{}).get('schema')!='cem_fullstate_30hz_v1':
        raise ValueError('Select a completed CEM result.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in directory.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.name not in ('console.log','STOP_REQUESTED','model.json'):
                archive.write(path,path.relative_to(directory).as_posix())
        model=read_json(directory/'model.json')
        model['motion_residual']['checkpoint']='assets/cable_residual.pt';model['fullstate_execution']['checkpoint']='assets/drone_model.json'
        archive.writestr('model.json',json.dumps(model,indent=2))
        archive.writestr('README.txt','CEM spline / native 30 Hz FullState offline result\n'
            'fullstate_30hz.csv contains the exact commanded tracked-origin P/V/A, yaw and yaw rate.\n'
            'rehearsal.json records initial hover, target, predicted hit and complete recovery limits.\n'
            'No force policy inference is needed. No ROS flight sender is included.\n'
            'Assets contain the fitted drone and cable components and both residuals; source_snapshot preserves the planner.\n'
            'The result is a nominal simulation prediction. Recovery has not been empirically validated.\n')
    return destination
