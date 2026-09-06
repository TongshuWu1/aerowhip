"""Generate a frozen plan from a verified policy package and one initial state.

This module neither connects to ROS nor authorizes a flight.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from simulator.cable import DderState
from simulator.live_flight import LiveFlight
from simulator.point_mass import ForceControlledPointCable


def verify_package(package):
    package = Path(package).resolve()
    manifest = json.loads((package/'policy_manifest.json').read_text(encoding='utf-8'))
    required = {'model.json', 'task.json', 'ppo.json', 'checkpoints/policy.pt'}
    if manifest.get('schema') != 'selected_ppo_package_v1' or not required.issubset(manifest.get('files', {})):
        raise ValueError('Incomplete or unsupported policy manifest')
    for relative, digest in manifest['files'].items():
        path = (package/relative).resolve()
        if not path.is_relative_to(package) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Policy package file failed verification: {relative}')
    return manifest


def controller_acceleration(force_world_n, controller_mass_kg, gravity_m_s2):
    """Mellinger acceleration setpoint, assuming the verified common Z-up frame."""
    force = np.asarray(force_world_n, dtype=np.float64)
    if force.shape != (3,) or not np.isfinite(force).all():
        raise ValueError('Expected a finite XYZ world force in newtons')
    if not np.isfinite(controller_mass_kg) or controller_mass_kg<=0:
        raise ValueError('Controller mass must be measured/configured and positive')
    if not np.isfinite(gravity_m_s2) or gravity_m_s2<=0:
        raise ValueError('Supply the verified firmware gravity magnitude')
    return force/controller_mass_kg - np.array([0.,0.,gravity_m_s2])


def force_at_elapsed(forces_world_n, physics_dt_s, elapsed_s):
    """Return a held command, or None at the frozen cutoff. Never wraps/repeats."""
    forces = np.asarray(forces_world_n)
    if forces.ndim!=2 or forces.shape[1]!=3 or len(forces)==0 or not np.isfinite(forces).all():
        raise ValueError('Expected a nonempty finite Nx3 sequence')
    if not np.isfinite(physics_dt_s) or physics_dt_s<=0 or not np.isfinite(elapsed_s) or elapsed_s<0:
        raise ValueError('Invalid time')
    cutoff = len(forces)*physics_dt_s
    if elapsed_s>=cutoff:
        return None
    index = min(int(np.floor(elapsed_s/physics_dt_s+1e-9)),len(forces)-1)
    return forces[index].copy()


def validate_initial_state(model_config, q, v):
    q,v = np.asarray(q,dtype=np.float64),np.asarray(v,dtype=np.float64)
    model = ForceControlledPointCable.from_mapping(model_config)
    expected = (model.cable_configuration.node_count,3)
    if q.shape!=expected or v.shape!=expected or not np.isfinite(q).all() or not np.isfinite(v).all():
        raise ValueError(f'Initial positions and velocities must be finite {expected} arrays')
    rest = np.asarray(model.cable_configuration.rest_lengths_m)
    if np.max(np.abs(np.linalg.norm(np.diff(q,axis=0),axis=1)-rest))>1e-3:
        raise ValueError('Initialize/project the measured cable geometry before planning (length error >1 mm)')
    return DderState(torch.from_numpy(q[None].copy()),torch.from_numpy(v[None].copy()))


def create_plan(package, initial_path, output):
    package, initial_path, output = Path(package),Path(initial_path),Path(output)
    verify_package(package)
    if output.exists():
        raise ValueError('Choose a new output directory')
    model,task,ppo = [json.loads((package/f'{name}.json').read_text(encoding='utf-8')) for name in ('model','task','ppo')]
    with np.load(initial_path,allow_pickle=False) as initial:
        state = validate_initial_state(model,initial['positions_m'],initial['velocities_m_s'])
        if 'state_time_s' not in initial or not np.isfinite(initial['state_time_s']).all() or initial['state_time_s'].size!=1:
            raise ValueError('Supply scalar state_time_s in the agreed tracking clock')
        state_time = float(initial['state_time_s'].item())
    torch.set_num_threads(1)
    started = time.perf_counter()
    flight = LiveFlight.from_checkpoint(model,task,ppo,package/'checkpoints/policy.pt')
    plan = flight.plan_strike(state)
    elapsed = time.perf_counter()-started
    forces = plan.forces_world_n.cpu().numpy()
    # Preserve the exact physics-rate trace; compress only identical adjacent holds.
    starts = np.r_[0,np.flatnonzero(np.any(forces[1:]!=forces[:-1],axis=1))+1]
    ends = np.r_[starts[1:],len(forces)]
    metadata = dict(schema='frozen_open_loop_force_plan_v1',flight_ready=False,
        state_time_s=state_time,initial_file_sha256=hashlib.sha256(initial_path.read_bytes()).hexdigest(),
        policy_sha256=hashlib.sha256((package/'checkpoints/policy.pt').read_bytes()).hexdigest(),
        policy_manifest_sha256=hashlib.sha256((package/'policy_manifest.json').read_bytes()).hexdigest(),
        physics_dt_s=plan.dt_s,policy_dt_s=task['control_dt_s'],cutoff_s=plan.duration_s,
        planning_wall_seconds=elapsed,force_units='N',frame='verified_common_world_z_up_required',
        gravity_convention='total commanded thrust vector; simulator applies gravity separately',
        launch_position_drift_m=ppo['deployment']['launch_position_drift_m'],
        launch_velocity_drift_m_s=ppo['deployment']['launch_velocity_drift_m_s'],
        next_step='Recheck current initial state and controller readiness before launch; no flight is authorized by this file')
    output.mkdir(parents=True)
    np.savez_compressed(output/'plan.npz',forces_world_n=forces,
        positions_m=state.positions_m[0].numpy(),velocities_m_s=state.velocities_m_s[0].numpy())
    with (output/'commands.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['time_s','until_s','fx_n','fy_n','fz_n'])
        for start,end in zip(starts,ends):
            writer.writerow([start*plan.dt_s,end*plan.dt_s,*forces[start]])
    (output/'plan.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,default=Path('policy'))
    parser.add_argument('--initial',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--example-initial',type=Path,help='Write a synthetic hanging state for offline checks only')
    args=parser.parse_args()
    verify_package(args.package)
    if args.verify:
        print('Policy package hashes verified; no vehicle connection.');return
    if args.example_initial:
        if args.example_initial.exists():raise ValueError('Example output already exists')
        model=json.loads((args.package/'model.json').read_text(encoding='utf-8'))
        task=json.loads((args.package/'task.json').read_text(encoding='utf-8'))
        state=ForceControlledPointCable.from_mapping(model).hanging_state(
            torch.tensor([task['initial_root_position_m']],dtype=torch.float64))
        np.savez_compressed(args.example_initial,positions_m=state.positions_m[0].numpy(),
            velocities_m_s=state.velocities_m_s[0].numpy(),state_time_s=0.)
        print('Synthetic state written. Do not substitute it for live measurements.');return
    if args.initial is None or args.output is None:parser.error('--initial and --output are required')
    print(json.dumps(create_plan(args.package,args.initial,args.output),indent=2))


if __name__=='__main__':main()
