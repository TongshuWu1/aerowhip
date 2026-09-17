"""Shared force/acceleration conversion and saved-plan validation helpers.

This module neither connects to ROS nor authorizes a flight.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from simulator.cable import DderState
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


def controller_force(force_world_n, controller_mass_kg, gravity_m_s2):
    """Force feedforward in N for a Z-up controller that adds its own m*g.

    Accept one XYZ vector or an Nx3 sequence. Do not clip negative residual Fz.
    Use exactly the controller's configured compensation mass. This may include
    the cable for a total-weight hover convention; physical model masses stay separate.
    """
    force = np.asarray(force_world_n, dtype=np.float64)
    if force.ndim not in (1, 2) or force.shape[-1] != 3 or not force.size or not np.isfinite(force).all():
        raise ValueError('Expected a finite XYZ world force in newtons')
    if not np.isfinite(controller_mass_kg) or controller_mass_kg<=0:
        raise ValueError('Controller mass must be measured/configured and positive')
    if not np.isfinite(gravity_m_s2) or gravity_m_s2<=0:
        raise ValueError('Supply the verified firmware gravity magnitude')
    return force - np.array([0., 0., controller_mass_kg*gravity_m_s2])


def controller_acceleration(force_world_n, controller_mass_kg, gravity_m_s2):
    """cmdFullState acceleration in m/s²; gravity is removed exactly once."""
    return controller_force(force_world_n, controller_mass_kg, gravity_m_s2)/controller_mass_kg


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


def write_plan(plan, output, *, task, metadata, controller_mass_kg=None, controller_gravity_m_s2=None):
    """Persist exact held commands and their cutoff before any execution."""
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a new output directory')
    forces = plan.forces_world_n.cpu().numpy()
    if (controller_mass_kg is None) != (controller_gravity_m_s2 is None):
        raise ValueError('Supply both controller mass and controller gravity')
    residual = None
    if controller_mass_kg is not None:
        residual = controller_force(forces, controller_mass_kg, controller_gravity_m_s2)
    starts = np.r_[0,np.flatnonzero(np.any(forces[1:]!=forces[:-1],axis=1))+1]
    ends = np.r_[starts[1:],len(forces)]
    metadata = dict(metadata, schema='frozen_open_loop_force_plan_v1', flight_ready=False,
        physics_dt_s=plan.dt_s, policy_dt_s=task['control_dt_s'], cutoff_s=plan.duration_s,
        target_position_m=task['target_position_m'],
        desired_strike_direction_world=task['desired_strike_direction_world'],
        force_units='N', frame='verified_common_world_z_up_required',
        gravity_convention='total commanded thrust vector; simulator applies gravity separately')
    if residual is not None:
        metadata['controller_export'] = dict(mass_kg=controller_mass_kg,
            gravity_m_s2=controller_gravity_m_s2, frame='world_z_up',
            force_file='controller_force.csv', acceleration_file='controller_acceleration.csv',
            force_convention='F_policy - [0, 0, controller_mass * gravity]',
            acceleration_convention='controller_force / controller_mass',
            total_force_file='commands.csv')
    output.mkdir(parents=True)
    np.savez_compressed(output/'plan.npz', forces_world_n=forces,
        positions_m=plan.initial_state.positions_m[0].numpy(),
        velocities_m_s=plan.initial_state.velocities_m_s[0].numpy())
    with (output/'commands.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['time_s','until_s','fx_n','fy_n','fz_n'])
        for start,end in zip(starts,ends):
            writer.writerow([start*plan.dt_s,end*plan.dt_s,*forces[start]])
    (output/'plan.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    if residual is not None:
        for filename, values, columns in (
            ('controller_force.csv', residual, ['fx_ff_n','fy_ff_n','fz_ff_n']),
            ('controller_acceleration.csv', residual/controller_mass_kg, ['ax_ff_m_s2','ay_ff_m_s2','az_ff_m_s2'])):
            with (output/filename).open('w', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream)
                writer.writerow(['time_s', 'until_s', *columns])
                for start, end in zip(starts, ends):
                    writer.writerow([start*plan.dt_s, end*plan.dt_s, *values[start]])
    return metadata
