"""Immutable, quality-masked inputs for the authorized all-historical fit.

No rejection based on prediction error or task success. Source files are never
modified. Cable and command validity are separate to avoid throwing away useful
motion merely because a command packet was absent.
"""
from pathlib import Path
import json
import shutil
import numpy as np
from .io import atomic_json, sha256_file, canonical_json_hash
from .force_dataset import _normalized_rotations
from .quality import _intervals
from simulator.workflow import stamp

TAKES = ('fig8_001','fig8_002','fig8_003','fig8vertical_001','fig8vertical_002',
         'osc_001','osc_002','osc_003','whip1_001','whip1_002','whip1_003')
FOLDS = (('fig8_001','fig8vertical_001','osc_001','whip1_001'),
         ('fig8_002','fig8vertical_002','osc_002','whip1_002'),
         ('fig8_003','osc_003','whip1_003'))


def transition_mask(values, threshold):
    bad = np.zeros(len(values), dtype=bool)
    jump = np.linalg.norm(np.diff(values, axis=0), axis=-1) > threshold
    if jump.ndim > 1:
        jump = jump.any(axis=1)
    bad[:-1] |= jump
    bad[1:] |= jump
    return bad


def quality_masks(a, payload):
    p, marker = a['position'], a['markers']
    rotation, rotation_valid = _normalized_rotations(a['quaternion'])
    root = p + np.einsum('tij,j->ti', rotation,
        payload['recorded_data']['optitrack_to_attachment_offset_body_m'])
    sites = np.concatenate((root[:,None], marker), axis=1)
    lengths = np.linalg.norm(np.diff(sites, axis=1), axis=-1)
    rest = np.asarray(payload['cable']['marker_interval_lengths_m'])
    # Chords may be shorter than arc lengths when bending. Only excess length is
    # a geometric inconsistency; do not reject sharp, physically possible bends.
    geometry = (lengths > rest[None] + .025).any(1)
    missing_drone = ~a['position_valid'] | ~np.isfinite(p).all(1)
    missing_marker = ~a['marker_valid'].all(1) | ~np.isfinite(marker).all((1,2))
    drone_jump = transition_mask(p, .10)
    marker_jump = transition_mask(marker, .10)
    near_plane = np.nanmin(marker[:,:,2], axis=1) < .05
    # Conservative possible-contact quarantine, not a claim to know floor height.
    contact = np.zeros(len(p), dtype=bool)
    for i in np.flatnonzero(near_plane):
        contact[max(0,i-20):min(len(p),i+21)] = True
    reasons = dict(missing_drone=missing_drone, missing_marker=missing_marker,
        invalid_orientation=~rotation_valid, drone_jump=drone_jump,
        marker_jump=marker_jump, excessive_chord=geometry,
        possible_contact_near_source_z_zero=contact)
    cable_valid = ~np.logical_or.reduce(list(reasons.values()))
    command_bad = (~a['command_valid'] | ~np.isfinite(a['commands']).all(1)
                   | ~np.isfinite(a['command_age']) | (a['command_age'] > .10)
                   | (a['command_age'] < -.000001))
    drone_valid = ~(missing_drone | drone_jump | contact | command_bad)
    reasons['missing_or_stale_command'] = command_bad
    return cable_valid, drone_valid, reasons


def load_source(root, name):
    if name.startswith('whip'):
        path = root/'data/adaptation_rounds/adaptation0/processed/20260907-194802-922485'/name/'dataset.npz'
        quality = json.loads((path.parent/'quality.json').read_text())
        if sha256_file(path) != quality['output_files']['dataset.npz']:
            raise ValueError(f'{name}: processed source checksum mismatch')
        with np.load(path, allow_pickle=False) as d:
            a = dict(time=d['optitrack_time_s'],position=d['drone_position_m'],
                quaternion=d['drone_quaternion_xyzw'],position_valid=d['drone_position_valid'],
                markers=d['cable_position_m'],marker_valid=d['cable_valid'],
                commands=d['reference_fullstate'][:,:9],command_valid=d['reference_valid'],
                command_age=d['reference_age_s'])
        details = dict(alignment=quality['alignment'],
            command_semantics='Actual logged full-state stream including early hold',
            source_quality=quality)
    else:
        path = root/'data/processed_takes'/name/'take.npz'
        with np.load(path,allow_pickle=False) as d:
            a = dict(time=d['time_s'],position=d['uav_position_m'],quaternion=d['uav_orientation_xyzw'],
                position_valid=d['uav_valid'],markers=d['cable_marker_positions_m'],marker_valid=d['cable_marker_valid'],
                commands=np.concatenate([d['command_position_m'],d['command_velocity_mps'],d['command_acceleration_mps2']],axis=1),
                command_valid=d['command_valid'],command_age=d['command_age_s'])
        sync = json.loads((path.parent/'sync_report.json').read_text())
        details = dict(alignment=sync,command_semantics='Logged FullState P/V/A, not measured thrust')
    a['time'] = a['time'] - a['time'][0]
    if not np.isfinite(a['time']).all() or not np.allclose(np.diff(a['time']),.01,atol=1e-7):
        raise ValueError(f'{name}: uniform 100 Hz timestamps required')
    return path, a, details


def prepare(root):
    root = Path(root).resolve()
    payload = json.loads((root/'config/model.json').read_text())
    job = root/'data/historical_model_runs'/stamp()
    (job/'inputs').mkdir(parents=True,exist_ok=False)
    atomic_json(job/'original_model.json',payload)
    audit, inputs = {}, {}
    for name in TAKES:
        path, a, details = load_source(root,name)
        cable_valid,drone_valid,reasons = quality_masks(a,payload)
        a.update(cable_fit_valid=cable_valid,drone_fit_valid=drone_valid)
        np.savez_compressed(job/'inputs'/f'{name}.npz',**a)
        audit[name] = dict(frames=len(a['time']),duration_s=float(a['time'][-1]),
            cable_frames_kept=int(cable_valid.sum()),drone_frames_kept=int(drone_valid.sum()),
            exclusions={key:dict(frames=int(mask.sum()),intervals=_intervals(mask,a['time'],key))
                        for key,mask in reasons.items()}, **details)
        inputs[name] = dict(source=str(path),source_sha256=sha256_file(path),
                           snapshot_sha256=sha256_file(job/'inputs'/f'{name}.npz'))
    atomic_json(job/'audit.json',audit)
    protocol = dict(schema='historical_joint_baseline_v1', takes=list(TAKES),folds=FOLDS,
        setup='User confirmed same drone and settings for preliminary and whip recordings',
        quality='Independent measurement/command masks, no loss-based or success-based exclusion',
        contact_quarantine='Any cable marker below source Z=0.05 m, plus 0.2 s either side; conservative possible contact',
        geometry='Reject chord excess >25 mm; short chords are allowed to bend',
        roles='Three whole-take development folds, followed by all-data final fit. No independent historical test.',
        seed=1729,device='cuda',cable_horizon_s=.5,cable_selection_horizon_s=2.,
        cable_windows_per_take=6,cable_training_windows_per_take=18,
        cable_updates=24,cable_hidden=32,cable_acceleration_limit=.5,
        cable_learning_rate=.0005,cable_regularization=.001,
        physical_EI_grid=[1e-9,1e-7,1e-5,2.8e-4],physical_Cb_grid=[1e-8,1e-6,1e-4,.002,.02],
        drag_mode='nn_only',external_drag_s_inv=0.,
        drone_horizon_s=1.,drone_windows_per_take=6,drone_updates=160,
        drone_hidden=32,drone_acceleration_limit=2.,drone_learning_rate=.001,
        drone_regularization=.001,drone_delay_candidates_s=[0.,.02,.04,.06,.08,.10],
        nominal_gain_bounds=[[.1]*6+[0.]*3,[80.]*3+[20.]*3+[2.]*3],
        provenance=inputs,initial_model_sha256=canonical_json_hash(payload))
    atomic_json(job/'protocol.json',protocol)
    for name in ('AGENTS.md','experimental_data/historical_dataset.py','experimental_data/historical_fit.py',
                 'experimental_data/differentiable_fit.py','experimental_data/cable_residual_fit.py',
                 'experimental_data/constrained_geometry.py','experimental_data/constrained_identification.py',
                 'experimental_data/state_initialization.py','experimental_data/force_dataset.py',
                 'simulator/drone_tracking.py','simulator/cable/dder.py','simulator/cable/config.py',
                 'simulator/cable/residual.py'):
        if not (root/name).exists(): continue
        target=job/'source_snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/name,target)
    atomic_json(job/'status.json',dict(status='PREPARED',active_model_changed=False,ppo_started=False))
    return job


def read_inputs(job):
    protocol=json.loads((job/'protocol.json').read_text())
    result={}
    for name,row in protocol['provenance'].items():
        path=job/'inputs'/f'{name}.npz'
        if sha256_file(path)!=row['snapshot_sha256']:
            raise ValueError(f'Input snapshot changed: {name}')
        with np.load(path,allow_pickle=False) as d:
            result[name]={k:d[k] for k in d.files}
    return result
