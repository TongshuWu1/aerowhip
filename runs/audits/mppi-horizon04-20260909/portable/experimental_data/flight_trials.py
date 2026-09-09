"""Immutable, explicitly synchronized flight imports for between-trial adaptation."""
from pathlib import Path
import csv
import json
import re
import shutil
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from simulator.cable import CableConfiguration, DderModel, START_PINNED_FREE_END
from .io import atomic_json, sha256_file

TRACKING_FIELDS = ['time_s', 'drone_x', 'drone_y', 'drone_z', 'qx', 'qy', 'qz', 'qw',
                   'drone_valid'] + [f'c{i}_{field}' for i in range(1, 11) for field in ('x', 'y', 'z', 'valid')]
COMMAND_FIELDS = ['time_s', 'fx_n', 'fy_n', 'fz_n']


def template(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    meta = dict(schema='aerial_whip_flight_v1', trial_id='flight_001', role='adaptation',
                world_frame='right_handed_z_up', position_units='m', force_units='N',
                quaternion='xyzw_body_to_world', force_convention='total_world_force_including_hover_support',
                tracking_time_offset_s=None, command_time_offset_s=None, clock_alignment_verified=False,
                strike_start_s=None, planned_cutoff_s=None, precontact_end_s=None,
                contact_evidence='unreviewed', outcome='unreviewed', controller_version='',
                policy_sha256='', model_sha256='', controller_interface_verified=False,
                notes='All event times use the common clock: common_time = source_time + offset. '
                      'precontact_end_s must precede possible contact/intervention and not exceed cutoff. '
                      'Record all attempts, including failures. Keep original controller/IMU logs in this folder.')
    atomic_json(destination/'trial.json', meta)
    for name, fields in [('tracking.csv', TRACKING_FIELDS), ('commands.csv', COMMAND_FIELDS)]:
        with (destination/name).open('w', newline='') as stream:
            csv.writer(stream).writerow(fields)
    (destination/'README.txt').write_text(
        'Fill trial.json and the two CSV files. No rows are fabricated.\n'
        'tracking.csv: 100 Hz, meters, right-handed Z-up, drone rigid-body origin, xyzw body-to-world quaternion, '
        'c1..c10 ordered attachment to tip; validity flags are 0 or 1. Include at least 0.1 s before strike.\n'
        'commands.csv: actual sent world-force commands in N, each timestamp starts a zero-order hold. '
        'Include the hover command immediately before launch. Do not add gravity twice.\n'
        'Copy original controller/IMU logs and the exact planned force sequence/initial estimate into this folder. '
        'They are preserved; sent force is not measured applied force. Convert clocks/axes explicitly.\n'
        'Snapshot model.json and task.json from the model used for planning. An import never guesses them.\n'
        'Contact/intervention timing must be reviewed before fitting; exclude uncertain contact frames.\n', encoding='utf-8')


def numeric_csv(path, fields):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not set(fields).issubset(reader.fieldnames or []):
            raise ValueError(f'{Path(path).name}: missing columns {set(fields)-set(reader.fieldnames or [])}')
        rows = [[float(row[key]) for key in fields] for row in reader]
    if len(rows) < 2:
        raise ValueError(f'{Path(path).name}: at least two rows required')
    return np.asarray(rows, dtype=float)


def validate_and_prepare(source):
    source = Path(source)
    meta = json.loads((source/'trial.json').read_text(encoding='utf-8'))
    model = json.loads((source/'model.json').read_text(encoding='utf-8'))
    task = json.loads((source/'task.json').read_text(encoding='utf-8'))
    expected = dict(schema='aerial_whip_flight_v1', world_frame='right_handed_z_up', position_units='m',
                    force_units='N', quaternion='xyzw_body_to_world',
                    force_convention='total_world_force_including_hover_support')
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f'{key} must be explicitly {value}')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', meta['trial_id']):
        raise ValueError('Use a simple unique trial_id')
    if meta.get('role') not in ('adaptation', 'validation', 'protected_test'):
        raise ValueError('Choose adaptation, validation or protected_test role')
    if meta['role'] == 'protected_test':
        raise ValueError('Protected tests cannot enter the development adaptation workflow')
    if not meta.get('clock_alignment_verified'):
        raise ValueError('Verify clock alignment and record explicit source-to-common offsets first')
    if meta.get('contact_evidence') in (None, '', 'unreviewed'):
        raise ValueError('Review contact/intervention evidence and the pre-contact end first')
    for key in ('tracking_time_offset_s', 'command_time_offset_s', 'strike_start_s', 'planned_cutoff_s', 'precontact_end_s'):
        if meta.get(key) is None or not np.isfinite(meta[key]):
            raise ValueError(f'Finite {key} required')
    start, end, cutoff = (float(meta[k]) for k in ('strike_start_s', 'precontact_end_s', 'planned_cutoff_s'))
    if not start < end <= cutoff or cutoff-start > float(task['episode_duration_s']) + 1e-8:
        raise ValueError('Require launch < pre-contact end <= frozen cutoff <= task horizon')
    tracking = numeric_csv(source/'tracking.csv', TRACKING_FIELDS)
    commands = numeric_csv(source/'commands.csv', COMMAND_FIELDS)
    tracking[:, 0] += meta['tracking_time_offset_s']
    commands[:, 0] += meta['command_time_offset_s']
    for data, name in ((tracking, 'tracking'), (commands, 'commands')):
        if not np.isfinite(data[:, 0]).all() or not (np.diff(data[:, 0]) > 0).all():
            raise ValueError(f'{name}: timestamps must be finite, unique, and increasing')
    if not np.isfinite(commands).all():
        raise ValueError('Commands must be finite')
    # Only use the causal prehistory and reviewed pre-contact interval. Never
    # bridge invalid marker gaps or include a potentially contacted endpoint.
    selected = (tracking[:, 0] >= start-.11-1e-8) & (tracking[:, 0] < end-1e-9)
    tracking = tracking[selected]
    if len(tracking) < 13 or not np.isfinite(tracking).all():
        raise ValueError('Need finite tracking covering 0.1 s prehistory and pre-contact flight')
    validity = tracking[:, [8]+list(range(12, len(TRACKING_FIELDS), 4))]
    if not (validity == 1).all():
        raise ValueError('Invalid drone/cable markers in identification interval; choose a complete interval')
    dt = float(model['simulation']['dt_s'])
    if not np.allclose(np.diff(tracking[:, 0]), dt, atol=dt*.05, rtol=0):
        raise ValueError('Tracking must match model timestep within 5%; no hidden resampling')
    launch = int(np.argmin(abs(tracking[:, 0]-start)))
    if abs(tracking[launch, 0]-start) > 1e-6 or launch < 10:
        raise ValueError('Strike start must coincide with a tracking frame and have 10 causal preceding frames')
    quat = tracking[:, 4:8]
    if not np.allclose(np.linalg.norm(quat, axis=1), 1, atol=.01):
        raise ValueError('Drone quaternion must be unit length')
    offset = np.asarray(model['recorded_data']['optitrack_to_attachment_offset_body_m'])
    attachment = tracking[:, 1:4] + Rotation.from_quat(quat).apply(offset)
    measured = tracking[:, 9:].reshape(-1, 10, 4)[:, :, :3]
    sites = np.concatenate((attachment[:, None], measured), axis=1)
    cable = CableConfiguration.from_mapping(model['cable'])
    nodes = [sites[:, 0]]
    for i, subdivisions in enumerate(cable.interval_subdivisions):
        nodes.extend(sites[:, i]*(1-j/subdivisions)+sites[:, i+1]*(j/subdivisions)
                     for j in range(1, subdivisions+1))
    nodes = np.stack(nodes, axis=1)
    history = nodes[launch-10:launch+1]
    history_time = tracking[launch-10:launch+1, 0]-start
    # Linear past-only estimate; save both raw shape and projection discrepancy.
    design = np.stack((np.ones(11), history_time), axis=1)
    coefficient = np.linalg.lstsq(design, history.reshape(11, -1), rcond=None)[0]
    q = torch.tensor(nodes[launch:launch+1], dtype=torch.float64)
    v = torch.tensor(coefficient[1].reshape(1, cable.node_count, 3), dtype=q.dtype)
    rod = DderModel(cable.dder_parameters(EI=model['cable']['EI_n_m2'], Cb=model['cable']['Cb_n_m2_s']))
    with torch.no_grad():
        for _ in range(8):
            q = rod.project_lengths(q, q[:, :1], pinned_endpoints=START_PINNED_FREE_END)
        v = rod.project_velocities(q, v, v[:, :1], pinned_endpoints=START_PINNED_FREE_END)
    times = tracking[launch:, 0]
    index = np.searchsorted(commands[:, 0], times[:-1]+1e-9, side='right')-1
    if (index < 0).any():
        raise ValueError('Commands do not cover the launch interval; include prelaunch hover')
    # Command recording must cover the full frozen maneuver, not end early.
    if commands[-1, 0] < cutoff-float(task['control_dt_s'])-1e-6:
        raise ValueError('Command log ends before the final command interval')
    projection = np.sqrt(np.mean(np.sum((q.numpy()[0, cable.marker_node_indices[1:]]-measured[launch])**2, axis=-1)))
    if projection > .02:
        raise ValueError(f'Initial cable projection error {projection*1000:.1f} mm exceeds 20 mm; inspect marker order/offset/geometry')
    arrays = dict(time_s=times-start, initial_positions_m=q.numpy(), initial_velocities_m_s=v.numpy(),
                  root_positions_m=attachment[launch:][None], measured_marker_positions_m=measured[launch:][None],
                  commanded_force_world_n=commands[index, 1:], command_time_s=commands[:, 0]-start,
                  command_values_n=commands[:, 1:])
    return meta, model, task, arrays, dict(initialization_marker_rmse_m=float(projection),
        frames=len(times), dt_s=dt, duration_s=float(times[-1]-times[0]),
        initialization='11-frame past-only linear velocity; measured launch shape; length/velocity projection',
        fit_interval='strictly before reviewed precontact_end_s', controller_logs_used_for_fit=False)


def import_trial(root, source):
    source = Path(source).resolve()
    meta, model, task, arrays, diagnostics = validate_and_prepare(source)
    destination = Path(root)/'data/flight_trials'/meta['trial_id']
    if destination.exists():
        raise ValueError('Trial already exists; immutable recordings cannot be overwritten')
    # Copy explicit files only, and preserve extra original log files without parsing.
    files = [p for p in source.iterdir() if p.is_file()]
    if any(p.is_symlink() for p in files):
        raise ValueError('Resolve source log links before importing')
    raw = destination/'raw'; raw.mkdir(parents=True)
    hashes = {}
    for path in files:
        shutil.copy2(path, raw/path.name)
        hashes[path.name] = sha256_file(raw/path.name)
    for name, payload in [('trial', meta), ('model', model), ('task', task), ('diagnostics', diagnostics)]:
        atomic_json(destination/f'{name}.json', payload)
    np.savez_compressed(destination/'prepared.npz', **arrays)
    atomic_json(destination/'import.json', dict(source=str(source), raw_sha256=hashes,
        snapshot_sha256={name:sha256_file(destination/name) for name in ('trial.json','model.json','task.json','diagnostics.json')},
        prepared_sha256=sha256_file(destination/'prepared.npz'), immutable=True))
    return destination
