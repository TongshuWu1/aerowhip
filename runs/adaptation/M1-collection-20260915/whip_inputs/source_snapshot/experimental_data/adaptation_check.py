"""Read-only comparison of a flown FullState CSV with its saved prediction."""
from pathlib import Path
from .paths import flight_batch_roots
import hashlib
import json
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp
from .adaptation_rounds import read_optitrack, read_controller, COMMAND_COLUMNS
from simulator.geometry import attachment_positions, normalized_rotations_xyzw


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def discover_batches(root):
    recorded={p.parent.parent for folder in flight_batch_roots(root)
              for p in folder.rglob('simulation_csv/fullstate_30hz.csv')}
    lab={p.parent.parent for p in (Path(root)/'experiments').glob('*/batches/*/simulation_csv/fullstate_30hz.csv')}
    return sorted(recorded|lab)


def flight_names(batch):
    folder = Path(batch) / 'flight_take'
    exclusion_file = Path(batch) / 'excluded_takes.json'
    excluded = json.loads(exclusion_file.read_text()).get('excluded', {}) if exclusion_file.exists() else {}
    return sorted(p.stem for p in folder.glob('*.csv') if not p.name.startswith('experiment_')
                  and p.stem not in excluded
                  and (folder / ('experiment_' + p.name)).exists())


def find_rehearsal(root, csv_path, selected=None):
    digest = sha256(csv_path)
    if selected is not None:
        candidates = [Path(selected)]
    else:
        candidates = sorted({p.parent for base in ('runs/rehearsals', 'runs/rehearsals_pva', 'runs/cem', 'runs/mppi')
                             for p in (Path(root) / base).rglob('rehearsal.npz')})
    for folder in candidates:
        ref = folder / 'fullstate_30hz.csv'
        if ref.exists() and sha256(ref) == digest:
            return folder
    raise ValueError('No saved prediction matches this exact CSV. Select its original rehearsal folder; do not use a different model or export.')


def command_onset(controller, reference):
    """Reception-time estimate from unique moving PVA packets; never align to target."""
    values = np.column_stack([reference[n] for n in reference.dtype.names[1:]])
    commands = np.column_stack([controller[n] for n in COMMAND_COLUMNS])
    good = (np.isfinite(commands).all(axis=1) & np.isfinite(controller['cmd_age']) &
            (controller['cmd_age'] >= 0) & (controller['cmd_valid'] > .5))
    if not good.any():
        raise ValueError('No valid FullState commands in controller log.')
    distance, ids = cKDTree(values).query(commands[good])
    # Repeated command values cannot uniquely identify a CSV knot.
    _,inverse,counts=np.unique(values,axis=0,return_inverse=True,return_counts=True)
    unique=counts[inverse]==1
    dynamic = np.linalg.norm(values[:, 3:9], axis=1) > 1e-5
    mask = (distance < 1e-8) & dynamic[ids] & unique[ids]
    indices = ids[mask]
    if len(np.unique(indices)) < 10:
        raise ValueError('Too few matching dynamic command packets to identify CSV timing.')
    receipt = (controller['time_s'] - controller['cmd_age'])[good][mask]
    estimates = np.array([np.median(receipt[indices == j]) - reference['time_s'][j] for j in np.unique(indices)])
    if np.ptp(estimates) > .1:
        raise ValueError('Ambiguous CSV timing (repeated execution or inconsistent commands). Split/review the log before replay.')
    return float(np.median(estimates)), int(len(np.unique(indices))), float(np.ptp(estimates))


def interpolate_positions(source_time, values, time):
    """Adjacent samples only: never extrapolate or bridge a missing tracking sample."""
    source_time, values, time = np.asarray(source_time), np.asarray(values), np.asarray(time)
    right = np.clip(np.searchsorted(source_time, time), 1, len(source_time)-1)
    left = right-1
    weight = (time-source_time[left])/(source_time[right]-source_time[left])
    w = weight.reshape((-1,)+(1,)*(values.ndim-1))
    out = values[left]*(1-w)+values[right]*w
    exact_left = np.isclose(time, source_time[left], atol=1e-10, rtol=0)
    exact_right = np.isclose(time, source_time[right], atol=1e-10, rtol=0)
    out[exact_left] = values[left[exact_left]]
    out[exact_right] = values[right[exact_right]]
    good = (time >= source_time[0]) & (time <= source_time[-1])
    good &= ((source_time[right]-source_time[left]) <= 1.5*np.median(np.diff(source_time))) | exact_left | exact_right
    out[~good] = np.nan
    return out


def recorded_alignment(root, batch, take, tracking_path, controller_path):
    """Use explicit clock alignment, never controller state or desired-motion fitting.

    Previously audited legacy pairs retain their exact historical offsets only
    when both source hashes match. These remain estimates, not synced clocks.
    """
    sidecar = Path(batch) / 'time_alignment.json'
    if sidecar.exists():
        entries = json.loads(sidecar.read_text())
        entry = entries.get(take)
        if entry is not None:
            for key, path in [('optitrack_sha256', tracking_path), ('controller_sha256', controller_path)]:
                if entry.get(key) != sha256(path):
                    raise ValueError('Time alignment source changed; review the offset for this exact pair.')
            offset = float(entry['offset_s'])
            if not np.isfinite(offset) or not str(entry.get('source', '')).strip():
                raise ValueError('Time alignment needs a finite offset and its timestamp/event source.')
            return dict(offset_s=offset, method=entry['source'], rms_m=None,
                        clock_verified=bool(entry.get('clock_verified', False)),
                        includes_logging_latency=not bool(entry.get('clock_verified', False)))
    audit = Path(root) / 'runs/audits/20260909-adp0-hover-height/hover_height.json'
    identity_path=Path(tracking_path).with_suffix('.tracking.json')
    identity=json.loads(identity_path.read_text()) if identity_path.exists() else {}
    if audit.exists() and identity.get('drone','cf_7')=='cf_7':
        for row in json.loads(audit.read_text())['rows']:
            hashes = {Path(k).name: v for k, v in row['hashes'].items()}
            if (row['take'] == take and
                hashes.get(Path(tracking_path).name) == sha256(tracking_path) and
                hashes.get(Path(controller_path).name) == sha256(controller_path)):
                result = dict(row['alignment'])
                result['method'] = 'Frozen legacy audit offset (originally measured-stream aligned)'
                return result
    raise ValueError('Clock alignment required: add this exact pair to time_alignment.json '
                     'using controller_time = OptiTrack_time + offset_s. '
                     'Use timestamps/shared events; do not align measured motion to the command trajectory.')


def load_comparison(root, batch, take, selected_rehearsal=None):
    batch = Path(batch)
    if take not in flight_names(batch):
        raise ValueError('Select a paired flight in this batch.')
    csv_path = batch / 'simulation_csv/fullstate_30hz.csv'
    # Identical commands can have several saved predictions. Prefer this batch's
    # frozen forecast, rather than whichever matching rehearsal sorts first.
    bound_forecast = None
    protocol_path = batch / 'protocol.json'
    if selected_rehearsal is None and protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if protocol.get('rehearsal'):
            selected_rehearsal = Path(protocol['rehearsal'])
            if not selected_rehearsal.is_absolute(): selected_rehearsal = Path(root)/selected_rehearsal
            bound_forecast = protocol.get('forecast_sha256')
    rehearsal = find_rehearsal(root, csv_path, selected_rehearsal)
    if bound_forecast is not None and sha256(rehearsal/'rehearsal.npz') != bound_forecast:
        raise ValueError('Saved prediction differs from the batch frozen forecast.')
    paths = [csv_path, batch/'flight_take'/f'{take}.csv', batch/'flight_take'/f'experiment_{take}.csv',
             *[rehearsal/name for name in ('rehearsal.npz', 'rehearsal.json', 'model.json', 'task.json')]]
    if (batch/'time_alignment.json').exists():
        paths.append(batch/'time_alignment.json')
    if paths[1].with_suffix('.tracking.json').exists():
        paths.append(paths[1].with_suffix('.tracking.json'))
    if (batch/'height_calibration.json').exists():
        paths.append(batch/'height_calibration.json')
    if bound_forecast is not None: paths.append(protocol_path)
    hashes = {str(p): sha256(p) for p in paths}
    metadata = json.loads((rehearsal/'rehearsal.json').read_text())
    model = json.loads((rehearsal/'model.json').read_text())
    task = json.loads((rehearsal/'task.json').read_text())
    m, c = read_optitrack(paths[1]), read_controller(paths[2], commands_only=True)
    reference = np.genfromtxt(csv_path, delimiter=',', names=True)
    onset, packets, jitter = command_onset(c, reference)
    alignment = recorded_alignment(root, batch, take, paths[1], paths[2])
    mt = m['time'] + alignment['offset_s'] - onset
    with np.load(rehearsal/'rehearsal.npz', allow_pickle=False) as archive:
        predicted = {key: archive[key].copy() for key in archive.files}
    ref_values = np.column_stack([reference[n] for n in reference.dtype.names[1:]])
    if (predicted['commands'].shape != ref_values.shape or
        not np.allclose(predicted['command_time_s'],reference['time_s'],rtol=0,atol=1e-9) or
        not np.allclose(predicted['commands'],ref_values,rtol=0,atol=1e-9)):
        raise ValueError('Saved prediction packets disagree with its CSV. Restore the original rehearsal bundle.')
    pt = predicted['prediction_time_s']
    mask = (mt >= pt[0]) & (mt <= min(pt[-1], metadata.get('prediction_valid_through_s', pt[-1])))
    if mask.sum() < 3:
        raise ValueError('Tracking and saved prediction have no usable overlap.')
    time = mt[mask]
    origin = m['drone'][mask]
    rotation, _ = normalized_rotations_xyzw(m['quaternion'][mask])
    attachment, _ = attachment_positions(origin, m['quaternion'][mask], model['recorded_data']['optitrack_to_attachment_offset_body_m'])
    cable = np.concatenate([attachment[:,None], m['cable'][mask]], axis=1)
    # Rotation interpolation preserves orthonormality. Measured frames remain native 100 Hz samples.
    pr = Slerp(pt, Rotation.from_matrix(predicted['origin_rotations']))(time).as_matrix()
    po = interpolate_positions(pt, predicted['origin_positions_m'], time)
    pq = interpolate_positions(pt, predicted['cable_positions_m'], time)
    marker_indices = np.cumsum(model['cable']['segments_per_marker_interval']).astype(int)
    if len(marker_indices) != 10 or marker_indices[-1] != pq.shape[1]-1:
        raise ValueError('Saved cable discretization does not match the ten recorded marker sites.')
    if any(sha256(p) != hashes[str(p)] for p in paths):
        raise ValueError('An input changed while loading. Wait for file copying to finish, then reload.')
    result=dict(time=time, measured_origin=origin, measured_rotation=rotation, measured_cable=cable,
                measured_state_source=f"OptiTrack {m.get('drone_label','selected rigid body')} pose and cable1 marker observations",
                drone_rigid_body=m.get('drone_label'),
                controller_source='FullState commands, validity and receipt timing only',
                predicted_origin=po, predicted_rotation=pr, predicted_cable=pq,
                predicted_marker_indices=marker_indices,
                target=predicted['target_position_m'], metadata=metadata, task=task,
                alignment=alignment, onset=onset, packet_count=packets, packet_jitter_s=jitter,
                take=take, batch=batch, rehearsal=rehearsal, hashes=hashes,
                tracking_span=(float(mt[0]),float(mt[-1])),
                drone_error=np.linalg.norm(origin-po, axis=1),
                tip_error=np.linalg.norm(cable[:,-1]-pq[:,-1], axis=1),
                target_error=np.linalg.norm(cable[:,-1]-predicted['target_position_m'], axis=1))
    from .hover_calibration import load_calibration
    try:
        calibration=load_calibration(batch,take)
        if calibration is not None:
            result['height_calibration']=calibration
            corrected_origin=origin.copy();corrected_origin[:,2]-=calibration['bias_z_m']
            corrected_cable=cable.copy();corrected_cable[:,:,2]-=calibration['bias_z_m']
            result['hover_normalized']=dict(measured_origin=corrected_origin,measured_cable=corrected_cable,
                drone_error=np.linalg.norm(corrected_origin-po,axis=1),tip_error=np.linalg.norm(corrected_cable[:,-1]-pq[:,-1],axis=1),
                target_error=np.linalg.norm(corrected_cable[:,-1]-predicted['target_position_m'],axis=1))
    except (ValueError,OSError,KeyError) as error:
        result['height_calibration_error']=str(error)  # Raw replay remains available.
    return result
