"""Read-only comparison of a flown FullState CSV with its saved prediction."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp
from .adaptation_rounds import read_optitrack, read_controller, align, COMMAND_COLUMNS
from simulator.geometry import attachment_positions, normalized_rotations_xyzw


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def discover_batches(root):
    folder = Path(root) / 'rehearsal_csv_and_result_in_real_flight'
    return sorted({p.parent.parent for p in folder.rglob('simulation_csv/fullstate_30hz.csv')})


def flight_names(batch):
    folder = Path(batch) / 'flight_take'
    return sorted(p.stem for p in folder.glob('*.csv') if not p.name.startswith('experiment_')
                  and (folder / ('experiment_' + p.name)).exists())


def find_rehearsal(root, csv_path, selected=None):
    digest = sha256(csv_path)
    if selected is not None:
        candidates = [Path(selected)]
    else:
        candidates = sorted({p.parent for base in ('runs/rehearsals', 'runs/cem')
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
    good = np.isfinite(commands).all(axis=1) & np.isfinite(controller['cmd_age']) & (controller['cmd_valid'] > .5)
    if not good.any():
        raise ValueError('No valid FullState commands in controller log.')
    distance, ids = cKDTree(values).query(commands[good])
    dynamic = np.linalg.norm(values[:, 3:9], axis=1) > 1e-5
    mask = (distance < 1e-8) & dynamic[ids]
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


def load_comparison(root, batch, take, selected_rehearsal=None):
    batch = Path(batch)
    if take not in flight_names(batch):
        raise ValueError('Select a paired flight in this batch.')
    csv_path = batch / 'simulation_csv/fullstate_30hz.csv'
    rehearsal = find_rehearsal(root, csv_path, selected_rehearsal)
    paths = [csv_path, batch/'flight_take'/f'{take}.csv', batch/'flight_take'/f'experiment_{take}.csv',
             *[rehearsal/name for name in ('rehearsal.npz', 'rehearsal.json', 'model.json', 'task.json')]]
    hashes = {str(p): sha256(p) for p in paths}
    metadata = json.loads((rehearsal/'rehearsal.json').read_text())
    model = json.loads((rehearsal/'model.json').read_text())
    task = json.loads((rehearsal/'task.json').read_text())
    m, c = read_optitrack(paths[1]), read_controller(paths[2])
    reference = np.genfromtxt(csv_path, delimiter=',', names=True)
    onset, packets, jitter = command_onset(c, reference)
    alignment = align(m, c)
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
    return dict(time=time, measured_origin=origin, measured_rotation=rotation, measured_cable=cable,
                predicted_origin=po, predicted_rotation=pr, predicted_cable=pq,
                predicted_marker_indices=marker_indices,
                target=predicted['target_position_m'], metadata=metadata, task=task,
                alignment=alignment, onset=onset, packet_count=packets, packet_jitter_s=jitter,
                take=take, batch=batch, rehearsal=rehearsal, hashes=hashes,
                tracking_span=(float(mt[0]),float(mt[-1])),
                drone_error=np.linalg.norm(origin-po, axis=1),
                tip_error=np.linalg.norm(cable[:,-1]-pq[:,-1], axis=1),
                target_error=np.linalg.norm(cable[:,-1]-predicted['target_position_m'], axis=1))
