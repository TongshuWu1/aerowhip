"""Display-only array helpers, shared by Isaac and geometry tests."""
import json
import io
from pathlib import Path
import numpy as np


def load_replay(directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'replay.json').read_text())
    if metadata['schema'] != 'multidrone_replay_v1':
        raise ValueError('Not a multi-drone replay.')
    import hashlib
    payload=(directory/'replay.npz').read_bytes()
    if hashlib.sha256(payload).hexdigest() != metadata['replay_sha256']:
        raise ValueError('Replay contents do not match their saved provenance.')
    with np.load(io.BytesIO(payload), allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    t, n = data['cable'].shape[:2]
    if n != metadata['num_envs'] or data['origin'].shape != (t,n,3) or data['rotation'].shape != (t,n,3,3):
        raise ValueError('Inconsistent replay dimensions.')
    if not all(np.isfinite(data[k]).all() for k in ('cable','origin','rotation','target','time_s')):
        raise ValueError('Nonfinite display geometry.')
    attached = data['origin'] + np.einsum('tbij,j->tbi', data['rotation'], data['attachment_offset_m'])
    if np.max(np.abs(attached-data['cable'][:,:,0])) > 1e-7:
        raise ValueError('Tracked origin and cable attachment disagree.')
    return data, metadata


def read_live_batch(run, previous_generation=None):
    """Read one committed cache slot; retry later if publication races this read."""
    import zipfile
    run=Path(run).resolve();pointer=run/'live_scene/latest.json'
    try:
        before=pointer.read_bytes();record=json.loads(before)
        if record['generation']==previous_generation:return None
        if record['slot'] not in ('slot_0','slot_1'):raise ValueError('Invalid scene cache slot')
        directory=pointer.parent/record['slot']
        data,metadata=load_replay(directory)
        if metadata.get('generation')!=record['generation'] or pointer.read_bytes()!=before:return None
        if not metadata.get('live_training') or Path(metadata['run']).resolve()!=run:return None
        return data,metadata,directory
    except (OSError,ValueError,KeyError,EOFError,zipfile.BadZipFile):
        return None


def grid_offsets(count, spacing=4.):
    side = int(np.ceil(np.sqrt(count)))
    xy = np.c_[np.arange(count)%side, np.arange(count)//side].astype(float)*spacing
    xy -= (xy.min(0)+xy.max(0))/2
    return np.c_[xy,np.zeros(count)]


def display_frame(data, frame, count, spacing=4.):
    count = min(int(count), data['origin'].shape[1])
    offsets = grid_offsets(count, spacing)
    return (data['origin'][frame,:count]+offsets,
            data['rotation'][frame,:count],
            data['cable'][frame,:count]+offsets[:,None],
            data['target'][:count]+offsets)
