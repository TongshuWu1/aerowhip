"""PPO rehearsal defaults in tracked-origin coordinates; CEM owns its settings."""
from pathlib import Path
import numpy as np
from .workflow import read_json


def launch_positions(root, fallback_origin, fallback_target):
    setup=read_json(Path(root)/'config/launch_setup.json',{})
    origin=np.asarray(setup.get('initial_tracking_origin_m',fallback_origin),float)
    target=np.asarray(setup.get('target_position_m',fallback_target),float)
    if origin.shape!=(3,) or target.shape!=(3,) or not np.isfinite([origin,target]).all():
        raise ValueError('Launch setup requires finite tracked-origin and target XYZ vectors.')
    return origin,target
