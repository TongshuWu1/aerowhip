"""CEM-owned launch coordinates, independent of PPO rehearsal/training settings."""
import numpy as np


DEFAULT_LAUNCH = dict(initial_tracking_origin_m=[-2., 0., 1.255],
                      target_position_m=[-1., 0., 1.1])


def resolve_launch_setup(setup=None):
    setup = DEFAULT_LAUNCH if setup is None else setup
    resolved = {}
    for key in DEFAULT_LAUNCH:
        try:
            vector = np.asarray(setup[key], dtype=float)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('CEM launch setup requires finite tracked-origin and target XYZ vectors.') from error
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError('CEM launch setup requires finite tracked-origin and target XYZ vectors.')
        resolved[key] = vector.tolist()
    return resolved
