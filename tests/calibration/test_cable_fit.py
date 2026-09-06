from __future__ import annotations

import numpy as np

from experimental_data.cable_fit import contiguous_window_starts


def test_fit_windows_do_not_cross_invalid_frames() -> None:
    valid = np.array(
        [False, True, True, True, True, False, True, True, True, True, True]
    )
    starts = contiguous_window_starts(valid, horizon_steps=2, stride_steps=2)
    assert starts == (1, 6, 8)
    for start in starts:
        assert np.all(valid[start : start + 3])


