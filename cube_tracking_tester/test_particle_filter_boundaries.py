"""Focused checks for cable endpoint boundary assumptions."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent
MAIN_SOURCE_DIR = PROJECT_DIR / "ZED_segmentation_viewer" / "source"
if str(MAIN_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_SOURCE_DIR))

from particle_filter import _resolve_endpoint_boundaries


class EndpointBoundaryTests(unittest.TestCase):
    def test_occluded_endpoint_uses_last_position_as_stationary_anchor(self) -> None:
        measured = np.zeros((2, 2, 3), dtype=np.float32)
        measured[0, 0] = (0.10, 0.20, -0.70)
        visible = np.array(((True, False), (False, False)), dtype=bool)
        initialized = np.array((True, False), dtype=bool)
        previous = np.full((2, 2, 3), np.nan, dtype=np.float32)
        previous[0, 0] = (0.09, 0.20, -0.70)
        previous[0, 1] = (0.30, 0.20, -0.70)

        boundaries, anchors, held = _resolve_endpoint_boundaries(
            measured,
            visible,
            initialized,
            previous,
            hold_occluded=True,
            allow_single_observed=False,
        )

        np.testing.assert_array_equal(anchors[0], (True, True))
        np.testing.assert_array_equal(held[0], (False, True))
        np.testing.assert_allclose(boundaries[0, 0], measured[0, 0])
        np.testing.assert_allclose(boundaries[0, 1], previous[0, 1])
        np.testing.assert_array_equal(anchors[1], (False, False))

        _, disabled_anchors, disabled_held = _resolve_endpoint_boundaries(
            measured,
            visible,
            initialized,
            previous,
            hold_occluded=False,
            allow_single_observed=False,
        )
        np.testing.assert_array_equal(disabled_anchors[0], (False, False))
        self.assertFalse(disabled_held.any())


if __name__ == "__main__":
    unittest.main()
