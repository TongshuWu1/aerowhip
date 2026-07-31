"""Exactness checks for native-resolution observation optimizations."""

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
VIEWER_SOURCE_DIR = PROJECT_DIR / "ZED_segmentation_viewer" / "source"
if str(VIEWER_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(VIEWER_SOURCE_DIR))

from observation import _component_roi, _mask_u8  # noqa: E402


class NativeResolutionObservationTests(unittest.TestCase):
    def test_uint8_runtime_masks_are_reused_without_changing_support(self):
        mask = np.zeros((108, 192), dtype=np.uint8)
        mask[17:83, 24:169] = 255

        prepared = _mask_u8(mask, mask.shape)

        self.assertTrue(np.shares_memory(prepared, mask))
        self.assertTrue(np.array_equal(prepared > 0, mask > 0))

    def test_component_roi_matches_the_former_full_frame_scan(self):
        mask = np.zeros((1080, 1920), dtype=np.uint8)
        cv2.polylines(
            mask,
            [np.asarray(((0, 80), (420, 360), (910, 190)), dtype=np.int32)],
            False,
            255,
            13,
            cv2.LINE_8,
        )
        cv2.polylines(
            mask,
            [np.asarray(((1030, 240), (1400, 470), (1919, 320)), dtype=np.int32)],
            False,
            255,
            13,
            cv2.LINE_8,
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        for component_label in range(1, count):
            full_component = labels == component_label
            component_y, component_x = np.nonzero(full_component)
            x0 = max(0, int(component_x.min()) - 2)
            x1 = min(labels.shape[1], int(component_x.max()) + 3)
            y0 = max(0, int(component_y.min()) - 2)
            y1 = min(labels.shape[0], int(component_y.max()) + 3)

            roi, origin_yx = _component_roi(labels, stats, component_label)

            self.assertEqual(origin_yx, (y0, x0))
            self.assertTrue(np.array_equal(roi, full_component[y0:y1, x0:x1]))


if __name__ == "__main__":
    unittest.main()
