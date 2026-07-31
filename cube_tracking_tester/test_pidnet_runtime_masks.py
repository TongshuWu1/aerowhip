"""Regression tests for the shared GUI/main PIDNet runtime mask contract."""

from pathlib import Path
import sys
import tomllib
import unittest

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
NN_SOURCE_DIR = PROJECT_DIR / "NN_collection_training" / "source"
if str(NN_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(NN_SOURCE_DIR))

from cable_detection import (  # noqa: E402
    PidNetMaskConfig,
    clean_binary_mask,
    pidnet_masks_from_probability,
    postprocess_pidnet_masks,
    remove_small_components,
)


class PidNetRuntimeMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime_path = NN_SOURCE_DIR / "config.toml"
        with cls.runtime_path.open("rb") as stream:
            cls.runtime_values = tomllib.load(stream)
        cls.mask_config = PidNetMaskConfig.from_mapping(cls.runtime_values)

    def test_main_configuration_references_the_gui_runtime_configuration(self):
        with (PROJECT_DIR / "ZED_segmentation_viewer" / "source" / "config.toml").open("rb") as stream:
            main_values = tomllib.load(stream)
        configured = PROJECT_DIR / main_values["pidnet"]["runtime_config"]
        self.assertEqual(configured.resolve(), self.runtime_path.resolve())
        self.assertNotIn("threshold", main_values["pidnet"])
        self.assertNotIn("endpoint_thresholds", main_values["pidnet"])

    def test_probability_and_thresholded_entry_points_are_bit_identical(self):
        rng = np.random.default_rng(17)
        probability = rng.random((73, 101, 3), dtype=np.float32)
        probability[8:18, 12:46, 0] = 0.99
        probability[12:14, 28:30, 0] = 0.0
        raw_masks = tuple(
            np.where(
                probability[:, :, channel] >= threshold,
                255,
                0,
            ).astype(np.uint8)
            for channel, threshold in enumerate(self.mask_config.thresholds)
        )

        gui_masks, gui_components = pidnet_masks_from_probability(
            probability,
            self.mask_config,
        )
        main_masks, main_components = postprocess_pidnet_masks(
            raw_masks,
            self.mask_config,
        )

        self.assertEqual(gui_components, main_components)
        for gui_mask, main_mask in zip(gui_masks, main_masks):
            self.assertTrue(np.array_equal(gui_mask, main_mask))

    def test_cleanup_changes_only_the_body_channel(self):
        body = np.zeros((32, 48), dtype=np.uint8)
        body[6:20, 6:22] = 255
        endpoint1 = np.zeros_like(body)
        endpoint2 = np.zeros_like(body)
        endpoint1[2, 3] = 255
        endpoint2[27, 42] = 255

        masks, _component_count = postprocess_pidnet_masks(
            (body, endpoint1, endpoint2),
            self.mask_config,
        )

        self.assertTrue(np.array_equal(masks[1], endpoint1))
        self.assertTrue(np.array_equal(masks[2], endpoint2))

    def test_roi_cleanup_is_bit_identical_to_full_frame_reference(self):
        mask = np.zeros((180, 260), dtype=np.uint8)
        mask[0:32, 0:28] = 255
        mask[54:61, 77:115] = 255
        mask[57:64, 119:158] = 255
        mask[150:154, 225:230] = 255
        mask[100, 190] = 255

        reference = mask.copy()
        if self.mask_config.open_kernel > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.mask_config.open_kernel, self.mask_config.open_kernel),
            )
            reference = cv2.morphologyEx(reference, cv2.MORPH_OPEN, kernel)
        if self.mask_config.close_kernel > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.mask_config.close_kernel, self.mask_config.close_kernel),
            )
            reference = cv2.morphologyEx(reference, cv2.MORPH_CLOSE, kernel)
        reference, reference_components = remove_small_components(
            reference,
            min_area=self.mask_config.min_area_px,
        )

        actual, actual_components = clean_binary_mask(
            mask,
            self.mask_config.min_area_px,
            self.mask_config.open_kernel,
            self.mask_config.close_kernel,
        )

        self.assertEqual(actual_components, reference_components)
        self.assertTrue(np.array_equal(actual, reference))


if __name__ == "__main__":
    unittest.main()
